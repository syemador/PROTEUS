"""
retrieval.py
============
Upgraded Retrieval engine for P.R.O.T.E.U.S.
* Includes a zero-dependency BM25 Sparse Index.
* Supports Hybrid Search (Dense + Sparse) via Reciprocal Rank Fusion (RRF).
* Supports Cross-Encoder Reranking over the candidate pool.
* Preserves Semantic Deduplication via dense embeddings post-reranking.
"""

from __future__ import annotations

import json
import logging
import pickle
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer

from data_processing import DocumentChunk

logger = logging.getLogger(__name__)

CLINICAL_BERT_ID = "emilyalsentzer/Bio_ClinicalBERT"
MEDCPT_QUERY_ID = "ncbi/MedCPT-Query-Encoder"
MEDCPT_DOC_ID = "ncbi/MedCPT-Article-Encoder"
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 768
INDEX_CACHE_VERSION = 2

class BM25Index:
    """Fast, local BM25 implementation for biomedical lexical matching."""
    def __init__(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        nd = {}
        num_wd = 0
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            num_wd += len(tokens)
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
                
        self.avgdl = num_wd / self.corpus_size if self.corpus_size else 0
        for word, freq in nd.items():
            self.idf[word] = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def get_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
        scores = [0.0] * self.corpus_size
        q_tokens = self._tokenize(query)
        for q in q_tokens:
            if q not in self.idf: 
                continue
            idf = self.idf[q]
            for i, doc in enumerate(self.doc_freqs):
                if q in doc:
                    f = doc[q]
                    num = f * (k1 + 1)
                    den = f + k1 * (1 - b + b * (self.doc_len[i] / self.avgdl))
                    scores[i] += idf * (num / den)
        return scores

class HFEncoder:
    def __init__(
        self, model_name: str, device: Optional[str] = None, max_length: int = 256,
        normalize: bool = True, pooling: str = "mean",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading encoder %s on %s (pooling=%s)", model_name, self.device, pooling)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.pooling = pooling

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        if not texts: return torch.empty((0, EMBED_DIM), device=self.device)
        out = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            if self.pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
            else:
                pooled = hidden[:, 0, :]
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            out.append(pooled)
        return torch.cat(out, dim=0)

    def encode_query(self, query: str) -> torch.Tensor:
        return self.encode([query], batch_size=1).squeeze(0)

class ClinicalBertEncoder(HFEncoder):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", CLINICAL_BERT_ID)
        kwargs.setdefault("pooling", "mean")
        super().__init__(**kwargs)

class MiniLmEncoder(HFEncoder):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", MINILM_ID)
        kwargs.setdefault("pooling", "mean")
        super().__init__(**kwargs)

class MedCPTEncoder:
    def __init__(self, device: Optional[str] = None, max_length: int = 256) -> None:
        self.query_encoder = HFEncoder(model_name=MEDCPT_QUERY_ID, device=device, max_length=max_length, pooling="cls")
        self.doc_encoder = HFEncoder(model_name=MEDCPT_DOC_ID, device=device, max_length=max_length, pooling="cls")
        self.device = self.query_encoder.device
        self.model_name = f"{MEDCPT_QUERY_ID} + {MEDCPT_DOC_ID}"

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        return self.doc_encoder.encode(texts, batch_size=batch_size)

    def encode_query(self, query: str) -> torch.Tensor:
        return self.query_encoder.encode_query(query)

EncoderLike = Union[HFEncoder, MedCPTEncoder]

def build_encoder(name: str, device: Optional[str] = None) -> EncoderLike:
    key = name.lower().strip()
    if key in {"clinicalbert", "bio_clinicalbert", "clinical"}: return ClinicalBertEncoder(device=device)
    if key == "medcpt": return MedCPTEncoder(device=device)
    if key in {"minilm", "generic", "sbert"}: return MiniLmEncoder(device=device)
    raise ValueError(f"Unknown encoder '{name}'.")

@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float

class InMemoryVectorStore:
    def __init__(self, encoder: EncoderLike) -> None:
        self.encoder = encoder
        self._chunks: List[DocumentChunk] = []
        self._embeddings: Optional[torch.Tensor] = None
        self._bm25: Optional[BM25Index] = None

    def add_chunks(self, chunks: Iterable[DocumentChunk], batch_size: int = 32) -> None:
        chunk_list = list(chunks)
        if not chunk_list: return
        logger.info("Embedding %d chunks (batch_size=%d)", len(chunk_list), batch_size)
        new_embeds = self.encoder.encode([c.text for c in chunk_list], batch_size=batch_size)
        
        if self._embeddings is None: self._embeddings = new_embeds
        else: self._embeddings = torch.cat([self._embeddings, new_embeds], dim=0)
            
        self._chunks.extend(chunk_list)
        self._bm25 = BM25Index([c.text for c in self._chunks])

    def __len__(self) -> int: return len(self._chunks)

    def save_index(self, index_dir: Union[str, Path], manifest_extra: Optional[Dict[str, Any]] = None) -> None:
        if self._embeddings is None or not self._chunks: raise RuntimeError("Empty store.")
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cpu_embeds = self._embeddings.detach().cpu().contiguous()
        torch.save(cpu_embeds, out_dir / "embeddings.pt")

        with (out_dir / "chunks.pkl").open("wb") as fh: pickle.dump(self._chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)

        manifest: Dict[str, Any] = {
            "version": INDEX_CACHE_VERSION, "encoder_model_name": getattr(self.encoder, "model_name", "unknown"),
            "n_chunks": len(self._chunks), "embed_dim": int(cpu_embeds.shape[1]), "dtype": str(cpu_embeds.dtype),
        }
        if manifest_extra:
            for k, v in manifest_extra.items():
                if k not in {"version", "n_chunks", "embed_dim", "dtype"}: manifest[k] = v

        with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh: json.dump(manifest, fh, indent=2, sort_keys=True)

    @classmethod
    def load_index(cls, index_dir: Union[str, Path], encoder: "EncoderLike", strict_encoder_match: bool = True) -> "InMemoryVectorStore":
        in_dir = Path(index_dir)
        with (in_dir / "manifest.json").open("r", encoding="utf-8") as fh: manifest = json.load(fh)
        if manifest.get("version") != INDEX_CACHE_VERSION: raise RuntimeError("Cache version mismatch.")

        embeds = torch.load(in_dir / "embeddings.pt", map_location="cpu", weights_only=True)
        with (in_dir / "chunks.pkl").open("rb") as fh: chunks = pickle.load(fh)

        store = cls(encoder=encoder)
        store._embeddings = embeds.to(getattr(encoder, "device", "cpu"))
        store._chunks = list(chunks)
        store._bm25 = BM25Index([c.text for c in store._chunks])
        return store

    @staticmethod
    def cache_exists(index_dir: Union[str, Path]) -> bool:
        d = Path(index_dir)
        return all((d / name).exists() for name in ("manifest.json", "embeddings.pt", "chunks.pkl"))

    @torch.inference_mode()
    def search(
        self, query: str, top_k: int = 5, pool_k: int = 50, retrieval_mode: str = "hybrid",
        reranker: Optional[Any] = None, redundancy_threshold: float = 0.85,
    ) -> List[RetrievedChunk]:
        if self._embeddings is None or len(self._chunks) == 0: return []

        q_embed = self.encoder.encode_query(query)
        dense_scores = (self._embeddings @ q_embed).detach().cpu().numpy()

        if retrieval_mode == "hybrid" and self._bm25 is not None:
            sparse_scores = np.array(self._bm25.get_scores(query))
            dense_rank_map = {idx: r for r, idx in enumerate(np.argsort(dense_scores)[::-1])}
            sparse_rank_map = {idx: r for r, idx in enumerate(np.argsort(sparse_scores)[::-1])}
            
            rrf_scores = np.array([(1.0 / (60 + dense_rank_map[i])) + (1.0 / (60 + sparse_rank_map[i])) for i in range(len(self._chunks))])
            pool_indices = np.argsort(rrf_scores)[::-1][:pool_k]
        else:
            pool_indices = np.argsort(dense_scores)[::-1][:pool_k]

        if reranker is not None:
            pairs = [[query, self._chunks[idx].text] for idx in pool_indices]
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            sorted_pool_idx = np.argsort(ce_scores)[::-1]
            pool_indices = [pool_indices[i] for i in sorted_pool_idx]
            final_scores = sorted(ce_scores, reverse=True)
        else:
            final_scores = [dense_scores[idx] for idx in pool_indices]

        selected, selected_embeds = [], []
        for score, idx in zip(final_scores, pool_indices):
            if len(selected) >= top_k: break
            candidate_embed = self._embeddings[idx]
            if selected_embeds:
                if float((torch.stack(selected_embeds, dim=0) @ candidate_embed).max().item()) > redundancy_threshold: continue
            selected.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))
            selected_embeds.append(candidate_embed)

        return selected

def format_context_block(retrieved: Sequence[RetrievedChunk], include_scores: bool = False) -> str:
    if not retrieved: return "(no retrieved context available)"
    lines = []
    for i, rc in enumerate(retrieved, start=1):
        lines.extend([f"[Doc {i} | cord_uid={rc.chunk.cord_uid}]", rc.chunk.text.strip(), ""])
    return "\n".join(lines).strip()