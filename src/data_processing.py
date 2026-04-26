"""
data_processing.py
==================

Data ingestion module for P.R.O.T.E.U.S.

The corpus is streamed from the Hugging Face Hub (``allenai/cord19``).
No local corpus archive is required. AI inference (ClinicalBERT / MedCPT
/ MiniLM encoders, Ollama) remains strictly on-device — only the raw
text of the biomedical corpus is fetched over the network, and only
during the one-time index build.

Privacy note
------------
Streaming fetches *corpus documents* (public research papers) from
the Hugging Face CDN. It does **not** transmit user queries, retrieved
context, or any PHI. The zero-egress invariant for user-facing traffic
is preserved: once the index is built, subsequent runs are fully
offline.

Pipeline
--------
    stream(allenai/cord19) -> chunk -> deduplicate -> List[DocumentChunk]
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

#: Default Hugging Face dataset identifier streamed by the loader.
#:
#: We use ``pritamdeka/cord-19-fulltext`` — a community mirror of CORD-19
#: that stores the full-text of ~369k papers as a single Parquet file.
#: This matters because the original ``allenai/cord19`` uses a custom
#: loader script that downloads and extracts a tar.gz archive, which
#: Hugging Face's streaming API does not support (it raises
#: NotImplementedError on any tar format). See HF datasets issue #4697.
#: Parquet, by contrast, streams cleanly row-group by row-group over HTTP.
DEFAULT_HF_DATASET = "pritamdeka/cord-19-fulltext"

#: Default config name. The parquet mirror has no configs (one flat table),
#: so this is ``None``. Only populated for datasets like ``allenai/cord19``
#: that expose ``BuilderConfig`` choices.
DEFAULT_HF_CONFIG: Optional[str] = None

#: Default split name.
DEFAULT_HF_SPLIT = "train"

#: Field-name mapping from the HF dataset schema onto our canonical
#: document record. The ``pritamdeka/cord-19-fulltext`` mirror exposes a
#: single ``fulltext`` column; we route that to our ``body`` key. Title,
#: abstract, and cord_uid are not available in this mirror — we synthesize
#: a cord_uid from the row index at load time and leave title/abstract
#: empty (the chunker handles missing fields gracefully).
#:
#: If you point the loader at a different mirror whose column names
#: differ (e.g. ``allenai/cord19`` with cord_uid/title/abstract/body_text),
#: override this mapping via ``HFStreamingLoader(field_map=...)``.
DEFAULT_FIELD_MAP: Dict[str, str] = {
    "fulltext": "body",
}


# ---------------------------------------------------------------------------
# Document container
# ---------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """A single retrievable unit from the corpus.

    Attributes
    ----------
    chunk_id : str
        Deterministic identifier ``{cord_uid}::{section}::{idx}``.
    cord_uid : str
        Source CORD-19 document identifier.
    section : str
        One of ``{"title", "abstract", "body"}``.
    text : str
        Raw chunk text (already truncated to the token budget).
    token_count : int
        Exact token count under the chunker's tokenizer.
    metadata : dict
        Optional bibliographic metadata (authors, publish_time, url, title).
    """

    chunk_id: str
    cord_uid: str
    section: str
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hugging Face streaming loader
# ---------------------------------------------------------------------------
class HFStreamingLoader:
    """Stream CORD-19 records from the Hugging Face Hub.

    Uses ``datasets.load_dataset(..., streaming=True)`` so no local
    archive is required. The stream is shuffled with a deterministic seed
    before ``take()`` so ``max_documents`` yields a reproducible
    distribution across the corpus rather than the first N insertion-order
    records.

    Hugging Face caches streamed shards client-side under
    ``~/.cache/huggingface/datasets/``, so repeated full builds do not
    re-download over the network — the first build is the only
    network-heavy step.

    Parameters
    ----------
    dataset_id : str
        Hugging Face dataset id.
    split : str
        Split name (``"train"`` is standard for CORD-19 mirrors).
    max_documents : int, optional
        Hard cap on documents streamed. ``None`` means the full corpus.
    shuffle_seed : int
        Seed for the deterministic stream-level shuffle.
    shuffle_buffer : int
        Buffer size for the approximate shuffle. Larger = better mixing,
        more memory. 10,000 is a sensible default for biomedical corpora.
    field_map : dict, optional
        Mapping from dataset column names to our canonical keys. Defaults
        to the ``allenai/cord19`` schema.
    hf_token : str, optional
        Hugging Face auth token. If ``None``, reads ``HF_TOKEN`` from the
        environment. If both are absent, streams anonymously (subject to
        tighter rate limits).
    """

    def __init__(
        self,
        dataset_id: str = DEFAULT_HF_DATASET,
        config: Optional[str] = DEFAULT_HF_CONFIG,
        split: str = DEFAULT_HF_SPLIT,
        max_documents: Optional[int] = None,
        shuffle_seed: int = 42,
        shuffle_buffer: int = 10_000,
        field_map: Optional[Dict[str, str]] = None,
        hf_token: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.dataset_id = dataset_id
        self.config = config
        self.split = split
        self.max_documents = max_documents
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer = shuffle_buffer
        self.field_map = field_map or DEFAULT_FIELD_MAP
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.trust_remote_code = trust_remote_code

    # ------------------------------------------------------------------
    def iter_documents(self) -> Iterator[dict]:
        """Yield normalized document records, one at a time.

        Yields
        ------
        dict
            Keys: ``cord_uid``, ``title``, ``abstract``, ``body``, ``metadata``.
        """
        # Import locally so the module import does not pull the full
        # `datasets` dependency when users aren't streaming (e.g. when
        # loading a pre-built cache only).
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The `datasets` package is required for streaming. "
                "Install with: pip install datasets"
            ) from exc

        auth_tag = "authenticated" if self.hf_token else "anonymous"
        logger.info(
            "Streaming %s [config=%s, split=%s, shuffle_seed=%d, buffer=%d, max=%s, %s]",
            self.dataset_id,
            self.config or "default",
            self.split,
            self.shuffle_seed,
            self.shuffle_buffer,
            self.max_documents if self.max_documents is not None else "unbounded",
            auth_tag,
        )

        load_kwargs: dict = {
            "streaming": True,
            "split": self.split,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        # Config (e.g. "fulltext") must be passed positionally as the
        # second argument to load_dataset — it can't go through **kwargs
        # with the name "name" in all `datasets` versions.
        load_args = [self.dataset_id]
        if self.config:
            load_args.append(self.config)

        try:
            ds = load_dataset(*load_args, **load_kwargs)
        except Exception as exc:  # noqa: BLE001 — network/auth errors vary
            raise RuntimeError(
                f"Failed to open HF stream for {self.dataset_id} "
                f"(config={self.config!r}): {exc}. "
                "Check network connectivity and (if required) HF_TOKEN."
            ) from exc

        # Shuffle -> take, in that order. Shuffle is buffer-approximate
        # but deterministic for a fixed seed.
        ds = ds.shuffle(seed=self.shuffle_seed, buffer_size=self.shuffle_buffer)
        if self.max_documents is not None:
            ds = ds.take(self.max_documents)

        n_yielded = 0
        for record in ds:
            normalized = self._normalize(record)
            # Some mirrors (e.g. pritamdeka/cord-19-fulltext) don't provide
            # a cord_uid column. Synthesize a stable one from the row index
            # so downstream chunk_id formatting still works.
            if not normalized["cord_uid"]:
                normalized["cord_uid"] = f"row_{n_yielded:08d}"
            yield normalized
            n_yielded += 1
            if n_yielded % 1000 == 0:
                logger.info("Streamed %d documents so far", n_yielded)

        logger.info("Stream complete: %d documents yielded", n_yielded)

    # ------------------------------------------------------------------
    def _normalize(self, record: dict) -> dict:
        """Translate a raw HF record through ``field_map`` into our schema.

        Any body-text field provided as a list of passages is joined with
        newlines. Missing fields become empty strings or empty dicts.
        """
        def _get(canonical: str) -> str:
            for src, dst in self.field_map.items():
                if dst == canonical:
                    val = record.get(src, "")
                    if isinstance(val, list):
                        # Some mirrors store body_text as a list of section dicts
                        parts: List[str] = []
                        for item in val:
                            if isinstance(item, dict):
                                parts.append(str(item.get("text", "")))
                            else:
                                parts.append(str(item))
                        return "\n".join(p for p in parts if p).strip()
                    return str(val or "").strip()
            return ""

        return {
            "cord_uid": _get("cord_uid"),
            "title": _get("title"),
            "abstract": _get("abstract"),
            "body": _get("body"),
            "metadata": {
                "authors": _get("authors"),
                "publish_time": _get("publish_time"),
                "url": _get("url"),
            },
        }


# ---------------------------------------------------------------------------
# Chunker (unchanged from offline pipeline)
# ---------------------------------------------------------------------------
class Chunker:
    """Token-aware chunker backed by a Hugging Face tokenizer.

    Parameters
    ----------
    tokenizer_name : str
        Hugging Face model id whose tokenizer defines the token budget.
    max_tokens : int
        Maximum tokens per chunk.
    stride : int
        Overlap tokens between consecutive chunks in the body text.
    """

    def __init__(
        self,
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_tokens: int = 256,
        stride: int = 32,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= stride < max_tokens:
            raise ValueError("stride must satisfy 0 <= stride < max_tokens")
        logger.info("Loading chunker tokenizer %s", tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.stride = stride

    def chunk_document(self, document: dict) -> List[DocumentChunk]:
        """Turn a single document record into a list of chunks."""
        chunks: List[DocumentChunk] = []
        cord_uid = document["cord_uid"]
        meta = {**document.get("metadata", {}), "title": document.get("title", "")}

        if document.get("title"):
            chunks.extend(
                self._section_to_chunks(document["title"], cord_uid, "title", meta)
            )
        if document.get("abstract"):
            chunks.extend(
                self._section_to_chunks(document["abstract"], cord_uid, "abstract", meta)
            )
        if document.get("body"):
            chunks.extend(
                self._section_to_chunks(document["body"], cord_uid, "body", meta)
            )
        return chunks

    def _section_to_chunks(
        self,
        text: str,
        cord_uid: str,
        section: str,
        metadata: dict,
    ) -> List[DocumentChunk]:
        """Sliding-window tokenization honoring ``max_tokens`` and ``stride``."""
        text = text.strip()
        if not text:
            return []

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return []

        window = self.max_tokens
        step = max(1, window - self.stride)
        chunks: List[DocumentChunk] = []
        idx = 0
        for start in range(0, len(ids), step):
            piece = ids[start : start + window]
            if not piece:
                break
            decoded = self.tokenizer.decode(piece, skip_special_tokens=True).strip()
            if not decoded:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{cord_uid}::{section}::{idx}",
                    cord_uid=cord_uid,
                    section=section,
                    text=decoded,
                    token_count=len(piece),
                    metadata=metadata,
                )
            )
            idx += 1
            if start + window >= len(ids):
                break
        return chunks


# ---------------------------------------------------------------------------
# Near-duplicate filter (unchanged)
# ---------------------------------------------------------------------------
class NearDuplicateFilter:
    """Shingle-hash based near-duplicate detector."""

    def __init__(self, shingle_size: int = 5, threshold: float = 0.85) -> None:
        if shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.shingle_size = shingle_size
        self.threshold = threshold
        self._seen_signatures: List[Set[int]] = []
        self._buckets: Dict[int, List[int]] = {}

    @staticmethod
    def _hash_shingle(shingle: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
            signed=False,
        )

    def _signature(self, text: str) -> Set[int]:
        tokens = text.lower().split()
        if len(tokens) < self.shingle_size:
            return {self._hash_shingle(" ".join(tokens))} if tokens else set()
        return {
            self._hash_shingle(" ".join(tokens[i : i + self.shingle_size]))
            for i in range(len(tokens) - self.shingle_size + 1)
        }

    def is_duplicate(self, text: str) -> bool:
        """Return ``True`` if ``text`` is a near-duplicate of a prior chunk."""
        sig = self._signature(text)
        if not sig:
            return True

        candidate_ids: Set[int] = set()
        for h in sig:
            candidate_ids.update(self._buckets.get(h, ()))

        for cid in candidate_ids:
            other = self._seen_signatures[cid]
            inter = len(sig & other)
            if inter == 0:
                continue
            union = len(sig | other)
            if inter / union >= self.threshold:
                return True

        new_idx = len(self._seen_signatures)
        self._seen_signatures.append(sig)
        for h in sig:
            self._buckets.setdefault(h, []).append(new_idx)
        return False


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------
def build_chunk_corpus(
    dataset_id: str = DEFAULT_HF_DATASET,
    config: Optional[str] = DEFAULT_HF_CONFIG,
    split: str = DEFAULT_HF_SPLIT,
    tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_documents: Optional[int] = None,
    max_tokens: int = 256,
    stride: int = 32,
    shingle_size: int = 5,
    dedup_threshold: float = 0.85,
    shuffle_seed: int = 42,
    shuffle_buffer: int = 10_000,
    field_map: Optional[Dict[str, str]] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
) -> List[DocumentChunk]:
    """Stream -> chunk -> deduplicate. Single entry point for index builds.

    Returns
    -------
    list[DocumentChunk]
        Ready-to-embed chunks.
    """
    loader = HFStreamingLoader(
        dataset_id=dataset_id,
        config=config,
        split=split,
        max_documents=max_documents,
        shuffle_seed=shuffle_seed,
        shuffle_buffer=shuffle_buffer,
        field_map=field_map,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    chunker = Chunker(
        tokenizer_name=tokenizer_name, max_tokens=max_tokens, stride=stride
    )
    dedup = NearDuplicateFilter(shingle_size=shingle_size, threshold=dedup_threshold)

    kept: List[DocumentChunk] = []
    n_seen = 0
    n_docs = 0
    for doc in loader.iter_documents():
        n_docs += 1
        for chunk in chunker.chunk_document(doc):
            n_seen += 1
            if dedup.is_duplicate(chunk.text):
                continue
            kept.append(chunk)
        if n_docs % 500 == 0:
            logger.info(
                "Processed %d docs -> %d chunks seen, %d kept",
                n_docs,
                n_seen,
                len(kept),
            )

    logger.info(
        "Ingest complete: %d documents, %d raw chunks, %d unique chunks",
        n_docs,
        n_seen,
        len(kept),
    )
    return kept


def iter_chunk_texts(chunks: Iterable[DocumentChunk]) -> Iterator[str]:
    """Convenience: yield the raw text of a chunk stream."""
    for c in chunks:
        yield c.text
