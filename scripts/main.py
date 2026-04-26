"""
main.py
=======
Command-line entry point for P.R.O.T.E.U.S. / H.E.R.A. evaluations.
(EMNLP Upgraded version)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from data_processing import build_chunk_corpus
from evaluation import (
    AtomicFactExtractor,
    ENTITY_CLASSES,
    KeywordScores,
    LatencyTracker,
    NliEnsembleHallucinationDetector,
    QaEvaluator,
    QaScores,
    aggregate_keyword_scores,
    pairwise_condition_report,
    score_keywords,
)
from llm_client import AnthropicClient, OllamaClient
from orchestrator import PipelineResult, ProteusOrchestrator, VerifierAgent
from retrieval import EncoderLike, InMemoryVectorStore, build_encoder

logger = logging.getLogger("proteus")

def load_or_build_vector_store(
    encoder: EncoderLike, index_dir: str, rag_enabled: bool, rebuild: bool = False,
    dataset_id: str = "pritamdeka/cord-19-fulltext", dataset_config: Optional[str] = None,
    dataset_split: str = "train", max_documents: Optional[int] = None,
    shuffle_seed: int = 42, shuffle_buffer: int = 10_000, hf_token: Optional[str] = None,
    tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
) -> InMemoryVectorStore:
    if not rag_enabled: return InMemoryVectorStore(encoder=encoder)
    if not rebuild and InMemoryVectorStore.cache_exists(index_dir):
        logger.info("Found existing embedding cache at %s — loading.", index_dir)
        return InMemoryVectorStore.load_index(index_dir, encoder)

    chunks = build_chunk_corpus(
        dataset_id=dataset_id, config=dataset_config, split=dataset_split,
        max_documents=max_documents, shuffle_seed=shuffle_seed, shuffle_buffer=shuffle_buffer,
        hf_token=hf_token, tokenizer_name=tokenizer_name,
    )
    if not chunks: raise RuntimeError("Stream produced zero usable chunks.")

    store = InMemoryVectorStore(encoder=encoder)
    store.add_chunks(chunks)
    store.save_index(index_dir, manifest_extra={
        "dataset_id": dataset_id, "dataset_config": dataset_config, "dataset_split": dataset_split,
        "max_documents": max_documents, "shuffle_seed": shuffle_seed,
    })
    return store

def load_queries(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"Query file not found: {path}")
    raw = p.read_text(encoding="utf-8").strip()
    if not raw: return []
    if p.suffix == ".jsonl": return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    return data if isinstance(data, list) else [data]

def run_qa_eval(
    orchestrator: ProteusOrchestrator,
    queries: List[Dict[str, Any]],
    evaluator: QaEvaluator,
    latency: LatencyTracker,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    per_example, score_objs = [], []

    for i, item in enumerate(queries, start=1):
        query, reference = item["query"], item.get("reference", "")
        logger.info("[QA %d/%d] %s", i, len(queries), query)

        result: PipelineResult = orchestrator.run(
            query, task="qa", temperature=temperature, max_tokens=max_tokens
        )
        latency.record(
            wall_seconds=result.llm_response.eval_duration_s,
            tokens=result.llm_response.eval_tokens,
        )

        # CRITICAL FIX: Pass individual chunks to support Max-Pooling Entailment
        retrieved_texts = [rc.chunk.text for rc in result.retrieved]

        scores = evaluator.compute(
            generated=result.text,
            reference=reference,
            retrieved_texts=retrieved_texts,
            query_id=f"q_{i:03d}",   # used by pairs-log for NLI calibration
        )
        score_objs.append(scores)

        per_example.append({
            "query": query, "reference": reference, "generated": result.text,
            "scores": {
                "rouge1_f": scores.rouge1_f, "rouge2_f": scores.rouge2_f,
                "rougeL_f": scores.rougeL_f, "bertscore_f1": scores.bertscore_f1,
                "hallucination_rate": scores.hallucination_rate,
                "n_sentences": scores.n_sentences, "n_hallucinated": scores.n_hallucinated,
                "parse_failed": scores.parse_failed,
            },
            "retrieved": [
                {"chunk_id": rc.chunk.chunk_id, "score": rc.score}
                for rc in result.retrieved
            ],
            "verifier": {
                "overall": result.verifier_verdict.overall,
                "n_supported": result.verifier_verdict.n_supported,
                "n_partial": result.verifier_verdict.n_partial,
                "n_unsupported": result.verifier_verdict.n_unsupported,
                "n_weak_entailment": result.verifier_verdict.n_weak_entailment,
                "n_citation_noncompliance": result.verifier_verdict.n_citation_noncompliance,
            } if result.verifier_verdict else None,
            "regenerated": result.regenerated,
            "tokens_per_second": result.llm_response.tokens_per_second,
        })

    return {
        "aggregate": evaluator.aggregate(score_objs),
        "latency": latency.summary(),
        "per_example": per_example,
    }

def run_keyword_eval(
    orchestrator: ProteusOrchestrator,
    queries: List[Dict[str, Any]],
    latency: LatencyTracker,
    temperature: float = 0.0,   # deterministic for JSON output
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    per_example, score_objs = [], []
    for i, item in enumerate(queries, start=1):
        query = item["query"]
        gold = item.get("gold", {cls: [] for cls in ENTITY_CLASSES})
        logger.info("[KW %d/%d] %s", i, len(queries), query)

        result: PipelineResult = orchestrator.run(
            query, task="keywords", temperature=temperature, max_tokens=max_tokens
        )
        latency.record(
            wall_seconds=result.llm_response.eval_duration_s,
            tokens=result.llm_response.eval_tokens,
        )
        predicted = result.parsed_output or {cls: [] for cls in ENTITY_CLASSES}
        scores = score_keywords(predicted=predicted, gold=gold)
        score_objs.append(scores)
        per_example.append({
            "query": query, "gold": gold, "predicted": predicted,
            "raw_output": result.text,
            "scores": {"per_class": scores.per_class, "micro": scores.micro},
            "tokens_per_second": result.llm_response.tokens_per_second,
        })
    return {
        "aggregate": aggregate_keyword_scores(score_objs),
        "latency": latency.summary(),
        "per_example": per_example,
    }

def _label_for(results: Dict[str, Any]) -> str:
    return f"{results.get('model', '?')}_{'rag' if results.get('rag_enabled') else 'norag'}_{results.get('retriever', '?')}_{'ver' if results.get('verifier_enabled') else 'nover'}"

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="proteus", description="P.R.O.T.E.U.S. / H.E.R.A. EMNLP-Grade Pipeline.")
    p.add_argument("--task", choices=("qa", "keywords"), required=True)
    p.add_argument("--model", default="llama3.1")
    p.add_argument("--rag", choices=("on", "off"), default="on")
    p.add_argument("--retriever", choices=("clinicalbert", "medcpt", "minilm"), default="clinicalbert")
    p.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default="hybrid")
    p.add_argument("--reranker", choices=("none", "cross-encoder"), default="cross-encoder")
    p.add_argument("--context-assembly", choices=("sequential", "lost-in-the-middle"), default="lost-in-the-middle")
    p.add_argument("--verifier", choices=("on", "off"), default="on")
    p.add_argument("--verifier-mode", choices=("standard", "split-partial"), default="standard",
                   help="'standard': original 3-label prompt. "
                        "'split-partial': 4-label prompt separating weak_entailment from "
                        "citation_noncompliance (resolves the §3.4 label-conflation confound).")
    p.add_argument("--regenerate-on-unsupported", action="store_true")
    p.add_argument("--ensemble-hallucination", choices=("on", "off"), default="on")
    p.add_argument("--atomic-facts", choices=("on", "off"), default="on")
    p.add_argument("--queries", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hf-dataset", default="pritamdeka/cord-19-fulltext")
    p.add_argument("--hf-config", default=None)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--max-documents", type=int, default=None)
    p.add_argument("--pool-k", type=int, default=50)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--redundancy-threshold", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--backend", choices=("ollama", "anthropic"), default="ollama",
                   help="LLM backend. 'anthropic' routes generation through the Anthropic "
                        "Messages API (cloud upper-bound baseline; public corpus only).")
    p.add_argument("--api-key", default=None,
                   help="Anthropic API key (required when --backend anthropic). "
                        "Alternatively set the ANTHROPIC_API_KEY environment variable.")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--significance-against", nargs="*", default=None)
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument(
        "--pairs-log", default=None, metavar="FILE",
        help="When set, every (atomic_claim, chunk, nli_decision) triple from "
             "h_ctx scoring is appended to this JSONL file. "
             "Used to sample N=50 NLI-unsupported pairs for the manual FNR "
             "calibration annotation task (TODO #1 in the paper). "
             "Only active for --task qa runs. "
             "Example: --pairs-log results/nli_pairs_raw.jsonl",
    )
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_encoder(args.retriever, device=device)
    index_dir = str(Path(args.cache_dir) / f"index.{args.retriever}")

    hf_model_map = {"clinicalbert": "emilyalsentzer/Bio_ClinicalBERT", "medcpt": "ncbi/MedCPT-Query-Encoder", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
    tokenizer_name = hf_model_map.get(args.retriever, "emilyalsentzer/Bio_ClinicalBERT")

    store = load_or_build_vector_store(
        encoder=encoder, index_dir=index_dir, rag_enabled=(args.rag == "on"), rebuild=args.rebuild_index,
        dataset_id=args.hf_dataset, dataset_config=args.hf_config, dataset_split=args.hf_split,
        max_documents=args.max_documents, shuffle_seed=args.shuffle_seed, shuffle_buffer=args.shuffle_buffer,
        tokenizer_name=tokenizer_name,
    )

    # ── LLM client selection ───────────────────────────────────────────────
    if args.backend == "anthropic":
        import os
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        jupyter_token = os.environ.get("JUPYTER_API_TOKEN")

        # Validation: Either we have an API key, or we are using the local hub proxy
        if not api_key and not jupyter_token:
            logger.error(
                "--backend anthropic requires --api-key, ANTHROPIC_API_KEY, "
                "or JUPYTER_API_TOKEN (for local mock proxy) to be set."
            )
            return 1
            
        # Supply a dummy key if utilizing the JupyterHub bypass
        api_key = api_key or "dummy-key-for-proxy"

        llm = AnthropicClient(api_key=api_key)
        logger.info("Using Anthropic API backend (model=%s)", args.model)
    else:
        llm = OllamaClient(host=args.ollama_host)

    verifier = (
        VerifierAgent(
            llm_client=llm,
            model_alias=args.model,
            verifier_mode=args.verifier_mode,
        )
        if args.verifier == "on" else None
    )

    orchestrator = ProteusOrchestrator(
        vector_store=store, llm_client=llm, model_alias=args.model,
        top_k=args.top_k, pool_k=args.pool_k, retrieval_mode=args.retrieval_mode,
        use_reranker=(args.reranker == "cross-encoder"), context_assembly=args.context_assembly,
        redundancy_threshold=args.redundancy_threshold, rag_enabled=(args.rag == "on"),
        verifier=verifier, regenerate_on_unsupported=args.regenerate_on_unsupported,
    )

    queries = load_queries(args.queries)
    latency = LatencyTracker()

    if args.task == "qa":
        detector = NliEnsembleHallucinationDetector(device=device) if args.ensemble_hallucination == "on" else None
        extractor = AtomicFactExtractor(llm_client=llm, model_alias=args.model) if args.atomic_facts == "on" else None
        evaluator = QaEvaluator(
            device=device,
            hallucination_detector=detector,
            fact_extractor=extractor,
            pairs_log_path=args.pairs_log,   # None by default; set to enable FNR logging
        )
        results = run_qa_eval(
            orchestrator, queries, evaluator, latency,
            temperature=args.temperature, max_tokens=args.max_tokens,
        )
    else:
        results = run_keyword_eval(
            orchestrator, queries, latency,
            temperature=args.temperature, max_tokens=args.max_tokens,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_meta = {
        "task": args.task, "model": args.model, "backend": args.backend,
        "rag_enabled": args.rag == "on", "verifier_enabled": args.verifier == "on",
        "verifier_mode": args.verifier_mode,
        "retriever": args.retriever, "retrieval_mode": args.retrieval_mode, "reranker": args.reranker,
        "context_assembly": args.context_assembly, "hf_dataset": args.hf_dataset, "max_documents": args.max_documents,
        "top_k": args.top_k, "n_queries": len(queries),
        "pairs_log": args.pairs_log,   # null unless --pairs-log was supplied
        **results,
    }

    out_path.write_text(json.dumps(results_meta, indent=2), encoding="utf-8")
    logger.info("Wrote results to %s", out_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())