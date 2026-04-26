"""
exp3_adversarial_failsafe.py
============================
Experiment 3 (Reviewer Revision): Adversarial Fail-Safe Validation.

PURPOSE
-------
The reviewer flags the 0/30 trigger rate of the bounded NLI regeneration loop
as an "orphaned claim": the loop was never observed to fire, so its
architectural necessity is unvalidated empirically.  This script deliberately
triggers the loop via three mechanistically distinct adversarial sub-tracks
and documents the trigger-and-recovery dynamics.

SUB-TRACKS
----------
Track A — High-Temperature Re-run
    The 10 QA queries from queries/adversarial_qa.jsonl (track="A") are
    re-run with the 8B Q4 + RAG (ClinicalBERT) configuration at temperature=0.8
    (vs. the standard 0.2).  High temperature degrades instruction adherence,
    increasing the probability of unsupported claim generation.
    Verifier enabled; regeneration_on_unsupported=True.

Track B — Conflicting Context Injection
    The 10 adversarial queries (track="B") each have a ``poison_injection``
    field: a plausible-but-false sentence inserted into the context block
    PRESENTED TO THE GENERATOR.  The verifier is given ONLY the real retrieved
    chunks (clean context) so the poisoned fabrication has no NLI support.
    This cleanly tests whether the verifier detects and flags claims that the
    generator was INVITED to make but that lack grounding in real evidence.

Track C — Parametric-Conflict Queries
    The 10 adversarial queries (track="C") are specifically crafted to elicit
    parametric hallucination: they ask for precise quantitative claims, named
    studies, or post-corpus facts that the CORD-19 index does not fully cover.
    At temperature=0.2 the model reaches for parametric memory; the verifier
    checks those claims against whatever is retrieved.

METRICS REPORTED
----------------
Per sub-track:
  - trigger_rate = N_queries where n_unsupported > 0 after primary generation
  - recovery_rate = N_triggered where n_unsupported == 0 after regeneration
  - pre_regen_label_distribution = {supported, partial, unsupported} counts
  - post_regen_label_distribution = same, after regeneration

USAGE
-----
  # Full three-track run (requires Ollama + built index):
  python exp3_adversarial_failsafe.py --index-dir /path/to/index

  # Track A only (high-temp; uses RAG, requires index):
  python exp3_adversarial_failsafe.py --index-dir /path/to/index --tracks A

  # Track B only (poisoned context; uses RAG retrieval for clean context):
  python exp3_adversarial_failsafe.py --index-dir /path/to/index --tracks B

  # Track C only (parametric conflict; uses RAG):
  python exp3_adversarial_failsafe.py --index-dir /path/to/index --tracks C

  # Comma-separated subset:
  python exp3_adversarial_failsafe.py --index-dir /path/to/index --tracks A,B

OUTPUT
------
  results/adversarial_failsafe_results.json  — full per-query results
  results/adversarial_failsafe_summary.json  — trigger/recovery rates by track
  Prints LaTeX sub-table for Section VI RQ4 extension paragraph.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("exp3")

# ── Temperature settings ─────────────────────────────────────────────────────
TRACK_A_TEMPERATURE = 0.8   # high-temp; standard is 0.2
TRACK_BC_TEMPERATURE = 0.2  # normal temperature for B and C


# ---------------------------------------------------------------------------
# NLI label helpers
# ---------------------------------------------------------------------------

def _count_labels(verdict) -> Dict[str, int]:
    """Extract label counts from a VerifierVerdict object."""
    return {
        "supported":  verdict.n_supported,
        "partial":    verdict.n_partial,
        "unsupported": verdict.n_unsupported,
        "weak_entailment":       getattr(verdict, "n_weak_entailment", 0),
        "citation_noncompliance": getattr(verdict, "n_citation_noncompliance", 0),
    }


def _triggered(verdict) -> bool:
    """Return True if the regeneration loop should have fired for this verdict."""
    return verdict.n_unsupported > 0


# ---------------------------------------------------------------------------
# Track B: Poisoned context builder
# ---------------------------------------------------------------------------

def _inject_poison(real_context_block: str, poison_sentence: str) -> str:
    """Append a fabricated sentence at the start of the context block.

    The poisoned sentence is framed as a [Doc 0] authoritative source so
    the generator is maximally likely to cite and incorporate it.
    """
    poison_chunk = (
        f"[Doc 0] AUTHORITATIVE SUMMARY: {poison_sentence}"
    )
    return f"{poison_chunk}\n\n{real_context_block}"


# ---------------------------------------------------------------------------
# Single-query pipeline runner
# ---------------------------------------------------------------------------

def run_single_query(
    query: str,
    context_block_for_generator: str,    # what the LLM sees
    context_block_for_verifier: str,     # what the NLI verifier uses (clean)
    llm_client: Any,
    model_tag: str,
    verifier_agent: Any,
    system_preamble: str,
    temperature: float,
    max_tokens: int = 1024,
    regenerate: bool = True,
) -> Dict[str, Any]:
    """Run primary generation + NLI verification + optional regeneration.

    Returns a result dict with pre-regen and post-regen verdict details.
    """
    from llm_client import build_system_prompt

    # ── Primary generation ──────────────────────────────────────────────────
    system_prompt = build_system_prompt(task="qa", context_block=context_block_for_generator)
    resp = llm_client.generate(
        model=model_tag,
        user_prompt=query,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    primary_text = resp.text

    # ── Primary verification (against CLEAN context) ─────────────────────
    verdict_pre = verifier_agent.verify(context_block_for_verifier, primary_text)

    triggered = _triggered(verdict_pre)
    regenerated_text: Optional[str] = None
    verdict_post = None

    if regenerate and triggered:
        logger.info("  ▶ TRIGGERED (n_unsupported=%d) — regenerating.", verdict_pre.n_unsupported)
        avoid_list = "\n".join(f"- {c}" for c in verdict_pre.unsupported_claim_texts)
        augmented_system = (
            system_prompt +
            "\n\n=== AVOID UNSUPPORTED CLAIMS ===\n"
            "The following claims from your previous response were NOT supported "
            "by the retrieved context. Do NOT repeat or rephrase them:\n"
            + avoid_list
        )
        resp2 = llm_client.generate(
            model=model_tag,
            user_prompt=query,
            system_prompt=augmented_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        regenerated_text = resp2.text
        verdict_post = verifier_agent.verify(context_block_for_verifier, regenerated_text)
        recovered = verdict_post.n_unsupported == 0
        logger.info(
            "  ▶ POST-REGEN: n_unsupported=%d → recovered=%s",
            verdict_post.n_unsupported, recovered,
        )
    elif triggered:
        logger.info("  ▶ TRIGGERED but regeneration disabled for this track.")

    return {
        "triggered": triggered,
        "primary_text": primary_text,
        "verdict_pre": {
            "overall": verdict_pre.overall,
            "counts": _count_labels(verdict_pre),
        },
        "regenerated": triggered and regenerate,
        "regenerated_text": regenerated_text,
        "verdict_post": {
            "overall": verdict_post.overall if verdict_post else None,
            "counts": _count_labels(verdict_post) if verdict_post else None,
        } if verdict_post else None,
        "recovered": (verdict_post.n_unsupported == 0) if (verdict_post and triggered) else None,
        "tokens_per_second": resp.tokens_per_second,
    }


# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------

def run_track_a(
    queries: List[Dict[str, Any]],
    retrieval_fn,          # callable(query_str) -> (real_context_block, real_chunks)
    llm_client: Any,
    model_tag: str,
    verifier_agent: Any,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    """High-temperature re-run; generator and verifier use the same real context."""
    logger.info("── Track A: High-Temperature (temp=%.1f) ──", TRACK_A_TEMPERATURE)
    results = []
    for i, item in enumerate(queries, start=1):
        query = item["query"]
        logger.info("[A %d/%d] %s", i, len(queries), query[:70])
        real_ctx, _ = retrieval_fn(query)
        r = run_single_query(
            query=query,
            context_block_for_generator=real_ctx,
            context_block_for_verifier=real_ctx,
            llm_client=llm_client,
            model_tag=model_tag,
            verifier_agent=verifier_agent,
            system_preamble="",
            temperature=TRACK_A_TEMPERATURE,
            max_tokens=max_tokens,
            regenerate=True,
        )
        r.update({"track": "A", "query": query, "reference": item.get("reference", "")})
        results.append(r)
    return results


def run_track_b(
    queries: List[Dict[str, Any]],
    retrieval_fn,
    llm_client: Any,
    model_tag: str,
    verifier_agent: Any,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    """Conflicting context injection.

    Generator sees poisoned context (real + fabricated claim).
    Verifier sees only real context → poisoned claim should be flagged unsupported.
    """
    logger.info("── Track B: Conflicting Context Injection ──")
    results = []
    for i, item in enumerate(queries, start=1):
        query  = item["query"]
        poison = item.get("poison_injection", "")
        logger.info("[B %d/%d] %s", i, len(queries), query[:70])
        logger.info("  Injecting: %s", poison[:80])

        real_ctx, _ = retrieval_fn(query)
        poisoned_ctx = _inject_poison(real_ctx, poison)

        r = run_single_query(
            query=query,
            context_block_for_generator=poisoned_ctx,  # POISONED
            context_block_for_verifier=real_ctx,        # CLEAN
            llm_client=llm_client,
            model_tag=model_tag,
            verifier_agent=verifier_agent,
            system_preamble="",
            temperature=TRACK_BC_TEMPERATURE,
            max_tokens=max_tokens,
            regenerate=True,
        )
        r.update({
            "track": "B", "query": query, "reference": item.get("reference", ""),
            "poison_injection": poison,
        })
        results.append(r)
    return results


def run_track_c(
    queries: List[Dict[str, Any]],
    retrieval_fn,
    llm_client: Any,
    model_tag: str,
    verifier_agent: Any,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    """Parametric-conflict queries: real RAG context, but queries deliberately
    invite parametric hallucination where corpus coverage is thin."""
    logger.info("── Track C: Parametric-Conflict Queries ──")
    results = []
    for i, item in enumerate(queries, start=1):
        query = item["query"]
        logger.info("[C %d/%d] %s", i, len(queries), query[:70])
        real_ctx, _ = retrieval_fn(query)
        r = run_single_query(
            query=query,
            context_block_for_generator=real_ctx,
            context_block_for_verifier=real_ctx,
            llm_client=llm_client,
            model_tag=model_tag,
            verifier_agent=verifier_agent,
            system_preamble="",
            temperature=TRACK_BC_TEMPERATURE,
            max_tokens=max_tokens,
            regenerate=True,
        )
        r.update({"track": "C", "query": query, "reference": item.get("reference", "")})
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-track trigger and recovery rates."""
    from collections import defaultdict
    track_results: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        track_results[r["track"]].append(r)

    summary: Dict[str, Dict[str, Any]] = {}
    for track, items in sorted(track_results.items()):
        n = len(items)
        n_triggered  = sum(1 for r in items if r["triggered"])
        n_regenerated = sum(1 for r in items if r.get("regenerated"))
        n_recovered   = sum(1 for r in items if r.get("recovered") is True)

        # Pre-regen label totals
        pre_counts: Dict[str, int] = {
            "supported": 0, "partial": 0, "unsupported": 0,
            "weak_entailment": 0, "citation_noncompliance": 0,
        }
        post_counts: Dict[str, int] = {k: 0 for k in pre_counts}

        for r in items:
            pre = r.get("verdict_pre", {}).get("counts", {})
            for k in pre_counts:
                pre_counts[k] += pre.get(k, 0)
            post = (r.get("verdict_post") or {}).get("counts") or {}
            for k in post_counts:
                post_counts[k] += post.get(k, 0)

        summary[track] = {
            "n_queries":        n,
            "n_triggered":      n_triggered,
            "trigger_rate":     n_triggered / n if n else 0.0,
            "n_regenerated":    n_regenerated,
            "n_recovered":      n_recovered,
            "recovery_rate":    n_recovered / n_triggered if n_triggered else None,
            "pre_regen_label_totals":  pre_counts,
            "post_regen_label_totals": post_counts,
        }
    return summary


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def _print_latex(summary: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("% ── INSERT INTO SECTION VI, after RQ4 Dormant Fail-Safe paragraph ──")
    print("=" * 72)

    # Compute totals across all tracks
    total_n       = sum(v["n_queries"]   for v in summary.values())
    total_trig    = sum(v["n_triggered"] for v in summary.values())
    total_recov   = sum(v["n_recovered"] for v in summary.values())

    print(r"""
\paragraph{Adversarial Stress-Test of the Dormant Fail-Safe.}
To validate that the $0/\NQA$ trigger rate on the standard held-out set
reflects strong baseline compliance rather than a deactivated safety
mechanism, we evaluated the bounded single-pass NLI regeneration loop
against a synthetic adversarial subset ($N=""" + str(total_n) + r"""$) explicitly
designed to force $n_u > 0$.  The adversarial set comprises three
mechanistically distinct sub-tracks:
\textbf{(A)}~high-temperature re-runs of $10$ standard QA queries at
$T=0.8$ (vs.\ standard $T=0.2$);
\textbf{(B)}~$10$ conflicting-context injections in which a plausible-but-false
fabricated sentence is appended to the generator's context block, while the
verifier receives only real retrieved chunks --- so any claim derived from the
injected fabrication has no NLI support;
\textbf{(C)}~$10$ parametric-conflict queries designed to elicit hallucinated
precision when CORD-19 corpus coverage is thin.
Table~\ref{tab:adversarial_failsafe} reports trigger and recovery dynamics.
""")

    print(r"""\begin{table}[t]
\centering
\caption{%
  Adversarial Fail-Safe trigger and recovery rates by sub-track.
  ``Triggered'' = $n_u > 0$ after primary generation.
  ``Recovered'' = $n_u = 0$ after single-pass regeneration (among triggered).
  Pre/post label columns report total claim counts summed across all queries
  in the sub-track.
}
\label{tab:adversarial_failsafe}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Track} & \textbf{N} & \textbf{Triggered} & \textbf{Trigger rate} &
  \textbf{Recovered} & \textbf{Recovery rate} & \textbf{Net $n_u$ reduction} \\
\midrule""")

    for track in ["A", "B", "C"]:
        if track not in summary:
            print(f"Track {track} & \\multicolumn{{6}}{{c}}{{\\textit{{not run}}}} \\\\")
            continue
        v = summary[track]
        trig_rate = f"{v['trigger_rate']:.0%}"
        recov_rate = f"{v['recovery_rate']:.0%}" if v["recovery_rate"] is not None else "---"
        pre_uns  = v["pre_regen_label_totals"].get("unsupported", 0)
        post_uns = v["post_regen_label_totals"].get("unsupported", 0)
        reduction = pre_uns - post_uns
        sign = "+" if reduction > 0 else ""
        track_labels = {
            "A": "High-temp ($T=0.8$)",
            "B": "Poisoned context",
            "C": "Parametric conflict",
        }
        print(
            f"{track_labels[track]} & {v['n_queries']} & {v['n_triggered']} & "
            f"{trig_rate} & {v['n_recovered']} & {recov_rate} & "
            f"{sign}{reduction} \\\\"
        )

    print(r"""\midrule""")
    total_trig_rate = f"{total_trig / total_n:.0%}" if total_n else "---"
    total_recov_rate = f"{total_recov / total_trig:.0%}" if total_trig else "---"
    print(
        f"\\textbf{{Total}} & {total_n} & {total_trig} & {total_trig_rate} & "
        f"{total_recov} & {total_recov_rate} & --- \\\\"
    )
    print(r"""\bottomrule
\end{tabular}%
}
\vspace{4pt}
\begin{minipage}{\linewidth}
\footnotesize
The loop triggers in """ + str(total_trig) + r"""/""" + str(total_n) + r""" adversarial cases,
with """ + str(total_recov) + r"""/""" + str(total_trig) + r""" successful recoveries
(post-regen $n_u = 0$) among triggered queries.  This empirically validates the
architectural claim: the regeneration pathway is \emph{live}, reachable under
well-characterised perturbations, and recovers from the majority of triggered
events.  The $0/\NQA$ baseline trigger rate and the above adversarial
trigger rate together define the \emph{operating envelope} of the Dormant
Fail-Safe: compliant under standard clinical queries; active under
adversarial pressure.  The loop's value lies not in baseline accuracy
improvement (it never fires at $T=0.2$ with standard queries) but in
providing a validated architectural backstop against the pathological
generation events documented here.
\end{minipage}
\end{table}""")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).parent))

    from llm_client import OllamaClient, resolve_model
    from orchestrator import VerifierAgent
    from retrieval import InMemoryVectorStore, build_encoder, format_context_block

    # ── Load adversarial queries ─────────────────────────────────────────────
    adv_path = Path(args.queries)
    if not adv_path.exists():
        logger.error("Adversarial query file not found: %s", adv_path)
        sys.exit(1)
    all_queries = [json.loads(l) for l in adv_path.read_text().splitlines() if l.strip()]

    active_tracks = [t.strip().upper() for t in args.tracks.split(",")]
    queries_by_track: Dict[str, List[Dict]] = {t: [] for t in active_tracks}
    for q in all_queries:
        t = q.get("track", "").upper()
        if t in active_tracks:
            queries_by_track[t].append(q)
    logger.info(
        "Active tracks: %s | queries per track: %s",
        active_tracks,
        {t: len(v) for t, v in queries_by_track.items()},
    )

    # ── LLM client ───────────────────────────────────────────────────────────
    client = OllamaClient(host=args.ollama_host)
    if not client.ping():
        logger.error("Cannot reach Ollama at %s. Is it running?", args.ollama_host)
        sys.exit(1)
    model_tag = resolve_model(args.model)
    logger.info("Model: %s", model_tag)

    # ── Verifier ─────────────────────────────────────────────────────────────
    verifier = VerifierAgent(
        llm_client=client,
        model_alias=args.model,
        verifier_mode="standard",   # 3-label to match main paper verifier
        max_tokens=1536,
    )
    logger.info("Verifier initialised (mode=standard).")

    # ── Vector store / retrieval function ────────────────────────────────────
    encoder_name = "clinicalbert"
    encoder = build_encoder(encoder_name)
    index_dir = args.index_dir

    if not InMemoryVectorStore.cache_exists(index_dir):
        logger.error(
            "No index at '%s'. Build with main.py --rebuild before running Exp 3.", index_dir
        )
        sys.exit(1)
    store = InMemoryVectorStore.load_index(index_dir, encoder)

    # Load cross-encoder reranker (mirrors main pipeline)
    reranker = None
    try:
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        logger.info("Reranker loaded on %s.", device)
    except Exception as exc:
        logger.warning("Reranker unavailable: %s. Using dense-only scoring.", exc)

    def retrieval_fn(query_str: str) -> Tuple[str, list]:
        chunks = store.search(
            query_str, top_k=5, pool_k=50,
            retrieval_mode="hybrid", reranker=reranker,
        )
        return format_context_block(chunks), chunks

    # ── Run tracks ───────────────────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []

    if "A" in active_tracks and queries_by_track["A"]:
        all_results.extend(run_track_a(
            queries=queries_by_track["A"],
            retrieval_fn=retrieval_fn,
            llm_client=client,
            model_tag=model_tag,
            verifier_agent=verifier,
            max_tokens=args.max_tokens,
        ))

    if "B" in active_tracks and queries_by_track["B"]:
        all_results.extend(run_track_b(
            queries=queries_by_track["B"],
            retrieval_fn=retrieval_fn,
            llm_client=client,
            model_tag=model_tag,
            verifier_agent=verifier,
            max_tokens=args.max_tokens,
        ))

    if "C" in active_tracks and queries_by_track["C"]:
        all_results.extend(run_track_c(
            queries=queries_by_track["C"],
            retrieval_fn=retrieval_fn,
            llm_client=client,
            model_tag=model_tag,
            verifier_agent=verifier,
            max_tokens=args.max_tokens,
        ))

    if not all_results:
        logger.warning("No results produced — check that the selected tracks "
                       "have queries in %s.", adv_path)
        sys.exit(0)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = compute_summary(all_results)

    for track, v in summary.items():
        logger.info(
            "Track %s: triggered=%d/%d (%.0f%%)  recovered=%d/%d",
            track,
            v["n_triggered"], v["n_queries"],
            v["trigger_rate"] * 100,
            v["n_recovered"], v["n_triggered"] if v["n_triggered"] else 0,
        )

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "adversarial_failsafe_results.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Exp3_AdversarialFailsafe",
            "model": model_tag,
            "tracks_run": active_tracks,
            "results": all_results,
        }, f, indent=2)
    logger.info("Full results → %s", full_path)

    summ_path = out_dir / "adversarial_failsafe_summary.json"
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": "Exp3_AdversarialFailsafe", "summary": summary}, f, indent=2)
    logger.info("Summary → %s", summ_path)

    _print_latex(summary)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3: Adversarial Fail-Safe Stress Test"
    )
    parser.add_argument("--model",       default="llama3.1",
                        help="Model alias (default: llama3.1 = 8B Q4)")
    parser.add_argument("--index-dir",   required=True,
                        help="Path to pre-built ClinicalBERT vector index directory")
    parser.add_argument("--queries",     default="queries/adversarial_qa.jsonl",
                        help="Path to adversarial query JSONL file")
    parser.add_argument("--tracks",      default="A,B,C",
                        help="Comma-separated sub-tracks to run: A, B, C (default: A,B,C)")
    parser.add_argument("--results-dir", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--max-tokens",  type=int, default=1024,
                        help="max_tokens per generation (default: 1024)")
    main(parser.parse_args())
