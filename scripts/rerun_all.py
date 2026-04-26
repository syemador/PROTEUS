"""
rerun_all.py
============
Master orchestration script that runs every evaluation configuration needed
to close TODOs #1 – #6 in the paper.

Phases
------
1  Re-run all 6 QA configs (local Ollama)
     → Generates qa_*.json with parse_failed field (TODO #2)
     → Fresh data for h_ref scoring (TODO #3/#4)
2  Run all 5 keyword configs (local Ollama)  (TODO #5)
3  Compute h_ref on the 6 QA outputs; print Table 2 paired stats (TODO #3/#4)
4  Run Claude API baselines — split into four independently runnable sub-phases:
   4.1 (--phases 41)  Claude QA  + RAG (ClinicalBERT)  → qa_claude_rag_cbert.json
   4.2 (--phases 42)  Claude QA  + No RAG               → qa_claude_norag.json
   4.3 (--phases 43)  Claude KW  + RAG (ClinicalBERT)  → kw_claude_rag_clinicalbert.json
   4.4 (--phases 44)  Claude KW  + No RAG               → kw_claude_norag.json
   All sub-phases require --api-key / ANTHROPIC_API_KEY.  (TODO #1)
5  Re-run split-partial verifier on 8B+RAG+ClinBERT config (TODO #6)

Usage
-----
# Full run (all phases, including all 4.x sub-phases):
python rerun_all.py --api-key sk-ant-...

# Local-only run (skips sub-phases 41–44 — no API key needed):
python rerun_all.py --local-only

# Run a single Phase 4 sub-phase in isolation:
python rerun_all.py --phases 41 --api-key sk-ant-...   # QA + RAG only
python rerun_all.py --phases 42 --api-key sk-ant-...   # QA No-RAG only
python rerun_all.py --phases 43 --api-key sk-ant-...   # KW + RAG only
python rerun_all.py --phases 44 --api-key sk-ant-...   # KW No-RAG only

# Select specific phases:
python rerun_all.py --phases 1 2 3 --api-key sk-ant-...

# Force re-run even if output files already exist:
python rerun_all.py --force --api-key sk-ant-...

# Dry run — print commands without executing:
python rerun_all.py --dry-run --api-key sk-ant-...
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("rerun_all")

# ---------------------------------------------------------------------------
# Run descriptor
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One invocation of main.py or compute_href.py."""
    phase: int
    label: str             # human-readable name used in summary
    output: str            # expected output path (used for skip-if-exists check)
    cmd: List[str]         # full argv passed to subprocess


# ---------------------------------------------------------------------------
# Run catalogue
# ---------------------------------------------------------------------------

def build_runs(
    queries_dir: str,
    results_dir: str,
    href_dir: str,
    cache_dir: str,
    ollama_host: str,
    api_key: Optional[str],
    claude_model: str,
) -> List[Run]:
    """Return the ordered list of all runs across all phases."""
    runs: List[Run] = []
    q = queries_dir
    r = results_dir
    h = href_dir
    c = cache_dir

    # ── PHASE 1: Re-run all 6 local QA configs ──────────────────────────────
    qa_local = [
        ("8B Q4 + RAG (ClinicalBERT)",   "llama3.1", "clinicalbert", "on",  "off", "standard", f"{r}/qa_8b_rag_cbert.json"),
        ("8B Q4 + RAG (MedCPT)",          "llama3.1", "medcpt",       "on",  "off", "standard", f"{r}/qa_8b_rag_medcpt.json"),
        ("8B Q4 + RAG (MiniLM)",          "llama3.1", "minilm",       "on",  "off", "standard", f"{r}/qa_8b_rag_minilm.json"),
        ("1B Q4 + RAG (ClinicalBERT)",    "llama3.2", "clinicalbert", "on",  "off", "standard", f"{r}/qa_1b_rag_cbert.json"),
        ("8B Q4 No RAG",                  "llama3.1", "clinicalbert", "off", "off", "standard", f"{r}/qa_8b_norag.json"),
        ("8B Q4 + RAG + NLI Check (std)", "llama3.1", "clinicalbert", "on",  "on",  "standard", f"{r}/qa_8b_rag_cbert_ver.json"),
    ]
    for label, model, retriever, rag, verifier, vmode, output in qa_local:
        runs.append(Run(
            phase=1, label=label, output=output,
            cmd=[
                sys.executable, "main.py",
                "--task", "qa",
                "--model", model,
                "--retriever", retriever,
                "--rag", rag,
                "--verifier", verifier,
                "--verifier-mode", vmode,
                "--queries", f"{q}/qa.jsonl",
                "--output", output,
                "--cache-dir", c,
                "--ollama-host", ollama_host,
            ],
        ))

    # ── PHASE 2: Keyword eval — 5 local configs ─────────────────────────────
    kw_local = [
        ("KW 8B RAG ClinicalBERT", "llama3.1", "clinicalbert", "on",  f"{r}/kw_8b_rag_clinicalbert.json"),
        ("KW 8B RAG MedCPT",       "llama3.1", "medcpt",       "on",  f"{r}/kw_8b_rag_medcpt.json"),
        ("KW 8B RAG MiniLM",       "llama3.1", "minilm",       "on",  f"{r}/kw_8b_rag_minilm.json"),
        ("KW 1B RAG ClinicalBERT", "llama3.2", "clinicalbert", "on",  f"{r}/kw_1b_rag_clinicalbert.json"),
        ("KW 8B No RAG",           "llama3.1", "clinicalbert", "off", f"{r}/kw_8b_norag.json"),
    ]
    for label, model, retriever, rag, output in kw_local:
        runs.append(Run(
            phase=2, label=label, output=output,
            cmd=[
                sys.executable, "main.py",
                "--task", "keywords",
                "--model", model,
                "--retriever", retriever,
                "--rag", rag,
                "--verifier", "off",
                "--queries", f"{q}/keywords.jsonl",
                "--output", output,
                "--cache-dir", c,
                "--ollama-host", ollama_host,
                "--temperature", "0.0",
                "--max-tokens", "512",
            ],
        ))

    # ── PHASE 3: h_ref scoring ───────────────────────────────────────────────
    qa_inputs = [
        f"{r}/qa_8b_rag_cbert.json",
        f"{r}/qa_8b_rag_medcpt.json",
        f"{r}/qa_8b_rag_minilm.json",
        f"{r}/qa_1b_rag_cbert.json",
        f"{r}/qa_8b_norag.json",
        f"{r}/qa_8b_rag_cbert_ver.json",
    ]
    # Sentinel output: presence of the RAG href file indicates phase 3 is done.
    href_sentinel = f"{h}/qa_8b_rag_cbert_href.json"
    runs.append(Run(
        phase=3,
        label="h_ref scoring + Table 2 paired stats",
        output=href_sentinel,
        cmd=[
            sys.executable, "compute_href.py",
            "--input", *qa_inputs,
            "--output", h,
            "--model", "llama3.1",
            "--ollama-host", ollama_host,
            "--paired-stats",
            f"{h}/qa_8b_rag_cbert_href.json",
            f"{h}/qa_8b_norag_href.json",
        ],
    ))

    # ── PHASE 4: Claude API baselines (split into sub-phases) ────────────────
    # 4.1 — Claude QA + RAG (ClinicalBERT)  → qa_claude_rag_cbert.json
    # 4.2 — Claude QA + No RAG              → qa_claude_norag.json
    # 4.3 — Claude Keywords + RAG           → kw_claude_rag_clinicalbert.json
    # 4.4 — Claude Keywords + No RAG        → kw_claude_norag.json
    if api_key:
        runs.append(Run(
            phase=41,
            label="[4.1] Claude QA + RAG (ClinicalBERT)",
            output=f"{r}/qa_claude_rag_cbert.json",
            cmd=[
                sys.executable, "main.py",
                "--task", "qa",
                "--model", claude_model,
                "--backend", "anthropic",
                "--api-key", api_key,
                "--retriever", "clinicalbert",
                "--rag", "on",
                "--verifier", "off",       # no local verifier on cloud model
                "--ensemble-hallucination", "on",
                "--atomic-facts", "on",
                "--queries", f"{q}/qa.jsonl",
                "--output", f"{r}/qa_claude_rag_cbert.json",
                "--cache-dir", c,
            ],
        ))

        runs.append(Run(
            phase=42,
            label="[4.2] Claude QA + No RAG",
            output=f"{r}/qa_claude_norag.json",
            cmd=[
                sys.executable, "main.py",
                "--task", "qa",
                "--model", claude_model,
                "--backend", "anthropic",
                "--api-key", api_key,
                "--retriever", "clinicalbert",
                "--rag", "off",
                "--verifier", "off",
                "--ensemble-hallucination", "on",
                "--atomic-facts", "on",
                "--queries", f"{q}/qa.jsonl",
                "--output", f"{r}/qa_claude_norag.json",
                "--cache-dir", c,
            ],
        ))

        runs.append(Run(
            phase=43,
            label="[4.3] Claude Keywords + RAG (ClinicalBERT)",
            output=f"{r}/kw_claude_rag_clinicalbert.json",
            cmd=[
                sys.executable, "main.py",
                "--task", "keywords",
                "--model", claude_model,
                "--backend", "anthropic",
                "--api-key", api_key,
                "--retriever", "clinicalbert",
                "--rag", "on",
                "--verifier", "off",
                "--queries", f"{q}/keywords.jsonl",
                "--output", f"{r}/kw_claude_rag_clinicalbert.json",
                "--cache-dir", c,
                "--temperature", "0.0",
                "--max-tokens", "512",
            ],
        ))

        runs.append(Run(
            phase=44,
            label="[4.4] Claude Keywords + No RAG",
            output=f"{r}/kw_claude_norag.json",
            cmd=[
                sys.executable, "main.py",
                "--task", "keywords",
                "--model", claude_model,
                "--backend", "anthropic",
                "--api-key", api_key,
                "--retriever", "clinicalbert",
                "--rag", "off",
                "--verifier", "off",
                "--queries", f"{q}/keywords.jsonl",
                "--output", f"{r}/kw_claude_norag.json",
                "--cache-dir", c,
                "--temperature", "0.0",
                "--max-tokens", "512",
            ],
        ))

    # ── PHASE 5: Split-partial verifier re-run ───────────────────────────────
    split_output = f"{r}/qa_8b_rag_cbert_ver_split.json"
    runs.append(Run(
        phase=5,
        label="8B Q4 + RAG + NLI Check (split-partial verifier)",
        output=split_output,
        cmd=[
            sys.executable, "main.py",
            "--task", "qa",
            "--model", "llama3.1",
            "--retriever", "clinicalbert",
            "--rag", "on",
            "--verifier", "on",
            "--verifier-mode", "split-partial",
            "--queries", f"{q}/qa.jsonl",
            "--output", split_output,
            "--cache-dir", c,
            "--ollama-host", ollama_host,
        ],
    ))

    return runs


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def run_one(run: Run, force: bool, dry_run: bool) -> bool:
    """Execute a single run. Returns True on success."""
    out_path = Path(run.output)
    if not force and out_path.exists():
        logger.info("[SKIP]  %s  (output exists: %s)", run.label, run.output)
        return True

    cmd_str = " ".join(str(x) for x in run.cmd)
    if dry_run:
        print(f"[DRY]  {cmd_str}")
        return True

    logger.info("[RUN]  Phase %d — %s", run.phase, run.label)
    logger.debug("CMD: %s", cmd_str)
    t0 = time.perf_counter()
    result = subprocess.run(run.cmd, cwd=Path(__file__).parent)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        logger.error(
            "[FAIL] Phase %d — %s  (rc=%d, %.1fs)",
            run.phase, run.label, result.returncode, elapsed,
        )
        return False

    logger.info("[OK]   Phase %d — %s  (%.1fs)", run.phase, run.label, elapsed)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rerun_all",
        description="Master script: run all evaluation configurations for TODOs #1–#6.",
    )
    p.add_argument(
        "--phases", nargs="+", type=int, default=[1, 2, 3, 41, 42, 43, 44, 5],
        metavar="N",
        help=(
            "Which phases to run (default: 1 2 3 41 42 43 44 5). "
            "Phase 4 is split into four independently runnable sub-phases: "
            "41=Claude QA+RAG, 42=Claude QA No-RAG, "
            "43=Claude KW+RAG, 44=Claude KW No-RAG. "
            "All sub-phases require --api-key."
        ),
    )
    p.add_argument(
        "--local-only", action="store_true",
        help="Skip all Phase 4 sub-phases (41-44). Equivalent to --phases 1 2 3 5.",
    )
    p.add_argument(
        "--api-key", default=None,
        help="Anthropic API key for Phase 4. "
             "Alternatively set ANTHROPIC_API_KEY environment variable.",
    )
    p.add_argument(
        "--claude-model", default="claude-sonnet",
        help="Claude model alias for baseline (default: claude-sonnet → claude-sonnet-4-6).",
    )
    p.add_argument(
        "--ollama-host", default="http://localhost:11434",
    )
    p.add_argument(
        "--queries-dir", default="queries",
        help="Directory containing qa.jsonl and keywords.jsonl.",
    )
    p.add_argument(
        "--results-dir", default="results",
        help="Directory to write qa_*.json and kw_*.json result files.",
    )
    p.add_argument(
        "--href-dir", default="results/href",
        help="Directory to write h_ref enriched JSONs.",
    )
    p.add_argument(
        "--cache-dir", default="cache",
        help="Directory for embedding index caches.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run even if the output file already exists.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    phases = set(args.phases)
    if args.local_only:
        phases -= {41, 42, 43, 44}
        logger.info("--local-only: skipping Phase 4 sub-phases (41–44, Claude API baselines)")

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if phases & {41, 42, 43, 44} and not api_key:
        logger.warning(
            "Phase 4 sub-phases selected (%s) but no API key found "
            "(--api-key or ANTHROPIC_API_KEY). Sub-phases 41–44 will be skipped.",
            sorted(phases & {41, 42, 43, 44}),
        )

    # Create output directories.
    for d in (args.results_dir, args.href_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    all_runs = build_runs(
        queries_dir=args.queries_dir,
        results_dir=args.results_dir,
        href_dir=args.href_dir,
        cache_dir=args.cache_dir,
        ollama_host=args.ollama_host,
        api_key=api_key if phases & {41, 42, 43, 44} else None,
        claude_model=args.claude_model,
    )

    selected = [r for r in all_runs if r.phase in phases]
    logger.info(
        "Selected %d run(s) across phase(s) %s",
        len(selected), sorted(phases),
    )

    # ── Execute ──────────────────────────────────────────────────────────────
    results: Dict[str, bool] = {}
    for run in selected:
        ok = run_one(run, force=args.force, dry_run=args.dry_run)
        results[run.label] = ok

    # ── Summary ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nDry run complete — {len(selected)} command(s) printed.")
        return 0

    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    skipped = [r.label for r in selected if not args.force and Path(r.output).exists()
               and r.label not in failed]

    print("\n" + "=" * 60)
    print(f"  SUMMARY  ({len(passed)} ok / {len(failed)} failed / {len(skipped)} skipped)")
    print("=" * 60)
    for label in passed:
        print(f"  ✓  {label}")
    for label in failed:
        print(f"  ✗  {label}")
    print()

    if failed:
        print("Re-run failed jobs with --force to retry.")
        return 1

    print("All selected phases complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())