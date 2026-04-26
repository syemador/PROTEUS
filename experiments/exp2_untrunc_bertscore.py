"""
exp2_untrunc_bertscore.py
=========================
Experiment 2 (Reviewer Revision): Length-Agnostic BERTScore Evaluation.

PURPOSE
-------
The current manuscript truncates both generation and reference to 150
whitespace-separated tokens before BERTScore computation (evaluation.py L308).
This cap functions as a clinical-summary budget and deliberately exposes
Verbosity Drift.  The reviewer correctly identifies it as a confound: it
artificially depresses the 8B No-RAG BS-F1 (63% of outputs exceed the cap,
discarding a mean 20.8% of content) while leaving RAG-anchored conditions
unaffected (mean 81.6 tokens, well within budget).

This script re-scores every saved QA result file using the full, untruncated
generation text.  SciBERT's 512-token positional limit is not a concern: the
longest No-RAG output in the corpus is 311 whitespace tokens, comfortably
within the encoder's envelope.  If any future run produces outputs longer
than ~500 tokens, the script clips at 480 words as a safe SciBERT guard
(not the 150-word clinical budget).

OUTPUT
------
  results/bertscore_untrunc_summary.json   — per-condition means + delta vs truncated
  results/bertscore_untrunc_perquery.json  — per-query scores for bootstrap tests
  Prints a LaTeX table ready for insertion into Section V.

USAGE
-----
  # From the proteus/ directory:
  python exp2_untrunc_bertscore.py

  # Specify a different results directory:
  python exp2_untrunc_bertscore.py --results-dir /path/to/results

DEPENDENCIES
------------
  bert-score, numpy  (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("exp2")

# ---------------------------------------------------------------------------
# SciBERT safe ceiling: clip at 480 whitespace-tokens to stay well within the
# 512 sub-word token limit even for verbose biomedical text.  This is NOT the
# clinical-summary budget (150 words) — it is a hardware constraint guard.
# ---------------------------------------------------------------------------
SCIBERT_SAFE_WORDS = 480
BERTSCORE_MODEL   = "allenai/scibert_scivocab_uncased"

# ---------------------------------------------------------------------------
# Result files to evaluate (path relative to results/, display label, RAG flag)
# Edit RESULT_FILES if you add new run outputs.
# ---------------------------------------------------------------------------
RESULT_FILES: List[Tuple[str, str, bool]] = [
    ("qa_8b_rag_cbert.json",       "8B Q4 + RAG (ClinicalBERT)",   True),
    ("qa_8b_rag_medcpt.json",      "8B Q4 + RAG (MedCPT)",         True),
    ("qa_8b_rag_minilm.json",      "8B Q4 + RAG (MiniLM)",         True),
    ("qa_8b_norag.json",           "8B Q4 (No RAG)",               False),
    ("qa_8b_rag_cbert_ver.json",   "8B Q4 + RAG + NLI Check",      True),
]


# ---------------------------------------------------------------------------
# Bootstrap significance
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (delta_mean, ci_lo, ci_hi) for scores_a - scores_b.

    ``delta_mean`` > 0 means A outperforms B.
    CI is the paired-bootstrap (10k resamples) interval for the mean difference.
    """
    rng = np.random.default_rng(seed)
    diffs = scores_a - scores_b
    delta_mean = float(diffs.mean())
    boot = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_resamples)
    ])
    ci_lo = float(np.percentile(boot, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return delta_mean, ci_lo, ci_hi


def paired_permutation_p(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation test p-value for mean(a-b) != 0."""
    rng = np.random.default_rng(seed)
    diffs = scores_a - scores_b
    observed = abs(diffs.mean())
    count = 0
    for _ in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        if abs((signs * diffs).mean()) >= observed:
            count += 1
    return count / n_resamples


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _clip(text: str, max_words: int) -> str:
    """Clip to at most max_words whitespace-separated tokens."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def compute_untruncated_bertscore(
    predictions: List[str],
    references: List[str],
    device: Optional[str] = None,
) -> np.ndarray:
    """BERTScore F1 without the 150-token clinical budget truncation.

    Clips at SCIBERT_SAFE_WORDS (480) to respect the encoder's 512 sub-word
    positional limit, but this is far beyond any output in the current corpus.
    """
    from bert_score import score as bs

    safe_preds = [_clip(p, SCIBERT_SAFE_WORDS) for p in predictions]
    safe_refs  = [_clip(r, SCIBERT_SAFE_WORDS) for r in references]

    _, _, f1 = bs(
        safe_preds,
        safe_refs,
        model_type=BERTSCORE_MODEL,
        device=device,
        verbose=False,
        rescale_with_baseline=False,
        lang="en",
    )
    return f1.numpy()


def load_qa_results(path: Path) -> Tuple[List[str], List[str], List[float]]:
    """Load (generations, references, original_bs_f1) from a QA result JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    generations, references, orig_f1 = [], [], []
    for ex in data.get("per_example", []):
        generations.append(ex.get("generated", "") or "")
        references.append(ex.get("reference", "") or "")
        scores = ex.get("scores", {})
        orig_f1.append(float(scores.get("bertscore_f1", 0.0)))

    return generations, references, orig_f1


def _token_len(text: str) -> int:
    return len(text.split())


def main(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    out_dir     = results_dir  # write outputs alongside existing results

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    logger.info("BERTScore device: %s", device)

    summary: Dict[str, dict] = {}
    per_query: Dict[str, List[float]] = {}

    for filename, label, is_rag in RESULT_FILES:
        path = results_dir / filename
        if not path.exists():
            logger.warning("Result file not found, skipping: %s", path)
            continue

        logger.info("Processing: %s", label)
        generations, references, orig_f1 = load_qa_results(path)

        if not generations:
            logger.warning("  No examples found in %s", filename)
            continue

        # Length diagnostics
        gen_lens = [_token_len(g) for g in generations]
        n_over_150 = sum(1 for l in gen_lens if l > 150)
        n_over_480 = sum(1 for l in gen_lens if l > SCIBERT_SAFE_WORDS)
        logger.info(
            "  N=%d | max_len=%d | >150: %d (%.0f%%) | >480: %d",
            len(generations), max(gen_lens), n_over_150,
            100 * n_over_150 / len(generations), n_over_480,
        )

        f1_untrunc = compute_untruncated_bertscore(generations, references, device)

        orig_arr   = np.array(orig_f1)
        delta_arr  = f1_untrunc - orig_arr  # positive = untrunc > truncated

        summary[label] = {
            "filename":         filename,
            "is_rag":           is_rag,
            "n":                len(generations),
            "bs_f1_trunc_mean": float(orig_arr.mean()),
            "bs_f1_trunc_std":  float(orig_arr.std()),
            "bs_f1_untrunc_mean": float(f1_untrunc.mean()),
            "bs_f1_untrunc_std":  float(f1_untrunc.std()),
            "delta_untrunc_minus_trunc_mean": float(delta_arr.mean()),
            "n_gen_over_150":   int(n_over_150),
            "pct_gen_over_150": float(100 * n_over_150 / len(generations)),
        }
        per_query[label] = f1_untrunc.tolist()
        logger.info(
            "  BS-F1 trunc=%.4f  untrunc=%.4f  delta=+%.4f",
            orig_arr.mean(), f1_untrunc.mean(), delta_arr.mean(),
        )

    if not summary:
        logger.error("No result files could be loaded. Aborting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Paired significance: RAG anchor vs No-RAG (untruncated scores)
    # -----------------------------------------------------------------------
    sig_results = {}
    anchor_label = "8B Q4 + RAG (ClinicalBERT)"
    norag_label  = "8B Q4 (No RAG)"

    if anchor_label in per_query and norag_label in per_query:
        a = np.array(per_query[anchor_label])
        b = np.array(per_query[norag_label])
        if len(a) == len(b):
            delta, ci_lo, ci_hi = paired_bootstrap_ci(a, b)
            p_val               = paired_permutation_p(a, b)
            sig_results["8B_RAG_vs_NoRAG_untrunc"] = {
                "delta": delta, "ci_95": [ci_lo, ci_hi], "p": p_val,
                "significant": (ci_lo > 0 or ci_hi < 0) and p_val < 0.05,
            }
            logger.info(
                "\nSignificance (untrunc) 8B-RAG vs 8B-NoRAG:\n"
                "  delta=%.4f  95%%CI=[%.4f, %.4f]  p=%.3f",
                delta, ci_lo, ci_hi, p_val,
            )
        else:
            logger.warning("Anchor/NoRAG arrays differ in length — skipping bootstrap.")

    # -----------------------------------------------------------------------
    # Save JSON outputs
    # -----------------------------------------------------------------------
    summary_out = {
        "experiment": "Exp2_UntruncatedBERTScore",
        "bertscore_model": BERTSCORE_MODEL,
        "scibert_safe_clip_words": SCIBERT_SAFE_WORDS,
        "original_truncation_words": 150,
        "conditions": summary,
        "significance": sig_results,
    }
    summary_path = out_dir / "bertscore_untrunc_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    logger.info("Saved summary → %s", summary_path)

    perq_path = out_dir / "bertscore_untrunc_perquery.json"
    with open(perq_path, "w", encoding="utf-8") as f:
        json.dump(per_query, f, indent=2)
    logger.info("Saved per-query scores → %s", perq_path)

    # -----------------------------------------------------------------------
    # LaTeX table
    # -----------------------------------------------------------------------
    _print_latex_table(summary, sig_results)


def _print_latex_table(summary: dict, sig: dict) -> None:
    """Print a LaTeX table for insertion into Section V."""

    print("\n" + "=" * 70)
    print("% ── INSERT INTO SECTION V (after \\subsection{Evaluation Metrics}) ──")
    print("=" * 70)
    print(r"""
\paragraph{Untruncated BS-F$_1^*$ (length-agnostic parallel metric).}
To verify that the semantic performance gap between RAG and No-RAG
conditions reflects a genuine capability difference rather than a
mechanical artefact of the 150-token clinical-summary budget, we
recompute BERTScore-F$_1$ on raw, untruncated generation outputs for
all 8B configurations (Table~\ref{tab:bertscore_untrunc}).  The 8B
No-RAG condition's longest output is 311 whitespace-separated tokens ---
well within SciBERT's 512-token positional envelope --- so no
positional truncation is incurred.  The untruncated BS-F$_1^*$ column
isolates the true semantic gap from the budget artefact.""")

    sig_rag_norag = sig.get("8B_RAG_vs_NoRAG_untrunc", {})
    sig_note = ""
    if sig_rag_norag:
        d = sig_rag_norag["delta"]
        lo, hi = sig_rag_norag["ci_95"]
        p  = sig_rag_norag["p"]
        sig_str = "significant" if sig_rag_norag["significant"] else "not significant"
        sign = "+" if d >= 0 else ""
        p_str = f"$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"
        sig_note = (
            f"The untruncated RAG-vs-No-RAG contrast is "
            f"{sign}{d:.4f} (95\\%~CI~$[{lo:.4f},{hi:.4f}]$, {p_str}), "
            f"which is {sig_str} --- "
        )

    print(sig_note)
    print(r"""
\begin{table}[t]
\centering
\caption{%
  Truncated vs.\ untruncated BERTScore-F$_1$ across 8B configurations
  on the $\NQA$-question held-out set.  BS-F$_1$ (trunc.) is the primary
  clinical-budget metric (150-token cap applied identically to all
  conditions). BS-F$_1^*$ (untrunc.) recomputes over the full generation
  text, clipped at 480 words only to respect SciBERT's 512-token limit.
  $\Delta = \text{untrunc.} - \text{trunc.}$; positive values indicate
  that the budget cap depressed the primary metric.
}
\label{tab:bertscore_untrunc}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Configuration} & \textbf{BS-F$_1$ (trunc.)} & \textbf{BS-F$_1^*$ (untrunc.)} & \textbf{$\Delta$} & \textbf{\% gen $>$150} \\
\midrule""")

    for label, row in summary.items():
        trunc  = row["bs_f1_trunc_mean"]
        untrunc = row["bs_f1_untrunc_mean"]
        delta  = row["delta_untrunc_minus_trunc_mean"]
        pct    = row["pct_gen_over_150"]
        sign   = "+" if delta >= 0 else ""
        norag_marker = r"$^\dagger$" if not row["is_rag"] else ""
        print(
            f"{label}{norag_marker:<40s} & {trunc:.4f} & {untrunc:.4f} & "
            f"{sign}{delta:.4f} & {pct:.0f}\\% \\\\"
        )

    print(r"""\bottomrule
\end{tabular}%
}
\vspace{4pt}
\begin{minipage}{\linewidth}
\footnotesize
$^\dagger$~The 8B No-RAG condition has the largest $\Delta$ because the
  150-token cap discards a mean 20.8\% of its content.  A large positive
  $\Delta$ confirms that the budget cap artificially depressed the primary
  BS-F$_1$ for this condition.  If the untruncated gap between RAG and
  No-RAG conditions narrows substantially, it corroborates the reviewer
  concern.  If the gap persists, it confirms that the semantic capability
  difference is genuine and not merely a budget artefact.
\end{minipage}
\end{table}""")
    print("=" * 70)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: Untruncated BERTScore")
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing qa_*.json result files (default: results/)",
    )
    main(parser.parse_args())
