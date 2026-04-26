"""
build_tables.py
===============
Reads result JSONs from results/ and results/href/ and generates the
complete set of LaTeX table sources needed for the paper.

Run after completing rerun_all.py phases 1–5 and the three reviewer
revision experiments (exp1/exp2/exp3).

Tables
------
  --table qa          Table 1 : QA performance (all local + Claude Sonnet rows)
  --table pairs       Table 2 : Pairwise significance (proper paired bootstrap)
  --table latency     Table 3 : Inference throughput
  --table genlength   Table 4 : Generation lengths + truncation rates
  --table keywords    Table 5 : Keyword extraction P / R / F1
  --table parse       Supp.   : Parse-failure rate per configuration
  --table split       Supp.   : Split-partial verifier label breakdown (RQ4)
  --table untrunc     Rev.R1  : Untruncated BERTScore (Exp 2)
  --table constrained Rev.R2  : Constrained structured decoding dual evaluation (Exp 1)
  --table adversarial Rev.R3  : Adversarial fail-safe trigger/recovery (Exp 3)
  --all               Print all tables in order

Usage
-----
python build_tables.py --table qa
python build_tables.py --all --results-dir results --href-dir results/href
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLACEHOLDER = r"\textcolor{red}{\textbf{[PENDING]}}"


def load(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def per_ex_scores(data: Dict[str, Any], field: str) -> List[float]:
    """Extract a per-example score field, excluding None and NaN."""
    out: List[float] = []
    for ex in data.get("per_example", []):
        v = ex.get("scores", {}).get(field)
        if v is not None and v == v:   # NaN check
            out.append(float(v))
    return out


def fmt_mean_ci(mean: Any, lo: Any, hi: Any,
                pct: bool = False, places: int = 4) -> str:
    if mean is None:
        return PLACEHOLDER
    s = 100.0 if pct else 1.0
    fmt = "{:.2f}" if pct else f"{{:.{places}f}}"
    return rf"{fmt.format(float(mean)*s)} [{fmt.format(float(lo)*s)}, {fmt.format(float(hi)*s)}]"


def fmt_pm(mean: float, std: float, places: int = 2) -> str:
    return rf"${mean:.{places}f} \pm {std:.{places}f}$"


def _f(val: Any, places: int = 4) -> str:
    """Format a float or return PLACEHOLDER."""
    if val is None:
        return PLACEHOLDER
    return f"{float(val):.{places}f}"


def _sign(val: Any) -> str:
    if val is None:
        return ""
    return "+" if float(val) >= 0 else ""


def _paired_stats(a: List[float], b: List[float],
                  n_resamples: int = 10_000,
                  seed: int = 42) -> Tuple[float, float, float, float]:
    """(delta, ci_lo, ci_hi, p_value) — pure numpy, no torch dependency.

    Paired bootstrap CI (10k resamples) and two-sided paired permutation test
    via random sign flips.  Matches evaluation.py implementations exactly.
    """
    if not a or not b or len(a) != len(b):
        nan = float("nan")
        return nan, nan, nan, nan
    import numpy as np
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    rng   = np.random.default_rng(seed)
    n     = len(a_arr)

    # --- bootstrap CI ---
    idx   = rng.integers(0, n, size=(n_resamples, n))
    boots = a_arr[idx].mean(axis=1) - b_arr[idx].mean(axis=1)
    ci_lo = float(np.quantile(boots, 0.025))
    ci_hi = float(np.quantile(boots, 0.975))

    # --- permutation p-value ---
    diffs    = a_arr - b_arr
    observed = abs(float(diffs.mean()))
    signs    = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
    extreme  = int(np.sum(np.abs((signs * diffs).mean(axis=1)) >= observed))
    p_val    = (extreme + 1) / (n_resamples + 1)

    return float(a_arr.mean() - b_arr.mean()), ci_lo, ci_hi, float(p_val)


def _fmt_stat(d: float, lo: float, hi: float, p: float,
              scale: float = 1.0) -> Tuple[str, str, str]:
    if any(v != v for v in (d, lo, hi, p)):
        return PLACEHOLDER, PLACEHOLDER, PLACEHOLDER
    return (
        rf"{d*scale:+.4f}",
        rf"[{lo*scale:+.4f}, {hi*scale:+.4f}]",
        rf"{p:.3f}",
    )


# ---------------------------------------------------------------------------
# Configuration registries
# ---------------------------------------------------------------------------

def _qa_configs(r: str) -> List[Dict[str, Any]]:
    return [
        {"label": "8B Q4 + RAG (ClinicalBERT)",              "path": f"{r}/qa_8b_rag_cbert.json",          "scale": "8B (Q4)"},
        {"label": "8B Q4 + RAG (MedCPT)",                     "path": f"{r}/qa_8b_rag_medcpt.json",         "scale": "8B (Q4)"},
        {"label": "8B Q4 + RAG (MiniLM)",                     "path": f"{r}/qa_8b_rag_minilm.json",         "scale": "8B (Q4)"},
        {"label": "1B Q4 + RAG (ClinicalBERT)",               "path": f"{r}/qa_1b_rag_cbert.json",          "scale": "1B (Q4)"},
        {"label": "8B Q4 (No RAG)",                           "path": f"{r}/qa_8b_norag.json",              "scale": "8B (Q4)"},
        {"label": "8B Q4 + RAG + NLI Check",                  "path": f"{r}/qa_8b_rag_cbert_ver.json",      "scale": "8B (Q4)"},
        {"label": r"Claude Sonnet + RAG (ClinicalBERT)",      "path": f"{r}/qa_claude_rag_cbert.json",      "scale": "API"},
        {"label": r"Claude Sonnet (No RAG)",                  "path": f"{r}/qa_claude_norag.json",          "scale": "API"},
    ]


def _kw_configs(r: str) -> List[Dict[str, Any]]:
    return [
        {"label": "8B Q4 + RAG (ClinicalBERT)",              "path": f"{r}/kw_8b_rag_clinicalbert.json"},
        {"label": "8B Q4 + RAG (MedCPT)",                    "path": f"{r}/kw_8b_rag_medcpt.json"},
        {"label": "8B Q4 + RAG (MiniLM)",                    "path": f"{r}/kw_8b_rag_minilm.json"},
        {"label": "1B Q4 + RAG (ClinicalBERT)",              "path": f"{r}/kw_1b_rag_clinicalbert.json"},
        {"label": "8B Q4 (No RAG)",                          "path": f"{r}/kw_8b_norag.json"},
        {"label": r"Claude Sonnet + RAG (ClinicalBERT)",     "path": f"{r}/kw_claude_rag_clinicalbert.json"},
        {"label": r"Claude Sonnet (No RAG)",                 "path": f"{r}/kw_claude_norag.json"},
    ]


# ---------------------------------------------------------------------------
# Table 1 — QA performance
# ---------------------------------------------------------------------------

def table_qa(results_dir: str, **kw: Any) -> str:
    rows: List[str] = []
    for c in _qa_configs(results_dir):
        d = load(c["path"])
        agg = d["aggregate"] if d else {}

        r1 = fmt_mean_ci(agg.get("rouge1_f_mean"),
                         agg.get("rouge1_f_ci_low"), agg.get("rouge1_f_ci_high")) if d else PLACEHOLDER
        rL = fmt_mean_ci(agg.get("rougeL_f_mean"),
                         agg.get("rougeL_f_ci_low"), agg.get("rougeL_f_ci_high")) if d else PLACEHOLDER
        bs = fmt_mean_ci(agg.get("bertscore_f1_mean"),
                         agg.get("bertscore_f1_ci_low"), agg.get("bertscore_f1_ci_high")) if d else PLACEHOLDER

        no_ctx = d and not d.get("rag_enabled", True)
        if no_ctx:
            hctx = "---"
        elif d and agg.get("hallucination_rate_mean") is not None:
            hctx = fmt_mean_ci(agg["hallucination_rate_mean"],
                               agg["hallucination_rate_ci_low"],
                               agg["hallucination_rate_ci_high"], pct=True)
        else:
            hctx = PLACEHOLDER

        rows.append(rf"{c['label']} & {r1} & {rL} & {bs} & {hctx} \\")

    return "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Clinical QA performance across configurations on the",
        r"$\NQA$-question high-precision held-out set. R-1 = ROUGE-1,",
        r"R-L = ROUGE-L, BS-F1 = BERTScore F$_1$, $h_{\mathrm{ctx}}$ (\%) =",
        r"context-grounded hallucination rate; all with $95\%$ bootstrap CIs.",
        r"$h_{\mathrm{ctx}}$ is undefined for No-RAG conditions (---).}",
        r"\label{tab:qa_final}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"\textbf{Configuration} & \textbf{R-1} & \textbf{R-L} "
        r"& \textbf{BS-F1} & \textbf{$h_{\mathrm{ctx}}$ (\%)} \\",
        r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table*}",
    ])


# ---------------------------------------------------------------------------
# Table 2 — Pairwise significance
# ---------------------------------------------------------------------------

def table_pairs(results_dir: str, href_dir: str,
                n_resamples: int = 10_000, **kw: Any) -> str:
    anchor  = load(f"{results_dir}/qa_8b_rag_cbert.json")
    lb_d    = load(f"{results_dir}/qa_1b_rag_cbert.json")
    norag_d = load(f"{results_dir}/qa_8b_norag.json")
    href_a  = load(f"{href_dir}/qa_8b_rag_cbert_href.json")
    href_n  = load(f"{href_dir}/qa_8b_norag_href.json")

    # Row 1: 8B-RAG vs 1B-RAG — BS-F1
    d1, l1, h1, p1 = _paired_stats(
        per_ex_scores(anchor, "bertscore_f1") if anchor else [],
        per_ex_scores(lb_d,   "bertscore_f1") if lb_d   else [],
        n_resamples)
    # Row 2: 8B-RAG vs 1B-RAG — h_ctx (percentage-point scale)
    d2, l2, h2, p2 = _paired_stats(
        per_ex_scores(anchor, "hallucination_rate") if anchor else [],
        per_ex_scores(lb_d,   "hallucination_rate") if lb_d   else [],
        n_resamples)
    # Row 3: 8B-RAG vs 8B-NoRAG — BS-F1
    d3, l3, h3, p3 = _paired_stats(
        per_ex_scores(anchor,  "bertscore_f1") if anchor  else [],
        per_ex_scores(norag_d, "bertscore_f1") if norag_d else [],
        n_resamples)
    # Row 4: 8B-RAG vs 8B-NoRAG — h_ref (percentage-point scale)
    d4, l4, h4, p4 = _paired_stats(
        per_ex_scores(href_a, "h_ref") if href_a else [],
        per_ex_scores(href_n, "h_ref") if href_n else [],
        n_resamples)

    r1d, r1ci, r1p = _fmt_stat(d1, l1, h1, p1)
    r2d, r2ci, r2p = _fmt_stat(d2, l2, h2, p2, scale=100.0)
    r3d, r3ci, r3p = _fmt_stat(d3, l3, h3, p3)
    r4d, r4ci, r4p = _fmt_stat(d4, l4, h4, p4, scale=100.0)

    href_note = (
        r"Computed from \texttt{results/href/} files."
        if href_a and href_n else
        r"\textcolor{red}{\textbf{[PENDING: run Phase~3 of rerun\_all.py]}}"
    )

    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Pairwise significance contrasts against the 8B Q4 + RAG",
        r"(ClinicalBERT) anchor. $\Delta = \text{anchor} - \text{comparison}$;",
        r"CIs are paired bootstrap ($\NBOOT$ resamples); $p$-values are",
        r"two-sided paired permutation tests. Significant iff CI excludes",
        r"zero \emph{and} $p<0.05$.}",
        r"\label{tab:pairwise_significance}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"\textbf{Contrast} & \textbf{Metric} & \textbf{$\Delta$} "
        r"& \textbf{95\% CI} & \textbf{$p$} \\",
        r"\midrule",
        rf"8B-RAG vs.\ 1B-RAG   & BS-F1 & {r1d} & {r1ci} & {r1p} \\",
        rf"8B-RAG vs.\ 1B-RAG   & $h_{{\mathrm{{ctx}}}}$ & {r2d} & {r2ci} & {r2p} \\",
        rf"8B-RAG vs.\ 8B-NoRAG & BS-F1 & {r3d} & {r3ci} & {r3p} \\",
        rf"8B-RAG vs.\ 8B-NoRAG & $h_{{\mathrm{{ref}}}}$ & {r4d} & {r4ci} & {r4p} \\",
        r"\bottomrule", r"\end{tabular}",
        r"\vspace{2pt}", r"\begin{flushleft}", r"\footnotesize{%",
        r"The $h_{\mathrm{ref}}$ row is the primary RAG-vs-parametric",
        r"comparison (Eq.~(\ref{eq:ref-ent})). " + href_note,
        r"}", r"\end{flushleft}", r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Table 3 — Latency
# ---------------------------------------------------------------------------

def table_latency(results_dir: str, **kw: Any) -> str:
    local_cfgs = [
        ("8B Q4 + RAG (ClinicalBERT)", f"{results_dir}/qa_8b_rag_cbert.json",     "8B (Q4)"),
        ("8B Q4 + RAG (MedCPT)",       f"{results_dir}/qa_8b_rag_medcpt.json",    "8B (Q4)"),
        ("8B Q4 + RAG (MiniLM)",       f"{results_dir}/qa_8b_rag_minilm.json",    "8B (Q4)"),
        ("1B Q4 + RAG (ClinicalBERT)", f"{results_dir}/qa_1b_rag_cbert.json",     "1B (Q4)"),
        ("8B Q4 (No RAG)",             f"{results_dir}/qa_8b_norag.json",          "8B (Q4)"),
        ("8B Q4 + RAG + NLI Check",    f"{results_dir}/qa_8b_rag_cbert_ver.json", "8B (Q4)"),
    ]
    rows: List[str] = []
    for label, path, scale in local_cfgs:
        d = load(path)
        if d:
            lat = d["latency"]
            tps = fmt_pm(lat["tokens_per_second_mean"],
                         lat.get("tokens_per_second_std", 0.0))
        else:
            tps = PLACEHOLDER
        rows.append(rf"{label} & {scale} & {tps} \\")
    rows.append(
        r"Claude Sonnet (API) & API & \textit{n/a (network-dominated)} \\"
    )

    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Inference throughput on the H.E.R.A.\ testbed",
        r"(RTX~3060, 12~GB). Mean $\pm$ std over $\NRUNS$ runs.",
        r"Cloud API throughput is network-latency dominated and not",
        r"reported as a hardware metric.}",
        r"\label{tab:latency}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
        r"\textbf{Configuration} & \textbf{Scale} & \textbf{Throughput (tok/s)} \\",
        r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Table 4 — Generation lengths and truncation rates
# ---------------------------------------------------------------------------

def table_genlength(results_dir: str, **kw: Any) -> str:
    import numpy as np
    cfgs = [
        ("8B Q4 + RAG (ClinicalBERT)", f"{results_dir}/qa_8b_rag_cbert.json"),
        ("8B Q4 + RAG (MedCPT)",       f"{results_dir}/qa_8b_rag_medcpt.json"),
        ("8B Q4 + RAG (MiniLM)",       f"{results_dir}/qa_8b_rag_minilm.json"),
        ("1B Q4 + RAG (ClinicalBERT)", f"{results_dir}/qa_1b_rag_cbert.json"),
        ("8B Q4 (No RAG)",             f"{results_dir}/qa_8b_norag.json"),
        ("8B Q4 + RAG + NLI Check",    f"{results_dir}/qa_8b_rag_cbert_ver.json"),
    ]
    rows: List[str] = []
    for label, path in cfgs:
        d = load(path)
        if not d:
            rows.append(rf"{label} & {PLACEHOLDER} & {PLACEHOLDER} & {PLACEHOLDER} & {PLACEHOLDER} \\")
            continue
        lens = np.array([len(ex["generated"].split()) for ex in d["per_example"]])
        pct_over = 100.0 * (lens > 150).mean()
        mean_disc = float(np.mean(
            [(l - 150) / l * 100 if l > 150 else 0.0 for l in lens]
        ))
        rows.append(
            rf"{label} & {lens.mean():.1f} & {int(lens.max())} & "
            rf"{pct_over:.0f}\% & {mean_disc:.1f}\% \\"
        )

    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Generation-length profile on the $\NQA$-question held-out",
        r"set. ``Mean len'' and ``Max'' are whitespace-separated tokens before",
        r"the $150$-token BERTScore truncation cap.}",
        r"\label{tab:gen_lengths}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"\textbf{Configuration} & \textbf{Mean len} & \textbf{Max} "
        r"& \textbf{\% $>$150} & \textbf{Mean \% disc.} \\",
        r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Table 5 — Keyword extraction
# ---------------------------------------------------------------------------

def table_keywords(results_dir: str, **kw: Any) -> str:
    CLASSES = ["Symptoms", "Diagnostics", "Pathogens"]
    rows: List[str] = []
    for c in _kw_configs(results_dir):
        d = load(c["path"])
        if not d:
            rows.append(rf"{c['label']} & " + " & ".join([PLACEHOLDER]*10) + r" \\")
            continue
        agg = d["aggregate"]
        parts: List[str] = []
        for cls in CLASSES:
            k = cls.lower()
            for metric in ("precision", "recall", "f1"):
                v = agg.get(f"{k}_{metric}_mean")
                parts.append(f"{v:.3f}" if v is not None else PLACEHOLDER)
        micro = agg.get("micro_f1_mean")
        parts.append(f"{micro:.3f}" if micro is not None else PLACEHOLDER)
        rows.append(rf"{c['label']} & {' & '.join(parts)} \\")

    hdr = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{\textbf{{{c}}}}}" for c in CLASSES
    )
    sub = " & ".join(["P & R & F1"] * 3) + r" & \textbf{$\mu$-F1}"

    return "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Keyword extraction performance on the $\NKW$-prompt",
        r"held-out set. P = Precision, R = Recall, F1, $\mu$-F1 = micro-averaged",
        r"F$_1$ over the flattened \texttt{\{class::term\}} set.}",
        r"\label{tab:keywords}",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}l{'rrr'*3}r@{{}}}}", r"\toprule",
        rf"\textbf{{Configuration}} & {hdr} & \\",
        rf" & {sub} \\",
        r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table*}",
    ])


# ---------------------------------------------------------------------------
# Supplementary — Parse failure rate
# ---------------------------------------------------------------------------

def table_parse(results_dir: str, **kw: Any) -> str:
    cfgs = [
        ("8B Q4 + RAG (ClinicalBERT)", f"{results_dir}/qa_8b_rag_cbert.json"),
        ("8B Q4 + RAG (MedCPT)",       f"{results_dir}/qa_8b_rag_medcpt.json"),
        ("8B Q4 + RAG (MiniLM)",       f"{results_dir}/qa_8b_rag_minilm.json"),
        ("1B Q4 + RAG (ClinicalBERT)", f"{results_dir}/qa_1b_rag_cbert.json"),
        ("8B Q4 (No RAG)",             f"{results_dir}/qa_8b_norag.json"),
        ("8B Q4 + RAG + NLI Check",    f"{results_dir}/qa_8b_rag_cbert_ver.json"),
    ]
    rows: List[str] = []
    for label, path in cfgs:
        d = load(path)
        if not d:
            rows.append(rf"{label} & --- & {PLACEHOLDER} & {PLACEHOLDER} \\")
            continue
        per = d["per_example"]
        n = len(per)
        has_flag = any("parse_failed" in ex.get("scores", {}) for ex in per)
        if not has_flag:
            rows.append(
                rf"{label} & {n} & "
                r"\textit{n/a (pre-instrumentation)} & \textit{n/a} \\"
            )
            continue
        n_fail = sum(1 for ex in per if ex["scores"].get("parse_failed", False))
        pct = 100.0 * n_fail / n
        rows.append(rf"{label} & {n} & {n_fail} & {pct:.1f}\% \\")

    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Atomic-fact decomposition parse-failure rate per",
        r"configuration. ``Failed'' = outputs where the fact-extractor LLM",
        r"response was not a valid JSON array and the regex sentence-splitter",
        r"fallback was invoked. Rows marked \textit{n/a} were produced before",
        r"the \texttt{parse\_failed} field was added; re-run Phase~1 of",
        r"\texttt{rerun\_all.py} to populate.}",
        r"\label{tab:parse_fail}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"\textbf{Configuration} & \textbf{N} & \textbf{Failed} & \textbf{\%} \\",
        r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Supplementary — Split-partial verifier breakdown
# ---------------------------------------------------------------------------

def table_split(results_dir: str, **kw: Any) -> str:
    std_d   = load(f"{results_dir}/qa_8b_rag_cbert_ver.json")
    split_d = load(f"{results_dir}/qa_8b_rag_cbert_ver_split.json")

    def _row(label: str, d: Optional[Dict[str, Any]],
             is_split: bool) -> str:
        if not d:
            return (
                rf"{label} & {PLACEHOLDER} & {PLACEHOLDER} & "
                rf"{PLACEHOLDER} & {PLACEHOLDER} & {PLACEHOLDER} & {PLACEHOLDER} \\"
            )
        per = [ex for ex in d.get("per_example", []) if ex.get("verifier")]
        total = sum(
            ex["verifier"]["n_supported"]
            + ex["verifier"].get("n_partial", 0)
            + ex["verifier"].get("n_weak_entailment", 0)
            + ex["verifier"].get("n_citation_noncompliance", 0)
            + ex["verifier"]["n_unsupported"]
            for ex in per
        )
        if total == 0:
            return rf"{label} & 0 & --- & --- & --- & --- & --- \\"
        n_sup = sum(ex["verifier"]["n_supported"]   for ex in per)
        n_uns = sum(ex["verifier"]["n_unsupported"] for ex in per)
        pct = lambda n: rf"{n} ({100*n/total:.1f}\%)"
        if is_split:
            n_we = sum(ex["verifier"].get("n_weak_entailment", 0)       for ex in per)
            n_cn = sum(ex["verifier"].get("n_citation_noncompliance", 0) for ex in per)
            return (
                rf"{label} & {total} & {pct(n_sup)} & --- "
                rf"& {pct(n_we)} & {pct(n_cn)} & {pct(n_uns)} \\"
            )
        else:
            n_par = sum(ex["verifier"].get("n_partial", 0) for ex in per)
            return (
                rf"{label} & {total} & {pct(n_sup)} & {pct(n_par)} "
                rf"& --- & --- & {pct(n_uns)} \\"
            )

    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{NLI consistency-check label distribution for 8B Q4 + RAG",
        r"(ClinicalBERT) under the standard (3-label) and split-partial",
        r"(4-label) verifier prompts. WE = \texttt{weak\_entailment}",
        r"(grounding-safety); CN = \texttt{citation\_noncompliance}",
        r"(formatting only). ``---'' = label not produced by that prompt mode.}",
        r"\label{tab:split_verifier}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrccccc@{}}", r"\toprule",
        r"\textbf{Mode} & \textbf{Claims} & \textbf{Supported} "
        r"& \textbf{Partial} & \textbf{WE} & \textbf{CN} & \textbf{Unsupported} \\",
        r"\midrule",
        _row("Standard (3-label)",      std_d,   is_split=False),
        _row("Split-partial (4-label)", split_d, is_split=True),
        r"\bottomrule",
        r"\end{tabular}", r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# Rev.R1 — Untruncated BERTScore (Experiment 2)
# ---------------------------------------------------------------------------

def table_untrunc(results_dir: str, **kw: Any) -> str:
    """
    Reads bertscore_untrunc_summary.json (produced by exp2_untrunc_bertscore.py)
    and the original qa_*.json files for the truncated baseline column.
    Falls back to PLACEHOLDER for any missing values.
    """
    summary = load(f"{results_dir}/bertscore_untrunc_summary.json")

    ORDERED = [
        ("8B Q4 + RAG (ClinicalBERT)", f"{results_dir}/qa_8b_rag_cbert.json",    False),
        ("8B Q4 + RAG (MedCPT)",       f"{results_dir}/qa_8b_rag_medcpt.json",   False),
        ("8B Q4 + RAG (MiniLM)",       f"{results_dir}/qa_8b_rag_minilm.json",   False),
        ("8B Q4 (No RAG)",             f"{results_dir}/qa_8b_norag.json",         True),
        ("8B Q4 + RAG + NLI Check",    f"{results_dir}/qa_8b_rag_cbert_ver.json", False),
    ]

    rows: List[str] = []
    for label, qa_path, is_norag in ORDERED:
        qa_d = load(qa_path)
        trunc = PLACEHOLDER
        if qa_d:
            agg = qa_d.get("aggregate", {})
            v = agg.get("bertscore_f1_mean")
            trunc = _f(v, 4) if v is not None else PLACEHOLDER

        unt = delta = pct = PLACEHOLDER
        if summary:
            cond = summary.get("conditions", {}).get(label, {})
            u = cond.get("bs_f1_untrunc_mean")
            t = cond.get("bs_f1_trunc_mean")
            p = cond.get("pct_gen_over_150")
            if u is not None and t is not None:
                d_val = u - t
                sign = _sign(d_val)
                unt   = _f(u, 4)
                delta = f"${sign}{d_val:.4f}$"
                pct   = f"{p:.0f}\\%" if p is not None else PLACEHOLDER

        dagger = r"$^\dagger$" if is_norag else ""
        rows.append(rf"{label}{dagger} & {trunc} & {unt} & {delta} & {pct} \\")

    # Significance block
    sig_line = ""
    if summary:
        sig = summary.get("significance", {}).get("8B_RAG_vs_NoRAG_untrunc", {})
        if sig:
            d_s  = sig.get("delta")
            lo_s = sig.get("ci_95", [None, None])[0]
            hi_s = sig.get("ci_95", [None, None])[1]
            p_s  = sig.get("p")
            if all(v is not None for v in (d_s, lo_s, hi_s, p_s)):
                p_str = r"$p<0.001$" if p_s < 0.001 else f"$p={p_s:.4f}$"
                sig_line = (
                    rf"% Significance (8B-RAG vs 8B-NoRAG, untruncated): "
                    rf"$\Delta={d_s:+.4f}$, "
                    rf"95\%~CI~$[{lo_s:+.4f},{hi_s:+.4f}]$, {p_str}"
                )

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{%",
        r"  Truncated vs.\ untruncated BERTScore-F$_1$ across 8B configurations",
        r"  on the $\NQA$-question held-out set.",
        r"  BS-F$_1$ (trunc.) is the primary clinical-budget metric (150-token cap).",
        r"  BS-F$_1^*$ (untrunc.) recomputes over the full generation text,",
        r"  clipped at 480 words to respect SciBERT's 512-token limit.",
        r"  $\Delta = \text{untrunc.} - \text{trunc.}$;",
        r"  positive values confirm the cap depressed the primary metric.",
        r"  ``\% gen $>$150'' = fraction of outputs affected by the cap.",
        r"}",
        r"\label{tab:bertscore_untrunc}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{BS-F$_1$ (trunc.)} "
        r"& \textbf{BS-F$_1^*$ (untrunc.)} & \textbf{$\Delta$} "
        r"& \textbf{\% gen $>$150} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{4pt}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"$^\dagger$~8B~No-RAG has the largest $|\Delta|$ because the 150-token",
        r"cap discards a mean 20.8\% of its content and the discarded suffix",
        r"diverges from the reference, so the untruncated score \emph{falls}.",
        r"RAG-anchored outputs ($\leq$81.6 tokens mean) are unaffected.",
    ]
    if sig_line:
        lines.append(sig_line)
    lines += [r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rev.R2 — Constrained Decoding Dual Evaluation (Experiment 1)
# ---------------------------------------------------------------------------

def table_constrained(results_dir: str, **kw: Any) -> str:
    """
    Reads kw_constrained_dual_table.json (produced by exp1_constrained_decoding.py)
    for constrained results, and the original kw_*.json files for native baselines.
    Renders both No-RAG and RAG (ClinicalBERT) condition pairs.
    """
    dual = load(f"{results_dir}/kw_constrained_dual_table.json")

    # Native aggregate from existing result files
    def _native_row(path: str, label: str) -> List[str]:
        d = load(path)
        if not d:
            return [rf"\textit{{Native}} ({label}) & " + " & ".join([PLACEHOLDER]*10) + r" \\"]
        agg = d["aggregate"]
        def _v(k: str) -> str:
            v = agg.get(k)
            return f"{v:.3f}" if v is not None else PLACEHOLDER
        return [
            rf"\textit{{Native}} ({label}) & "
            rf"{_v('symptoms_precision_mean')} & {_v('symptoms_recall_mean')} & {_v('symptoms_f1_mean')} & "
            rf"{_v('diagnostics_precision_mean')} & {_v('diagnostics_recall_mean')} & {_v('diagnostics_f1_mean')} & "
            rf"{_v('pathogens_precision_mean')} & {_v('pathogens_recall_mean')} & {_v('pathogens_f1_mean')} & "
            rf"{_v('micro_f1_mean')} \\"
        ]

    # Constrained aggregate from dual table JSON
    def _constrained_row(label: str, tag: str) -> List[str]:
        if not dual:
            return [rf"\textit{{Constrained}}$^\ddagger$ ({label}) & " + " & ".join([PLACEHOLDER]*10) + r" \\"]
        con = dual.get("constrained_aggregate", {})
        pc  = con.get("per_class", {})
        mu  = con.get("micro_f1")

        def _pc(cls: str, m: str) -> str:
            v = pc.get(cls, {}).get(m)
            return f"{v:.3f}" if v is not None else PLACEHOLDER

        return [
            rf"\textit{{Constrained}}$^\ddagger$ ({label}) & "
            rf"{_pc('Symptoms','precision')} & {_pc('Symptoms','recall')} & {_pc('Symptoms','f1')} & "
            rf"{_pc('Diagnostics','precision')} & {_pc('Diagnostics','recall')} & {_pc('Diagnostics','f1')} & "
            rf"{_pc('Pathogens','precision')} & {_pc('Pathogens','recall')} & {_pc('Pathogens','f1')} & "
            rf"{_f(mu,3) if mu is not None else PLACEHOLDER} \\"
        ]

    def _delta_row(nat_path: str, label: str) -> List[str]:
        nat_d = load(nat_path)
        if not nat_d or not dual:
            return [rf"$\Delta$ (constrained $-$ native) & " + " & ".join([PLACEHOLDER]*10) + r" \\"]
        agg = nat_d["aggregate"]
        con = dual.get("constrained_aggregate", {})
        pc  = con.get("per_class", {})
        mu_con = con.get("micro_f1", 0.0)
        mu_nat = agg.get("micro_f1_mean", 0.0)

        METRICS = [
            ("Symptoms",    "precision", "symptoms_precision_mean"),
            ("Symptoms",    "recall",    "symptoms_recall_mean"),
            ("Symptoms",    "f1",        "symptoms_f1_mean"),
            ("Diagnostics", "precision", "diagnostics_precision_mean"),
            ("Diagnostics", "recall",    "diagnostics_recall_mean"),
            ("Diagnostics", "f1",        "diagnostics_f1_mean"),
            ("Pathogens",   "precision", "pathogens_precision_mean"),
            ("Pathogens",   "recall",    "pathogens_recall_mean"),
            ("Pathogens",   "f1",        "pathogens_f1_mean"),
        ]
        deltas = []
        for cls, m, nat_key in METRICS:
            v_con = pc.get(cls, {}).get(m, 0.0)
            v_nat = agg.get(nat_key, 0.0)
            d_v = v_con - v_nat
            deltas.append(f"${_sign(d_v)}{d_v:.3f}$")

        d_mu = mu_con - mu_nat
        deltas.append(f"${_sign(d_mu)}{d_mu:.3f}$")
        return [rf"$\Delta$ (constrained $-$ native) & {' & '.join(deltas)} \\"]

    CLASSES = ["Symptoms", "Diagnostics", "Pathogens"]
    hdr = " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{{c}}}}}" for c in CLASSES)
    sub = " & ".join(["P & R & F1"] * 3) + r" & \textbf{$\mu$-F$_1$}"

    norag_path = f"{results_dir}/kw_8b_norag.json"
    rag_path   = f"{results_dir}/kw_8b_rag_clinicalbert.json"

    rows: List[str] = (
        _native_row(norag_path, "8B Q4, No RAG")
        + _constrained_row("8B Q4, No RAG", "norag")
        + [r"\midrule"]
        + _delta_row(norag_path, "No RAG")
        + [r"\midrule"]
        + _native_row(rag_path,   "8B Q4 + RAG, ClinicalBERT")
        + _constrained_row("8B Q4 + RAG, ClinicalBERT", "cbert")
        + [r"\midrule"]
        + _delta_row(rag_path, "ClinicalBERT")
    )

    return "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{%",
        r"  Dual keyword-extraction evaluation: native generation vs.\ constrained",
        r"  structured decoding (Ollama JSON-Schema grammar enforcement,",
        r"  \texttt{format} parameter, Ollama $\geq$0.1.24).",
        r"  P\,=\,Precision, R\,=\,Recall, $\mu$-F$_1$\,=\,micro-averaged F$_1$.",
        r"  $\Delta$ rows (constrained $-$ native) quantify the Format Collapse",
        r"  penalty per entity class. A large positive $\Delta$ on Symptoms or",
        r"  Pathogens proves entity knowledge was present but syntactically",
        r"  inaccessible under native 4-bit generation",
        r"  (\emph{compliance gap, not a knowledge gap}).",
        r"  Near-zero $\Delta$ on Diagnostics identifies a residual",
        r"  \emph{knowledge gap} that format enforcement cannot resolve.",
        r"}",
        r"\label{tab:keywords_constrained}",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{@{{}}l{'rrr'*3}r@{{}}}}",
        r"\toprule",
        rf"\textbf{{Configuration}} & {hdr} & \\",
        rf" & {sub} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"$^\ddagger$~Constrained decoding via Ollama JSON-Schema format",
        r"enforcement guarantees syntactically valid output by construction.",
        r"Temperature is $T=0.0$ (deterministic) for both native and constrained",
        r"runs, isolating format effects from sampling variance.",
        r"\end{minipage}",
        r"\end{table*}",
    ])


# ---------------------------------------------------------------------------
# Rev.R3 — Adversarial Fail-Safe (Experiment 3)
# ---------------------------------------------------------------------------

def table_adversarial(results_dir: str, **kw: Any) -> str:
    """
    Reads adversarial_failsafe_summary.json (produced by exp3_adversarial_failsafe.py).
    Generates the per-track trigger/recovery table for Section VI RQ4.
    """
    summary = load(f"{results_dir}/adversarial_failsafe_summary.json")
    td = summary.get("summary", {}) if summary else {}

    TRACK_LABELS = {
        "A": r"(A) High-temp ($T=0.8$)",
        "B": r"(B) Poisoned context",
        "C": r"(C) Parametric conflict",
    }

    rows: List[str] = []
    total_n = total_trig = total_recov = 0

    for track in ["A", "B", "C"]:
        label = TRACK_LABELS[track]
        if track not in td:
            rows.append(rf"{label} & 10 & {PLACEHOLDER} & {PLACEHOLDER} & {PLACEHOLDER} \\")
            continue
        v = td[track]
        n      = v.get("n_queries",   0)
        trig   = v.get("n_triggered", 0)
        recov  = v.get("n_recovered", 0)
        tr_rt  = v.get("trigger_rate",  0.0)
        rc_rt  = v.get("recovery_rate")
        pre_u  = v.get("pre_regen_label_totals",  {}).get("unsupported", 0)
        post_u = v.get("post_regen_label_totals", {}).get("unsupported", 0)
        net    = pre_u - post_u

        trig_s  = rf"{trig}/{n} ({tr_rt*100:.0f}\%)"
        recov_s = rf"{recov}/{trig} ({rc_rt*100:.0f}\%)" if (trig > 0 and rc_rt is not None) else "---"
        net_s   = rf"${_sign(net)}{net}$ claims" if trig > 0 else "---"

        rows.append(rf"{label} & {n} & {trig_s} & {recov_s} & {net_s} \\")
        total_n += n; total_trig += trig; total_recov += recov

    if td:
        tot_tr_s  = rf"{total_trig}/{total_n} ({total_trig*100//total_n if total_n else 0}\%)"
        tot_rc_s  = (rf"{total_recov}/{total_trig} ({total_recov*100//total_trig if total_trig else 0}\%)"
                     if total_trig else "---")
    else:
        tot_tr_s = PLACEHOLDER
        tot_rc_s = PLACEHOLDER

    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{%",
        r"  Adversarial Dormant Fail-Safe trigger and recovery rates by sub-track.",
        r"  \textbf{Triggered} = $n_u > 0$ after primary generation.",
        r"  \textbf{Recovered} = $n_u = 0$ after single-pass regeneration",
        r"  (among triggered queries).",
        r"  ``Net $n_u$ reduction'' = total claim-level improvement.",
        r"  (A) High-temperature ($T=0.8$) re-run of standard QA queries.",
        r"  (B) Conflicting-context injection: generator sees poisoned context;",
        r"  verifier sees only real retrieved chunks (split-context design).",
        r"  (C) Parametric-conflict queries targeting thin CORD-19 coverage.",
        r"}",
        r"\label{tab:adversarial_failsafe}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Sub-track} & \textbf{N} & \textbf{Triggered} "
        r"& \textbf{Recovered} & \textbf{Net $n_u$ reduction} \\",
        r"\midrule",
        *rows,
        r"\midrule",
        rf"\textbf{{Total}} & {total_n if td else 30} & {tot_tr_s} & {tot_rc_s} & --- \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{4pt}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"Sub-track~(B) uses a \emph{split context}: the generator sees the",
        r"fabricated sentence framed as \texttt{[Doc~0]}; the verifier receives",
        r"only real retrieved chunks, isolating verifier sensitivity from",
        r"generator compliance. The low recovery rate ($\leq$12.5\% across",
        r"triggered cases) motivates multi-pass or NLI-guided generation",
        r"constraints for adversarial deployment scenarios.",
        r"\end{minipage}",
        r"\end{table}",
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TABLE_MAP: Dict[str, Any] = {
    "qa":          (table_qa,          "Table 1  — QA performance"),
    "pairs":       (table_pairs,       "Table 2  — Pairwise significance"),
    "latency":     (table_latency,     "Table 3  — Inference throughput"),
    "genlength":   (table_genlength,   "Table 4  — Generation lengths"),
    "keywords":    (table_keywords,    "Table 5  — Keyword extraction"),
    "parse":       (table_parse,       "Supp.    — Parse failure rate"),
    "split":       (table_split,       "Supp.    — Split-partial verifier"),
    "untrunc":     (table_untrunc,     "Rev.R1   — Untruncated BERTScore"),
    "constrained": (table_constrained, "Rev.R2   — Constrained decoding"),
    "adversarial": (table_adversarial, "Rev.R3   — Adversarial fail-safe"),
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_tables",
        description="Generate LaTeX tables for the P.R.O.T.E.U.S. paper.",
    )
    p.add_argument("--table", choices=list(TABLE_MAP.keys()),
                   help="Single table to generate.")
    p.add_argument("--all", action="store_true",
                   help="Generate all tables in order.")
    p.add_argument("--results-dir", default="results",
                   help="Directory containing result JSON files (default: results/).")
    p.add_argument("--href-dir", default="results/href",
                   help="Directory containing h_ref result files (default: results/href/).")
    p.add_argument("--n-resamples", type=int, default=10_000,
                   help="Bootstrap/permutation resamples for significance tests.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if not args.table and not args.all:
        build_argparser().print_help()
        return 1
    kwargs = dict(
        results_dir=args.results_dir,
        href_dir=args.href_dir,
        n_resamples=args.n_resamples,
    )
    if args.all:
        for key, (fn, title) in TABLE_MAP.items():
            sep = "─" * max(1, 50 - len(title))
            print(f"% ── {title} {sep}")
            print(fn(**kwargs))
            print()
    else:
        fn, title = TABLE_MAP[args.table]
        sep = "─" * max(1, 50 - len(title))
        print(f"% ── {title} {sep}")
        print(fn(**kwargs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
