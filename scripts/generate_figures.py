"""
generate_figures.py
===================
Master plotting script for all P.R.O.T.E.U.S. paper figures (2-8).

Degrades gracefully when 70B result files are absent — those figures
are produced with a PENDING watermark so the layout is fixed and only
the data needs dropping in later.

Usage
-----
python generate_figures.py                        # all figures
python generate_figures.py --figures 2 4 6        # specific subset
python generate_figures.py --format pdf           # PDF instead of PNG
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ── Global style ─────────────────────────────────────────────────────────────
PALETTE = {
    "8b_cbert":  "#4C72B0",
    "8b_medcpt": "#DD8452",
    "8b_minilm": "#55A868",
    "8b_norag":  "#C44E52",
    "8b_nlick":  "#8172B2",
    "70b_rag":   "#937860",
    "70b_norag": "#DA8BC3",
}
GREY = "#AAAAAA"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def load(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def per_ex(data, field):
    return [float(ex["scores"][field])
            for ex in data.get("per_example", [])
            if field in ex.get("scores", {})
            and ex["scores"][field] is not None
            and ex["scores"][field] == ex["scores"][field]]

def pending_watermark(ax):
    ax.text(0.5, 0.5, "PENDING\n(Phase 4)", transform=ax.transAxes,
            fontsize=16, color=GREY, alpha=0.45, ha="center", va="center",
            rotation=25, fontweight="bold")

def paired_bootstrap(a, b, n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.array(a) - np.array(b)
    obs = diffs.mean()
    boots = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                      for _ in range(n)])
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def save(fig, out, name, fmt):
    path = out / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")

# ── Figure 2: h_ctx violin ────────────────────────────────────────────────────
def fig2_hctx_violin(r, h, out, fmt):
    cfgs = [
        ("8B RAG\nClinBERT",  f"{r}/qa_8b_rag_cbert.json",     PALETTE["8b_cbert"]),
        ("8B RAG\nMedCPT",    f"{r}/qa_8b_rag_medcpt.json",    PALETTE["8b_medcpt"]),
        ("8B RAG\nMiniLM",    f"{r}/qa_8b_rag_minilm.json",    PALETTE["8b_minilm"]),
        ("8B\nRAG+NLI",       f"{r}/qa_8b_rag_cbert_ver.json", PALETTE["8b_nlick"]),
        ("70B RAG\nClinBERT", f"{r}/qa_70b_rag_cbert.json",    PALETTE["70b_rag"]),
    ]
    labels, data_all, colors, pending = [], [], [], False
    for label, path, color in cfgs:
        d = load(path)
        if d is None:
            pending = True; continue
        vals = [v * 100 for v in per_ex(d, "hallucination_rate")]
        if vals:
            labels.append(label); data_all.append(vals); colors.append(color)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if data_all:
        parts = ax.violinplot(data_all, positions=range(len(data_all)),
                              showmedians=True, showextrema=False, widths=0.65)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_alpha(0.65)
            pc.set_edgecolor("white"); pc.set_linewidth(0.8)
        parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
        for i, (vals, c) in enumerate(zip(data_all, colors)):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=14, color=c, alpha=0.55, zorder=3, edgecolors="none")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, linespacing=1.2)
        ax.set_ylabel(r"$h_{\mathrm{ctx}}$ (%)")
        ax.set_ylim(0, 108)
        ax.axhline(50, color=GREY, lw=0.8, ls="--", alpha=0.6)
        ax.set_title(r"Per-query $h_{\mathrm{ctx}}$ distribution by configuration", pad=8)
    if pending:
        pending_watermark(ax)
    save(fig, out, "fig2_hctx_violin", fmt)

# ── Figure 3: Forest plot ─────────────────────────────────────────────────────
def fig3_forest_plot(r, h, out, fmt):
    anchor  = load(f"{r}/qa_8b_rag_cbert.json")
    norag   = load(f"{r}/qa_8b_norag.json")
    href_a  = load(f"{h}/qa_8b_rag_cbert_href.json")
    href_n  = load(f"{h}/qa_8b_norag_href.json")
    u70     = load(f"{r}/qa_70b_rag_cbert.json")

    rows = []
    for label, metric, scale, a_src, b_src, field in [
        ("8B-RAG vs 70B-RAG",  "BS-F1",                   1.0,   anchor, u70,    "bertscore_f1"),
        ("8B-RAG vs 70B-RAG",  r"$h_{\rm ctx}$ (pp)",     100.0, anchor, u70,    "hallucination_rate"),
        ("8B-RAG vs 8B-NoRAG", "BS-F1",                   1.0,   anchor, norag,  "bertscore_f1"),
        ("8B-RAG vs 8B-NoRAG", r"$h_{\rm ref}$ (pp)",     100.0, href_a, href_n, "h_ref"),
    ]:
        if a_src and b_src:
            a = [v * scale for v in per_ex(a_src, field)]
            b = [v * scale for v in per_ex(b_src, field)]
            if a and b and len(a) == len(b):
                d, lo, hi = paired_bootstrap(a, b)
                rows.append(dict(label=label, metric=metric,
                                 d=d, lo=lo, hi=hi, pending=False))
                continue
        rows.append(dict(label=label, metric=metric,
                         d=0, lo=0, hi=0, pending=True))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y_pos = list(range(len(rows)))[::-1]
    for row, y in zip(rows, y_pos):
        lbl = f"{row['label']}\n{row['metric']}"
        if row["pending"]:
            ax.scatter(0, y, marker="D", s=55, color=GREY,
                       zorder=5, edgecolors="#888888", lw=0.8)
            ax.text(0.25, y, "PENDING", color=GREY, va="center",
                    fontsize=8.5, style="italic")
        else:
            c = PALETTE["8b_cbert"]
            ax.errorbar(row["d"], y,
                        xerr=[[row["d"] - row["lo"]], [row["hi"] - row["d"]]],
                        fmt="D", color=c, ecolor=c,
                        capsize=4, capthick=1.5, ms=6, lw=1.8, zorder=5)
            ax.barh(y, row["hi"] - row["lo"], left=row["lo"],
                    height=0.28, color=c, alpha=0.18, zorder=2)
    ax.axvline(0, color="black", lw=1.0, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['label']}\n{r['metric']}" for r in rows], fontsize=8.5)
    ax.set_xlabel(r"$\Delta$ (anchor $-$ comparison)")
    ax.set_title("Pairwise contrasts — paired bootstrap 95% CI\n"
                 "Anchor: 8B Q4 + RAG (ClinicalBERT)", pad=8)
    ax.text(0.01, -0.14,
            "CI left of 0 → anchor better   |   CI right of 0 → comparison better",
            transform=ax.transAxes, fontsize=7.5, color=GREY)
    save(fig, out, "fig3_forest_plot", fmt)

# ── Figure 4: Keyword class bars ──────────────────────────────────────────────
def fig4_keyword_bars(r, h, out, fmt):
    cfgs = [
        ("8B ClinBERT",  f"{r}/kw_8b_rag_clinicalbert.json", PALETTE["8b_cbert"]),
        ("8B MedCPT",    f"{r}/kw_8b_rag_medcpt.json",       PALETTE["8b_medcpt"]),
        ("8B MiniLM",    f"{r}/kw_8b_rag_minilm.json",       PALETTE["8b_minilm"]),
        ("8B NoRAG",     f"{r}/kw_8b_norag.json",             PALETTE["8b_norag"]),
        ("70B ClinBERT", f"{r}/kw_70b_rag_clinicalbert.json", PALETTE["70b_rag"]),
        ("70B NoRAG",    f"{r}/kw_70b_norag.json",            PALETTE["70b_norag"]),
    ]
    cls_keys   = ["symptoms_f1_mean", "diagnostics_f1_mean", "pathogens_f1_mean"]
    cls_labels = ["Symptoms", "Diagnostics", "Pathogens"]
    cls_colors = ["#4393C3", "#74C476", "#FC8D59"]

    labels, matrix, pending = [], [], False
    for label, path, _ in cfgs:
        d = load(path)
        if d is None:
            pending = True; continue
        labels.append(label)
        matrix.append([d["aggregate"].get(k, 0.0) for k in cls_keys])

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 4.5))
    if not matrix:
        pending_watermark(ax)
    else:
        x = np.arange(len(labels))
        w = 0.26
        for ci, (cls, cls_c, off) in enumerate(zip(cls_labels, cls_colors,
                                                    [-w, 0, w])):
            vals = [row[ci] for row in matrix]
            bars = ax.bar(x + off, vals, w - 0.02, color=cls_c,
                          alpha=0.85, label=cls, zorder=3)
            for bar, val in zip(bars, vals):
                if val > 0.005:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.003,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=6.5, color="#333")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("F$_1$")
        top = max(v for row in matrix for v in row)
        ax.set_ylim(0, max(0.15, top * 1.35))
        ax.legend(loc="upper right", framealpha=0.85)
        ax.set_title("Keyword extraction F$_1$ by entity class", pad=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
    if pending:
        ax.text(0.98, 0.97, "70B rows pending",
                transform=ax.transAxes, fontsize=8, color=GREY,
                ha="right", va="top", style="italic")
    save(fig, out, "fig4_keyword_bars", fmt)

# ── Figure 5: Retriever radar ─────────────────────────────────────────────────
def fig5_retriever_radar(r, h, out, fmt):
    retrievers = [
        ("ClinicalBERT", f"{r}/qa_8b_rag_cbert.json",
                         f"{r}/kw_8b_rag_clinicalbert.json", PALETTE["8b_cbert"]),
        ("MedCPT",       f"{r}/qa_8b_rag_medcpt.json",
                         f"{r}/kw_8b_rag_medcpt.json",       PALETTE["8b_medcpt"]),
        ("MiniLM",       f"{r}/qa_8b_rag_minilm.json",
                         f"{r}/kw_8b_rag_minilm.json",       PALETTE["8b_minilm"]),
    ]
    axes_labels = ["BS-F1", "ROUGE-L", r"$1-h_{\rm ctx}$", r"KW $\mu$-F1"]
    n = len(axes_labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.00"], size=7, color=GREY)
    ax.grid(color=GREY, alpha=0.4)

    raw, all_vals = [], {i: [] for i in range(n)}
    for label, qa_p, kw_p, color in retrievers:
        qa = load(qa_p); kw = load(kw_p)
        if not qa or not kw:
            continue
        vals = [qa["aggregate"]["bertscore_f1_mean"],
                qa["aggregate"]["rougeL_f_mean"],
                1 - qa["aggregate"]["hallucination_rate_mean"],
                kw["aggregate"]["micro_f1_mean"]]
        for i, v in enumerate(vals):
            all_vals[i].append(v)
        raw.append((label, vals, color))

    if not raw:
        pending_watermark(ax)
    else:
        lo = [min(all_vals[i]) - 0.01 for i in range(n)]
        hi = [max(all_vals[i]) + 0.01 for i in range(n)]
        for label, vals, color in raw:
            norm = [(v - l)/(h_ - l) if h_ > l else 0.5
                    for v, l, h_ in zip(vals, lo, hi)]
            norm += norm[:1]
            ax.plot(angles, norm, color=color, lw=2, label=label)
            ax.fill(angles, norm, color=color, alpha=0.15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12),
                  framealpha=0.85, fontsize=9)

    ax.set_title("Retriever profile (normalised per axis)\n8B Q4 configurations",
                 pad=18, size=10)
    save(fig, out, "fig5_retriever_radar", fmt)

# ── Figure 6: Gen-length vs BS-F1 scatter ─────────────────────────────────────
def fig6_genlength_scatter(r, h, out, fmt):
    cfgs = [
        ("8B RAG ClinBERT",  f"{r}/qa_8b_rag_cbert.json",     PALETTE["8b_cbert"],  "o"),
        ("8B RAG MedCPT",    f"{r}/qa_8b_rag_medcpt.json",    PALETTE["8b_medcpt"], "s"),
        ("8B RAG MiniLM",    f"{r}/qa_8b_rag_minilm.json",    PALETTE["8b_minilm"], "^"),
        ("8B NoRAG",         f"{r}/qa_8b_norag.json",          PALETTE["8b_norag"],  "D"),
        ("8B RAG+NLI",       f"{r}/qa_8b_rag_cbert_ver.json", PALETTE["8b_nlick"],  "P"),
        ("70B RAG ClinBERT", f"{r}/qa_70b_rag_cbert.json",    PALETTE["70b_rag"],   "o"),
        ("70B NoRAG",        f"{r}/qa_70b_norag.json",          PALETTE["70b_norag"], "D"),
    ]
    TRUNC = 150
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    pending = False
    for label, path, color, marker in cfgs:
        d = load(path)
        if d is None:
            pending = True; continue
        lens = [len(ex["generated"].split()) for ex in d["per_example"]]
        bsf1 = per_ex(d, "bertscore_f1")
        if not lens: continue
        ax.scatter(lens, bsf1, c=color, marker=marker,
                   s=38, alpha=0.65, edgecolors="none", zorder=4, label=label)
    ax.axvline(TRUNC, color="#C44E52", lw=1.2, ls="--", zorder=5,
               label=f"{TRUNC}-token truncation cap")
    xlim = ax.get_xlim()
    ax.set_xlim(left=0)
    ax.axvspan(TRUNC, max(xlim[1], TRUNC + 50), alpha=0.07, color="#C44E52", zorder=1)
    ax.text(TRUNC + 5, ax.get_ylim()[0] + 0.01,
            "truncated\nregion", fontsize=7.5, color="#C44E52", va="bottom")
    ax.set_xlabel("Generation length (tokens, pre-truncation)")
    ax.set_ylabel("BERTScore F$_1$")
    ax.set_title("Generation length vs. BERTScore F$_1$: truncation confound\n"
                 "Each point = one QA output", pad=8)
    ax.legend(fontsize=8, framealpha=0.85, loc="upper right", markerscale=1.2)
    ax.grid(alpha=0.25, zorder=0)
    if pending:
        ax.text(0.02, 0.97, "70B points pending",
                transform=ax.transAxes, fontsize=8, color=GREY,
                ha="left", va="top", style="italic")
    save(fig, out, "fig6_genlength_scatter", fmt)

# ── Figure 7: Split-partial stacked bar ──────────────────────────────────────
def fig7_split_bar(r, h, out, fmt):
    std_d   = load(f"{r}/qa_8b_rag_cbert_ver.json")
    split_d = load(f"{r}/qa_8b_rag_cbert_ver_split.json")

    SEG_COLORS = {
        "Supported":              "#55A868",
        "Partial\n(collapsed)":   "#FFC107",
        "Weak\nEntailment":       "#DD8452",
        "Citation\nNon-comply":   "#9E9E9E",
        "Unsupported":            "#C44E52",
    }

    def counts(data, split_mode):
        per = [ex for ex in data.get("per_example", []) if ex.get("verifier")]
        tot = sum(ex["verifier"]["n_supported"]
                  + ex["verifier"].get("n_partial", 0)
                  + ex["verifier"].get("n_weak_entailment", 0)
                  + ex["verifier"].get("n_citation_noncompliance", 0)
                  + ex["verifier"]["n_unsupported"] for ex in per)
        if tot == 0: return {}
        sup = sum(ex["verifier"]["n_supported"] for ex in per) / tot
        uns = sum(ex["verifier"]["n_unsupported"] for ex in per) / tot
        if split_mode:
            we = sum(ex["verifier"].get("n_weak_entailment", 0) for ex in per) / tot
            cn = sum(ex["verifier"].get("n_citation_noncompliance", 0) for ex in per) / tot
            return {"Supported": sup, "Weak\nEntailment": we,
                    "Citation\nNon-comply": cn, "Unsupported": uns}
        par = sum(ex["verifier"].get("n_partial", 0) for ex in per) / tot
        return {"Supported": sup, "Partial\n(collapsed)": par, "Unsupported": uns}

    rows = []
    if std_d:
        rows.append(("Standard\n(3-label)", counts(std_d, False)))
    if split_d:
        rows.append(("Split-partial\n(4-label)", counts(split_d, True)))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    if not rows:
        pending_watermark(ax)
    else:
        x = np.arange(len(rows))
        bottoms = np.zeros(len(rows))
        for seg, color in SEG_COLORS.items():
            vals = np.array([row[1].get(seg, 0.0) for row in rows])
            if vals.sum() == 0: continue
            bars = ax.bar(x, vals, bottom=bottoms, color=color,
                          width=0.42, label=seg, zorder=3,
                          edgecolor="white", linewidth=0.6)
            for bar, val, bot in zip(bars, vals, bottoms):
                if val > 0.04:
                    ax.text(bar.get_x() + bar.get_width()/2, bot + val/2,
                            f"{val*100:.1f}%", ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold",
                            path_effects=[
                                pe.withStroke(linewidth=1.5, foreground="black")])
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([row[0] for row in rows], fontsize=10)
        ax.set_ylabel("Fraction of atomic claims")
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%","25%","50%","75%","100%"])
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9,
                  title="Claim label", title_fontsize=8)
        ax.set_title("NLI verifier label distribution:\nstandard vs. split-partial prompt", pad=8)
        if split_d:
            ax.annotate("WE > CN\n(grounding safety\ndominates formatting)",
                        xy=(1, 0.5), xytext=(1.45, 0.65),
                        fontsize=7.5, color="#DD8452",
                        arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2))
    save(fig, out, "fig7_split_partial_bar", fmt)

# ── Figure 8: Throughput vs h_ctx Pareto ─────────────────────────────────────
def fig8_pareto(r, h, out, fmt):
    cfgs = [
        ("8B RAG ClinBERT",  f"{r}/qa_8b_rag_cbert.json",     PALETTE["8b_cbert"],  "o"),
        ("8B RAG MedCPT",    f"{r}/qa_8b_rag_medcpt.json",    PALETTE["8b_medcpt"], "s"),
        ("8B RAG MiniLM",    f"{r}/qa_8b_rag_minilm.json",    PALETTE["8b_minilm"], "^"),
        ("8B RAG+NLI",       f"{r}/qa_8b_rag_cbert_ver.json", PALETTE["8b_nlick"],  "P"),
        ("70B RAG ClinBERT", f"{r}/qa_70b_rag_cbert.json",    PALETTE["70b_rag"],   "o"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    points, pending = [], False
    for label, path, color, marker in cfgs:
        d = load(path)
        if d is None:
            pending = True; continue
        tps  = d["latency"]["tokens_per_second_mean"]
        hctx = d["aggregate"].get("hallucination_rate_mean")
        if hctx is None: continue
        h_pct = hctx * 100
        ax.scatter(tps, h_pct, c=color, marker=marker,
                   s=90, zorder=5, edgecolors="white", lw=0.8)
        ax.annotate(label, (tps, h_pct),
                    textcoords="offset points", xytext=(6, 2),
                    fontsize=7.5, color=color)
        points.append((tps, h_pct))
    if len(points) >= 2:
        pts = sorted(points, key=lambda p: p[0])
        pareto, min_h = [], float("inf")
        for p in pts:
            if p[1] < min_h:
                pareto.append(p); min_h = p[1]
        if len(pareto) >= 2:
            ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
                    color=GREY, lw=1.2, ls="--", zorder=2,
                    label="Pareto frontier")
    ax.set_xlabel("Decoding throughput (tok/s)")
    ax.set_ylabel(r"$h_{\mathrm{ctx}}$ (%) — lower is better")
    ax.set_title("Speed–grounding tradeoff (lower-right = preferable)", pad=8)
    ax.invert_yaxis()
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8.5)
    if pending:
        ax.text(0.98, 0.03, "70B point pending Phase 4",
                transform=ax.transAxes, fontsize=8, color=GREY,
                ha="right", va="bottom", style="italic")
        pending_watermark(ax)
    save(fig, out, "fig8_pareto_frontier", fmt)

# ── CLI ───────────────────────────────────────────────────────────────────────
FIG_MAP = {
    2: ("h_ctx violin",        fig2_hctx_violin),
    3: ("forest plot",         fig3_forest_plot),
    4: ("keyword bars",        fig4_keyword_bars),
    5: ("retriever radar",     fig5_retriever_radar),
    6: ("gen-length scatter",  fig6_genlength_scatter),
    7: ("split-partial bar",   fig7_split_bar),
    8: ("pareto frontier",     fig8_pareto),
}

def main(argv=None):
    p = argparse.ArgumentParser(prog="generate_figures",
        description="Generate all P.R.O.T.E.U.S. paper figures.")
    p.add_argument("--figures", nargs="+", type=int,
                   default=list(FIG_MAP), metavar="N")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--href-dir",    default="results/href")
    p.add_argument("--out",         default="figs")
    p.add_argument("--format",      default="png",
                   choices=("png", "pdf", "svg"))
    args = p.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    r, h, fmt = args.results_dir, args.href_dir, args.format

    print(f"Output: {out}/  |  Format: {fmt}  |  Figs: {args.figures}\n")
    for n in sorted(set(args.figures)):
        if n not in FIG_MAP:
            print(f"  [skip] unknown figure {n}"); continue
        desc, fn = FIG_MAP[n]
        print(f"Fig {n}: {desc}")
        try:
            fn(r, h, out, fmt)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
    print("\nAll done.")

if __name__ == "__main__":
    sys.exit(main())
