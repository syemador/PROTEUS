"""
generate_figures.py  (v2 — visual upgrade)
==========================================
Master plotting script for all P.R.O.T.E.U.S. paper figures (2-8).

New in v2
---------
- Richer global style: tighter font stack, subtle warm-white background,
  stronger title weight, consistent spine/grid treatment.
- Vibrant but academically restrained palette with alpha-gradient fills.
- Per-figure enhancements documented inline.
- --format both  →  writes PNG (300 dpi) AND PDF in one pass.

Usage
-----
python generate_figures.py                            # all, PNG
python generate_figures.py --figures 2 4 6            # subset
python generate_figures.py --format pdf               # PDF only
python generate_figures.py --format both              # PNG + PDF
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.colors as mcolors
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType, IEEE PDF eXpress compliant
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ── Palette ───────────────────────────────────────────────────────────────────
# Slightly more saturated than v1 — still IEEE-printable in greyscale.
PALETTE = {
    "8b_cbert":  "#2D6DB5",   # strong blue
    "8b_medcpt": "#E07B39",   # warm orange
    "8b_minilm": "#2E9E5B",   # forest green
    "8b_norag":  "#C0392B",   # crimson
    "8b_nlick":  "#6C5FC7",   # violet
    "70b_rag":   "#7B5EA7",   # purple (scale-upgrade signal)
    "70b_norag": "#B03A7D",   # magenta-rose
}
ACCENT  = "#E63946"   # highlight red (significance stars, key annotations)
GREY    = "#999999"
LGREY   = "#E8E8E8"
BG      = "#FAFAFA"   # warm off-white figure background

# ── Global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    # Font
    "font.family":          "DejaVu Sans",
    "font.size":            10,
    "axes.titlesize":       12,
    "axes.titleweight":     "bold",
    "axes.labelsize":       10,
    "axes.labelweight":     "semibold",
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "legend.title_fontsize": 9,
    # Spines — keep left + bottom only
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.edgecolor":       "#555555",
    "axes.linewidth":       0.9,
    # Grid
    "axes.grid":            True,
    "grid.color":           LGREY,
    "grid.linewidth":       0.6,
    "grid.alpha":           0.9,
    "axes.axisbelow":       True,
    # Background
    "axes.facecolor":       BG,
    "figure.facecolor":     "white",
    # Output
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.08,
    "savefig.facecolor":    "white",
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
    ax.text(0.5, 0.5, "PENDING", transform=ax.transAxes,
            fontsize=22, color=LGREY, alpha=0.6, ha="center", va="center",
            rotation=25, fontweight="bold", style="italic")

def paired_bootstrap(a, b, n=10_000, seed=42):
    rng  = np.random.default_rng(seed)
    diffs = np.array(a) - np.array(b)
    obs   = diffs.mean()
    boots = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                      for _ in range(n)])
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def is_significant(lo, hi):
    """CI entirely on one side of zero."""
    return (lo > 0) or (hi < 0)

def save(fig, out, name, fmt):
    """Save in one or both formats."""
    formats = ["png", "pdf"] if fmt == "both" else [fmt]
    for f in formats:
        path = out / f"{name}.{f}"
        fig.savefig(path, format=f)
        print(f"  -> {path}")
    plt.close(fig)


# ── Figure 2: h_ctx violin ────────────────────────────────────────────────────
# Upgrades:
#   - Wider violins with a stronger alpha gradient (lighter top, denser body)
#   - Median line styled white on a dark edge for visibility
#   - Individual points use darker edge so they pop against the body
#   - 50 % reference line annotated with a label box
#   - Subtle horizontal zebra-stripe background bands
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

    fig, ax = plt.subplots(figsize=(9, 5))

    if data_all:
        # Zebra bands every 25 pp for depth
        for band_lo, band_hi in [(0, 25), (50, 75), (100, 108)]:
            ax.axhspan(band_lo, band_hi, color=LGREY, alpha=0.25, zorder=0)

        parts = ax.violinplot(data_all, positions=range(len(data_all)),
                              showmedians=True, showextrema=False, widths=0.72)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.70)
            pc.set_edgecolor(mcolors.to_rgba(c, 0.9))
            pc.set_linewidth(1.2)
        # Median: white line with coloured halo
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2.5)
        parts["cmedians"].set_path_effects(
            [pe.withStroke(linewidth=4.0, foreground="#333333")])

        for i, (vals, c) in enumerate(zip(data_all, colors)):
            jitter = np.random.default_rng(42).uniform(-0.14, 0.14, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=18, color=c, alpha=0.70, zorder=4,
                       edgecolors=mcolors.to_rgba(c, 0.5), linewidths=0.6)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, linespacing=1.3, fontsize=9.5)
        ax.set_ylabel(r"$h_{\mathrm{ctx}}$ (%)", fontsize=10)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])

        # Annotated 50 % reference
        ax.axhline(50, color=ACCENT, lw=1.0, ls="--", alpha=0.7, zorder=3)
        ax.text(len(data_all) - 0.45, 52, "50 %", fontsize=8,
                color=ACCENT, va="bottom", ha="right", alpha=0.85)

        ax.set_title(r"Per-query $h_{\mathrm{ctx}}$ distribution by configuration", pad=10)
        ax.set_xlabel("Configuration")

    if pending:
        pending_watermark(ax)
    save(fig, out, "fig2_hctx_violin", fmt)


# ── Figure 3: Forest plot ─────────────────────────────────────────────────────
# Upgrades:
#   - Significant rows: filled diamond + green CI band; non-significant: grey
#   - Significance star annotation on the significant row
#   - Alternating row background for readability
#   - Cleaner bottom guide text via fig.text
def fig3_forest_plot(r, h, out, fmt):
    anchor = load(f"{r}/qa_8b_rag_cbert.json")
    norag  = load(f"{r}/qa_8b_norag.json")
    href_a = load(f"{h}/qa_8b_rag_cbert_href.json")
    href_n = load(f"{h}/qa_8b_norag_href.json")
    u70    = load(f"{r}/qa_70b_rag_cbert.json")

    rows = []
    for label, metric, scale, a_src, b_src, field in [
        ("8B-RAG vs 70B-RAG",  "BS-F1",             1.0,   anchor, u70,    "bertscore_f1"),
        ("8B-RAG vs 70B-RAG",  r"$h_\mathrm{ctx}$ (pp)", 100.0, anchor, u70, "hallucination_rate"),
        ("8B-RAG vs 8B-NoRAG", "BS-F1",             1.0,   anchor, norag,  "bertscore_f1"),
        ("8B-RAG vs 8B-NoRAG", r"$h_\mathrm{ref}$ (pp)", 100.0, href_a, href_n, "h_ref"),
    ]:
        if a_src and b_src:
            a = [v * scale for v in per_ex(a_src, field)]
            b = [v * scale for v in per_ex(b_src, field)]
            if a and b and len(a) == len(b):
                d, lo, hi = paired_bootstrap(a, b)
                rows.append(dict(label=label, metric=metric,
                                 d=d, lo=lo, hi=hi,
                                 sig=is_significant(lo, hi), pending=False))
                continue
        rows.append(dict(label=label, metric=metric,
                         d=0, lo=0, hi=0, sig=False, pending=True))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y_pos = list(range(len(rows)))[::-1]

    # Alternating row bands
    for y in y_pos:
        if y % 2 == 0:
            ax.axhspan(y - 0.45, y + 0.45, color=LGREY, alpha=0.35, zorder=0)

    for row, y in zip(rows, y_pos):
        if row["pending"]:
            ax.scatter(0, y, marker="D", s=55, color=GREY,
                       zorder=5, edgecolors="#888888", lw=0.8)
            ax.text(0.3, y, "PENDING", color=GREY, va="center",
                    fontsize=8.5, style="italic")
            continue

        sig   = row["sig"]
        c_pt  = "#1A7340" if sig else PALETTE["8b_cbert"]  # green=sig, blue=not
        c_ci  = mcolors.to_rgba(c_pt, 0.20)
        # CI band
        ax.barh(y, row["hi"] - row["lo"], left=row["lo"],
                height=0.38, color=c_ci, zorder=2)
        # Error bar
        ax.errorbar(row["d"], y,
                    xerr=[[row["d"] - row["lo"]], [row["hi"] - row["d"]]],
                    fmt="D" if sig else "o",
                    color=c_pt, ecolor=c_pt,
                    capsize=5, capthick=1.8,
                    ms=7 if sig else 6, lw=2.0, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
        # Significance star
        if sig:
            ax.text(row["d"], y + 0.28, "★", color=c_pt,
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axvline(0, color="#333333", lw=1.2, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['label']}\n{r['metric']}" for r in rows], fontsize=9)
    ax.set_xlabel(r"$\Delta$ (anchor $-$ comparison)")
    ax.set_title("Pairwise contrasts — paired bootstrap 95% CI\n"
                 "Anchor: 8B Q4 + RAG (ClinicalBERT)", pad=10)

    # Legend for colour coding
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#1A7340",
               markersize=8, label="Significant (CI excludes 0)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=PALETTE["8b_cbert"],
               markersize=7, label="Not significant"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5,
              framealpha=0.9, edgecolor=LGREY)

    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.03,
             "CI entirely left of 0  →  anchor better   |   "
             "CI entirely right of 0  →  comparison better",
             ha="center", va="bottom", fontsize=8, color=GREY)
    save(fig, out, "fig3_forest_plot", fmt)


# ── Figure 4: Keyword class bars ──────────────────────────────────────────────
# Upgrades:
#   - Gradient-like effect via layered bar + edge (edge 20% darker)
#   - μ-F1 overlaid as a line+scatter on a twin axis for easy comparison
#   - Value labels only on non-zero bars, slightly larger font
#   - Grid on y only, removed from x
def fig4_keyword_bars(r, h, out, fmt):
    cfgs = [
        ("8B ClinBERT",  f"{r}/kw_8b_rag_clinicalbert.json"),
        ("8B MedCPT",    f"{r}/kw_8b_rag_medcpt.json"),
        ("8B MiniLM",    f"{r}/kw_8b_rag_minilm.json"),
        ("8B NoRAG",     f"{r}/kw_8b_norag.json"),
        ("70B ClinBERT", f"{r}/kw_70b_rag_clinicalbert.json"),
        ("70B NoRAG",    f"{r}/kw_70b_norag.json"),
    ]
    cls_keys   = ["symptoms_f1_mean", "diagnostics_f1_mean", "pathogens_f1_mean"]
    cls_labels = ["Symptoms", "Diagnostics", "Pathogens"]
    # Vivid class colours — distinct in colour AND greyscale
    cls_colors = ["#2166AC", "#4DAC26", "#D6604D"]

    labels, matrix, micro_vals, pending = [], [], [], False
    for label, path in cfgs:
        d = load(path)
        if d is None:
            pending = True; continue
        labels.append(label)
        matrix.append([d["aggregate"].get(k, 0.0) for k in cls_keys])
        micro_vals.append(d["aggregate"].get("micro_f1_mean", 0.0))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    ax2 = ax.twinx()   # twin for μ-F1 line

    if not matrix:
        pending_watermark(ax)
    else:
        x   = np.arange(len(labels))
        w   = 0.25
        offsets = [-w, 0, w]

        for ci, (cls, c, off) in enumerate(zip(cls_labels, cls_colors, offsets)):
            vals = [row[ci] for row in matrix]
            # Main bar
            bars = ax.bar(x + off, vals, w - 0.025,
                          color=c, alpha=0.82, label=cls,
                          zorder=3, edgecolor=mcolors.to_rgba(c, 1.0),
                          linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0.008:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.004,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=7, color="#222222", fontweight="semibold")

        # μ-F1 overlay line
        ax2.plot(x, micro_vals, color="#333333", lw=2.0, ls="-.",
                 marker="D", ms=7, zorder=6,
                 markerfacecolor="white", markeredgecolor="#333333",
                 markeredgewidth=1.5, label=r"$\mu$-F1")
        for xi, mv in zip(x, micro_vals):
            ax2.text(xi + 0.12, mv + 0.004, f"{mv:.3f}",
                     fontsize=7.5, color="#333333", va="bottom")
        ax2.set_ylabel(r"Micro-averaged F$_1$ ($\mu$-F1)", fontsize=9.5,
                       labelpad=6)
        top = max(v for row in matrix for v in row)
        ax2.set_ylim(0, max(0.18, top * 1.4))
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=16, ha="right", fontsize=9.5)
        ax.set_ylabel("Per-class F$_1$")
        ax.set_ylim(0, max(0.18, top * 1.4))

        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5,
                  framealpha=0.92, edgecolor=LGREY)

        ax.set_title("Keyword extraction F$_1$ by entity class and configuration",
                     pad=10)

    if pending:
        ax.text(0.98, 0.97, "70B rows pending",
                transform=ax.transAxes, fontsize=8, color=GREY,
                ha="right", va="top", style="italic")
    save(fig, out, "fig4_keyword_bars", fmt)


# ── Figure 5: Retriever radar ─────────────────────────────────────────────────
# Upgrades:
#   - Bolder fill (alpha 0.22) + strong line (lw 2.5) + coloured gridlines
#   - Vertex dots so exact normalised values are visible
#   - Radial tick labels pushed outward to avoid axis collision
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
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor("#F5F5F8")          # slightly blue-grey polar background
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, size=10.5, fontweight="semibold")
    ax.tick_params(axis="x", pad=16)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       size=7.5, color="#777777")
    # Coloured concentric gridlines
    for ytick, alpha in zip([0.25, 0.5, 0.75, 1.0], [0.3, 0.4, 0.5, 0.6]):
        ax.plot(angles, [ytick] * (n + 1), color="#AAAACC",
                lw=0.8, alpha=alpha, zorder=1)
    ax.set_rlabel_position(45)

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
            norm = [(v - l) / (hv - l) if hv > l else 0.5
                    for v, l, hv in zip(vals, lo, hi)]
            norm_closed = norm + norm[:1]
            ax.plot(angles, norm_closed, color=color, lw=2.5,
                    label=label, zorder=4)
            ax.fill(angles, norm_closed, color=color, alpha=0.22, zorder=3)
            # Vertex dots
            ax.scatter(angles[:-1], norm, s=50, color=color,
                       zorder=5, edgecolors="white", linewidths=0.8)

        ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.14),
                  framealpha=0.92, fontsize=9.5, edgecolor=LGREY)

    ax.set_title("Retriever profile (normalised per axis) — 8B Q4",
                 pad=22, size=11, fontweight="bold")
    save(fig, out, "fig5_retriever_radar", fmt)


# ── Figure 6: Gen-length vs BS-F1 scatter ─────────────────────────────────────
# Upgrades:
#   - Larger, edge-bordered points (pop against background)
#   - Truncation zone: gradient fill (dark at cap, fading right)
#   - Per-cluster convex hull or ellipse would be overkill; instead add
#     cluster mean crosshair markers
#   - Text labels use path_effects halo for legibility
def fig6_genlength_scatter(r, h, out, fmt):
    cfgs = [
        ("8B RAG ClinBERT",  f"{r}/qa_8b_rag_cbert.json",     PALETTE["8b_cbert"],  "o"),
        ("8B RAG MedCPT",    f"{r}/qa_8b_rag_medcpt.json",    PALETTE["8b_medcpt"], "s"),
        ("8B RAG MiniLM",    f"{r}/qa_8b_rag_minilm.json",    PALETTE["8b_minilm"], "^"),
        ("8B NoRAG",         f"{r}/qa_8b_norag.json",         PALETTE["8b_norag"],  "D"),
        ("8B RAG+NLI",       f"{r}/qa_8b_rag_cbert_ver.json", PALETTE["8b_nlick"],  "P"),
        ("70B RAG ClinBERT", f"{r}/qa_70b_rag_cbert.json",    PALETTE["70b_rag"],   "*"),
    ]
    TRUNC = 150
    ACCENT = "#C44E52"  # Fallback definition to prevent NameError
    LGREY = "#CCCCCC"   # Fallback definition to prevent NameError
    
    fig, ax = plt.subplots(figsize=(8, 5))
    pending = False

    for label, path, color, marker in cfgs:
        d = load(path)
        if d is None:
            pending = True; continue
        lens = [len(ex["generated"].split()) for ex in d["per_example"]]
        bsf1 = per_ex(d, "bertscore_f1")
        if not lens:
            continue
        ax.scatter(lens, bsf1, c=color, marker=marker,
                   s=55 if marker == "*" else 42,
                   alpha=0.78, zorder=4, label=label,
                   edgecolors=mcolors.to_rgba(color, 0.6),
                   linewidths=0.7)
        # Cluster mean crosshair
        mx, my = float(np.mean(lens)), float(np.mean(bsf1))
        ax.plot(mx, my, marker="+", ms=12, color=color,
                markeredgewidth=2.0, zorder=5, alpha=0.9)

    # Truncation cap line
    ax.axvline(TRUNC, color=ACCENT, lw=1.4, ls="--", zorder=6,
               label=f"{TRUNC}-token truncation cap")
    ax.set_xlim(left=0)
    xlim = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # FIX: Corrected pcolormesh grid alignment for the gradient zone
    # X and Y must be 2D meshgrids. C maps the color fade from 0 to 1.
    grad_x = np.linspace(TRUNC, max(xlim[1], TRUNC + 50), 100)
    grad_y = np.linspace(ymin, ymax, 2)
    X, Y = np.meshgrid(grad_x, grad_y)
    
    # C must be 2D array of shape (len(grad_y)-1, len(grad_x)-1)
    C = np.linspace(0, 1, len(grad_x) - 1).reshape(1, -1)
    
    # Custom colormap: ACCENT color fading from 18% opacity to 0% opacity
    color_rgb = mcolors.to_rgb(ACCENT)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_fade", 
        [(*color_rgb, 0.18), (*color_rgb, 0.0)]
    )
    
    ax.pcolormesh(X, Y, C, cmap=cmap, shading="flat", zorder=1)

    ax.text(TRUNC + 6, ymin + 0.005,
            "truncated region", fontsize=8, color=ACCENT, va="bottom",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.set_xlabel("Generation length (whitespace tokens, pre-truncation)")
    ax.set_ylabel("BERTScore F$_1$")
    ax.set_title("Generation length vs. BERTScore F$_1$ — truncation confound\n"
                 "Each point = one QA output   |   + = cluster mean",
                 pad=10)
    ax.legend(fontsize=8.5, framealpha=0.92, loc="upper right",
              markerscale=1.3, edgecolor=LGREY)

    if pending:
        ax.text(0.02, 0.97, "Some 70B points pending",
                transform=ax.transAxes, fontsize=8, color=GREY,
                ha="left", va="top", style="italic")
    save(fig, out, "fig6_genlength_scatter", fmt)


# ── Figure 7: Split-partial stacked bar ──────────────────────────────────────
def fig7_split_bar(r, h, out, fmt):
    std_d   = load(f"{r}/qa_8b_rag_cbert_ver.json")
    split_d = load(f"{r}/qa_8b_rag_cbert_ver_split.json")

# FIX: Removed \n characters to force single-line legend labels
    SEG_COLORS = {
        "Supported":           "#1A7340",   # deep green  (safe)
        "Partial (collapsed)": "#F0A500",   # amber       (warning)
        "Weak Entailment":     "#E07B39",   # orange      (grounding risk)
        "Citation Non-comply": "#9B9B9B",   # grey        (formatting)
        "Unsupported":         "#C0392B",   # crimson     (unsafe)
    }

    def counts(data, split_mode):
        per = [ex for ex in data.get("per_example", []) if ex.get("verifier")]
        tot = sum(
            ex["verifier"]["n_supported"]
            + ex["verifier"].get("n_partial", 0)
            + ex["verifier"].get("n_weak_entailment", 0)
            + ex["verifier"].get("n_citation_noncompliance", 0)
            + ex["verifier"]["n_unsupported"]
            for ex in per)
        if tot == 0:
            return {}
        sup = sum(ex["verifier"]["n_supported"] for ex in per) / tot
        uns = sum(ex["verifier"]["n_unsupported"] for ex in per) / tot
        if split_mode:
            we = sum(ex["verifier"].get("n_weak_entailment", 0) for ex in per) / tot
            cn = sum(ex["verifier"].get("n_citation_noncompliance", 0) for ex in per) / tot
            # FIX: Updated keys to match the new single-line strings in SEG_COLORS
            return {"Supported": sup, "Weak Entailment": we,
                    "Citation Non-comply": cn, "Unsupported": uns}
        par = sum(ex["verifier"].get("n_partial", 0) for ex in per) / tot
        # FIX: Updated keys to match the new single-line strings in SEG_COLORS
        return {"Supported": sup, "Partial (collapsed)": par, "Unsupported": uns}

    rows = []
    if std_d:
        rows.append(("Standard\n(3-label)", counts(std_d, False)))
    if split_d:
        rows.append(("Split-partial\n(4-label)", counts(split_d, True)))

    # FIX 1: Narrower figure to bring bars closer, but use a larger right margin 
    # to protect the callout box from being cut off.
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.subplots_adjust(bottom=0.30, right=0.68)  

    if not rows:
        pending_watermark(ax)
    else:
        x = np.arange(len(rows))
        bottoms = np.zeros(len(rows))

        for seg, color in SEG_COLORS.items():
            vals = np.array([row[1].get(seg, 0.0) for row in rows])
            if vals.sum() == 0: continue
            # FIX 2: Slightly wider bars (0.50) to balance the negative space
            bars = ax.bar(x, vals, bottom=bottoms, color=color,
                          width=0.50, label=seg, zorder=3,
                          edgecolor="white", linewidth=0.8)
            for bar, val, bot in zip(bars, vals, bottoms):
                if val > 0.04:
                    ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
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
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        
        # FIX 3: Clamp the x-limits to trim excessive outer whitespace
        ax.set_xlim(-0.45, 1.45)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=3, fontsize=8.5, framealpha=0.9,
                  title="Claim label", title_fontsize=8)

        ax.set_title("NLI verifier label distribution:\nstandard vs. split-partial prompt", pad=8)

        if split_d:
            ax.annotate(
                "WE > CN\n(grounding safety\ndominates formatting)",
                # Point to the right edge of the bar (x=1.25) to avoid crossing the text
                xy=(1.25, 0.638 + 0.117 / 2),         
                xycoords="data",
                xytext=(1.08, 0.73),                
                textcoords="axes fraction",
                fontsize=7.5, color="#DD8452",
                ha="left", va="center",
                # Removed connectionstyle to make the arrow straight
                arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#DD8452", lw=0.9, alpha=0.95),
                annotation_clip=False,
            )

    save(fig, out, "fig7_split_partial_bar", fmt)


# ── Figure 8: Throughput vs h_ctx Pareto ─────────────────────────────────────
# Upgrades:
#   - Shaded quadrant: bottom-right is the "sweet spot" — faint green fill
#   - Larger points with glowing edge (path_effects)
#   - Pareto frontier: dashed + arrow annotation pointing to best config
#   - Label text uses white halo for legibility against any background
def fig8_pareto(r, h, out, fmt):
    cfgs = [
        ("8B RAG\nClinBERT",  f"{r}/qa_8b_rag_cbert.json",     PALETTE["8b_cbert"],  "o"),
        ("8B RAG\nMedCPT",    f"{r}/qa_8b_rag_medcpt.json",    PALETTE["8b_medcpt"], "s"),
        ("8B RAG\nMiniLM",    f"{r}/qa_8b_rag_minilm.json",    PALETTE["8b_minilm"], "^"),
        ("8B RAG+NLI",        f"{r}/qa_8b_rag_cbert_ver.json", PALETTE["8b_nlick"],  "P"),
        ("70B RAG\nClinBERT", f"{r}/qa_70b_rag_cbert.json",    PALETTE["70b_rag"],   "*"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    points, pending = [], False

    for label, path, color, marker in cfgs:
        d = load(path)
        if d is None:
            pending = True; continue
        tps  = d["latency"]["tokens_per_second_mean"]
        hctx = d["aggregate"].get("hallucination_rate_mean")
        if hctx is None:
            continue
        h_pct = hctx * 100
        # Glowing edge on points
        ax.scatter(tps, h_pct, c=color,
                   marker=marker, s=160 if marker == "*" else 110,
                   zorder=5,
                   edgecolors=mcolors.to_rgba(color, 0.5),
                   linewidths=4.0)
        ax.scatter(tps, h_pct, c=color, marker=marker,
                   s=90 if marker == "*" else 70,
                   zorder=6, edgecolors="white", linewidths=1.2)
        ax.annotate(label, (tps, h_pct),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color, fontweight="semibold",
                    path_effects=[pe.withStroke(linewidth=2.5,
                                                foreground="white")])
        points.append((tps, h_pct, color))

    # Pareto frontier
    if len(points) >= 2:
        pts    = sorted(points, key=lambda p: p[0])
        pareto, min_h = [], float("inf")
        for p in pts:
            if p[1] < min_h:
                pareto.append(p); min_h = p[1]
        if len(pareto) >= 2:
            ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
                    color="#333333", lw=1.8, ls="--", zorder=3,
                    label="Pareto frontier")
            # Arrow to best point (rightmost Pareto point = fastest+lowest h_ctx)
            bx, by = pareto[-1][0], pareto[-1][1]
            ax.annotate("best\navailable",
                        xy=(bx, by), xytext=(bx - 8, by - 6),
                        fontsize=7.5, color="#1A7340",
                        arrowprops=dict(arrowstyle="->",
                                        color="#1A7340", lw=1.2))

    # Sweet-spot quadrant: fast AND low h_ctx — shaded light green
    ax.invert_yaxis()
    ylim = ax.get_ylim()   # after invert, ylim[0] > ylim[1]
    xlim = ax.get_xlim()
    mid_x = (xlim[0] + xlim[1]) / 2
    mid_y = (ylim[0] + ylim[1]) / 2
    # Bottom-right in display = high x, low h_ctx (low value = bottom after invert)
    ax.fill_betweenx([ylim[1], mid_y], mid_x, xlim[1],
                     color="#1A7340", alpha=0.06, zorder=0)
    ax.text(xlim[1] * 0.97, ylim[1] + (mid_y - ylim[1]) * 0.05,
            "preferred\nzone", fontsize=7.5, color="#1A7340",
            ha="right", va="bottom", alpha=0.7, style="italic")

    ax.set_xlabel("Decoding throughput (tok/s)", fontsize=10)
    ax.set_ylabel(r"$h_{\mathrm{ctx}}$ (%)  ↑ lower is better", fontsize=10)
    ax.set_title("Speed–grounding Pareto frontier\n"
                 "Lower-right = faster and better-grounded", pad=10)
    ax.legend(fontsize=8.5, framealpha=0.92, edgecolor=LGREY, loc="upper right")

    if pending:
        ax.text(0.98, 0.04, "Some configs pending",
                transform=ax.transAxes, fontsize=8, color=GREY,
                ha="right", va="bottom", style="italic")

    save(fig, out, "fig8_pareto_frontier", fmt)


# ── CLI ───────────────────────────────────────────────────────────────────────
FIG_MAP = {
    2: ("h_ctx violin",       fig2_hctx_violin),
    3: ("forest plot",        fig3_forest_plot),
    4: ("keyword bars",       fig4_keyword_bars),
    5: ("retriever radar",    fig5_retriever_radar),
    6: ("gen-length scatter", fig6_genlength_scatter),
    7: ("split-partial bar",  fig7_split_bar),
    8: ("pareto frontier",    fig8_pareto),
}

def main(argv=None):
    p = argparse.ArgumentParser(prog="generate_figures",
        description="Generate all P.R.O.T.E.U.S. paper figures (v2 — visual upgrade).")
    p.add_argument("--figures", nargs="+", type=int,
                   default=list(FIG_MAP), metavar="N")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--href-dir",    default="results/href")
    p.add_argument("--out",         default="figs")
    p.add_argument("--format",      default="png",
                   choices=("png", "pdf", "svg", "both"),
                   help="'both' writes PNG + PDF in one pass.")
    args = p.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    r, h, fmt = args.results_dir, args.href_dir, args.format

    print(f"Output : {out}/")
    print(f"Format : {fmt}  ({'PNG + PDF' if fmt == 'both' else fmt.upper()})")
    print(f"Figures: {args.figures}\n")

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