"""
preflight.py
============
Submission-readiness validator for the P.R.O.T.E.U.S. paper.

Run after rerun_all.py phases 1–5 and the three reviewer revision
experiments (exp1/exp2/exp3) to confirm every result file, computed
field, and derived artefact is present and internally consistent before
camera-ready submission.  Prints a clear pass/fail line for every check,
an exact FIX command for every failure, and a final summary.

Usage
-----
python preflight.py
python preflight.py --results-dir results --href-dir results/href
python preflight.py --strict        # non-zero exit on warnings too
python preflight.py --no-color      # plain output for CI logs
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
DIVIDER = "__DIVIDER__"


class Check:
    def __init__(self, name: str, status: str, message: str,
                 fix: Optional[str] = None):
        self.name    = name
        self.status  = status
        self.message = message
        self.fix     = fix


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _per_ex(data: Dict[str, Any], key: str) -> List[float]:
    return [
        float(ex["scores"][key])
        for ex in data.get("per_example", [])
        if key in ex.get("scores", {})
        and ex["scores"][key] is not None
        and ex["scores"][key] == ex["scores"][key]   # NaN guard
    ]


def _div(title: str) -> Check:
    return Check(DIVIDER, PASS, title)


# ---------------------------------------------------------------------------
# Original check helpers (phases 1–5)
# ---------------------------------------------------------------------------

def _check_qa(label: str, path: str, fix: str, n: int = 30) -> List[Check]:
    out: List[Check] = []
    d = _load(path)
    if d is None:
        out.append(Check(f"{label} / file", FAIL, f"missing: {path}", fix))
        return out
    out.append(Check(f"{label} / file", PASS,
                     f"ok ({Path(path).stat().st_size:,} B)"))
    n_ex = len(d.get("per_example", []))
    out.append(Check(
        f"{label} / n_examples",
        PASS if n_ex == n else FAIL,
        str(n_ex) if n_ex == n else f"expected {n}, got {n_ex}",
        fix if n_ex != n else None,
    ))
    has_flag = any(
        "parse_failed" in ex.get("scores", {})
        for ex in d.get("per_example", [])
    )
    if has_flag:
        n_f = sum(
            1 for ex in d["per_example"]
            if ex.get("scores", {}).get("parse_failed", False)
        )
        out.append(Check(f"{label} / parse_failed", PASS,
                         f"instrumented: {n_f}/{n_ex} failures"))
    else:
        out.append(Check(f"{label} / parse_failed", WARN,
                         "pre-instrumentation run — field absent",
                         "python rerun_all.py --phases 1"))
    agg = d.get("aggregate", {})
    needed = [
        "rouge1_f_mean", "rouge1_f_ci_low", "rouge1_f_ci_high",
        "rougeL_f_mean", "bertscore_f1_mean", "hallucination_rate_mean",
    ]
    missing = [k for k in needed if k not in agg]
    out.append(Check(
        f"{label} / aggregate",
        FAIL if missing else PASS,
        f"missing: {', '.join(missing)}" if missing else "all keys present",
        fix if missing else None,
    ))
    for meta in ("verifier_mode", "backend"):
        if meta not in d:
            out.append(Check(f"{label} / {meta}", WARN,
                             "absent — pre-update run",
                             "python rerun_all.py --phases 1"))
    return out


def _check_kw(label: str, path: str, fix: str) -> List[Check]:
    out: List[Check] = []
    d = _load(path)
    if d is None:
        out.append(Check(f"{label} / file", FAIL, f"missing: {path}", fix))
        return out
    out.append(Check(f"{label} / file", PASS,
                     f"ok ({Path(path).stat().st_size:,} B)"))
    agg = d.get("aggregate", {})
    for cls in ("symptoms", "diagnostics", "pathogens"):
        k = f"{cls}_f1_mean"
        out.append(Check(
            f"{label} / {cls}_F1",
            PASS if k in agg else FAIL,
            f"{agg[k]:.4f}" if k in agg else "absent",
            fix if k not in agg else None,
        ))
    k = "micro_f1_mean"
    out.append(Check(
        f"{label} / micro_F1",
        PASS if k in agg else FAIL,
        f"{agg[k]:.4f}" if k in agg else "absent",
        fix if k not in agg else None,
    ))
    return out


def _check_href(label: str, path: str, fix: str, n: int = 30) -> List[Check]:
    out: List[Check] = []
    d = _load(path)
    if d is None:
        out.append(Check(f"{label} / file", FAIL, f"missing: {path}", fix))
        return out
    out.append(Check(f"{label} / file", PASS, "ok"))
    vals = _per_ex(d, "h_ref")
    ok   = len(vals) == n
    out.append(Check(
        f"{label} / h_ref coverage",
        PASS if ok else FAIL,
        f"{len(vals)}/{n} valid"
        + (f" (mean={sum(vals)/len(vals):.4f})" if vals else ""),
        fix if not ok else None,
    ))
    agg   = d.get("aggregate", {})
    ci_ok = all(
        k in agg and agg[k] == agg[k]
        for k in ("h_ref_mean", "h_ref_ci_low", "h_ref_ci_high")
    )
    out.append(Check(
        f"{label} / h_ref CI",
        PASS if ci_ok else FAIL,
        (f"{agg.get('h_ref_mean', float('nan')):.4f} "
         f"[{agg.get('h_ref_ci_low', float('nan')):.4f}, "
         f"{agg.get('h_ref_ci_high', float('nan')):.4f}]")
        if ci_ok else "missing or NaN",
        fix if not ci_ok else None,
    ))
    return out


def _check_split(path: str, fix: str) -> List[Check]:
    out: List[Check] = []
    d = _load(path)
    if d is None:
        out.append(Check("Split-partial verifier / file", FAIL,
                         f"missing: {path}", fix))
        return out
    out.append(Check("Split-partial verifier / file", PASS, "ok"))
    per    = [ex for ex in d.get("per_example", []) if ex.get("verifier")]
    has_we = any("n_weak_entailment"       in ex["verifier"] for ex in per)
    has_cn = any("n_citation_noncompliance" in ex["verifier"] for ex in per)
    if has_we and has_cn:
        nwe = sum(ex["verifier"].get("n_weak_entailment",       0) for ex in per)
        ncn = sum(ex["verifier"].get("n_citation_noncompliance", 0) for ex in per)
        out.append(Check("Split-partial verifier / split labels", PASS,
                         f"WE={nwe}, CN={ncn}"))
    else:
        out.append(Check("Split-partial verifier / split labels", FAIL,
                         "split fields absent — standard prompt used", fix))
    return out


def _check_pairs(r: str, h: str) -> List[Check]:
    out: List[Check] = []
    P1 = "python rerun_all.py --phases 1"
    P3 = "python rerun_all.py --phases 3"
    anchor  = _load(f"{r}/qa_8b_rag_cbert.json")
    norag_d = _load(f"{r}/qa_8b_norag.json")
    lb_d    = _load(f"{r}/qa_1b_rag_cbert.json")
    href_a  = _load(f"{h}/qa_8b_rag_cbert_href.json")
    href_n  = _load(f"{h}/qa_8b_norag_href.json")
    for label, a, b, metric, fx in [
        ("Table 2 row 1 — 8B-RAG vs 1B-RAG / BS-F1",
         anchor, lb_d,    "bertscore_f1",       P1),
        ("Table 2 row 2 — 8B-RAG vs 1B-RAG / h_ctx",
         anchor, lb_d,    "hallucination_rate",  P1),
        ("Table 2 row 3 — 8B-RAG vs NoRAG / BS-F1",
         anchor, norag_d, "bertscore_f1",        P1),
    ]:
        if not a or not b:
            out.append(Check(label, FAIL, "source file missing", fx))
            continue
        va, vb = _per_ex(a, metric), _per_ex(b, metric)
        ok = len(va) == len(vb) == 30
        out.append(Check(
            label,
            PASS if ok else FAIL,
            "30 pairs available" if ok else f"lengths: {len(va)} vs {len(vb)}",
            fx if not ok else None,
        ))
    if href_a and href_n:
        va, vb = _per_ex(href_a, "h_ref"), _per_ex(href_n, "h_ref")
        ok = len(va) == len(vb) == 30
        out.append(Check(
            "Table 2 row 4 — 8B-RAG vs NoRAG / h_ref",
            PASS if ok else FAIL,
            "30 pairs — row 4 computable" if ok else f"{len(va)} vs {len(vb)}",
            None if ok else P3,
        ))
    else:
        out.append(Check("Table 2 row 4 — 8B-RAG vs NoRAG / h_ref",
                         FAIL, "href files missing", P3))
    if anchor and href_a:
        nq = len(anchor.get("per_example", []))
        nh = len(href_a.get("per_example", []))
        out.append(Check(
            "Consistency — QA vs h_ref length",
            PASS if nq == nh else FAIL,
            f"both {nq}" if nq == nh else f"qa={nq}, href={nh}",
            None if nq == nh else "Re-run Phase 3 after Phase 1",
        ))
    return out


# ---------------------------------------------------------------------------
# Reviewer revision check helpers (Exp1 / Exp2 / Exp3)
# ---------------------------------------------------------------------------

def _check_exp2_untrunc(r: str) -> List[Check]:
    """Rev.R1 — Untruncated BERTScore (exp2_untrunc_bertscore.py)."""
    out: List[Check] = []
    FIX = "python exp2_untrunc_bertscore.py"

    summary_path = f"{r}/bertscore_untrunc_summary.json"
    perq_path    = f"{r}/bertscore_untrunc_perquery.json"

    s = _load(summary_path)
    if s is None:
        out.append(Check("Rev.R1 / bertscore_untrunc_summary.json",
                         FAIL, f"missing: {summary_path}", FIX))
        return out
    out.append(Check("Rev.R1 / bertscore_untrunc_summary.json", PASS, "ok"))

    pq = _load(perq_path)
    out.append(Check(
        "Rev.R1 / bertscore_untrunc_perquery.json",
        PASS if pq else FAIL,
        "ok" if pq else f"missing: {perq_path}",
        FIX if not pq else None,
    ))

    # Check all expected conditions are present
    EXPECTED = [
        "8B Q4 + RAG (ClinicalBERT)",
        "8B Q4 + RAG (MedCPT)",
        "8B Q4 + RAG (MiniLM)",
        "8B Q4 (No RAG)",
        "8B Q4 + RAG + NLI Check",
    ]
    conds = s.get("conditions", {})
    missing_conds = [c for c in EXPECTED if c not in conds]
    out.append(Check(
        "Rev.R1 / conditions coverage",
        FAIL if missing_conds else PASS,
        f"missing: {missing_conds}" if missing_conds
        else f"{len(conds)}/5 conditions present",
        FIX if missing_conds else None,
    ))

    # Check each condition has the key fields
    for label, v in conds.items():
        has_keys = all(
            k in v for k in
            ("bs_f1_trunc_mean", "bs_f1_untrunc_mean",
             "delta_untrunc_minus_trunc_mean", "pct_gen_over_150")
        )
        out.append(Check(
            f"Rev.R1 / {label[:35]} fields",
            PASS if has_keys else FAIL,
            (f"trunc={v.get('bs_f1_trunc_mean',float('nan')):.4f}  "
             f"untrunc={v.get('bs_f1_untrunc_mean',float('nan')):.4f}  "
             f"Δ={v.get('delta_untrunc_minus_trunc_mean',float('nan')):+.4f}")
            if has_keys else "key fields missing",
            FIX if not has_keys else None,
        ))

    # Check significance block
    sig = s.get("significance", {}).get("8B_RAG_vs_NoRAG_untrunc", {})
    has_sig = all(k in sig for k in ("delta", "ci_95", "p", "significant"))
    if has_sig:
        d_v = sig["delta"]
        lo, hi = sig["ci_95"]
        p_v = sig["p"]
        out.append(Check(
            "Rev.R1 / RAG-vs-NoRAG significance",
            PASS,
            f"Δ={d_v:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
            f"p={p_v:.4f}  significant={sig['significant']}",
        ))
    else:
        out.append(Check("Rev.R1 / RAG-vs-NoRAG significance", WARN,
                         "significance block absent or incomplete", FIX))

    return out


def _check_exp1_constrained(r: str) -> List[Check]:
    """Rev.R2 — Constrained decoding (exp1_constrained_decoding.py)."""
    out: List[Check] = []
    FIX_NORAG = "python exp1_constrained_decoding.py --model llama3.1 --no-rag"
    FIX_RAG   = ("python exp1_constrained_decoding.py --model llama3.1 "
                 "--retriever cbert --index-dir cache/index.clinicalbert")
    FIX_DUAL  = FIX_NORAG

    # 1. Dual comparison table (primary output)
    dual = _load(f"{r}/kw_constrained_dual_table.json")
    if dual is None:
        out.append(Check("Rev.R2 / kw_constrained_dual_table.json",
                         FAIL,
                         f"missing: {r}/kw_constrained_dual_table.json",
                         FIX_DUAL))
        return out
    out.append(Check("Rev.R2 / kw_constrained_dual_table.json", PASS, "ok"))

    con = dual.get("constrained_aggregate", {})
    pc  = con.get("per_class", {})
    mu  = con.get("micro_f1")

    has_mu = mu is not None
    out.append(Check(
        "Rev.R2 / constrained micro_F1",
        PASS if has_mu else FAIL,
        f"{mu:.4f}" if has_mu else "missing",
        FIX_DUAL if not has_mu else None,
    ))

    for cls in ("Symptoms", "Diagnostics", "Pathogens"):
        v = pc.get(cls, {}).get("f1")
        out.append(Check(
            f"Rev.R2 / constrained {cls}-F1",
            PASS if v is not None else FAIL,
            f"{v:.4f}" if v is not None else "missing",
            FIX_DUAL if v is None else None,
        ))

    nat = dual.get("native_aggregate", {})
    nat_mu = (nat.get("micro_f1_mean")
              or nat.get("micro", {}).get("f1")
              or nat.get("micro_f1"))
    if nat_mu is not None and mu is not None:
        delta = mu - nat_mu
        out.append(Check(
            "Rev.R2 / delta_micro_F1",
            PASS,
            f"constrained {mu:.4f} − native {nat_mu:.4f} = {delta:+.4f}",
        ))

    # 2. No-RAG constrained per-example result file
    norag_path = f"{r}/kw_constrained_llama31_norag.json"
    norag_d    = _load(norag_path)
    out.append(Check(
        "Rev.R2 / kw_constrained_llama31_norag.json",
        PASS if norag_d else FAIL,
        f"ok (n={norag_d.get('n_prompts',0)})" if norag_d
        else f"missing: {norag_path}",
        FIX_NORAG if not norag_d else None,
    ))

    # 3. RAG constrained per-example result file
    rag_paths = [
        f"{r}/kw_constrained_llama31_rag_cbert.json",
        f"{r}/kw_constrained_llama31_8b-instruct-q4_K_M_rag_cbert.json",
    ]
    rag_d = next((_load(p) for p in rag_paths if _load(p) is not None), None)
    out.append(Check(
        "Rev.R2 / kw_constrained_*_rag_cbert.json",
        PASS if rag_d else WARN,
        f"ok (n={rag_d.get('n_prompts',0)})" if rag_d
        else "not found — No-RAG only was run",
        FIX_RAG if not rag_d else None,
    ))

    return out


def _check_exp3_adversarial(r: str) -> List[Check]:
    """Rev.R3 — Adversarial fail-safe (exp3_adversarial_failsafe.py)."""
    out: List[Check] = []
    INDEX = "cache/index.clinicalbert"
    FIX_BC = (f"python exp3_adversarial_failsafe.py "
              f"--index-dir {INDEX} --tracks B,C")
    FIX_A  = (f"python exp3_adversarial_failsafe.py "
              f"--index-dir {INDEX} --tracks A")
    FIX_ALL = (f"python exp3_adversarial_failsafe.py "
               f"--index-dir {INDEX} --tracks A,B,C")

    summary = _load(f"{r}/adversarial_failsafe_summary.json")
    results = _load(f"{r}/adversarial_failsafe_results.json")

    if summary is None:
        out.append(Check("Rev.R3 / adversarial_failsafe_summary.json",
                         FAIL,
                         f"missing: {r}/adversarial_failsafe_summary.json",
                         FIX_ALL))
        return out
    out.append(Check("Rev.R3 / adversarial_failsafe_summary.json", PASS, "ok"))

    out.append(Check(
        "Rev.R3 / adversarial_failsafe_results.json",
        PASS if results else WARN,
        "ok" if results else "full per-query results missing (summary sufficient for tables)",
        FIX_ALL if not results else None,
    ))

    td = summary.get("summary", {})

    # Track-level checks
    for track, fix_cmd, required in [
        ("A", FIX_A,  False),   # Track A optional (not yet run)
        ("B", FIX_BC, True),
        ("C", FIX_BC, True),
    ]:
        if track not in td:
            status = FAIL if required else WARN
            msg    = "not run"
            out.append(Check(f"Rev.R3 / Track {track}", status, msg,
                             fix_cmd if required else FIX_A if track == "A" else None))
            continue

        v = td[track]
        n      = v.get("n_queries",   0)
        trig   = v.get("n_triggered", 0)
        recov  = v.get("n_recovered", 0)
        tr_rt  = v.get("trigger_rate",  0.0)
        rc_rt  = v.get("recovery_rate") or 0.0
        pre_u  = v.get("pre_regen_label_totals",  {}).get("unsupported", 0)
        post_u = v.get("post_regen_label_totals", {}).get("unsupported", 0)

        out.append(Check(
            f"Rev.R3 / Track {track} trigger rate",
            PASS if trig > 0 else WARN,
            f"{trig}/{n} triggered ({tr_rt*100:.0f}%)",
            FIX_A if (trig == 0 and track == "A") else None,
        ))
        out.append(Check(
            f"Rev.R3 / Track {track} recovery rate",
            PASS,
            f"{recov}/{trig} recovered ({rc_rt*100:.0f}%)"
            + (" — single-pass insufficient; motivates multi-pass"
               if rc_rt < 0.5 and trig > 0 else ""),
        ))
        out.append(Check(
            f"Rev.R3 / Track {track} n_u reduction",
            PASS if (pre_u - post_u) >= 0 else WARN,
            f"pre={pre_u}  post={post_u}  net={pre_u - post_u:+d}",
        ))

    # Overall summary line
    if td:
        total_n    = sum(v.get("n_queries",   0) for v in td.values())
        total_trig = sum(v.get("n_triggered", 0) for v in td.values())
        total_rec  = sum(v.get("n_recovered", 0) for v in td.values())
        out.append(Check(
            "Rev.R3 / overall trigger/recovery",
            PASS,
            f"{total_trig}/{total_n} triggered  "
            f"{total_rec}/{total_trig} recovered across all run tracks",
        ))

    return out


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all(results_dir: str, href_dir: str) -> List[Check]:
    r, h = results_dir, href_dir
    P1 = "python rerun_all.py --phases 1"
    P2 = "python rerun_all.py --phases 2"
    P3 = "python rerun_all.py --phases 3"
    P4 = "python rerun_all.py --phases 4 --api-key $ANTHROPIC_API_KEY"
    P5 = "python rerun_all.py --phases 5"

    out: List[Check] = []

    # ── Original phases ──────────────────────────────────────────────────────
    out.append(_div("PHASE 1 — Local QA runs (6 configurations)"))
    for label, path in [
        ("8B Q4 + RAG (ClinicalBERT)",    f"{r}/qa_8b_rag_cbert.json"),
        ("8B Q4 + RAG (MedCPT)",          f"{r}/qa_8b_rag_medcpt.json"),
        ("8B Q4 + RAG (MiniLM)",          f"{r}/qa_8b_rag_minilm.json"),
        ("1B Q4 + RAG (ClinicalBERT)",    f"{r}/qa_1b_rag_cbert.json"),
        ("8B Q4 (No RAG)",                f"{r}/qa_8b_norag.json"),
        ("8B Q4 + RAG + NLI Check (std)", f"{r}/qa_8b_rag_cbert_ver.json"),
    ]:
        out.extend(_check_qa(label, path, P1))

    out.append(_div("PHASE 2 — Keyword evaluation (5 configurations)"))
    for label, path in [
        ("KW 8B RAG ClinicalBERT", f"{r}/kw_8b_rag_clinicalbert.json"),
        ("KW 8B RAG MedCPT",       f"{r}/kw_8b_rag_medcpt.json"),
        ("KW 8B RAG MiniLM",       f"{r}/kw_8b_rag_minilm.json"),
        ("KW 1B RAG ClinicalBERT", f"{r}/kw_1b_rag_clinicalbert.json"),
        ("KW 8B No RAG",           f"{r}/kw_8b_norag.json"),
    ]:
        out.extend(_check_kw(label, path, P2))

    out.append(_div("PHASE 3 — h_ref scoring (6 href files + Table 2 stats)"))
    for label, path in [
        ("h_ref 8B RAG ClinicalBERT", f"{h}/qa_8b_rag_cbert_href.json"),
        ("h_ref 8B NoRAG",            f"{h}/qa_8b_norag_href.json"),
        ("h_ref 8B RAG MedCPT",       f"{h}/qa_8b_rag_medcpt_href.json"),
        ("h_ref 8B RAG MiniLM",       f"{h}/qa_8b_rag_minilm_href.json"),
        ("h_ref 1B RAG ClinicalBERT", f"{h}/qa_1b_rag_cbert_href.json"),
        ("h_ref 8B + NLI Check",      f"{h}/qa_8b_rag_cbert_ver_href.json"),
    ]:
        out.extend(_check_href(label, path, P3))
    out.extend(_check_pairs(r, h))

    out.append(_div("PHASE 4 — Claude Sonnet API baseline"))
    for label, path in [
        ("Claude Sonnet QA RAG",   f"{r}/qa_claude_rag_cbert.json"),
        ("Claude Sonnet QA NoRAG", f"{r}/qa_claude_norag.json"),
    ]:
        out.extend(_check_qa(label, path, P4))
    for label, path in [
        ("Claude Sonnet KW RAG",   f"{r}/kw_claude_rag_clinicalbert.json"),
        ("Claude Sonnet KW NoRAG", f"{r}/kw_claude_norag.json"),
    ]:
        out.extend(_check_kw(label, path, P4))

    out.append(_div("PHASE 5 — Split-partial verifier"))
    out.extend(_check_split(f"{r}/qa_8b_rag_cbert_ver_split.json", P5))

    # ── Reviewer revision experiments ────────────────────────────────────────
    out.append(_div("REV.R1 — Untruncated BERTScore (exp2_untrunc_bertscore.py)"))
    out.extend(_check_exp2_untrunc(r))

    out.append(_div("REV.R2 — Constrained Decoding (exp1_constrained_decoding.py)"))
    out.extend(_check_exp1_constrained(r))

    out.append(_div("REV.R3 — Adversarial Fail-Safe (exp3_adversarial_failsafe.py)"))
    out.extend(_check_exp3_adversarial(r))

    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SYM = {PASS: "✓", WARN: "⚠", FAIL: "✗"}
CLR = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
RST = "\033[0m"


def _c(t: str, s: str, color: bool) -> str:
    return f"{CLR[s]}{t}{RST}" if color else t


def print_report(checks: List[Check], color: bool) -> None:
    for c in checks:
        if c.name == DIVIDER:
            print(f"\n  ── {c.message}")
            continue
        print(f"  {_c(SYM.get(c.status, '?'), c.status, color)}  {c.name}")
        print(f"       {c.message}")
        if c.fix:
            dim = "\033[2m" if color else ""
            print(f"       {dim}FIX: {c.fix}{RST if color else ''}")


def print_summary(checks: List[Check], strict: bool, color: bool) -> int:
    real = [c for c in checks if c.name != DIVIDER]
    np_  = sum(1 for c in real if c.status == PASS)
    nw   = sum(1 for c in real if c.status == WARN)
    nf   = sum(1 for c in real if c.status == FAIL)
    print("\n" + "=" * 60)
    print(
        f"  {_c(f'{np_} passed', PASS, color)}"
        f"  {_c(f'{nw} warnings', WARN, color)}"
        f"  {_c(f'{nf} failed', FAIL, color)}"
    )
    print("=" * 60)
    if nf == 0 and nw == 0:
        print("  ✓  All checks passed — ready for camera-ready submission.")
        return 0
    if nf == 0:
        print("  ⚠  Warnings present — review before submission.")
        return 1 if strict else 0
    print("  ✗  Failures present — paper tables are incomplete.")
    print("     Run the FIX commands above, then re-run preflight.py.")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="preflight",
        description="Submission-readiness validator for the P.R.O.T.E.U.S. paper.",
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument("--href-dir",    default="results/href")
    p.add_argument("--strict",    action="store_true",
                   help="Non-zero exit on warnings too.")
    p.add_argument("--no-color",  action="store_true",
                   help="Plain output for CI logs.")
    args = p.parse_args(argv)
    color = not args.no_color and sys.stdout.isatty()

    print("P.R.O.T.E.U.S. preflight check")
    print(f"results-dir : {args.results_dir}")
    print(f"href-dir    : {args.href_dir}\n")

    checks = run_all(args.results_dir, args.href_dir)
    print_report(checks, color)
    return print_summary(checks, args.strict, color)


if __name__ == "__main__":
    sys.exit(main())