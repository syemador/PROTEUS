# PROTEUS — Release Provenance

Consolidated artifact assembled **2026-07-30** for the ICTIIA 2026 camera-ready
reproducibility statement. This file records where every file came from and why,
so the release is auditable.

---

## Source archives

Two overlapping exports from the RTX 3060 / WSL2 development host were merged.
Neither was complete on its own.

| Tag | Archive | Exported | Layout | Distinguishing content |
|---|---|---|---|---|
| **NEW** | `P_R_O_T_E_U_S-20260730T002721Z-1-001.zip` | 2026-04-26 | reorganized (`src/`, `scripts/`, `experiments/`), git-tracked | reviewer experiments, superset of results, `README.md`, `adversarial_qa.jsonl` |
| **OLD** | `Proteus_project-20260730T002634Z-1-001.zip` | 2026-04-18 | flat (`script/`) | **all 8 figures**, **expert annotation CSVs**, all 3 index manifests, `test_index_cache.py` |

Two earlier paper-source archives (`PROTEUS_main.zip` Jun 3, `PROTEUS_paper_src.zip`
Jun 15) contain LaTeX only, no code. `PROTEUS_main.zip` is fully superseded by the
Jun 15 archive and was discarded.

### Upstream repository

The NEW archive carries an intact `.git`:

**Repository:** https://github.com/syemador/PROTEUS

```
remote origin  https://github.com/syemador/PROTEUS.git
(renamed from P.R.O.T.E.U.S; the dotted form persists in source docstrings
 and in the generator system prompt, which is left unchanged because altering
 it would invalidate the reported results)
local main == origin/main  @ 49bd16a344613f82f88575a96191fb7c6ab69904
```

Local and remote heads matched at export, so the codebase was fully pushed.
Commits authored by `adorMQ <syemshiblyador@gmail.com>`.

**Visibility status:** public. The paper's availability statement cites the URL
above.

---

## Merge decisions

Of 13 code files present in both archives, **9 are byte-identical**. The four
conflicts were resolved individually rather than by preferring one archive.

| File | Winner | Reason |
|---|---|---|
| `build_tables.py` | NEW (+16,413 b) | superset; adds reviewer-table builders |
| `preflight.py` | NEW (+11,605 b) | superset; adds result-directory validation |
| `llm_client.py` | NEW (+998 b) | superset |
| `generate_figures.py` | **OLD (+11,349 b)** | **OLD produced the published figures.** `figs/fig4_keyword_bars.pdf` in OLD is byte-identical to the figure in the submitted paper; the `fig4` plotting function differs between versions (3,729 vs 2,513 chars). OLD is canonical for figure reproduction. |
| 9 others | identical | no decision needed |

Exclusive content, all retained:

- **NEW-only:** `experiments/exp1-3`, `tables/build_reviewer_tables.py`, `reviewer_tables/`, `queries/adversarial_qa.jsonl`, `results/nli_pairs_raw.jsonl`, 10 additional result files
- **OLD-only:** `annotations/blinded.csv`, `annotations/master.csv`, `scripts/test_index_cache.py`, all 8 figures, medcpt/minilm manifests

`results/`: 19 files identical, 0 divergent, 10 NEW-only, 0 OLD-only — NEW taken whole.

---

## Modification applied: figure font compliance

The published figure PDFs embedded **Type 3** fonts, a standard IEEE PDF eXpress
rejection cause. Neither `generate_figures.py` set `pdf.fonttype`.

One line was added to the canonical (OLD) generator:

```python
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType
matplotlib.rcParams['ps.fonttype']  = 42
```

All 8 figures were regenerated from the surviving result JSONs. Verification:

- Type 3 font count, all figures: **0** (was 3 in `fig4_keyword_bars.pdf`)
- `fig4` numeric fidelity confirmed identical to the published version — all 18
  plotted values match exactly:
  `0.012 0.016 0.028 0.030 0.043 0.045 0.045 0.049 0.054 0.080 0.083 0.089 0.116 0.120 0.260 0.367 0.387 0.467`

No plotted value changed. Only font embedding differs.

`figures/arch_diagram.pdf` is TikZ-generated (Type 1 CM fonts, already compliant)
and ships with its `.tex` source.

---

## Deliberate exclusions

| Excluded | Size | Rebuild |
|---|---|---|
| `cache/index.clinicalbert/embeddings.pt` | 141 MB | `python scripts/main.py` (index build stage) |
| `cache/index.clinicalbert/chunks.pkl` | 55 MB | same |
| medcpt / minilm index tensors | ~1.8 MB, incomplete | same, with `--retriever` switched |

Index **manifests** are retained in `cache_manifests/` because they record the
exact build parameters needed to reproduce the index:

```json
{ "dataset_id": "pritamdeka/cord-19-fulltext", "dataset_split": "train",
  "encoder_model_name": "emilyalsentzer/Bio_ClinicalBERT", "embed_dim": 768,
  "max_documents": 50000, "n_chunks": 48114, "shuffle_seed": 42,
  "dtype": "torch.float32", "version": 2 }
```

Git internals (`.git/`) were excluded; the upstream remote is recorded above.

---

## Values verified against this artifact

Checks run during consolidation, confirming paper claims against raw results:

| Claim | Source | Result |
|---|---|---|
| 94 atomic claims (Split-Partial denominator) | `results/qa_8b_rag_cbert_ver_split.json`, summed over 30 queries | **confirmed**: 60 supported / 11 weak-entailment / 6 citation-noncompliance / 17 unsupported = 94 |
| Split-Partial percentages 63.8 / 11.7 / 6.4 / 18.1 % | same | all four match to 2 dp |
| QA set n = 30 | `queries/qa.jsonl` | 30 lines |
| Keyword set n = 25 | `queries/keywords.jsonl` | 25 lines |
| Expert recheck N = 50 | `annotations/master.csv` | 50 rows incl. header |
| Index size 50,000 documents | `cache_manifests/` | `max_documents: 50000`, `n_chunks: 48114` |

Derived statistics computable from this artifact with no GPU (exact
Clopper-Pearson CIs, Holm-corrected p-values, post-hoc detectable effect size)
are reported separately in the camera-ready revision notes.

---

## Known gaps

1. **The LaTeX source of the submitted 6-page paper is not in any archive.** The
   Jun 15 paper archive is a 9-page US-Letter pre-cut draft that predates the
   final corrections. Only the compiled submission PDF survives. The camera-ready
   requires LaTeX source, so this must be reconstructed from the Jun 15 fragments
   plus the documented edit list.
2. **No further experiments are possible** — access to the RTX 3060 host and the
   4-GPU node was lost after 2026-04-26. This artifact is the complete
   experimental record.
3. `annotations/` contains a single annotator's scores. A second annotator can
   score `blinded.csv` independently to yield an inter-annotator agreement
   statistic without any compute.

---

## Layout

```
PROTEUS/
├── README.md               project documentation (NEW)
├── PROVENANCE.md           this file
├── requirements.txt        pinned dependencies
├── preflight.py            environment + results validation (NEW)
├── src/                    5 core modules: data_processing, retrieval,
│                           llm_client, orchestrator, evaluation
├── scripts/                main, rerun_all, compute_href, dry_run, verify,
│                           generate_figures (OLD + fonttype patch),
│                           test_index_cache
├── experiments/            exp1 constrained decoding, exp2 untruncated
│                           BERTScore, exp3 adversarial fail-safe
├── tables/                 build_tables, build_reviewer_tables
├── queries/                qa.jsonl (30), keywords.jsonl (25),
│                           adversarial_qa.jsonl (30)
├── annotations/            blinded.csv, master.csv (expert recheck, N=50)
├── results/                29 result JSONs + href/ + nli_pairs_raw.jsonl
├── reviewer_tables/        generated LaTeX tables and paragraphs
├── figures/                8 regenerated PDFs (TrueType) + arch_diagram + .tex
└── cache_manifests/        index build parameters (tensors excluded)
```
