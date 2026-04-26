# P.R.O.T.E.U.S.: Privacy-Preserving Biomedical RAG

**Official implementation** of the paper:  
*P.R.O.T.E.U.S.: Privacy-Preserving Retrieval and Orchestrator for Text Extraction and Understanding Systems — A Local Biomedical RAG Harness for CORD-19 Clinical QA and Keyword Extraction*

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)

---

## 🔬 Overview

**P.R.O.T.E.U.S.** is a fully local, privacy-preserving Retrieval-Augmented Generation (RAG) framework for clinical biomedical text, evaluated on CORD-19. All model inference runs on-device via Ollama — no query text, retrieved context, or patient data ever leaves the machine. The system supports two tasks: **Clinical QA** (BERTScore, ROUGE, hallucination rate) and **Medical Keyword Extraction** (structured `{Symptoms, Diagnostics, Pathogens}` entity classification), with an NLI-based consistency verifier that detects and suppresses hallucinated claims at inference time.

The framework benchmarks 4-bit quantized local models (Llama 3.1 8B, Llama 3.2 1B) against a Claude Sonnet API reference baseline across three retriever families, a hybrid BM25+dense search engine with cross-encoder reranking, and a bounded single-pass NLI regeneration loop (the *Dormant Fail-Safe*). Three reviewer-mandated empirical extensions — untruncated BERTScore re-evaluation, constrained structured decoding, and adversarial fail-safe stress-testing — are implemented as standalone experiment scripts.

Core methodological pillars:

* **Zero-egress inference:** corpus streaming is the only network operation; query and context traffic never leave the machine
* **Hybrid retrieval:** BM25 + dense encoding (ClinicalBERT / MedCPT / MiniLM) fused via Reciprocal Rank Fusion (RRF) and reranked by a cross-encoder
* **Lost-in-the-Middle context assembly:** retrieved chunks are reordered to place highest-ranked evidence at context boundaries, mitigating LLM edge-bias degradation
* **NLI-based hallucination verification:** three-model cross-encoder ensemble scores atomic claims against retrieved context; unsupported claims trigger bounded single-pass regeneration
* **Constrained structured decoding:** Ollama JSON-Schema grammar enforcement at the sampler level bypasses Format Collapse for keyword extraction

---

## ✨ Key Features

* **Streaming CORD-19 Corpus Ingestion with On-Disk Caching**
  Streams `pritamdeka/cord-19-fulltext` from HuggingFace Datasets as a Parquet mirror — no local corpus download required. Sentence-level chunking, SHA-256 deduplication, and ClinicalBERT tokenizer-aware splitting produce a 48,114-chunk index cached to `cache/index.clinicalbert/` on the first run and reused thereafter. The privacy-preserving data pipeline is implemented in `src/data_processing.py`, with the full encoder and vector store in `src/retrieval.py`.

* **Hybrid BM25 + Dense Retrieval with Cross-Encoder Reranking**
  A zero-dependency BM25 sparse index is combined with dense ClinicalBERT / MedCPT / MiniLM encoders via RRF, then refined by a `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder over a candidate pool of 50 passages. Semantic deduplication (cosine threshold 0.85) ensures retrieved chunks are maximally diverse. Top-*k* chunks (default *k* = 5) are assembled using a lost-in-the-middle-aware interleaving strategy. Core implementation: `src/retrieval.py`.

* **Split-Partial NLI Verifier with Dormant Fail-Safe**
  Each generated response is atomically decomposed into individual claims and verified against retrieved context by a three-model NLI ensemble. The **standard verifier** uses a 3-label scheme (supported / partial / unsupported); the **split-partial verifier** refines `partial` into `weak_entailment` (content grounded but citation absent) and `citation_noncompliance` (citation present but formatting incorrect), resolving a label-conflation confound identified in RQ4. Queries where the regeneration loop fires are stress-tested via three adversarial sub-tracks in `experiments/exp3_adversarial_failsafe.py`. Core pipeline: `src/orchestrator.py`.

* **Constrained Structured Decoding for Entity Extraction**
  `OllamaClient.generate()` in `src/llm_client.py` accepts an optional `format=` JSON-Schema parameter (requires Ollama ≥ 0.1.24) that constrains the sampler to produce only valid `{Symptoms, Diagnostics, Pathogens}` instances, bypassing the Format Collapse failure mode observed under 4-bit quantization. A dual evaluation separating compliance gaps from knowledge gaps is implemented in `experiments/exp1_constrained_decoding.py`, with findings reported in `output/results/kw_constrained_dual_table.json`.

---

## 📂 Repository Structure

```
proteus/
├── src/                             # Core library
│   ├── orchestrator.py              #   ProteusOrchestrator + VerifierAgent
│   │                                #   (standard 3-label + split-partial 4-label)
│   ├── retrieval.py                 #   BM25Index, InMemoryVectorStore,
│   │                                #   build_encoder, RRF, cross-encoder reranker
│   ├── evaluation.py                #   QaEvaluator, KeywordEvaluator,
│   │                                #   AtomicFactExtractor, NliEnsemble,
│   │                                #   paired bootstrap + permutation stats
│   ├── llm_client.py                #   OllamaClient (+ format= patch),
│   │                                #   AnthropicClient, MODEL_REGISTRY
│   └── data_processing.py           #   CORD-19 streaming chunker,
│                                    #   SHA-256 deduplication, cache management
│
├── scripts/                         # Runnable entry points
│   ├── main.py                      #   Primary CLI — all task/model/retriever combos
│   ├── rerun_all.py                 #   5-phase master orchestration script
│   ├── compute_href.py              #   Post-hoc h_ref scoring on saved QA JSONs
│   ├── generate_figures.py          #   Paper figures 2–8 (degrades gracefully)
│   ├── dry_run.py                   #   Smoke-test pipeline (no corpus required)
│   └── verify.py                    #   End-to-end stack verification (synthetic chunks)
│
├── experiments/                     # Reviewer-mandated empirical extensions
│   ├── exp1_constrained_decoding.py #   Rev.R2: constrained JSON-Schema decoding
│   ├── exp2_untrunc_bertscore.py    #   Rev.R1: untruncated BERTScore re-evaluation
│   └── exp3_adversarial_failsafe.py #   Rev.R3: adversarial Dormant Fail-Safe stress-test
│
├── tables/                          # LaTeX table generators
│   ├── build_tables.py              #   10-table generator (Tables 1–5 + Supp + Rev.R1–R3)
│   └── build_reviewer_tables.py     #   Standalone reviewer paragraph + table inserts
│
├── queries/                         # Held-out evaluation sets (not used for training)
│   ├── qa.jsonl                     #   30 high-precision clinical QA prompts
│   ├── keywords.jsonl               #   25 medical keyword extraction prompts
│   └── adversarial_qa.jsonl         #   30 adversarial queries (Tracks A / B / C)
│
├── output/                          # All generated artefacts
│   ├── results/                     #   Evaluation result JSONs
│   │   ├── qa_8b_rag_cbert.json
│   │   ├── qa_8b_rag_medcpt.json
│   │   ├── qa_8b_rag_minilm.json
│   │   ├── qa_8b_rag_cbert_ver.json
│   │   ├── qa_8b_rag_cbert_ver_split.json
│   │   ├── qa_8b_norag.json
│   │   ├── qa_70b_rag_cbert.json        ← Claude Sonnet + RAG baseline
│   │   ├── qa_70b_norag.json            ← Claude Sonnet No-RAG baseline
│   │   ├── kw_*.json                    ← Keyword extraction results
│   │   ├── kw_constrained_*.json        ← Constrained decoding (Rev.R2)
│   │   ├── bertscore_untrunc_*.json     ← Untruncated BERTScore (Rev.R1)
│   │   ├── adversarial_failsafe_*.json  ← Adversarial fail-safe (Rev.R3)
│   │   └── href/                        ← h_ref enriched QA results
│   ├── reviewer_tables/             #   Pre-generated .tex paragraph inserts
│   │   ├── para_{untrunc,constrained,adversarial}.tex
│   │   └── tab_{bertscore_untrunc,kw_constrained,adversarial_failsafe}.tex
│   └── tables/
│       └── tables_final.tex         #   All 10 LaTeX tables (camera-ready)
│
├── cache/                           # Vector index cache (auto-created on first run)
│   └── index.clinicalbert/
│       ├── chunks.pkl               #   48,114 DocumentChunk objects
│       ├── embeddings.pt            #   ClinicalBERT dense vectors
│       └── manifest.json            #   Index metadata + version hash
│
├── preflight.py                     # Submission-readiness validator (8 phases)
└── requirements.txt                 # Pip dependency snapshot
```

---

## ✅ Setup

### Prerequisites

Install PyTorch with CUDA 12.1 support **before** the requirements file:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install [Ollama](https://ollama.com) and pull the required local models:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull llama3.2:1b-instruct-q4_K_M
```

> **Constrained decoding** (`experiments/exp1_constrained_decoding.py`) requires **Ollama ≥ 0.1.24** for JSON-Schema `format` enforcement. Check with `ollama --version`.

### Option A: Pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda create -n proteus python=3.10
conda activate proteus
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 📦 Data & External Dependencies

### CORD-19 Corpus

No manual download is required. The pipeline streams `pritamdeka/cord-19-fulltext` — a community Parquet mirror of CORD-19 (~369k papers) — directly from the HuggingFace Hub at index-build time. The original `allenai/cord19` dataset uses a custom loader that is incompatible with HuggingFace streaming; the Parquet mirror resolves this transparently (see `src/data_processing.py` for the full explanation).

The encoded index is cached on the first run to `cache/index.clinicalbert/` (and equivalent sibling directories for MedCPT and MiniLM). **Subsequent runs are fully offline.** Only public research paper text is fetched — no user queries or retrieved context ever leave the machine.

### Claude Sonnet API Baseline (optional, Phase 4 only)

Phase 4 of `scripts/rerun_all.py` uses the Anthropic API for the large-model reference baseline. Set your API key before running Phase 4:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

All other phases (1–3, 5, and all reviewer experiments) run fully locally with no API calls.

### External Model Weights

All encoder and cross-encoder weights are downloaded automatically from HuggingFace Hub on first use:

| Model | Role | HF ID |
|---|---|---|
| Bio\_ClinicalBERT | Dense retriever (primary) | `emilyalsentzer/Bio_ClinicalBERT` |
| MedCPT | Dense retriever (alternate) | `ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder` |
| MiniLM-L6 | Dense retriever (lightweight) | `sentence-transformers/all-MiniLM-L6-v2` |
| MiniLM cross-encoder | Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| SciBERT | BERTScore computation | `allenai/scibert_scivocab_uncased` |

---

## 🚀 Quick Start

### 1) Verify the stack (no corpus or Ollama required)

```bash
python scripts/verify.py
```

Runs end-to-end on synthetic in-memory chunks. All layers should report `OK`.

### 2) Run all local evaluation phases

```bash
python scripts/rerun_all.py --local-only
```

Executes Phases 1–3 and 5 in sequence:
- **Phase 1** — 6 QA configurations (8B Q4 × 3 retrievers + 1B + No-RAG + NLI-Check)
- **Phase 2** — 5 keyword extraction configurations
- **Phase 3** — h_ref post-hoc scoring via `scripts/compute_href.py`
- **Phase 5** — split-partial verifier re-run on the anchor configuration

### 3) Run the Claude Sonnet API baseline (Phase 4)

```bash
python scripts/rerun_all.py --phases 41 42 43 44 --api-key $ANTHROPIC_API_KEY
```

Or run all five phases at once:

```bash
python scripts/rerun_all.py --api-key $ANTHROPIC_API_KEY
```

### 4) Run reviewer revision experiments

```bash
# Rev.R1 — Untruncated BERTScore (zero new inference; uses saved results)
python experiments/exp2_untrunc_bertscore.py

# Rev.R2 — Constrained structured decoding for keyword extraction
python experiments/exp1_constrained_decoding.py --model llama3.1 --no-rag
python experiments/exp1_constrained_decoding.py \
    --model llama3.1 --retriever cbert \
    --index-dir cache/index.clinicalbert

# Rev.R3 — Adversarial Dormant Fail-Safe stress-test
python experiments/exp3_adversarial_failsafe.py \
    --index-dir cache/index.clinicalbert --tracks B,C
```

### 5) Validate submission readiness

```bash
python preflight.py --results-dir output/results --href-dir output/results/href
```

Checks all 8 phases (original + reviewer extensions) and prints per-check pass/warn/fail with exact `FIX:` commands.

### 6) Generate all LaTeX tables

```bash
python tables/build_tables.py --all \
    --results-dir output/results \
    --href-dir output/results/href \
    > output/tables/tables_final.tex
```

Individual tables (all degrade gracefully to `[PENDING]` if a result file is absent):

```bash
python tables/build_tables.py --table qa          # Table 1  — QA performance
python tables/build_tables.py --table pairs       # Table 2  — Pairwise significance
python tables/build_tables.py --table latency     # Table 3  — Throughput
python tables/build_tables.py --table genlength   # Table 4  — Generation lengths
python tables/build_tables.py --table keywords    # Table 5  — Keyword extraction
python tables/build_tables.py --table untrunc     # Rev.R1   — Untruncated BERTScore
python tables/build_tables.py --table constrained # Rev.R2   — Constrained decoding
python tables/build_tables.py --table adversarial # Rev.R3   — Adversarial fail-safe
```

### 7) Generate paper figures

```bash
python scripts/generate_figures.py                        # all figures (2–8)
python scripts/generate_figures.py --figures 2 4 6        # specific subset
python scripts/generate_figures.py --format pdf           # PDF output
```

Figures that depend on missing result files (e.g., 70B local results) are rendered with a `PENDING` watermark to preserve layout.

---

## 🔧 CLI Reference — `scripts/main.py`

```
--task              qa | keywords                      [required]
--model             llama3.1 | llama3.2 | claude-sonnet | claude-opus | claude-haiku
--rag               on | off                           [default: on]
--retriever         clinicalbert | medcpt | minilm     [default: clinicalbert]
--retrieval-mode    dense | hybrid                     [default: hybrid]
--reranker          none | cross-encoder               [default: cross-encoder]
--context-assembly  sequential | lost-in-the-middle    [default: lost-in-the-middle]
--verifier          on | off                           [default: on]
--verifier-mode     standard | split-partial           [default: standard]
--regenerate-on-unsupported                            [flag]
--queries           path to .jsonl query file          [required]
--output            path for result JSON               [required]
--cache-dir         embedding index cache dir          [default: cache/]
--rebuild-index                                        [flag]
--top-k             retrieved chunks per query         [default: 5]
--pool-k            reranker candidate pool size       [default: 50]
--temperature       generation temperature             [default: 0.2]
--max-tokens        max generation tokens              [default: 1024]
--backend           ollama | anthropic                 [default: ollama]
--api-key           Anthropic API key
--ollama-host       Ollama server URL                  [default: http://localhost:11434]
```

### Example invocations

```bash
# Primary anchor — 8B Q4 + hybrid RAG (ClinicalBERT) + NLI verifier
python scripts/main.py --task qa --model llama3.1 \
    --retriever clinicalbert --verifier on \
    --queries queries/qa.jsonl \
    --output output/results/qa_8b_rag_cbert_ver.json

# Parametric baseline — No RAG
python scripts/main.py --task qa --model llama3.1 \
    --rag off --verifier off \
    --queries queries/qa.jsonl \
    --output output/results/qa_8b_norag.json

# Split-partial verifier (4-label NLI)
python scripts/main.py --task qa --model llama3.1 \
    --retriever clinicalbert --verifier on \
    --verifier-mode split-partial \
    --queries queries/qa.jsonl \
    --output output/results/qa_8b_rag_cbert_ver_split.json

# Keyword extraction (deterministic; T=0)
python scripts/main.py --task keywords --model llama3.1 \
    --retriever clinicalbert --verifier off \
    --temperature 0.0 --max-tokens 512 \
    --queries queries/keywords.jsonl \
    --output output/results/kw_8b_rag_clinicalbert.json

# Claude Sonnet + RAG baseline (API)
python scripts/main.py --task qa --model claude-sonnet \
    --backend anthropic --api-key $ANTHROPIC_API_KEY \
    --retriever clinicalbert \
    --queries queries/qa.jsonl \
    --output output/results/qa_70b_rag_cbert.json
```

---

## 📊 Included Outputs

All generated artefacts are written to `output/`:

| Path | Contents |
|---|---|
| `output/results/qa_*.json` | Per-example QA scores: ROUGE-{1,2,L}, BERTScore-F₁, h_ctx, verifier verdicts, latency |
| `output/results/kw_*.json` | Per-example keyword P/R/F₁ by entity class (Symptoms, Diagnostics, Pathogens) |
| `output/results/href/*.json` | h_ref enriched QA results with atomic-fact decomposition |
| `output/results/kw_constrained_*.json` | Constrained decoding dual-evaluation (Rev.R2) |
| `output/results/bertscore_untrunc_summary.json` | Untruncated BS-F₁ per condition + significance (Rev.R1) |
| `output/results/adversarial_failsafe_summary.json` | Per-track trigger/recovery rates (Rev.R3) |
| `output/reviewer_tables/*.tex` | Ready-to-insert LaTeX paragraph and table sources |
| `output/tables/tables_final.tex` | All 10 paper tables (camera-ready) |

**Hardware notes (H.E.R.A. testbed — RTX 3060, 12 GB GDDR6, CUDA 12.1):**

| Configuration | Throughput |
|---|---|
| Llama 3.1 8B Q4 + RAG | ~63.4 tok/s |
| Llama 3.1 8B Q4, No-RAG | ~63.7 tok/s |
| Llama 3.2 1B Q4 + RAG | ~267 tok/s |

VRAM budget at top-*k* = 5: 8B Q4 (≈4.7 GB) + ClinicalBERT encoder (≈0.5 GB) + cross-encoder reranker (≈0.25 GB) + NLI ensemble (≈1.5 GB) ≈ 7.0 GB, leaving ~5 GB headroom on a 12 GB card.

---

## 📜 Citation

```bibtex
@article{antar2025proteus,
  title     = {{P.R.O.T.E.U.S.}: Privacy-Preserving Retrieval and Orchestrator
               for Text Extraction and Understanding Systems ---
               A Local Biomedical {RAG} Harness for {CORD-19}
               Clinical {QA} and Keyword Extraction},
<<<<<<< HEAD
  author    = {Antar, Siam Shibly and Ador, Syem Shibly and Fung, Benjamin C. M. and Ding, Steven},
=======
  author    = {Antar, Siam Shibly and Ador, Syem Shibly and Fung, Benjamin C. M. and Ding, Steven},
>>>>>>> parent of 7ae079e (minor fix)
  journal   = {Under Review},
  year      = {2025}
}
```

---

## 📄 License

MIT License (see `LICENSE`).
