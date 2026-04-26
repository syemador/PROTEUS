# P.R.O.T.E.U.S.: Privacy-Preserving Biomedical RAG

**Official implementation** of the paper:  
*P.R.O.T.E.U.S.: Privacy-preserving Retrieval and Orchestrator for Text Extraction and Understanding Systems*

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)
![OS](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange.svg)

---

## 🔬 Overview

**P.R.O.T.E.U.S.** is a fully local, privacy-preserving Retrieval-Augmented Generation (RAG) framework for clinical biomedical text, evaluated on CORD-19. All model inference runs on-premise — no query text, retrieved context, or patient data ever leaves the institutional network. The system supports two tasks: **Clinical QA** (BERTScore, ROUGE, hallucination rate) and **Medical Keyword Extraction** (structured `{Symptoms, Diagnostics, Pathogens}` entity classification), with an NLI-based consistency verifier that detects and suppresses hallucinated claims at inference time.

The framework benchmarks 4-bit quantized local models (**Llama 3.1 8B Q4** on a single RTX 3060, **Llama 3.3 70B NF4** sharded across a 4×RTX 6000 JupyterHub node via a local FastAPI proxy that exposes an Ollama-compatible endpoint) across three retriever families, a hybrid BM25+dense search engine with cross-encoder reranking, and a bounded single-pass NLI regeneration loop (the *Dormant Fail-Safe*). Three reviewer-mandated empirical extensions — untruncated BERTScore re-evaluation, constrained structured decoding, and adversarial fail-safe stress-testing — are implemented as standalone experiment scripts.

Core methodological pillars:

* **Zero-egress inference:** corpus streaming is the only network operation; query and context traffic never leave the institutional boundary
* **Hybrid retrieval:** BM25 + dense encoding (Bio\_ClinicalBERT / MedCPT / MiniLM) fused via Reciprocal Rank Fusion (RRF) and reranked by a cross-encoder
* **Lost-in-the-Middle context assembly:** retrieved chunks are reordered to place highest-ranked evidence at context boundaries, mitigating LLM edge-bias degradation
* **NLI-based hallucination verification:** three-model cross-encoder ensemble scores atomic claims against retrieved context; unsupported claims trigger bounded single-pass regeneration
* **Constrained structured decoding:** Ollama JSON-Schema grammar enforcement at the sampler level bypasses Format Collapse for keyword extraction

---

## ✨ Key Features

* **Streaming CORD-19 Corpus Ingestion with On-Disk Caching**
  Streams `pritamdeka/cord-19-fulltext` from HuggingFace Datasets as a Parquet mirror — no local corpus download required. Sentence-level chunking, SHA-256 deduplication, and Bio\_ClinicalBERT tokenizer-aware splitting produce a 48,114-chunk index cached to `cache/index.clinicalbert/` on the first run and reused thereafter. The privacy-preserving data pipeline is implemented in `src/data_processing.py`, with the full encoder and vector store in `src/retrieval.py`.

* **Hybrid BM25 + Dense Retrieval with Cross-Encoder Reranking**
  A zero-dependency BM25 sparse index is combined with dense Bio\_ClinicalBERT / MedCPT / MiniLM encoders via RRF, then refined by a `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder over a candidate pool of 50 passages. Semantic deduplication (cosine threshold 0.85) ensures retrieved chunks are maximally diverse. Top-*k* chunks (default *k* = 5) are assembled using a lost-in-the-middle-aware interleaving strategy. Core implementation: `src/retrieval.py`.

* **Split-Partial NLI Verifier with Dormant Fail-Safe**
  Each generated response is atomically decomposed into individual claims and verified against retrieved context by a three-model NLI ensemble. The **standard verifier** uses a 3-label scheme (supported / partial / unsupported); the **split-partial verifier** refines `partial` into `weak_entailment` (content grounded but citation absent) and `citation_noncompliance` (citation present but formatting incorrect), resolving a label-conflation confound identified in RQ4. The regeneration loop is then stress-tested via two adversarial sub-tracks (B: poisoned context, C: parametric conflict) in `experiments/exp3_adversarial_failsafe.py`. Core pipeline: `src/orchestrator.py`.

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
│   │                                #   MODEL_REGISTRY (8B local + 70B proxy)
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
│   └── adversarial_qa.jsonl         #   20 adversarial queries (Sub-tracks B / C)
│
├── output/                          # All generated artefacts
│   ├── results/                     #   Evaluation result JSONs
│   │   ├── qa_8b_rag_cbert.json
│   │   ├── qa_8b_rag_medcpt.json
│   │   ├── qa_8b_rag_minilm.json
│   │   ├── qa_8b_rag_cbert_ver.json
│   │   ├── qa_8b_rag_cbert_ver_split.json
│   │   ├── qa_8b_norag.json
│   │   ├── qa_70b_rag_cbert.json        ← Llama 3.3 70B NF4 + RAG (4×RTX 6000)
│   │   ├── qa_70b_norag.json            ← Llama 3.3 70B NF4 No-RAG (4×RTX 6000)
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
│       ├── embeddings.pt            #   Bio_ClinicalBERT dense vectors
│       └── manifest.json            #   Index metadata + version hash
│
├── preflight.py                     # Submission-readiness validator (8 phases)
└── requirements.txt                 # Pip dependency snapshot
```

---

## ✅ Setup

### System Environment

The H.E.R.A. testbed is validated on the following configuration:

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 LTS (or WSL2 Ubuntu 22.04) |
| Python | 3.10 |
| CUDA | 12.1 |
| PyTorch | 2.3.1 (cu121 wheel) |
| Ollama | ≥ 0.1.24 (required for JSON-Schema `format` enforcement) |

### Prerequisites

Install PyTorch with CUDA 12.1 support **before** the requirements file:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install [Ollama](https://ollama.com) and pull the 8B local model:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

For the 70B local upper-bound baseline (requires the multi-GPU JupyterHub inference node — see *§ Llama 3.3 70B FastAPI Proxy* below):

```bash
ollama pull llama3.3:70b-instruct-q4_K_M
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

### Llama 3.3 70B FastAPI Proxy (optional, Phase 4 only)

Phase 4 of `scripts/rerun_all.py` evaluates Llama 3.3 70B (NF4) as the local upper-bound baseline. The 70B model is sharded across **4×NVIDIA RTX 6000 (24 GB GDDR6 each)** using `device_map="auto"` with an 18 GB per-card `max_memory` cap to preserve KV-cache headroom. A small FastAPI proxy fronts the sharded model and exposes a standard Ollama-compatible `/api/generate` endpoint, so the rest of the pipeline remains unmodified — point `--ollama-host` at the proxy URL:

```bash
export PROTEUS_70B_HOST=http://multi-gpu-host:8080
```

The proxy runs entirely within the institutional network. No external API calls are made; egress-free operation is preserved end-to-end. Phases 1–3, 5 and all reviewer experiments run on the single 8B node and do not require the 70B proxy.

### External Model Weights

All encoder and cross-encoder weights are downloaded automatically from HuggingFace Hub on first use:

| Model | Role | HF ID |
|---|---|---|
| Bio\_ClinicalBERT | Dense retriever (primary) | `emilyalsentzer/Bio_ClinicalBERT` |
| MedCPT | Dense retriever (alternate) | `ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder` |
| MiniLM-L6 | Dense retriever (lightweight) | `sentence-transformers/all-MiniLM-L6-v2` |
| MiniLM cross-encoder | Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| SciBERT | BERTScore computation | `allenai/scibert_scivocab_uncased` |
| NLI ensemble (×3) | Atomic-fact entailment | `cross-encoder/nli-deberta-v3-base`, `nli-deberta-v3-large`, `nli-roberta-base` |

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

Executes Phases 1–3 and 5 in sequence on the single-GPU 8B node:
- **Phase 1** — 5 QA configurations (8B Q4 × 3 retrievers + No-RAG + NLI-Check)
- **Phase 2** — 4 keyword extraction configurations (8B Q4 × 3 retrievers + No-RAG)
- **Phase 3** — h_ref post-hoc scoring via `scripts/compute_href.py`
- **Phase 5** — split-partial verifier re-run on the anchor configuration

### 3) Run the 70B local upper-bound baseline (Phase 4)

Requires the 4×RTX 6000 JupyterHub node and FastAPI proxy (see *§ Llama 3.3 70B FastAPI Proxy*):

```bash
python scripts/rerun_all.py --phases 41 42 43 44 \
    --ollama-host $PROTEUS_70B_HOST
```

Or run all five phases at once (proxy must be reachable):

```bash
python scripts/rerun_all.py --ollama-host $PROTEUS_70B_HOST
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

# Rev.R3 — Adversarial Dormant Fail-Safe stress-test (Sub-tracks B + C, N=20)
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

Figures that depend on missing result files (e.g., 70B results without the proxy) are rendered with a `PENDING` watermark to preserve layout.

---

## 🔧 CLI Reference — `scripts/main.py`

```
--task              qa | keywords                      [required]
--model             llama3.1 | llama3.3                [default: llama3.1]
                                                       (llama3.1 = 8B Q4 local;
                                                        llama3.3 = 70B NF4 via FastAPI proxy)
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
--ollama-host       Ollama server URL                  [default: http://localhost:11434]
                                                       (set to FastAPI proxy URL for 70B)
```

### Example invocations

```bash
# Primary anchor — 8B Q4 + hybrid RAG (Bio_ClinicalBERT) + NLI verifier
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

# Llama 3.3 70B (NF4) + RAG baseline — sharded across 4×RTX 6000 via FastAPI proxy
python scripts/main.py --task qa --model llama3.3 \
    --ollama-host $PROTEUS_70B_HOST \
    --retriever clinicalbert --verifier off \
    --queries queries/qa.jsonl \
    --output output/results/qa_70b_rag_cbert.json

# Llama 3.3 70B (NF4) No-RAG baseline — for the Scale-Dependent Grounding Policy finding
python scripts/main.py --task qa --model llama3.3 \
    --ollama-host $PROTEUS_70B_HOST \
    --rag off --verifier off \
    --queries queries/qa.jsonl \
    --output output/results/qa_70b_norag.json
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

### H.E.R.A. Testbed — Hardware Profile

The harness runs across two physically distinct, network-isolated nodes within the institutional security perimeter.

**Node A — 8B pipeline (single-GPU edge profile):**
- NVIDIA RTX 3060, 12 GB GDDR6
- 64 GB DDR4 RAM
- Ubuntu 22.04 LTS, CUDA 12.1
- Hosts: 8B generator, retrieval encoders, cross-encoder reranker, NLI ensemble

**Node B — 70B upper-bound baseline (multi-GPU JupyterHub node):**
- 4× NVIDIA RTX 6000, 24 GB GDDR6 each (96 GB aggregate)
- Sharded via `device_map="auto"`, 18 GB per-card `max_memory` cap
- FastAPI proxy exposing Ollama-compatible `/api/generate` endpoint

### Throughput

| Configuration | Hardware | Throughput |
|---|---|---|
| Llama 3.1 8B Q4 + RAG (Bio\_ClinicalBERT) | RTX 3060 | 63.43 ± 1.42 tok/s |
| Llama 3.1 8B Q4 + RAG (MedCPT) | RTX 3060 | 60.54 ± 1.24 tok/s |
| Llama 3.1 8B Q4 + RAG (MiniLM) | RTX 3060 | 61.50 ± 1.52 tok/s |
| Llama 3.1 8B Q4, No-RAG | RTX 3060 | 63.72 ± 1.53 tok/s |
| Llama 3.1 8B Q4 + RAG + NLI Check | RTX 3060 | 63.10 ± 1.26 tok/s |
| Llama 3.3 70B NF4 + RAG (Bio\_ClinicalBERT) | 4× RTX 6000 | 4.15 ± 0.26 tok/s |
| Llama 3.3 70B NF4, No-RAG† | 4× RTX 6000 | 2.33 ± 0.41 tok/s |

†The 70B No-RAG condition uniformly returns `"Insufficient evidence."` (≈3 tokens) for all queries — the *Scale-Dependent Grounding Policy* finding. The low tok/s reflects the very short decode, not a hardware limit.

### VRAM Budget — 12 GB Edge Envelope (Node A)

| Component | Footprint |
|---|---|
| Llama 3.1 8B Q4 (weights) | ~4.7 GB |
| KV-cache @ 8192 ctx | ~1.0–1.5 GB |
| Bio\_ClinicalBERT encoder | ~0.5 GB |
| Cross-encoder reranker | ~0.25 GB |
| 3-model NLI ensemble | ~1.5 GB |
| **Total resident** | **~8.45 GB** |
| **Free headroom (12 GB card)** | **~3.5 GB** |

The four-card 70B configuration on Node B retains ~24 GB aggregate headroom across the inference node under the 18 GB per-card cap.

---

## 📜 Citation

```bibtex
@article{antar2026proteus,
  title     = {{P.R.O.T.E.U.S.}: Privacy-preserving Retrieval and Orchestrator
               for Text Extraction and Understanding Systems},
  author    = {Antar, Siam Shibly and Ador, Syem Shibly and
               Ding, Steven H. H. and Fung, Benjamin C. M.},
  journal   = {Under Review (IEEE TKDE)},
  year      = {2026}
}
```

---

## 📄 License

MIT License (see `LICENSE`).
