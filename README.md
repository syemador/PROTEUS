# PROTEUS

**Privacy-Preserving On-Premise RAG for Biomedical Text Understanding**

Reference implementation and experimental record for the ICTIIA 2026 paper.
PROTEUS is an egress-free retrieval-augmented generation pipeline for
biomedical text: retrieval, generation, and grounding verification all run on
local hardware, so no query, context, or output leaves the local environment at
inference.

---

## Paper

> S. S. Antar and S. S. Ador, "PROTEUS: Privacy-Preserving On-Premise RAG for
> Biomedical Text Understanding," in *Proc. 3rd Int. Conf. on Technology
> Innovation and Its Applications (ICTIIA)*, Surabaya, Indonesia, 2026.

```bibtex
@inproceedings{antar2026proteus,
  title     = {{PROTEUS}: Privacy-Preserving On-Premise {RAG} for Biomedical Text Understanding},
  author    = {Antar, Siam Shibly and Ador, Syem Shibly},
  booktitle = {Proc. 3rd Int. Conf. on Technology Innovation and Its Applications (ICTIIA)},
  year      = {2026}
}
```

---

## What this is

A four-stage pipeline, of which stages 2–4 run on every query:

1. **Corpus ingestion and chunking** (one-time) — streams CORD-19 from a Parquet
   mirror, chunks at 256 tokens with 32-token overlap, deduplicates with MinHash.
2. **Hybrid retrieval** — dense encoder plus a zero-dependency BM25 index, fused
   by Reciprocal Rank Fusion, reranked by a cross-encoder, assembled outside-in
   under the Lost-in-the-Middle heuristic.
3. **Local generation** — quantized Llama served by Ollama on loopback.
4. **Inline NLI verification** — decomposes each answer into atomic claims and
   scores them against the retrieved context.

The contribution is the **egress-free integration**, not the novelty of any
single component, together with a reproducible account of how local 4-bit models
fail: *Verbosity Drift*, *Format Collapse*, and *reference-metric failure*.

---

## Repository layout

```
src/                  core modules
  data_processing.py    streaming ingest, chunking, MinHash dedup
  retrieval.py          dense + BM25 hybrid, RRF, rerank, LITM assembly
  llm_client.py         Ollama client (loopback only)
  orchestrator.py       end-to-end query path
  evaluation.py         ROUGE, BERTScore, h_ctx, h_ref

scripts/
  main.py               build index and run a task
  rerun_all.py          reproduce every reported condition
  compute_href.py       reference-grounded probe
  generate_figures.py   all figures (TrueType output)
  verify.py             standalone verifier
  dry_run.py            smoke test without a GPU
  test_index_cache.py   index cache integrity

experiments/
  exp1_constrained_decoding.py    JSON-Schema grammar enforcement
  exp2_untrunc_bertscore.py       untruncated BERTScore recompute
  exp3_adversarial_failsafe.py    poisoned-context and parametric-conflict probe

tables/                 LaTeX table builders
queries/                qa.jsonl (30), keywords.jsonl (25), adversarial_qa.jsonl (30)
annotations/            blinded.csv, master.csv  (expert recheck, N=50)
results/                per-condition result records
reviewer_tables/        generated LaTeX tables and paragraphs
figures/                published figures
cache_manifests/        index build parameters
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) bound to `127.0.0.1`
- One CUDA GPU with 12 GB for the 8B pipeline; the 70B generator was served
  from a separate four-card node
- `pip install -r requirements.txt`

Pull the generators:

```bash
ollama pull llama3.1     # 8B Q4
ollama pull llama3.3     # 70B NF4, optional
```

---

## Reproducing

```bash
# 1. environment and results check
python preflight.py

# 2. build the index (one-time; downloads the public corpus)
python scripts/main.py --build-index --retriever clinicalbert --max-documents 50000

# 3. run a single condition
python scripts/main.py --task qa --retriever clinicalbert --verifier

# 4. reproduce every reported condition
python scripts/rerun_all.py

# 5. regenerate figures and tables
python scripts/generate_figures.py --figures 2 3 4 5 6 7 8 --out figures/
python tables/build_tables.py
```

The regeneration loop is **off by default**. It is armed explicitly, and only
the adversarial protocol uses it:

```bash
python experiments/exp3_adversarial_failsafe.py --regenerate-on-unsupported
```

---

## Notes on the artifact

**Index tensors are not distributed.** `cache/` holds roughly 200 MB of
embeddings and pickled chunks per retriever. `cache_manifests/` records the
exact build parameters instead — dataset id, encoder, embedding dimension,
document cap, chunk count, and shuffle seed — so the index can be rebuilt
deterministically.

**Two distinct verifier instruments.** The labeling pass assigns each atomic
claim `supported` / `partial` / `unsupported` and gates regeneration. The NLI
entailment scorer computes `h_ctx` by max-pooling over retained chunks. They are
not expected to agree numerically: entailment is the stricter test, which is why
the reported false-negative rate is high and `h_ctx` is read only as a relative
ordering.

**Length statistics are whitespace words.** `generate_figures.py` and the
150-word truncation budget in `evaluation.py` count whitespace-delimited words,
not model tokens. Chunking parameters and throughput are in model tokens.

**Only body text is chunked.** The `pritamdeka/cord-19-fulltext` mirror exposes
a single `fulltext` column; title and abstract fields are empty in this mirror.

**The dotted project name persists in the code.** Source docstrings and the
generator's system prompt still read `P.R.O.T.E.U.S.`, the original project
name. The system prompt is left unchanged deliberately: it is part of the
experimental record, and editing it would mean the released code no longer
reproduces the reported numbers.

See `PROVENANCE.md` for how this release was assembled and which files came from
which export.

---

## Limitations

CORD-19 is public literature, not PHI; no clinical notes were evaluated. The
egress-free property is established by data-path inspection rather than a
packet-level audit, and the 70B configuration crosses a local network boundary
to a second node. Evaluation sets are small (30 QA questions, 25 keyword
prompts, 20 adversarial items), the 8B–70B comparison is confounded by model
generation and quantization as well as size, and pipeline components are not
ablated individually.

---

## License

Code released under the MIT License. CORD-19 is redistributed under its own
terms; this repository ships chunk identifiers and manifests rather than corpus
text.

---

## Contact

Siam Shibly Antar — siam.antar@mail.mcgill.ca
Syem Shibly Ador — syemshibly.ador@students.mq.edu.au
