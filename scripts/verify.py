"""
verify.py
=========

Standalone P.R.O.T.E.U.S. stack verification — streaming edition.

Exercises every layer of the pipeline without requiring CORD-19 on disk
or a network connection (uses synthetic in-memory chunks). Run after
any environment change to confirm the stack is healthy.

    python verify.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"  OK    {name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name}: {exc}")
        return False


def main() -> int:
    print("\n=== P.R.O.T.E.U.S. Stack Verification (streaming edition) ===\n")
    ok = True

    # ----------------------------------------------------------------
    # 1. CUDA
    # ----------------------------------------------------------------
    ok &= check(
        "CUDA available",
        lambda: (
            torch.cuda.is_available()
            or (_ for _ in ()).throw(RuntimeError("no GPU"))
        ),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"        -> {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU only'}"
    )

    # ----------------------------------------------------------------
    # 2. `datasets` library importable
    # ----------------------------------------------------------------
    ok &= check(
        "Hugging Face `datasets` library importable",
        lambda: __import__("datasets"),
    )

    # ----------------------------------------------------------------
    # 3. ClinicalBERT loads and embeds into VRAM
    # ----------------------------------------------------------------
    def _cbert():
        from retrieval import ClinicalBertEncoder

        enc = ClinicalBertEncoder(device=device)
        v = enc.encode_query("acute respiratory distress syndrome")
        assert v.shape == (768,), f"Unexpected shape: {v.shape}"

    ok &= check("ClinicalBERT loads and embeds", _cbert)

    # ----------------------------------------------------------------
    # 4. Ollama REST API responding
    # ----------------------------------------------------------------
    def _ping():
        from llm_client import OllamaClient

        assert OllamaClient().ping(), "Ollama daemon not reachable"

    ok &= check("Ollama REST API responding", _ping)

    # ----------------------------------------------------------------
    # 5. Llama 3.1 8B Q4 generates
    # ----------------------------------------------------------------
    def _gen_8b():
        from llm_client import OllamaClient, resolve_model

        c = OllamaClient()
        assert "llama3.1:8b-instruct-q4_K_M" in c.list_models(), "8B Q4 not pulled"
        r = c.generate(
            model=resolve_model("llama3.1"),
            user_prompt="Reply with the single word: OK",
            temperature=0.0,
            max_tokens=16,
        )
        assert r.text.strip() and r.eval_tokens > 0

    ok &= check("Llama 3.1 8B Q4 generates via Ollama", _gen_8b)

    # ----------------------------------------------------------------
    # 6. Llama 3.2 1B Q4 generates
    # ----------------------------------------------------------------
    def _gen_1b():
        from llm_client import OllamaClient, resolve_model

        c = OllamaClient()
        assert "llama3.2:1b-instruct-q4_K_M" in c.list_models(), "1B Q4 not pulled"
        r = c.generate(
            model=resolve_model("llama3.2"),
            user_prompt="Reply with the single word: OK",
            temperature=0.0,
            max_tokens=16,
        )
        assert r.text.strip()

    ok &= check("Llama 3.2 1B Q4 generates via Ollama", _gen_1b)

    # ----------------------------------------------------------------
    # 7. QaEvaluator end-to-end (ROUGE + BERTScore + NLI)
    # ----------------------------------------------------------------
    def _eval():
        from evaluation import QaEvaluator

        ev = QaEvaluator(device=device)
        s = ev.compute(
            generated="Fever is a common COVID-19 symptom.",
            reference="COVID-19 commonly presents with fever.",
            retrieved_context="COVID-19 patients frequently experience fever.",
        )
        assert 0.0 <= s.bertscore_f1 <= 1.0
        assert 0.0 <= s.hallucination_rate <= 1.0

    ok &= check("QaEvaluator end-to-end (ROUGE + BERTScore + NLI)", _eval)

    # ----------------------------------------------------------------
    # 8. Index persistence round-trip (save_index -> load_index)
    # ----------------------------------------------------------------
    def _round_trip():
        from data_processing import DocumentChunk
        from retrieval import ClinicalBertEncoder, InMemoryVectorStore

        enc = ClinicalBertEncoder(device=device)
        store = InMemoryVectorStore(encoder=enc)
        chunks = [
            DocumentChunk(
                chunk_id="rt::0",
                cord_uid="rt",
                section="title",
                text="COVID-19 patients present with fever, cough, and fatigue.",
                token_count=12,
            ),
            DocumentChunk(
                chunk_id="rt::1",
                cord_uid="rt",
                section="abstract",
                text="SARS-CoV-2 enters cells via the ACE2 receptor.",
                token_count=11,
            ),
        ]
        store.add_chunks(chunks)

        # Save to a temp directory
        tmp = tempfile.mkdtemp(prefix="proteus_verify_")
        try:
            store.save_index(
                tmp,
                manifest_extra={"test": True, "dataset_id": "synthetic"},
            )

            # Verify files exist
            for name in ("manifest.json", "embeddings.pt", "chunks.pkl"):
                assert (Path(tmp) / name).exists(), f"{name} missing from cache"

            # Verify manifest content
            with open(Path(tmp) / "manifest.json") as fh:
                manifest = json.load(fh)
            assert manifest["n_chunks"] == 2
            assert manifest["test"] is True

            # Reload into a fresh store
            loaded = InMemoryVectorStore.load_index(tmp, enc)
            assert len(loaded) == 2, f"Expected 2 chunks, got {len(loaded)}"

            # Verify search works on the reloaded store
            results = loaded.search("What are COVID-19 symptoms?", top_k=1)
            assert len(results) == 1
            assert results[0].chunk.chunk_id == "rt::0"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    ok &= check(
        "Index persistence round-trip (save -> reload -> search)", _round_trip
    )

    # ----------------------------------------------------------------
    # 9. Full orchestrator pipeline (retrieve -> generate -> verify)
    # ----------------------------------------------------------------
    def _orch():
        from data_processing import DocumentChunk
        from llm_client import OllamaClient
        from orchestrator import ProteusOrchestrator, VerifierAgent
        from retrieval import ClinicalBertEncoder, InMemoryVectorStore

        enc = ClinicalBertEncoder(device=device)
        store = InMemoryVectorStore(encoder=enc)
        store.add_chunks(
            [
                DocumentChunk(
                    chunk_id="orch::0",
                    cord_uid="orch",
                    section="title",
                    text="COVID-19 patients commonly present with fever, cough, and fatigue.",
                    token_count=12,
                ),
            ]
        )
        llm = OllamaClient()
        orch = ProteusOrchestrator(
            vector_store=store,
            llm_client=llm,
            model_alias="llama3.1",
            rag_enabled=True,
            verifier=VerifierAgent(llm_client=llm, model_alias="llama3.1"),
        )
        result = orch.run("What are common COVID-19 symptoms?", task="qa")
        assert result.text.strip(), "Empty generation"
        assert len(result.retrieved) > 0, "No chunks retrieved"
        assert result.verifier_verdict is not None, "Verifier did not run"

    ok &= check("Full orchestrator pipeline (retrieve -> generate -> verify)", _orch)

    # ----------------------------------------------------------------
    # 10. HFStreamingLoader import (no network call — just confirms class exists)
    # ----------------------------------------------------------------
    def _loader_import():
        from data_processing import HFStreamingLoader, DEFAULT_HF_DATASET

        loader = HFStreamingLoader(
            dataset_id=DEFAULT_HF_DATASET, max_documents=0
        )
        assert loader.dataset_id == "allenai/cord19"

    ok &= check("HFStreamingLoader importable and configurable", _loader_import)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print()
    if ok:
        print("  All checks passed. Stack is healthy.")
    else:
        print("  One or more checks failed. Review errors above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
