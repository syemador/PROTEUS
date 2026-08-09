"""
test_index_cache.py
===================

Fast, self-contained test for the ``save_index`` / ``load_index``
round-trip. Does NOT require a GPU, Ollama, or network access — it
uses synthetic chunks and a tiny MiniLM encoder on CPU.

Run from the project root:
    python test_index_cache.py

Exit code 0 = pass, 1 = fail.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    print("\n=== Index Cache Round-Trip Test ===\n")

    # We need torch and the retrieval module. If they're not available,
    # fail fast with a helpful message rather than a traceback.
    try:
        import torch
        from data_processing import DocumentChunk
        from retrieval import (
            INDEX_CACHE_VERSION,
            InMemoryVectorStore,
            MiniLmEncoder,
        )
    except ImportError as exc:
        print(f"FAIL: missing dependency — {exc}")
        print("Run: pip install torch transformers sentence-transformers")
        return 1

    device = "cpu"  # deliberately CPU-only so this runs anywhere
    print(f"  Using device: {device}")

    # 1. Build a tiny store
    enc = MiniLmEncoder(device=device)
    store = InMemoryVectorStore(encoder=enc)

    chunks = [
        DocumentChunk(
            chunk_id=f"test::{i}",
            cord_uid="test",
            section="abstract",
            text=text,
            token_count=10,
            metadata={"title": "Test Document"},
        )
        for i, text in enumerate(
            [
                "COVID-19 patients commonly present with fever and dry cough.",
                "SARS-CoV-2 enters host cells through the ACE2 receptor.",
                "Dexamethasone reduces mortality in hospitalized patients.",
                "Long COVID involves persistent fatigue and cognitive impairment.",
                "mRNA vaccines demonstrated high efficacy against severe disease.",
            ]
        )
    ]
    store.add_chunks(chunks)
    print(f"  Built store with {len(store)} chunks")

    # 2. Save to a temporary directory
    tmp = tempfile.mkdtemp(prefix="proteus_cache_test_")
    try:
        store.save_index(
            tmp,
            manifest_extra={
                "dataset_id": "synthetic",
                "max_documents": 5,
                "shuffle_seed": 42,
            },
        )
        print(f"  Saved index to {tmp}")

        # Verify files
        for name in ("manifest.json", "embeddings.pt", "chunks.pkl"):
            p = Path(tmp) / name
            assert p.exists(), f"Missing: {p}"
            assert p.stat().st_size > 0, f"Empty: {p}"
        print("  Verified: manifest.json, embeddings.pt, chunks.pkl all present")

        # Verify manifest
        with open(Path(tmp) / "manifest.json") as fh:
            manifest = json.load(fh)
        assert manifest["version"] == INDEX_CACHE_VERSION, (
            f"Version mismatch: {manifest['version']} != {INDEX_CACHE_VERSION}"
        )
        assert manifest["n_chunks"] == 5
        assert manifest["dataset_id"] == "synthetic"
        assert manifest["embed_dim"] == 384  # MiniLM output dim
        print(f"  Manifest valid: version={manifest['version']}, "
              f"n_chunks={manifest['n_chunks']}, dim={manifest['embed_dim']}")

        # 3. cache_exists should return True
        assert InMemoryVectorStore.cache_exists(tmp), "cache_exists returned False"
        print("  cache_exists() returns True ✓")

        # 4. Load into a fresh store
        loaded = InMemoryVectorStore.load_index(tmp, enc)
        assert len(loaded) == 5, f"Expected 5 chunks, got {len(loaded)}"
        print(f"  Loaded store: {len(loaded)} chunks")

        # 5. Search works on the reloaded store
        results = loaded.search("What are COVID-19 symptoms?", top_k=2)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        # The fever/cough chunk should rank highest
        assert "fever" in results[0].chunk.text.lower(), (
            f"Top result doesn't mention fever: {results[0].chunk.text!r}"
        )
        print(f"  Search on loaded index: top result mentions fever ✓")
        print(f"  Top-1 score: {results[0].score:.4f}")

        # 6. Encoder mismatch rejection
        from retrieval import ClinicalBertEncoder

        try:
            # This should raise because we saved with MiniLM but load with ClinicalBERT
            cbert = ClinicalBertEncoder(device=device)
            InMemoryVectorStore.load_index(tmp, cbert, strict_encoder_match=True)
            print("  FAIL: encoder mismatch was NOT rejected")
            return 1
        except RuntimeError as exc:
            assert "Encoder mismatch" in str(exc)
            print(f"  Encoder mismatch correctly rejected ✓")

        # 7. Version mismatch rejection
        # Temporarily corrupt the version in the manifest
        manifest["version"] = -1
        with open(Path(tmp) / "manifest.json", "w") as fh:
            json.dump(manifest, fh)
        try:
            InMemoryVectorStore.load_index(tmp, enc)
            print("  FAIL: version mismatch was NOT rejected")
            return 1
        except RuntimeError as exc:
            assert "version mismatch" in str(exc).lower()
            print(f"  Version mismatch correctly rejected ✓")

        # 8. Incomplete cache rejection
        (Path(tmp) / "embeddings.pt").unlink()
        assert not InMemoryVectorStore.cache_exists(tmp), (
            "cache_exists should return False after deleting embeddings.pt"
        )
        print("  Incomplete cache correctly detected by cache_exists() ✓")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n  All tests passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
