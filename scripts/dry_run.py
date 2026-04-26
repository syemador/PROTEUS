import json
from pathlib import Path
from evaluation import QaEvaluator
from orchestrator import ProteusOrchestrator
from llm_client import OllamaClient
from retrieval import build_encoder, InMemoryVectorStore

def verify_stack():
    # 1. Initialize Components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_encoder("clinicalbert", device=device)
    store = InMemoryVectorStore(encoder=encoder)
    
    # 2. Mock a Retrieval Chunk
    from data_processing import DocumentChunk
    mock_chunk = DocumentChunk(
        chunk_id="test_0", text="Clinical studies show Llama 3 is highly efficient.",
        cord_uid="test_uid", section="abstract", token_count=10
    )
    store.add_chunks([mock_chunk])
    
    # 3. Setup Orchestrator
    llm = OllamaClient()
    orch = ProteusOrchestrator(vector_store=store, llm_client=llm, model_alias="llama3.1")
    
    # 4. Execute Pipeline
    query = "How efficient is Llama 3?"
    reference = "Llama 3 is highly efficient according to clinical studies."
    result = orch.run(query, task="qa")
    
    # 5. Run Evaluator (Tests Patch 1 & 2)
    evaluator = QaEvaluator(device=device)
    scores = evaluator.compute(
        generated=result.text, 
        reference=reference, 
        retrieved_texts=[rc.chunk.text for rc in result.retrieved]
    )
    
    # 6. Verify Table Metadata (Tests Patch 3)
    aggregate = evaluator.aggregate([scores])
    print(f"BERTScore: {aggregate['bertscore_f1_mean']:.4f}")
    print(f"Hallucination Rate: {aggregate['hallucination_rate_mean']:.4f}")

if __name__ == "__main__":
    verify_stack()