"""Manual sanity test for bare retrieval across manual test queries."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import RerankedRetriever

def safe_str(text: str) -> str:
    """Sanitize text for Windows console printing."""
    return text.encode("ascii", "ignore").decode("ascii")

def test_manual_queries():
    print("=" * 70)
    print(" MANUAL SANITY TEST: BARE RETRIEVAL PIPELINE")
    print("=" * 70)
    
    test_queries = [
        "How does DPO derive closed-form optimal policy from Bradley-Terry preference model?",
        "What is reward model overoptimization and Goodhart's law in RLHF?",
        "Constitutional AI harmlessness from AI feedback without human labels"
    ]
    
    dense_retriever = DenseRetriever()
    hybrid_retriever = HybridRetriever()
    reranked_retriever = RerankedRetriever()
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n[{idx}] Test Query: '{safe_str(query)}'\n")
        
        print("--- [1] Naive Dense Top-3 ---")
        dense_chunks = dense_retriever.retrieve(query, k=3)
        for c in dense_chunks:
            title = safe_str(c.paper_title[:45])
            snippet = safe_str(c.text[:120].replace('\n', ' '))
            print(f"  * [{c.chunk_id}] (Score: {c.score:.3f}, Page {c.page}) {title}")
            print(f"    Snippet: {snippet}...")
            
        print("\n--- [2] Hybrid (BM25 + Dense RRF) Top-3 ---")
        hybrid_chunks = hybrid_retriever.retrieve(query, k=3)
        for c in hybrid_chunks:
            title = safe_str(c.paper_title[:45])
            snippet = safe_str(c.text[:120].replace('\n', ' '))
            print(f"  * [{c.chunk_id}] (RRF Score: {c.score:.4f}, Page {c.page}) {title}")
            print(f"    Snippet: {snippet}...")
            
        print("\n--- [3] Hybrid + Cross-Encoder Reranked Top-3 ---")
        reranked_chunks = reranked_retriever.retrieve(query, k=3)
        for c in reranked_chunks:
            title = safe_str(c.paper_title[:45])
            snippet = safe_str(c.text[:120].replace('\n', ' '))
            print(f"  * [{c.chunk_id}] (Cross-Encoder Score: {c.score:.3f}, Page {c.page}) {title}")
            print(f"    Snippet: {snippet}...")
            
    print("\n" + "=" * 70)
    print(" BARE RETRIEVAL SANITY TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_manual_queries()
