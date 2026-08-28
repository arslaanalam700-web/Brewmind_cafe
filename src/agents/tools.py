"""ADK and Python tools for document search and chunk lookup."""
from typing import List, Dict, Any, Optional
from src.retrieval.factory import get_retriever, RetrievalStrategy
from src.ingestion.indexer import get_chunk_by_id

# Global active retrieval strategy (defaults to hybrid_rerank)
_ACTIVE_STRATEGY: str = RetrievalStrategy.HYBRID_RERANK.value

def set_active_retrieval_strategy(strategy: str):
    """Set the active retrieval strategy used by the retriever tool."""
    global _ACTIVE_STRATEGY
    _ACTIVE_STRATEGY = strategy

def get_active_retrieval_strategy() -> str:
    """Get currently active retrieval strategy."""
    return _ACTIVE_STRATEGY

def search_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the indexed paper corpus using the active retrieval strategy.
    
    Args:
        query: The search query or sub-question.
        k: Number of relevant chunks to retrieve.
        
    Returns:
        List of dictionaries with keys: chunk_id, paper_id, paper_title, page, section, text, score.
    """
    retriever = get_retriever(_ACTIVE_STRATEGY)
    results = retriever.retrieve(query=query, k=k)
    return [chunk.to_dict() for chunk in results]

def get_chunk(chunk_id: str) -> Dict[str, Any]:
    """
    Fetch the exact chunk text, paper title, page, and section by its chunk_id.
    
    Args:
        chunk_id: Unique chunk identifier (e.g., 'dpo:p04_c02').
        
    Returns:
        Dictionary containing the full chunk record or an error dictionary if not found.
    """
    chunk = get_chunk_by_id(chunk_id)
    if chunk:
        return chunk
    return {
        "chunk_id": chunk_id,
        "error": f"Chunk '{chunk_id}' not found in indexed corpus.",
        "text": ""
    }
