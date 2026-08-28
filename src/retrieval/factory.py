"""Factory to dynamically instantiate and cache configurable retrieval strategies."""
from enum import Enum
from typing import Union, Dict
from .base import BaseRetriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .reranker import RerankedRetriever

class RetrievalStrategy(str, Enum):
    NAIVE_DENSE = "naive_dense"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"

_RETRIEVER_INSTANCES: Dict[str, BaseRetriever] = {}

def get_retriever(strategy: Union[str, RetrievalStrategy] = RetrievalStrategy.HYBRID_RERANK) -> BaseRetriever:
    """Instantiate, cache, and return the configured retrieval strategy."""
    strat_str = str(strategy).lower()
    
    if strat_str in _RETRIEVER_INSTANCES:
        return _RETRIEVER_INSTANCES[strat_str]
        
    if "naive" in strat_str or strat_str == "naive_dense":
        retriever = DenseRetriever()
    elif "hybrid_rrf" in strat_str or strat_str == "hybrid":
        retriever = HybridRetriever()
    elif "rerank" in strat_str or strat_str == "hybrid_rerank":
        retriever = RerankedRetriever()
    else:
        retriever = DenseRetriever()
        
    _RETRIEVER_INSTANCES[strat_str] = retriever
    return retriever
