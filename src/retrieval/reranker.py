"""Strategy C: Hybrid Retrieval + Cross-Encoder Reranker."""
from typing import List
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL_NAME,
    DEFAULT_RETRIEVAL_K,
    RERANK_CANDIDATE_POOL_K
)
from .base import BaseRetriever, RetrievedChunk
from .hybrid import HybridRetriever

class RerankedRetriever(BaseRetriever):
    """Retrieves candidates via Hybrid RRF, then scores them with a Cross-Encoder."""
    
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.hybrid_retriever = HybridRetriever()
        self.model_name = model_name
        self._reranker = None  # Lazy loading

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self.model_name)
        return self._reranker

    def retrieve(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[RetrievedChunk]:
        """Fetch candidates via Hybrid search and rerank using Cross-Encoder."""
        candidate_pool = self.hybrid_retriever.retrieve(
            query,
            k=max(k * 3, RERANK_CANDIDATE_POOL_K)
        )
        
        if not candidate_pool:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [[query, chunk.text] for chunk in candidate_pool]
        scores = self.reranker.predict(pairs)
        
        # Attach cross-encoder scores
        for chunk, score in zip(candidate_pool, scores):
            chunk.score = float(score)
            chunk.retrieval_strategy = "hybrid_reranked"
            
        # Sort descending by cross-encoder score
        reranked = sorted(candidate_pool, key=lambda c: c.score, reverse=True)[:k]
        
        for rank, chunk in enumerate(reranked, 1):
            chunk.rank = rank
            
        return reranked
