"""Strategy B: Hybrid BM25 + Dense Retrieval with Reciprocal Rank Fusion (RRF)."""
import pickle
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from config import (
    CHROMA_PERSIST_DIR,
    BM25_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    DEFAULT_RETRIEVAL_K,
    RRF_K_CONSTANT
)
from src.ingestion.indexer import tokenize_for_bm25, get_chunk_by_id
from .base import BaseRetriever, RetrievedChunk
from .dense import DenseRetriever

class HybridRetriever(BaseRetriever):
    """Combines Dense similarity search and BM25 sparse search using RRF."""
    
    def __init__(self, persist_dir: str = str(CHROMA_PERSIST_DIR), bm25_path: str = str(BM25_INDEX_PATH)):
        self.dense_retriever = DenseRetriever(persist_dir=persist_dir)
        self.bm25 = None
        self.bm25_chunk_ids = []
        
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.bm25_chunk_ids = data["chunk_ids"]

    def retrieve(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[RetrievedChunk]:
        """Perform hybrid search combining BM25 and Dense with Reciprocal Rank Fusion."""
        candidate_pool_size = max(k * 3, 20)
        
        # 1. Dense retrieval
        dense_results = self.dense_retriever.retrieve(query, k=candidate_pool_size)
        
        # 2. BM25 retrieval
        bm25_ranks: Dict[str, int] = {}
        if self.bm25 and self.bm25_chunk_ids:
            tokenized_query = tokenize_for_bm25(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top candidate indices
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_pool_size]
            for rank_idx, idx in enumerate(top_indices, 1):
                c_id = self.bm25_chunk_ids[idx]
                if bm25_scores[idx] > 0:
                    bm25_ranks[c_id] = rank_idx

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_lookup: Dict[str, RetrievedChunk] = {}
        
        for rank, d_chunk in enumerate(dense_results, 1):
            c_id = d_chunk.chunk_id
            rrf_scores[c_id] += 1.0 / (RRF_K_CONSTANT + rank)
            chunk_lookup[c_id] = d_chunk
            
        for c_id, rank in bm25_ranks.items():
            rrf_scores[c_id] += 1.0 / (RRF_K_CONSTANT + rank)
            if c_id not in chunk_lookup:
                raw_chunk = get_chunk_by_id(c_id)
                if raw_chunk:
                    chunk_lookup[c_id] = RetrievedChunk(
                        chunk_id=c_id,
                        paper_id=raw_chunk["paper_id"],
                        paper_title=raw_chunk["paper_title"],
                        page=int(raw_chunk["page"]),
                        section=raw_chunk["section"],
                        text=raw_chunk["text"],
                        score=0.0,
                        retrieval_strategy="hybrid_rrf",
                        rank=rank
                    )

        # 4. Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:k]
        
        final_results: List[RetrievedChunk] = []
        for rank, cid in enumerate(sorted_chunk_ids, 1):
            if cid in chunk_lookup:
                chunk = chunk_lookup[cid]
                chunk.score = float(rrf_scores[cid])
                chunk.retrieval_strategy = "hybrid_rrf"
                chunk.rank = rank
                final_results.append(chunk)
                
        return final_results
