"""Strategy A: Naive Dense Top-k Vector Retrieval."""
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, DEFAULT_RETRIEVAL_K
from src.ingestion.indexer import COLLECTION_NAME
from .base import BaseRetriever, RetrievedChunk

class DenseRetriever(BaseRetriever):
    """Dense vector retriever querying ChromaDB with cosine similarity."""
    
    def __init__(self, persist_dir: str = str(CHROMA_PERSIST_DIR)):
        self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[RetrievedChunk]:
        """Query ChromaDB and return top-k chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved: List[RetrievedChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved
            
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for rank, (c_id, text, meta, dist) in enumerate(zip(ids, docs, metas, distances), 1):
            # Cosine distance to similarity: similarity = 1 - distance
            similarity = 1.0 - float(dist)
            retrieved.append(RetrievedChunk(
                chunk_id=c_id,
                paper_id=meta.get("paper_id", "unknown"),
                paper_title=meta.get("paper_title", "Unknown Title"),
                page=int(meta.get("page", 1)),
                section=meta.get("section", "General"),
                text=text,
                score=similarity,
                retrieval_strategy="naive_dense",
                rank=rank
            ))
            
        return retrieved
