"""ChromaDB dense indexing and BM25 sparse index serialization for coffee shop menu."""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_PERSIST_DIR,
    BM25_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    EMBEDDING_MODEL_NAME
)

COLLECTION_NAME = "coffee_menu_chunks"

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text into lowercased alphanumeric words for BM25."""
    import re
    return re.findall(r'\b\w+\b', text.lower())

def build_indices(all_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build both Chroma vector store and BM25 sparse index from all extracted menu chunks.
    """
    if not all_chunks:
        raise ValueError("No chunks provided to index.")
        
    print(f"Building indices for {len(all_chunks)} menu chunks...")
    
    # 1. Setup ChromaDB persistent collection
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    # Use SentenceTransformers embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # Reset or get collection
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Prepare Chroma batch data
    ids = []
    documents = []
    metadatas = []
    
    # Prepare chunk lookup map and BM25 tokenized corpus
    chunks_lookup = {}
    bm25_corpus_tokens = []
    bm25_chunk_ids = []
    
    for chunk in all_chunks:
        c_id = chunk["chunk_id"]
        c_text = chunk["text"]
        
        ids.append(c_id)
        documents.append(c_text)
        metadatas.append({
            "chunk_id": c_id,
            "paper_id": chunk["paper_id"],
            "paper_title": chunk["paper_title"],
            "page": int(chunk["page"]),
            "section": chunk["section"]
        })
        
        chunks_lookup[c_id] = chunk
        bm25_corpus_tokens.append(tokenize_for_bm25(c_text))
        bm25_chunk_ids.append(c_id)
        
    # Chroma upsert in batches
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        
    print(f"  [OK] Indexed {len(ids)} chunks in Chroma collection '{COLLECTION_NAME}'")
    
    # 2. Build BM25 index
    bm25 = BM25Okapi(bm25_corpus_tokens)
    bm25_data = {
        "bm25": bm25,
        "chunk_ids": bm25_chunk_ids
    }
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"  [OK] Saved BM25 index to {BM25_INDEX_PATH}")
    
    # 3. Save chunks lookup JSON
    with open(CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_lookup, f, indent=2)
    print(f"  [OK] Saved chunks lookup dictionary to {CHUNKS_METADATA_PATH}")
    
    return {
        "total_chunks": len(all_chunks),
        "collection": collection,
        "bm25_data": bm25_data,
        "chunks_lookup": chunks_lookup
    }

def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve raw menu chunk content and metadata by its unique chunk_id (item_id)."""
    if CHUNKS_METADATA_PATH.exists():
        with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
            lookup = json.load(f)
            return lookup.get(chunk_id)
    return None

def load_or_build_indices() -> Dict[str, Any]:
    """Load existing Chroma collection, BM25 index, and chunks lookup."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    
    bm25_data = None
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
            
    chunks_lookup = {}
    if CHUNKS_METADATA_PATH.exists():
        with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
            chunks_lookup = json.load(f)
            
    return {
        "collection": collection,
        "bm25_data": bm25_data,
        "chunks_lookup": chunks_lookup
    }
