"""Retrieval and generation evaluation metrics computation."""
from typing import List, Set, Dict, Any

def normalize_id(cid: str) -> str:
    """Normalize chunk or paper ID for matching."""
    return cid.strip().lower()

def precision_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Calculate Precision@k: (number of relevant items in top-k) / k
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
        
    top_k = retrieved_ids[:k]
    # Check exact chunk ID match or matching paper ID prefix
    expected_set = {normalize_id(eid) for eid in expected_ids}
    
    hits = 0
    for rid in top_k:
        norm_rid = normalize_id(rid)
        # Match either exact chunk_id or paper_id prefix
        if norm_rid in expected_set or any(norm_rid.startswith(eid.split(':')[0]) for eid in expected_set):
            hits += 1
            
    return hits / float(k)

def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Calculate Recall@k: (number of relevant items in top-k) / (total number of relevant items)
    """
    if not expected_ids or not retrieved_ids:
        return 0.0
        
    top_k = retrieved_ids[:k]
    expected_set = {normalize_id(eid) for eid in expected_ids}
    
    matched_expected = set()
    for rid in top_k:
        norm_rid = normalize_id(rid)
        for eid in expected_set:
            if norm_rid == eid or norm_rid.startswith(eid.split(':')[0]):
                matched_expected.add(eid)
                
    return len(matched_expected) / float(len(expected_ids))

def mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR): 1 / (rank of first relevant item)
    """
    if not expected_ids or not retrieved_ids:
        return 0.0
        
    expected_set = {normalize_id(eid) for eid in expected_ids}
    
    for rank, rid in enumerate(retrieved_ids, 1):
        norm_rid = normalize_id(rid)
        if norm_rid in expected_set or any(norm_rid.startswith(eid.split(':')[0]) for eid in expected_set):
            return 1.0 / rank
            
    return 0.0

def hit_rate_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Calculate Hit Rate@k: 1.0 if at least one expected item is in top-k, else 0.0
    """
    return 1.0 if precision_at_k(retrieved_ids, expected_ids, k) > 0 else 0.0

def compute_retrieval_metrics(
    retrieved_ids: List[str],
    expected_ids: List[str],
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, float]:
    """Compute full suite of retrieval metrics for a query."""
    metrics = {
        "mrr": mrr(retrieved_ids, expected_ids)
    }
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, expected_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, expected_ids, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, expected_ids, k)
    return metrics
