"""Retriever Agent: Executes retrieval for each sub-question using the active retrieval strategy."""
import time
from typing import Dict, Any, List
from .tools import search_documents
from src.observability.tracer import global_tracer

def run_retriever_agent(plan_data: Dict[str, Any], k_per_query: int = 4) -> Dict[str, Any]:
    """Execute retrieval across all planned sub-questions."""
    start_time = time.time()
    sub_questions = plan_data.get("sub_questions", [])
    
    retrieval_results = {}
    total_retrieved = 0
    all_chunk_ids_seen = set()
    
    for sq in sub_questions:
        sq_id = str(sq.get("id", 1))
        question_text = sq.get("question", "")
        queries = sq.get("search_queries", [question_text])
        
        sq_chunks = []
        sq_chunk_ids = set()
        
        # Search using the main question text
        main_results = search_documents(question_text, k=k_per_query)
        for chunk in main_results:
            if chunk["chunk_id"] not in sq_chunk_ids:
                sq_chunk_ids.add(chunk["chunk_id"])
                sq_chunks.append(chunk)
                
        # Search using specific keywords
        for q in queries:
            if len(sq_chunks) >= k_per_query * 2:
                break
            res = search_documents(q, k=k_per_query)
            for chunk in res:
                if chunk["chunk_id"] not in sq_chunk_ids:
                    sq_chunk_ids.add(chunk["chunk_id"])
                    sq_chunks.append(chunk)
                    
        retrieval_results[sq_id] = {
            "sub_question": question_text,
            "evidence_type": sq.get("evidence_type", ""),
            "chunks": sq_chunks
        }
        
        total_retrieved += len(sq_chunks)
        all_chunk_ids_seen.update(sq_chunk_ids)
        
    latency_ms = (time.time() - start_time) * 1000
    
    global_tracer.log_trace(
        agent_name="RetrieverAgent",
        step_type="retrieval",
        input_text=f"Retrieving evidence for {len(sub_questions)} sub-questions",
        output_text=f"Retrieved {total_retrieved} total chunk candidates ({len(all_chunk_ids_seen)} unique across sub-questions)",
        latency_ms=latency_ms,
        metadata={
            "total_chunks": total_retrieved,
            "unique_chunk_count": len(all_chunk_ids_seen),
            "sub_question_count": len(sub_questions)
        }
    )
    
    return {
        "sub_questions_evidence": retrieval_results,
        "unique_chunk_ids": list(all_chunk_ids_seen)
    }
