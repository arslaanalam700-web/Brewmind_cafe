"""Step 3 Baseline: Single ADK Agent RAG wrapping search_documents tool."""
import os
import sys
import time
from pathlib import Path
from typing import Optional
from google import genai

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GEMINI_API_KEY, DEFAULT_MODEL
from src.agents.tools import search_documents, set_active_retrieval_strategy
from src.utils.helpers import call_gemini_with_fallback

BASELINE_SYSTEM_PROMPT = """You are a Baseline Single-Agent RAG Literature Assistant.
Your task is to answer the user's research question by calling the search_documents tool to retrieve relevant evidence chunks.

RULES:
1. First, search for relevant documents using search_documents(query).
2. Write a clear, structured response based on the retrieved chunks.
3. Attach bracketed citations [chunk_id] (e.g. [dpo:p04_c02] or [item_espresso_classic]) to every factual claim."""

def safe_str(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")

def run_baseline_single_agent(query: str, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> str:
    """Execute single baseline agent with RAG tool."""
    print("=" * 70)
    print(" STEP 3 BASELINE: SINGLE-AGENT RAG EXECUTION")
    print(f" Query: '{safe_str(query)}'")
    print("=" * 70)
    
    # 1. Retrieve evidence chunks via search_documents tool
    set_active_retrieval_strategy("naive_dense")
    retrieved_chunks = search_documents(query, k=5)
    
    print(f"Retrieved {len(retrieved_chunks)} baseline chunks via Naive Dense strategy:\n")
    context_str = ""
    for c in retrieved_chunks:
        title = safe_str(c['paper_title'][:50])
        print(f"  * [{c['chunk_id']}] (Page {c['page']}) {title}")
        context_str += f"\n--- CHUNK [{c['chunk_id']}] ({c['paper_title']}, Page {c['page']}) ---\n{c['text']}\n"
        
    key_to_use = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not key_to_use:
        print("\n[!] Notice: No GEMINI_API_KEY found in .env. Returning structured evidence summary from baseline tool.")
        answer = f"### Baseline Retrieval Response for: {query}\n\n"
        answer += "Based on the retrieved baseline chunks:\n"
        for c in retrieved_chunks:
            answer += f"- According to *{c['paper_title']}* [{c['chunk_id']}] (Page {c['page']}):\n"
            answer += f"  > \"{safe_str(c['text'][:200])}...\"\n\n"
        print("\n" + "=" * 70)
        print(" BASELINE SINGLE-AGENT RESPONSE (OFFLINE FALLBACK):")
        print("=" * 70)
        print(answer)
        print("=" * 70)
        return answer

    # 2. Call Gemini for baseline single-agent answer with fallback
    client = genai.Client(api_key=key_to_use)
    prompt = f"USER RESEARCH QUESTION: {query}\n\nRETRIEVED CONTEXT CHUNKS:\n{context_str}\n\nPlease generate a cited answer based ONLY on the chunks above."
    
    answer = call_gemini_with_fallback(
        client=client,
        contents=prompt,
        system_instruction=BASELINE_SYSTEM_PROMPT,
        temperature=0.2,
        primary_model=model
    )
    
    print("\n" + "=" * 70)
    print(" BASELINE SINGLE-AGENT RESPONSE:")
    print("=" * 70)
    print(safe_str(answer))
    print("=" * 70)
    
    return answer

if __name__ == "__main__":
    test_q = "How does Direct Preference Optimization (DPO) avoid training an explicit reward model compared to PPO?"
    run_baseline_single_agent(test_q)
