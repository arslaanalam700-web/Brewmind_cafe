"""Barista Recommender Agent: Formulates tailored coffee, beverage, and pairing recommendations with item citations."""
import json
import time
from typing import Dict, Any, List, Optional
from google import genai

from config import GEMINI_API_KEY, DEFAULT_MODEL
from src.observability.tracer import global_tracer
from src.utils.helpers import approximate_tokens, call_gemini_with_fallback

BARISTA_SYSTEM_PROMPT = """You are a Master Artisanal Coffee Barista & Recommendation Agent.
Your role is to craft personalized coffee, tea, and pastry recommendations based SOLELY on the provided source menu items.

RULES:
1. CITATION MANDATE: Every recommended beverage or food item MUST be cited using its exact item_id in brackets [item_id] (e.g., [item_nitro_oat_latte]).
2. DIETARY ACCURACY: Strictly honor customer dietary constraints (vegan, dairy-free, gluten-free, keto, sugar-free). If a drink requires a milk swap (e.g. sub oat milk), explicitly mention the custom barista tweak.
3. PAIRING: Suggest a complementary bakery or food pairing if appropriate.
4. ZERO UNGROUNDED ITEMS: Recommend ONLY items present in the retrieved menu passages.

OUTPUT FORMAT:
Provide your recommendation in clean Markdown format with sections for:
- Recommended Beverage
- Custom Barista Modifications (Milk swaps, syrups, shots)
- Complementary Bakery Pairing
- Flavor Profile & Tasting Notes"""

def format_menu_evidence(evidence_data: Dict[str, Any]) -> str:
    """Format retrieved menu chunks for the prompt."""
    sub_questions_evidence = evidence_data.get("sub_questions_evidence", {})
    blocks = []
    
    for sq_id, data in sub_questions_evidence.items():
        chunks = data.get("chunks", [])
        for c in chunks:
            blocks.append(f"--- MENU ITEM [{c['chunk_id']}] ---")
            blocks.append(c['text'])
            blocks.append(f"--- END ITEM [{c['chunk_id']}] ---\n")
            
    return "\n".join(blocks)

def run_writer(
    research_question: str,
    evidence_data: Dict[str, Any],
    verifier_feedback: Optional[str] = None,
    iteration: int = 1,
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """Execute Barista Recommender Agent."""
    start_time = time.time()
    
    if client is None and GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
    menu_context = format_menu_evidence(evidence_data)
    
    prompt = f"Customer Preference Request: {research_question}\n\n"
    prompt += f"AVAILABLE MENU ITEMS:\n{menu_context}\n\n"
    
    if verifier_feedback:
        prompt += f"REVISION NEEDED - DIETARY VERIFIER FEEDBACK (Iteration {iteration - 1}):\n{verifier_feedback}\n\n"
        prompt += "Please revise your recommendation to strictly fix the dietary or caffeine violation above.\n"
    else:
        prompt += "Please create a personalized recommendation card citing exact [item_id] tags for all items.\n"
        
    if not client:
        # Fallback offline recommendation
        retrieved_chunks = []
        for sq_id, data in evidence_data.get("sub_questions_evidence", {}).items():
            retrieved_chunks.extend(data.get("chunks", []))
            
        rec_item = retrieved_chunks[0] if retrieved_chunks else {"chunk_id": "item_espresso_classic", "paper_title": "Artisanal Double Espresso"}
        pairing_item = retrieved_chunks[1] if len(retrieved_chunks) > 1 else {"chunk_id": "item_croissant_butter", "paper_title": "French Butter Croissant"}
        
        draft = f"### Recommended Beverage: {rec_item.get('paper_title')} [{rec_item.get('chunk_id')}]\n"
        draft += f"A perfect choice based on your preferences. Enjoy smooth flavor notes and quality brewing [{rec_item.get('chunk_id')}].\n\n"
        draft += f"### Recommended Pairing: {pairing_item.get('paper_title')} [{pairing_item.get('chunk_id')}]\n"
        draft += f"Pairs excellently with your drink [{pairing_item.get('chunk_id')}].\n"
    else:
        try:
            draft = call_gemini_with_fallback(
                client=client,
                contents=prompt,
                system_instruction=BARISTA_SYSTEM_PROMPT,
                temperature=0.3,
                primary_model=model
            )
        except Exception as e:
            draft = f"Recommendation generation encountered an issue: {e}"
        
    latency_ms = (time.time() - start_time) * 1000
    
    global_tracer.log_trace(
        agent_name=f"BaristaAgent (Iter {iteration})",
        step_type="writing",
        input_text=research_question,
        output_text=draft,
        latency_ms=latency_ms,
        metadata={"iteration": iteration}
    )
    
    return {"draft": draft, "iteration": iteration}
