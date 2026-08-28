"""Concierge Editor Agent: Formulates an artisanal cafe menu card with pricing and item indices."""
import time
from typing import Dict, Any, List, Optional
from google import genai

from config import GEMINI_API_KEY, DEFAULT_MODEL
from src.agents.tools import get_chunk
from src.observability.tracer import global_tracer
from src.utils.helpers import extract_citations, approximate_tokens, call_gemini_with_fallback

EDITOR_SYSTEM_PROMPT = """You are the Lead Concierge Editor for an Artisanal Coffee House.
Your job is to take the verified recommendation draft and polish it into a cafe recommendation menu card.

EDITORIAL MANDATES:
1. CITATION PRESERVATION: Retain every single [item_id] citation attached to beverages and food pairings.
2. ELEGANT FORMATTING: Use clean headings, emoji accents, tasting notes, and barista customization instructions.
3. ITEM SUMMARY TABLE: Include a clear price and item breakdown table.
4. BARISTA NOTES: Add a friendly 2-sentence note from the head barista about flavor profile and brewing."""

def build_menu_references_section(cited_ids: List[str]) -> str:
    """Generate a structured item index with pricing and dietary tags."""
    if not cited_ids:
        return ""
        
    ref_lines = ["\n\n### 🧾 Menu Order & Item Index\n"]
    total_price = 0.0
    
    for cid in cited_ids:
        chunk = get_chunk(cid)
        item_data = chunk.get("item_data", {})
        name = chunk.get("paper_title", "Custom Item")
        price = item_data.get("price", 4.50)
        category = chunk.get("section", "General")
        dietary = ", ".join(item_data.get("dietary_tags", []))
        
        total_price += price
        ref_lines.append(f"- **`[{cid}]`** **{name}** (`{category}`) — **${price:.2f}**  \n  *Dietary Tags:* `{dietary or 'Standard'}`")
        
    ref_lines.append(f"\n> 💳 **Estimated Order Total:** **${total_price:.2f}**\n")
    return "\n".join(ref_lines)

def run_editor(
    research_question: str,
    verified_draft: str,
    unverified_claims: Optional[List[Dict[str, Any]]] = None,
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """Execute Concierge Editor Agent."""
    start_time = time.time()
    
    if client is None and GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
    prompt = f"Customer Request: {research_question}\n\nVERIFIED RECOMMENDATION DRAFT:\n{verified_draft}\n\n"
    prompt += "Format into an artisanal cafe recommendation card. Retain all [item_id] citations.\n"
    
    if not client:
        edited_text = verified_draft
    else:
        try:
            edited_text = call_gemini_with_fallback(
                client=client,
                contents=prompt,
                system_instruction=EDITOR_SYSTEM_PROMPT,
                temperature=0.2,
                primary_model=model
            )
        except Exception:
            edited_text = verified_draft
            
    cited_ids = extract_citations(edited_text)
    # If edited_text lost citations, fall back to extracting citations from verified_draft
    if not cited_ids:
        cited_ids = extract_citations(verified_draft)
        
    references = build_menu_references_section(cited_ids)
    
    final_report = f"# ☕ BaristaAI Personalized Recommendation\n\n" + edited_text + references
    latency_ms = (time.time() - start_time) * 1000
    
    global_tracer.log_trace(
        agent_name="ConciergeEditorAgent",
        step_type="editing",
        input_text=research_question,
        output_text=final_report,
        latency_ms=latency_ms,
        metadata={"cited_item_count": len(cited_ids)}
    )
    
    return {
        "final_report": final_report,
        "cited_chunk_ids": cited_ids,
        "unverified_claims": unverified_claims or []
    }
