"""Profiler Agent: Decomposes customer preferences, dietary constraints, and flavor goals."""
import json
import time
from typing import Dict, Any, List, Optional
from google import genai

from config import GEMINI_API_KEY, DEFAULT_MODEL
from src.observability.tracer import global_tracer
from src.utils.helpers import approximate_tokens, call_gemini_with_fallback

PROFILER_SYSTEM_PROMPT = """You are an expert Coffee Profiler & Sommelier Agent for an Artisanal Coffee Shop.
Your task is to analyze the customer's request and decode their exact preferences and dietary constraints.

Extract the following into a structured JSON:
1. sub_questions: 2-3 focused sub-queries to retrieve matching menu items.
2. caffeine_preference: "high", "medium", "low", "decaf", or "any".
3. temp_preference: "hot", "iced", or "any".
4. dietary_constraints: List of tags required (e.g., ["vegan", "dairy-free", "keto", "gluten-free", "sugar-free"]).
5. flavor_profile: List of target flavor notes (e.g., ["nutty", "chocolate", "vanilla", "fruity", "spiced"]).
6. pairing_requested: boolean.

Respond ONLY with a valid JSON object matching:
{
  "sub_questions": [
    {
      "id": 1,
      "question": "Recommended iced non-dairy low caffeine espresso or tea drink",
      "evidence_type": "Menu Item Selection",
      "search_queries": ["iced oat milk espresso", "matcha iced oat milk"]
    }
  ],
  "caffeine_preference": "low",
  "temp_preference": "iced",
  "dietary_constraints": ["vegan", "dairy-free"],
  "flavor_profile": ["creamy", "vanilla"],
  "pairing_requested": true
}"""

def run_planner(research_question: str, client: Optional[genai.Client] = None, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Execute the Profiler Agent to decode customer preferences."""
    start_time = time.time()
    
    if client is None and GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
    prompt = f"Customer Request: {research_question}\n\nDecode customer preferences, dietary restrictions, flavor profile, and search queries."
    
    if not client:
        # Fallback profiling
        data = {
            "sub_questions": [
                {
                    "id": 1,
                    "question": f"Menu search for: {research_question}",
                    "evidence_type": "Menu Item Search",
                    "search_queries": [research_question]
                }
            ],
            "caffeine_preference": "any",
            "temp_preference": "any",
            "dietary_constraints": [],
            "flavor_profile": [],
            "pairing_requested": True
        }
    else:
        try:
            raw_text = call_gemini_with_fallback(
                client=client,
                contents=prompt,
                system_instruction=PROFILER_SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                primary_model=model
            )
            data = json.loads(raw_text or "{}")
        except Exception:
            data = {
                "sub_questions": [{"id": 1, "question": research_question, "evidence_type": "Menu Search", "search_queries": [research_question]}],
                "caffeine_preference": "any",
                "temp_preference": "any",
                "dietary_constraints": [],
                "flavor_profile": [],
                "pairing_requested": True
            }
            
    latency_ms = (time.time() - start_time) * 1000
    
    global_tracer.log_trace(
        agent_name="ProfilerAgent",
        step_type="planning",
        input_text=research_question,
        output_text=json.dumps(data, indent=2),
        latency_ms=latency_ms,
        metadata={"caffeine": data.get("caffeine_preference"), "dietary": data.get("dietary_constraints")}
    )
    
    return data
