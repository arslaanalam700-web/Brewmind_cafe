"""Helper utilities for citation extraction, token estimation, markdown rendering, and robust multi-model LLM generation."""
import re
import time
import json
from typing import List, Set, Dict, Any, Optional
import requests

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None
    types = None

from config import GEMINI_API_KEY, DEFAULT_MODEL, MODEL_CANDIDATES

# Matches cafe menu item IDs like [item_espresso_classic] or paper citations like [dpo:p04_c02]
CITATION_REGEX = re.compile(r'\[(item_[a-zA-Z0-9_]+|[a-zA-Z0-9_\-]+:p\d+_c\d+|item-[a-zA-Z0-9_]+)\]')

def extract_citations(text: str) -> List[str]:
    """Extract all unique chunk citations from generated text in order of appearance."""
    citations = CITATION_REGEX.findall(text)
    # Deduplicate while preserving order
    seen = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)
    return unique_citations

def split_text_into_claims(text: str) -> List[str]:
    """
    Split generated report text into individual claim sentences.
    Focuses on sentences that contain or should contain citations.
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15 and not s.strip().startswith('#')]
    return clean_sentences

def approximate_tokens(text: str) -> int:
    """Approximate token count (1 token ~= 4 chars or 0.75 words)."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))

def call_gemini_rest_fallback(
    contents: Any,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    response_mime_type: Optional[str] = None,
    candidate_models: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> str:
    """Fallback REST API caller if google-genai package is not initialized."""
    key = api_key or GEMINI_API_KEY
    models = candidate_models or MODEL_CANDIDATES
    prompt_text = contents if isinstance(contents, str) else str(contents)
    
    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": temperature}
        }
        if response_mime_type:
            payload["generationConfig"]["responseMimeType"] = response_mime_type
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                data = r.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
        except Exception:
            continue
            
    raise RuntimeError("All Gemini model endpoints failed via REST fallback.")

def call_gemini_with_fallback(
    client: Optional[Any] = None,
    contents: Any = "",
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    response_mime_type: Optional[str] = None,
    candidate_models: Optional[List[str]] = None,
    primary_model: Optional[str] = None,
    max_retries_per_model: int = 2,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Execute generate_content with automatic model fallback and retry on 503/404/demand spikes.
    Falls back to direct REST calls if genai library is unavailable.
    """
    effective_api_key = api_key or GEMINI_API_KEY

    models_to_try = []
    if primary_model:
        models_to_try.append(primary_model)
    if candidate_models:
        for m in candidate_models:
            if m not in models_to_try:
                models_to_try.append(m)
    else:
        for m in MODEL_CANDIDATES:
            if m not in models_to_try:
                models_to_try.append(m)

    if not HAS_GENAI:
        return call_gemini_rest_fallback(
            contents=contents,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type=response_mime_type,
            candidate_models=models_to_try,
            api_key=effective_api_key
        )

    if client is None:
        try:
            client = genai.Client(api_key=effective_api_key)
        except Exception:
            return call_gemini_rest_fallback(
                contents=contents,
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type=response_mime_type,
                candidate_models=models_to_try,
                api_key=effective_api_key
            )

    config_kwargs: Dict[str, Any] = {"temperature": temperature}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    config = types.GenerateContentConfig(**config_kwargs)

    last_error = None
    for model_name in models_to_try:
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                if response and response.text is not None:
                    return response.text
            except Exception as ex:
                last_error = ex
                err_str = str(ex).lower()
                # If 404 (model deprecated/unavailable), don't retry same model; move to next fallback
                if "404" in err_str or "not_found" in err_str or "no longer available" in err_str:
                    break
                # For 503, demand spikes, rate limits, or transient errors, wait briefly before retrying
                time.sleep(0.5 * (attempt + 1))
                continue

    # Final attempt via REST if SDK calls encountered issues
    try:
        return call_gemini_rest_fallback(
            contents=contents,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type=response_mime_type,
            candidate_models=models_to_try,
            api_key=effective_api_key
        )
    except Exception:
        pass

    if last_error:
        raise last_error
    return ""
