"""Global configuration and paths for BaristaAI Coffee Shop Assistant."""
import os
from pathlib import Path

# Safely load environment variables from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MENU_JSON_PATH = DATA_DIR / "coffee_menu.json"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
CHUNKS_METADATA_PATH = DATA_DIR / "chunks_lookup.json"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"
RESULTS_DIR = BASE_DIR / "results"
EVAL_REPORT_PATH = RESULTS_DIR / "eval_report.md"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
def get_configured_api_key() -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

GEMINI_API_KEY = get_configured_api_key()

# LLM Models
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = "gemini-3.5-flash-lite"
MODEL_CANDIDATES = [
    os.getenv("DEFAULT_MODEL", "gemini-3.5-flash"),
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

# Embedding & Reranking Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval Defaults
DEFAULT_RETRIEVAL_K = 5
RERANK_CANDIDATE_POOL_K = 15
RRF_K_CONSTANT = 60

# Agent Parameters
MAX_VERIFIER_LOOPS = 3

# Pricing & Cost Estimation
COST_PER_MILLION_INPUT_TOKENS = 0.15
COST_PER_MILLION_OUTPUT_TOKENS = 0.60
