"""BrewMind Café — Sleek, Minimalist, 100% Emoji-Free AI Barista Chatbot Website with Crash-Proof RAG Imports."""
import os
import sys
import json
import time
import base64
from pathlib import Path
import streamlit as st

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Safely import google-genai or fallback
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Safely import RAG retrieval module or fallback to direct keyword search
try:
    from src.retrieval.factory import get_retriever, RetrievalStrategy
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

from config import (
    GEMINI_API_KEY,
    DEFAULT_MODEL,
    MODEL_CANDIDATES,
    MENU_JSON_PATH,
    CHUNKS_METADATA_PATH
)
import importlib
import src.utils.helpers
try:
    importlib.reload(src.utils.helpers)
except Exception:
    pass
from src.utils.helpers import call_gemini_with_fallback

# High-Res Aesthetic Cafe Background Image Path
HD_BG_IMAGE_PATH = BASE_DIR / "data" / "cafe_bg.jpg"

def get_base64_bg(image_path: Path) -> str:
    """Convert image to base64 string for CSS background injection."""
    if image_path.exists():
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

bg_base64 = get_base64_bg(HD_BG_IMAGE_PATH)

# Page configuration (No emojis)
st.set_page_config(
    page_title="BrewMind Café • AI Barista",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Aesthetic Minimalist CSS (Emoji-Free)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700;900&family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    /* 4K Crystal Clear Full Page Background Image */
    .stApp {{
        background: linear-gradient(rgba(10, 5, 3, 0.60), rgba(10, 5, 3, 0.70)), url("data:image/jpg;base64,{bg_base64}") no-repeat center center fixed;
        background-size: cover;
        background-position: center;
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1080px;
    }}

    /* Blurry Transparent Glassmorphic Sidebar */
    [data-testid="stSidebar"] {{
        background: rgba(15, 8, 6, 0.45) !important;
        backdrop-filter: blur(22px) !important;
        -webkit-backdrop-filter: blur(22px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 6px 0 24px rgba(0, 0, 0, 0.4) !important;
    }}

    /* Sidebar Text & Label Color Fix */
    [data-testid="stSidebar"] * {{
        color: #FFF8EE !important;
    }}

    /* Sidebar Expanders Frosted Glass */
    [data-testid="stSidebar"] div[data-baseweb="accordion"] {{
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        margin-bottom: 8px !important;
    }}

    /* Glassmorphic Hero Banner */
    .hero-banner {{
        background: rgba(20, 9, 7, 0.65);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 20px;
        padding: 2.2rem 2.6rem;
        color: #FFFDF9;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 30px -8px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }}

    /* Elegant Luxury Cinzel Decorative Title */
    .brand-title {{
        font-family: 'Cinzel Decorative', 'Cormorant Garamond', serif;
        font-size: 2.7rem;
        font-weight: 900;
        letter-spacing: 0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #FDE68A 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 0.4rem;
    }}

    .brand-subtitle {{
        font-size: 1.05rem;
        color: #E2D1C3;
        font-weight: 500;
        max-width: 820px;
        line-height: 1.5;
    }}

    /* Tag Badges */
    .tag-pill {{
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #FEF3C7;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 50px;
        margin-right: 6px;
        margin-top: 10px;
    }}

    /* Glassmorphic Chat Messages */
    .stChatMessage {{
        background: rgba(20, 10, 8, 0.70) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 14px !important;
        color: #FDFBF7 !important;
        margin-bottom: 0.8rem !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45) !important;
    }}

    /* FORCE TRANSPARENCY ON ALL STREAMLIT BOTTOM CONTAINER SELECTORS */
    [data-testid="stBottomBlockContainer"],
    div[data-testid="stBottom"],
    section[data-testid="stBottom"],
    .stBottomBlockContainer,
    footer,
    header[data-testid="stHeader"] {{
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }}

    /* Styling the Chat Input Pill */
    div[data-testid="stChatInput"] {{
        background: rgba(20, 9, 7, 0.75) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    }}

    div[data-testid="stChatInput"] textarea {{
        color: #FFF8EE !important;
        background: transparent !important;
    }}

    div[data-testid="stChatInput"] textarea::placeholder {{
        color: #D1C4B6 !important;
    }}

    /* Price Badge */
    .price-badge {{
        background: #FEF3C7;
        color: #92400E;
        font-weight: 800;
        font-size: 0.85rem;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid #FDE68A;
    }}

    /* Custom Chat Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.85) 0%, rgba(69, 26, 3, 0.85) 100%);
        backdrop-filter: blur(10px);
        color: #FFFDF9 !important;
        font-weight: 700;
        font-size: 0.92rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease;
    }}

    .stButton>button:hover {{
        background: linear-gradient(135deg, rgba(146, 64, 14, 0.95) 0%, rgba(120, 53, 15, 0.95) 100%);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.6);
        transform: translateY(-1px);
    }}
</style>
""", unsafe_allow_html=True)

# Load Menu Database
@st.cache_data
def get_menu_data():
    if MENU_JSON_PATH.exists():
        with open(MENU_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

menu_items = get_menu_data()

# Fallback Menu Search if RAG (ChromaDB) module is not loaded
def fallback_menu_search(query: str):
    q = query.lower()
    matches = []
    for item in menu_items:
        text = f"{item.get('name', '')} {item.get('category', '')} {item.get('description', '')} {' '.join(item.get('flavor_notes', []))} {' '.join(item.get('dietary_tags', []))}".lower()
        if any(w in text for w in q.split()):
            matches.append(item)
    return matches[:5] if matches else menu_items[:5]

# Helper to get active API key dynamically
def get_current_api_key() -> str:
    # 1. Streamlit Secrets (for Streamlit Community Cloud)
    try:
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    # 2. Environment Variables (.env)
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env", override=True)
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY or ""

# Sidebar: Customer Preference & Menu Navigator (Emoji-Free)
with st.sidebar:
    st.markdown("## BrewMind Café")
    st.caption("Artisanal Coffee & AI Barista")
    
    st.markdown("---")
    st.markdown("### API Key & Controls")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=get_current_api_key(),
        type="password",
        help="Reads automatically from .env or paste here"
    )
    
    if st.button("Reset Chat Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("### Dietary & Taste Filter")
    f_vegan = st.checkbox("Vegan", value=False)
    f_dairy_free = st.checkbox("Dairy-Free", value=False)
    f_gluten_free = st.checkbox("Gluten-Free", value=False)
    f_keto = st.checkbox("Keto Suitable", value=False)
    f_sugar_free = st.checkbox("Sugar-Free", value=False)
    
    st.markdown("---")
    st.markdown("### Menu Explorer")
    
    # Filter menu based on sidebar toggles
    filtered_menu = []
    for item in menu_items:
        tags = item.get("dietary_tags", [])
        if f_vegan and "vegan" not in tags:
            continue
        if f_dairy_free and "dairy-free" not in tags:
            continue
        if f_gluten_free and "gluten-free" not in tags:
            continue
        if f_keto and "keto" not in tags:
            continue
        if f_sugar_free and "sugar-free" not in tags:
            continue
        filtered_menu.append(item)
        
    st.caption(f"Showing **{len(filtered_menu)}** matching items")
    
    categories = sorted(list(set(i.get("category", "General") for i in filtered_menu)))
    for cat in categories:
        with st.expander(f"{cat}"):
            cat_items = [i for i in filtered_menu if i.get("category") == cat]
            for item in cat_items:
                st.markdown(f"**{item['name']}** <span class='price-badge'>${item['price']:.2f}</span>", unsafe_allow_html=True)
                st.caption(f"Notes: _{', '.join(item.get('flavor_notes', []))}_")
                st.caption(f"Tags: `{', '.join(item.get('dietary_tags', []))}`")
                st.markdown("---")

# Main Header Banner (Emoji-Free)
st.markdown("""
<div class="hero-banner">
    <div class="brand-title">BrewMind Artisanal AI Barista</div>
    <div class="brand-subtitle">
        Welcome to BrewMind Café. Ask our AI Barista for beverage recommendations, dietary advice, origin profiles, brewing methods, or food pairings.
    </div>
    <div>
        <span class="tag-pill">Single-Origin Pourovers</span>
        <span class="tag-pill">Nitro Cold Brews</span>
        <span class="tag-pill">Uji Ceremonial Matcha</span>
        <span class="tag-pill">Vegan & Keto Options</span>
        <span class="tag-pill">Fresh Bakery Pairings</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Chat History (Emoji-Free)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to **BrewMind Café**. I am your Head Barista. How can I craft your experience today? Tell me what you are craving (e.g., *\"I need an iced non-dairy coffee under $6\"* or *\"Recommend a bakery pairing for an espresso\"*)."
        }
    ]

# Render Chat History (Emoji-Free)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick Prompt Action Chips (Emoji-Free)
st.markdown("##### Popular Requests:")
chip_cols = st.columns(4)
selected_prompt = None

if chip_cols[0].button("Iced Oat Milk Special", use_container_width=True):
    selected_prompt = "Recommend an iced oat milk coffee drink with flavor notes and exact price."
if chip_cols[1].button("Coffee & Pastry Combo", use_container_width=True):
    selected_prompt = "Suggest a perfect coffee and bakery pairing under $10."
if chip_cols[2].button("Vegan & Dairy-Free", use_container_width=True):
    selected_prompt = "What vegan and dairy-free options do you offer on the menu?"
if chip_cols[3].button("Strong Single-Origin", use_container_width=True):
    selected_prompt = "What single-origin pourover coffee do you recommend for a high caffeine kick?"

# User Input
chat_input_val = st.chat_input("Type your order or ask a question about our coffee menu...")
prompt = selected_prompt or chat_input_val

if prompt:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate RAG Barista Response
    with st.chat_message("assistant"):
        with st.spinner("Barista is preparing your recommendation..."):
            try:
                # 1. Execute Retrieval Search
                if HAS_RAG:
                    retriever = get_retriever("hybrid_rerank")
                    retrieved_chunks = retriever.retrieve(prompt, k=5)
                    context_blocks = []
                    for c in retrieved_chunks:
                        context_blocks.append(f"--- MENU ITEM ID: {c.paper_id} ---")
                        context_blocks.append(f"Name: {c.paper_title}")
                        context_blocks.append(f"Category: {c.section}")
                        context_blocks.append(c.text)
                        context_blocks.append("--- END ITEM ---\n")
                    context_str = "\n".join(context_blocks)
                else:
                    matched_items = fallback_menu_search(prompt)
                    context_blocks = []
                    for m in matched_items:
                        context_blocks.append(f"Name: {m['name']} (${m['price']:.2f}) | Category: {m['category']}")
                        context_blocks.append(f"Description: {m.get('description', '')} | Tags: {', '.join(m.get('dietary_tags', []))}")
                    context_str = "\n".join(context_blocks)
                
                # Active Dietary Constraints string
                active_constraints = []
                if f_vegan: active_constraints.append("Vegan")
                if f_dairy_free: active_constraints.append("Dairy-Free")
                if f_gluten_free: active_constraints.append("Gluten-Free")
                if f_keto: active_constraints.append("Keto")
                if f_sugar_free: active_constraints.append("Sugar-Free")
                
                constraint_note = f"CUSTOMER DIETARY FILTERS: {', '.join(active_constraints)}" if active_constraints else "No active dietary filters."
                
                # 2. Call Gemini LLM
                key_to_use = (api_key_input.strip() if api_key_input and api_key_input.strip() else get_current_api_key())
                
                client = genai.Client(api_key=key_to_use) if (HAS_GENAI and key_to_use) else None
                
                sys_prompt = f"""You are the Master Head Barista & Sommelier at BrewMind Café.
Your mission is to provide warm, knowledgeable, and elegant coffee, tea, and pastry recommendations based SOLELY on the provided coffee menu context.

{constraint_note}

RULES:
1. Be warm, professional, and welcoming like a luxury café barista.
2. ALWAYS provide exact item names, prices ($), flavor notes, and key ingredients from the menu context.
3. Offer custom barista tweaks (e.g. "Sub oat milk", "Extra shot", "Add sugar-free syrup") and food pairings.
4. Format your answer with clean Markdown headers, bullet points, and pricing summaries.
5. DO NOT use any emojis in your response. Keep text clean, sophisticated, and professional."""

                full_user_prompt = f"CUSTOMER QUERY: {prompt}\n\nRETRIEVED MENU CONTEXT:\n{context_str}\n\nCraft a detailed Barista response without emojis."
                
                try:
                    bot_text = call_gemini_with_fallback(
                        client=client,
                        contents=full_user_prompt,
                        system_instruction=sys_prompt,
                        temperature=0.3,
                        candidate_models=MODEL_CANDIDATES,
                        primary_model=DEFAULT_MODEL,
                        api_key=key_to_use
                    )
                except Exception as ex:
                    bot_text = f"I retrieved these matching menu items for your request:\n\n{context_str}\n\n*(API notice: {ex})*"
                    
                st.markdown(bot_text)
                st.session_state.messages.append({"role": "assistant", "content": bot_text})
                
            except Exception as e:
                err_msg = f"Barista Error: {e}"
                st.error(err_msg)
