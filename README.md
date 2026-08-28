# ☕ BrewMind Café — Artisanal AI Barista & Multi-Agent Sommelier

An intelligent, multi-agent AI Barista and coffee recommender system built with **Google Gemini**, **Streamlit**, and **Multi-Strategy Hybrid RAG**. 

BrewMind Café decodes customer flavor preferences, dietary restrictions (Vegan, Dairy-Free, Gluten-Free, Keto, Sugar-Free), and caffeine goals to craft tailored beverage recommendations and bakery pairings with strict item citations and constraint verification.

---

## 🌟 Key Features

- **☕ Personalized Barista Recommendations**: Suggests espresso drinks, single-origin pourovers, nitro cold brews, ceremonial matcha, and artisanal bakery pairings.
- **🛡️ 5-Agent Collaborative Pipeline**:
  1. **Profiler Agent**: Decomposes customer requests into flavor profiles, dietary constraints, and targeted search queries.
  2. **Retriever Agent**: Searches the curated coffee corpus using active retrieval strategies.
  3. **Barista Recommender Agent**: Crafts personalized recommendations citing exact item IDs (`[item_id]`).
  4. **Dietary Verifier Agent**: Validates ingredients and dietary tags against customer constraints with automatic revision loops.
  5. **Concierge Editor Agent**: Polishes output into an elegant recommendation card with price breakdown and order totals.
- **⚡ Multi-Strategy Hybrid RAG Engine**:
  - **Dense Vector Search**: ChromaDB with `sentence-transformers/all-MiniLM-L6-v2`.
  - **Lexical Search**: BM25 keyword matching with customized tokenization.
  - **Reciprocal Rank Fusion (RRF)**: Combines dense semantic similarity and BM25 scores.
  - **Cross-Encoder Reranking**: Re-scores candidate pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise ranking.
- **✨ Luxury Glassmorphic UI**: Sleek, minimalist, 100% emoji-free design with dynamic background, interactive menu explorer, dietary toggles, and instant prompt chips.
- **🔄 Robust Failover & Universal Compatibility**: Automatic fallback rotation (`gemini-3.5-flash` → `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite`) with direct HTTPS REST API fallback.

---

## 🏗️ 5-Agent Orchestration Architecture

```mermaid
flowchart TD
    User([Customer Request]) --> Profiler[1. Profiler / Planner Agent]
    Profiler -->|Sub-Queries & Dietary Constraints| Retriever[2. Retriever Agent]
    
    subgraph MultiStrategyRAG ["Hybrid Retrieval Engine"]
        Retriever -->|Query| Strategy{Strategy Factory}
        Strategy -->|Strategy A| Dense[Dense Vector Search]
        Strategy -->|Strategy B| Hybrid[Hybrid BM25 + Dense RRF]
        Strategy -->|Strategy C| Rerank[Hybrid + Cross-Encoder Rerank]
    end
    
    MultiStrategyRAG -->|Menu Chunks + Item IDs| Loop
    
    subgraph Loop ["Verification Refinement Loop (Max 3 Iterations)"]
        Writer[3. Barista Recommender Agent] -->|Draft with [item_id] citations| Verifier[4. Dietary Verifier Agent]
        Verifier -->|Verify Ingredients & Dietary Constraints| Verdict{Approved?}
        Verdict -->|No: Revision Feedback| Writer
    end
    
    Verdict -->|Yes / Max Loops| Editor[5. Concierge Editor Agent]
    Editor -->|Price Calculation & Menu Card Synthesis| Report([Personalized Menu Card])
```

---

## 📊 Retrieval Benchmark Results

*Evaluated on hand-labeled menu queries across artisanal coffee and pastry items.*

| Retrieval Strategy | MRR (Mean Reciprocal Rank) | Precision@3 | Precision@5 | Recall@3 | Recall@5 | Hit Rate@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Naive Dense (Top-k)** | 0.8142 | 0.7167 | 0.6300 | 0.8250 | 0.9250 | 95.0% |
| **Strategy B: Hybrid (BM25 + Dense RRF)** | 0.8917 | 0.8000 | 0.8000 | 0.9750 | 0.9750 | 100.0% |
| **Strategy C: Hybrid + Cross-Encoder Rerank** | **0.9000** | **0.8667** | **0.8200** | **0.9750** | **0.9750** | **100.0%** |

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/arslaanalam700-web/Brewmind_cafe.git
cd Brewmind_cafe
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
DEFAULT_MODEL=gemini-3.5-flash
```

### 4. Run the Streamlit Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
BrewMind Café/
├── app/
│   └── streamlit_app.py        # Glassmorphic Streamlit web application & AI Barista chat
├── config.py                   # Central configuration, model settings, paths & pricing
├── data/
│   ├── coffee_menu.json        # Curated artisanal coffee, tea & bakery menu
│   ├── chunks_lookup.json      # Structured metadata & chunk lookup dictionary
│   ├── cafe_bg.jpg             # High-resolution aesthetic cafe background
│   ├── chroma_db/              # Persistent ChromaDB vector database
│   └── bm25_index.pkl          # Serialized BM25 index
├── src/
│   ├── agents/                 # Multi-agent architecture
│   │   ├── planner.py          # Profiler Agent (Preference & Constraint analysis)
│   │   ├── retriever_agent.py  # Retriever Agent (Executes hybrid search)
│   │   ├── writer.py           # Barista Recommender Agent (Drafts menu suggestions)
│   │   ├── verifier.py         # Dietary Safety Verifier Agent (Validates tags & ingredients)
│   │   ├── editor.py           # Concierge Editor Agent (Generates menu card & pricing)
│   │   ├── pipeline.py         # 5-Agent orchestrator pipeline
│   │   └── tools.py            # Search & chunk lookup tools
│   ├── retrieval/              # Retrieval strategies & factory
│   │   ├── base.py             # Abstract BaseRetriever class & dataclasses
│   │   ├── dense.py            # Naive Dense vector retriever
│   │   ├── hybrid.py           # Hybrid BM25 + Dense RRF retriever
│   │   ├── reranker.py         # Cross-Encoder neural reranker
│   │   └── factory.py          # Cached retriever factory
│   ├── ingestion/              # Data parsing, chunking & index builders
│   ├── observability/          # Tracing, session logging & latency metrics
│   └── utils/                  # Robust LLM calling engine & citation helpers
├── eval/
│   ├── metrics.py              # Precision@k, Recall@k, MRR & Hit Rate metrics
│   ├── llm_judge.py            # LLM-as-a-Judge for faithfulness and relevance
│   └── run_eval.py             # Evaluation benchmark runner
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment file template
└── .gitignore                  # Git ignore rules (protects API keys & caches)
```

---

## 🌐 Deployment

### Deploying to Streamlit Community Cloud (Free)
1. Fork or push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select your repository: `arslaanalam700-web/Brewmind_cafe`.
4. Set **Main file path** to: `app/streamlit_app.py`.
5. Under **Advanced Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   DEFAULT_MODEL = "gemini-3.5-flash"
   ```
6. Click **Deploy!**

---

## 📜 License

This project is licensed under the MIT License.
