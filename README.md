# 📚 Agentic Literature Review Assistant & Multi-Strategy RAG Benchmark

An end-to-end, interview-grade AI literature review system built with **Google ADK (Python)** and **Gemini**. The assistant decomposes complex research questions, retrieves evidence chunks across a 25-paper RLHF corpus, drafts literature reviews with strict claim-level chunk citations (`[chunk_id]`), verifies every single claim against source passages in a refinement loop, and synthesizes publication-ready reports.

Alongside the generation pipeline, a custom **Evaluation Harness** empirically benchmarks 3 retrieval strategies (Naive Dense vs. Hybrid BM25+Dense RRF vs. Hybrid + Cross-Encoder Rerank) using hand-labeled ground-truth evaluation sets.

---

## 📊 Empirical Retrieval Benchmark Results

*Evaluated on 20 hand-labeled research queries across 25 landmark papers (767 indexed chunks).*

| Retrieval Strategy | MRR (Mean Reciprocal Rank) | Precision@3 | Precision@5 | Recall@3 | Recall@5 | Hit Rate@5 | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Naive Dense (Top-k)** | 0.8142 | 0.7167 | 0.6300 | 0.8250 | 0.9250 | 95.0% | **25.4 ms** |
| **Strategy B: Hybrid (BM25 + Dense RRF)** | 0.8917 | 0.8000 | 0.8000 | 0.9750 | 0.9750 | 100.0% | 261.6 ms |
| **Strategy C: Hybrid + Cross-Encoder Rerank** | **0.9000** | **0.8667** | **0.8200** | **0.9750** | **0.9750** | **100.0%** | 2734.4 ms |

### 🔑 Key Takeaway & Winner
**Strategy C (Hybrid + Cross-Encoder Rerank)** is the empirical winner, achieving **0.9000 MRR** and **86.67% Precision@3**. Combining BM25 keyword matching with dense embeddings via Reciprocal Rank Fusion ($k=60$) prevents semantic drift on technical formulas, while the Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) accurately ranks the exact proof/derivation chunks above peripheral mentions.

---

## 🏗️ 5-Agent ADK Orchestration Architecture

```mermaid
flowchart TD
    User([User Research Question]) --> Planner[1. Planner Agent]
    Planner -->|Sub-Questions & Evidence Criteria| Retriever[2. Retriever Agent]
    
    subgraph MultiStrategyRAG ["Active Retrieval Engine"]
        Retriever -->|Query| Strategy{Strategy Factory}
        Strategy -->|A| Dense[Naive Dense Top-k]
        Strategy -->|B| Hybrid[Hybrid BM25 + Dense RRF]
        Strategy -->|C| Rerank[Hybrid + Cross-Encoder Rerank]
    end
    
    MultiStrategyRAG -->|Chunks + chunk_ids| Loop
    
    subgraph Loop ["Refinement Loop (LoopAgent, Max 3 Iterations)"]
        Writer[3. Writer Agent] -->|Draft with bracketed citations| Verifier[4. Verifier Agent]
        Verifier -->|get_chunk(chunk_id) Verification| Verdict{Approved?}
        Verdict -->|No: Critique Feedback| Writer
    end
    
    Verdict -->|Yes / Max Loops| Editor[5. Editor Agent]
    Editor -->|Reconcile Contradictions + References Index| Report([Final Traceable Literature Review])
```

### Agent Responsibilities
1. **Planner Agent**: Decomposes the research question into 3–5 focused sub-questions and identifies required evidence types (empirical, mathematical, benchmark comparison, failure modes).
2. **Retriever Agent**: Executes search queries via the `search_documents(query, k)` tool using the active retrieval strategy.
3. **Writer Agent**: Drafts sections. **Constraint**: Every factual assertion must cite a source chunk `[chunk_id]` (e.g. `[dpo:p04_c02]`). Uncited claims are strictly forbidden.
4. **Verifier Agent**: Inspects each cited claim by pulling raw chunk text via `get_chunk(chunk_id)`. Verifies support (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`) and loops back revision feedback if ungrounded claims exist.
5. **Editor Agent**: Synthesizes verified sections, explicitly highlights cross-paper contradictions, and appends a complete References Index mapping `[chunk_id]` to paper title, page number, and section.

---

## 🚀 Quickstart & Setup

### 1. Installation & Environment
```bash
# Clone repository
git clone https://github.com/your-username/Literature_review.git
cd Literature_review

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Keys (Optional for local UI / Required for LLM Agents)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to include your `GEMINI_API_KEY`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Ingest Paper Corpus
Download 25 landmark RLHF/Alignment papers from arXiv and build persistent ChromaDB & BM25 indices:
```bash
python scripts/ingest.py
```

### 4. Run Retrieval Benchmark
Evaluate all 3 retrieval strategies over the hand-labeled test set (`data/eval_set.json`) and generate `results/eval_report.md`:
```bash
python eval/run_eval.py
```

### 5. Launch Interactive Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Repository Structure

```
Literature_review/
├── README.md                          # Full architecture diagrams & benchmark comparison
├── requirements.txt                   # Dependency manifest
├── config.py                          # Global configuration, model selection, paths & pricing
├── data/
│   ├── papers/                        # Downloaded 25 landmark PDF papers
│   ├── chroma_db/                     # Persistent Chroma vector store
│   ├── bm25_index.pkl                 # Serialized BM25 index
│   └── eval_set.json                  # Hand-labeled test set (20 queries, expected chunk IDs, gold answers)
├── src/
│   ├── ingestion/                     # PDF extraction, paragraph-aligned chunking & index building
│   ├── retrieval/                     # Naive Dense, Hybrid RRF, Cross-Encoder reranker & factory
│   ├── agents/                        # Planner, Retriever, Writer, Verifier, Editor & LiteratureReviewPipeline
│   ├── observability/                 # Structured JSON trace logger & latency/cost estimator
│   └── utils/                         # Citation parsing & text sanitization helpers
├── eval/
│   ├── metrics.py                     # Precision@k, Recall@k, MRR, Hit Rate implementations
│   ├── llm_judge.py                   # LLM-as-a-Judge for claim faithfulness and relevance
│   ├── run_eval.py                    # Benchmark runner across all strategies
│   └── generate_report.py             # Markdown evaluation report generator
├── results/
│   └── eval_report.md                 # Generated empirical evaluation report
├── scripts/
│   ├── download_corpus.py             # Curated arXiv PDF downloader
│   ├── ingest.py                      # Master ingestion runner
│   ├── test_retrieval_manual.py       # Bare retrieval sanity test script
│   └── baseline_single_agent.py       # Baseline single-agent RAG script
└── app/
    └── streamlit_app.py               # Streamlit application with live agent trace viewer & citation modal
```

---

## 💡 What I'd Improve with More Time (Production Roadmap)

1. **Hierarchical Parent-Child Chunking**: Store small child chunks (~150 tokens) for precise vector/BM25 retrieval, but return larger parent section chunks (~800 tokens) to the Writer Agent for richer context.
2. **HyDE (Hypothetical Document Embeddings)**: Have the Planner Agent generate hypothetical passage answers to align query vector space closer to academic paper formulations.
3. **GPU-Accelerated Cross-Encoder Reranking**: Batch Cross-Encoder inference asynchronously or deploy as a vLLM/TRT-LLM sidecar service to reduce reranking latency from 2.7s down to <200ms.
4. **Dynamic Graph Workflow**: Transition the SequentialAgent/LoopAgent pipeline definition to ADK 2.0 Graph Runtimes to enable dynamic branching when heavy contradictions are flagged by the Verifier.
