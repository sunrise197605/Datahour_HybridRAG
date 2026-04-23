# Hybrid RAG with Re-Ranker

A Retrieval-Augmented Generation system that combines dense (semantic) and sparse (BM25) retrieval with Reciprocal Rank Fusion, plus a **cross-encoder re-ranker** as a precision stage, over 500 Wikipedia articles in the travel domain.

Built for the Analytics Vidhya **DataHour** session *"Hybrid RAG with Re-Ranker"*.

---

## Task-wise Python File Mapping

All logic is under `src/` plus two root-level driver scripts.

### Task 1 — Data Collection

| What it does | Python File |
|---|---|
| Fetch & clean Wikipedia pages (HTML → text) | `src/ingestion/fetch_wikipedia.py` |
| Build 300 random URLs via Wikipedia API | `src/ingestion/build_random_urls.py` |
| Merge fixed + random URL lists | `src/ingestion/sample_urls.py` |
| Fetch all URLs → chunk → save corpus | `src/ingestion/build_corpus.py` |
| Text chunking (200–400 token overlapping windows) | `src/utils/chunking.py` |
| Text cleaning (whitespace, citation removal) | `src/utils/text_cleaning.py` |
| JSON / JSONL file I/O helpers | `src/utils/io.py` |

### Task 2 — Index Building

| What it does | Python File |
|---|---|
| Dense index (SentenceTransformers + FAISS) | `src/retrieval/dense.py` |
| BM25 sparse index (Okapi BM25, k1=1.5, b=0.75) | `src/retrieval/bm25.py` |
| Data types (Chunk, RetrievedChunk, RAGAnswer) | `src/types.py` |

### Task 3 — Hybrid RAG Pipeline

| What it does | Python File |
|---|---|
| End-to-end pipeline (retrieve → fuse → re-rank → generate) | `src/rag/pipeline.py` |
| Reciprocal Rank Fusion (RRF) | `src/retrieval/rrf.py` |
| **Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)** | **`src/retrieval/reranker.py`** |
| LLM answer generation (Flan-T5-base) | `src/generation/llm.py` |
| Prompt template builder | `src/generation/prompt.py` |
| Mistral-7B wrapper (for LLM-judge, optional) | `src/generation/mistral_chat.py` |
| Standalone test script (5 sample queries) | `test_rag_pipeline.py` |

### Task 4 — Evaluation

| What it does | Python File |
|---|---|
| **Main evaluation script (run this to reproduce)** | **`run_evaluation.py`** |
| Metrics: MRR, HitRate@K, CSFS, CUS, ACS | `src/evaluation/metrics.py` |
| Evaluation runner with LLM-judge support | `src/evaluation/run_eval.py` |
| Ablation study (Hybrid vs Dense-only vs Sparse-only) | `src/evaluation/ablation.py` |
| LLM-as-Judge evaluation (Mistral) | `src/evaluation/llm_judge.py` |
| Automated question generation (Mistral) | `src/evaluation/question_gen.py` |
| Report table generation | `src/evaluation/report.py` |

### Additional

| What it does | File |
|---|---|
| Streamlit web UI (reranker toggle, source explainability) | `app/streamlit_app.py` |
| Self-contained demo notebook (no `src/` imports) | `notebooks/HybridRAG_With_Reranker.ipynb` |
| Same notebook with pre-rendered outputs | `notebooks/HybridRAG_With_Reranker_executed.ipynb` |
| Default hyperparameters for pipeline | `src/config.py` |

---

## Project Structure

```
Datahour_HybridRAG/
├── app/
│   └── streamlit_app.py              # Streamlit web interface
├── assets/
│   ├── AnalyticsVidya.png            # Title-slide background
│   ├── vriksha-logo.png              # Speaker logo
│   └── screenshots/                  # UI screenshots for the deck
├── data/
│   ├── urls/                         # 500 Wikipedia URLs (200 fixed + 300 random)
│   ├── corpus/chunks.jsonl           # 9,083 text chunks
│   └── eval/                         # Evaluation questions and results
├── docs/
│   ├── REPORT.md                     # Full project report
│   ├── Report.pdf                    # PDF version of the report
│   ├── DataHour_HybridRAG_With_ReRanker.pptx  # 30-slide session deck
│   └── DataHour_HybridRAG_Worked_Problems.pdf  # Audience handout with worked metric problems
├── indexes/
│   ├── dense/                        # FAISS index + embeddings
│   └── bm25/                         # BM25 model + tokenized docs
├── src/
│   ├── config.py                     # Hyperparameter defaults
│   ├── types.py                      # Chunk, RetrievedChunk, RAGAnswer
│   ├── ingestion/                    # Data collection & corpus building
│   ├── retrieval/
│   │   ├── dense.py                  # all-mpnet-base-v2 + FAISS
│   │   ├── bm25.py                   # Okapi BM25
│   │   ├── rrf.py                    # Reciprocal Rank Fusion
│   │   └── reranker.py               # Cross-encoder re-ranker
│   ├── generation/
│   │   ├── llm.py                    # Flan-T5-base (main generator)
│   │   ├── prompt.py
│   │   └── mistral_chat.py           # Mistral-7B (judge only)
│   ├── rag/
│   │   └── pipeline.py               # HybridRAG end-to-end pipeline
│   ├── evaluation/                   # All evaluation logic
│   └── utils/                        # chunking, text_cleaning, io
├── notebooks/
│   ├── HybridRAG_With_Reranker.ipynb           # Self-contained demo notebook
│   └── HybridRAG_With_Reranker_executed.ipynb  # Notebook with rendered outputs
├── README.md                         # This file
├── requirements.txt
├── run_evaluation.py                 # Main script to reproduce all results
└── test_rag_pipeline.py              # Quick smoke-test (5 queries)
```

---

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# 1. Quick test — 5 sample queries, prints answers + top sources
python test_rag_pipeline.py

# 2. Full evaluation — computes MRR, HitRate, CSFS, CUS, ACS + ablation
python run_evaluation.py

# 3. Web UI — interactive Q&A with re-ranker toggle
streamlit run app/streamlit_app.py

# 4. Self-contained notebook (demo fallback)
jupyter lab notebooks/HybridRAG_With_Reranker.ipynb
```

## Rebuilding from Scratch

Pre-built indexes and corpus are included. To rebuild:

```bash
# Step 1: Build corpus from URLs
python -m src.ingestion.build_corpus --urls data/urls/all_urls.json --out data/corpus/chunks.jsonl

# Step 2: Build dense index (~1 hour on CPU)
python -m src.retrieval.dense --chunks data/corpus/chunks.jsonl --out indexes/dense

# Step 3: Build BM25 index
python -m src.retrieval.bm25 --chunks data/corpus/chunks.jsonl --out indexes/bm25

# Step 4: Generate evaluation questions (requires GPU + Mistral-7B)
python -m src.evaluation.question_gen --chunks data/corpus/chunks.jsonl --out data/eval/questions.json
```

---

## How It Works

```
Query
  ├──► Dense Index (all-mpnet-base-v2 + FAISS) ──► Top-50 by cosine similarity
  └──► BM25 Index (Okapi BM25)                 ──► Top-50 by term frequency
                        │
                        ▼
              RRF Fusion (k=60)
           Score(doc) = 1/(60+rank_dense) + 1/(60+rank_bm25)
                        │
                        ▼
              Top-25 candidate pool
                        │
                        ▼
              Cross-Encoder Re-Ranker
              (cross-encoder/ms-marco-MiniLM-L-6-v2)
              Joint (query, chunk) scoring
                        │
                        ▼
              Top-6 chunks as context
                        │
                        ▼
              Flan-T5-base generates grounded answer with citations
```

The reranker is optional but on by default — toggle it in the Streamlit sidebar or pass `use_reranker=False` to `HybridRAG.answer(...)` to compare.

---

## Custom Metrics

Standard metrics like MRR only measure retrieval quality, not answer quality. We built two custom metrics in `src/evaluation/metrics.py`:

**CUS (Context Utilization Score)** — Does the answer actually use the retrieved context?
- Combines semantic similarity (55%) + word-level containment (45%)
- Catches cases where the LLM ignores context and hallucinates from memory

**ACS (Answer Completeness Score)** — Does the answer address the question?
- Semantic similarity between question and answer + factual content detection
- Penalizes evasive non-answers ("I don't know", "no information")
- Does not penalize short factual answers (e.g., "828 metres" is valid)

---

## Results

| Metric | Value |
|--------|-------|
| MRR (URL-level) | 0.7978 |
| HitRate@5 | 0.9300 |
| HitRate@10 | 0.9300 |
| CSFS (Faithfulness) | 0.0122 |
| **CUS (Context Utilization)** | **0.7208** |
| **ACS (Answer Completeness)** | **0.6368** |
| Avg Latency (no reranker) | 1600 ms |
| Questions evaluated | 100 |

### Ablation Study

| Method | MRR | HitRate@5 |
|--------|-----|-----------|
| Hybrid (Dense + BM25 + RRF) | 0.80 | 0.93 |
| Dense only | 0.85 | 0.93 |
| Sparse only | 0.74 | 0.92 |

### Re-Ranker Impact

On the Burj Khalifa compound query the cross-encoder lifted the correct chunk from BM25 rank 12 / RRF rank ~5 to **final position 1** (rerank score 3.41), with about 300–800 ms added latency per query on CPU after model warm-up.

---

## Documents

| Document | Location | Description |
|----------|----------|-------------|
| Full project report | `docs/REPORT.md` / `docs/Report.pdf` | Architecture, dataset, evaluation, results, error analysis |
| Worked problems handout | `docs/DataHour_HybridRAG_Worked_Problems.pdf` | Audience-facing numerical problems on MRR, HitRate, RRF, and the re-ranker |
| Session deck | `docs/DataHour_HybridRAG_With_ReRanker.pptx` | 30-slide deck for the DataHour session |

---

## Notes

- Dense index building takes ~1 hour on CPU; pre-built indexes are included
- Cross-encoder model (~80 MB) downloads on first reranker call, then cached
- Streamlit caches models after first load for fast subsequent queries
- Question generation and LLM-judge require GPU + Mistral-7B-Instruct
- All other components (pipeline, evaluation, UI, notebook) run on CPU
