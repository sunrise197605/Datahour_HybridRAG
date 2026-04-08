# Hybrid RAG System

A Retrieval-Augmented Generation system that combines dense (semantic) and sparse (BM25) retrieval using Reciprocal Rank Fusion, over 500 Wikipedia articles in the travel domain.

---

## Task-wise Python File Mapping

All logic is in `.py` files under `src/` and the project root.

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
| End-to-end pipeline (retrieve → fuse → generate) | `src/rag/pipeline.py` |
| Reciprocal Rank Fusion (RRF) | `src/retrieval/rrf.py` |
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

| What it does | Python File |
|---|---|
| Streamlit web UI for interactive Q&A | `app/streamlit_app.py` |
| Default hyperparameters for pipeline | `src/config.py` |

---

## Project Structure

```
Group_114_Hybrid_RAG/
├── app/
│   └── streamlit_app.py          # Streamlit web interface
├── data/
│   ├── urls/                     # 500 Wikipedia URLs (200 fixed + 300 random)
│   ├── corpus/chunks.jsonl       # 9,083 text chunks
│   └── eval/                     # Evaluation questions and results
├── indexes/
│   ├── dense/                    # FAISS index + embeddings
│   └── bm25/                     # BM25 model + tokenized docs
├── src/
│   ├── config.py                 # Hyperparameter defaults
│   ├── types.py                  # Chunk, RetrievedChunk, RAGAnswer
│   ├── ingestion/                # Data collection & corpus building
│   │   ├── fetch_wikipedia.py
│   │   ├── build_random_urls.py
│   │   ├── build_corpus.py
│   │   └── sample_urls.py
│   ├── retrieval/                # Dense, BM25, and RRF fusion
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   └── rrf.py
│   ├── generation/               # LLM answer generation
│   │   ├── llm.py                # Flan-T5 (main)
│   │   ├── prompt.py
│   │   └── mistral_chat.py       # Mistral-7B (judge only)
│   ├── rag/
│   │   └── pipeline.py           # HybridRAG end-to-end pipeline
│   ├── evaluation/               # All evaluation logic
│   │   ├── metrics.py            # MRR, HitRate, CSFS, CUS, ACS
│   │   ├── run_eval.py
│   │   ├── ablation.py
│   │   ├── llm_judge.py
│   │   ├── question_gen.py
│   │   └── report.py
│   └── utils/                    # Shared helpers
│       ├── chunking.py
│       ├── text_cleaning.py
│       └── io.py
├── README.md                    # Project overview, setup, and file mapping
├── REPORT.md                    # Detailed project report
├── requirements.txt
├── run_evaluation.py             # Main script to reproduce all results
└── test_rag_pipeline.py          # Quick smoke-test (5 queries)
```

---

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# 1. Quick test — runs 5 sample queries and prints answers
python test_rag_pipeline.py

# 2. Full evaluation — computes MRR, HitRate, CSFS, CUS, ACS + ablation study
python run_evaluation.py

# 3. Web UI — interactive question answering with source viewer
streamlit run app/streamlit_app.py
```

## Rebuilding from Scratch

Pre-built indexes and corpus are included in `data/` and `indexes/`. To rebuild:

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
              Top-6 chunks as context
                        │
                        ▼
              Flan-T5-base generates answer
```

---

## Custom Metrics (Our Contribution)

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
| Avg Latency | 1600 ms |
| Questions evaluated | 100 |

### Ablation Study

| Method | MRR | HitRate@5 |
|--------|-----|-----------|
| Hybrid (Dense + BM25 + RRF) | 0.80 | 0.93 |
| Dense only | 0.85 | 0.93 |
| Sparse only | 0.74 | 0.92 |

---

## Documents

| Document | Description |
|----------|-------------|
| `REPORT.md` | Full project report — architecture, dataset, evaluation, results, error analysis |

---

## Notes

- Dense index building takes ~1 hour on CPU; pre-built indexes are included
- Streamlit caches models after first load for fast subsequent queries
- Question generation and LLM-judge require GPU + Mistral-7B-Instruct
- All other components (pipeline, evaluation, UI) run on CPU
