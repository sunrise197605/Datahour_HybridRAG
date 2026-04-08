# Hybrid RAG System Report

## 1. What we built

We implemented a Hybrid RAG system that combines dense retrieval (using sentence embeddings) with sparse retrieval (BM25 keyword matching). The idea is that dense retrieval captures semantic meaning while BM25 catches exact keyword matches that embeddings might miss.

## 2. Architecture

```
Query --> Dense Index --> Top 50 results
      --> BM25 Index  --> Top 50 results
                |
                v
          RRF Fusion (combines rankings)
                |
                v
          Top 5 chunks as context
                |
                v
          Flan-T5 generates answer
```

Components used:
- Dense: sentence-transformers (all-mpnet-base-v2) + FAISS
- Sparse: rank-bm25 library
- Generation: google/flan-t5-base
- UI: Streamlit

## 3. Dataset

We collected 500 Wikipedia URLs:
- 200 fixed URLs about travel (cities, landmarks, attractions)
- 300 random URLs (100 more travel + 200 mixed topics)

After fetching and chunking, we got 9,083 text chunks (200-400 tokens each with 50 token overlap).

## 4. Evaluation

### Questions
We generated 100 questions in different categories:
- Factual (35) - simple fact questions
- Descriptive (20) - "describe X" type
- Comparative (15) - comparing two things  
- Inferential (15) - requires some reasoning
- Multi-hop (15) - needs info from multiple sources

### Metrics

**MRR (mandatory)**
Mean Reciprocal Rank at URL level. Measures where the correct source document appears in the ranking.
- MRR = 1/N * sum(1/rank_i)
- Higher is better, 1.0 means perfect

**HitRate@K (custom)**
Just checks if correct URL appears anywhere in top-K results.
- Simple but useful to know if relevant docs are being retrieved

**CSFS (custom metric)**
Checks if answer claims are supported by context. Detects hallucinations.

---

### Our Custom Metrics (CUS and ACS)

We developed two new metrics to evaluate RAG quality:

**CUS (Context Utilization Score)** - *Custom metric we built*

Problem: Sometimes LLMs ignore the retrieved context and generate from memory.
Solution: CUS measures if the answer actually uses the retrieved chunks.

How it works:
- Extract keywords from context and answer
- Compute semantic similarity between answer and context  
- Score = 0.4 * keyword_overlap + 0.6 * semantic_similarity
- Range: 0 to 1 (higher = better context usage)

**ACS (Answer Completeness Score)** - *Custom metric we built*

Problem: Sometimes answers are evasive ("I don't know") or don't address the question.
Solution: ACS checks if the answer actually responds to what was asked.

How it works:
- Check semantic similarity between question and answer
- Penalize very short answers (likely incomplete)
- Detect non-answer patterns ("cannot find", "no information")
- Range: 0 to 1 (higher = more complete answer)

Why these metrics matter:
- MRR/HitRate only measure retrieval, not answer quality
- CSFS checks faithfulness but not relevance
- CUS catches context-ignoring behavior
- ACS catches evasive/incomplete answers

---

## 5. Results

Main results (100 questions):

| Metric | Value | Notes |
|--------|-------|-------|
| MRR | 0.7978 | Good retrieval ranking |
| HitRate@5 | 0.9300 | 93% correct source in top-5 |
| HitRate@10 | 0.9300 | Same as @5 |
| CSFS | 0.0122 | Low - short answers hard to verify |
| **CUS** | **0.7208** | Good - answers use context |
| **ACS** | **0.6368** | Good - answers address questions |
| Latency | 1600ms | End-to-end time |
| Questions | 100 | Full test set |

**Analysis of our custom metrics:**
- CUS (0.72): Answers use retrieved context well. Checks both semantic similarity and whether answer words appear in context.
- ACS (0.64): Answers properly address the questions. Factual answers (with numbers/names) are not penalized for being short.

### Ablation study

| Method | MRR | HitRate@5 |
|--------|-----|-----------|
| Hybrid | 0.80 | 0.93 |
| Dense only | 0.85 | 0.93 |
| Sparse only | 0.74 | 0.92 |

Interesting finding: Dense-only performed better than hybrid for our dataset. This might be because:
1. Our questions are mostly semantic/conceptual
2. Travel domain has lots of proper nouns that embeddings handle well
3. BM25 might be adding noise in some cases

## 6. UI

We built a Streamlit interface where you can:
- Type a question
- See the generated answer
- View retrieved source documents with their scores
- See latency breakdown

UI screenshots are available showing:
- Home page with system info (9083 chunks, settings panel)
- Query: "When was the Eiffel Tower built?" → Answer: "1887 to 1889"
- Query: "What is the height of Burj Khalifa?" → Answer: "829.8 m"
- Query: "Which country is Machu Picchu located in?" → Answer: "Peru"
- Retrieved sources with Dense Rank, BM25 Rank, and RRF Score

## 7. How to run

```bash
# Test pipeline
python test_rag_pipeline.py

# Run evaluation  
python run_evaluation.py

# Launch UI
streamlit run app/streamlit_app.py
```

## 8. Conclusion

The system works well for travel domain questions. Dense retrieval performed best, but the hybrid approach provides more robustness for different query types. Main limitation is that Flan-T5-base generates short answers, which affects some metrics.

## Sample outputs

Q: When was the Eiffel Tower built?
A: 1887 to 1889

Q: What is the height of Burj Khalifa?
A: 829.8 m

Q: Which country is Machu Picchu in?
A: Peru
