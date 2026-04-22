"""
Core data structures used throughout the RAG pipeline.
- Chunk: a piece of a Wikipedia article (id, url, title, text).
- RetrievedChunk: a Chunk along with its dense, BM25, and RRF scores/ranks.
- RAGAnswer: the final output containing the query, generated answer, retrieved
  context chunks, latency breakdown, and any debug info.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Chunk:
    """A single passage of source text with its provenance metadata."""
    chunk_id: str
    url: str
    title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk paired with every per-stage score and rank it earned.

    Carries the dense and BM25 raw scores and 1-based ranks, the fused
    RRF score, and (if the cross-encoder ran) the rerank score and rank.
    Keeping all stages on one object makes explainability trivial: the UI
    simply prints the four numbers side by side for each source.
    """
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None


@dataclass(frozen=True)
class RAGAnswer:
    """The final envelope returned by HybridRAG.answer().

    Bundles the original question, the generated answer, the context
    chunks used to produce it (for citation and audit), a latency
    breakdown across retrieve / generate / total, and a free-form debug
    dict for anything the pipeline wants to surface to the caller.
    """
    query: str
    answer: str
    context_chunks: List[RetrievedChunk]
    latency_ms: Dict[str, float]
    debug: Dict[str, Any]
