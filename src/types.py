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
    chunk_id: str
    url: str
    title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


@dataclass(frozen=True)
class RAGAnswer:
    query: str
    answer: str
    context_chunks: List[RetrievedChunk]
    latency_ms: Dict[str, float]
    debug: Dict[str, Any]
