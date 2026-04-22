"""
Cross-encoder re-ranker. Takes the fused candidate list from RRF and re-scores
each (query, chunk) pair jointly using a cross-encoder model. Cross-encoders
are slower than bi-encoders but much more accurate because they attend over
query and document together.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, CPU-friendly,
trained on MS MARCO passage ranking).
"""

from dataclasses import replace
from typing import List

from src.types import RetrievedChunk


class CrossEncoderReranker:
    """Second-stage re-ranker using a cross-encoder model.

    A bi-encoder dense retriever and BM25 score queries and documents
    separately, so they are fast but can miss subtle relevance. A
    cross-encoder instead concatenates the query and the candidate chunk
    and runs full self-attention across both, producing a much sharper
    relevance score at the cost of being too slow to score the whole
    corpus. The Hybrid RAG pipeline takes advantage of both: fuse the
    dense and BM25 top-K into roughly 25 candidates, then let this
    cross-encoder rescore every (query, chunk) pair and reorder the
    final context window. The default model, cross-encoder/ms-marco-
    MiniLM-L-6-v2, is small enough to run on CPU and trained on the
    MS MARCO passage-ranking task, which makes it a strong default for
    open-domain question answering.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []
        self.load()

        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self._model.predict(pairs)

        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)

        out: List[RetrievedChunk] = []
        for rank, (cand, score) in enumerate(scored[:top_n], start=1):
            out.append(replace(cand, rerank_score=score, rerank_rank=rank))
        return out
