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
