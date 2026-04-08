"""
End-to-end Hybrid RAG pipeline — the heart of the system.
1. Retrieves top-K chunks from BOTH Dense and BM25 indexes.
2. Fuses the two ranked lists using Reciprocal Rank Fusion (RRF).
3. Selects the top-N chunks by RRF score as context for the LLM.
4. Builds a prompt and generates an answer with Flan-T5.
Returns a RAGAnswer with the answer text, source chunks, and latency breakdown.
"""

import time
from typing import Dict, List

import numpy as np

from src.generation.llm import LLMGenerator
from src.generation.prompt import build_prompt
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex
from src.retrieval.rrf import reciprocal_rank_fusion, top_n_by_score
from src.types import Chunk, RAGAnswer, RetrievedChunk


class HybridRAG:
    def __init__(
        self,
        chunks: List[Chunk],
        dense_index: DenseIndex,
        bm25_index: BM25Index,
        generator: LLMGenerator,
        rrf_constant: int = 60,
    ):
        self.chunks = chunks
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.generator = generator
        self.rrf_constant = rrf_constant

    def build(self) -> None:
        self.dense_index.build(self.chunks)
        self.bm25_index.build(self.chunks)
        self.generator.load()

    def retrieve(self, query: str, top_k: int = 50, context_size: int = 6) -> List[RetrievedChunk]:
        dense_indices, dense_scores = self.dense_index.search(query, top_k=top_k)
        sparse_indices, sparse_scores = self.bm25_index.search(query, top_k=top_k)

        fused_scores = reciprocal_rank_fusion(
            dense_ranked_indices=dense_indices,
            sparse_ranked_indices=sparse_indices,
            fusion_constant=self.rrf_constant,
            limit=top_k,
        )
        fused_top = top_n_by_score(fused_scores, top_n=context_size)

        dense_rank_lookup = {int(doc_index): rank + 1 for rank, doc_index in enumerate(dense_indices)}
        sparse_rank_lookup = {int(doc_index): rank + 1 for rank, doc_index in enumerate(sparse_indices)}
        dense_score_lookup = {int(doc_index): float(score) for doc_index, score in zip(dense_indices, dense_scores)}
        sparse_score_lookup = {int(doc_index): float(score) for doc_index, score in zip(sparse_indices, sparse_scores)}

        retrieved_chunks: List[RetrievedChunk] = []
        for doc_index, rrf_score in fused_top:
            chunk = self.chunks[int(doc_index)]
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=dense_score_lookup.get(int(doc_index), 0.0),
                    bm25_score=sparse_score_lookup.get(int(doc_index), 0.0),
                    rrf_score=float(rrf_score),
                    dense_rank=dense_rank_lookup.get(int(doc_index)),
                    bm25_rank=sparse_rank_lookup.get(int(doc_index)),
                )
            )
        return retrieved_chunks

    def answer(self, query: str, top_k: int = 50, context_size: int = 6, max_new_tokens: int = 128) -> RAGAnswer:
        overall_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        context_chunks = self.retrieve(query, top_k=top_k, context_size=context_size)
        retrieval_end = time.perf_counter()

        prompt = build_prompt(query=query, context_chunks=context_chunks)

        generation_start = time.perf_counter()
        answer_text = self.generator.generate(prompt, max_new_tokens=max_new_tokens)
        generation_end = time.perf_counter()

        overall_end = time.perf_counter()

        latency_ms: Dict[str, float] = {
            "retrieve_total": (retrieval_end - retrieval_start) * 1000.0,
            "generate": (generation_end - generation_start) * 1000.0,
            "total": (overall_end - overall_start) * 1000.0,
        }

        return RAGAnswer(
            query=query,
            answer=answer_text,
            context_chunks=context_chunks,
            latency_ms=latency_ms,
            debug={},
        )
