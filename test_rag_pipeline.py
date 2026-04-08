"""
Quick smoke-test for the Hybrid RAG pipeline.
Loads all components (chunks, Dense index, BM25 index, Flan-T5 LLM),
runs 5 sample travel queries, and prints the answers with their top-3
source documents and latency. Useful for verifying the setup works before
running the full evaluation.
Usage: python test_rag_pipeline.py
"""

import json
from src.generation.llm import LLMGenerator
from src.rag.pipeline import HybridRAG
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex
from src.types import Chunk
from src.utils.io import read_jsonl


def load_chunks(path):
    records = read_jsonl(path)
    return [Chunk(**r) for r in records]


def main():
    print("=" * 60)
    print("HYBRID RAG PIPELINE TEST")
    print("=" * 60)
    print()

    # Load components
    print("[1/4] Loading chunks...")
    chunks = load_chunks("data/corpus/chunks.jsonl")
    print(f"      Loaded {len(chunks)} chunks")

    print("[2/4] Loading Dense index...")
    dense_index = DenseIndex.load(chunks, "indexes/dense")

    print("[3/4] Loading BM25 index...")
    bm25_index = BM25Index.load(chunks, "indexes/bm25")

    print("[4/4] Loading LLM...")
    generator = LLMGenerator(model_name="google/flan-t5-base", device="cpu")
    generator.load()

    rag = HybridRAG(
        chunks=chunks,
        dense_index=dense_index,
        bm25_index=bm25_index,
        generator=generator,
        rrf_constant=60,
    )

    print()
    print("=" * 60)
    print("SAMPLE QUERIES")
    print("=" * 60)

    queries = [
        "When was the Eiffel Tower built?",
        "What is the height of Burj Khalifa?",
        "Which country is Machu Picchu located in?",
        "What is the Great Barrier Reef made of?",
        "Who designed the Sagrada Familia?",
    ]

    for i, query in enumerate(queries, 1):
        print()
        print(f"Query {i}: {query}")
        print()

        result = rag.answer(query, top_k=50, context_size=5, max_new_tokens=100)

        print(f"Answer: {result.answer}")
        print()
        print("Top 3 Sources:")
        for j, chunk in enumerate(result.context_chunks[:3], 1):
            print(f"  {j}. {chunk.chunk.title}")
            print(f"     Dense Rank: {chunk.dense_rank}, BM25 Rank: {chunk.bm25_rank}")
        print()
        print(f"Latency: {result.latency_ms['total']:.0f}ms")
        print("-" * 60)

    print()
    print("TEST COMPLETE")


if __name__ == "__main__":
    main()
