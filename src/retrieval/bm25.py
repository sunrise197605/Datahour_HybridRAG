"""
Sparse (keyword) retrieval using the BM25 Okapi algorithm.
Tokenizes each chunk into lowercased words, builds a BM25Okapi model from
rank_bm25, and scores queries by term-frequency / inverse-document-frequency.
Complements the dense index — BM25 excels at exact keyword matches while
dense retrieval captures semantic meaning.
Usage: python -m src.retrieval.bm25 --chunks data/corpus/chunks.jsonl --out indexes/bm25
"""

import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.types import Chunk
from src.utils.io import read_jsonl, write_json


def simple_tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.model = None  # BM25Okapi instance, built in build() or load()
        self.tokenized_documents: List[List[str]] = []
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_documents = [simple_tokenize(c.text) for c in chunks]
        self.model = BM25Okapi(self.tokenized_documents, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("BM25Index is not built. Call build() first.")
        query_tokens = simple_tokenize(query)
        scores = np.asarray(self.model.get_scores(query_tokens), dtype=np.float32)
        ranked = np.argsort(-scores)[:top_k]
        return ranked, scores[ranked]

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        meta = {
            "k1": self.k1,
            "b": self.b,
            "chunk_count": len(self.chunks),
        }
        write_json(meta, os.path.join(output_dir, "bm25_meta.json"))
        with open(os.path.join(output_dir, "bm25_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(output_dir, "tokenized_docs.pkl"), "wb") as f:
            pickle.dump(self.tokenized_documents, f)

    @classmethod
    def load(cls, chunks: List[Chunk], input_dir: str) -> "BM25Index":
        import json
        meta_path = os.path.join(input_dir, "bm25_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(k1=meta["k1"], b=meta["b"])
        instance.chunks = chunks
        with open(os.path.join(input_dir, "bm25_model.pkl"), "rb") as f:
            instance.model = pickle.load(f)
        with open(os.path.join(input_dir, "tokenized_docs.pkl"), "rb") as f:
            instance.tokenized_documents = pickle.load(f)
        return instance


def _load_chunks_from_jsonl(path: str) -> List[Chunk]:
    records = read_jsonl(path)
    return [Chunk(**r) for r in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--out", required=True, help="Output directory for BM25 index")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    chunks = _load_chunks_from_jsonl(args.chunks)
    bm25_index = BM25Index(k1=args.k1, b=args.b)
    bm25_index.build(chunks)
    bm25_index.save(args.out)
    print(f"BM25 index saved to {args.out}")


if __name__ == "__main__":
    main()
