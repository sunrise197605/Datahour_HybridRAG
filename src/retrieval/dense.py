"""
Dense (semantic) retrieval using SentenceTransformers + FAISS.
Encodes every chunk into a 768-dim vector with all-mpnet-base-v2, stores them
in a FAISS IndexFlatIP for fast cosine-similarity search. At query time the
query is encoded the same way and the top-K nearest chunks are returned.
Usage: python -m src.retrieval.dense --chunks data/corpus/chunks.jsonl --out indexes/dense
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

from src.types import Chunk
from src.utils.io import read_jsonl, write_json


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


class DenseIndex:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.encoder = None    # SentenceTransformer, loaded in build() or load()
        self.embeddings = None # numpy array of shape (num_chunks, 768)
        self.chunks: List[Chunk] = []
        self.faiss_index = None

    def build(self, chunks: List[Chunk], use_faiss: bool = True) -> None:
        self.chunks = chunks
        self.encoder = SentenceTransformer(self.model_name)

        texts = [c.text for c in chunks]
        vectors = self.encoder.encode(texts, batch_size=64, show_progress_bar=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors = _normalize_matrix(vectors)

        self.embeddings = vectors

        if use_faiss and faiss is not None:
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self.faiss_index = index

    def search(self, query: str, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if self.encoder is None or self.embeddings is None:
            raise RuntimeError("DenseIndex is not built. Call build() first.")

        query_vector = self.encoder.encode([query])
        query_vector = np.asarray(query_vector, dtype=np.float32)
        query_vector = _normalize_matrix(query_vector)

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vector, top_k)
            return indices[0], scores[0]

        # Fallback: cosine similarity via dot product on normalized embeddings
        scores = np.dot(self.embeddings, query_vector[0])
        ranked = np.argsort(-scores)[:top_k]
        return ranked, scores[ranked]

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "chunk_count": len(self.chunks),
        }
        write_json(meta, os.path.join(output_dir, "dense_meta.json"))
        np.save(os.path.join(output_dir, "embeddings.npy"), self.embeddings)

        if self.faiss_index is not None and faiss is not None:
            faiss.write_index(self.faiss_index, os.path.join(output_dir, "faiss.index"))

    @classmethod
    def load(cls, chunks: List[Chunk], input_dir: str) -> "DenseIndex":
        meta_path = os.path.join(input_dir, "dense_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            import json
            meta = json.load(f)

        instance = cls(model_name=meta["model_name"])
        instance.chunks = chunks
        instance.encoder = SentenceTransformer(instance.model_name)
        instance.embeddings = np.load(os.path.join(input_dir, "embeddings.npy"))

        if faiss is not None:
            index_path = os.path.join(input_dir, "faiss.index")
            if os.path.exists(index_path):
                instance.faiss_index = faiss.read_index(index_path)

        return instance


def _load_chunks_from_jsonl(path: str) -> List[Chunk]:
    records = read_jsonl(path)
    return [Chunk(**r) for r in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--out", required=True, help="Output directory for dense index")
    parser.add_argument("--model", default="all-mpnet-base-v2")
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args()

    chunks = _load_chunks_from_jsonl(args.chunks)
    dense_index = DenseIndex(model_name=args.model)
    dense_index.build(chunks, use_faiss=not args.no_faiss)
    dense_index.save(args.out)
    print(f"Dense index saved to {args.out}")


if __name__ == "__main__":
    main()
