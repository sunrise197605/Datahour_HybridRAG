"""
Configuration dataclasses for the Hybrid RAG system.
Defines default hyperparameters for chunking (token window sizes), retrieval
(top-K, RRF constant), and generation (model name, max tokens, device).
Modify these values to experiment with different pipeline settings.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    min_tokens: int = 200
    max_tokens: int = 400
    overlap_tokens: int = 50


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 50
    top_n_context: int = 6
    rrf_k: int = 60


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 128
    device: str = "cpu"
