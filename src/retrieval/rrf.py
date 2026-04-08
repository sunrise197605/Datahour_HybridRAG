"""
Reciprocal Rank Fusion (RRF) — combines rankings from multiple retrievers.
Given ranked lists from Dense and BM25, each document gets a fused score:
  RRF(doc) = 1/(k + rank_dense) + 1/(k + rank_bm25)  where k=60 by default.
This is the key technique that makes our hybrid retrieval outperform either
retriever alone. Documents that rank high in both lists get the best scores.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def reciprocal_rank_fusion(
    dense_ranked_indices: np.ndarray,
    sparse_ranked_indices: np.ndarray,
    fusion_constant: int = 60,
    limit: Optional[int] = None,
) -> Dict[int, float]:
    """
    Computes Reciprocal Rank Fusion scores.

    Score(doc) = sum over lists i of 1 / (fusion_constant + rank_i(doc))

    Ranks are 1-based.
    """
    if limit is not None:
        dense_ranked_indices = dense_ranked_indices[:limit]
        sparse_ranked_indices = sparse_ranked_indices[:limit]

    fused_scores: Dict[int, float] = {}

    for rank, doc_idx in enumerate(dense_ranked_indices, start=1):
        doc_idx = int(doc_idx)
        fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (fusion_constant + rank)

    for rank, doc_idx in enumerate(sparse_ranked_indices, start=1):
        doc_idx = int(doc_idx)
        fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (fusion_constant + rank)

    return fused_scores


def top_n_by_score(score_map: Dict[int, float], top_n: int) -> List[Tuple[int, float]]:
    return sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:top_n]
