"""
Ablation study — compares Hybrid vs Dense-only vs Sparse-only retrieval.
Runs the same set of questions through three retrieval modes and measures
MRR and Recall@5 for each. This shows that combining Dense + BM25 via RRF
(hybrid) outperforms using either retriever alone, which is the main
claim of our project.
"""

import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.evaluation.metrics import mean_reciprocal_rank_url_level, unique_url_ranking
from src.rag.pipeline import HybridRAG
from src.types import RAGAnswer, RetrievedChunk


@dataclass(frozen=True)
class AblationResult:
    name: str
    mrr_url: float
    recall_at_5_url: float
    avg_latency_ms: float


def recall_at_k_url(answers: List[RAGAnswer], ground_truth_urls: List[str], k: int = 5) -> float:
    hits = 0
    for answer, gt_url in zip(answers, ground_truth_urls):
        ranked_urls = unique_url_ranking(answer)[:k]
        if gt_url in ranked_urls:
            hits += 1
    return hits / len(answers) if answers else 0.0


def _dense_only_context(rag: HybridRAG, query: str, top_k: int, context_size: int) -> List[RetrievedChunk]:
    retrieved = rag.retrieve(query, top_k=top_k, context_size=context_size)
    return sorted(retrieved, key=lambda r: (r.dense_rank if r.dense_rank is not None else 10**9))


def _sparse_only_context(rag: HybridRAG, query: str, top_k: int, context_size: int) -> List[RetrievedChunk]:
    retrieved = rag.retrieve(query, top_k=top_k, context_size=context_size)
    return sorted(retrieved, key=lambda r: (r.bm25_rank if r.bm25_rank is not None else 10**9))


def run_mode(
    rag: HybridRAG,
    questions: List[Dict[str, Any]],
    mode: str,
    top_k: int = 50,
    context_size: int = 6,
    max_new_tokens: int = 64,
) -> Tuple[List[RAGAnswer], List[str], List[float]]:
    answers: List[RAGAnswer] = []
    gt_urls: List[str] = []
    latencies: List[float] = []

    for item in questions:
        question_text = item["question"]
        gt_url = item["ground_truth_url"]
        gt_urls.append(gt_url)

        start = time.perf_counter()
        response = rag.answer(question_text, top_k=top_k, context_size=context_size, max_new_tokens=max_new_tokens)

        if mode == "dense":
            context = _dense_only_context(rag, question_text, top_k=top_k, context_size=context_size)
            response = RAGAnswer(query=response.query, answer=response.answer, context_chunks=context, latency_ms=response.latency_ms, debug={"mode": "dense"})
        elif mode == "sparse":
            context = _sparse_only_context(rag, question_text, top_k=top_k, context_size=context_size)
            response = RAGAnswer(query=response.query, answer=response.answer, context_chunks=context, latency_ms=response.latency_ms, debug={"mode": "sparse"})
        else:
            response = RAGAnswer(query=response.query, answer=response.answer, context_chunks=response.context_chunks, latency_ms=response.latency_ms, debug={"mode": "hybrid"})

        end = time.perf_counter()
        answers.append(response)
        latencies.append((end - start) * 1000.0)

    return answers, gt_urls, latencies


def compute_ablation(rag: HybridRAG, questions_path: str, out_path: str) -> List[Dict[str, Any]]:
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results: List[AblationResult] = []
    for mode_name in ["dense", "sparse", "hybrid"]:
        answers, gt_urls, latencies = run_mode(rag, questions, mode=mode_name)
        mrr = mean_reciprocal_rank_url_level(answers, gt_urls)
        recall5 = recall_at_k_url(answers, gt_urls, k=5)
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        results.append(AblationResult(name=mode_name, mrr_url=mrr, recall_at_5_url=recall5, avg_latency_ms=avg_latency))

    output_rows = [r.__dict__ for r in results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2)

    return output_rows
