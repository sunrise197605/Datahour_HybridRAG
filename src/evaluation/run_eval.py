"""
Evaluation runner that scores the RAG pipeline on a set of test questions.
For each question it runs the full pipeline, computes CSFS faithfulness,
records per-question results, and optionally uses an LLM judge (Mistral)
to score factual accuracy, completeness, and hallucination. Aggregates
MRR and HitRate@K across all questions.
"""

import json
import time
from typing import Any, Dict, List, Optional

from src.evaluation.llm_judge import judge_one
from src.evaluation.metrics import (
    claim_supported_faithfulness,
    hit_rate_at_k_url_level,
    mean_reciprocal_rank_url_level,
    unique_url_ranking,
)
from src.generation.mistral_chat import MistralChat
from src.rag.pipeline import HybridRAG
from src.types import RAGAnswer, RetrievedChunk


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def context_as_text(context_chunks: List[RetrievedChunk], max_chars: int = 6000) -> str:
    parts: List[str] = []
    current = 0
    for retrieved in context_chunks:
        block = f"{retrieved.chunk.title} ({retrieved.chunk.url})\n{retrieved.chunk.text}"
        if current + len(block) > max_chars:
            break
        parts.append(block)
        current += len(block) + 2
    return "\n\n".join(parts)


def run_evaluation(
    rag: HybridRAG,
    questions: List[Dict[str, Any]],
    max_new_tokens: int = 128,
    judge_model_name: Optional[str] = None,
    judge_device: str = "cuda",
    hitrate_k: int = 5,
) -> Dict[str, Any]:
    answers: List[RAGAnswer] = []
    ground_truth_urls: List[str] = []
    rows: List[Dict[str, Any]] = []

    judge: Optional[MistralChat] = None
    if judge_model_name:
        judge = MistralChat(model_name=judge_model_name, device=judge_device)
        judge.load()

    for item in questions:
        qid = item.get("qid", "")
        question_text = item["question"]
        ground_truth_url = item["ground_truth_url"]
        ground_truth_answer = item.get("ground_truth_answer", "")
        category = item.get("category", "")

        ground_truth_urls.append(ground_truth_url)

        start = time.perf_counter()
        response = rag.answer(question_text, max_new_tokens=max_new_tokens)
        end = time.perf_counter()

        answers.append(response)
        ranked_urls = unique_url_ranking(response)

        # Custom metric #1: claim-level faithfulness grounded to retrieved chunks.
        csfs_faithfulness = claim_supported_faithfulness(response)

        judge_payload: Dict[str, Any] = {}
        if judge is not None:
            judge_payload = judge_one(
                chat=judge,
                question=question_text,
                ground_truth_answer=ground_truth_answer,
                generated_answer=response.answer,
                context=context_as_text(response.context_chunks),
            )

        rows.append(
            {
                "qid": qid,
                "category": category,
                "question": question_text,
                "ground_truth_url": ground_truth_url,
                "ground_truth_answer": ground_truth_answer,
                "generated_answer": response.answer,
                "ranked_urls": ranked_urls,
                "csfs_faithfulness": csfs_faithfulness,
                "latency_ms_total": (end - start) * 1000.0,
                "factual_score": judge_payload.get("factual_score"),
                "completeness_score": judge_payload.get("completeness_score"),
                "faithfulness_score": judge_payload.get("faithfulness_score"),
                "hallucination": judge_payload.get("hallucination"),
                "judge_explanation": judge_payload.get("judge_explanation"),
            }
        )

    mrr_value = mean_reciprocal_rank_url_level(answers, ground_truth_urls)
    hitrate_value = hit_rate_at_k_url_level(answers, ground_truth_urls, k=hitrate_k)
    avg_csfs = (
        sum(r.get("csfs_faithfulness", 0.0) for r in rows) / len(rows)
        if rows
        else 0.0
    )

    return {
        "mrr_url_level": mrr_value,
        f"hitrate@{hitrate_k}_url_level": hitrate_value,
        "avg_csfs_faithfulness": avg_csfs,
        "row_count": len(rows),
        "rows": rows,
    }
