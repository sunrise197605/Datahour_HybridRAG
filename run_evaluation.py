"""
Main evaluation script — run this to reproduce all results.
Loads the full RAG pipeline, evaluates on 100 test questions computing MRR,
HitRate@5, HitRate@10, CSFS, CUS, and ACS metrics, then runs an ablation
study (Hybrid vs Dense-only vs Sparse-only). Results are saved to
data/eval/evaluation_results.json.
Usage: python run_evaluation.py
"""

import json
import time
from typing import Any, Dict, List

from src.evaluation.metrics import (
    claim_supported_faithfulness,
    hit_rate_at_k_url_level,
    mean_reciprocal_rank_url_level,
    context_utilization_score,
    answer_completeness_score,
)
from src.generation.llm import LLMGenerator
from src.rag.pipeline import HybridRAG
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex
from src.types import Chunk, RAGAnswer
from src.utils.io import read_jsonl


def load_chunks(path):
    records = read_jsonl(path)
    return [Chunk(**r) for r in records]


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(rag, questions, max_questions=100):
    answers = []
    ground_truth_urls = []
    csfs_scores = []
    cus_scores = []
    acs_scores = []
    latencies = []
    
    eval_questions = questions[:max_questions]
    
    print(f"Evaluating {len(eval_questions)} questions...")
    
    for i, q in enumerate(eval_questions, 1):
        start = time.perf_counter()
        result = rag.answer(q["question"], top_k=50, context_size=6, max_new_tokens=100)
        elapsed = (time.perf_counter() - start) * 1000
        
        answers.append(result)
        ground_truth_urls.append(q["ground_truth_url"])
        latencies.append(elapsed)
        
        try:
            csfs = claim_supported_faithfulness(result)
            csfs_scores.append(csfs)
        except Exception:
            csfs_scores.append(0.0)
        
        try:
            cus = context_utilization_score(result)
            cus_scores.append(cus)
        except Exception:
            cus_scores.append(0.0)
        
        try:
            acs = answer_completeness_score(result, q["question"])
            acs_scores.append(acs)
        except Exception:
            acs_scores.append(0.0)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(eval_questions)}")
    
    mrr = mean_reciprocal_rank_url_level(answers, ground_truth_urls)
    hit_rate_5 = hit_rate_at_k_url_level(answers, ground_truth_urls, k=5)
    hit_rate_10 = hit_rate_at_k_url_level(answers, ground_truth_urls, k=10)
    avg_csfs = sum(csfs_scores) / len(csfs_scores) if csfs_scores else 0.0
    avg_cus = sum(cus_scores) / len(cus_scores) if cus_scores else 0.0
    avg_acs = sum(acs_scores) / len(acs_scores) if acs_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    return {
        "mrr_url": mrr,
        "hit_rate_at_5": hit_rate_5,
        "hit_rate_at_10": hit_rate_10,
        "avg_csfs": avg_csfs,
        "avg_cus": avg_cus,
        "avg_acs": avg_acs,
        "avg_latency_ms": avg_latency,
        "num_questions": len(eval_questions),
    }


def run_ablation(chunks, dense_index, bm25_index, generator, questions, max_questions=100):
    eval_questions = questions[:max_questions]
    results = {}
    
    rag = HybridRAG(chunks, dense_index, bm25_index, generator, rrf_constant=60)
    
    # Hybrid
    print("Ablation: Hybrid...")
    hybrid_answers = []
    hybrid_urls = []
    hybrid_latencies = []
    
    for q in eval_questions:
        start = time.perf_counter()
        result = rag.answer(q["question"], top_k=50, context_size=6, max_new_tokens=64)
        elapsed = (time.perf_counter() - start) * 1000
        hybrid_answers.append(result)
        hybrid_urls.append(q["ground_truth_url"])
        hybrid_latencies.append(elapsed)
    
    results["hybrid"] = {
        "mrr_url": mean_reciprocal_rank_url_level(hybrid_answers, hybrid_urls),
        "hit_rate_at_5": hit_rate_at_k_url_level(hybrid_answers, hybrid_urls, k=5),
        "avg_latency_ms": sum(hybrid_latencies) / len(hybrid_latencies),
    }
    
    # Dense-only
    print("Ablation: Dense-only...")
    dense_answers = []
    for ans in hybrid_answers:
        sorted_chunks = sorted(ans.context_chunks, key=lambda c: c.dense_rank if c.dense_rank else 9999)
        dense_answers.append(RAGAnswer(
            query=ans.query,
            answer=ans.answer,
            context_chunks=sorted_chunks,
            latency_ms=ans.latency_ms,
            debug={"mode": "dense"},
        ))
    
    results["dense_only"] = {
        "mrr_url": mean_reciprocal_rank_url_level(dense_answers, hybrid_urls),
        "hit_rate_at_5": hit_rate_at_k_url_level(dense_answers, hybrid_urls, k=5),
        "avg_latency_ms": results["hybrid"]["avg_latency_ms"],
    }
    
    # Sparse-only
    print("Ablation: Sparse-only...")
    sparse_answers = []
    for ans in hybrid_answers:
        sorted_chunks = sorted(ans.context_chunks, key=lambda c: c.bm25_rank if c.bm25_rank else 9999)
        sparse_answers.append(RAGAnswer(
            query=ans.query,
            answer=ans.answer,
            context_chunks=sorted_chunks,
            latency_ms=ans.latency_ms,
            debug={"mode": "sparse"},
        ))
    
    results["sparse_only"] = {
        "mrr_url": mean_reciprocal_rank_url_level(sparse_answers, hybrid_urls),
        "hit_rate_at_5": hit_rate_at_k_url_level(sparse_answers, hybrid_urls, k=5),
        "avg_latency_ms": results["hybrid"]["avg_latency_ms"],
    }
    
    return results


def main():
    print("=" * 60)
    print("HYBRID RAG EVALUATION")
    print("=" * 60)
    print()
    
    print("[1/5] Loading chunks...")
    chunks = load_chunks("data/corpus/chunks.jsonl")
    print(f"      {len(chunks)} chunks")
    
    print("[2/5] Loading Dense index...")
    dense_index = DenseIndex.load(chunks, "indexes/dense")
    
    print("[3/5] Loading BM25 index...")
    bm25_index = BM25Index.load(chunks, "indexes/bm25")
    
    print("[4/5] Loading LLM...")
    generator = LLMGenerator(model_name="google/flan-t5-base", device="cpu")
    generator.load()
    
    print("[5/5] Loading questions...")
    questions = load_questions("data/eval/questions.json")
    print(f"      {len(questions)} questions")
    
    rag = HybridRAG(
        chunks=chunks,
        dense_index=dense_index,
        bm25_index=bm25_index,
        generator=generator,
        rrf_constant=60,
    )
    
    print()
    print("=" * 60)
    print("MAIN EVALUATION")
    print("=" * 60)
    print()
    
    eval_results = run_evaluation(rag, questions, max_questions=100)
    
    print()
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"  Questions:   {eval_results['num_questions']}")
    print(f"  MRR:         {eval_results['mrr_url']:.4f}")
    print(f"  HitRate@5:   {eval_results['hit_rate_at_5']:.4f}")
    print(f"  HitRate@10:  {eval_results['hit_rate_at_10']:.4f}")
    print(f"  Avg CSFS:    {eval_results['avg_csfs']:.4f}")
    print(f"  Avg CUS:     {eval_results['avg_cus']:.4f}")
    print(f"  Avg ACS:     {eval_results['avg_acs']:.4f}")
    print(f"  Avg Latency: {eval_results['avg_latency_ms']:.0f} ms")
    print("-" * 60)
    
    print()
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    print()
    
    ablation_results = run_ablation(
        chunks, dense_index, bm25_index, generator, questions, max_questions=100
    )
    
    print()
    print("-" * 60)
    print("ABLATION RESULTS")
    print("-" * 60)
    print()
    print(f"{'Method':<15} {'MRR':<10} {'HitRate@5':<12} {'Latency':<10}")
    print("-" * 50)
    for method, metrics in ablation_results.items():
        print(f"{method:<15} {metrics['mrr_url']:<10.4f} {metrics['hit_rate_at_5']:<12.4f} {metrics['avg_latency_ms']:<10.0f}")
    print("-" * 60)
    
    # Save results
    output = {
        "evaluation": eval_results,
        "ablation": ablation_results,
    }
    
    with open("data/eval/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print()
    print("Results saved to data/eval/evaluation_results.json")


if __name__ == "__main__":
    main()
