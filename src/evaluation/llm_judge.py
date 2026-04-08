"""
LLM-as-Judge evaluation using Mistral-7B-Instruct.
Sends the question, ground-truth answer, generated answer, and context to
Mistral and asks it to return a JSON with scores for factual accuracy,
completeness, faithfulness (0-5 each), and a hallucination flag.
This provides a richer evaluation than automated metrics alone.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.generation.mistral_chat import ChatMessage, MistralChat


def extract_json_object(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(extract_json_object(text))
    except Exception:
        return None


def build_judge_messages(question: str, ground_truth_answer: str, generated_answer: str, context: str) -> List[ChatMessage]:
    system = (
        "You are an impartial evaluator for a RAG system. "
        "Use only the provided context. "
        "Return strict JSON only. No markdown."
    )

    user = (
        f"Question: {question}\n"
        f"Ground truth answer: {ground_truth_answer}\n"
        f"Generated answer: {generated_answer}\n\n"
        f"Context:\n{context}\n\n"
        "Return a strict JSON object with fields:\n"
        "factual_score (0-5), completeness_score (0-5), faithfulness_score (0-5), "
        "hallucination (true/false), judge_explanation (one line)."
    )

    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def judge_one(chat: MistralChat, question: str, ground_truth_answer: str, generated_answer: str, context: str) -> Dict[str, Any]:
    messages = build_judge_messages(question, ground_truth_answer, generated_answer, context)
    raw = chat.generate(messages, max_new_tokens=350, temperature=0.0)
    parsed = safe_json_load(raw)
    if parsed is None:
        return {
            "factual_score": 0,
            "completeness_score": 0,
            "faithfulness_score": 0,
            "hallucination": True,
            "judge_explanation": "Judge output could not be parsed as JSON.",
        }
    return parsed
