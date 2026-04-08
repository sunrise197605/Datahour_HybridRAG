"""
Automated evaluation question generator using Mistral-7B-Instruct.
Picks random Wikipedia articles from the corpus, sends their text to
the LLM, and asks it to generate questions across 5 categories:
factual, descriptive, comparative, inferential, and multi-hop.
Produces 100 questions with ground-truth answers saved to questions.json.
Usage: python -m src.evaluation.question_gen --chunks data/corpus/chunks.jsonl --out data/eval/questions.json
"""

import argparse
import json
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.generation.mistral_chat import ChatMessage, MistralChat
from src.types import Chunk
from src.utils.io import read_jsonl


TARGET_DISTRIBUTION: Dict[str, int] = {
    "factual": 35,
    "descriptive": 20,
    "comparative": 15,
    "inferential": 15,
    "multi_hop": 15,
}


def load_chunks(chunks_jsonl_path: str) -> List[Chunk]:
    records = read_jsonl(chunks_jsonl_path)
    return [Chunk(**r) for r in records]


def group_chunks_by_url(chunks: List[Chunk]) -> Dict[str, List[Chunk]]:
    grouped: Dict[str, List[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.url].append(chunk)
    for url in grouped:
        grouped[url] = sorted(grouped[url], key=lambda c: c.chunk_index)
    return grouped


def extract_json_block(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def safe_json_load(text: str) -> Optional[Any]:
    try:
        return json.loads(extract_json_block(text))
    except Exception:
        return None


def build_source_context(chunks_for_url: List[Chunk], max_chars: int = 3500) -> Tuple[str, List[str]]:
    selected_ids: List[str] = []
    parts: List[str] = []
    current = 0

    for chunk in chunks_for_url[:6]:
        block = f"CHUNK_ID: {chunk.chunk_id}\n{chunk.text}"
        if current + len(block) > max_chars:
            break
        parts.append(block)
        selected_ids.append(chunk.chunk_id)
        current += len(block) + 2

    return "\n\n".join(parts), selected_ids


def messages_for_question_batch(url: str, title: str, context: str, category: str, count: int) -> List[ChatMessage]:
    system = (
        "You generate evaluation questions for a RAG system. "
        "Rules: (1) Questions must be answerable only from the provided context. "
        "(2) Do not use outside knowledge. (3) Answers must be short and precise. "
        "(4) Return strict JSON only."
    )

    user = (
        f"Source URL: {url}\n"
        f"Title: {title}\n\n"
        f"Context:\n{context}\n\n"
        f"Generate {count} questions of type '{category}'.\n"
        "Return a JSON array where each item has exactly these keys:\n"
        "- question\n"
        "- ground_truth_answer\n"
        "No other keys. No explanations."
    )

    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def make_qid(index: int) -> str:
    return f"q{index:03d}"


def generate_questions(
    chunks_jsonl_path: str,
    output_path: str,
    question_count: int = 100,
    seed: int = 114,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    device: str = "cuda",
) -> None:
    rng = random.Random(seed)
    chunks = load_chunks(chunks_jsonl_path)
    grouped = group_chunks_by_url(chunks)

    urls = list(grouped.keys())
    rng.shuffle(urls)

    chat = MistralChat(model_name=model_name, device=device)
    chat.load()

    remaining = dict(TARGET_DISTRIBUTION)
    questions: List[Dict[str, Any]] = []
    seen_questions: set[str] = set()

    url_cursor = 0
    attempts = 0
    max_attempts = 600

    while len(questions) < question_count and attempts < max_attempts:
        attempts += 1
        open_categories = [c for c, v in remaining.items() if v > 0]
        if not open_categories:
            break
        category = rng.choice(open_categories)

        if url_cursor >= len(urls):
            url_cursor = 0
            rng.shuffle(urls)

        url = urls[url_cursor]
        url_cursor += 1

        url_chunks = grouped[url]
        if not url_chunks:
            continue

        title = url_chunks[0].title
        context, source_chunk_ids = build_source_context(url_chunks)

        batch_size = 3 if remaining[category] >= 3 else remaining[category]
        messages = messages_for_question_batch(url, title, context, category, batch_size)
        raw = chat.generate(messages, max_new_tokens=700, temperature=0.2)

        parsed = safe_json_load(raw)
        if not isinstance(parsed, list):
            continue

        for item in parsed:
            if not isinstance(item, dict):
                continue
            question_text = str(item.get("question", "")).strip()
            answer_text = str(item.get("ground_truth_answer", "")).strip()

            if len(question_text) < 10 or len(answer_text) < 1:
                continue
            key = question_text.lower()
            if key in seen_questions:
                continue

            qid = make_qid(len(questions) + 1)
            questions.append(
                {
                    "qid": qid,
                    "question": question_text,
                    "ground_truth_answer": answer_text,
                    "ground_truth_url": url,
                    "category": category,
                    "source_chunk_ids": source_chunk_ids,
                }
            )
            seen_questions.add(key)
            remaining[category] -= 1

            if len(questions) >= question_count or remaining[category] <= 0:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    counts = {k: 0 for k in TARGET_DISTRIBUTION}
    for q in questions:
        counts[q["category"]] += 1

    print(f"Wrote {len(questions)} questions to {output_path}")
    print("Category counts:", counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--out", required=True, help="Output questions.json")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    generate_questions(
        chunks_jsonl_path=args.chunks,
        output_path=args.out,
        question_count=args.count,
        seed=args.seed,
        model_name=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
