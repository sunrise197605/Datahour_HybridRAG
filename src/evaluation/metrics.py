"""
Evaluation metrics for measuring RAG system quality.
Standard metrics:
  - MRR (Mean Reciprocal Rank) at URL level: how high the correct source ranks.
  - HitRate@K: whether the correct source appears in the top-K results.
  - CSFS (Claim-Supported Faithfulness): splits the answer into claims and checks
    each one is semantically supported by the retrieved context.
Custom metrics (our contribution):
  - CUS (Context Utilization Score): does the answer actually use the context?
  - ACS (Answer Completeness Score): does the answer address the question?
"""

import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.types import RAGAnswer


_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_EMBEDDER = None       # cached SentenceTransformer instance
_EMBEDDER_NAME = None  # name of the cached model


def _get_embedder(model_name: str) -> SentenceTransformer:
    """Load embedder once and reuse across calls to avoid reloading."""
    global _EMBEDDER, _EMBEDDER_NAME
    if _EMBEDDER is None or _EMBEDDER_NAME != model_name:
        _EMBEDDER = SentenceTransformer(model_name)
        _EMBEDDER_NAME = model_name
    return _EMBEDDER


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def _split_into_claims(answer_text: str) -> List[str]:
    """Split answer into individual claims/facts."""
    text = (answer_text or "").replace("\n", " ").strip()
    if not text:
        return []

    parts = re.split(r"[.;!?]+", text)
    claims: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        subparts = re.split(r"\b(?:and|but|however|also)\b", p, flags=re.IGNORECASE)
        for sp in subparts:
            sp = sp.strip(" ,")
            if len(sp) >= 8:
                claims.append(sp)

    return claims or [text]


def _extract_numbers(text: str) -> List[str]:
    return [m.group(0) for m in _NUM_RE.finditer(text or "")]


def _has_numeric_contradiction(claim: str, evidence: str) -> bool:
    """Check if numbers in claim match numbers in evidence."""
    claim_nums = _extract_numbers(claim)
    if not claim_nums:
        return False
    evidence_nums = set(_extract_numbers(evidence))
    return any(n not in evidence_nums for n in claim_nums)


def claim_supported_faithfulness(
    answer: RAGAnswer,
    model_name: str = "all-mpnet-base-v2",
    sim_threshold: float = 0.78,
    max_chunks_to_check: int = 8,
) -> float:
    """CSFS: checks if answer claims are supported by retrieved context."""
    claims = _split_into_claims(answer.answer)
    if not claims:
        return 0.0

    chunks = answer.context_chunks[:max_chunks_to_check]
    if not chunks:
        return 0.0

    embedder = _get_embedder(model_name)
    claim_vecs = embedder.encode(claims, convert_to_numpy=True)
    chunk_texts = [c.chunk.text for c in chunks]
    chunk_vecs = embedder.encode(chunk_texts, convert_to_numpy=True)

    supported = 0
    for i, claim in enumerate(claims):
        sims = [_cosine(claim_vecs[i], chunk_vecs[j]) for j in range(len(chunks))]
        best_j = int(np.argmax(sims))
        best_sim = sims[best_j]
        best_chunk = chunk_texts[best_j]

        if best_sim >= sim_threshold and not _has_numeric_contradiction(claim, best_chunk):
            supported += 1

    return supported / len(claims)


def unique_url_ranking(answer: RAGAnswer) -> List[str]:
    seen_urls = set()
    url_ranking: List[str] = []
    for retrieved in answer.context_chunks:
        url = retrieved.chunk.url
        if url not in seen_urls:
            url_ranking.append(url)
            seen_urls.add(url)
    return url_ranking


def mean_reciprocal_rank_url_level(answers: List[RAGAnswer], ground_truth_urls: List[str]) -> float:
    if len(answers) != len(ground_truth_urls):
        raise ValueError("Answers and ground truth URL lists must have the same length.")

    reciprocal_ranks: List[float] = []
    for answer, ground_truth_url in zip(answers, ground_truth_urls):
        ranked_urls = unique_url_ranking(answer)
        if ground_truth_url in ranked_urls:
            rank_position = ranked_urls.index(ground_truth_url) + 1
            reciprocal_ranks.append(1.0 / rank_position)
        else:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def hit_rate_at_k_url_level(answers: List[RAGAnswer], ground_truth_urls: List[str], k: int = 5) -> float:
    """Check if correct URL appears in top-K results."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(answers) != len(ground_truth_urls):
        raise ValueError("Answers and ground truth URL lists must have the same length.")

    hits = 0
    for answer, ground_truth_url in zip(answers, ground_truth_urls):
        ranked_urls = unique_url_ranking(answer)[:k]
        if ground_truth_url in ranked_urls:
            hits += 1
    return hits / len(answers) if answers else 0.0


def _extract_keywords(text: str, min_len: int = 4) -> set:
    """Get important words, skip common ones."""
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'this', 'that',
            'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'and', 'but', 'or', 'if', 'so', 'than', 'too', 'very', 'just', 'also', 'not'}
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return {w for w in words if len(w) >= min_len and w not in stop}


def context_utilization_score(answer: RAGAnswer, model_name: str = "all-mpnet-base-v2", max_chunks: int = 5) -> float:
    """CUS (Context Utilization Score) — Custom Metric #1.

    Measures whether the generated answer actually draws from the retrieved
    context rather than hallucinating or ignoring it. This matters because a
    RAG system should ground its answers in the documents it retrieved.

    How it works (two signals combined):
      1. Semantic similarity (weight 0.55): Embeds the answer and the concatenated
         context using SentenceTransformers and computes cosine similarity. A high
         score means the answer is talking about the same topics as the context.
      2. Word-level containment (weight 0.45): Checks what fraction of words in
         the answer also appear in the context. If the full answer string is found
         verbatim in the context, containment = 1.0 (perfect grounding).

    Final score = 0.55 * semantic_sim + 0.45 * containment + 0.05 base boost,
    capped at 1.0.

    Args:
        answer:     RAGAnswer object containing the generated answer and retrieved chunks.
        model_name: SentenceTransformer model for computing embeddings (default: all-mpnet-base-v2).
        max_chunks: Number of top retrieved chunks to use as context (default: 5).

    Returns:
        Float between 0.0 and 1.0. Higher = answer is better grounded in context.
    """
    answer_text = (answer.answer or "").strip()
    if not answer_text:
        return 0.0
    
    chunks = answer.context_chunks[:max_chunks]
    if not chunks:
        return 0.0
    
    context_text = " ".join([c.chunk.text for c in chunks])
    
    # Semantic similarity
    embedder = _get_embedder(model_name)
    ans_vec = embedder.encode(answer_text, convert_to_numpy=True)
    ctx_vec = embedder.encode(context_text[:2000], convert_to_numpy=True)
    sem_score = max(_cosine(ans_vec, ctx_vec), 0.0)
    
    # Check if answer content appears in context (exact or partial match)
    ans_lower = answer_text.lower()
    ctx_lower = context_text.lower()
    
    # Direct containment check
    if ans_lower in ctx_lower:
        containment_score = 1.0
    else:
        # Check word-level overlap
        ans_words = set(ans_lower.split())
        ctx_words = set(ctx_lower.split())
        if ans_words:
            containment_score = len(ans_words & ctx_words) / len(ans_words)
        else:
            containment_score = 0.0
    
    # Combine: semantic similarity + containment bonus
    score = 0.55 * sem_score + 0.45 * containment_score
    return min(score + 0.05, 1.0)  # Small base boost


def answer_completeness_score(answer: RAGAnswer, question: str, model_name: str = "all-mpnet-base-v2") -> float:
    """ACS (Answer Completeness Score) — Custom Metric #2.

    Measures whether the generated answer actually addresses the user's question
    instead of giving a vague, evasive, or off-topic response. Standard metrics
    like MRR only check retrieval quality — ACS checks generation quality.

    How it works (three checks):
      1. Non-answer detection: If the answer matches evasive patterns like
         "I don't know" or "no information", the score is penalized to 0.1.
      2. Semantic relevance (weight 0.65): Embeds both the question and answer
         with SentenceTransformers and computes cosine similarity. A high score
         means the answer is topically relevant to what was asked.
      3. Factual content bonus: If the answer contains numbers or proper nouns
         (capitalized words), it is treated as factual and not penalized for
         being short (e.g., "828 metres" is a valid complete answer).

    Final score = (0.65 * semantic_sim + 0.35 base) * length_bonus, capped at 1.0.

    Args:
        answer:     RAGAnswer object containing the generated answer.
        question:   The original question string the user asked.
        model_name: SentenceTransformer model for computing embeddings (default: all-mpnet-base-v2).

    Returns:
        Float between 0.0 and 1.0. Higher = answer more completely addresses the question.
    """
    ans_text = (answer.answer or "").strip()
    q_text = (question or "").strip()
    
    if not ans_text or not q_text:
        return 0.0
    
    # Check for non-answers first
    bad_patterns = [r"i don'?t know", r"cannot find", r"no information", r"not found", r"unclear"]
    for p in bad_patterns:
        if re.search(p, ans_text.lower()):
            return 0.1
    
    embedder = _get_embedder(model_name)
    q_vec = embedder.encode(q_text, convert_to_numpy=True)
    a_vec = embedder.encode(ans_text, convert_to_numpy=True)
    sem_sim = max(_cosine(q_vec, a_vec), 0.0)
    
    # Check if answer contains factual content (numbers, proper nouns, etc.)
    has_numbers = bool(re.search(r'\d+', ans_text))
    has_caps = bool(re.search(r'[A-Z][a-z]+', ans_text))
    is_factual = has_numbers or has_caps
    
    # For factual questions, short answers are OK if they contain facts
    word_count = len(ans_text.split())
    if is_factual and word_count >= 1:
        length_bonus = 1.0  # Don't penalize short factual answers
    elif word_count < 2:
        length_bonus = 0.6
    else:
        length_bonus = 1.0
    
    # Base score from semantic similarity
    score = (0.65 * sem_sim + 0.35) * length_bonus
    return min(score, 1.0)
