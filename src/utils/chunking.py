"""
Text chunking module — splits long article text into overlapping windows.
Each chunk is 200-400 tokens (approximated by word count) with 50-token
overlap so that no information is lost at chunk boundaries. Every chunk
gets a unique ID derived from its source URL and position index.
"""

from typing import List

from src.types import Chunk
from src.utils.text_cleaning import normalize_whitespace


def chunk_text(
    url: str,
    title: str,
    text: str,
    chunk_tokens_min: int = 200,
    chunk_tokens_max: int = 400,
    overlap_tokens: int = 50,
) -> List[Chunk]:
    """
    Splits text into overlapping token windows.
    We approximate tokens by splitting on whitespace (1 word ~ 1 token).
    Windows are 200-400 words with 50-word overlap between consecutive chunks.
    """
    cleaned_text = normalize_whitespace(text)
    words = cleaned_text.split(" ")
    if not words:
        return []

    # We approximate token count with word count (close enough for our use case)
    min_len = max(1, chunk_tokens_min)
    max_len = max(min_len, chunk_tokens_max)
    overlap = max(0, min(overlap_tokens, max_len - 1))

    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(len(words), start + max_len)
        window = words[start:end]
        if len(window) < min_len:
            break
        chunk_str = " ".join(window).strip()
        chunk_id = f"{abs(hash(url))}_{chunk_index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                url=url,
                title=title,
                chunk_index=chunk_index,
                text=chunk_str,
            )
        )
        chunk_index += 1
        start = end - overlap

    return chunks
