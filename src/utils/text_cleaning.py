"""
Text cleaning utilities used during Wikipedia ingestion.
- normalize_whitespace: collapses multiple spaces/newlines into a single space.
- remove_citation_markers: strips Wikipedia-style citations like [1], [42].
- basic_clean: applies both cleaners in sequence to produce clean body text.
"""

import re


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_citation_markers(text: str) -> str:
    # Removes patterns like [1], [12], etc.
    return re.sub(r"\[[0-9]+\]", "", text)


def basic_clean(text: str) -> str:
    text = remove_citation_markers(text)
    text = normalize_whitespace(text)
    return text
