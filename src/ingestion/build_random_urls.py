"""
Generates 300 random Wikipedia URLs for the corpus.
Uses the Wikipedia API to search for travel-related articles (100 URLs) and
fetch truly random articles (200 URLs). Each candidate URL is validated by
downloading the page and checking it has at least 200 words of content.
Usage: python -m src.ingestion.build_random_urls --out data/urls/random_urls.json
"""

import argparse
import json
import random
import time
from typing import Iterable, List, Optional, Set
from urllib.parse import quote

import requests

from src.ingestion.fetch_wikipedia import fetch_and_clean, is_valid_page


WIKI_API = "https://en.wikipedia.org/w/api.php"

TRAVEL_QUERIES = [
    "tourism",
    "travel",
    "airport",
    "railway station",
    "national park",
    "UNESCO World Heritage Site",
    "capital city",
    "island",
    "mountain pass",
    "metro system",
    "airline",
    "museum",
    "beach",
    "temple",
]


def _request_json(params: dict, timeout_seconds: int = 20) -> dict:
    response = requests.get(WIKI_API, params=params, timeout=timeout_seconds, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.json()


def wikipedia_search_titles(query: str, limit: int = 30) -> List[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
        "srnamespace": 0,
    }
    data = _request_json(params)
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def wikipedia_random_titles(limit: int = 50) -> List[str]:
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": limit,
        "format": "json",
    }
    data = _request_json(params)
    return [item["title"] for item in data.get("query", {}).get("random", [])]


def title_to_url(title: str) -> str:
    safe_title = quote(title.replace(" ", "_"))
    return f"https://en.wikipedia.org/wiki/{safe_title}"


def validate_urls(urls: Iterable[str], min_words: int = 200, max_keep: Optional[int] = None) -> List[str]:
    kept: List[str] = []
    for url in urls:
        try:
            _, text = fetch_and_clean(url)
            if is_valid_page(text, min_words=min_words):
                kept.append(url)
                if max_keep is not None and len(kept) >= max_keep:
                    break
        except Exception:
            continue
    return kept


def build_travel_urls(target_count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    candidates: List[str] = []
    seen: Set[str] = set()

    queries = TRAVEL_QUERIES[:]
    rng.shuffle(queries)

    for q in queries:
        titles = wikipedia_search_titles(q, limit=40)
        rng.shuffle(titles)
        for title in titles:
            url = title_to_url(title)
            if url in seen:
                continue
            seen.add(url)
            candidates.append(url)
            if len(candidates) >= target_count * 3:
                break
        if len(candidates) >= target_count * 3:
            break

    return validate_urls(candidates, min_words=200, max_keep=target_count)


def build_mixed_random_urls(target_count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    candidates: List[str] = []
    seen: Set[str] = set()

    while len(candidates) < target_count * 2:
        titles = wikipedia_random_titles(limit=50)
        rng.shuffle(titles)
        for title in titles:
            url = title_to_url(title)
            if url in seen:
                continue
            seen.add(url)
            candidates.append(url)
            if len(candidates) >= target_count * 2:
                break
        time.sleep(0.25)

    return validate_urls(candidates, min_words=200, max_keep=target_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output path for random_urls.json")
    parser.add_argument("--travel_count", type=int, default=100)
    parser.add_argument("--mixed_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=114)
    args = parser.parse_args()

    travel_urls = build_travel_urls(target_count=args.travel_count, seed=args.seed)
    mixed_urls = build_mixed_random_urls(target_count=args.mixed_count, seed=args.seed + 1)

    random_urls = travel_urls + mixed_urls

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(random_urls, f, indent=2)

    print(
        f"Saved {len(random_urls)} random URLs to {args.out} "
        f"(travel={len(travel_urls)}, mixed={len(mixed_urls)})"
    )


if __name__ == "__main__":
    main()
