"""
Builds the text corpus from a list of Wikipedia URLs.
For each URL it fetches the page, extracts and cleans the text, then splits
it into overlapping chunks (200-400 tokens). The resulting chunks are saved
as a JSONL file that serves as the knowledge base for retrieval.
Usage: python -m src.ingestion.build_corpus --urls data/urls/all_urls.json --out data/corpus/chunks.jsonl
"""

import argparse
from typing import List

from src.ingestion.fetch_wikipedia import fetch_and_clean, is_valid_page
from src.types import Chunk
from src.utils.chunking import chunk_text
from src.utils.io import read_json, write_jsonl


def build_chunks(urls: List[str]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    failed_urls: List[str] = []
    for i, url in enumerate(urls):
        try:
            title, text = fetch_and_clean(url)
            if not is_valid_page(text):
                print(f"[{i+1}/{len(urls)}] Skipped (too short): {url}")
                continue
            chunks = chunk_text(url=url, title=title, text=text)
            all_chunks.extend(chunks)
            print(f"[{i+1}/{len(urls)}] OK: {title} ({len(chunks)} chunks)")
        except Exception as e:
            failed_urls.append(url)
            print(f"[{i+1}/{len(urls)}] FAILED: {url} - {e}")
    if failed_urls:
        print(f"\n{len(failed_urls)} URLs failed to fetch.")
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", required=True, help="Path to all_urls.json")
    parser.add_argument("--out", required=True, help="Output chunks jsonl path")
    args = parser.parse_args()

    urls = read_json(args.urls)
    chunks = build_chunks(urls)
    write_jsonl([c.__dict__ for c in chunks], args.out)
    print(f"Wrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()
