"""
URL sampling and merging utilities.
Provides helpers to load/save URL lists from JSON, randomly sample a subset,
and combine the fixed (curated travel) URLs with randomly sampled URLs into
a single all_urls.json file used for corpus building.
"""

import json
import random
from typing import List, Optional


def load_urls(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_urls(urls: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=2)


def sample_random_urls(candidate_urls: List[str], count: int, seed: Optional[int] = None) -> List[str]:
    rng = random.Random(seed)
    return rng.sample(candidate_urls, count)


def combine_fixed_and_random(fixed_urls_path: str, random_urls_path: str, output_path: str) -> None:
    fixed_urls = load_urls(fixed_urls_path)
    random_urls = load_urls(random_urls_path)
    all_urls = fixed_urls + random_urls
    save_urls(all_urls, output_path)
