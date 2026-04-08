"""
File I/O helpers for JSON and JSONL formats.
Provides read_json / write_json for single JSON files (e.g. URL lists,
config), and read_jsonl / write_jsonl for line-delimited JSON files
(e.g. the chunks corpus where each line is one chunk dictionary).
"""

import json
from typing import Any, Dict, Iterable, List


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
