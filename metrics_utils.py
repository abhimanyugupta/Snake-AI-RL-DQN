from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def append_metric_entry(path: str | Path, entry: dict) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def append_metric_entries(path: str | Path, entries: Iterable[dict]) -> None:
    rows = list(entries)
    if not rows:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for entry in rows:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_metric_entries(path: str | Path | None) -> List[dict]:
    if not path:
        return []

    log_path = Path(path)
    if not log_path.exists():
        return []

    rows: List[dict] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def rewrite_metric_entries(path: str | Path, entries: Iterable[dict]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def group_entries_by_algo(entries: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for entry in entries:
        algo = str(entry.get("algo", "unknown"))
        grouped.setdefault(algo, []).append(entry)

    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("episode", 0)))
    return grouped


def build_history(entries: Iterable[dict]) -> dict:
    rows = list(entries)
    return {
        "entries": rows,
        "scores": [int(item.get("score", 0)) for item in rows],
        "moving_avg": [float(item.get("moving_avg_20", 0.0)) for item in rows],
        "best_scores": [int(item.get("best_score", 0)) for item in rows],
        "episode_rewards": [float(item.get("episode_reward", 0.0)) for item in rows],
    }


def load_histories_from_log(path: str | Path | None) -> Dict[str, dict]:
    grouped = group_entries_by_algo(load_metric_entries(path))
    return {algo: build_history(entries) for algo, entries in grouped.items()}
