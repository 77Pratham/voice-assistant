#!/usr/bin/env python3
"""Summarise latency recorded by real browser sessions (logs/latency.jsonl).

The server writes one JSON line per answered turn. Times are measured on the server from the
moment the utterance arrived, so they exclude the browser's end-of-speech wait; the web UI
shows the full perceived number (speech end -> first sound) live.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["stt_ms", "llm_first_token_ms", "first_sentence_ms", "first_audio_ms", "total_ms"]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "latency.jsonl"
    if not path.exists():
        print(f"No log at {path}. Talk to the assistant first.")
        return 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audio_rows = [r for r in rows if r.get("input") == "audio"]
    print(f"{len(rows)} turns logged ({len(audio_rows)} spoken, {len(rows) - len(audio_rows)} typed)\n")
    print(f"{'metric':<22}{'n':>5}{'p50':>9}{'p95':>9}{'max':>9}   (ms)")
    for key in KEYS:
        vals = [r[key] for r in audio_rows if key in r]
        if vals:
            print(f"{key:<22}{len(vals):>5}{statistics.median(vals):>9.0f}{pct(vals, 95):>9.0f}{max(vals):>9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
