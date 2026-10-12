#!/usr/bin/env python3
"""Headless latency benchmark + conversation-memory check (no microphone needed).

It uses the real models. Test utterances are rendered with Piper, resampled to 16 kHz and fed
through exactly the same `VoicePipeline.run_turn` that the web app uses, so the numbers
describe the actual pipeline on *your* machine. Results are printed and written to
docs/BENCHMARK_RESULTS.md.

  python scripts/benchmark.py              # 1 warm-up + 3 measured runs per sentence
  python scripts/benchmark.py --runs 5

Note: synthetic speech is clean, so STT accuracy here is optimistic. Latency is representative.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.pipeline import AudioOut, VoicePipeline  # noqa: E402
from app.session import Session  # noqa: E402

CASES = [
    ("en", "What is the capital of France, and what is it famous for?"),
    ("en", "Tell me one interesting fact about the planet Mars."),
    ("hi", "भारत की राजधानी क्या है और वहाँ कौन सी मशहूर जगहें हैं?"),
    ("hi", "मुझे चाय बनाने का आसान तरीका बताओ।"),
]
DIALOGUE = [
    "Hi, my name is Asha and I live in Bengaluru.",
    "Can you suggest one nice thing to do there on a weekend?",
    "Sorry, what was my name again, and which city did I say I live in?",
]
METRICS = [
    ("stt_ms", "Speech to text"),
    ("llm_first_token_ms", "LLM first token (from request start)"),
    ("first_sentence_ms", "First sentence ready for TTS"),
    ("first_audio_ms", "First audio produced (server total)"),
    ("total_ms", "Whole reply generated + synthesised"),
]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def hardware() -> dict:
    info = {"OS": platform.platform(), "CPU cores": str(os.cpu_count())}
    cpu = platform.processor() or platform.machine()
    try:
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        elif sys.platform == "darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    info["CPU"] = cpu
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["RAM"] = f"{int(line.split()[1]) / 1024 / 1024:.1f} GB"
                break
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
            ).strip()
            info["GPU"] = gpu.replace("\n", "; ")
        except Exception:
            pass
    info.setdefault("GPU", "none detected")
    return info


async def synth_audio(pipe: VoicePipeline, text: str, lang: str) -> np.ndarray:
    chunks, rate = [], 16_000
    async for rate, pcm in pipe.tts.stream(text, lang):
        chunks.append(np.frombuffer(pcm, dtype="<i2"))
    x = np.concatenate(chunks).astype(np.float32) / 32768.0
    if rate != 16_000:
        n = int(len(x) * 16_000 / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    pad = np.zeros(int(0.3 * 16_000), dtype=np.float32)
    return np.concatenate([pad, x, pad])


async def one_turn(pipe: VoicePipeline, session: Session, audio: np.ndarray, turn: int) -> dict:
    result: dict = {"reply": ""}
    async for ev in pipe.run_turn(session, turn=turn, audio=audio):
        if isinstance(ev, AudioOut):
            continue
        kind = ev["type"]
        if kind == "transcript":
            result["transcript"], result["language"] = ev["text"], ev["language"]
        elif kind == "llm_delta":
            result["reply"] += ev["text"]
        elif kind == "turn_done":
            result.update(ev)
        elif kind == "error":
            raise RuntimeError(ev["message"])
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="measured runs per sentence")
    parser.add_argument("--warmup", type=int, default=1, help="un-measured runs per sentence")
    parser.add_argument("--hangover-ms", type=int, default=650, help="UI end-of-speech wait used for the estimate")
    parser.add_argument("--out", default=str(ROOT / "docs" / "BENCHMARK_RESULTS.md"))
    args = parser.parse_args()

    settings = Settings.from_env()
    pipe = VoicePipeline.create(settings)
    await pipe.start()
    info = pipe.info()
    print("Models:", info)

    rows: dict[str, list[dict]] = {"en": [], "hi": []}
    transcripts: list[tuple[str, str, str]] = []
    for lang, text in CASES:
        if lang not in pipe.tts.voices:
            print(f"(skipping {lang}: no voice installed)")
            continue
        audio = await synth_audio(pipe, text, lang)
        for i in range(args.warmup + args.runs):
            r = await one_turn(pipe, Session(), audio, turn=1)
            if i == args.warmup:
                transcripts.append((lang, text, r.get("transcript", "")))
            if i >= args.warmup and "first_audio_ms" in r:
                rows[lang].append(r)
        print(f"  [{lang}] {text[:48]!r} -> {len(rows[lang])} runs collected")

    # --- multi-turn memory check ------------------------------------------------------
    print("\nConversation-memory check:")
    session, dialogue_log = Session(), []
    for i, line in enumerate(DIALOGUE, 1):
        audio = await synth_audio(pipe, line, "en")
        r = await one_turn(pipe, session, audio, turn=i)
        dialogue_log.append((line, r.get("transcript", ""), r["reply"].strip()))
        print(f"  you      : {line}\n  heard    : {r.get('transcript', '')}\n  assistant: {r['reply'].strip()}\n")

    # --- report -----------------------------------------------------------------------
    out = ["# Benchmark results", "", f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/benchmark.py`._", ""]
    out += ["## Setup", ""]
    out += [f"- **STT:** {info['stt_model']} on {info['stt_device']}"]
    out += [f"- **LLM:** {info['llm_model']} (Ollama)"]
    out += [f"- **TTS:** Piper, voices: {', '.join(info['voices'].values())}"]
    out += [f"- **{k}:** {v}" for k, v in hardware().items()]
    out += ["", "## Latency (milliseconds, median / p95 over all measured runs)", ""]
    for lang, name in (("en", "English"), ("hi", "Hindi")):
        data = rows[lang]
        if not data:
            continue
        out += [f"### {name} ({len(data)} runs)", "", "| Stage | median | p95 |", "|---|---:|---:|"]
        for key, label in METRICS:
            vals = [d[key] for d in data if key in d]
            if vals:
                out.append(f"| {label} | {statistics.median(vals):.0f} | {pct(vals, 95):.0f} |")
        fa = [d["first_audio_ms"] for d in data]
        out.append(f"| **Estimated perceived latency** (UI wait {args.hangover_ms} ms + server first audio + ~50 ms) "
                   f"| **{args.hangover_ms + statistics.median(fa) + 50:.0f}** | **{args.hangover_ms + pct(fa, 95) + 50:.0f}** |")
        stt_rtf = [d["stt_ms"] / (d["audio_in_s"] * 1000) for d in data if d.get("audio_in_s")]
        if stt_rtf:
            out += ["", f"STT real-time factor: {statistics.median(stt_rtf):.2f}x (lower is faster)."]
        out.append("")
    out += ["## Speech recognition sanity check (synthetic speech, so optimistic)", "", "| Lang | Said | Heard |", "|---|---|---|"]
    out += [f"| {l} | {a} | {b} |" for l, a, b in transcripts]
    out += ["", "## Multi-turn memory check", ""]
    for said, heard, reply in dialogue_log:
        out += [f"- **User:** {said}", f"  - heard as: {heard}", f"  - **Assistant:** {reply}"]
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")
    await pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
