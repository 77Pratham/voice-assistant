#!/usr/bin/env python3
"""Fetch everything the assistant needs to run offline.

  python scripts/download_models.py            # Piper voices (en + hi) and the Whisper model
  python scripts/download_models.py --llm      # additionally `ollama pull <LLM_MODEL>`
  python scripts/download_models.py --no-whisper

Piper voices come from https://huggingface.co/rhasspy/piper-voices (MIT-licensed voice
collection; check the individual voice cards for dataset licences).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402

HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def voice_urls(voice: str) -> tuple[str, str]:
    """'hi_IN-pratham-medium' -> URLs of the .onnx model and its .onnx.json config."""
    locale, name, quality = voice.split("-", 2)
    lang = locale.split("_")[0]
    base = f"{HF}/{lang}/{locale}/{name}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already present: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {dest.name}: {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, hook)
    tmp.rename(dest)
    print(f"\r  {dest.name}: done ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="also run `ollama pull` for LLM_MODEL")
    parser.add_argument("--no-whisper", action="store_true", help="skip pre-fetching the Whisper model")
    args = parser.parse_args()

    s = Settings.from_env()
    print(f"Piper voices -> {s.voices_dir}")
    for voice in (s.voice_en, s.voice_hi):
        model_url, config_url = voice_urls(voice)
        download(model_url, s.voices_dir / f"{voice}.onnx")
        download(config_url, s.voices_dir / f"{voice}.onnx.json")

    if not args.no_whisper:
        name = "small" if s.whisper_model == "auto" else s.whisper_model
        print(f"Whisper '{name}' (cached in the Hugging Face cache)")
        from faster_whisper import WhisperModel

        WhisperModel(name, device="cpu", compute_type="int8")
        print("  done")

    if args.llm:
        if shutil.which("ollama") is None:
            print("`ollama` not found on PATH. Install it from https://ollama.com first.", file=sys.stderr)
            return 1
        print(f"ollama pull {s.llm_model}")
        subprocess.run(["ollama", "pull", s.llm_model], check=True)

    print("\nAll set. Start the assistant with:  python -m app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
