"""Text-to-speech using Piper (fast, fully local neural TTS, runs well on CPU)."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

from .config import Settings

log = logging.getLogger("voice.tts")

ROOT = Path(__file__).resolve().parent.parent


class TextToSpeech:
    """Loads one Piper voice per language and streams int16 PCM chunks.

    Synthesis runs in a worker thread (ONNX Runtime releases the GIL) and pushes chunks
    onto an asyncio queue, so the event loop stays free to handle audio, the WebSocket
    and the LLM stream while a sentence is being rendered.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.voices: dict[str, object] = {}
        self.voice_names: dict[str, str] = {}
        self._syn_config = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    # ------------------------------------------------------------------ setup
    def _ensure_voices(self) -> None:
        """Download any missing Piper voice files on first run (Whisper is fetched separately)."""
        names = {self.settings.voice_en, self.settings.voice_hi}
        folder = Path(self.settings.voices_dir)
        missing = sorted(n for n in names if not (folder / f"{n}.onnx").exists() or not (folder / f"{n}.onnx.json").exists())
        if not missing:
            return
        log.warning("Piper voices missing: %s. Downloading them now.", ", ".join(missing))
        script = ROOT / "scripts" / "download_models.py"
        try:
            subprocess.run([sys.executable, str(script), "--no-whisper"], check=True, cwd=str(ROOT))
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"Could not download the Piper voices automatically ({exc}). "
                "Run manually: python scripts/download_models.py"
            ) from exc

    def load(self) -> None:
        from piper import PiperVoice, SynthesisConfig

        self._ensure_voices()
        wanted = {"en": self.settings.voice_en, "hi": self.settings.voice_hi}
        for lang, name in wanted.items():
            model = Path(self.settings.voices_dir) / f"{name}.onnx"
            if not model.exists():
                log.warning("Piper voice %s not found at %s (run scripts/download_models.py)", name, model)
                continue
            log.info("Loading Piper voice %s", name)
            self.voices[lang] = PiperVoice.load(model, use_cuda=self.settings.tts_use_cuda)
            self.voice_names[lang] = name
        if not self.voices:
            raise RuntimeError(
                f"No Piper voices found in {self.settings.voices_dir}. "
                "Run: python scripts/download_models.py"
            )
        if self.settings.tts_length_scale != 1.0:
            self._syn_config = SynthesisConfig(length_scale=self.settings.tts_length_scale)

    def warmup(self) -> None:
        for lang, voice in self.voices.items():
            for _ in voice.synthesize("Hello." if lang == "en" else "नमस्ते।", self._syn_config):
                pass

    def resolve_lang(self, lang: str) -> str:
        """Return a language we actually have a voice for."""
        if lang in self.voices:
            return lang
        return next(iter(self.voices))

    # -------------------------------------------------------------- inference
    async def stream(self, text: str, lang: str) -> AsyncIterator[tuple[int, bytes]]:
        """Yield `(sample_rate, int16_pcm_bytes)` chunks for `text`."""
        voice = self.voices[self.resolve_lang(lang)]
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()

        def push(item) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # event loop already closed (shutdown)
                pass

        def work() -> None:
            try:
                for chunk in voice.synthesize(text, self._syn_config):
                    if cancel.is_set():
                        break
                    push((chunk.sample_rate, chunk.audio_int16_bytes))
            except Exception as exc:  # surfaced to the caller below
                push(exc)
            finally:
                push(None)

        loop.run_in_executor(self._executor, work)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancel.set()  # stop the worker early if the consumer went away (barge-in)
