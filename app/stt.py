"""Speech-to-text using faster-whisper (CTranslate2 build of OpenAI Whisper)."""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .config import Settings

log = logging.getLogger("voice.stt")

SAMPLE_RATE = 16_000

# Whisper is known to "hear" these on near-silent clips. We only discard them when the
# voiced part of the clip is tiny, so a genuine "thank you" still goes through.
_HALLUCINATIONS = {
    "you", "thank you", "thanks", "thanks for watching", "bye", "okay", "ok",
    "subtitles by the amara.org community",
}


def preload_cuda_libs() -> None:
    """pip's `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels put their libraries under
    site-packages/nvidia/*/lib, a folder the dynamic loader does not search, so CTranslate2 fails
    with "libcublas.so.12 is not found". Loading them by path first makes them resolvable."""
    if not sys.platform.startswith("linux"):
        return
    import ctypes
    import glob
    import importlib.util

    wanted = (
        ("nvidia.cublas", ("libcublasLt.so.12*", "libcublas.so.12*")),
        ("nvidia.cudnn", ("libcudnn.so.9*",)),
    )
    for package, patterns in wanted:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            continue
        for location in (spec.submodule_search_locations or []) if spec else []:
            for pattern in patterns:
                for path in sorted(glob.glob(f"{location}/lib/{pattern}")):
                    try:
                        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    except OSError as exc:
                        log.debug("could not preload %s: %s", path, exc)


def normalize_level(audio: np.ndarray) -> np.ndarray:
    """Quiet microphones are the most common reason Whisper 'hears' nothing. Bring clips whose
    peak is low up to a healthy level; leave normal and silent clips alone."""
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if 0.003 < peak < 0.3:
        return (audio * (0.5 / peak)).astype(np.float32)
    return audio


@dataclass
class Transcript:
    text: str
    language: str
    language_prob: float
    audio_s: float
    ms: float


class SpeechToText:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.model_name = ""
        self.device = ""
        self.compute_type = ""
        # One worker: CTranslate2 releases the GIL, but serialising keeps latency predictable.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    # ------------------------------------------------------------------ setup
    def _resolve_device(self) -> tuple[str, str]:
        device = self.settings.whisper_device
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute = self.settings.whisper_compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return device, compute

    def load(self) -> None:
        from faster_whisper import WhisperModel

        device, compute = self._resolve_device()
        if device == "cuda":
            preload_cuda_libs()
        name = self.settings.whisper_model
        if name == "auto":
            name = "large-v3-turbo" if device == "cuda" else "small"
        log.info("Loading Whisper '%s' on %s (%s)...", name, device, compute)
        self.model = WhisperModel(name, device=device, compute_type=compute)
        self.model_name, self.device, self.compute_type = name, device, compute

    def warmup(self) -> None:
        try:
            self.transcribe_sync(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except RuntimeError as exc:
            if "libcu" in str(exc) or "CUDA" in str(exc):
                raise RuntimeError(
                    f"{exc}\nGPU libraries are missing. Run: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
                    "(or set WHISPER_DEVICE=cpu)."
                ) from exc
            raise

    # -------------------------------------------------------------- inference
    def _run(self, audio: np.ndarray, language: str | None):
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=self.settings.whisper_beam_size or (5 if self.device == "cuda" else 1),
            temperature=(0.0, 0.2, 0.4),  # retry a bad decode instead of returning garbage
            condition_on_previous_text=False,
            without_timestamps=True,
            no_speech_threshold=0.6,
            # Stage 2 of our two-stage VAD (stage 1 is the browser): Silero VAD trims
            # silence/noise so Whisper only sees speech and does not hallucinate.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info

    def transcribe_sync(
        self,
        audio: np.ndarray,
        forced_language: str | None = None,
        session_language: str = "en",
    ) -> Transcript:
        t0 = time.perf_counter()
        audio = normalize_level(audio)
        langs = self.settings.languages
        forced = forced_language if forced_language in langs else None

        text, info = self._run(audio, forced)
        language = forced or info.language
        prob = 1.0 if forced else info.language_probability

        if not forced and language not in langs:
            # Whisper guessed a language we don't support (common on short clips).
            # Pick the likelier of our supported languages, nudged towards the
            # language used earlier in the conversation, and decode again.
            probs = dict(info.all_language_probs or [])
            language = max(
                langs, key=lambda l: probs.get(l, 0.0) + (0.05 if l == session_language else 0.0)
            )
            prob = probs.get(language, 0.0)
            text, info = self._run(audio, language)

        voiced = getattr(info, "duration_after_vad", None) or 0.0
        if text.lower().strip(" .!?,") in _HALLUCINATIONS and voiced < 0.5:
            text = ""

        return Transcript(
            text=text,
            language=language,
            language_prob=float(prob),
            audio_s=len(audio) / SAMPLE_RATE,
            ms=(time.perf_counter() - t0) * 1000,
        )

    async def transcribe(
        self,
        audio: np.ndarray,
        forced_language: str | None = None,
        session_language: str = "en",
    ) -> Transcript:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.transcribe_sync, audio, forced_language, session_language
        )
