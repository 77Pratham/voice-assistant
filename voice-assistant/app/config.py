"""Central configuration.

Every knob can be set through an environment variable (or a `.env` file in the
project root). Defaults are chosen so that `python -m app.server` works on a
laptop after running `scripts/download_models.py` and `ollama pull gemma3:4b`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- web server ---------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # --- speech-to-text (faster-whisper) ------------------------------------
    # "auto" -> large-v3-turbo on a CUDA GPU, "small" on CPU.
    whisper_model: str = "auto"
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_compute_type: str = "auto"  # auto -> float16 (cuda) / int8 (cpu)
    whisper_beam_size: int = 0  # 0 = auto: beam 5 on GPU (more accurate), 1 (greedy) on CPU

    # --- language model (Ollama) --------------------------------------------
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "gemma3:4b"
    llm_keep_alive: str = "30m"  # keep weights in memory between turns
    llm_num_ctx: int = 4096
    llm_temperature: float = 0.6
    llm_max_tokens: int = 220  # voice answers should be short

    # --- conversation context -----------------------------------------------
    max_history_messages: int = 24  # user+assistant messages kept verbatim

    # --- text-to-speech (Piper) ---------------------------------------------
    voices_dir: Path = ROOT / "models" / "piper"
    voice_en: str = "en_US-lessac-medium"
    voice_hi: str = "hi_IN-pratham-medium"
    tts_length_scale: float = 1.0  # <1 faster speech, >1 slower
    tts_use_cuda: bool = False

    # --- misc ---------------------------------------------------------------
    languages: tuple[str, ...] = ("en", "hi")
    log_dir: Path = ROOT / "logs"
    save_utterances: bool = False  # write each received clip to logs/utterances/*.wav for debugging

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=_str("HOST", cls.host),
            port=_int("PORT", cls.port),
            whisper_model=_str("WHISPER_MODEL", cls.whisper_model),
            whisper_device=_str("WHISPER_DEVICE", cls.whisper_device),
            whisper_compute_type=_str("WHISPER_COMPUTE_TYPE", cls.whisper_compute_type),
            whisper_beam_size=_int("WHISPER_BEAM_SIZE", cls.whisper_beam_size),
            ollama_url=_str("OLLAMA_URL", cls.ollama_url).rstrip("/"),
            llm_model=_str("LLM_MODEL", cls.llm_model),
            llm_keep_alive=_str("LLM_KEEP_ALIVE", cls.llm_keep_alive),
            llm_num_ctx=_int("LLM_NUM_CTX", cls.llm_num_ctx),
            llm_temperature=_float("LLM_TEMPERATURE", cls.llm_temperature),
            llm_max_tokens=_int("LLM_MAX_TOKENS", cls.llm_max_tokens),
            max_history_messages=_int("MAX_HISTORY_MESSAGES", cls.max_history_messages),
            voices_dir=Path(_str("VOICES_DIR", str(cls.voices_dir))),
            voice_en=_str("VOICE_EN", cls.voice_en),
            voice_hi=_str("VOICE_HI", cls.voice_hi),
            tts_length_scale=_float("TTS_LENGTH_SCALE", cls.tts_length_scale),
            tts_use_cuda=_bool("TTS_USE_CUDA", cls.tts_use_cuda),
            log_dir=Path(_str("LOG_DIR", str(cls.log_dir))),
            save_utterances=_bool("SAVE_UTTERANCES", cls.save_utterances),
        )
