"""The STT -> LLM -> TTS orchestrator.

`VoicePipeline.run_turn` is an async generator that yields JSON-able event dicts and
`AudioOut` chunks. Both the WebSocket server and the offline benchmark consume it, so
what we measure is exactly what users get.

Latency design (see docs/ARCHITECTURE.md):

    audio ──► STT ──► LLM tokens ──► SentenceChunker ──► TTS ──► PCM chunks
                         │  (streaming)      │   (queue)       │  (streaming)
                         └── runs concurrently with TTS: sentence N+1 is generated
                             while sentence N is being synthesised and played.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from .config import Settings
from .llm import LLMError, OllamaClient
from .session import Session
from .stt import SpeechToText
from .text_utils import SentenceChunker, ThinkStripper, clean_for_speech, script_language
from .tts import TextToSpeech

log = logging.getLogger("voice.pipeline")

_DONE = object()


@dataclass
class AudioOut:
    sample_rate: int
    pcm: bytes  # int16 mono


def _friendly_error(exc: Exception) -> str:
    import httpx

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "Can't reach the language model. Is `ollama serve` running?"
    if isinstance(exc, LLMError):
        return f"Language model error: {exc}"
    return f"{type(exc).__name__}: {exc}"


class VoicePipeline:
    def __init__(self, settings: Settings, stt: SpeechToText, llm: OllamaClient, tts: TextToSpeech):
        self.settings = settings
        self.stt = stt
        self.llm = llm
        self.tts = tts

    @classmethod
    def create(cls, settings: Settings) -> "VoicePipeline":
        return cls(settings, SpeechToText(settings), OllamaClient(settings), TextToSpeech(settings))

    # --------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Load every model and run a dummy inference so the first user turn is fast."""
        t0 = time.perf_counter()
        # TTS first: it is quick and fails with an actionable message if the voices are missing.
        await asyncio.to_thread(self.tts.load)
        await asyncio.to_thread(self.stt.load)
        await asyncio.gather(asyncio.to_thread(self.stt.warmup), asyncio.to_thread(self.tts.warmup))
        status = await self.llm.status()
        if not status.get("ok"):
            log.error("Ollama is not reachable at %s: %s", self.settings.ollama_url, status.get("error"))
        elif not status.get("model_pulled"):
            log.error("Model '%s' is not pulled. Run: ollama pull %s", status["model"], status["model"])
        else:
            try:
                await self.llm.warmup()
            except Exception as exc:  # warm-up is best effort
                log.warning("LLM warm-up failed: %s", exc)
        log.info("Pipeline ready in %.1fs", time.perf_counter() - t0)

    async def close(self) -> None:
        await self.llm.aclose()

    def info(self) -> dict:
        return {
            "stt_model": f"whisper-{self.stt.model_name}",
            "stt_device": f"{self.stt.device}/{self.stt.compute_type}",
            "llm_model": self.settings.llm_model,
            "voices": dict(self.tts.voice_names),
            "languages": list(self.settings.languages),
        }

    async def health(self) -> dict:
        return {
            "stt": self.stt.model is not None,
            "tts_voices": dict(self.tts.voice_names),
            "llm": await self.llm.status(),
        }

    # -------------------------------------------------------------------- turn
    async def run_turn(
        self,
        session: Session,
        *,
        turn: int,
        audio: np.ndarray | None = None,
        text: str | None = None,
    ) -> AsyncIterator[dict | AudioOut]:
        t0 = time.perf_counter()

        def ms() -> float:
            return round((time.perf_counter() - t0) * 1000, 1)

        timing: dict = {}

        # ---- 1. speech -> text ------------------------------------------------
        if audio is not None:
            yield {"type": "status", "state": "transcribing"}
            tr = await self.stt.transcribe(audio, session.forced_language, session.language)
            timing["stt_ms"] = round(tr.ms, 1)
            timing["audio_in_s"] = round(tr.audio_s, 2)
            user_text, lang = tr.text.strip(), tr.language
            log.info("heard [%s] %r (stt %.0f ms, %.1fs audio)", lang, user_text, tr.ms, tr.audio_s)
            if not user_text:
                yield {"type": "transcript", "text": "", "language": lang, "empty": True}
                yield {"type": "turn_done", "empty": True}
                yield {"type": "status", "state": "idle"}
                return
        else:
            user_text = (text or "").strip()
            lang = session.forced_language or script_language(user_text, session.language)
            timing["stt_ms"] = 0.0
            if not user_text:
                yield {"type": "turn_done", "empty": True}
                yield {"type": "status", "state": "idle"}
                return

        session.language = lang
        yield {"type": "transcript", "text": user_text, "language": lang}

        llm_user_text = f"{session.carry} {user_text}".strip() if session.carry else user_text
        messages = session.messages_for(llm_user_text, lang)

        # ---- 2+3. LLM -> sentences -> TTS, running concurrently ----------------
        yield {"type": "status", "state": "thinking"}

        out: asyncio.Queue = asyncio.Queue()
        sentences: asyncio.Queue = asyncio.Queue()
        spoken: list[tuple[str, float]] = []  # (sentence, audio seconds preceding it)
        stats = {"audio_out_s": 0.0, "reply_chars": 0, "completed": False}

        def mark_first_sentence() -> None:
            timing.setdefault("first_sentence_ms", ms())

        async def llm_task() -> None:
            chunker, stripper = SentenceChunker(), ThinkStripper()

            async def emit(piece: str) -> None:
                if not piece:
                    return
                stats["reply_chars"] += len(piece)
                await out.put({"type": "llm_delta", "text": piece})
                for s in chunker.feed(piece):
                    mark_first_sentence()
                    await sentences.put(s)

            try:
                async with aclosing(self.llm.stream_chat(messages)) as stream:
                    async for token in stream:
                        timing.setdefault("llm_first_token_ms", ms())
                        await emit(stripper.feed(token))
                await emit(stripper.flush())
                for s in chunker.flush():
                    mark_first_sentence()
                    await sentences.put(s)
                timing["llm_total_ms"] = ms()
            finally:
                sentences.put_nowait(None)

        async def tts_task() -> None:
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    return
                clean = clean_for_speech(sentence)
                if not clean:
                    continue
                voice_lang = self.tts.resolve_lang(script_language(clean, lang))
                started = False
                async with aclosing(self.tts.stream(clean, voice_lang)) as stream:
                    async for sample_rate, pcm in stream:
                        if not started:
                            started = True
                            spoken.append((clean, round(stats["audio_out_s"], 3)))
                            if "first_audio_ms" not in timing:
                                timing["first_audio_ms"] = ms()
                                await out.put({"type": "status", "state": "speaking"})
                                await out.put({"type": "timing", **timing})
                        stats["audio_out_s"] += len(pcm) / 2 / sample_rate
                        await out.put(AudioOut(sample_rate, pcm))

        async def supervise() -> None:
            llm_t = asyncio.create_task(llm_task())
            tts_t = asyncio.create_task(tts_task())
            try:
                await llm_t
                await tts_t
                stats["completed"] = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Turn %s failed", turn)
                out.put_nowait({"type": "error", "message": _friendly_error(exc)})
            finally:
                for t in (llm_t, tts_t):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(llm_t, tts_t, return_exceptions=True)
                out.put_nowait(_DONE)

        supervisor = asyncio.create_task(supervise())
        try:
            while True:
                item = await out.get()
                if item is _DONE:
                    break
                yield item
            timing["total_ms"] = ms()
            yield {
                "type": "turn_done",
                "empty": False,
                "completed": stats["completed"],
                "audio_out_s": round(stats["audio_out_s"], 2),
                "reply_chars": stats["reply_chars"],
                **timing,
            }
            yield {"type": "status", "state": "idle"}
        finally:
            # Runs on normal completion *and* when the turn is cancelled by a barge-in.
            if not supervisor.done():
                supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)
            if spoken:
                session.record_exchange(turn, llm_user_text, spoken)
            elif not stats["completed"]:
                session.stash_unanswered(llm_user_text)
            else:
                session.carry = ""
