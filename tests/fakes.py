"""Tiny stand-ins for the real models so the orchestration logic can be tested anywhere."""
from __future__ import annotations

import asyncio
import numpy as np

from app.config import Settings
from app.stt import Transcript


class FakeSTT:
    model = object()
    model_name, device, compute_type = "fake", "cpu", "int8"

    def __init__(self, text="what is the capital of france", language="en"):
        self.text, self.language = text, language

    def load(self):
        pass

    def warmup(self):
        pass

    async def transcribe(self, audio, forced_language=None, session_language="en"):
        await asyncio.sleep(0.01)
        return Transcript(self.text, forced_language or self.language, 0.99, len(audio) / 16000, 10.0)


class FakeLLM:
    """Streams a scripted reply token by token; `delay` is the pause between tokens."""

    def __init__(self, reply="Paris is the capital of France. It is famous for the Eiffel Tower. Enjoy your trip!", delay=0.005):
        self.reply, self.delay = reply, delay
        self.calls: list[list[dict]] = []
        self.closed_early = False

    async def stream_chat(self, messages, max_tokens=None):
        self.calls.append(messages)
        try:
            for word in self.reply.split(" "):
                await asyncio.sleep(self.delay)
                yield word + " "
        except (asyncio.CancelledError, GeneratorExit):
            self.closed_early = True
            raise

    async def status(self):
        return {"ok": True, "model": "fake", "model_pulled": True}

    async def warmup(self):
        pass

    async def aclose(self):
        pass


class FakeTTS:
    """Emits 100 ms of silence (as 2 chunks) per sentence at 22.05 kHz."""

    voices = {"en": object(), "hi": object()}
    voice_names = {"en": "fake-en", "hi": "fake-hi"}

    def __init__(self, delay=0.005):
        self.delay = delay
        self.spoken: list[tuple[str, str]] = []

    def load(self):
        pass

    def warmup(self):
        pass

    def resolve_lang(self, lang):
        return lang if lang in self.voices else "en"

    async def stream(self, text, lang):
        self.spoken.append((lang, text))
        for _ in range(2):
            await asyncio.sleep(self.delay)
            yield 22050, np.zeros(1102, dtype="<i2").tobytes()  # ~50 ms


def make_pipeline(stt=None, llm=None, tts=None):
    from app.pipeline import VoicePipeline

    return VoicePipeline(Settings(), stt or FakeSTT(), llm or FakeLLM(), tts or FakeTTS())
