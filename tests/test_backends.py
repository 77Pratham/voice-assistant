"""Tests for the real STT/TTS wrapper classes using stand-ins for the heavy model objects."""
import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.stt import SpeechToText
from app.tts import TextToSpeech


# ----------------------------------------------------------------------------- TTS
class FakeVoice:
    def __init__(self, n=5, delay=0.01, fail_at=None):
        self.n, self.delay, self.fail_at = n, delay, fail_at
        self.produced = 0

    def synthesize(self, text, syn_config=None):
        for i in range(self.n):
            if self.fail_at == i:
                raise RuntimeError("synthesis failed")
            time.sleep(self.delay)
            self.produced += 1
            yield SimpleNamespace(sample_rate=22050, audio_int16_bytes=b"\x00\x00" * 100)


def make_tts(voice, hi=None):
    tts = TextToSpeech(Settings())
    tts.voices = {"en": voice, **({"hi": hi} if hi else {})}
    return tts


async def test_tts_streams_all_chunks():
    voice = FakeVoice(n=3)
    chunks = [c async for c in make_tts(voice).stream("hello", "en")]
    assert len(chunks) == 3 and chunks[0] == (22050, b"\x00\x00" * 100)


async def test_tts_cancel_stops_the_worker_thread():
    voice = FakeVoice(n=200, delay=0.01)
    tts = make_tts(voice)
    gen = tts.stream("long text", "en")
    await gen.__anext__()
    await gen.aclose()  # barge-in
    await asyncio.sleep(0.2)
    assert voice.produced < 50, "worker should stop soon after the consumer goes away"


async def test_tts_propagates_errors():
    with pytest.raises(RuntimeError, match="synthesis failed"):
        async for _ in make_tts(FakeVoice(fail_at=1)).stream("x", "en"):
            pass


async def test_tts_does_not_block_the_event_loop():
    tts = make_tts(FakeVoice(n=5, delay=0.05))
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    _ = [c async for c in tts.stream("x", "en")]
    t.cancel()
    assert ticks > 10


def test_tts_falls_back_to_available_voice():
    tts = make_tts(FakeVoice())
    assert tts.resolve_lang("hi") == "en"


# ----------------------------------------------------------------------------- STT
class FakeWhisper:
    """Mimics faster_whisper.WhisperModel.transcribe for a scripted clip."""

    def __init__(self, detected="fr", probs=None, text="hello", voiced=2.0):
        self.detected, self.text, self.voiced = detected, text, voiced
        self.probs = probs or [("fr", 0.5), ("hi", 0.3), ("en", 0.2)]
        self.calls = []

    def transcribe(self, audio, language=None, **kw):
        self.calls.append(language)
        info = SimpleNamespace(
            language=language or self.detected,
            language_probability=0.9,
            all_language_probs=None if language else self.probs,
            duration_after_vad=self.voiced,
        )
        return [SimpleNamespace(text=f" {self.text} ")], info


def make_stt(model):
    stt = SpeechToText(Settings())
    stt.model = model
    return stt


AUDIO = np.zeros(16000, dtype=np.float32)


def test_stt_supported_language_decodes_once():
    m = FakeWhisper(detected="en")
    r = make_stt(m).transcribe_sync(AUDIO)
    assert (r.text, r.language) == ("hello", "en") and m.calls == [None]


def test_stt_unsupported_language_falls_back_to_best_supported():
    m = FakeWhisper(detected="fr")  # hi (0.3) beats en (0.2)
    r = make_stt(m).transcribe_sync(AUDIO, session_language="en")
    assert r.language == "hi" and m.calls == [None, "hi"]


def test_stt_session_language_breaks_near_ties():
    m = FakeWhisper(detected="fr", probs=[("fr", 0.6), ("hi", 0.21), ("en", 0.19)])
    r = make_stt(m).transcribe_sync(AUDIO, session_language="en")
    assert r.language == "en"  # 0.19 + 0.05 bonus > 0.21


def test_stt_forced_language_skips_detection():
    m = FakeWhisper(detected="fr")
    r = make_stt(m).transcribe_sync(AUDIO, forced_language="hi")
    assert r.language == "hi" and m.calls == ["hi"]


def test_stt_drops_known_hallucination_on_nearly_silent_clip():
    assert make_stt(FakeWhisper(detected="en", text="Thank you.", voiced=0.1)).transcribe_sync(AUDIO).text == ""
    assert make_stt(FakeWhisper(detected="en", text="Thank you.", voiced=1.5)).transcribe_sync(AUDIO).text == "Thank you."


async def test_stt_async_wrapper_runs_in_thread():
    ident = {}

    class M(FakeWhisper):
        def transcribe(self, *a, **k):
            ident["thread"] = threading.current_thread().name
            return super().transcribe(*a, **k)

    r = await make_stt(M(detected="en")).transcribe(AUDIO)
    assert r.text == "hello" and ident["thread"].startswith("stt")


def test_quiet_audio_is_boosted_but_silence_and_loud_audio_are_not():
    from app.stt import normalize_level

    quiet = (np.sin(np.linspace(0, 200, 16000)) * 0.02).astype(np.float32)
    assert abs(float(np.abs(normalize_level(quiet)).max()) - 0.5) < 1e-3
    silence = np.zeros(16000, dtype=np.float32)
    assert float(np.abs(normalize_level(silence)).max()) == 0.0
    loud = (np.sin(np.linspace(0, 200, 16000)) * 0.8).astype(np.float32)
    assert normalize_level(loud) is loud


def test_preload_cuda_libs_never_raises_when_libs_are_absent():
    from app.stt import preload_cuda_libs

    preload_cuda_libs()  # no nvidia wheels here: must be a silent no-op


def test_stt_warmup_gives_actionable_hint_for_missing_cuda_libs():
    class Broken(FakeWhisper):
        def transcribe(self, *a, **k):
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")

    with pytest.raises(RuntimeError, match="nvidia-cublas-cu12"):
        make_stt(Broken()).warmup()


def test_tts_downloads_missing_voices_with_no_whisper_flag(tmp_path, monkeypatch):
    import subprocess

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    TextToSpeech(Settings(voices_dir=tmp_path))._ensure_voices()
    assert calls and calls[0][-1] == "--no-whisper" and calls[0][1].endswith("download_models.py")


def test_tts_skips_download_when_voices_exist(tmp_path, monkeypatch):
    import subprocess

    s = Settings(voices_dir=tmp_path)
    for name in (s.voice_en, s.voice_hi):
        (tmp_path / f"{name}.onnx").write_bytes(b"x")
        (tmp_path / f"{name}.onnx.json").write_text("{}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("should not download"))
    TextToSpeech(s)._ensure_voices()


def test_tts_reports_download_failure_clearly(tmp_path, monkeypatch):
    import subprocess

    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="download_models.py"):
        TextToSpeech(Settings(voices_dir=tmp_path))._ensure_voices()
