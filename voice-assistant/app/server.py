"""FastAPI app: serves the web UI and the real-time voice WebSocket.

Wire protocol (all JSON frames are text frames):

  client -> server
    binary  : 4-byte little-endian turn id + 16 kHz mono int16 PCM  (one utterance)
    {"type":"text","turn":N,"text":"..."}        typed message (same pipeline, skips STT)
    {"type":"interrupt","turn":N,"played_ms":X}  barge-in: stop generating, X ms were heard
    {"type":"config","language":"auto|en|hi"}
    {"type":"reset"}                             clear conversation memory

  server -> client
    {"type":"ready", ...model info}
    {"type":"status","state":"transcribing|thinking|speaking|idle","turn":N}
    {"type":"transcript","text","language","turn":N}
    {"type":"llm_delta","text","turn":N}         streamed reply text
    {"type":"timing", stt_ms, first_sentence_ms, first_audio_ms, ...}
    {"type":"turn_done", ...totals}
    {"type":"error","message"}
    binary  : 4-byte turn id + 4-byte sample rate (LE uint32) + int16 mono PCM
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .pipeline import AudioOut, VoicePipeline
from .session import Session

log = logging.getLogger("voice.server")

STATIC_DIR = Path(__file__).parent / "static"
IN_HEADER = struct.Struct("<I")
OUT_HEADER = struct.Struct("<II")
MIN_SAMPLES = 16_000 // 5  # ignore clips shorter than 200 ms
MAX_SAMPLES = 16_000 * 60  # cap utterances at 60 s


def _pcm16_to_float(data: bytes) -> np.ndarray:
    data = data[: len(data) - (len(data) % 2)]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _save_wav(settings: Settings, audio: np.ndarray, turn: int) -> None:
    import wave

    try:
        folder = settings.log_dir / "utterances"
        folder.mkdir(parents=True, exist_ok=True)
        with wave.open(str(folder / f"{int(time.time() * 1000)}_turn{turn}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16_000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    except OSError:
        pass


def _log_latency(settings: Settings, event: dict, source: str) -> None:
    if event.get("empty") or "first_audio_ms" not in event:
        return
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        row = {"ts": round(time.time(), 3), "input": source, **event}
        with open(settings.log_dir / "latency.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:  # logging must never break a conversation
        pass


def create_app(settings: Settings | None = None, pipeline=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or VoicePipeline.create(settings)
        await app.state.pipeline.start()
        yield
        await app.state.pipeline.close()

    app = FastAPI(title="Open-weight voice assistant", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        return await app.state.pipeline.health()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        pipe = app.state.pipeline
        session = Session(max_messages=settings.max_history_messages)
        current: asyncio.Task | None = None

        async def run(turn: int, *, audio: np.ndarray | None = None, text: str | None = None):
            source = "audio" if audio is not None else "text"
            try:
                async for ev in pipe.run_turn(session, turn=turn, audio=audio, text=text):
                    if isinstance(ev, AudioOut):
                        await ws.send_bytes(OUT_HEADER.pack(turn, ev.sample_rate) + ev.pcm)
                    else:
                        if ev.get("type") == "turn_done":
                            _log_latency(settings, ev, source)
                        await ws.send_json({**ev, "turn": turn})
            except asyncio.CancelledError:
                raise
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                log.exception("turn %s crashed", turn)
                try:
                    await ws.send_json({"type": "error", "message": str(exc), "turn": turn})
                except Exception:
                    pass

        async def cancel_current() -> None:
            nonlocal current
            task, current = current, None
            if task and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        try:
            await ws.send_json({"type": "ready", **pipe.info()})
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                data = msg.get("bytes")
                if data is not None:
                    if len(data) < IN_HEADER.size + 2 * MIN_SAMPLES:
                        continue
                    (turn,) = IN_HEADER.unpack_from(data)
                    audio = _pcm16_to_float(data[IN_HEADER.size:])[:MAX_SAMPLES]
                    log.info(
                        "utterance turn=%d %.1fs level(rms)=%.4f peak=%.3f",
                        turn, len(audio) / 16000, float(np.sqrt(np.mean(audio**2))), float(np.abs(audio).max()),
                    )
                    if settings.save_utterances:
                        _save_wav(settings, audio, turn)
                    await cancel_current()
                    current = asyncio.create_task(run(turn, audio=audio))
                    continue

                raw = msg.get("text")
                if raw is None:
                    continue
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = body.get("type")

                if kind == "text":
                    await cancel_current()
                    current = asyncio.create_task(
                        run(int(body.get("turn", 0)), text=str(body.get("text", "")))
                    )
                elif kind == "interrupt":
                    await cancel_current()
                    session.apply_playback_cut(
                        int(body.get("turn", -1)), float(body.get("played_ms", 0.0))
                    )
                    await ws.send_json({"type": "status", "state": "idle"})
                elif kind == "config":
                    lang = body.get("language", "auto")
                    session.forced_language = lang if lang in settings.languages else None
                elif kind == "reset":
                    await cancel_current()
                    session.reset()
                    await ws.send_json({"type": "reset_done"})
        except WebSocketDisconnect:
            pass
        finally:
            await cancel_current()

    return app


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
