import json
import struct

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.server import create_app
from tests.fakes import FakeLLM, make_pipeline


def utterance(turn: int, seconds: float = 1.0) -> bytes:
    pcm = (np.random.default_rng(0).normal(0, 0.1, int(16000 * seconds)) * 32767).astype("<i2")
    return struct.pack("<I", turn) + pcm.tobytes()


def drain(ws, until="turn_done"):
    """Collect JSON events and binary audio frames until `until` arrives."""
    events, audio = [], []
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            audio.append(msg["bytes"])
        elif msg.get("text") is not None:
            ev = json.loads(msg["text"])
            events.append(ev)
            if ev["type"] == until:
                return events, audio


def client(tmp_path, pipeline=None):
    settings = Settings(log_dir=tmp_path)
    return TestClient(create_app(settings, pipeline or make_pipeline()))


def test_index_and_health(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/").status_code == 200
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/api/health").json()["llm"]["ok"] is True


def test_spoken_turn_roundtrip_and_latency_log(tmp_path):
    with client(tmp_path) as c, c.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["llm_model"]
        ws.send_bytes(utterance(turn=4))
        events, audio = drain(ws)
        assert any(e["type"] == "transcript" and e["turn"] == 4 for e in events)
        assert "".join(e["text"] for e in events if e["type"] == "llm_delta").startswith("Paris")
        assert audio, "binary audio frames expected"
        turn, rate = struct.unpack_from("<II", audio[0])
        assert (turn, rate) == (4, 22050) and len(audio[0]) > 8
    line = json.loads((tmp_path / "latency.jsonl").read_text().splitlines()[0])
    assert line["input"] == "audio" and line["first_audio_ms"] > 0


def test_typed_message_and_memory_and_reset(tmp_path):
    llm = FakeLLM(reply="Hello Asha, nice to meet you today.")
    with client(tmp_path, make_pipeline(llm=llm)) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "text", "turn": 1, "text": "my name is Asha"}))
        drain(ws)
        ws.send_text(json.dumps({"type": "text", "turn": 2, "text": "what is my name"}))
        drain(ws)
        assert len(llm.calls[1]) == 4  # system + previous exchange + new question
        ws.send_text(json.dumps({"type": "reset"}))
        assert drain(ws, "reset_done")[0][-1]["type"] == "reset_done"
        ws.send_text(json.dumps({"type": "text", "turn": 3, "text": "hello again"}))
        drain(ws)
        assert len(llm.calls[2]) == 2  # memory wiped: system + user only


def test_short_clips_are_ignored_and_interrupt_is_acknowledged(tmp_path):
    with client(tmp_path) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(utterance(1, seconds=0.05))  # too short: silently dropped
        ws.send_text(json.dumps({"type": "interrupt", "turn": 1, "played_ms": 0}))
        events, _ = drain(ws, "status")
        assert events[-1] == {"type": "status", "state": "idle"}


def test_language_config_is_applied(tmp_path):
    llm = FakeLLM()
    with client(tmp_path, make_pipeline(llm=llm)) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "config", "language": "hi"}))
        ws.send_bytes(utterance(1))
        events, _ = drain(ws)
        assert next(e for e in events if e["type"] == "transcript")["language"] == "hi"
        assert "Devanagari" in llm.calls[0][0]["content"]


def test_utterances_can_be_saved_for_debugging(tmp_path):
    import wave

    settings = Settings(log_dir=tmp_path, save_utterances=True)
    with TestClient(create_app(settings, make_pipeline())) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(utterance(turn=2))
        drain(ws)
    files = list((tmp_path / "utterances").glob("*_turn2.wav"))
    assert len(files) == 1
    with wave.open(str(files[0])) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1) and w.getnframes() == 16000
