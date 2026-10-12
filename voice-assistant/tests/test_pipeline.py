import asyncio

import numpy as np

from app.pipeline import AudioOut
from app.session import Session
from tests.fakes import FakeLLM, FakeSTT, FakeTTS, make_pipeline

AUDIO = np.zeros(16000, dtype=np.float32)


async def collect(pipe, session, **kw):
    events = []
    async for ev in pipe.run_turn(session, turn=1, **kw):
        events.append(ev)
    return events


def find(events, kind):
    return next(e for e in events if isinstance(e, dict) and e.get("type") == kind)


def kinds(events):
    return ["audio" if isinstance(e, AudioOut) else e["type"] for e in events]


async def test_full_turn_event_order_and_history():
    pipe, session = make_pipeline(), Session()
    events = await collect(pipe, session, audio=AUDIO)
    k = kinds(events)
    assert k[0] == "status" and events[0]["state"] == "transcribing"
    assert k.index("transcript") < k.index("llm_delta") < k.index("audio") < k.index("turn_done")
    done = find(events, "turn_done")
    assert done["completed"] and done["first_audio_ms"] <= done["total_ms"]
    assert done["stt_ms"] == 10.0
    assert [m["role"] for m in session.history] == ["user", "assistant"]
    assert "Eiffel Tower" in session.history[1]["content"]


async def test_audio_starts_before_the_llm_has_finished():
    # slow LLM: the first sentence must reach the speaker long before the last token arrives
    pipe = make_pipeline(llm=FakeLLM(delay=0.03))
    events = await collect(pipe, Session(), audio=AUDIO)
    done = find(events, "turn_done")
    assert done["first_audio_ms"] < done["llm_total_ms"] - 100


async def test_each_sentence_is_synthesised_separately():
    tts = FakeTTS()
    await collect(make_pipeline(tts=tts), Session(), audio=AUDIO)
    assert len(tts.spoken) >= 2
    assert all(lang == "en" for lang, _ in tts.spoken)


async def test_hindi_reply_picks_hindi_voice():
    tts = FakeTTS()
    llm = FakeLLM(reply="भारत की राजधानी नई दिल्ली है। वहाँ बहुत सी मशहूर जगहें हैं।")
    pipe = make_pipeline(stt=FakeSTT("भारत की राजधानी क्या है", "hi"), llm=llm, tts=tts)
    session = Session()
    await collect(pipe, session, audio=AUDIO)
    assert tts.spoken and all(lang == "hi" for lang, _ in tts.spoken)
    assert session.language == "hi"
    assert "Devanagari" in llm.calls[0][0]["content"]


async def test_context_is_sent_on_the_second_turn():
    llm = FakeLLM(reply="Nice to meet you, Asha. How can I help today?")
    pipe, session = make_pipeline(stt=FakeSTT("my name is asha"), llm=llm), Session()
    await collect(pipe, session, audio=AUDIO)
    pipe.stt.text = "what is my name"
    await collect(pipe, session, audio=AUDIO)
    second_call = llm.calls[1]
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]
    assert second_call[1]["content"] == "my name is asha"
    assert second_call[-1]["content"] == "what is my name"


async def test_empty_transcript_ends_turn_quietly():
    pipe, session = make_pipeline(stt=FakeSTT("")), Session()
    events = await collect(pipe, session, audio=AUDIO)
    assert kinds(events)[-2:] == ["turn_done", "status"]
    assert events[-2]["empty"] and session.history == []


async def test_typed_text_skips_stt():
    pipe, session = make_pipeline(), Session()
    events = await collect(pipe, session, text="hello there")
    assert events[0]["type"] == "transcript" and session.history[0]["content"] == "hello there"


async def test_llm_failure_is_reported_and_turn_terminates():
    class Boom(FakeLLM):
        async def stream_chat(self, messages, max_tokens=None):
            raise RuntimeError("model exploded")
            yield  # pragma: no cover

    events = await collect(make_pipeline(llm=Boom()), Session(), audio=AUDIO)
    err = find(events, "error")
    assert "model exploded" in err["message"]
    assert events[-1]["type"] == "status"


async def test_cancel_before_any_audio_stashes_the_users_words():
    llm = FakeLLM(delay=0.2)
    pipe, session = make_pipeline(llm=llm), Session()

    async def run():
        async for _ in pipe.run_turn(session, turn=1, audio=AUDIO):
            pass

    task = asyncio.create_task(run())
    await asyncio.sleep(0.15)  # STT done, LLM still thinking
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert session.history == []
    assert session.carry == "what is the capital of france"

    # ...and the next utterance is merged with it
    pipe.stt.text = "and also its population"
    llm.delay = 0.001
    await collect(pipe, session, audio=AUDIO)
    assert llm.calls[-1][-1]["content"] == "what is the capital of france and also its population"
    assert session.carry == ""


async def test_cancel_mid_reply_keeps_only_what_was_spoken_and_stops_the_llm():
    llm = FakeLLM(reply="One sentence goes here for you. " * 20, delay=0.01)
    pipe, session = make_pipeline(llm=llm), Session()
    got_audio = asyncio.Event()

    async def run():
        async for ev in pipe.run_turn(session, turn=5, audio=AUDIO):
            if isinstance(ev, AudioOut):
                got_audio.set()

    task = asyncio.create_task(run())
    await asyncio.wait_for(got_audio.wait(), 2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert llm.closed_early, "LLM generation must be cancelled on barge-in"
    reply = session.history[-1]["content"]
    assert reply and len(reply) < len(llm.reply)
    # the browser reports it only played the first sentence
    session.apply_playback_cut(5, played_ms=0)
    assert session.history[-1]["content"].count("One sentence") == 1
