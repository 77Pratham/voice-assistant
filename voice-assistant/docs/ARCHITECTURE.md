# Architecture

![Architecture: STT -> LLM -> TTS](architecture.svg)

## One turn, step by step

1. **Browser VAD (stage 1).** The microphone is captured with an AudioWorklet, down-sampled to 16 kHz and
   gated by an adaptive energy detector (noise floor tracking, hysteresis, 380 ms pre-roll, configurable
   hang-over). One complete utterance is sent over the WebSocket.
2. **STT.** `faster-whisper` transcribes it. Silero VAD (built into faster-whisper) trims silence so Whisper
   does not hallucinate on noise (stage 2). The language is auto-detected and restricted to English/Hindi.
3. **LLM.** The transcript plus the rolling history goes to Ollama, which streams tokens back.
4. **Sentence chunker + TTS.** As soon as one sentence (or a first clause of ~32 chars) is complete it is
   handed to Piper while the LLM keeps generating. Audio is streamed to the browser as raw PCM per sentence.
5. **Playback.** The browser schedules PCM buffers back-to-back (gapless) on the Web Audio clock.

The first sound therefore depends on *STT + time to the first sentence + one sentence of TTS*, not on the
length of the full answer.

## Design decisions

| Decision | Why |
|---|---|
| Sentence-level pipelining of LLM and TTS | Biggest single latency win; reply length no longer matters for time-to-first-audio |
| Two-stage VAD | The browser decides when you stopped talking (cheap, instant); Silero on the server keeps Whisper honest |
| Barge-in cancels the asyncio task | Closing the Ollama HTTP stream stops generation; the Piper worker thread is signalled and stops after its current chunk |
| History records what was *heard* | The browser reports `played_ms` on interruption; the server trims the stored reply to the sentences that actually started playing |
| Unanswered speech is carried over | If you cut in before hearing anything (e.g. a thinking pause), your earlier words are merged into the next utterance |
| TTS voice chosen per sentence by script | Devanagari text -> Hindi voice, otherwise English; more reliable than trusting the STT language tag |
| Prompt forces short, markdown-free replies | Spoken answers should be 1-3 sentences; `clean_for_speech` strips anything that slips through |
| Models warmed at startup, `keep_alive` on Ollama | First user turn does not pay model-load cost |
| Browser UI shows the latency breakdown | Makes the optimisation visible and measurable during the demo |

## WebSocket protocol

See the docstring at the top of `app/server.py`. In short: binary uplink = 4-byte turn id + 16 kHz int16 PCM;
binary downlink = turn id + sample rate + int16 PCM; JSON frames carry status, transcript, streamed LLM text,
timings and control messages (`interrupt`, `config`, `reset`). Every frame carries a turn id so stale audio
from an interrupted turn is dropped on arrival.

## Known limitations

- STT is utterance-based (not streaming partials); the end-of-speech wait (default 650 ms) is part of perceived latency.
- Energy-based stage-1 VAD can be fooled by loud background speech; adjust *sensitivity* in the UI.
- Barge-in relies on browser echo cancellation; headphones are the most reliable setup.
- Piper Hindi voices are clear but less expressive than the English ones. Romanised Hindi ("Hinglish") is read with the English voice.
- One conversation per WebSocket; no authentication (intended for local / SSH-tunnelled use).
