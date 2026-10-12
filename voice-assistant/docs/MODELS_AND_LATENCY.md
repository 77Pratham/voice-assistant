# Models, hardware and latency

## Models selected

| Stage | Default | Why | Alternatives (set in `.env`) |
|---|---|---|---|
| STT | `faster-whisper` **small** on CPU (int8); **large-v3-turbo** auto-selected on CUDA | Multilingual (Hindi + English), CTranslate2 is ~4x faster than reference Whisper, Silero VAD included | `base` (faster, weaker Hindi), `medium` (better Hindi on CPU, slower) |
| LLM | **Gemma 3 4B** via Ollama | Good multilingual quality incl. Hindi at 4B, ~3 GB quantised, streams quickly | `llama3.2:3b`, `qwen2.5:3b` (faster, weaker Hindi), `gemma3:12b` (GPU) |
| TTS | **Piper** `en_US-lessac-medium`, `hi_IN-pratham-medium` | Runs in real time on CPU, tiny (~60 MB per voice), streams per sentence | any Piper voice via `VOICE_EN` / `VOICE_HI` |

All models are open-weight and run locally. No hosted inference API is used anywhere.

## Hardware

| Setup | What to expect |
|---|---|
| CPU only, 8 GB RAM, 4+ cores | Works with `WHISPER_MODEL=base` and a 3B LLM; expect multi-second replies |
| CPU only, 16 GB RAM (default config) | Comfortable memory-wise: roughly 1 GB for Whisper small, 3-4 GB for the LLM, 0.2 GB for Piper |
| NVIDIA GPU, 8 GB+ VRAM | Whisper large-v3-turbo + Gemma 3 4B fit together; this is the low-latency configuration |

Sizes above are approximate. Your measured numbers go below.

## Measuring latency

```bash
python scripts/benchmark.py            # headless, real models, writes docs/BENCHMARK_RESULTS.md
python scripts/summarize_latency.py    # p50/p95 from real browser sessions (logs/latency.jsonl)
```

The web UI also shows, for every reply, the *perceived* latency (end of your speech -> first sound) with a
breakdown: wait for end-of-speech, STT, LLM first sentence, TTS first audio, network/playback.

### Observed results

Measured with `python scripts/benchmark.py` on a free Google Colab T4 (full output in
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)). Medians over 6 runs per language, using synthetic test speech.

| Hardware | Language | STT | LLM first sentence (cumulative) | First audio, server-side | Estimated perceived* |
|---|---|---:|---:|---:|---:|
| Colab T4 (15 GB), 2 vCPU, 12.7 GB RAM | English | 676 ms | 909 ms | 1,236 ms | ~1.9 s |
| Colab T4 (15 GB), 2 vCPU, 12.7 GB RAM | Hindi | 750 ms | 1,013 ms | 1,437 ms | ~2.1 s |

\*Perceived = 650 ms end-of-speech wait + server first audio + an assumed 50 ms. In the author's browser sessions the
Cloudflare tunnel from Colab (US) to India added roughly another 0.5 s, shown in the app's latency panel as
"Network and playback start". STT real-time factor is about 0.2x.

For comparison, on the author's Windows laptop (CPU only, `llama3.2` + Whisper `base`) replies took roughly
7-8 seconds. That was an informal observation, not a benchmark run.

## Tuning knobs

- **End-of-speech wait** (UI slider, 300-1200 ms): the largest fixed component; lower = snappier, but may cut you off.
- `WHISPER_MODEL`: `base` roughly halves STT time vs `small`.
- `LLM_MODEL` / `LLM_MAX_TOKENS`: smaller model or shorter answers cut time to first sentence.
- `TTS_LENGTH_SCALE`: <1.0 speaks faster.
- GPU: `WHISPER_DEVICE=cuda`, Ollama uses the GPU automatically.
