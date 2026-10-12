# Benchmark results

_Generated 2026-10-11 11:55 by `scripts/benchmark.py` on a free Google Colab T4 session._

## Setup

- **STT:** whisper-large-v3-turbo on cuda/float16
- **LLM:** gemma3:4b (Ollama)
- **TTS:** Piper, voices: en_US-lessac-medium, hi_IN-pratham-medium
- **OS:** Linux-6.6.122+-x86_64-with-glibc2.39
- **CPU cores:** 2
- **CPU:** Intel(R) Xeon(R) CPU @ 2.00GHz
- **RAM:** 12.7 GB
- **GPU:** Tesla T4, 15360 MiB

## Latency (milliseconds, median / p95 over all measured runs)

### English (6 runs)

| Stage | median | p95 |
|---|---:|---:|
| Speech to text | 676 | 692 |
| LLM first token (from request start) | 752 | 777 |
| First sentence ready for TTS | 909 | 1043 |
| First audio produced (server total) | 1236 | 1614 |
| Whole reply generated + synthesised | 1702 | 2200 |
| **Estimated perceived latency** (UI wait 650 ms + server first audio + ~50 ms) | **1936** | **2314** |

STT real-time factor: 0.19x (lower is faster).

### Hindi (6 runs)

| Stage | median | p95 |
|---|---:|---:|
| Speech to text | 750 | 774 |
| LLM first token (from request start) | 832 | 853 |
| First sentence ready for TTS | 1013 | 1094 |
| First audio produced (server total) | 1437 | 1948 |
| Whole reply generated + synthesised | 2172 | 2685 |
| **Estimated perceived latency** (UI wait 650 ms + server first audio + ~50 ms) | **2137** | **2648** |

STT real-time factor: 0.20x (lower is faster).

## Speech recognition sanity check (synthetic speech, so optimistic)

| Lang | Said | Heard |
|---|---|---|
| en | What is the capital of France, and what is it famous for? | What is the capital of France, and what is it famous for? |
| en | Tell me one interesting fact about the planet Mars. | tell me one interesting fact about the planet mars |
| hi | भारत की राजधानी क्या है और वहाँ कौन सी मशहूर जगहें हैं? | भारत की राजधानी क्या है और बहां कौन सी मश्वूर जगहें हैं। |
| hi | मुझे चाय बनाने का आसान तरीका बताओ। | मुझे चाय बनाने का आसान तरीका बताओ. |

## Multi-turn memory check

- **User:** Hi, my name is Asha and I live in Bengaluru.
  - heard as: Hi, my name is Asha and I live in Bengaluru.
  - **Assistant:** Hi Asha, it’s nice to meet you! Bengaluru is a fantastic city – I’ve heard great things.
- **User:** Can you suggest one nice thing to do there on a weekend?
  - heard as: Can you suggest one nice thing to do there on a weekend?
  - **Assistant:** How about visiting the Lal Bagh Botanical Garden? It’s a beautiful place to spend a relaxing weekend afternoon.
- **User:** Sorry, what was my name again, and which city did I say I live in?
  - heard as: Sorry, what was my name again, and which city did I say I live in?
  - **Assistant:** You said your name was Asha, and you live in Bengaluru. Just wanted to double-check!
