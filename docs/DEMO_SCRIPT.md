# 2-3 minute demo video script

Record the browser window plus your voice (OBS or Windows Game Bar). Use a wired headset mic so interruptions
are clean, set the Windows input volume to 80-100%, and keep the latency panel visible.

| Time | On screen | Say / do |
|---|---|---|
| 0:00-0:20 | Face-cam (optional), then the architecture diagram | "A voice assistant built entirely on open-weight models: Whisper large-v3-turbo for speech to text, Gemma 3 4B through Ollama, and Piper for the voice. It runs on a free Colab T4 GPU that I set up myself, with no hosted AI APIs." |
| 0:20-0:55 | App, microphone on | Ask "What's a good name for a coffee shop in Bengaluru?", then "Make it sound more modern." Shows streaming text, voice, and context |
| 0:55-1:25 | Same page | Ask in Hindi: "भारत की राजधानी क्या है?" then a follow-up. Language is detected automatically, reply is Hindi text and a Hindi voice |
| 1:25-1:55 | Same page | Ask "Tell me about the history of Bengaluru", talk over it after a sentence or two. Audio stops at once; ask "what was I asking about?" to show it remembers only what was heard |
| 1:55-2:20 | Latency panel | Read the number off the panel: wait for end of speech, speech to text, LLM first sentence, voice synthesis, network |
| 2:20-2:40 | GitHub repo, README | Mention the tests, the benchmark script, and be upfront about the tunnel delay and small Hindi recognition errors |
