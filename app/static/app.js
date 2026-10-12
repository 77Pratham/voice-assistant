"use strict";
/* Browser client: mic capture -> voice-activity detection -> WebSocket -> streamed audio playback.
 *
 * Stage-1 VAD lives here (adaptive energy gate with hysteresis, pre-roll and hang-over).
 * Stage-2 VAD (Silero) runs on the server inside faster-whisper.
 * Barge-in: speech detected while the assistant is busy stops playback immediately and tells the
 * server how many milliseconds were really heard, so its memory matches what the user heard.
 */

const $ = (s) => document.querySelector(s);

const RATE = 16000;
const FRAME = 512;                 // samples per VAD frame at 16 kHz
const FRAME_MS = (FRAME / RATE) * 1000;   // 32 ms
const PREROLL = 12;                // frames kept before speech onset (~380 ms)
const START_FRAMES = 3;            // consecutive loud frames to start an utterance (~96 ms)
const BARGE_FRAMES = 8;            // stricter while the assistant is talking (~256 ms)
const MIN_SPEECH_FRAMES = 8;       // ignore blips shorter than ~256 ms
const MAX_FRAMES = Math.round(30000 / FRAME_MS);   // hard cap: 30 s per utterance
const TAIL_FRAMES = 6;             // trailing silence kept in the clip (~190 ms)
const MIN_START_BY_SENS = { 1: 0.035, 2: 0.025, 3: 0.017, 4: 0.011, 5: 0.007 };

const cfg = { hangMs: 650, sens: 3, barge: true, lang: "auto" };
try { Object.assign(cfg, JSON.parse(localStorage.getItem("va-cfg") || "{}")); } catch (_) {}
const saveCfg = () => { try { localStorage.setItem("va-cfg", JSON.stringify(cfg)); } catch (_) {} };

const S = {
  ws: null, ctx: null, player: null, stream: null, node: null, src: null, mute: null,
  workletReady: false, running: false, rs: null,
  turn: 0, status: "idle", meta: {}, totals: [], bot: null, frame: new Float32Array(FRAME), fi: 0,
};
const vad = { noise: 0.006, inSpeech: false, hot: 0, silence: 0, speech: 0, frames: [], pre: [], lastVoiceAt: 0, level: 0 };

/* ------------------------------------------------------------------ utilities */
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const fmt = (ms) => Math.round(ms).toLocaleString("en-US");
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
function send(obj) {
  if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(obj));
}

/* ------------------------------------------------------------------ playback */
class Player {
  constructor(ctx) {
    this.ctx = ctx;
    this.sources = new Set();
    this.next = 0;
    this.items = [];            // this turn's buffers: {start, dur, off}
    this.offset = 0;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.connect(ctx.destination);
    this.scratch = new Float32Array(this.analyser.fftSize);
    this.onDrain = () => {};
  }
  get playing() { return this.sources.size > 0; }
  resetTurn() { this.items = []; this.offset = 0; }
  enqueue(rate, int16) {
    const buf = this.ctx.createBuffer(1, int16.length, rate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.analyser);
    const start = Math.max(this.ctx.currentTime + 0.03, this.next);
    src.start(start);
    this.next = start + buf.duration;
    this.items.push({ start, dur: buf.duration, off: this.offset });
    this.offset += buf.duration;
    this.sources.add(src);
    src.onended = () => { this.sources.delete(src); if (!this.sources.size) this.onDrain(); };
    return start;
  }
  playedMs() {
    const now = this.ctx.currentTime;
    let played = 0;
    for (const it of this.items) {
      if (now >= it.start + it.dur) played = it.off + it.dur;
      else { if (now > it.start) played = it.off + (now - it.start); break; }
    }
    return played * 1000;
  }
  stop() {
    for (const s of this.sources) { s.onended = null; try { s.stop(); } catch (_) {} }
    this.sources.clear();
    this.next = 0;
  }
  level() {
    this.analyser.getFloatTimeDomainData(this.scratch);
    let sum = 0;
    for (const v of this.scratch) sum += v * v;
    return Math.sqrt(sum / this.scratch.length);
  }
}

function ensureAudio() {
  if (!S.ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    S.ctx = new AC({ latencyHint: "interactive" });
    S.player = new Player(S.ctx);
    S.player.onDrain = render;
  }
  if (S.ctx.state === "suspended") S.ctx.resume();
}

/* ------------------------------------------------------------------ websocket */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { setConn(true); send({ type: "config", language: cfg.lang }); };
  ws.onclose = () => { setConn(false); S.status = "idle"; render(); setTimeout(connect, 1500); };
  ws.onmessage = onMessage;
  S.ws = ws;
}
function setConn(up) {
  $("#conn").dataset.state = up ? "up" : "down";
  $("#connText").textContent = up ? "Connected" : "Reconnecting…";
}

function onMessage(e) {
  if (typeof e.data !== "string") return onAudio(e.data);
  const m = JSON.parse(e.data);
  if (m.type === "ready") {
    const voices = Object.values(m.voices || {}).join(", ");
    $("#models").textContent = `${m.stt_model} (${m.stt_device})  →  ${m.llm_model} via Ollama  →  Piper ${voices}`;
    return;
  }
  if (m.type === "error") {
    addMessage("err", m.message);
    if (m.turn === undefined || m.turn === S.turn) S.status = "idle";
    return render();
  }
  if (m.turn !== undefined && m.turn !== S.turn) return;   // stale event from an interrupted turn
  switch (m.type) {
    case "status": S.status = m.state; break;
    case "transcript":
      if (m.empty) addMessage("note", "Didn't catch that. Watch the level bar while you speak; if it barely moves, raise the microphone sensitivity or check your input device.");
      else addMessage("user", m.text, m.language);
      break;
    case "llm_delta": appendBot(m.text); break;
    case "timing": (S.meta[m.turn] ||= {}).server = m; break;
    case "turn_done": break;
  }
  render();
}

function onAudio(buf) {
  const dv = new DataView(buf);
  const turn = dv.getUint32(0, true);
  const rate = dv.getUint32(4, true);
  if (turn !== S.turn || !S.player) return;
  const pcm = new Int16Array(buf, 8, (buf.byteLength - 8) >> 1);
  S.player.enqueue(rate, pcm);
  const meta = (S.meta[turn] ||= {});
  if (!meta.firstAudioAt) {
    meta.firstAudioAt = performance.now();
    showTrace(meta);
  }
  render();
}

/* ------------------------------------------------------------------ conversation log */
function hideEmpty() { const e = $("#empty"); if (e) e.remove(); }
function addMessage(kind, text, lang) {
  hideEmpty();
  const li = el("li", `msg ${kind}`);
  const t = el("span", "text", text);
  li.append(t);
  if (lang) { const l = el("span", "lang", lang); l.title = lang === "hi" ? "Hindi" : "English"; li.append(l); }
  $("#log").append(li);
  scrollLog();
  return { li, t };
}
function appendBot(text) {
  if (!S.bot) S.bot = addMessage("bot", "");
  S.bot.t.textContent += text;
  scrollLog();
}
function scrollLog() { const l = $("#log"); l.scrollTop = l.scrollHeight; }

/* ------------------------------------------------------------------ latency trace */
const SEGMENTS = [
  ["wait", "Waiting for you to stop talking", "var(--wait)"],
  ["stt", "Speech to text", "var(--stt)"],
  ["llm", "Language model, first sentence", "var(--llm)"],
  ["tts", "Voice synthesis, first audio", "var(--tts)"],
  ["net", "Network and playback start", "var(--wait)"],
];
function showTrace(meta) {
  const sv = meta.server;
  if (!sv) return;
  const wait = meta.voiceEnd ? Math.max(0, meta.sent - meta.voiceEnd) : 0;
  const stt = sv.stt_ms || 0;
  const firstSentence = sv.first_sentence_ms ?? sv.first_audio_ms;
  const llm = Math.max(0, firstSentence - stt);
  const tts = Math.max(0, sv.first_audio_ms - firstSentence);
  const total = wait + (meta.firstAudioAt - meta.sent) + 30;
  const net = Math.max(0, total - wait - stt - llm - tts);
  const vals = { wait, stt, llm, tts, net };

  $("#totalNum").textContent = fmt(total);
  const wf = $("#wf"), legend = $("#legend");
  wf.replaceChildren(); legend.replaceChildren();
  for (const [key, label, color] of SEGMENTS) {
    const seg = el("div", "seg");
    seg.style.flexGrow = String(Math.max(vals[key], 1));
    seg.style.background = color;
    wf.append(seg);
    const li = el("li");
    const sw = el("i"); sw.style.background = color;
    li.append(sw, el("span", "", label), el("b", "", `${fmt(vals[key])} ms`));
    legend.append(li);
  }
  S.totals.push(total);
  const sorted = [...S.totals].sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  $("#median").textContent = `Median over ${S.totals.length} repl${S.totals.length === 1 ? "y" : "ies"}: ${fmt(med)} ms`;
}

/* ------------------------------------------------------------------ UI state */
function mode() {
  if (S.running && vad.inSpeech) return "hearing";
  if ((S.player && S.player.playing) || S.status === "speaking") return "speaking";
  if (S.status === "transcribing" || S.status === "thinking") return "thinking";
  return S.running ? "listening" : "off";
}
const LABELS = {
  off: ["Microphone is off", "Press the button to start. Headphones give the best interruption handling."],
  listening: ["Listening", "Say something in English or Hindi."],
  hearing: ["Hearing you", "Pause when you're done."],
  speaking: ["Speaking", "Start talking to interrupt."],
};
function render() {
  const m = mode();
  const mic = $("#mic");
  mic.dataset.mode = m;
  mic.setAttribute("aria-pressed", String(S.running));
  mic.setAttribute("aria-label", S.running ? "Stop conversation" : "Start conversation");
  let [label, hint] = LABELS[m] || ["", ""];
  if (m === "thinking") { label = S.status === "transcribing" ? "Transcribing…" : "Thinking…"; hint = "Speak to cancel and say something else."; }
  if (m === "speaking" && !cfg.barge) hint = "Interrupting is turned off in settings.";
  $("#stateLabel").textContent = label;
  $("#stateHint").textContent = hint;
}
function frameLoop() {
  let lvl = 0;
  if (S.running) lvl = Math.max(lvl, vad.level * 9);
  if (S.player && S.player.playing) lvl = Math.max(lvl, S.player.level() * 4);
  document.documentElement.style.setProperty("--lvl", clamp(lvl, 0, 1).toFixed(3));
  const fill = $("#meterFill");
  fill.style.width = `${S.running ? Math.round(clamp(vad.level / 0.08, 0, 1) * 100) : 0}%`;
  fill.dataset.hot = vad.inSpeech ? "1" : "0";
  requestAnimationFrame(frameLoop);
}

/* ------------------------------------------------------------------ microphone + VAD */
class Resampler {
  // Box-filter decimation (anti-aliased enough for speech) from the context rate down to 16 kHz.
  constructor(inRate, outRate) {
    this.pass = inRate === outRate;
    this.step = inRate / outRate;
    this.sum = 0; this.cnt = 0; this.pos = 0; this.next = this.step;
  }
  process(input) {
    if (this.pass) return input;
    const out = [];
    for (let i = 0; i < input.length; i++) {
      this.sum += input[i]; this.cnt++; this.pos++;
      if (this.pos >= this.next) {
        out.push(this.sum / this.cnt);
        this.sum = 0; this.cnt = 0; this.next += this.step;
      }
    }
    if (this.pos > 1e6) { this.pos -= 1e6; this.next -= 1e6; }
    return out;
  }
}

async function startMic() {
  ensureAudio();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return addMessage("err", "This page needs HTTPS (or localhost) before the browser will allow microphone access. You can still type messages.");
  }
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
    });
  } catch (err) {
    return addMessage("err", "Microphone access was blocked. Allow it from the address bar and try again, or type a message instead.");
  }
  if (!S.workletReady) {
    await S.ctx.audioWorklet.addModule("/static/worklet.js");
    S.workletReady = true;
  }
  S.rs = new Resampler(S.ctx.sampleRate, RATE);
  S.src = S.ctx.createMediaStreamSource(S.stream);
  S.node = new AudioWorkletNode(S.ctx, "capture");
  S.mute = S.ctx.createGain();
  S.mute.gain.value = 0;                          // keeps the graph alive without echoing the mic
  S.src.connect(S.node); S.node.connect(S.mute); S.mute.connect(S.ctx.destination);
  S.node.port.onmessage = (e) => onSamples(e.data);
  Object.assign(vad, { inSpeech: false, hot: 0, silence: 0, frames: [], pre: [], level: 0 });
  S.fi = 0;
  S.running = true;
  render();
}

function stopMic() {
  S.running = false;
  try { S.node.port.onmessage = null; S.src.disconnect(); S.node.disconnect(); S.mute.disconnect(); } catch (_) {}
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
  Object.assign(vad, { inSpeech: false, hot: 0, frames: [], pre: [], level: 0 });
  render();
}

function onSamples(f32) {
  const out = S.rs.process(f32);
  for (let i = 0; i < out.length; i++) {
    S.frame[S.fi++] = out[i];
    if (S.fi === FRAME) { onFrame(S.frame.slice(0)); S.fi = 0; }
  }
}

function busy() { return S.status !== "idle" || (S.player && S.player.playing); }

function onFrame(frame) {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  const rms = Math.sqrt(sum / frame.length);
  vad.level = vad.level * 0.7 + rms * 0.3;

  const playing = S.player && S.player.playing;
  const minStart = MIN_START_BY_SENS[cfg.sens];
  let startT = Math.max(minStart, vad.noise * 3);
  const endT = Math.max(minStart * 0.6, vad.noise * 1.8);
  if (playing) startT *= 2;                       // the speaker leaks into the mic; be stricter
  const now = performance.now();

  if (!vad.inSpeech) {
    vad.pre.push(frame);
    if (vad.pre.length > PREROLL) vad.pre.shift();
    if (rms > startT) {
      vad.hot++;
    } else {
      vad.hot = 0;
      if (!playing) vad.noise = clamp(vad.noise * 0.97 + rms * 0.03, 0.001, 0.05);
    }
    if (vad.hot >= (playing ? BARGE_FRAMES : START_FRAMES)) {
      const wasBusy = busy();
      if (wasBusy && !cfg.barge) { vad.hot = 0; return; }
      vad.inSpeech = true;
      vad.frames = vad.pre.slice(); vad.pre = [];
      vad.silence = 0; vad.speech = vad.hot; vad.lastVoiceAt = now;
      if (wasBusy) bargeIn();
      render();
    }
    return;
  }

  vad.frames.push(frame);
  if (rms > endT) { vad.silence = 0; vad.speech++; vad.lastVoiceAt = now; } else { vad.silence++; }
  if (vad.silence >= Math.round(cfg.hangMs / FRAME_MS) || vad.frames.length >= MAX_FRAMES) endUtterance();
}

function endUtterance() {
  const frames = vad.frames, silence = vad.silence;
  vad.inSpeech = false; vad.frames = []; vad.hot = 0;
  if (vad.speech >= MIN_SPEECH_FRAMES) {
    const keep = frames.length - Math.max(0, silence - TAIL_FRAMES);
    sendUtterance(frames.slice(0, keep), vad.lastVoiceAt);
  }
  render();
}

function bargeIn() {
  const played = S.player ? S.player.playedMs() : 0;
  if (S.player) S.player.stop();
  send({ type: "interrupt", turn: S.turn, played_ms: Math.round(played) });
  if (S.bot) S.bot.li.classList.add("cut");
  S.bot = null;
  S.turn++;                                        // anything still in flight for the old turn is now stale
  S.status = "idle";
}

function sendUtterance(frames, voiceEnd) {
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN) return;
  const n = frames.length * FRAME;
  const buf = new ArrayBuffer(4 + n * 2);
  const pcm = new Int16Array(buf, 4, n);
  let k = 0;
  for (const f of frames) for (let i = 0; i < FRAME; i++) pcm[k++] = clamp(Math.round(f[i] * 32767), -32768, 32767);
  S.turn++;
  new DataView(buf).setUint32(0, S.turn, true);
  S.meta[S.turn] = { voiceEnd, sent: performance.now() };
  S.player.resetTurn();
  S.bot = null;
  S.status = "transcribing";
  S.ws.send(buf);
}

function sendTyped(text) {
  ensureAudio();
  if (busy()) bargeIn();
  S.turn++;
  S.meta[S.turn] = { voiceEnd: null, sent: performance.now() };
  S.player.resetTurn();
  S.bot = null;
  S.status = "thinking";
  send({ type: "text", turn: S.turn, text });
  render();
}

/* ------------------------------------------------------------------ controls */
$("#mic").addEventListener("click", () => { S.running ? stopMic() : startMic(); });
$("#typed").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("#typedInput");
  const text = input.value.trim();
  if (text) { sendTyped(text); input.value = ""; }
});
$("#lang").addEventListener("change", (e) => { cfg.lang = e.target.value; saveCfg(); send({ type: "config", language: cfg.lang }); });
$("#hang").addEventListener("input", (e) => { cfg.hangMs = +e.target.value; $("#hangOut").textContent = `${cfg.hangMs} ms`; saveCfg(); });
$("#sens").addEventListener("input", (e) => { cfg.sens = +e.target.value; $("#sensOut").textContent = String(cfg.sens); saveCfg(); });
$("#barge").addEventListener("change", (e) => { cfg.barge = e.target.checked; saveCfg(); render(); });
$("#reset").addEventListener("click", () => {
  if (S.player) S.player.stop();
  send({ type: "reset" });
  S.turn++; S.status = "idle"; S.bot = null;
  const note = el("li", "empty", "New conversation. The assistant has forgotten everything so far.");
  note.id = "empty";
  $("#log").replaceChildren(note);
  render();
});

// restore saved settings into the controls
$("#lang").value = cfg.lang;
$("#hang").value = cfg.hangMs; $("#hangOut").textContent = `${cfg.hangMs} ms`;
$("#sens").value = cfg.sens; $("#sensOut").textContent = String(cfg.sens);
$("#barge").checked = cfg.barge;

connect();
render();
requestAnimationFrame(frameLoop);
