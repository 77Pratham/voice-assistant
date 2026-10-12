// Headless check of the browser client's VAD / barge-in state machine.
//   node tests/js/vad.test.js
// Loads the real app.js into a VM with stubbed browser APIs and feeds it synthetic 16 kHz audio.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

function stubEl() {
  const el = {
    style: { setProperty() {} }, dataset: {}, classList: { add() {}, remove() {} },
    addEventListener() {}, setAttribute() {}, append() {}, replaceChildren() {}, remove() {},
    textContent: "", value: "", checked: true, scrollTop: 0, scrollHeight: 0, firstChild: null,
  };
  return el;
}

function load() {
  const sent = [];
  class WS {
    constructor() { this.readyState = 1; WS.last = this; setTimeout(() => this.onopen && this.onopen(), 0); }
    send(d) { sent.push(d); }
  }
  WS.OPEN = 1;
  const ctx = {
    document: { querySelector: stubEl, createElement: stubEl, documentElement: stubEl() },
    WebSocket: WS, localStorage: { getItem: () => null, setItem() {} },
    location: { protocol: "http:", host: "x" }, navigator: {}, performance,
    requestAnimationFrame() {}, setTimeout, console, Math, JSON, Object, Float32Array, Int16Array,
    ArrayBuffer, DataView, Set, Promise,
  };
  ctx.window = ctx;
  vm.createContext(ctx);
  const src = fs.readFileSync(path.join(__dirname, "../../app/static/app.js"), "utf8");
  vm.runInContext(src + "\n;globalThis.__t = { S, vad, cfg, onSamples, Resampler, FRAME };", ctx);
  const t = ctx.__t;
  const player = { playing: false, stopped: 0, resetTurn() {}, playedMs: () => 1234, stop() { this.stopped++; } };
  t.S.player = player;
  t.S.rs = new t.Resampler(16000, 16000);
  t.S.running = true;
  return { t, sent, player };
}

const rate = 16000;
let phase = 0;
function tone(seconds, amp) {
  const n = Math.round(seconds * rate), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((phase += 2 * Math.PI * 220 / rate));
  return out;
}
function noise(seconds, amp) {
  const n = Math.round(seconds * rate), out = new Float32Array(n);
  let x = 12345;
  for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; out[i] = amp * (x / 0x7fffffff - 0.5) * 2; }
  return out;
}
function feed(t, samples) {
  for (let i = 0; i < samples.length; i += 1024) t.onSamples(samples.subarray(i, i + 1024));
}
const isBinary = (m) => typeof m !== "string";
let passed = 0;
function test(name, fn) { fn(); passed++; console.log("ok -", name); }

test("resampler: 48 kHz -> 16 kHz keeps the right sample count and level", () => {
  const { t } = load();
  const rs = new t.Resampler(48000, 16000);
  const input = new Float32Array(48000).fill(0.5);
  const out = rs.process(input);
  assert.ok(Math.abs(out.length - 16000) <= 1);
  assert.ok(Math.abs(out[100] - 0.5) < 1e-6);
});

test("a spoken phrase is sent once, after the hang-over, with the right header", () => {
  const { t, sent } = load();
  feed(t, noise(0.6, 0.003));
  feed(t, tone(1.0, 0.12));
  assert.strictEqual(sent.length, 0, "must not send while the user is still talking");
  feed(t, noise(1.0, 0.003));
  const bin = sent.filter(isBinary);
  assert.strictEqual(bin.length, 1);
  assert.strictEqual(new DataView(bin[0]).getUint32(0, true), 1, "turn id");
  const samples = (bin[0].byteLength - 4) / 2;
  assert.ok(samples > rate * 1.0 && samples < rate * 1.9, `clip length ${samples}`);
  assert.strictEqual(t.S.status, "transcribing");
});

test("a short blip is ignored", () => {
  const { t, sent } = load();
  feed(t, noise(0.5, 0.003));
  feed(t, tone(0.12, 0.12));
  feed(t, noise(1.2, 0.003));
  assert.strictEqual(sent.length, 0);
});

test("barge-in: speaking over the assistant interrupts with played_ms, then sends the new utterance", () => {
  const { t, sent, player } = load();
  feed(t, noise(0.6, 0.003));
  t.S.status = "speaking"; player.playing = true; t.S.turn = 3;
  feed(t, tone(1.0, 0.2));
  const first = sent.find((m) => typeof m === "string");
  assert.ok(first, "an interrupt message should be sent as soon as speech is confirmed");
  assert.deepStrictEqual(JSON.parse(first), { type: "interrupt", turn: 3, played_ms: 1234 });
  assert.strictEqual(player.stopped, 1);
  player.playing = false; t.S.status = "idle";
  feed(t, noise(1.0, 0.003));
  const bin = sent.filter(isBinary);
  assert.strictEqual(bin.length, 1);
  assert.strictEqual(new DataView(bin[0]).getUint32(0, true), 5, "turn id moves past the cancelled turn");
});

test("quiet speaker bleed does not trigger barge-in", () => {
  const { t, sent, player } = load();
  feed(t, noise(0.6, 0.003));
  t.S.status = "speaking"; player.playing = true;
  feed(t, tone(1.0, 0.025));  // loud enough to start a normal turn, not loud enough while the bot talks
  assert.strictEqual(sent.length, 0);
  assert.strictEqual(player.stopped, 0);
});

test("with interruption disabled the user is not allowed to cut in", () => {
  const { t, sent, player } = load();
  t.cfg.barge = false;
  feed(t, noise(0.6, 0.003));
  t.S.status = "speaking"; player.playing = true;
  feed(t, tone(1.0, 0.2));
  feed(t, noise(1.0, 0.003));
  assert.strictEqual(sent.length, 0);
});

console.log(`\n${passed} client checks passed`);
