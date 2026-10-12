"""Text helpers used between the LLM and the TTS engine.

The key idea for low latency is *sentence-level pipelining*: as soon as the LLM
has produced one speakable sentence we hand it to TTS, while the LLM keeps
generating the next one. `SentenceChunker` decides where those cut points are.
"""
from __future__ import annotations

import re

# Sentence terminators for English and Hindi (danda "।" and double danda "॥"),
# optionally followed by closing quotes/brackets, then whitespace.
_END = re.compile(r"[.!?।॥]+[\"'”’)\]]*\s+")
# Clause boundaries, used only to release the *first* chunk early.
_CLAUSE = re.compile(r"[,;:—–]\s+")

_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "no", "approx",
}


class SentenceChunker:
    """Incrementally splits a token stream into speakable chunks.

    * `feed(text)` returns the chunks that became complete.
    * `flush()` returns whatever is left once the stream ends.

    Tiny sentences ("Sure.") are merged with the following one so the TTS engine
    gets enough context for natural prosody. The first chunk may be released at a
    comma once it is long enough, which shaves a few hundred milliseconds off the
    time-to-first-audio.
    """

    def __init__(self, min_chars: int = 14, first_clause_chars: int = 32, max_chars: int = 220):
        self.min_chars = min_chars
        self.first_clause_chars = first_clause_chars
        self.max_chars = max_chars
        self._buf = ""
        self._held = ""
        self._emitted_any = False

    # -- public -------------------------------------------------------------
    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if sentence:
                out.extend(self._emit(sentence))
        return out

    def flush(self) -> list[str]:
        rest = f"{self._held} {self._buf}".strip()
        self._held = ""
        self._buf = ""
        if rest:
            self._emitted_any = True
            return [rest]
        return []

    # -- internals ----------------------------------------------------------
    def _emit(self, sentence: str) -> list[str]:
        if self._held:
            sentence = f"{self._held} {sentence}"
            self._held = ""
        if len(sentence) < self.min_chars:
            self._held = sentence
            return []
        self._emitted_any = True
        return [sentence]

    def _find_cut(self) -> int | None:
        buf = self._buf
        for m in _END.finditer(buf):
            if self._is_false_boundary(buf[: m.start()], m.group()):
                continue
            return m.end()
        if not self._emitted_any and not self._held:
            for m in _CLAUSE.finditer(buf):
                if m.end() >= self.first_clause_chars:
                    return m.end()
        if len(buf) > self.max_chars:
            space = buf.rfind(" ", 0, self.max_chars)
            return space + 1 if space > 0 else self.max_chars
        return None

    @staticmethod
    def _is_false_boundary(head: str, terminator: str) -> bool:
        if not terminator.startswith("."):
            return False
        m = re.search(r"(\S+)$", head)
        if not m:
            return False
        word = m.group(1).lower().strip("\"'([")
        if word in _ABBREVIATIONS:
            return True
        # single capital initial like "J. K. Rowling"
        return len(word) == 1 and word.isalpha()


class ThinkStripper:
    """Removes `<think>...</think>` blocks emitted by reasoning models (e.g. Qwen3)
    from a streaming token feed. Tags may be split across tokens."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.inside = False
        self._pending = ""

    @staticmethod
    def _partial_suffix(data: str, tag: str) -> int:
        for k in range(min(len(data), len(tag) - 1), 0, -1):
            if tag.startswith(data[-k:]):
                return k
        return 0

    def feed(self, text: str) -> str:
        data = self._pending + text
        self._pending = ""
        out: list[str] = []
        while data:
            if self.inside:
                i = data.find(self.CLOSE)
                if i == -1:
                    k = self._partial_suffix(data, self.CLOSE)
                    self._pending = data[len(data) - k:] if k else ""
                    data = ""
                else:
                    data = data[i + len(self.CLOSE):]
                    self.inside = False
            else:
                i = data.find(self.OPEN)
                if i == -1:
                    k = self._partial_suffix(data, self.OPEN)
                    out.append(data[: len(data) - k] if k else data)
                    self._pending = data[len(data) - k:] if k else ""
                    data = ""
                else:
                    out.append(data[:i])
                    data = data[i + len(self.OPEN):]
                    self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self._pending
        self._pending = ""
        return rest


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"(?m)^\s*#+\s*")
_BULLET = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")


def clean_for_speech(text: str) -> str:
    """Strip markdown, URLs and emoji so the TTS engine never reads them aloud.
    Returns "" if nothing speakable is left."""
    t = _MD_LINK.sub(r"\1", text)
    t = _URL.sub("", t)
    t = _HEADING.sub("", t)
    t = _BULLET.sub("", t)
    t = t.replace("`", "").replace("*", "")
    t = re.sub(r"(?<!\w)_+|_+(?!\w)", "", t)  # _emphasis_ markers
    t = t.replace("_", " ")  # snake_case -> snake case
    t = _EMOJI.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if any(c.isalnum() for c in t) else ""


def script_language(text: str, default: str = "en") -> str:
    """'hi' if the text is mostly Devanagari, otherwise 'en'. Used to choose the
    TTS voice per sentence, which is more reliable than trusting the STT result."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return default
    deva = sum("\u0900" <= c <= "\u097f" for c in letters)
    return "hi" if deva / len(letters) >= 0.4 else "en"
