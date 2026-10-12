"""Per-connection conversation state and prompt construction."""
from __future__ import annotations

from dataclasses import dataclass, field

_BASE_PROMPT = (
    "You are a friendly, quick-witted voice assistant. Everything you write is spoken "
    "aloud by a text-to-speech engine, so follow these rules strictly:\n"
    "- Answer in one to three short sentences unless the user explicitly asks for detail.\n"
    "- Use plain conversational speech. Never use markdown, bullet points, numbered lists, "
    "emojis, URLs, code blocks or special symbols.\n"
    "- Write numbers, units and abbreviations the way a person would say them.\n"
    "- The user's words come from speech recognition and may contain small mistakes. "
    "If something is unclear, ask a short clarifying question instead of guessing.\n"
    "- Remember earlier turns of this conversation and refer back to them naturally.\n"
    "- Reply in the same language the user speaks."
)

_LANG_HINT = {
    "hi": (
        "The user is speaking Hindi. Reply in natural spoken Hindi written in Devanagari "
        "script (never in Roman letters). Common English words may stay in Devanagari."
    ),
    "en": "The user is speaking English. Reply in clear, simple English.",
}


def build_system_prompt(lang: str) -> str:
    return f"{_BASE_PROMPT}\n{_LANG_HINT.get(lang, _LANG_HINT['en'])}"


@dataclass
class _LastReply:
    turn: int
    user_text: str
    # (sentence text, seconds of audio that precede it in this reply)
    sentences: list[tuple[str, float]]


@dataclass
class Session:
    """Holds the rolling conversation for one WebSocket connection.

    Context policy: keep the most recent `max_messages` messages, additionally capped
    by a character budget (roughly the model's context window), always dropping whole
    user/assistant pairs from the oldest end so the history stays well-formed.

    Interruption policy: the history must reflect what the user actually *heard*.
    `record_exchange` stores what was sent to the speaker; when the browser reports how
    much audio really played before a barge-in, `apply_playback_cut` trims the stored
    reply to match. If the user cut in before any reply was heard, their words are kept
    in `carry` and merged into their next utterance (so a mid-thought pause doesn't
    lose the first half of a sentence).
    """

    max_messages: int = 24
    max_chars: int = 9000
    history: list[dict] = field(default_factory=list)
    language: str = "en"  # language of the most recent user turn
    forced_language: str | None = None  # None = auto-detect
    carry: str = ""
    _last: _LastReply | None = None

    # ---------------------------------------------------------------- prompts
    def messages_for(self, user_text: str, lang: str) -> list[dict]:
        return [
            {"role": "system", "content": build_system_prompt(lang)},
            *self.history,
            {"role": "user", "content": user_text},
        ]

    # ---------------------------------------------------------------- updates
    def record_exchange(
        self, turn: int, user_text: str, sentences: list[tuple[str, float]]
    ) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": " ".join(t for t, _ in sentences)})
        self._last = _LastReply(turn, user_text, list(sentences))
        self.carry = ""
        self._trim()

    def stash_unanswered(self, user_text: str) -> None:
        """The turn was cancelled before the user heard anything."""
        self.carry = user_text
        self._last = None

    def apply_playback_cut(self, turn: int, played_ms: float) -> None:
        last = self._last
        if last is None or last.turn != turn:
            return
        if len(self.history) < 2 or self.history[-1]["role"] != "assistant":
            return
        kept = [(t, s) for t, s in last.sentences if s * 1000.0 <= played_ms]
        if len(kept) == len(last.sentences):
            return
        if kept:
            self.history[-1]["content"] = " ".join(t for t, _ in kept)
            self._last = _LastReply(turn, last.user_text, kept)
        else:
            del self.history[-2:]
            self.stash_unanswered(last.user_text)

    def reset(self) -> None:
        self.history.clear()
        self.carry = ""
        self._last = None

    # --------------------------------------------------------------- trimming
    def _size(self) -> int:
        return sum(len(m["content"]) for m in self.history)

    def _trim(self) -> None:
        while len(self.history) > self.max_messages or (
            self._size() > self.max_chars and len(self.history) > 2
        ):
            del self.history[:2]
