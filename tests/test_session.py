from app.session import Session, build_system_prompt


def test_messages_include_system_history_and_user():
    s = Session()
    s.record_exchange(1, "hi", [("Hello there.", 0.0)])
    msgs = s.messages_for("how are you", "en")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "how are you"


def test_language_hint_in_prompt():
    assert "Devanagari" in build_system_prompt("hi")
    assert "English" in build_system_prompt("en")


def test_history_trims_in_pairs():
    s = Session(max_messages=4)
    for i in range(5):
        s.record_exchange(i, f"q{i}", [(f"a{i}", 0.0)])
    assert len(s.history) == 4
    assert s.history[0] == {"role": "user", "content": "q3"}


def test_char_budget_trims_oldest_pairs():
    s = Session(max_chars=100)
    for i in range(6):
        s.record_exchange(i, "x" * 30, [("y" * 30, 0.0)])
    assert s._size() <= 100 and s.history[0]["role"] == "user"


def test_playback_cut_keeps_only_heard_sentences():
    s = Session()
    s.record_exchange(7, "tell me about paris", [("Paris is big.", 0.0), ("It has a tower.", 1.5), ("Enjoy.", 3.0)])
    s.apply_playback_cut(7, played_ms=1600)  # heard the first sentence and part of the second
    assert s.history[-1]["content"] == "Paris is big. It has a tower."


def test_playback_cut_with_nothing_heard_stashes_user_text():
    s = Session()
    s.record_exchange(3, "tell me about paris", [("Paris is big.", 0.0)])
    s.apply_playback_cut(3, played_ms=-1)
    assert s.history == [] and s.carry == "tell me about paris"


def test_playback_cut_ignores_other_turns():
    s = Session()
    s.record_exchange(3, "q", [("a", 0.0), ("b", 2.0)])
    s.apply_playback_cut(99, played_ms=0)
    assert s.history[-1]["content"] == "a b"


def test_reset_clears_everything():
    s = Session()
    s.record_exchange(1, "q", [("a", 0.0)])
    s.carry = "x"
    s.reset()
    assert s.history == [] and s.carry == ""
