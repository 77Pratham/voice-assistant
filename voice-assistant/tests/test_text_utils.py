from app.text_utils import SentenceChunker, ThinkStripper, clean_for_speech, script_language


def stream(chunker, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out.extend(chunker.feed(text[i : i + step]))
    out.extend(chunker.flush())
    return out


def test_splits_on_sentence_ends():
    chunks = stream(SentenceChunker(), "Paris is the capital of France. It is famous for the Eiffel Tower.")
    assert chunks == ["Paris is the capital of France.", "It is famous for the Eiffel Tower."]


def test_short_sentences_are_merged():
    chunks = stream(SentenceChunker(), "Sure. I can help with that today. Anything else?")
    assert chunks[0].startswith("Sure. I can help")
    assert "".join(chunks).replace(" ", "") == "Sure.Icanhelpwiththattoday.Anythingelse?".replace(" ", "")


def test_decimals_and_abbreviations_do_not_split():
    chunks = stream(SentenceChunker(), "Dr. Rao paid 3.5 dollars for the book yesterday. Great deal.")
    assert chunks[0] == "Dr. Rao paid 3.5 dollars for the book yesterday."


def test_hindi_danda_splits():
    chunks = stream(SentenceChunker(), "भारत की राजधानी नई दिल्ली है। वहाँ इंडिया गेट और लाल क़िला हैं।")
    assert chunks == ["भारत की राजधानी नई दिल्ली है।", "वहाँ इंडिया गेट और लाल क़िला हैं।"]


def test_first_chunk_released_early_at_a_comma():
    c = SentenceChunker(first_clause_chars=20)
    got = c.feed("Well, the answer to your question is quite long, and then ")
    assert got and got[0].endswith(",")
    # after the first chunk, commas no longer split
    assert c.feed("it continues with more words, and more words, and more ") == []


def test_overlong_text_is_force_split():
    c = SentenceChunker(max_chars=50)
    got = c.feed("word " * 40)
    assert got and all(len(g) <= 50 for g in got)


def test_flush_returns_tail():
    c = SentenceChunker()
    assert c.feed("This has no terminator yet") == []
    assert c.flush() == ["This has no terminator yet"]
    assert c.flush() == []


def test_think_stripper_handles_split_tags():
    s = ThinkStripper()
    pieces = ["Hel", "lo <thi", "nk>secret ", "reasoning</th", "ink> world"]
    assert "".join(s.feed(p) for p in pieces) + s.flush() == "Hello  world"


def test_think_stripper_passes_plain_text_and_stray_lt():
    s = ThinkStripper()
    assert s.feed("2 < 3") + s.flush() == "2 < 3"


def test_clean_for_speech_keeps_snake_case_readable():
    assert clean_for_speech("use the max_value setting") == "use the max value setting"


def test_clean_for_speech():
    assert clean_for_speech("**Hello** _there_! See [docs](http://x.io) 😀") == "Hello there! See docs"
    assert clean_for_speech("- one\n- two") == "one two"
    assert clean_for_speech("...") == ""
    assert clean_for_speech("😀😀") == ""


def test_script_language():
    assert script_language("नमस्ते दोस्त") == "hi"
    assert script_language("hello there") == "en"
    assert script_language("12345", default="hi") == "hi"
    assert script_language("Hello दुनिया", "en") in {"en", "hi"}
