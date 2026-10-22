import json
import random
import re
from pathlib import Path

import omnivoice_tts.models.omnivoice as ov
from omnivoice_tts.utils.text import ABBREVIATIONS, CLOSING_MARKS, SPLIT_PUNCTUATION, chunk_text_punctuation

UP = json.loads((Path(__file__).parent / "data/upstream_08be0b4.json").read_text(encoding="utf-8"))
CASES = json.loads((Path(__file__).parent / "data/split_cases.json").read_text(encoding="utf-8"))  # [cat, text, gold]


def test_upstream_chunker_untouched():
    chunker = UP["chunker"]
    assert sorted(SPLIT_PUNCTUATION) == chunker["split_punctuation"]
    assert sorted(CLOSING_MARKS) == chunker["closing_marks"]
    assert sorted(ABBREVIATIONS) == chunker["abbreviations"]
    for case in chunker["cases"]:
        assert chunk_text_punctuation(case["text"], case["chunk_len"], 3) == case["out"]


def test_cases():
    assert len(CASES) == 73
    for cat, text, gold in CASES:
        out = ov._split_sentences(text)
        assert "".join(out) == text, (cat, text, out)
        got = out if len(out) == 1 else [p.strip() for p in out]  # one piece stays byte for byte
        assert got == (gold if cat == "ident" else [g.strip() for g in gold]), (cat, text, out)


def test_single_sentence_unchanged():
    for t in ["Hola, ¿qué tal?", "  Hola  ", "Son las 21:30 y llegué.", "[laughter] jaja", "...", ""]:
        assert ov._split_sentences(t) == [t]


# The parts and the 4 invariants of the design session's fixed_variant_check.py (stream-critic).
PARTS = ["Hola", "chau", "Sí", "no", "3.5", "21:30", "1.500", "$1.250,50", "8/11", "kohai.com", "a@b.com",
         "Sr.", "etc.", "J.", "EE.", "UU.", "[laughter]", "[sigh]", "😂", "¿Qué", "¡Ja", "…", "...", ".", "!",
         "?", "?!", ",", '"', "'", "»", ")", "(", "。", "你好", "\n", "\n\n", " ", "  ", "Bueno", "La",
         "¿qué", "¡no", '"sí', "(y"]


def speakable(s):
    return any(c.isalnum() for c in ov._NONVERBAL_PATTERN.sub("", s))


def test_fuzz():
    rnd = random.Random(20261020)
    for _ in range(20000):
        t = "".join(rnd.choice(PARTS) + rnd.choice(["", " ", " ", "\n"]) for _ in range(rnd.randint(1, 14)))
        out = ov._split_sentences(t)
        assert "".join(out) == t  # 1. lossless
        if speakable(t):
            assert all(speakable(p) for p in out), (t, out)  # 2. no piece without something to say
        for a, b in zip(out, out[1:]):
            assert not (re.search(r"\d[.,:]\Z", a) and b[:1].isdigit()), (t, out)  # 3. no split inside 3.5, 21:30
        for p in out:
            assert ov._split_sentences(p) == [p], (t, p)  # 4. every piece is a fixed point
