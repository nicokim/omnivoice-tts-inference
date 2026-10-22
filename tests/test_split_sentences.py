import json
from pathlib import Path

from omnivoice_tts.utils.text import ABBREVIATIONS, CLOSING_MARKS, SPLIT_PUNCTUATION, chunk_text_punctuation

UP = json.loads((Path(__file__).parent / "data/upstream_08be0b4.json").read_text(encoding="utf-8"))


def test_upstream_chunker_untouched():
    chunker = UP["chunker"]
    assert sorted(SPLIT_PUNCTUATION) == chunker["split_punctuation"]
    assert sorted(CLOSING_MARKS) == chunker["closing_marks"]
    assert sorted(ABBREVIATIONS) == chunker["abbreviations"]
    for case in chunker["cases"]:
        assert chunk_text_punctuation(case["text"], case["chunk_len"], 3) == case["out"]
