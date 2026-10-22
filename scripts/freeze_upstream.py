"""Freeze upstream k2-fsa/OmniVoice's behavior into the JSON fixture that the tests compare the library against.

Run it in the reference venv, where upstream is installed editable as ``omnivoice``:

    . ./.plan-env.sh && "$KV" -I -B scripts/freeze_upstream.py tests/data/upstream_08be0b4.json

It runs upstream's own ``omnivoice`` package, never ``omnivoice_tts``, and reads ``UP`` (the upstream clone) and
``REV`` (the commit it must be at) from the environment. ``-I`` ignores PYTHONHASHSEED, so two runs that write the
same bytes prove that the fixture does not depend on set order.
"""

import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))  # -I keeps the script's directory off sys.path

import numpy as np
import omnivoice
import omnivoice.utils.voice_design as vd
import torch
import transformers
from omnivoice.models.omnivoice import OmniVoice, _combine_text, _resolve_instruct, _resolve_language
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID
from omnivoice.utils.text import ABBREVIATIONS, CLOSING_MARKS, SPLIT_PUNCTUATION, chunk_text_punctuation
from stubs import Recorder, stub_model

SENT = "Ayer fuimos al centro con mi hermana a comprar regalos para toda la familia, y la verdad es que caminamos muchísimo. "
TEXTS_CALLS = ["  Hola che, ¿todo bien?  ", "[laughter] No puedo más, me ganaste otra vez.", (SENT * 6).strip(),
               "你好，世界。" * 40]
KWARGS = [{}, {"duration": 3.0}, {"speed": 1.3}, {"duration": 40.0, "language": "es"},
          {"instruct": "female, low pitch"}, {"num_step": 16}, {"speed": 0.4}]
S = "Hoy arrancamos el stream con todo, chat, y vamos a jugar un rato largo. "
LONG_TEXTS = [S * 12, ("La verdad que no sé qué decirte, " * 30) + "fin.", "Uno, dos, tres; cuatro: cinco. " * 25]
CHUNKED_KWARGS = [{}, {"speed": 1.3}, {"duration": 45.0}]
CHUNK_TEXTS = ["Ya llegó la lluvia, che[laughter]. No puedo más, me ganaste otra vez.",
               "[laughter] No puedo más, me ganaste otra vez. ¿Viste? Te lo dije.", "Hola.[laughter] ¿Qué tal?",
               "Son las 21:30 y cuesta 3.5 pesos, o sea 1,000 centavos.",
               "La Sra. Pérez y el Sr. Gómez llegaron. Dr. House también.", "  Hola che, ¿todo bien?  ",
               "Hola... ¿estás ahí? ¡Sí!"]
KOHAI = ["angry", "happy", "sad", "surprised", "afraid", "melancholic", "shy", "confused"]
EDGE = ["", "   ", ",", "female,", " ,male", "，男，", "Female , Low Pitch", "FEMALE，LOW PITCH", "female, female",
        "male, female", "american accent，河南话", "河南话, female"]
ALPHABET = "abc Hola,. ¿?¡! [laughter] 你好（）\t  "


def check_upstream():
    """Abort unless ``omnivoice`` is the clone at $UP, checked out at $REV, with its package untouched."""
    up, rev = os.environ["UP"], os.environ["REV"]
    if not Path(omnivoice.__file__).resolve().is_relative_to(Path(up).resolve()):
        sys.exit(f"omnivoice comes from {omnivoice.__file__}, not from {up}")
    git = ["git", "--no-optional-locks", "-C", up]  # a plain `git status` may rewrite the clone's index
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != rev:
        sys.exit(f"{up} is at {head}, not at {rev}")
    dirty = subprocess.check_output([*git, "status", "--porcelain", "--", "omnivoice"], text=True)
    if dirty:
        sys.exit(f"{up}/omnivoice has local changes:\n{dirty}")


def freeze_meta():
    return {"upstream": os.environ["REV"], "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__}


def run_case(text, kwargs):
    """One generate() call on a stubbed upstream model: what reached _generate_iterative, and a hash of the audio."""
    recorder = Recorder()
    out = stub_model(OmniVoice, RuleDurationEstimator(), recorder).generate(text=text, **kwargs)[0]
    pcm = np.ascontiguousarray(out, dtype="<f4").tobytes()
    return {"text": text, "kwargs": kwargs, "calls": recorder.calls, "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "n_samples": len(out)}


def freeze_calls():
    return [run_case(t, dict(kw)) for t in TEXTS_CALLS for kw in KWARGS]


def freeze_chunked():
    return [run_case(t, {**kw, "language": "es"}) for t in LONG_TEXTS for kw in CHUNKED_KWARGS]


def freeze_chunker():
    cases = [(t, n) for t in CHUNK_TEXTS for n in (0, 20, 1000)] + [(t, n) for t in LONG_TEXTS for n in (100, 200, 400)]
    return {"split_punctuation": sorted(SPLIT_PUNCTUATION), "closing_marks": sorted(CLOSING_MARKS),
            "abbreviations": sorted(ABBREVIATIONS),
            "cases": [{"text": t, "chunk_len": n, "out": chunk_text_punctuation(t, n, 3)} for t, n in cases]}


def freeze_duration():
    estimator = RuleDurationEstimator()
    texts = TEXTS_CALLS + LONG_TEXTS + CHUNK_TEXTS + ["", "a", "1234", "[laughter]", "你好", "Hello world.", "¿Qué?"]
    return [[t, estimator.calculate_total_weight(t), estimator.estimate_duration(t, "Nice to meet you.", 25)]
            for t in texts]


def freeze_instruct():
    # The design session's equiv_check recipe: upstream's own vocabulary in pairs, with the separators, the case and
    # the padding mangled, plus the KohAI words. EDGE is new. Sets are sorted: their order depends on hashing.
    cats =[list(c.keys()) + list(c.values()) if isinstance(c, dict) else sorted(c) for c in vd._INSTRUCT_CATEGORIES]
    cases = []
    for a, b in itertools.combinations(range(len(cats)), 2):
        for x in cats[a][:3]:
            for y in cats[b][:3]:
                cases += [f"{x}, {y}", f"{x.upper()}，{y}", f" {x} ,{y},"]
    cases += [w for c in cats for w in c]
    # On purpose: upstream accepts the KohAI words only once they are in its set, and this shows how it then
    # normalizes a word it was not built for. Nothing after this point needs upstream to reject them.
    vd._INSTRUCT_ALL_VALID.update(KOHAI)
    cases += KOHAI + ["Happy", "happy, female", "shy，男"] + EDGE
    rows = []
    for s, use_zh in itertools.product(cases, (False, True)):
        try:
            rows.append([s, use_zh, _resolve_instruct(s, use_zh=use_zh), False])
        except ValueError:
            rows.append([s, use_zh, None, True])
    return rows


def freeze_language():
    names = sorted(LANG_NAME_TO_ID)
    inputs = sorted(LANG_IDS) + names + [n.title() for n in names] + [None, "None", "none"]
    inputs += ["ES", "Es", "SPANISH", " es ", "es-AR", "castellano", "Klingon"]  # upstream logs a warning for most
    rows = []
    for x in inputs:
        out = _resolve_language(x)
        rows.append([x, out, out is not None or x in (None, "None", "none")])
    return rows


def freeze_combine_text():
    random.seed(0)
    strs = ["".join(random.choice(ALPHABET) for _ in range(random.randint(0, 40))) for _ in range(1000)]
    return [[t, ref, _combine_text(t, ref)] for t, ref in zip(strs, strs[1:] + [None])]


def main(out_path: str) -> None:
    check_upstream()
    data = {
        "meta": freeze_meta(),
        "calls": freeze_calls(),
        "chunked": freeze_chunked(),
        "chunker": freeze_chunker(),
        "duration": freeze_duration(),
        "instruct": freeze_instruct(),
        "language": freeze_language(),
        "combine_text": freeze_combine_text(),
    }
    # indent=1: one value per line, so a fixture that moves shows up in `git diff` as the values that moved
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    n_instruct, n_language = len(data["instruct"]), len(data["language"])
    instruct_rejected = sum(row[3] for row in data["instruct"])
    language_accepted = sum(row[2] for row in data["language"])
    print("calls", len(data["calls"]))
    print("chunked", len(data["chunked"]))
    print("chunker cases", len(data["chunker"]["cases"]))
    print("duration", len(data["duration"]))
    print(f"instruct {n_instruct} ({n_instruct - instruct_rejected} accepted, {instruct_rejected} rejected)")
    print(f"language {n_language} ({language_accepted} accepted, {n_language - language_accepted} rejected)")
    print("combine_text", len(data["combine_text"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
