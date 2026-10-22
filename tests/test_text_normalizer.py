"""TextNormalizer (§8). The tests commented "no NeMo" run without nemo-text-processing and never import it; the
others skip without it."""

import contextlib
import hashlib
import json
import os
import pickle
import re
import signal
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

import omnivoice_tts.utils.text as text
from omnivoice_tts.models.omnivoice import _NONVERBAL_PATTERN
from omnivoice_tts.utils import es_rules

ES_RULES_SHA256 = "40023499a9d15abf8f3c5128075cd3e1bd07db69aa0318e0436b10f44365b66e"  # indextts25-inference c98cabb8
PROBES = json.loads((Path(__file__).parent / "data" / "tn_probes.json").read_text(encoding="utf-8"))
EXACT = {
    "Ese [B EY1 S] cuesta $450.": "Ese [B EY1 S] cuesta cuatrocientos cincuenta pesos.",
    "[laughter]21:30 arrancamos.": "[laughter]veintiuna y media arrancamos.",
    "  Hola [laughter]  21 años  ": "  Hola [laughter]  veintiún años  ",
}


def python(code, *args, **kwargs):
    """Run ``code`` in a fresh ``python -I``, so nothing this process imported carries over."""
    return subprocess.run([sys.executable, "-I", "-c", code, *args], capture_output=True, text=True, **kwargs)


@pytest.mark.parametrize("probe", PROBES + [t for t in EXACT if t not in PROBES])
def test_tags_survive_byte_for_byte(tn, probe):
    out = tn.normalize(probe, "es")
    assert text._BRACKET_TAG_RE.findall(out) == text._BRACKET_TAG_RE.findall(probe)
    assert _NONVERBAL_PATTERN.findall(out) == _NONVERBAL_PATTERN.findall(probe)
    # IndexTTS <word|pronunciation> annotations pass through es_rules as written, digits included
    assert not re.search(r"\d", text._BRACKET_TAG_RE.sub("", es_rules._ANNOTATION.sub("", out))), out
    assert tn.normalize(out, "es") == out
    assert out == EXACT.get(probe, out)


def test_only_spanish_is_normalized(tn):
    for language in ("en", "fr", None, "spa", "es-AR"):
        assert tn.normalize("Tengo 2 gatos.", language) == "Tengo 2 gatos.", language
    for language in ("ES", "Spanish", " es "):
        assert tn.normalize("Tengo 2 gatos.", language) == "Tengo dos gatos.", language
    with pytest.raises(TypeError):
        tn.normalize("x")


NO_NEMO = """
import sys
for name in ("nemo_text_processing", "nemo_text_processing.text_normalization",
             "nemo_text_processing.text_normalization.normalize"):
    sys.modules[name] = None  # as if nemo-text-processing were not installed
from omnivoice_tts.utils.text import TextNormalizer
tn = TextNormalizer()
assert tn.normalize("Tengo 2 gatos.", "en") == tn.normalize("Tengo 2 gatos.", None) == "Tengo 2 gatos."
errors = []
for _ in range(2):  # every Spanish call raises, not only the first
    try:
        tn.normalize("Tengo 2 gatos.", "es")
    except ImportError as e:
        errors.append(str(e))
assert len(errors) == 2 and all("omnivoice-tts-inference[tn]" in e for e in errors), errors
"""


def test_non_spanish_needs_no_nemo():  # no NeMo
    r = python(NO_NEMO)
    assert r.returncode == 0, r.stderr


class FakeNormalizer:
    """Stands in for NeMo's Normalizer: counts its builds and pickles in no time."""

    builds = 0

    def __init__(self, **kwargs):
        FakeNormalizer.builds += 1


class Payload:
    """Unpickling it creates the file ``mark``."""

    def __init__(self, mark):
        self.mark = mark

    def __reduce__(self):
        return open, (self.mark, "w")


class Hung(BaseException):
    """Raised by ``deadline``: a BaseException, so the library's ``except Exception`` cannot swallow it."""


@contextlib.contextmanager
def deadline(seconds):
    """Raise ``Hung`` in a call still blocked after ``seconds`` (SIGALRM interrupts a blocking ``open``)."""

    def hung(signum, frame):
        raise Hung(f"still blocked after {seconds} s")

    old = signal.signal(signal.SIGALRM, hung)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_cache_loads_only_own_private_pickle(tmp_path, monkeypatch):  # no NeMo
    fake = types.ModuleType("nemo_text_processing.text_normalization.normalize")
    fake.Normalizer = FakeNormalizer
    for name in ("nemo_text_processing", "nemo_text_processing.text_normalization", fake.__name__):
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(text, "version", lambda _: "0")  # the pickle name reads both distributions' metadata
    monkeypatch.setattr(FakeNormalizer, "builds", 0)
    cache, mark, payload = tmp_path / "cache", tmp_path / "PAYLOAD_RAN", tmp_path / "payload.pkl"
    path = cache / "es-cased-nemo0-pynini0.pkl"

    def mode(p):
        return stat.S_IMODE(p.lstat().st_mode)

    def load():
        return text._spanish_nemo(str(cache))

    if os.geteuid() != 0:  # root writes anyway
        unwritable = tmp_path / "ro"
        unwritable.mkdir(mode=0o500)
        with pytest.raises(OSError):
            text._spanish_nemo(str(unwritable))
        assert FakeNormalizer.builds == 0 and list(unwritable.iterdir()) == []  # it failed before the build

    assert isinstance(load(), FakeNormalizer) and FakeNormalizer.builds == 1
    assert mode(cache) == 0o700 and mode(path) == 0o600  # the library created the directory
    assert isinstance(load(), FakeNormalizer) and FakeNormalizer.builds == 1  # our own pickle loads, no build

    payload.write_bytes(pickle.dumps(Payload(str(mark))))
    payload.chmod(0o600)
    pickle.loads(payload.read_bytes()).close()  # the payload works: unpickling it creates mark
    mark.unlink()
    for planted in ("0664", "0646", "symlink", "corrupt", "fifo", "fed fifo"):
        path.unlink()
        writer = None
        if planted == "symlink":
            path.symlink_to(payload)  # a regular 0600 file this user owns, behind a link
        elif planted.endswith("fifo"):  # 0600 and this user's, but not a regular file
            os.mkfifo(path, 0o600)
            if planted == "fed fifo":  # the payload queued in it, behind an open writer
                writer = os.open(path, os.O_RDWR)
                os.write(writer, payload.read_bytes())
        else:
            path.write_bytes(b"corrupt" if planted == "corrupt" else payload.read_bytes())
            path.chmod(0o600 if planted == "corrupt" else int(planted, 8))
        builds = FakeNormalizer.builds
        with deadline(10):  # opening a FIFO nobody writes waits for a writer, unless O_NONBLOCK
            got = load()
        if writer is not None:
            os.close(writer)
        assert isinstance(got, FakeNormalizer) and FakeNormalizer.builds == builds + 1, planted  # rebuilt
        assert not mark.exists() and path.is_file() and not path.is_symlink() and mode(path) == 0o600, planted
    assert [p.name for p in cache.iterdir()] == [path.name]  # no .tmp file left behind

    monkeypatch.setattr(fake, "Normalizer", None)  # the build itself fails
    path.unlink()
    with pytest.raises(TypeError):
        load()
    assert list(cache.iterdir()) == []  # its .tmp file is gone too


SILENT = """
import logging, os, sys, warnings
import nemo_text_processing.text_normalization.normalize  # the app imports NeMo, then sets its logger
from omnivoice_tts.utils.text import TextNormalizer
nemo = logging.getLogger("NeMo-text-processing")
nemo.setLevel(logging.ERROR)


def snapshot():
    return (nemo.level, nemo.handlers[:], nemo.filters[:], logging.root.handlers[:], dict(os.environ),
            sys.getrecursionlimit(), warnings.filters[:])


before = snapshot()
print("<tn>", flush=True)
print("<tn>", file=sys.stderr, flush=True)  # whatever follows is the library's
tn = TextNormalizer(cache_dir=sys.argv[1])
out = tn.normalize("[laughter]21:30 arrancamos.", "es")
tn.normalize("hola " * 501, "es")  # NeMo resets its level to INFO, then warns above 500 words
assert out == "[laughter]veintiuna y media arrancamos.", out
assert snapshot() == before, (before, snapshot())
"""


def test_tn_is_silent_and_global_free(tmp_path):
    pytest.importorskip("nemo_text_processing")
    env = {**os.environ, "HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "xdg"),
           "TMPDIR": str(tmp_path / "tmp")}
    (tmp_path / "tmp").mkdir()
    cache = tmp_path / "cache"
    for start in ("cold", "warm"):
        r = python(SILENT, str(cache), env=env)
        assert r.returncode == 0, r.stderr
        assert r.stdout.partition("<tn>\n")[2] == r.stderr.partition("<tn>\n")[2] == "", (start, r.stdout, r.stderr)
        assert [stat.S_IMODE(p.stat().st_mode) for p in cache.iterdir()] == [0o600], start  # one pickle
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cache", "tmp"] and not any((tmp_path / "tmp").iterdir())


def test_importing_text_does_not_import_nemo():  # no NeMo
    r = python("import sys, omnivoice_tts.utils.text; assert 'nemo_text_processing' not in sys.modules")
    assert r.returncode == 0, r.stderr


def test_es_rules_is_the_indextts25_copy():  # no NeMo
    source = Path(es_rules.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == ES_RULES_SHA256  # fix it in indextts25-inference, then copy it
    assert source.split(b"\n")[:2] == [b"# Copyright 2026 Kimox", b"# SPDX-License-Identifier: Apache-2.0"]
