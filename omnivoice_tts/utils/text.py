#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
# Modified by omnivoice-tts-inference from k2-fsa/OmniVoice 08be0b4
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text processing utilities for TTS inference.

Provides:
- ``chunk_text_punctuation()``: Splits long text into model-friendly chunks at
  sentence boundaries, with abbreviation-aware punctuation splitting.
- ``TextNormalizer``: Optional Spanish text normalization (numbers, dates,
  currency, etc.) into their spoken form, while preserving inline control syntax.
"""

import contextlib
import logging
import os
import pickle
import re
import tempfile
from importlib.metadata import version

from omnivoice_tts.utils import es_rules
from omnivoice_tts.utils.lang_map import LANG_NAME_TO_ID

logger = logging.getLogger(__name__)


SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
CLOSING_MARKS = set("\"'“”‘’）]》>」】")


ABBREVIATIONS = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "Rev.",
    "Fr.",
    "Hon.",
    "Pres.",
    "Gov.",
    "Capt.",
    "Gen.",
    "Sen.",
    "Rep.",
    "Col.",
    "Maj.",
    "Lt.",
    "Cmdr.",
    "Sgt.",
    "Cpl.",
    "Co.",
    "Corp.",
    "Inc.",
    "Ltd.",
    "Est.",
    "Dept.",
    "St.",
    "Ave.",
    "Blvd.",
    "Rd.",
    "Mt.",
    "Ft.",
    "No.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
    "i.e.",
    "e.g.",
    "vs.",
    "Vs.",
    "Etc.",
    "approx.",
    "fig.",
    "def.",
}


def chunk_text_punctuation(
    text: str,
    chunk_len: int,
    min_chunk_len: int | None = None,
) -> list[str]:
    """
    Splits the input tokens list into chunks according to punctuations,
    avoiding splits on common abbreviations (e.g., Mr., No.).
    """

    # 1. Split the tokens according to punctuations.
    sentences = []
    current_sentence = []

    tokens_list = list(text)

    for token in tokens_list:
        # If the first token of current sentence is punctuation,
        # append it to the end of the previous sentence.
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in SPLIT_PUNCTUATION or token in CLOSING_MARKS)
        ):
            sentences[-1].append(token)
        # Otherwise, append the current token to the current sentence.
        else:
            current_sentence.append(token)

            # Split the sentence in positions of punctuations.
            if token in SPLIT_PUNCTUATION:
                is_abbreviation = False

                if token == ".":
                    temp_str = "".join(current_sentence).strip()
                    if temp_str:
                        last_word = temp_str.split()[-1]
                        if last_word in ABBREVIATIONS:
                            is_abbreviation = True

                if not is_abbreviation:
                    sentences.append(current_sentence)
                    current_sentence = []
    # Assume the last few tokens are also a sentence
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    # 2. Merge short sentences.
    merged_chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_len:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                merged_chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) > 0:
        merged_chunks.append(current_chunk)

    # 4. Post-process: Check for undersized chunks and merge them
    #  with the previous chunk or next chunk (if it's the first chunk).
    if min_chunk_len is not None:
        first_chunk_short_flag = (
            len(merged_chunks) > 0 and len(merged_chunks[0]) < min_chunk_len
        )
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            if i == 1 and first_chunk_short_flag:
                final_chunks[-1].extend(chunk)
            else:
                if len(chunk) >= min_chunk_len:
                    final_chunks.append(chunk)
                else:
                    if len(final_chunks) == 0:
                        final_chunks.append(chunk)
                    else:
                        final_chunks[-1].extend(chunk)
    else:
        final_chunks = merged_chunks

    chunk_strings = [
        "".join(chunk).strip() for chunk in final_chunks if "".join(chunk).strip()
    ]
    return chunk_strings


# ---------------------------------------------------------------------------
# Optional text normalization: call TextNormalizer.normalize() before generate()
# ---------------------------------------------------------------------------
#
# Spanish only: NeMo's grammar wrapped by ``es_rules``, a byte-identical copy of
# nicokim/indextts25-inference indextts25/utils/es_rules.py (fix it there first,
# then copy it). Upstream's zh/en (WeTextProcessing) and num2words paths are gone.
#
# The OmniVoice inline control syntax must survive normalization: every ``[...]``
# span (non-verbal tags such as ``[laughter]``; CMU pronunciation overrides such
# as ``[B EY1 S]``, whose stress digit would otherwise be read as a number) is
# held out and re-inserted verbatim.

_BRACKET_TAG_RE = re.compile(r"\[[^\[\]]*\]")

_TN_INSTALL_MSG = (
    "Spanish text normalization needs nemo-text-processing: "
    "pip install 'omnivoice-tts-inference[tn]' (Linux x86_64, CPython 3.10-3.13, a C compiler). "
    "Elsewhere, install pynini from conda-forge first, then nemo-text-processing."
)


class TextNormalizer:
    """Spell out numbers, money, dates, times... before ``generate()``.

    ``normalize(text, language)`` takes the ``language`` you pass to ``generate()``.
    Only Spanish (``"es"``, ``"Spanish"``) is normalized: NeMo's grammar plus the
    rules in ``es_rules``, which follow Argentine/Latin American usage (``$`` is
    pesos, ``US$``/``USD`` are dollars, ``21.30`` is a time). Any other language,
    or ``None``, comes back unchanged. ``[...]`` spans (non-verbal tags, CMU
    pronunciations) pass through byte for byte; numbers do not agree across them
    (``1 [laughter] perro`` -> ``uno``).

    The Spanish grammar is built on the first Spanish call: about 20 s and 500 MB
    of RAM. With ``cache_dir``, the built grammar is pickled there (about 60 MB,
    mode 0600) and later instances load it in about 0.5 s; only a regular file
    owned by this user and not writable by group or others is unpickled.

    One call at a time per instance: NeMo's parser keeps state. pynini holds the
    GIL while it runs (up to ~30 ms per sentence), so keep it off event loops.
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir
        self._es = None

    def normalize(self, text: str, language: str | None) -> str:
        code = (language or "").strip().lower()
        if LANG_NAME_TO_ID.get(code, code) != "es" or not text.strip():
            return text
        if self._es is None:
            self._es = _spanish_nemo(self.cache_dir)
        with _quiet_nemo():
            return _apply_with_protection(text, self._spanish)

    def _spanish(self, text: str) -> str:
        return es_rules.normalize(text, lambda t: self._es.normalize(t, verbose=False))


def _spanish_nemo(cache_dir: str | None):
    """NeMo's Spanish Normalizer, from ``cache_dir`` when it holds our own pickle."""
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer
    except ImportError as e:
        raise ImportError(_TN_INSTALL_MSG) from e
    if cache_dir is None:
        with _quiet_nemo():
            return Normalizer(input_case="cased", lang="es")
    # indextts25's nemo_tn.py cache. NeMo's own cache_dir is not used: its FAR
    # export optimizes again and reads some decimals differently ("1,80").
    path = os.path.join(
        cache_dir,
        f"es-cased-nemo{version('nemo_text_processing')}-pynini{version('pynini')}.pkl",
    )
    try:
        # Unpickling runs code: only a regular file (not a symlink) that this user owns and
        # nobody else can write. ponytail: POSIX only (os.getuid, O_NOFOLLOW); elsewhere
        # the cache is never trusted and every start rebuilds.
        with open(path, "rb", opener=lambda p, fl: os.open(p, fl | os.O_NOFOLLOW)) as f:
            st = os.fstat(f.fileno())
            if st.st_uid == os.getuid() and not st.st_mode & 0o022:
                return pickle.load(f)
    except Exception:  # missing, corrupt, or written by another version: rebuild it
        pass
    os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    # mode 0600 whatever the umask; an unwritable cache_dir fails here, before the ~20 s build
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            with _quiet_nemo():
                normalizer = Normalizer(input_case="cased", lang="es")
            pickle.dump(normalizer, f)
        os.replace(tmp, path)  # atomic: a process starting meanwhile never reads half a file
    except BaseException:
        os.unlink(tmp)
        raise
    return normalizer


@contextlib.contextmanager
def _quiet_nemo():
    """NeMo logs through its own stderr handler and resets its logger to INFO on
    every normalize(). For the length of our call, its records go to our logger at
    DEBUG instead, and its level is put back afterwards: the net change is none.
    ponytail: not coordinated across threads; two instances normalizing at the same
    time can leave NeMo's logger at INFO (measured). A module-level lock fixes it."""
    nemo_logger = logging.getLogger("NeMo-text-processing")
    level = nemo_logger.level

    def to_debug(record):
        logger.debug("NeMo: %s", record.getMessage())
        return False

    nemo_logger.addFilter(to_debug)
    try:
        yield
    finally:
        nemo_logger.removeFilter(to_debug)
        nemo_logger.setLevel(level)


def _normalize_segment(fn, segment: str) -> str:
    """Normalize one segment, keeping its outer whitespace: the normalizer strips
    it, which would glue words to an adjacent span (``the [B EY1 S] guitar``)."""
    if not segment.strip():
        return segment
    lead = segment[: len(segment) - len(segment.lstrip())]
    trail = segment[len(segment.rstrip()) :]
    return lead + fn(segment.strip()) + trail


def _apply_with_protection(text: str, fn) -> str:
    """Run ``fn`` on the text between ``[...]`` spans; the spans pass through verbatim."""
    out: list[str] = []
    last = 0
    for m in _BRACKET_TAG_RE.finditer(text):
        out += [_normalize_segment(fn, text[last : m.start()]), m.group()]
        last = m.end()
    out.append(_normalize_segment(fn, text[last:]))
    return "".join(out)
