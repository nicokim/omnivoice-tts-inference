"""GPU checks on a local KohAI fp16 export (spec section 19: G2 and G3a). They need OMNIVOICE_TEST_KOHAI_DIR and CUDA;
the FlashInfer case also needs flashinfer and, on SM 12.x, FLASHINFER_CUDA_ARCH_LIST."""

import hashlib
import importlib
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from omnivoice_tts.models.omnivoice import OmniVoice, _split_sentences
from omnivoice_tts.utils.duration import RuleDurationEstimator

KOHAI_DIR = os.environ.get("OMNIVOICE_TEST_KOHAI_DIR")  # a local KohAI fp16 export
pytestmark = pytest.mark.skipif(not KOHAI_DIR or not torch.cuda.is_available(),
                                reason="needs OMNIVOICE_TEST_KOHAI_DIR and CUDA")
needs_flashinfer = pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None,
                                      reason="flashinfer is not installed")

SEED = 1234
# The design session's 12 bench lines (bench2/perf/perf.py LINES) and KohAI's duration rule (rule): the rule weight
# over the emotion's rate, plus 1.2 s per [laughter], at least 1 s. Neutral lines get no instruct.
RATES = {"happy": 14.11, "neutral": 13.14, "angry": 15.09, "sad": 12.92, "afraid": 14.73, "surprised": 12.34,
         "confused": 12.89, "shy": 12.84, "melancholic": 15.4}
LINES = [
    ("angry", "Mejor hacete youtuber, total ya sos un inútil."),
    ("angry", "No, es un puesto de verdulería tercermundista donde venden cebollas que te hacen llorar más que tu "
              "vida."),
    ("happy", "Para robar una tienda, rompé el vidrio y llevate todo mientras los empleados lloran en el piso."),
    ("sad", "El sol ya se está yendo a dormir porque se aburrió de iluminar esta porquería."),
    ("neutral", "Esa soy yo, KohAI, con traje de calabaza porque Halloween me da hambre."),
    ("angry", "Twitter es un antro de virgos peleando por política y Instagram es para mostrarle el culo a "
              "desconocidos."),
    ("happy", "es un video de un tipo surfeando y una reportera entrevistando a alguien."),
    ("sad", "El existencialismo es darte cuenta que tu vida no tiene sentido y que vas a morir solo."),
    ("surprised", "¿En serio me regalaste una suscripción? No te puedo creer, sos un divino."),
    ("afraid", "Me tiemblan las piernas, es mi primer evento en vivo con tanta gente mirando."),
    ("melancholic", "Hay noches en las que me pongo a pensar si alguien se acuerda de mí."),
    ("confused", "Esperá, no entiendo, ¿me estás diciendo que el gato se comió el cable?"),
]
EST = RuleDurationEstimator()


def rule(text, emo):
    return max(1.0, EST.calculate_total_weight(text) / RATES[emo] + 1.2 * text.count("[laughter]"))


def pcm_hash(audio):  # bench2's int16 PCM sha256
    return hashlib.sha256((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()).hexdigest()


def seeded(private):
    """A fresh private CUDA generator, or None after seeding the global RNG; both seeded SEED."""
    if private:
        return torch.Generator("cuda:0").manual_seed(SEED)
    torch.manual_seed(SEED)
    return None


@pytest.fixture(scope="module")
def sdpa():
    return OmniVoice.from_pretrained(KOHAI_DIR, device_map="cuda:0", dtype=torch.float16)


@pytest.fixture(scope="module")
def flashinfer(tmp_path_factory):
    """FlashInfer built CPU-first. flashinfer is imported as in test_flashinfer_cpu.py, with HOME and TMPDIR in a temp
    dir, unless an earlier test already imported it."""
    with pytest.MonkeyPatch.context() as mp:
        if "flashinfer" not in sys.modules:
            sandbox = str(tmp_path_factory.mktemp("home"))
            mp.setenv("HOME", sandbox)
            mp.setenv("TMPDIR", sandbox)
        fi = importlib.import_module("omnivoice_tts.models.omnivoice_flashinfer")
    if not fi.flashinfer.jit.current_compilation_context.TARGET_CUDA_ARCHS:
        pytest.skip("FlashInfer has no CUDA arch for this GPU: start Python with FLASHINFER_CUDA_ARCH_LIST set")
    return fi.apply_flashinfer(OmniVoice.from_pretrained(KOHAI_DIR, dtype=torch.float16)).to("cuda:0")


def test_private_generator_matches_global_seed_gpu(sdpa):
    emo, text = LINES[0]
    kw = dict(instruct=emo, duration=rule(text, emo))
    seen = []
    sdpa._decode_and_post_process = lambda tokens, gen_config: seen.append(tokens.clone()) or np.zeros(1, np.float32)
    try:
        torch.manual_seed(SEED)
        sdpa.generate(text, "es", **kw)
        state = torch.cuda.get_rng_state()
        sdpa.generate(text, "es", generator=torch.Generator("cuda:0").manual_seed(SEED), **kw)
        assert torch.equal(torch.cuda.get_rng_state(), state)
        sdpa.generate(text, "es", **kw)  # control: without a generator the CUDA RNG does move
        assert not torch.equal(torch.cuda.get_rng_state(), state)
        torch.manual_seed(SEED + 1)
        sdpa.generate(text, "es", **kw)
        with pytest.raises(RuntimeError, match="Expected a 'cuda' device type for generator"):
            sdpa.generate(text, "es", generator=torch.Generator(), **kw)
    finally:
        del sdpa._decode_and_post_process
    assert len(seen) == 4
    assert torch.equal(seen[0], seen[1]) and not torch.equal(seen[0], seen[3])  # another seed, other tokens


@pytest.mark.parametrize("backend", ["sdpa", pytest.param("flashinfer", marks=needs_flashinfer)])
def test_stream_single_sentence_parity_gpu(backend, request):
    model = request.getfixturevalue(backend)
    whole = [(emo, text) for emo, text in LINES if _split_sentences(text) == [text]]
    assert len(whole) == 11  # the 9th line splits in two (G3b, multi-sentence)
    for emo, text in whole:
        kw = dict(instruct=None if emo == "neutral" else emo)

        def dur(spoken):
            return rule(spoken, emo)

        hashes = set()
        for private in (False, True):
            [(audio, sentence)] = model.generate_streaming(text, "es", duration=dur, generator=seeded(private), **kw)
            want = model.generate(text, "es", duration=dur(text), generator=seeded(private), **kw)[0]
            assert sentence == text
            assert pcm_hash(audio) == pcm_hash(want), (backend, text, private)
            hashes.add(pcm_hash(audio))
        assert len(hashes) == 1, (backend, text)  # the private generator gives the global seed's audio
