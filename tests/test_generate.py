import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import cfg_batch, tiny_model
from stubs import Recorder, stub_model

import omnivoice_tts.models.omnivoice as ov
from omnivoice_tts.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice_tts.utils.duration import RuleDurationEstimator

UP = json.loads((Path(__file__).parent / "data/upstream_08be0b4.json").read_text(encoding="utf-8"))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()


def run(text, kwargs, rec=None):
    rec = rec or Recorder()
    return rec, stub_model(OmniVoice, RuleDurationEstimator(), rec).generate(text, **kwargs)[0]


def case_id(case):
    return f"{len(case['text'])}chars-" + ",".join(f"{k}={v}" for k, v in case["kwargs"].items())


# Fix 10 moves the later chunks of every text upstream split into several calls with a speed or a duration;
# _assert_fix10 covers those, and every other case stays identical to upstream.
PARITY = [
    c
    for c in UP["calls"] + UP["chunked"]
    if not (len(c["calls"]) > 1 and {"speed", "duration"} & c["kwargs"].keys())
]


def chunked_with(field):
    """The frozen cases upstream split into several calls, with ``field`` (speed or duration) among the kwargs."""
    return [c for c in UP["calls"] + UP["chunked"] if len(c["calls"]) > 1 and field in c["kwargs"]]


def _assert_fix10(case):
    """Same first call and chunk texts as upstream, but every later chunk is timed from chunk 0 at speed 1.0.

    Chunk 0's length already carries the speed and the later chunks are estimated from its rate, so upstream
    divided by the speed a second time.
    """
    rec = Recorder()
    m = stub_model(OmniVoice, RuleDurationEstimator(), rec)
    m.generate(case["text"], **case["kwargs"])
    frozen = case["calls"]
    assert rec.calls[0] == frozen[0]
    assert [c[1] for c in rec.calls] == [c[1] for c in frozen]
    t0, n0 = rec.calls[0][1][0], rec.calls[0][2][0]
    for call, old in zip(rec.calls[1:], frozen[1:]):
        assert call[2] == [m._estimate_target_tokens(call[1][0], t0, n0, speed=1.0)]
        assert call[2] != old[2]


@pytest.mark.parametrize("case", PARITY, ids=case_id)
def test_generate_matches_upstream_call_sequence(case):
    rec, out = run(case["text"], case["kwargs"])
    assert rec.calls == case["calls"] and sha(out) == case["pcm_sha256"] and len(out) == case["n_samples"]


@pytest.mark.parametrize("case", chunked_with("speed"), ids=case_id)
def test_later_chunks_not_sped_up_twice(case):
    _assert_fix10(case)


PLAIN_CHUNKED = [c for c in UP["chunked"] if not {"speed", "duration"} & c["kwargs"].keys()]


@pytest.mark.parametrize("case", PLAIN_CHUNKED + chunked_with("duration"), ids=case_id)
def test_generate_chunked_matches_upstream(case):
    if "duration" in case["kwargs"]:
        _assert_fix10(case)
    else:
        rec, out = run(case["text"], case["kwargs"])
        assert rec.calls == case["calls"] and sha(out) == case["pcm_sha256"] and len(out) == case["n_samples"]


def test_duration_estimator_matches_upstream():
    est = RuleDurationEstimator()
    for text, weight, dur in UP["duration"]:  # isclose: libm's pow may differ in the last bit on another machine
        assert math.isclose(est.calculate_total_weight(text), weight, rel_tol=1e-12), text
        assert math.isclose(est.estimate_duration(text, "Nice to meet you.", 25), dur, rel_tol=1e-12), text


def test_combine_text_unchanged_without_newlines():
    assert all(ov._combine_text(t, r) == out for t, r, out in UP["combine_text"])


def test_newline_becomes_space():
    assert ov._combine_text("Hola che.\nTodo bien?") == "Hola che. Todo bien?"
    assert ov._combine_text("a\r\n\r\nb") == "a b"
    assert ov._combine_text("你好\n世界") == "你好世界"


def test_instruct_normalised_like_upstream_and_never_rejected():
    for s, zh, out, rejected in UP["instruct"]:
        got = ov._resolve_instruct(s, use_zh=zh)  # never raises
        assert got is None or isinstance(got, str)
        if not rejected:
            assert got == out
    assert ov._resolve_instruct("Melancholic") == "melancholic"  # upstream rejected it without the KohAI hack


def test_language_accepted_like_upstream():
    assert all(ov._resolve_language(x) == out for x, out, accepted in UP["language"] if accepted)


@pytest.mark.parametrize("text", [["Hola."], b"Hola.", None, 3])
def test_list_text_raises_typeerror(text):
    rec = Recorder()
    with pytest.raises(TypeError, match="text must be a str"):
        run(text, {}, rec)
    assert rec.calls == []


def test_positional_after_language_raises():
    rec = Recorder()
    with pytest.raises(TypeError):
        stub_model(OmniVoice, RuleDurationEstimator(), rec).generate("Hola.", "es", "female")
    assert rec.calls == []


def test_generate_needs_from_pretrained():
    rec = Recorder()
    m = stub_model(OmniVoice, RuleDurationEstimator(), rec)
    m.text_tokenizer = None
    with pytest.raises(RuntimeError, match="from_pretrained"):
        m.generate("Hola.")
    assert rec.calls == []


def test_generate_end_to_end_on_tiny_model():
    m, outs = tiny_model(), []
    for _ in range(2):
        torch.manual_seed(1)
        outs.append(m.generate("hola [laughter] hola.", "es", num_step=4, postprocess_output=False))
    a, b = outs
    assert len(a) == 1 and a[0].dtype == np.float32 and a[0].shape == (45120,)  # 42 frames x 960 + 2 x 2400 pad
    assert np.array_equal(a[0], b[0])


def test_near_silent_output_returns_empty():
    quiet = (np.random.default_rng(0).standard_normal((1, 24000)) * 1e-4).astype(np.float32)
    stub = stub_model(OmniVoice, RuleDurationEstimator(), Recorder())
    assert stub._post_process_audio(quiet, OmniVoiceGenerationConfig()).shape == (1, 0)
    stub.audio_tokenizer.decode = lambda t: SimpleNamespace(audio_values=torch.from_numpy(quiet)[None])
    assert stub.generate("Hola.")[0].shape == (0,)


def test_unknown_generate_kwarg_raises_typeerror():
    for extra in ({}, {"generation_config": OmniVoiceGenerationConfig()}):
        rec = Recorder()
        with pytest.raises(TypeError):
            run("Hola.", {"num_steps": 8, **extra}, rec)
        assert rec.calls == []


def test_kwargs_override_generation_config():
    rec, cfg = Recorder(), OmniVoiceGenerationConfig(num_step=32)
    run("Hola.", {"generation_config": cfg, "num_step": 16}, rec)
    assert rec.configs[0].num_step == 16 and cfg.num_step == 32


@pytest.mark.parametrize("language", ["castellano", "Klingon", "es-AR", " es "])  # P7: no strip, as upstream
def test_unknown_language_raises(language):
    rec = Recorder()
    with pytest.raises(ValueError, match="supported_language_ids"):
        run("Hola.", {"language": language}, rec)
    assert rec.calls == []


def test_language_id_case_insensitive():
    assert [ov._resolve_language(x) for x in ("ES", "es", "Spanish")] == ["es", "es", "es"]
    assert all(ov._resolve_language(x) == out for x, out, accepted in UP["language"] if accepted)


def test_audio_chunk_duration_zero_disables_split():
    text = UP["chunked"][0]["text"]
    assert len(run(text, {"audio_chunk_duration": 0.0})[0].calls) == 1
    assert len(run(text, {})[0].calls) > 1


def test_forward_requires_attention_mask():
    m, (input_ids, audio_mask, attention_mask), seen = tiny_model(), cfg_batch(), []
    with pytest.raises(TypeError):
        m(input_ids, audio_mask)

    original = m.llm.forward

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    m.llm.forward = spy
    out = m(input_ids, audio_mask, attention_mask)
    assert out.logits.shape == (2, 8, 40, 1025)
    assert seen[0]["use_cache"] is False


def test_text_without_speech_raises():
    for text, kwargs in [("", {}), ("   ", {}), ("...", {}), ("😀", {}), ("¿?", {}), ("", {"duration": 40.0})]:
        rec = Recorder()
        with pytest.raises(ValueError, match="nothing to speak"):
            run(text, kwargs, rec)
        assert rec.calls == []
    stub = stub_model(OmniVoice, RuleDurationEstimator(), Recorder())
    assert len(stub.generate("[laughter]")) == 1  # a tag alone is speech


def test_gumbel_sample_matches_rand_like():
    logits = torch.randn(1, 8, 30, 1025)
    for lg in (logits, logits.permute(0, 2, 1, 3)):
        torch.manual_seed(3)
        s = lg / 0.7
        u = torch.rand_like(s)
        want = s + -torch.log(-torch.log(u + 1e-10) + 1e-10)  # upstream's formula
        torch.manual_seed(3)
        assert torch.equal(ov._gumbel_sample(lg, 0.7), want)
        assert torch.equal(ov._gumbel_sample(lg, 0.7, torch.Generator().manual_seed(3)), want)


def test_generator_reaches_every_chunk():
    g, rec = torch.Generator().manual_seed(1), Recorder(draw=True)
    run(UP["chunked"][0]["text"], {"language": "es", "generator": g}, rec)
    assert len(rec.calls) > 1 and all(x is g for x in rec.gens)


def test_private_generator_matches_global_seed():
    m, seen = tiny_model(), []
    m._decode_and_post_process = lambda tokens, gen_config: seen.append(tokens) or np.zeros(1, np.float32)
    kw = dict(num_step=4, class_temperature=1.0)
    torch.manual_seed(7)
    m.generate("hola [laughter] hola.", "es", **kw)
    m.generate("hola [laughter] hola.", "es", generator=torch.Generator().manual_seed(7), **kw)
    torch.manual_seed(8)
    m.generate("hola [laughter] hola.", "es", **kw)
    assert torch.equal(seen[0], seen[1]) and not torch.equal(seen[0], seen[2])  # tokens: tiny audio hides the RNG


def test_generator_leaves_global_rng_untouched():
    m = tiny_model()
    state = torch.get_rng_state()
    kw = dict(num_step=4, class_temperature=1.0, postprocess_output=False)  # both gumbel draws are live
    m.generate("hola [laughter] hola.", "es", generator=torch.Generator().manual_seed(7), **kw)
    assert torch.equal(torch.get_rng_state(), state)
    m.generate("hola [laughter] hola.", "es", **kw)  # control: without a generator the global RNG does move
    assert not torch.equal(torch.get_rng_state(), state)
