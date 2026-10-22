import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import tiny_model
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


@pytest.mark.parametrize("case", UP["calls"] + UP["chunked"])
def test_generate_matches_upstream_call_sequence(case):
    rec, out = run(case["text"], case["kwargs"])
    assert rec.calls == case["calls"] and sha(out) == case["pcm_sha256"] and len(out) == case["n_samples"]


def test_duration_estimator_matches_upstream():
    est = RuleDurationEstimator()
    for text, weight, dur in UP["duration"]:
        assert est.calculate_total_weight(text) == weight and est.estimate_duration(text, "Nice to meet you.", 25) == dur


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
    with pytest.raises(TypeError):
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
