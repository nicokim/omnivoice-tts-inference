import gc

import numpy as np
import pytest
import torch
from stubs import Recorder, stub_model

import omnivoice_tts.models.omnivoice as ov
from omnivoice_tts.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice_tts.utils.duration import RuleDurationEstimator

# One sentence of about 57 s by the rule (1,437 frames, past audio_chunk_threshold), which the stub sees as 5 calls.
# It is scripts/freeze_upstream.py's LONG_TEXTS[1]; [0] and [2] hold many sentences, so they would stream as many items.
LONG = ("La verdad que no sé qué decirte, " * 30) + "fin."


def stub(draw=False):
    rec = Recorder(draw=draw)
    return stub_model(OmniVoice, RuleDurationEstimator(), rec), rec


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def spied(monkeypatch):
    """Wrap the module's splitter and return ``(log, normalize)``: both write to ``log``, so ``log == []`` shows that
    neither the split nor any TN ran."""
    log, real = [], ov._split_sentences
    monkeypatch.setattr(ov, "_split_sentences", lambda t: log.append(t) or real(t))
    return log, lambda t, lang: log.append(t) or t


def test_one_sentence_is_generate():
    text = "Hola che, ¿todo bien?"
    m, rec = stub()
    items = list(m.generate_streaming(text, "es"))
    m2, rec2 = stub()
    assert len(items) == 1 and np.array_equal(items[0][0], m2.generate(text, "es")[0]) and rec.calls == rec2.calls
    assert items[0][1] is text  # the caller's own string
    assert [s for _, s in stub()[0].generate_streaming("  Hola  ", "es")] == ["  Hola  "]


def test_sentences_are_independent_generate_calls():
    text, sentences = "Hola. ¿Qué tal? Bien.", ["Hola.", "¿Qué tal?", "Bien."]
    m, _ = stub(draw=True)
    torch.manual_seed(5)
    got = [a for a, _ in m.generate_streaming(text, "es")]
    torch.manual_seed(5)
    assert len(got) == 3 and same(got, [m.generate(s, "es")[0] for s in sentences])
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    got = [a for a, _ in m.generate_streaming(text, "es", generator=g1)]
    assert same(got, [m.generate(s, "es", generator=g2)[0] for s in sentences])
    torch.manual_seed(5)
    assert not np.array_equal(got[1], m.generate("¿Qué tal?", "es")[0])  # item 1 goes on from item 0's draws


def test_hooks_order():
    events = []
    m, rec = stub()
    inner = m._generate_iterative
    m._generate_iterative = lambda task, *a: events.append(("gen", task.texts[0])) or inner(task, *a)

    def normalize(sentence, language):
        events.append(("norm", sentence, language))
        return sentence + " uno"

    def duration(spoken):
        events.append(("dur", spoken))
        return 1.234

    items = list(m.generate_streaming("Hola. ¿Qué tal?", "Spanish", normalize=normalize, duration=duration))
    assert events == [("norm", "Hola.", "es"), ("dur", "Hola. uno"), ("gen", "Hola. uno"),
                      ("norm", "¿Qué tal?", "es"), ("dur", "¿Qué tal? uno"), ("gen", "¿Qué tal? uno")]
    # texts, target frames (max(1, int(1.234 * 25))) and languages that reached the model
    assert [c[1:4] for c in rec.calls] == [[["Hola. uno"], [30], ["es"]], [["¿Qué tal? uno"], [30], ["es"]]]
    assert [s for _, s in items] == ["Hola.", "¿Qué tal?"]  # the yielded text stays raw


@pytest.mark.parametrize("bad", [3.0, 3, np.float64(3.0), True])
def test_duration_must_be_callable(bad, monkeypatch):
    log, normalize = spied(monkeypatch)
    m, rec = stub()
    s = m.generate_streaming("Hola. Chau.", "es", duration=bad, normalize=normalize)
    assert rec.calls == [] and log == []  # creating the stream records nothing
    with pytest.raises(TypeError, match="callable"):
        next(s)
    with pytest.raises(StopIteration):
        next(s)
    assert rec.calls == [] and log == []  # it raised before the split and before any TN


@pytest.mark.parametrize("bad, exc", [({"num_steps": 8}, TypeError), ({"normalize_text": True}, TypeError),
                                      ({"language": "Klingon"}, ValueError)])
def test_config_fields_checked_first(bad, exc, monkeypatch):
    log, normalize = spied(monkeypatch)
    m, rec = stub()
    s = m.generate_streaming("Hola. Chau.", **{"language": "es", "normalize": normalize, **bad})
    with pytest.raises(exc):
        next(s)
    assert log == [] and rec.calls == []


def test_lazy_and_closes():
    seen = []
    kw = dict(normalize=lambda t, lang: seen.append(t) or t)
    m, rec = stub()
    s = m.generate_streaming("Uno. Dos. Tres.", "es", **kw)
    assert rec.calls == [] and seen == []  # creating the stream runs nothing
    s.close()
    assert next(s, None) is None and rec.calls == [] and seen == []
    for how in ("close", "del"):
        seen.clear()
        m, rec = stub()
        s = m.generate_streaming("Uno. Dos. Tres.", "es", **kw)
        next(s)
        assert len(rec.calls) == 1 and seen == ["Uno."]
        if how == "close":
            s.close()
            assert next(s, None) is None
        else:
            del s
            gc.collect()
        assert len(rec.calls) == 1 and seen == ["Uno."]  # nothing ran after the sentence in flight


def test_error_mid_stream():
    def normalize(sentence, language):
        if sentence == "Dos.":
            raise RuntimeError("tn")
        return sentence

    m, rec = stub()
    s = m.generate_streaming("Uno. Dos. Tres.", "es", normalize=normalize)
    audio, sentence = next(s)
    before = audio.copy()
    with pytest.raises(RuntimeError, match="tn"):
        next(s)
    assert sentence == "Uno." and audio.size > 0 and np.array_equal(audio, before)
    with pytest.raises(StopIteration):
        next(s)
    assert len(rec.calls) == 1


@pytest.mark.parametrize("text", ["", "   ", "...", "😀", "¿?"])
def test_without_speech_raises(text):
    seen = []
    m, rec = stub()
    s = m.generate_streaming(text, "es", normalize=lambda t, lang: seen.append(t) or t)
    with pytest.raises(ValueError, match="nothing to speak"):
        next(s)
    assert seen == [] and rec.calls == []


def test_lone_tag_is_spoken():
    items = list(stub()[0].generate_streaming("[laughter]"))
    assert len(items) == 1 and items[0][1] == "[laughter]"
    assert np.array_equal(items[0][0], stub()[0].generate("[laughter]")[0])


@pytest.mark.parametrize("empty", ["", " ", ".", "😀"])
def test_skips_sentences_tn_empties(empty):
    durations = []
    m, rec = stub(draw=True)
    g, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    items = list(m.generate_streaming("Hola. <3. Chau.", "es", generator=g,
                                      normalize=lambda t, lang: empty if t == "<3." else t,
                                      duration=lambda spoken: durations.append(spoken) or 1.0))
    assert [s for _, s in items] == ["Hola.", "Chau."] and len(rec.calls) == 2 and durations == ["Hola.", "Chau."]
    assert same([a for a, _ in items], [m.generate(s, "es", duration=1.0, generator=g2)[0] for s in ("Hola.", "Chau.")])
    m, rec = stub()
    assert list(m.generate_streaming("Hola. <3. Chau.", "es", normalize=lambda t, lang: empty)) == []
    assert rec.calls == []


def test_arguments_reach_every_sentence():
    def targets(speed):
        m, rec = stub()
        for s in ("Hola.", "¿Qué tal?"):
            m.generate(s, "es", speed=speed)
        return [c[2] for c in rec.calls]

    m, rec = stub()
    items = list(m.generate_streaming("Hola. ¿Qué tal?", "es", instruct="Female", speed=2.0, num_step=8,
                                      generation_config=OmniVoiceGenerationConfig(guidance_scale=1.5)))
    assert len(items) == 2 and [c[4] for c in rec.calls] == [["female"], ["female"]]
    assert [c[2] for c in rec.calls] == targets(2.0) == [[t[0] // 2] for t in targets(1.0)]  # generate()'s, halved
    assert all(c.num_step == 8 and c.guidance_scale == 1.5 for c in rec.configs) and len(rec.configs) == 2


@pytest.mark.parametrize("text", [["Hola."], b"Hola.", None, 3])
def test_rejects_non_str(text):
    m, rec = stub()
    s = m.generate_streaming(text, "es")
    with pytest.raises(TypeError):
        next(s)
    assert rec.calls == []


def test_caller_outside_inference_mode():
    hooks, between = [], []
    m, rec = stub()
    for _ in m.generate_streaming("Uno. Dos. Tres.", "es",
                                  normalize=lambda t, lang: hooks.append(torch.is_inference_mode_enabled()) or t,
                                  duration=lambda s: hooks.append(torch.is_inference_mode_enabled()) or 1.0):
        between.append(torch.is_inference_mode_enabled())
    assert hooks == [False] * 6 and between == [False] * 3 and rec.inference == [True] * 3
    m, _ = stub()
    with torch.inference_mode():  # the caller's own inference_mode
        items = list(m.generate_streaming("Uno. Dos.", "es"))
    assert len(items) == 2 and all(isinstance(a, np.ndarray) for a, _ in items)


def test_long_sentence_is_one_chunked_item():
    m, rec = stub()
    assert m._estimate_target_tokens(LONG, None, None) == 1437
    items = list(m.generate_streaming(LONG, "es"))
    assert [s for _, s in items] == [LONG] and len(rec.calls) == 5  # generate()'s chunked path, one item
    m, rec = stub()
    s = m.generate_streaming(LONG + " ¿Y ahora?", "es")
    audio, text = next(s)
    assert text == LONG and audio.size > 0 and len(rec.calls) == 5
    assert next(s)[1] == "¿Y ahora?" and len(rec.calls) == 6
    with pytest.raises(StopIteration):
        next(s)


def test_normalize_keeps_raw_subtitles(tn, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    m, rec = stub()
    items = list(m.generate_streaming("Son las 21:30. Me donaron $1.500.", "es", normalize=tn.normalize))
    assert [s for _, s in items] == ["Son las 21:30.", "Me donaron $1.500."]
    assert len(rec.calls) == 2 and not any(c.isdigit() for call in rec.calls for c in call[1][0])  # words reached it
    assert list(tmp_path.iterdir()) == []  # nothing written to HOME or XDG_CACHE_HOME
