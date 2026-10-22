import logging
import os
import shutil
import sys
import warnings

import numpy as np
import pytest
import torch
from conftest import cfg_batch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, HiggsAudioV2TokenizerModel, PreTrainedModel
from transformers.models.auto import CONFIG_MAPPING

import omnivoice_tts.models.omnivoice as ov
from omnivoice_tts.models.omnivoice import OmniVoice, OmniVoiceConfig


def tensors(m):
    """Parameters and buffers (inv_freq included) by name, without the codec."""
    return {k: t for k, t in [*m.named_parameters(), *m.named_buffers()] if not k.startswith("audio_tokenizer.")}


def logits(m):
    with torch.inference_mode():
        return m(*cfg_batch()).logits


def follow_capfd(monkeypatch):
    """transformers and huggingface_hub bound their log handlers at import, to the sys.stderr pytest's capture had
    then, so capfd would not see their warnings; point them at capfd's stream for this test."""
    for name in ("transformers", "huggingface_hub"):
        for h in logging.getLogger(name).handlers:
            if type(h) is logging.StreamHandler:  # pytest's own handlers are subclasses
                monkeypatch.setattr(h, "stream", sys.stderr)


def test_loader_matches_transformers_from_pretrained(tiny_dir):
    ref = PreTrainedModel.from_pretrained.__func__(OmniVoice, tiny_dir, dtype=torch.float16)
    m = OmniVoice.from_pretrained(str(tiny_dir), dtype=torch.float16)
    a, b = tensors(ref), tensors(m)
    assert {k: t.dtype for k, t in a.items()} == {k: t.dtype for k, t in b.items()}
    assert [k for k in a if not torch.equal(a[k], b[k])] == []
    assert m.llm.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.equal(logits(ref), logits(m))

    half = OmniVoice.from_pretrained(str(tiny_dir), dtype=torch.float32).half()  # what the README warns against
    assert half.llm.rotary_emb.inv_freq.dtype == torch.float16
    assert not torch.equal(logits(half), logits(m))


def test_load_is_silent_and_global_free(tiny_dir, capfd, monkeypatch):
    OmniVoice.from_pretrained(str(tiny_dir))  # transformers' lazy imports happen here, before the snapshot

    def snapshot():
        return (torch.get_rng_state(), dict(os.environ), list(warnings.filters), logging.root.manager.disable,
                torch.get_default_dtype(), "omnivoice" in CONFIG_MAPPING)

    before, calls, real = snapshot(), [], torch.set_default_dtype
    monkeypatch.setattr(torch, "set_default_dtype", lambda d: calls.append(d) or real(d))
    follow_capfd(monkeypatch)
    capfd.readouterr()
    OmniVoice.from_pretrained(str(tiny_dir))
    assert capfd.readouterr() == ("", "")
    after = snapshot()
    assert torch.equal(after[0], before[0]) and after[1:] == before[1:]
    assert calls == []


def assert_decode_only_matches_full(path, frames):
    full = HiggsAudioV2TokenizerModel.from_pretrained(path).eval()
    slim = ov._load_decode_only_codec(str(path), torch.device("cpu"))
    codes = torch.randint(0, 1024, (1, 8, frames), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        assert torch.equal(full.decode(codes).audio_values, slim.decode(codes).audio_values)
    return slim


def test_decode_only_codec_matches_full_tiny(tiny_dir):
    slim = assert_decode_only_matches_full(tiny_dir / "audio_tokenizer", 6)
    assert not hasattr(slim, "acoustic_encoder")


def test_decode_only_codec_matches_full():
    try:
        path = snapshot_download(ov._CODEC_REPO, revision=ov._CODEC_REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip("eustlb/higgs-audio-v2-tokenizer@528e871c is not in the local HF cache")
    slim = assert_decode_only_matches_full(path, 60)
    assert len(slim.state_dict()) == 176


def test_tokenizer_needs_no_auto_registry(tiny_dir, capfd, monkeypatch):
    follow_capfd(monkeypatch)
    capfd.readouterr()
    tok = AutoTokenizer.from_pretrained(tiny_dir, config=OmniVoiceConfig.from_pretrained(tiny_dir))
    assert capfd.readouterr().err == ""
    assert tok.convert_tokens_to_ids("<|denoise|>") == 2
    assert "omnivoice" not in CONFIG_MAPPING


def test_hub_kwargs_forwarded_and_codec_pinned(tiny_dir, tmp_path, monkeypatch):
    repo = shutil.copytree(tiny_dir, tmp_path / "repo", ignore=shutil.ignore_patterns("audio_tokenizer"))
    calls = []

    def spy(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(repo if repo_id == "org/model" else tiny_dir / "audio_tokenizer")

    monkeypatch.setattr(ov, "snapshot_download", spy)
    cache = str(tmp_path / "c")
    OmniVoice.from_pretrained("org/model", revision="r1", token="t", cache_dir=cache, local_files_only=True)
    hub = dict(token="t", cache_dir=cache, local_files_only=True, tqdm_class=ov._NoBar)
    assert calls == [
        ("org/model", dict(revision="r1", allow_patterns=ov._MODEL_FILES, **hub)),
        (ov._CODEC_REPO, dict(revision=ov._CODEC_REVISION, allow_patterns=["config.json", "model.safetensors"], **hub)),
    ]


@pytest.mark.parametrize("kwarg", ["train", "load_asr", "asr_model_name", "asr_device", "attn_implementation",
                                   "torch_dtype", "trust_remote_code"])
def test_from_pretrained_rejects_upstream_kwargs(kwarg):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        OmniVoice.from_pretrained("org/model", **{kwarg: True})


def test_device_map_rejects_auto_and_dicts(monkeypatch):
    def no_io(*args, **kwargs):
        raise AssertionError("snapshot_download ran before the device_map check")

    monkeypatch.setattr(ov, "snapshot_download", no_io)
    for device_map in ("auto", "balanced", "balanced_low_0", "sequential", {"": "cpu"}):
        with pytest.raises(ValueError, match="device_map"):
            OmniVoice.from_pretrained("org/model", device_map=device_map)


def test_bad_checkpoints_raise(tiny_dir, tmp_path):
    d = shutil.copytree(tiny_dir, tmp_path / "no_weights")
    (d / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        OmniVoice.from_pretrained(str(d))

    d = shutil.copytree(tiny_dir, tmp_path / "missing_key")
    sd = load_file(tiny_dir / "model.safetensors")  # read the original: the copy is rewritten below
    del sd["audio_heads.weight"]
    save_file(sd, d / "model.safetensors")
    with pytest.raises(RuntimeError, match="audio_heads.weight"):
        OmniVoice.from_pretrained(str(d))


def test_config_keeps_unknown_keys():
    weights = [8, 8, 6, 6, 4, 4, 2, 2]
    assert OmniVoiceConfig(audio_codebook_weights=weights).audio_codebook_weights == weights
    assert "audio_codebook_weights" not in OmniVoiceConfig().to_dict()


def test_loaded_model_generates(tiny_dir):
    out = OmniVoice.from_pretrained(str(tiny_dir)).generate("hola.", "es", num_step=2, postprocess_output=False)[0]
    assert out.dtype == np.float32 and out.shape == (30720,)  # 27 frames x 960 + 2 x 2400 of padding
