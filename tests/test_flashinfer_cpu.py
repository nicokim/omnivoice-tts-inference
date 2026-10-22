import gc
import importlib.util
import re
import sys
import weakref

import pytest
import torch
from conftest import tiny_model
from torch import nn

from omnivoice_tts.models.omnivoice import OmniVoice
from omnivoice_tts.models.omnivoice_int8 import Int8Linear, apply_int8

MODULE = "omnivoice_tts.models.omnivoice_flashinfer"
# find_spec, not importorskip: collection must not import flashinfer outside the sandbox below
needs_flashinfer = pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None,
                                      reason="flashinfer is not installed")


def test_import_error_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    with pytest.raises(ImportError, match=re.escape("flashinfer-jit-cache==0.6.15.post1+cu128")):
        importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def fi(tmp_path_factory):
    """The module, imported with HOME and TMPDIR in a temp dir: flashinfer, tvm-ffi and CuTe write caches there."""
    assert "flashinfer" not in sys.modules  # else its import already wrote them under the real HOME
    sandbox = str(tmp_path_factory.mktemp("home"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", sandbox)
        mp.setenv("TMPDIR", sandbox)
        return importlib.import_module(MODULE)


@pytest.fixture
def fake_cuda(fi, monkeypatch):
    """``fi``, with CUDA available on an SM 12.0 card and an arch FlashInfer could build for it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(fi.flashinfer.jit.current_compilation_context, "TARGET_CUDA_ARCHS", {(12, "0a")})
    return fi


def plain_tensors(*objs):
    return [k for o in objs for k, v in vars(o).items() if isinstance(v, torch.Tensor)]


@needs_flashinfer
def test_fusion_frees_originals_and_keeps_4_linears_per_layer(fake_cuda):
    model = tiny_model(torch.float16)
    layers = model.llm.layers
    n_params = sum(p.numel() for p in model.parameters())
    want = [(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).detach(),
             torch.cat([m.gate_proj.weight, m.up_proj.weight]).detach())
            for a, m in ((layer.self_attn, layer.mlp) for layer in layers)]
    refs = [weakref.ref(lin.weight) for layer in layers for lin in (
        layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.mlp.gate_proj, layer.mlp.up_proj)]

    assert fake_cuda.apply_flashinfer(model) is model  # not under no_grad: the module must not need it
    gc.collect()
    assert [i for i, r in enumerate(refs) if r() is not None] == []
    for layer, (qkv, gate_up) in zip(layers, want):
        fused = (layer.self_attn.qkv_proj.weight, layer.mlp.gate_up_proj.weight)
        assert torch.equal(fused[0], qkv) and torch.equal(fused[1], gate_up)
        assert all(not w.requires_grad and w.grad_fn is None for w in fused)
        assert sum(isinstance(m, nn.Linear) for m in layer.modules()) == 4  # qkv, o, gate_up, down
    assert sum(p.numel() for p in model.parameters()) == n_params
    assert model._fi_runner.wrapper is None  # lazy: nothing allocated until the first plan
    assert model._fi_runner.dtype == torch.float16  # the weights' dtype: tiny_model's config says None


GUARDS = {  # case: (weights, exception, message)
    "no_cuda": (torch.float16, ValueError, "needs CUDA"),
    "fp32": (torch.float32, ValueError, "needs fp16 or bf16 weights, got torch.float32"),
    "int8": (torch.float16, TypeError, "apply_int8"),
    "no_arch": (torch.float16, RuntimeError, "FLASHINFER_CUDA_ARCH_LIST"),
}


@needs_flashinfer
@pytest.mark.parametrize("case", GUARDS)
def test_guards_leave_the_model_untouched(fake_cuda, monkeypatch, case):
    dtype, error, message = GUARDS[case]
    model = tiny_model(dtype)
    if case == "no_cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elif case == "int8":
        apply_int8(model)
    elif case == "no_arch":
        monkeypatch.setattr(fake_cuda.flashinfer.jit.current_compilation_context, "TARGET_CUDA_ARCHS", set())
    with pytest.raises(error, match=re.escape(message)):
        fake_cuda.apply_flashinfer(model)
    assert not hasattr(model, "_fi_runner") and hasattr(model.llm.layers[0].self_attn, "q_proj")
    assert "_generate_iterative" not in vars(model)


@needs_flashinfer
def test_flashinfer_then_int8(fake_cuda):
    model = apply_int8(fake_cuda.apply_flashinfer(tiny_model(torch.float16)))
    layers = model.llm.layers
    assert [sum(isinstance(m, Int8Linear) for m in layer.modules()) for layer in layers] == [4, 4]
    assert not any(isinstance(m, nn.Linear) for m in layers.modules())
    assert plain_tensors(*layers.modules()) == []


@needs_flashinfer
def test_apply_flashinfer_is_idempotent(fake_cuda):
    model = fake_cuda.apply_flashinfer(tiny_model(torch.float16))
    runner, modules = model._fi_runner, list(model.named_modules())
    assert fake_cuda.apply_flashinfer(model) is model and model._fi_runner is runner
    assert list(model.named_modules()) == modules  # nothing fused or swapped again


@needs_flashinfer
@pytest.mark.parametrize("int8", [False, True], ids=["flashinfer", "flashinfer_int8"])
def test_patched_model_moves_whole(fake_cuda, int8):
    model = fake_cuda.apply_flashinfer(tiny_model(torch.float16))
    if int8:
        apply_int8(model)
    assert plain_tensors(*model.modules(), model._fi_runner) == []  # the runner holds none until its first plan
    model.llm.to("meta")
    tensors = [*model.llm.parameters(), *model.llm.buffers()]
    assert sum(t.dtype == torch.int8 for t in tensors) == (8 if int8 else 0) and all(t.is_meta for t in tensors)


def class_attrs(classes):
    """What a class-level patch would replace for every model in the process, unseen by an instance's vars():
    OmniVoice's own _generate_iterative and each module class's own forward."""
    return vars(OmniVoice)["_generate_iterative"], {c: vars(c).get("forward") for c in classes}


@needs_flashinfer
def test_two_models_share_no_state(fake_cuda):
    a, b = tiny_model(torch.float16), tiny_model(torch.float16)
    b_modules, classes = list(b.named_modules()), {type(m) for m in b.modules()}  # a's classes too
    before = class_attrs(classes)
    fake_cuda.apply_flashinfer(a)
    assert class_attrs(classes) == before
    assert not hasattr(b, "_fi_runner") and "_generate_iterative" not in vars(b)
    assert b._generate_iterative.__func__ is OmniVoice._generate_iterative
    assert list(b.named_modules()) == b_modules and not any("forward" in vars(m) for m in b.modules())
    assert not hasattr(fake_cuda, "_CTX") and plain_tensors(fake_cuda) == []

    fake_cuda.apply_flashinfer(b)
    assert class_attrs(classes) == before
    assert b._fi_runner is not a._fi_runner
    assert all(layer.self_attn._fi_runner is m._fi_runner for m in (a, b) for layer in m.llm.layers)
    assert a._generate_iterative.__self__ is a and b._generate_iterative.__self__ is b
