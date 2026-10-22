import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import cfg_batch, tiny_model
from torch import nn
from torch.nn.modules.module import register_module_forward_hook
from torch.utils._python_dispatch import TorchDispatchMode

from omnivoice_tts.models.omnivoice import OmniVoice
from omnivoice_tts.models.omnivoice_int8 import Int8Linear, apply_int8

# bench2's int8.per_layer_sha256 (perf/raw/listen_B.json): 28 layers x 8 hashes, which equal slim.py's int8_layers run
# on CUDA. slim.py on the CPU gives f0dfc9af... instead, and is not a reference.
KOHAI_ROWS = json.loads((Path(__file__).parent / "data/int8_kohai_per_layer_sha256.json").read_text(encoding="utf-8"))
# the hash of those rows, as bench2 hashes them
KOHAI_SHA256 = "d8e4b4726f653815be095faa661c3ca3ffee8f35404bea4100771f3b3f0d5011"
KOHAI_DIR = os.environ.get("OMNIVOICE_TEST_KOHAI_DIR")  # a local KohAI fp16 export

# bench2's recipe (perf/perf.py int8_report): per layer, one sha256 per group and kind, in this order, over the bytes
# of the separate modules or of the fused one (the same bytes either way)
GROUPS = {"qkv": (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("self_attn.qkv_proj",)),
          "o": (("self_attn.o_proj",),), "gate_up": (("mlp.gate_proj", "mlp.up_proj"), ("mlp.gate_up_proj",)),
          "down": (("mlp.down_proj",),)}


def per_layer_sha256(model):
    rows = []
    for layer in model.llm.layers:
        named, row = dict(layer.named_modules()), {}
        for g, alts in GROUPS.items():
            mods = [named[n] for n in next(a for a in alts if all(n in named for n in a))]
            for kind, attr in (("q", "qweight"), ("scale", "scale")):
                h = hashlib.sha256()
                for m in mods:
                    h.update(getattr(m, attr).detach().cpu().contiguous().numpy().tobytes())
                row[f"{g}.{kind}"] = h.hexdigest()
        rows.append(row)
    return rows


class Ops(TorchDispatchMode):
    """The names of the aten ops that run under it."""

    def __init__(self):
        super().__init__()
        self.names = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def linear(dtype, bias=False):
    """A 64 -> 96 Linear with an all-zero row (3) and an outlier row (5)."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(96, 64, generator=g) * 0.05
    w[3] = 0
    w[5, 7] = 4.0
    lin = nn.Linear(64, 96, bias=bias).to(dtype)
    with torch.no_grad():
        lin.weight.copy_(w)
        if bias:
            lin.bias.copy_(torch.randn(96, generator=g))
    return lin


def logits(m):
    with torch.inference_mode():
        return m(*cfg_batch()).logits


DTYPES = pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["float16", "float32"])


@DTYPES
def test_roundtrip_error_bound(dtype):
    lin = linear(dtype)
    q = Int8Linear(lin)
    assert q.qweight.dtype == torch.int8 and q.qweight.min() >= -127 and q.qweight.max() <= 127
    assert q.scale.dtype == dtype and q.scale.shape == (96, 1)
    err = (q.qweight.float() * q.scale.float() - lin.weight.float()).abs()
    assert (err <= 0.63 * q.scale.float() + 1e-12).all()  # s / 2, plus s rounded to fp16 times up to 127
    deq = q.qweight * q.scale
    assert deq.dtype == dtype and torch.equal(deq, q.qweight.to(dtype) * q.scale)  # slim.py's dequant, bit for bit
    assert not deq[3].any()


def test_quantization_formula():
    for dtype in (torch.float16, torch.float32):
        lin = linear(dtype)
        w = lin.weight.float()
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) * (1 / 127)  # in fp32, whatever the weight's dtype
        q = Int8Linear(lin)
        assert torch.equal(q.qweight, torch.round(w / s).to(torch.int8))
        assert torch.equal(q.scale, s.to(dtype))
    # slim.py writes `/ 127`, which CUDA runs as the multiplication above and the CPU as a true division. This row is
    # where they part: the CPU's division rounds 63.5 down to 63
    row = torch.tensor([[0.0106658935546875, 0.00533294677734375]], dtype=torch.float16)
    lin = nn.Linear(2, 1, bias=False).to(torch.float16)
    with torch.no_grad():
        lin.weight.copy_(row)
    assert Int8Linear(lin).qweight.tolist() == [[127, 64]]
    w = row.float()
    slim = torch.round(w / (w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127)).to(torch.int8)
    assert slim.tolist() == [[127, 63]]


@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
@DTYPES
def test_forward_is_one_expression(dtype, bias):
    lin = linear(dtype, bias)
    m = Int8Linear(lin)
    assert m.bias is lin.bias  # carried over, not dropped
    x = torch.randn(10, 64, generator=torch.Generator().manual_seed(1)).to(dtype)
    assert torch.equal(m(x), F.linear(x, m.qweight * m.scale, m.bias))
    # slim.py's `q.to(x.dtype) * scale` gives the same bits, with a second kernel for the cast
    with Ops() as ops:
        m(x)
    assert ops.names.count("mul") == 1 and "_to_copy" not in ops.names


def test_apply_int8_scope_and_idempotence():
    model = tiny_model(torch.float16)
    codec = list(model.audio_tokenizer.modules())
    fp16 = logits(model)
    assert apply_int8(model) is model
    layers = list(model.llm.layers)
    assert [sum(isinstance(m, Int8Linear) for m in layer.modules()) for layer in layers] == [7, 7]
    assert not any(isinstance(m, nn.Linear) for layer in layers for m in layer.modules())
    assert all(m.qweight.dtype == torch.int8 and m.scale.dtype == torch.float16
               for m in model.llm.layers.modules() if isinstance(m, Int8Linear))
    assert type(model.audio_heads) is nn.Linear and model.audio_heads.weight.dtype == torch.float16
    # the rest of the model: embeddings, norms and heads stay fp16, the RoPE buffer fp32, the codec the same objects
    assert {p.dtype for k, p in model.named_parameters() if not k.startswith("audio_tokenizer.")} == {torch.float16}
    assert model.llm.rotary_emb.inv_freq.dtype == torch.float32
    assert list(model.audio_tokenizer.modules()) == codec  # Module has no __eq__: the same objects, in order

    int8 = logits(model)
    assert (fp16.float() - int8.float()).norm() / fp16.float().norm() < 0.05
    modules = list(model.named_modules())
    assert apply_int8(model) is model
    assert list(model.named_modules()) == modules  # the same objects under the same names: nothing was swapped again
    assert torch.equal(logits(model), int8)


def test_int8_calls_per_llm_forward():
    model = apply_int8(tiny_model(torch.float16))
    calls = {Int8Linear: 0, nn.Linear: 0}

    def count(module, args, output):
        if type(module) in calls:
            calls[type(module)] += 1

    handle = register_module_forward_hook(count)  # every module call in the process, so a Linear anywhere shows up
    try:
        with torch.inference_mode():
            model.llm(inputs_embeds=torch.randn(2, 40, 64, dtype=torch.float16), attention_mask=cfg_batch()[2],
                      use_cache=False)
    finally:
        handle.remove()
    assert calls == {Int8Linear: 14, nn.Linear: 0}


def test_int8_model_moves_whole():
    model = apply_int8(tiny_model(torch.float16))
    assert [k for m in model.modules() for k, v in vars(m).items() if isinstance(v, torch.Tensor)] == []
    model.llm.to("meta")
    tensors = [*model.llm.parameters(), *model.llm.buffers()]
    assert sum(t.dtype == torch.int8 for t in tensors) == 14 and all(t.is_meta for t in tensors)


def test_kohai_fixture_integrity():
    assert len(KOHAI_ROWS) == 28 and all(len(row) == 8 for row in KOHAI_ROWS)
    assert hashlib.sha256(json.dumps(KOHAI_ROWS).encode()).hexdigest() == KOHAI_SHA256


@pytest.mark.skipif(not KOHAI_DIR, reason="OMNIVOICE_TEST_KOHAI_DIR is not set")
def test_kohai_per_layer_sha256():
    rows = per_layer_sha256(apply_int8(OmniVoice.from_pretrained(KOHAI_DIR, dtype=torch.float16)))
    assert len(rows) == len(KOHAI_ROWS)
    assert [(i, k) for i, (r, e) in enumerate(zip(rows, KOHAI_ROWS)) for k in e if r[k] != e[k]] == []
    assert hashlib.sha256(json.dumps(rows).encode()).hexdigest() == KOHAI_SHA256
