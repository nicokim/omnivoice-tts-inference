"""Optional int8 weight-only mode for the LLM decoder layers.

Every nn.Linear inside ``model.llm.layers`` (q/k/v/o/gate/up/down) keeps an int8
weight plus one scale per output channel and is dequantized on each call. Text
embedding, audio embeddings, audio heads, norms and the codec stay as loaded.
Audio is NOT bit-identical to the fp16 model.

Quantize on CPU, then move, so the fp16 layer weights never reach the GPU::

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", dtype=torch.float16)
    apply_int8(model).to("cuda:0")

Idempotent. A quantized model cannot be saved.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Int8Linear(nn.Module):
    """Symmetric int8 weight, one scale per output channel, in the weight's dtype."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().float()
        # * (1 / 127), not / 127: CUDA turns a scalar division into this multiply, so the
        # weights come out identical on CPU and GPU.
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) * (1 / 127)
        self.register_buffer("qweight", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.to(linear.weight.dtype))
        self.bias = linear.bias

    def forward(self, x):
        # int8 * fp16 -> fp16 in one kernel. ponytail: dequantizes on every call, RTF +8 % on sdpa
        # (0.0530 -> 0.0572) and +31-34 % on FlashInfer; a fused int8 GEMM if that matters.
        return F.linear(x, self.qweight * self.scale, self.bias)


def apply_int8(model):
    """Swap every nn.Linear in ``model.llm.layers`` for an Int8Linear. Returns ``model``."""
    for layer in model.llm.layers:
        for module in list(layer.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(module, name, Int8Linear(child))
    return model
