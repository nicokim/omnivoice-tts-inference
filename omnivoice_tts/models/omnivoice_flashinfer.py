# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by omnivoice-tts-inference from k2-fsa/OmniVoice 08be0b4
# (omnivoice/models/omnivoice_flashinfer.py): per-model state instead of the
# module-level _CTX and the global "omnivoice_fi" attention registration, fused
# q/k/v and gate/up weights replace the originals, plan dtype follows the model,
# CUDA graphs removed, private generator for the Gumbel noise.
"""FlashInfer-accelerated iterative decoding for OmniVoice.

- Sequence packing: the baseline pads the uncond (CFG) sequence to the cond
  length and runs batch=2 with a (2,1,S,S) bool mask. Here cond+uncond are
  packed into ONE row of length c_len+u_len with per-document positions and
  flashinfer ragged attention (qo_indptr = document boundaries) - no pad
  compute, no S^2 mask materialization.
- Kernels: flashinfer RMSNorm, in-place RoPE in NHD layout, one GEMM for q/k/v
  and one for gate/up followed by silu_and_mul; audio heads run on the target
  positions only. No KV cache.

Usage:
    from omnivoice_tts.models.omnivoice_flashinfer import apply_flashinfer
    apply_flashinfer(model)

Needs flashinfer-python 0.6.15.post1 and the matching flashinfer-jit-cache
wheel; see the README for the install command and FLASHINFER_CUDA_ARCH_LIST.
"""

import math
from types import MethodType

import torch
from torch import nn

try:
    import flashinfer
except ImportError as e:
    raise ImportError(
        "apply_flashinfer needs FlashInfer: pip install flashinfer-python==0.6.15.post1 "
        '"flashinfer-jit-cache==0.6.15.post1+cu128" --extra-index-url '
        "https://flashinfer.ai/whl/cu128/ (cu129 for a CUDA 12.9 torch)"
    ) from e

from omnivoice_tts.models.omnivoice import (
    GenerationTask,
    OmniVoiceGenerationConfig,
    _get_time_steps,
    _gumbel_sample,
)

_WORKSPACE_SIZE = 128 * 1024 * 1024


def _fi_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Single-kernel replacement for Qwen3RMSNorm.forward (a 7-kernel
    fp32-upcast chain in eager mode). flashinfer.norm.rmsnorm computes in
    fp32 internally and matches to fp16 rounding."""
    shape = hidden_states.shape
    out = flashinfer.norm.rmsnorm(
        hidden_states.reshape(-1, shape[-1]).contiguous(),
        self.weight,
        eps=self.variance_epsilon,
    )
    return out.view(shape)


def _patch_rmsnorm(llm):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    for module in llm.modules():
        if isinstance(module, Qwen3RMSNorm):
            module.forward = MethodType(_fi_rmsnorm_forward, module)


def _fused_linear(*linears: nn.Linear) -> nn.Linear:
    """One bias-free Linear whose weight is the row-concatenation of ``linears``."""
    # ponytail: bias-free like upstream; every OmniVoice LLM is Qwen3 with attention_bias=False
    weight = torch.cat([lin.weight for lin in linears])
    fused = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    fused.weight = nn.Parameter(weight, requires_grad=False)
    return fused


def _fi_attention_module_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    """NHD-layout replacement for Qwen3Attention.forward (packed batch=1).

    The stock forward works in (B, H, S, D): the rotate-half RoPE costs a cat
    plus four elementwise passes, and handing (B,H,S,D) to the ragged wrapper
    costs three transpose copies. Keeping everything in (S, H, D) removes all
    of that; RoPE is one fused in-place kernel driven by the packed position
    ids the runner holds for the current generation."""
    s = hidden_states.shape[1]
    q, k, v = self.qkv_proj(hidden_states[0]).split(self._fi_qkv_split, dim=-1)
    # split views are strided; reshape materializes contiguous copies
    # (q/k would be copied inside the fused rmsnorm anyway)
    q = self.q_norm(q.reshape(s, -1, self.head_dim))
    k = self.k_norm(k.reshape(s, -1, self.head_dim))
    v = v.reshape(s, -1, self.head_dim)
    runner = self._fi_runner
    flashinfer.rope.apply_rope_pos_ids_inplace(
        q, k, runner.pos_ids, rope_theta=self._fi_rope_theta, interleave=False
    )
    out = runner.wrapper.run(q, k, v)  # (S, Hq, D)
    return self.o_proj(out.reshape(s, -1)).unsqueeze(0), None


def _patch_attention_forward(llm, runner):
    theta = llm.config.rope_parameters["rope_theta"]
    for layer in llm.layers:
        attn = layer.self_attn
        attn._fi_rope_theta = theta
        attn._fi_runner = runner
        attn._fi_qkv_split = [
            attn.q_proj.out_features,
            attn.k_proj.out_features,
            attn.v_proj.out_features,
        ]
        attn.qkv_proj = _fused_linear(attn.q_proj, attn.k_proj, attn.v_proj)
        del attn.q_proj, attn.k_proj, attn.v_proj  # free the originals now
        attn.forward = MethodType(_fi_attention_module_forward, attn)


def _fi_mlp_forward(self, x):
    """Qwen3MLP with fused gate+up GEMM and flashinfer silu_and_mul
    (2 GEMMs + silu + mul -> 1 GEMM + 1 fused kernel)."""
    y = flashinfer.activation.silu_and_mul(self.gate_up_proj(x[0]))  # (S, inter)
    return self.down_proj(y).unsqueeze(0)


def _patch_mlp(llm):
    for layer in llm.layers:
        mlp = layer.mlp
        mlp.gate_up_proj = _fused_linear(mlp.gate_proj, mlp.up_proj)
        del mlp.gate_proj, mlp.up_proj
        mlp.forward = MethodType(_fi_mlp_forward, mlp)


class PackedAttnRunner:
    """Per-model flashinfer state: the ragged-prefill wrapper planned for the
    current packed layout, and that layout's int32 position ids."""

    def __init__(self, num_qo_heads, num_kv_heads, head_dim, dtype):
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.wrapper = None
        self._planned_key = None
        self.pos_ids = None

    def plan(self, doc_lens: list[int], device):
        key = tuple(doc_lens)
        if key == self._planned_key:
            return
        # ponytail: the runner's buffers are allocated on the first plan, on the model's device at that time;
        # move the model before the first generate()
        if self.wrapper is None:  # first use: lets apply_flashinfer run before model.to("cuda")
            workspace = torch.zeros(_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
            self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        indptr = torch.zeros(len(doc_lens) + 1, dtype=torch.int32, device=device)
        indptr[1:] = torch.cumsum(
            torch.tensor(doc_lens, dtype=torch.int32, device=device), dim=0
        )
        self.wrapper.plan(
            indptr,
            indptr,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=False,
            sm_scale=self.head_dim**-0.5,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
        self._planned_key = key


def _generate_iterative_packed(
    self,
    task: GenerationTask,
    gen_config: OmniVoiceGenerationConfig,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Packed-sequence rewrite of OmniVoice._generate_iterative.

    Documents are packed as [cond_0, uncond_0, cond_1, uncond_1, ...] into a
    single batch row; the scoring/unmasking math is identical to the original,
    and at batch 1 the Gumbel noise is drawn with the same shapes and order.
    """
    B = task.batch_size
    inputs_list = [
        self._prepare_inference_inputs(
            task.texts[i],
            task.target_lens[i],
            task.ref_texts[i],
            task.ref_audio_tokens[i],
            task.langs[i],
            task.instructs[i],
            gen_config.denoise,
        )
        for i in range(B)
    ]

    c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
    u_lens = list(task.target_lens)
    doc_lens = []
    for c, u in zip(c_lens, u_lens):
        doc_lens.extend([c, u])
    offsets = [0]
    for n in doc_lens[:-1]:
        offsets.append(offsets[-1] + n)
    total_len = sum(doc_lens)

    C = self.config.num_audio_codebook
    packed_ids = torch.full(
        (1, C, total_len),
        self.config.audio_mask_id,
        dtype=torch.long,
        device=self.device,
    )
    packed_audio_mask = torch.zeros(
        (1, total_len), dtype=torch.bool, device=self.device
    )
    position_ids = torch.zeros((1, total_len), dtype=torch.long, device=self.device)

    for i, inp in enumerate(inputs_list):
        c_off, u_off = offsets[2 * i], offsets[2 * i + 1]
        c_len, u_len = c_lens[i], u_lens[i]
        packed_ids[0, :, c_off : c_off + c_len] = inp["input_ids"][0]
        packed_audio_mask[0, c_off : c_off + c_len] = inp["audio_mask"][0]
        position_ids[0, c_off : c_off + c_len] = torch.arange(c_len, device=self.device)
        # uncond doc = target region only
        packed_ids[0, :, u_off : u_off + u_len] = inp["input_ids"][0, :, -u_len:]
        packed_audio_mask[0, u_off : u_off + u_len] = inp["audio_mask"][0, -u_len:]
        position_ids[0, u_off : u_off + u_len] = torch.arange(u_len, device=self.device)

    timesteps = _get_time_steps(
        t_start=0.0, t_end=1.0, num_step=gen_config.num_step, t_shift=gen_config.t_shift
    ).tolist()
    schedules = []
    for t_len in task.target_lens:
        total_mask = t_len * C
        rem = total_mask
        sched = []
        for step in range(gen_config.num_step):
            num = (
                rem
                if step == gen_config.num_step - 1
                else min(
                    math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem
                )
            )
            sched.append(int(num))
            rem -= int(num)
        schedules.append(sched)

    layer_ids = torch.arange(C, device=self.device).view(1, -1, 1)

    # gather indices of the logits-consuming positions, laid out as
    # [all cond-target blocks | all uncond blocks] so the guidance/scoring
    # math can run over every item in one batched pass. flat_spans[i] gives
    # the item's (start, len) within each half.
    cond_ranges, uncond_ranges = [], []
    flat_spans = []
    for i in range(B):
        c_off, u_off = offsets[2 * i], offsets[2 * i + 1]
        c_len, t_len = c_lens[i], task.target_lens[i]
        prev = 0 if i == 0 else flat_spans[-1][0] + flat_spans[-1][1]
        flat_spans.append((prev, t_len))
        cond_ranges.append(
            torch.arange(c_off + c_len - t_len, c_off + c_len, device=self.device)
        )
        uncond_ranges.append(torch.arange(u_off, u_off + t_len, device=self.device))
    T_flat = sum(task.target_lens)
    tgt_index = torch.cat(cond_ranges + uncond_ranges)

    # flat per-position token state aligned with the cond half of the gathered layout
    tokens_flat = torch.full(
        (C, T_flat), self.config.audio_mask_id, dtype=torch.long, device=self.device
    )

    runner = self._fi_runner
    runner.plan(doc_lens, self.device)
    runner.pos_ids = position_ids[0].to(torch.int32)

    for step in range(gen_config.num_step):
        batch_logits = _forward_logits(
            self, packed_ids, packed_audio_mask, position_ids, tgt_index
        ).to(torch.float32)

        # batched scoring over every item at once: the guidance/log_softmax/
        # argmax/gumbel chain (the GPU-heavy part) runs on the whole
        # [cond | uncond] halves; only topk + scatter stay per item.
        c_logits_all = batch_logits[:, :, :T_flat, :]
        u_logits_all = batch_logits[:, :, T_flat:, :]
        pred_all, scores_all = self._predict_tokens_with_scoring(
            c_logits_all, u_logits_all, gen_config, generator
        )
        scores_all = scores_all - (layer_ids * gen_config.layer_penalty_factor)
        if gen_config.position_temperature > 0.0:
            scores_all = _gumbel_sample(
                scores_all, gen_config.position_temperature, generator
            )
        # -inf for already-unmasked positions
        scores_all.masked_fill_(
            (tokens_flat != self.config.audio_mask_id).unsqueeze(0), -float("inf")
        )
        pred_all, scores_all = pred_all[0], scores_all[0]  # (C, T_flat)

        for i in range(B):
            k = schedules[i][step]
            if k <= 0:
                continue
            c_off, u_off = offsets[2 * i], offsets[2 * i + 1]
            c_len, t_len = c_lens[i], task.target_lens[i]
            st, _ = flat_spans[i]

            _, topk_idx = torch.topk(scores_all[:, st : st + t_len].reshape(-1), k)
            flat_tokens = tokens_flat[:, st : st + t_len].reshape(-1)
            flat_tokens[topk_idx] = pred_all[:, st : st + t_len].reshape(-1)[topk_idx]
            new_tokens = flat_tokens.view(C, t_len)
            tokens_flat[:, st : st + t_len] = new_tokens

            packed_ids[0, :, c_off + c_len - t_len : c_off + c_len] = new_tokens
            packed_ids[0, :, u_off : u_off + t_len] = new_tokens

    return [tokens_flat[:, st : st + t_len] for (st, t_len) in flat_spans]


def _forward_logits(model, input_ids, audio_mask, position_ids, tgt_index):
    """LLM forward + audio head over target positions only.

    The scoring step consumes logits at the cond-target and uncond ranges
    (2*sum(t_len) of the packed positions); running the 1024->8200 audio_heads
    GEMM and the fp32 upcast on the full packed length is wasted work.
    Returns logits of shape (1, C, 2*sum(t_len), V).
    """
    inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
    hidden = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask={"full_attention": None},  # skips HF mask construction
        position_ids=position_ids,
        use_cache=False,  # bidirectional decoding recomputes everything each step
        return_dict=True,
    )[0]
    tgt_hidden = hidden[0, tgt_index]  # (2T, hidden)
    logits_flat = model.audio_heads(tgt_hidden)
    n = tgt_hidden.shape[0]
    return logits_flat.view(
        1, n, model.config.num_audio_codebook, model.config.audio_vocab_size
    ).permute(0, 2, 1, 3)


_NO_ARCH = (
    "FlashInfer found no CUDA arch it can build or load for this GPU (compute "
    "capability {}.{}). flashinfer 0.6.15 only accepts SM 12.x when the CUDA it "
    "detects (nvcc from CUDA_HOME, PATH or /usr/local/cuda, else torch.version.cuda) "
    "is >= 12.9. Start Python with FLASHINFER_CUDA_ARCH_LIST=12.0a "
    "(flashinfer-jit-cache +cu128) or 12.0f (+cu129). omnivoice_tts never sets "
    "environment variables."
)


@torch.no_grad()
def apply_flashinfer(model):
    """Patch an OmniVoice instance in place to decode with flashinfer kernels.

    Run it after any weight edit (e.g. a LoRA merge) and before int8
    quantization. It may run on a CPU-loaded model that is moved with
    ``model.to("cuda")`` afterwards: the fused weights are then allocated once
    on the GPU. Afterwards the model only runs through ``generate()`` and
    ``generate_streaming()``, one call at a time per instance. Returns ``model``.
    """
    if hasattr(model, "_fi_runner"):
        return model
    llm = model.llm
    if not torch.cuda.is_available():
        raise ValueError("apply_flashinfer needs CUDA")
    if llm.dtype not in (torch.float16, torch.bfloat16):  # the weights' dtype: the config may still say float32
        raise ValueError(f"apply_flashinfer needs fp16 or bf16 weights, got {llm.dtype}")
    if not isinstance(llm.layers[0].self_attn.q_proj, nn.Linear):
        raise TypeError("apply_flashinfer: q_proj is not an nn.Linear; call it before apply_int8")
    from flashinfer.jit import current_compilation_context

    if not current_compilation_context.TARGET_CUDA_ARCHS:
        raise RuntimeError(_NO_ARCH.format(*torch.cuda.get_device_capability()))

    cfg = model.config.llm_config
    model._fi_runner = PackedAttnRunner(
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, llm.dtype
    )
    _patch_rmsnorm(llm)
    _patch_attention_forward(llm, model._fi_runner)
    _patch_mlp(llm)
    model._generate_iterative = MethodType(_generate_iterative_packed, model)
    return model
