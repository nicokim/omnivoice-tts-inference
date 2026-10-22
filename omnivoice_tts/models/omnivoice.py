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

"""Core OmniVoice model implementation, trimmed to inference.

Defines the ``OmniVoice`` model class, its config, the generation config and
the inference pipeline: ``OmniVoice.from_pretrained()`` loads the model, then
``model.generate()`` synthesizes one text, in voice design mode
(``instruct``) or with an auto voice.

Upstream's training (the loss and packed attention), voice cloning and ASR
are removed.
"""

import logging
import math
import os
import re
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm
from safetensors import safe_open
from transformers import (
    AutoTokenizer,
    HiggsAudioV2TokenizerConfig,
    HiggsAudioV2TokenizerModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING, MODEL_MAPPING, AutoConfig

from omnivoice_tts.utils.audio import (
    cross_fade_chunks,
    fade_and_pad_audio,
    remove_silence,
)
from omnivoice_tts.utils.duration import RuleDurationEstimator
from omnivoice_tts.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID, LANG_NAMES
from omnivoice_tts.utils.text import chunk_text_punctuation
from omnivoice_tts.utils.voice_design import (
    _INSTRUCT_EN_TO_ZH,
    _INSTRUCT_ZH_TO_EN,
    _ZH_RE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OmniVoiceGenerationConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1


@dataclass
class GenerationTask:
    batch_size: int
    texts: list[str]
    target_lens: list[int]
    langs: list[str | None]
    instructs: list[str | None]
    ref_texts: list[str | None]
    ref_audio_tokens: list[torch.Tensor | None]
    speed: list[float] | None = None


@dataclass
class OmniVoiceModelOutput(ModelOutput):
    logits: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Config & Model
# ---------------------------------------------------------------------------


class OmniVoiceConfig(PretrainedConfig):
    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        llm_config: dict | PretrainedConfig | None = None,
        **kwargs,
    ):
        # training-only keys in config.json (audio_codebook_weights) still load as attributes, through **kwargs
        if isinstance(llm_config, dict):
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)

        self.llm_config = llm_config

        super().__init__(**kwargs)
        self.audio_vocab_size = audio_vocab_size
        self.audio_mask_id = audio_mask_id
        self.num_audio_codebook = num_audio_codebook


_MODEL_FILES = ["*.json", "*.txt", "model.safetensors", "audio_tokenizer/*"]
# the codec k2-fsa ships in audio_tokenizer/: the fallback for repos without one
_CODEC_REPO, _CODEC_REVISION = "eustlb/higgs-audio-v2-tokenizer", "528e871c2a26c4f0f7773b9754e2e1acae20899d"
_CODEC_ENCODER = ("acoustic_encoder", "encoder_semantic", "decoder_semantic", "semantic_model", "fc", "fc1")
_CODEC_DECODER = ("quantizer.", "fc2.", "acoustic_decoder.")  # all that HiggsAudioV2TokenizerModel.decode() uses


class _NoBar(hf_tqdm):
    """No progress bar for this download only (HF_HUB_DISABLE_PROGRESS_BARS would be process-wide)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "disable": True})


def _resolve_model_path(name_or_path: str, allow_patterns: list[str], revision: str | None = None, **hub) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    return snapshot_download(name_or_path, revision=revision, allow_patterns=allow_patterns, tqdm_class=_NoBar, **hub)


def _load_safetensors(
    module: nn.Module,
    path: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
    prefixes: tuple[str, ...] | None = None,
) -> None:
    """Strict assign-load into a meta-built module; only floating checkpoint tensors are cast to ``dtype``."""
    sd = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if prefixes is None or k.startswith(prefixes):
                t = f.get_tensor(k)
                sd[k] = t.to(device, dtype if dtype is not None and t.is_floating_point() else t.dtype)
    module.load_state_dict(sd, strict=True, assign=True)


def _load_decode_only_codec(path: str, device: torch.device) -> HiggsAudioV2TokenizerModel:
    """The Higgs codec without its encoders, in the checkpoint's dtype; ``encode()`` is not available."""
    config = HiggsAudioV2TokenizerConfig.from_pretrained(path)
    with torch.device("meta"):
        codec = HiggsAudioV2TokenizerModel(config)  # __init__ keeps _adjust_dac_decoder (output_padding, no tanh)
    for name in _CODEC_ENCODER:
        delattr(codec, name)
    _load_safetensors(codec, os.path.join(path, "model.safetensors"), device, prefixes=_CODEC_DECODER)
    return codec.eval()


class OmniVoice(PreTrainedModel):
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(self, config: OmniVoiceConfig):
        super().__init__(config)

        # the model class itself: AutoModel.from_config writes the process default dtype while it builds
        self.llm = MODEL_MAPPING[type(config.llm_config)](config.llm_config)

        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            self.config.llm_config.hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )

        self.audio_heads = nn.Linear(
            self.config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        self.post_init()

        # Inference-only attributes, set by from_pretrained; generate() refuses to run while they are None
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.duration_estimator = None
        self.sampling_rate = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        device_map: str | int | torch.device | None = None,
        dtype: torch.dtype | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> "OmniVoice":
        """Load the model, its text tokenizer and a decode-only audio codec, in eval mode.

        Everything after the path is keyword-only, and an unknown keyword (upstream's ``train``,
        ``load_asr``, ``torch_dtype``, ...) raises ``TypeError``.

        Args:
            pretrained_model_name_or_path: A local directory, used as is, or a Hub repo id.
            device_map: One device (str, int or ``torch.device``); ``None`` means CPU. ``"auto"``,
                ``"balanced"``, ``"balanced_low_0"``, ``"sequential"`` and dicts raise ``ValueError``.
                The codec goes to CPU when the model is on MPS.
            dtype: ``None`` means the config's dtype (and the checkpoint's own dtypes when the config
                names none). RoPE's ``inv_freq`` stays fp32 and the codec keeps its checkpoint dtype.
            revision, token, cache_dir, local_files_only: Passed to ``snapshot_download``, also for the
                codec fallback (``eustlb/higgs-audio-v2-tokenizer`` at a pinned revision) that a repo
                without ``audio_tokenizer/`` uses.

        The checkpoint is a single ``model.safetensors`` (sharded ones are not supported); its keys must
        match the model exactly.

        Never call ``.half()`` or ``.to(dtype)`` after loading: they also cast the fp32 RoPE buffers and
        the codec. Pass ``dtype=`` instead.
        """
        if isinstance(device_map, dict) or device_map in ("auto", "balanced", "balanced_low_0", "sequential"):
            raise ValueError(f"device_map must be one device (str, int or torch.device), got {device_map!r}")
        device = torch.device("cpu" if device_map is None else device_map)
        hub = dict(token=token, cache_dir=cache_dir, local_files_only=local_files_only)
        path = _resolve_model_path(pretrained_model_name_or_path, _MODEL_FILES, revision, **hub)
        config = OmniVoiceConfig.from_pretrained(path)
        with torch.device("meta"):  # no allocation, no random init, no RNG draw
            model = cls(config)
        _load_safetensors(
            model, os.path.join(path, "model.safetensors"), device, dtype if dtype is not None else config.dtype
        )
        # inv_freq is a non-persistent fp32 buffer, not in the checkpoint: rebuilt on CPU, then moved, as
        # transformers' from_pretrained does
        model.llm.rotary_emb = type(model.llm.rotary_emb)(model.llm.config).to(device)
        model.text_tokenizer = AutoTokenizer.from_pretrained(path, config=config)  # config=: silent, no Auto registry
        codec_path = os.path.join(path, "audio_tokenizer")
        if not os.path.isdir(codec_path):
            codec_path = _resolve_model_path(_CODEC_REPO, ["config.json", "model.safetensors"], _CODEC_REVISION, **hub)
        # higgs-audio-v2-tokenizer does not support MPS (output channels > 65536)
        model.audio_tokenizer = _load_decode_only_codec(
            codec_path, torch.device("cpu") if device.type == "mps" else device
        )
        if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
            raise RuntimeError("tensors left on the meta device after loading (unsupported transformers version?)")
        model.sampling_rate = model.audio_tokenizer.config.sample_rate
        model.duration_estimator = RuleDurationEstimator()
        return model.eval()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def _prepare_embed_inputs(
        self, input_ids: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepares embeddings from input_ids of shape (batch_size, layers, seq_length).
        Embedding shape is (batch_size, seq_length, hidden_size).
        """
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])

        # Apply shift to audio IDs based on codebook layer
        # audio_ids: [Batch, 8, Seq]
        # codebook_layer_offsets: [1, 8, 1]
        # Result: Layer 0 ID Layer 1 ID + Layer 2 ID + 2050...
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1)
        ) + self.codebook_layer_offsets.view(1, -1, 1)

        # input: [Batch, 8, Seq] -> output: [Batch, Seq, Hidden]
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> OmniVoiceModelOutput:
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)

        # use_cache=False: no KV cache is ever reused, so building one per step is wasted work
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = llm_outputs[0]

        # Shape: [B, S, C * Vocab]
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)
        # Shape: [B, S, C, Vocab] -> [B, C, S, Vocab]
        audio_logits = logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)

        return OmniVoiceModelOutput(logits=audio_logits)

    def supported_language_ids(self) -> set[str]:
        """Return a list of supported language IDs."""
        return LANG_IDS

    def supported_language_names(self) -> set[str]:
        """Return a list of supported language names."""
        return LANG_NAMES

    # -------------------------------------------------------------------
    # Inference API
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        language: str | None = None,
        *,
        instruct: str | None = None,
        duration: float | None = None,
        speed: float | None = None,
        generation_config: OmniVoiceGenerationConfig | None = None,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Generate speech audio for one text.

        Supports two modes:

        1. **Voice design** — provide ``instruct`` text describing
           the desired voice style.
        2. **Auto** — omit it; the model picks a voice itself.

        Everything after ``language`` is keyword-only.

        Args:
            text: Target text, a single ``str`` (a list raises ``TypeError``).
            language: Language name (e.g. ``"English"``) or code
                (e.g. ``"en"``). ``None`` for language-agnostic mode.
                Performance is slightly better if you specify the language.
            instruct: Style instruction for voice design mode.
            duration: Fixed output duration in seconds. ``None`` (default)
                lets the model estimate duration from text.
                Overrides ``speed`` when both are provided.
            speed: Speaking speed factor. ``> 1.0`` for faster, ``< 1.0`` for
                slower. ``None`` (default) uses the model's default estimation.
            generation_config: Base config, ``OmniVoiceGenerationConfig()`` by
                default. It is not modified: ``**kwargs`` override its fields
                on a copy.
            generator: A ``torch.Generator`` on the model's device; ``None`` uses the global RNG.
            **kwargs: Fields of ``OmniVoiceGenerationConfig``; an unknown name
                raises ``TypeError``:
                denoise: Whether to prepend the ``<|denoise|>`` token.
                num_step: Number of iterative decoding steps.
                guidance_scale: Classifier-free guidance scale.
                t_shift: Time-step shift (smaller → emphasise low-SNR).
                postprocess_output: Shorten pauses longer than 1 s to 1 s and
                    trim leading and trailing silence to 100 ms. Peak
                    normalization to 0.5 and the fade/pad always run.
                layer_penalty_factor: Penalty encouraging earlier codebook
                    layers to unmask first.
                position_temperature: Temperature for position selection.
                class_temperature: Temperature for token sampling (0 = greedy).
                audio_chunk_duration: If > 0, split long text into chunks of
                    this duration (seconds) and generate chunk by chunk.
                audio_chunk_threshold: Only apply chunking if estimated audio
                    duration exceeds this threshold (seconds).
                pad_duration: Silence padding duration per side in seconds
                    (0 to disable).
                fade_duration: Fade-in/out curve duration in seconds
                    (0 to disable).
        Returns:
            A list with one 1-D float32 ``np.ndarray`` of shape ``(T,)`` at
            ``model.sampling_rate`` (usually 24 000 Hz). Save it e.g. with
            soundfile (not a dependency):
            ``soundfile.write("out.wav", audios[0], model.sampling_rate)``.
        """

        if self.audio_tokenizer is None or self.text_tokenizer is None:
            raise RuntimeError(
                "Model is not loaded with audio/text tokenizers. Make sure you "
                "loaded the model with OmniVoice.from_pretrained()."
            )
        gen_config = replace(generation_config or OmniVoiceGenerationConfig(), **kwargs)

        self.eval()

        task = self._preprocess_all(text, language, instruct, duration, speed)

        if gen_config.audio_chunk_duration > 0 and task.target_lens[0] > int(
            gen_config.audio_chunk_threshold * self.audio_tokenizer.config.frame_rate
        ):
            tokens = self._generate_chunked(task, gen_config, generator)
        else:
            tokens = self._generate_iterative(task, gen_config, generator)[0]

        return [self._decode_and_post_process(tokens, gen_config)]

    def _decode_and_post_process(
        self,
        tokens: torch.Tensor | list[torch.Tensor],
        gen_config: OmniVoiceGenerationConfig,
    ) -> np.ndarray:
        """
        Args:
            tokens: Audio tokens — either a single tensor of shape
                (num_codebooks, seq_len) or a list of chunk tensors.
            gen_config: Generation config for post-processing options.
        Returns:
            Decoded and post-processed audio array of shape (T,).
        """
        tokenizer_device = self.audio_tokenizer.device
        if isinstance(tokens, list):
            chunk_audios = [
                self.audio_tokenizer.decode(t.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .cpu()
                .numpy()
                for t in tokens
            ]
            audio_waveform = cross_fade_chunks(chunk_audios, self.sampling_rate)
        else:
            audio_waveform = (
                self.audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .cpu()
                .numpy()
            )

        audio_waveform = self._post_process_audio(audio_waveform, gen_config)
        return audio_waveform.squeeze(0)

    def _post_process_audio(
        self,
        generated_audio: np.ndarray,
        gen_config: OmniVoiceGenerationConfig,
    ) -> np.ndarray:
        """Optionally remove long silences, adjust volume, and add edge padding.

        Args:
            generated_audio: Numpy array of shape (1, T).
            gen_config: Generation config controlling post-processing behaviour.
        Returns:
            Processed numpy array of shape (1, T).
        """
        if gen_config.postprocess_output:
            generated_audio = remove_silence(
                generated_audio,
                self.sampling_rate,
                mid_sil=500,
                lead_sil=100,
                trail_sil=100,
            )

        # initial=0.0: remove_silence returns an empty clip when everything is below -50 dBFS, and max() of
        # nothing raises (upstream crashed here)
        peak = np.abs(generated_audio).max(initial=0.0)
        if peak > 1e-6:
            generated_audio = generated_audio / peak * 0.5

        generated_audio = fade_and_pad_audio(
            generated_audio,
            pad_duration=gen_config.pad_duration,
            fade_duration=gen_config.fade_duration,
            sample_rate=self.sampling_rate,
        )
        return generated_audio

    def _generate_chunked(
        self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig, generator: torch.Generator | None = None
    ) -> list[torch.Tensor]:
        """Generate long audio by splitting the text into chunks.

        Chunk 0 is generated without a reference; every later chunk is
        prompted with chunk 0's tokens and text.

        Args:
            task: A one-item :class:`GenerationTask` whose estimated audio
                exceeds ``audio_chunk_threshold``.
            gen_config: Generation config (``audio_chunk_duration`` controls
                chunk size).
            generator: Shared by every chunk, so the chunks draw one after
                another from the same stream (``None``: the global RNG).
        Returns:
            The audio token tensors, one per text chunk.
        """
        avg_tokens_per_char = task.target_lens[0] / len(task.texts[0])
        text_chunk_len = int(
            gen_config.audio_chunk_duration
            * self.audio_tokenizer.config.frame_rate
            / avg_tokens_per_char
        )
        chunks = chunk_text_punctuation(
            text=task.texts[0],
            chunk_len=text_chunk_len,
            min_chunk_len=3,
        )
        logger.debug("Text chunked into %d pieces: %s", len(chunks), chunks)

        speed = task.speed[0] if task.speed else 1.0
        tokens = []
        for chunk in chunks:
            ref_text, ref_tokens = (chunks[0], tokens[0]) if tokens else (None, None)
            sub_task = GenerationTask(
                batch_size=1,
                texts=[chunk],
                # chunk 0's length already carries the speed, and later chunks are timed from its rate, so they
                # use 1.0 (upstream divided by the speed a second time)
                target_lens=[
                    self._estimate_target_tokens(
                        chunk,
                        ref_text,
                        ref_tokens.size(-1) if ref_tokens is not None else None,
                        speed=1.0 if tokens else speed,
                    )
                ],
                langs=task.langs,
                instructs=task.instructs,
                ref_texts=[ref_text],
                ref_audio_tokens=[ref_tokens],
                speed=task.speed,
            )
            tokens.append(self._generate_iterative(sub_task, gen_config, generator)[0])
        return tokens

    def _preprocess_all(
        self,
        text: str,
        language: str | None = None,
        instruct: str | None = None,
        duration: float | None = None,
        speed: float | None = None,
    ) -> GenerationTask:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not any(c.isalnum() for c in text):
            raise ValueError(f"text has nothing to speak: {text!r}")
        language = _resolve_language(language)
        instruct = None if instruct is None else _resolve_instruct(instruct, use_zh=bool(_ZH_RE.search(text)))

        # duration overrides speed for estimation: use speed=1.0 to get the raw
        # estimate, then override the target length below.
        est = self._estimate_target_tokens(
            text, None, None, speed=1.0 if duration is not None else (float(speed) if speed is not None else 1.0)
        )
        if duration is not None:
            # Exact frame count, and a speed ratio so chunked generation
            # scales proportionally.
            target = max(1, int(float(duration) * self.audio_tokenizer.config.frame_rate))
            speed_list = [est / target]
            est = target
        else:
            speed_list = [float(speed)] if speed is not None else None

        return GenerationTask(1, [text], [est], [language], [instruct], [None], [None], speed_list)

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        """Estimate number of target audio tokens."""
        if num_ref_audio_tokens is None or ref_text is None or len(ref_text) == 0:
            # Fall back to a simple heuristic
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = 25

        est = self.duration_estimator.estimate_duration(
            text, ref_text, num_ref_audio_tokens
        )
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def _prepare_inference_inputs(
        self,
        text: str,
        num_target_tokens: int,
        ref_text: str | None = None,
        ref_audio_tokens: torch.Tensor | None = None,
        lang: str | None = None,
        instruct: str | None = None,
        denoise: bool = True,
    ):
        """Prepare input_ids and audio masks for inference.
        Args:
            text: Target text to generate.
            num_target_tokens: Number of audio tokens to generate.
            ref_text: Optional reference text for voice cloning.
            ref_audio_tokens: Optional reference audio tokens for voice cloning.
                with shape (C, T).
            lang: Optional language ID.
            instruct: Optional style instruction for voice design.
            denoise: Whether to include the <|denoise|> token.
        """

        # Build style tokens: <|denoise|> + <|lang_start|>...<|lang_end|>
        #                      + <|instruct_start|>...<|instruct_end|>
        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        lang_str = lang if lang else "None"
        instruct_str = instruct if instruct else "None"
        style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

        style_tokens = (
            self.text_tokenizer(style_text, return_tensors="pt")
            .input_ids.repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(self.device)  # [1, C, N1]

        # Build text tokens
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_tokens = (
            _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer)
            .repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(self.device)  # [1, C, N2]

        # Target: all MASK
        target_audio_tokens = torch.full(
            (1, self.config.num_audio_codebook, num_target_tokens),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        # Conditional input
        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
        parts.append(target_audio_tokens)
        cond_input_ids = torch.cat(parts, dim=2)

        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)

        cond_audio_mask = torch.zeros(
            1, cond_total_length, dtype=torch.bool, device=self.device
        )
        cond_audio_mask[0, cond_audio_start_idx:] = True

        return {
            "input_ids": cond_input_ids,
            "audio_mask": cond_audio_mask,
        }

    def _generate_iterative(
        self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig, generator: torch.Generator | None = None
    ) -> list[torch.Tensor]:
        """N-step iterative unmasked decoding.

        Args:
            task: A :class:`GenerationTask` containing batch texts, target
                lengths, languages, instructions, and optional reference data.
            gen_config: A :class:`OmniVoiceGenerationConfig` controlling
                decoding steps, guidance, temperatures, etc.
            generator: Source of every random draw (token sampling and
                position selection); ``None`` uses the global RNG.
        Returns:
            List of generated audio token tensors of shape (C, T) (one per
            input text).
        """

        B = task.batch_size

        for i in range(B):
            logger.debug(
                "Item %d — text: %s | ref_text: %s | instruct: %s | lang: %s | target_tokens: %d",
                i,
                task.texts[i],
                task.ref_texts[i],
                task.instructs[i],
                task.langs[i],
                task.target_lens[i],
            )

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
        max_c_len = max(c_lens)
        pad_id = self.config.audio_mask_id  # Or any other tokens

        batch_input_ids = torch.full(
            (2 * B, self.config.num_audio_codebook, max_c_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        batch_audio_mask = torch.zeros(
            (2 * B, max_c_len), dtype=torch.bool, device=self.device
        )
        batch_attention_mask = torch.zeros(
            (2 * B, 1, max_c_len, max_c_len), dtype=torch.bool, device=self.device
        )

        for i, inp in enumerate(inputs_list):
            c_len, u_len = c_lens[i], task.target_lens[i]

            # Cond (0 ~ B-1)
            batch_input_ids[i, :, :c_len] = inp["input_ids"]
            batch_audio_mask[i, :c_len] = inp["audio_mask"]
            batch_attention_mask[i, :, :c_len, :c_len] = True

            # Uncond (B ~ 2B-1)
            batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]
            batch_audio_mask[B + i, :u_len] = inp["audio_mask"][..., -u_len:]
            batch_attention_mask[B + i, :, :u_len, :u_len] = True
            if max_c_len > u_len:
                pad_diag = torch.arange(u_len, max_c_len, device=self.device)
                batch_attention_mask[B + i, :, pad_diag, pad_diag] = True

        tokens = torch.full(
            (B, self.config.num_audio_codebook, max(task.target_lens)),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=gen_config.num_step,
            t_shift=gen_config.t_shift,
        ).tolist()
        schedules = []
        for t_len in task.target_lens:
            total_mask = t_len * self.config.num_audio_codebook
            rem = total_mask
            sched = []
            for step in range(gen_config.num_step):
                num = (
                    rem
                    if step == gen_config.num_step - 1
                    else min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        rem,
                    )
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)

        layer_ids = torch.arange(
            self.config.num_audio_codebook, device=self.device
        ).view(1, -1, 1)

        for step in range(gen_config.num_step):
            batch_logits = self(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)

            for i in range(B):
                k = schedules[i][step]
                if k <= 0:
                    continue

                c_len, t_len = c_lens[i], task.target_lens[i]

                # Extract real target Logits
                # [1, C, T, V]
                c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :]
                u_logits = batch_logits[B + i : B + i + 1, :, :t_len, :]

                pred_tokens, scores = self._predict_tokens_with_scoring(
                    c_logits, u_logits, gen_config, generator
                )

                scores = scores - (layer_ids * gen_config.layer_penalty_factor)

                if gen_config.position_temperature > 0.0:
                    scores = _gumbel_sample(scores, gen_config.position_temperature, generator)

                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(
                    sample_tokens != self.config.audio_mask_id, -float("inf")
                )

                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                # Update individual slices into batched structure
                tokens[i : i + 1, :, :t_len] = sample_tokens
                batch_input_ids[i : i + 1, :, c_len - t_len : c_len] = sample_tokens
                batch_input_ids[B + i : B + i + 1, :, :t_len] = sample_tokens

        return [tokens[i, :, : task.target_lens[i]] for i in range(B)]

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config, generator=None):
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)

        log_probs[..., self.config.audio_mask_id] = -float("inf")

        if gen_config.class_temperature > 0.0:
            filtered_probs = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(
                filtered_probs, gen_config.class_temperature, generator
            ).argmax(dim=-1)
        else:
            pred_tokens = log_probs.argmax(dim=-1)

        confidence_scores = log_probs.max(dim=-1)[0]

        return pred_tokens, confidence_scores


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _resolve_language(language: str | None) -> str | None:
    """Return the language ID for an ID or a name, or ``None`` for language-agnostic mode.

    ``None`` and ``"none"`` give ``None``. Otherwise IDs, then names, are matched case-insensitively and
    without stripping whitespace. Anything else raises ``ValueError`` (upstream warned and used ``None``).
    """
    if language is None or language.lower() == "none":
        return None
    key = language.lower()
    if key in LANG_IDS:
        return key
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]
    raise ValueError(
        f"unknown language {language!r}: use a language ID (e.g. 'es', 'en', 'zh') from supported_language_ids() "
        "or a language name (e.g. 'Spanish') from supported_language_names()"
    )


def _resolve_instruct(
    instruct: str | None, use_zh: bool = False
) -> str | None:
    """Normalise a voice-design instruct string; nothing is rejected.

    The string is split on half- and full-width commas, and each item is
    stripped and lowercased. Items of upstream's bilingual vocabulary are
    unified to one language: a Chinese dialect (``...话``) forces Chinese, an
    English accent (``... accent``) forces English, otherwise ``use_zh``
    decides. Any other item passes through unchanged. Items are joined with a
    full-width comma when any of them is Chinese, else with ``", "``.

    Args:
        instruct: Raw instruct string, or ``None``.
        use_zh: If True, normalise items to Chinese (used when the
            synthesis text contains Chinese and no accent is specified).

    Returns:
        Normalised instruct string, or ``None`` if it is empty.
    """
    if instruct is None or not instruct.strip():
        return None
    items = [x.strip().lower() for x in re.split(r"\s*[,，]\s*", instruct.strip()) if x]
    if any(n.endswith("话") for n in items):  # a dialect forces Chinese
        use_zh = True
    elif any(" accent" in n for n in items):  # an accent forces English
        use_zh = False
    mapping = _INSTRUCT_EN_TO_ZH if use_zh else _INSTRUCT_ZH_TO_EN
    items = [mapping.get(n, n) for n in items]
    return ("，" if any(any("一" <= c <= "鿿" for c in n) for n in items) else ", ").join(items)


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def _gumbel_sample(logits: torch.Tensor, temperature: float, generator: torch.Generator | None = None) -> torch.Tensor:
    scaled_logits = logits / temperature
    # rand_like is empty_like + uniform_ (same layout, same stream); this form only adds the generator
    u = torch.empty_like(scaled_logits).uniform_(generator=generator)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs tensor of shape (1, seq_len).
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)

    if not parts:
        result = tokenizer(text, return_tensors="pt").input_ids
    else:
        combined = []
        for p in parts:
            combined.extend(p)
        result = torch.tensor([combined], dtype=torch.long)
    return result


def _combine_text(text, ref_text: str | None = None) -> str:
    # combine with reference text if not None
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()

    # a run of newline / carriage-return characters becomes one space (upstream deleted it, so "a.\nb" read "a.b");
    # the rules below collapse the spaces and drop any left between Chinese characters
    full_text = re.sub(r"[\r\n]+", " ", full_text)

    # replace Chinese parentheses with English ones
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")

    # collapse consecutive spaces / tabs into a single space
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # remove spaces around chinese characters
    chinese_range = r"[\u4e00-\u9fff]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    return full_text
