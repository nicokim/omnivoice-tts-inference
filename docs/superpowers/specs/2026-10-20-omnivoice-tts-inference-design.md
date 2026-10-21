# omnivoice-tts-inference v0.1 design

- Date: 2026-10-20
- Status: the user approved the design in six parts on 2026-10-20, and bench2's GPU results were folded in on
  2026-10-21. The user approved this written spec on 2026-10-21; the implementation plan is
  `docs/superpowers/plans/2026-10-21-omnivoice-tts-inference-v0.1.md`.
- Upstream: k2-fsa/OmniVoice @ 08be0b4, which is master HEAD on 2026-10-20 and identical in core to PyPI 0.2.1.
- Import name `omnivoice_tts`; distribution `omnivoice-tts-inference`.
- Line references:
  - `M:` is upstream `omnivoice/models/omnivoice.py`, `T:` is upstream `omnivoice/utils/text.py` and `D:` is
    upstream `omnivoice/utils/duration.py`, all @ 08be0b4.
  - `R:` is `indextts25/utils/es_rules.py` in nicokim/indextts25-inference at dfb3286f, the head of PR #6.
- Evidence directory (outside the repo): the design session's records in
  `~/.claude/projects/-home-kimox-orca-workspaces-omnivoice-tts-inference-feat-initial-implementation/afd1a81d-ce0d-4c3f-b201-4690f4fd694d/`,
  with `stream/` (part 3), `tn/` (part 4), `rtf/` (GPU measurements, 2026-10-20) and `bench2/` (int8, FlashInfer +
  int8, quality, TN audio, LLM contention and soak, 2026-10-21; summary in `bench2/FINAL.md`). bench2 runs upstream
  code with the library's changes emulated ("lib-like") in four configs: A is sdpa fp16, B sdpa int8, C FlashInfer
  fp16 and D FlashInfer + int8.

## 1. Goal, scope, success criteria

**Goal.** Shrink upstream OmniVoice into an inference-only library that keeps OmniVoice's own API trimmed. The first
consumer is the KohAI voice (`kohai-channel/kohai-omni-tts`) inside kohai-v3. This is not a port of
indextts25-inference's API: there is no facade, no `synthesize`/`warm`/`Chunk`.

**In v0.1**
- Any OmniVoice checkpoint, without voice cloning.
- `generate()`.
- `generate_streaming()`. It works at sentence level and needs no framework. It yields each sentence's audio together
  with the raw text that audio speaks (§7).
- Spanish text normalization, as a step the caller runs: `TextNormalizer` (§8).
- Optional FlashInfer (§9) and int8 (§10).

**Out of v0.1**
- Voice cloning: no reference audio, so the codec is decode-only.
- Batch input (lists).
- Streaming input as LLM deltas (`Iterable[str]`), conditioning a sentence on earlier ones, a public splitter.
- Text normalization for languages other than Spanish.
- ASR, training, the CLI, LoRA utilities.
- Model-specific settings in code. KohAI's emotions, duration rates and tags are caller config; the README only
  recommends them.

**Success criteria**
1. With defaults (sdpa, fp16 weights, global RNG), `generate(text)[0]` is bit-identical (int16 PCM sha256) to upstream
   `generate(text)[0]` for any single speakable text without CR/LF, short or chunked. `generate_streaming(text)`
   yields that same array when `text` has no sentence break (§7).
2. Library code prints nothing to stdout/stderr and leaves no lasting change to process-global state (§11). The one
   exception is documented: two `TextNormalizer` instances normalizing at the same time can leave NeMo's logger at
   INFO (§8; no module-level lock, T3-A).
3. Every upstream bug fixed in place has a CPU regression test named after it (§16), except fix 6, which only
   corrects a docstring.
4. The package shrinks from 15,088 to about 3.0k Python lines, not counting the 1,177 lines of copied Spanish rules
   (§3).

## 2. Decision log

| # | Decision | By |
|---|---|---|
| 1 | Generic library, no cloning, no batch; FlashInfer, int8 and streaming are in | user, 2026-10-20 |
| 2 | Keep upstream's API shape trimmed; do not copy indextts25-inference's contract | user |
| 3 | No validation of model config (instruct passes through, normalized like upstream) | user |
| 4 | Text normalization is in (native digit fine-tune failed) | user |
| 5 | Part 1 approved: shrink map and API (str-only `generate`, keyword-only after `language`, `generator=`, `generate_streaming`, `apply_flashinfer`, `apply_int8`) | user |
| 6 | Part 2 approved: own loader; `dtype=None` keeps upstream/transformers semantics (config dtype). KohAI Hub config fixed to float16 (commit 696ac256) | user |
| 7 | Streaming: the library segments any text itself, following best practice, independent of Pipecat (§7) | user delegated, lead decided |
| 8 | Part 5 approved: vendored FlashInfer without CUDA graphs or global state; int8; FlashInfer then int8 allowed (experimental) | user |
| 9 | Part 6 approved: repo infra like indextts25-inference; licenses; tests; GPU check list. KohAI card relabelled cc-by-nc-4.0 (commit 499a9c55) | user |
| 10 | Unknown language raises ValueError. The double-speed fix for chunks ≥ 1 is in (fix 10); G13 checks it by ear before the release | lead, approved with part 3 |
| 11 | Keep pydub (bit-identity); document its warnings | lead, approved with part 2 |
| 12 | Part 3 approved: sentence-level `generate_streaming` with `normalize=` and a callable `duration=`, yielding `(audio, raw sentence)`; independent sentences; `str` input only. The user listens to the stream before the release (§19); it passed (decision #24) | user |
| 13 | Part 4 approved: caller-held `TextNormalizer`; Spanish only (NeMo plus the copied PR #6 rules); no `normalize_text=` flag; opt-in guarded grammar cache; errors propagate; the stream skips sentences that normalization empties | user |
| 14 | `use_cache=False` in the sdpa forward is adopted: G10 measured PCM-identical output (36/36) and 6.7 % lower RTF on upstream code | rule from part 6 ("if identical, adopt"), measured 2026-10-20 |
| 15 | CI's test job installs the `[tn]` extra, so the PR #6 suite and the TN regression tests run on every PR (§13) | lead, from success criterion 3 |
| 16 | ruff as in indextts25-inference; the kept upstream code is converted once to PEP 604/585 annotations (§13) | lead, from part 6 (repo infra like indextts25-inference) |
| 17 | The es_rules copy waits for PR #6's merge, with the user's license header, so TN is the last implementation task (§8) | lead, from T6-A |
| 18 | int8's forward stays `F.linear(x, self.qweight * self.scale, self.bias)`. It is bit-identical to slim.py's `self.q.to(x.dtype) * self.scale` and costs +8 % instead of +15 % RTF on sdpa, and +31-34 % instead of +42-46 % on FlashInfer | lead, from bench2 (2026-10-21) |
| 19 | int8 and FlashInfer + int8 stay in v0.1, experimental until the blind listening passes. The README says when they pay and what they cost; kohai-v3 runs FlashInfer fp16 | user, 2026-10-21 |
| 20 | The README recommends FlashInfer fp16 for a TTS that shares its GPU with an LLM | lead, from bench2 contention |
| 21 | Neighbors on the same machine are tuned by the caller, never by the library (§11). The README lists faster-whisper in its own process (or `OPENBLAS_NUM_THREADS=1`), a high-priority CUDA stream and `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT` | lead, from bench2 contention and soak |
| 22 | G-TN1's pass rule is the one fixed before scoring. TN must lie in the third of the dígitos → palabras interval nearest palabras, on CER and on ok ≤ 5 %. dur/ref must be within 1.00 ± 0.03, and the CER CI against dígitos must exclude 0. TN passed | lead, from bench2 tn |
| 23 | The blind listening passed, so int8 and FlashInfer + int8 are no longer experimental (decision #19's rule). The user heard no difference in 28 of the 29 pairs answered; the one preference went to sdpa int8 | user, 2026-10-21 (blind listening) |
| 24 | The listening gate passed: the stream, the lone short sentences and G-TN3/4's takes sounded right, so streaming ships as designed, with no `pad_duration=0.05` advice and no minimum-length merge | user, 2026-10-21 |

## 3. Package layout

| Path | Upstream lines | Action | After |
|---|---|---|---|
| `omnivoice_tts/__init__.py` | 34 | Drop the warnings filters and `VoiceClonePrompt`. Static `__version__ = "0.1.0"  # x-release-please-version`. Export `OmniVoice`, `OmniVoiceConfig`, `OmniVoiceGenerationConfig`. | ~12 |
| `omnivoice_tts/models/omnivoice.py` | 1727 | §5, §6, §7 | ~1,000 |
| `omnivoice_tts/models/omnivoice_flashinfer.py` | 654 | §9 | ~400 |
| `omnivoice_tts/models/omnivoice_int8.py` | new | §10 | ~50 |
| `omnivoice_tts/utils/audio.py` | 343 | Keep the pydub silence helpers, `fade_and_pad_audio`, `cross_fade_chunks`. Delete loaders, `trim_long_audio`, unused imports. | ~202 |
| `omnivoice_tts/utils/duration.py` | 282 | Delete the `__main__` demo (D:250-282), the `@lru_cache` on `_get_char_weight` (D:165) and its import (D:27). Weights are identical without it on 34,090 sentences. | ~247 |
| `omnivoice_tts/utils/lang_map.py` | 698 | Delete `_TITLE_EXCEPTIONS`, `lang_display_name`. | ~679 (data) |
| `omnivoice_tts/utils/text.py` | 429 | Keep `SPLIT_PUNCTUATION`, `CLOSING_MARKS`, `ABBREVIATIONS`, `chunk_text_punctuation`. Delete `END_PUNCTUATION`, `add_punctuation`. Replace the TN section (T:228-429) per §8. | ~332 |
| `omnivoice_tts/utils/es_rules.py` | new | Byte-identical copy of indextts25-inference's Spanish rules (§8). Never edited here. | 1,177 |
| `omnivoice_tts/utils/voice_design.py` | 97 | Keep `_ZH_RE`, the bilingual dicts and EN↔ZH maps; delete accent/dialect and validation sets. | ~55 |
| `omnivoice_tts/utils/{common,data_utils,lora}.py`, `cli/`, `training/`, `data/`, `scripts/`, `eval/` | | Delete. | 0 |

Repository files:
- Keep LICENSE byte-identical. Each modified file gets the header line
  `Modified by omnivoice-tts-inference from k2-fsa/OmniVoice 08be0b4` (Apache-2.0 §4(b)).
- Trim `docs/generation-parameters.md` and `docs/voice-design.md`; delete the rest of `docs/`, `examples/`,
  `tests/test_lora.py`, upstream `.github/`, `uv.lock`, `.pre-commit-config.yaml`, `CONTRIBUTING.md`.
- Add `tests/test_es_normalization.py` from indextts25-inference PR #6 (§8), and `tests/data/` with the splitter's
  hand cases, the TN tag probes and int8's per-layer hashes (§18).
- Add a repo-root `scripts/benchmark.py` (not part of the package), as in indextts25-inference.
- Add the CI, release-please and release workflows, ruff config and a fresh `uv.lock` (§13).

## 4. Public API

```python
from omnivoice_tts import OmniVoice, OmniVoiceConfig, OmniVoiceGenerationConfig
from omnivoice_tts.utils.text import TextNormalizer

model = OmniVoice.from_pretrained(name_or_path, *, device_map=None, dtype=None, revision=None,
                                  token=None, cache_dir=None, local_files_only=False)
audios = model.generate(text, language=None, *, instruct=None, duration=None, speed=None,
                        generation_config=None, generator=None, **config_fields)   # list[np.ndarray], 1 item
stream = model.generate_streaming(text, language=None, *, instruct=None, duration=None, speed=None,
                                  normalize=None, generation_config=None, generator=None,
                                  **config_fields)                    # Iterator[tuple[np.ndarray, str]], §7
model.sampling_rate                      # 24000, from the codec config
model.duration_estimator                 # RuleDurationEstimator
model.supported_language_ids(); model.supported_language_names()

tn = TextNormalizer(cache_dir=None)      # §8
spoken = tn.normalize(text, language)    # -> str

from omnivoice_tts.models.omnivoice_flashinfer import apply_flashinfer   # apply_flashinfer(model) -> model
from omnivoice_tts.models.omnivoice_int8 import apply_int8               # apply_int8(model) -> model
```

**`OmniVoiceGenerationConfig`** keeps upstream's fields and defaults: `num_step` 32, `guidance_scale` 2.0, `t_shift`
0.1, `layer_penalty_factor` 5.0, `position_temperature` 5.0, `class_temperature` 0.0, `denoise` True,
`postprocess_output` True, `audio_chunk_duration` 15.0, `audio_chunk_threshold` 30.0, `pad_duration` 0.1,
`fade_duration` 0.1. `preprocess_prompt` and `from_dict` are removed.

**Signature changes from upstream**
- `text` is `str` only; a list raises TypeError.
- Everything after `language` is keyword-only. Upstream's third positional argument was `ref_text`, so a stale call
  raises instead of binding to `instruct`.
- `generator: torch.Generator | None`. `None` keeps upstream's global RNG with the same op.
- Config fields passed as keywords override a passed `generation_config` (built with `dataclasses.replace`); an
  unknown field raises TypeError. Upstream dropped both silently.
- `instruct` is normalized like upstream (split, lowercase, EN↔ZH) and never rejected.
- `generate_streaming` is new. Its `duration` is a callable `(spoken text) -> seconds`, and its `normalize` is a
  callable `(sentence, language id) -> spoken text` (§7).
- `OmniVoiceConfig` drops `audio_codebook_weights` (training only; config.json keys still load as attributes) and keeps
  `model_type = "omnivoice"`. There is no Auto registration.

**Removed:** `VoiceClonePrompt`, `create_voice_clone_prompt`, `load_asr_model`, `transcribe`, the
`ref_text`/`ref_audio`/`voice_clone_prompt` parameters, the `train`/`load_asr`/`asr_*` kwargs, `set_input_embeddings`,
the loss, the flex/FA2 flags and `feature_extractor`. Also removed: the `normalize_text` parameter and the module
function `normalize_text()`. TN is a step the caller runs: `TextNormalizer.normalize()` before `generate()`, or
`generate_streaming(normalize=tn.normalize)` (§8).

**Errors**

| Exception | When |
|---|---|
| TypeError | `text` not a str; positional arguments after `language`; unknown config field; unknown `from_pretrained` kwarg; a number as `generate_streaming`'s `duration`; `TextNormalizer.normalize()` without `language`; `apply_flashinfer` after `apply_int8` |
| ValueError | nothing to speak (`not any(c.isalnum() for c in text)`; `'[laughter]'` alone is speakable); unknown language; `device_map` "auto" or a dict; `apply_flashinfer` without CUDA or with fp32 weights |
| RuntimeError | model not built by `from_pretrained`; tensors left on meta; strict-load mismatch; FlashInfer arch check |
| FileNotFoundError | no single `model.safetensors` |
| ImportError | a Spanish `TextNormalizer.normalize()` without nemo-text-processing; importing `omnivoice_flashinfer` without flashinfer. Both messages give the install command |
| OSError | `TextNormalizer` cannot create or write its `cache_dir`; raised before the grammar build |

`generate_streaming` raises nothing at the call except Python's own argument errors (positional arguments after
`language`, a missing `text`). The checks listed in §7 raise at the first `next()`; with non-str text the TypeError
comes from the splitter's `re`. Errors raised by `normalize`, NeMo or generation come out of the `next()` that hits
them.

## 5. Loader (replaces upstream M:270-275 and M:332-384)

1. **Path.** A local directory is used as is. Otherwise `snapshot_download(repo, revision=, allow_patterns=['*.json',
   '*.txt', 'model.safetensors', 'audio_tokenizer/*'], token=, cache_dir=, local_files_only=, tqdm_class=_NoBar)`,
   where `_NoBar` subclasses `huggingface_hub.utils.tqdm` with `disable=True`.
2. **LLM.** `config = OmniVoiceConfig.from_pretrained(path)`; `with torch.device("meta"): model = cls(config)`.
   `__init__(config)` builds the LLM with `MODEL_MAPPING[type(config.llm_config)](config.llm_config)` (not
   `AutoModel.from_config`, which writes the process default dtype). Strict `load_state_dict(sd, assign=True)` from
   `safe_open`; floating tensors are cast to `dtype or config.dtype` and moved to the device.
3. **RoPE.** `model.llm.rotary_emb = type(model.llm.rotary_emb)(model.llm.config).to(device)`: inv_freq is rebuilt in
   fp32 on CPU, as transformers does.
4. **Tokenizer.** `AutoTokenizer.from_pretrained(path, config=config)`; silent without the Auto registry.
5. **Codec, decode-only.**
   - `HiggsAudioV2TokenizerConfig.from_pretrained(codec_path)`, then a meta build.
   - `delattr` acoustic_encoder, encoder_semantic, decoder_semantic, semantic_model, fc and fc1.
   - Strict-load only `quantizer.`, `fc2.` and `acoustic_decoder.` (176 keys, 86.3 MiB fp32, checkpoint dtype).
   - CPU when the model is on MPS. Assigned to `model.audio_tokenizer` (a registered submodule).
   - Without `audio_tokenizer/` in the repo, the fallback is `eustlb/higgs-audio-v2-tokenizer` pinned at
     `528e871c2a26c4f0f7773b9754e2e1acae20899d`, the same blob k2-fsa ships. The KohAI repo has no
     `audio_tokenizer/`, so it always uses the fallback.
6. **Finish.** RuntimeError if anything is left on meta; `model.sampling_rate = codec_config.sample_rate`;
   `model.duration_estimator = RuleDurationEstimator()`; `return model.eval()`.

**Arguments.** Keyword-only after the path. `device_map` is one device (str, int or `torch.device`); `None` means CPU;
`"auto"` or a dict raises ValueError. `dtype=None` means `config.dtype`. Hub kwargs are forwarded. A single `model.safetensors`
only (sharded checkpoints are out of v0.1). The README says never to call `.half()` or `.to(dtype)` after loading.

**Evidence (CPU).** On the KohAI fp16 export, 315/315 tensors are bit-identical to upstream
`from_pretrained(dtype=fp16)`, inv_freq stays fp32, 0 bytes are printed and the global RNG is untouched. The
decode-only codec decodes bit-identically to the full codec.

## 6. Generation (`generate`)

- **`generate()`** keeps upstream's structure, decorated with `@torch.inference_mode()`, in this order:
  1. the tokenizer guard;
  2. `gen_config = replace(generation_config or OmniVoiceGenerationConfig(), **kwargs)`;
  3. `self.eval()`;
  4. `task = self._preprocess_all(text, language, instruct, duration, speed)`;
  5. the dispatch: `_generate_chunked` when `audio_chunk_duration > 0 and task.target_lens[0] >
     int(audio_chunk_threshold * frame_rate)`, otherwise `_generate_iterative(task, gen_config, generator)[0]`;
  6. `return [self._decode_and_post_process(tokens, gen_config)]`.
- **Long text** follows upstream (M:913-1022) for one text, in `_generate_chunked(task, gen_config, generator=None) ->
  list[torch.Tensor]`:
  - `chunk_text_punctuation(min_chunk_len=3)`;
  - chunk 0 is unprompted; later chunks are prompted with chunk 0's tokens and text plus `<|denoise|>`;
  - each chunk's target comes from `_estimate_target_tokens(..., speed=1.0 if tokens else speed)`. That is fix 10:
    chunk 0's rate already carries `speed`, and upstream divided by it again;
  - `_decode_and_post_process` cross-fades the chunk tensors as upstream does and post-processes once.
- **`_preprocess_all(text, language, instruct, duration, speed)`** is a str-only rewrite of M:1024-1167.
  - TypeError/ValueError checks come first. Nothing to speak raises `ValueError(f"text has nothing to speak: {text!r}")`.
  - `use_zh = bool(_ZH_RE.search(text))`.
  - With `duration`, the estimate is computed at speed 1, then `target = max(1, int(duration * frame_rate))` and
    `speed = est / target`.
  - It returns a one-item `GenerationTask` (`ref_rms`, `get_indices` and `slice_task` are dropped).
- **Unchanged and batch-generic:**
  - `_generate_iterative(task, gen_config, generator=None)`, which passes `generator` on;
  - `_predict_tokens_with_scoring(..., generator=None)`, `_prepare_inference_inputs`, `_prepare_embed_inputs`,
    `_estimate_target_tokens`, `_tokenize_with_nonverbal_tags`, `_NONVERBAL_PATTERN`, `_get_time_steps`,
    `_filter_top_k`;
  - the `logger.debug` lines.
- **`_gumbel_sample(logits, temperature, generator=None)`** draws with
  `torch.empty_like(scaled).uniform_(generator=generator)` (rand_like's own implementation).
- **Post-processing.** `_decode_and_post_process(tokens, gen_config)` and `_post_process_audio(audio, gen_config)`
  lose the rms parameter. The peak is `np.abs(audio).max(initial=0.0)`.
- **`forward(input_ids, audio_mask, attention_mask)`:** `attention_mask` is required, and the LLM call passes
  `use_cache=False` (G10: PCM-identical on the 12 bench lines, 262 → 246 ms per call, same peak memory). The loss,
  labels, document_ids, position_ids and the flex block mask are deleted.
- **Language and instruct.** `_resolve_language` takes IDs case-insensitively, which is new: upstream matched them
  case-sensitively, so 'ES' became None with a warning. Names stay case-insensitive as upstream. An unknown value
  raises ValueError. It is identical on 1,941 upstream-accepted inputs. `_resolve_instruct` only normalizes and
  never raises; it is identical on 784 inputs.
- **`_combine_text`:** runs of CR/LF become a space.
- **Deleted:** the flex autocast/registration, packed-mask helpers, `VoiceClonePrompt`, `create_voice_clone_prompt`,
  ASR, `_ensure_list`, the Auto registration, unused imports.
- **RNG.**
  - `generator=None` keeps the global RNG.
  - A `torch.Generator` on the model's device drives every chunk in order.
  - Decoding draws nothing.
  - A CPU generator with a CUDA model raises (torch).
- **Concurrency:** one call at a time per model instance (also with FlashInfer).

## 7. Streaming (`generate_streaming`)

**What it does.**
1. It splits the raw text into sentences with the private `_split_sentences`.
2. For each sentence it runs `normalize`, then `duration`, then `generate()`.
3. It yields `(audio, raw sentence)` as soon as each sentence is ready.

It takes one sentence or a whole LLM reply, with no framework. Text without a sentence break yields exactly
`generate(text)[0]`.

**Items and arguments**
- **`audio`** is a 1-D float32 array at `model.sampling_rate`. It is exactly `generate(spoken, ...)[0]` for that
  sentence alone.
- **`sentence`** is the raw slice of `text`: before TN, tags kept.
  - With one sentence it is the caller's string, byte for byte.
  - With several, each piece is `.strip()`ped.
- **`duration`**:
  - `None` means upstream's estimate for each sentence, including its boost below 2 s.
  - A callable is called once per sentence, on the spoken text, and returns seconds.
  - A number (int, float, np.float64, bool) raises TypeError. One float applied to every sentence would be a median
    +106 % per sentence.
- **`normalize`** is `None` or `(sentence, language id) -> spoken text`; `TextNormalizer.normalize` fits as it is.
  - A sentence whose spoken text has nothing to say (`not any(c.isalnum() ...)`, e.g. `<3` → `''`) is skipped. It
    yields no item and draws no RNG.
  - If every sentence is skipped, the stream is empty.
- **The other arguments** are `generate()`'s and apply to every sentence. The language is resolved once and config
  fields are validated once, both before the split.

**Code** (`models/omnivoice.py`; imports gain `from collections.abc import Callable, Iterator`, and `ABBREVIATIONS`
next to `chunk_text_punctuation`; annotations in the §13 style):

```python
    def generate_streaming(self, text: str, language: str | None = None, *, instruct: str | None = None,
                           duration: Callable[[str], float] | None = None, speed: float | None = None,
                           normalize: Callable[[str, str | None], str] | None = None,
                           generation_config: OmniVoiceGenerationConfig | None = None,
                           generator: torch.Generator | None = None,
                           **kwargs) -> Iterator[tuple[np.ndarray, str]]:
        """Speak ``text`` one sentence at a time: yield ``(audio, sentence)`` as soon as each is ready.

        ``sentence`` is the raw slice of ``text`` that ``audio`` speaks (tags kept; stripped when there are
        several). Each item is ``generate(spoken, ...)[0]`` for that sentence alone, so text without a
        sentence break yields exactly ``generate(text, ...)[0]``. ``normalize(sentence, language) -> str``
        turns a sentence into the text the model speaks (``TextNormalizer(...).normalize`` fits); a sentence
        it leaves with nothing to say is skipped. ``duration(spoken) -> seconds`` is called on that text,
        because one float cannot be split across sentences. The other arguments are generate()'s.
        Nothing runs before the first ``next()``; ``close()`` stops after the sentence in flight.
        Step and close it from one thread.
        """
        if duration is not None and not callable(duration):
            raise TypeError("generate_streaming(): duration must be None or a callable (spoken text -> seconds)")
        language = _resolve_language(language)  # one ID for normalize and generate(); unknown fails before TN
        gen_config = replace(generation_config or OmniVoiceGenerationConfig(), **kwargs)  # a typo raises now
        segments = _split_sentences(text)
        if not any(c.isalnum() for c in text):
            raise ValueError(f"text has nothing to speak: {text!r}")
        for segment in segments if len(segments) == 1 else [s.strip() for s in segments]:
            spoken = segment if normalize is None else normalize(segment, language)
            if not any(c.isalnum() for c in spoken):  # TN emptied it ("<3" -> ""): nothing to say
                continue
            yield self.generate(spoken, language, instruct=instruct, speed=speed,
                                duration=None if duration is None else duration(spoken),
                                generation_config=gen_config, generator=generator)[0], segment


# ---- module level, next to _combine_text
# A run of . ! ? … plus closing quotes/brackets, then whitespace, the end or a non-verbal tag; or 。！？.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’»)\]]*(?:\s+|$|(?=\[))|[。！？]+[”’」』）]*\s*")
# Upstream's English list is shared with generate()'s >30 s chunker, so it stays untouched; Spanish is added here.
_SENTENCE_ABBREVIATIONS = frozenset(ABBREVIATIONS | {"Sra.", "Srta.", "Dra.", "Lic.", "Ing.", "Av.", "Gral.", "Ud.",
                                                     "Uds.", "EE.", "pág.", "Pág.", "aprox.", "núm.", "nro.", "Nro.",
                                                     "tel."})


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` after each sentence end, keeping every character (``"".join(result) == text``).

    No split inside 3.5, 21:30, 1.500 or URLs (no whitespace follows), after a listed abbreviation or an
    initial, or before a lowercase letter, also behind ¿ ¡ quotes or "(" ("Bueno... ¿qué?"). A piece with
    nothing to say but non-verbal tags, punctuation or emoji joins the next piece; at the end, the previous one.
    ponytail: Latin and CJK marks only; no split without whitespace after the mark ("¡Hola!¿Qué?"); the
    abbreviation list is finite ("Bs. As.", "Sta. Fe" split: add entries when seen) and keeps upstream's
    English "No." ("No. Mañana" merges); list markers ("1. Pizza") split like sentence ends; a one-letter
    word before the mark ("¿Y?", "vitamina D.", "好。") merges; each skipped boundary re-scans from the
    last split, O(n*k) on long texts without sentence ends.
    """
    pieces, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        word = (text[start : m.start() + 1].split() or [""])[-1].lstrip("¿¡(\"'«“‘")
        nxt = text[m.end() : m.end() + 8].lstrip("¿¡(\"'«“‘")[:1]  # first letter past opening marks
        if nxt and (nxt.islower() or word in _SENTENCE_ABBREVIATIONS or (len(word) == 2 and word[0].isalpha())):
            continue
        pieces.append(text[start : m.end()])
        start = m.end()
    out = []
    for piece in pieces + [text[start:]]:
        if out and not any(c.isalnum() for c in _NONVERBAL_PATTERN.sub("", out[-1])):
            out[-1] += piece
        else:
            out.append(piece)
    if len(out) > 1 and not any(c.isalnum() for c in _NONVERBAL_PATTERN.sub("", out[-1])):
        out[-2:] = [out[-2] + out[-1]]
    return out
```

**Conditioning.** Sentences are independent `generate()` calls: no prompt, no `<|denoise|>`, no config field.
- The KohAI LoRA was trained with `prompt_ratio_range [0,0]` (train_config_omni5h_v5r2b.json:17-20,30).
- The maintainer warns against prompting with earlier segments (k2-fsa/OmniVoice#241).
- Kokoro, Piper and XTTS do the same.
- The voice inside each call is unchanged. Base-model auto voices and designed voices can change between sentences
  (#255). The fix for that is a fixed anchor, i.e. cloning, which is out of v0.1; the README says so.

**Input.** A `str` only, decided under the user's delegation.
- Grouping LLM deltas into sentences is the orchestration layer's job, as with Kokoro, Piper and XTTS.
- In kohai-v3, Pipecat already sends sentences, so each call is usually one item. Without Pipecat the caller passes
  the whole reply.
- `text: str | Iterable[str]`, plus a hold while a `[` tag is open, can be added later without breaking anything.

**Segmentation** (`_split_sentences`)
- **Lossless:** `"".join(pieces) == text`.
- **Boundaries:**
  - a run of `. ! ? …` plus closing quotes or brackets, followed by whitespace, the end or a non-verbal tag;
  - a run of `。！？`.
- **No split:**
  - inside 3.5, 21:30, 1.500 or URLs, since no whitespace follows the mark;
  - after a listed abbreviation (upstream's `ABBREVIATIONS` plus the 17 Spanish ones above) or an initial;
  - before a lowercase letter, also behind ¿ ¡ quotes or `(`.
- **Merges.** A piece with nothing to say but tags, punctuation or emoji joins the next piece, or the previous one
  at the end.
  - `'[laughter]'` alone is one item.
  - `'Hola. [laughter] ¿Qué tal?'` gives `'Hola.'` and `'[laughter] ¿Qué tal?'`.
  - `'Qué bueno. [laughter]'` stays one piece.
- **Corpus** (23,881 KohAI replies):
  - 35,709 pieces, 1.50 per reply;
  - 12,567 one-piece replies, byte-identical;
  - all 11,828 boundaries are followed by whitespace;
  - 0 pieces start lowercase, also when the first letter past opening marks is checked;
  - 0 pieces have nothing to speak.
- **Suite:** 68/68 hand cases and 20,000 fuzz strings.
- **Known ceilings** (in the docstring and the README, no code). Two of them occur in the KohAI corpus (23,881
  replies), in 2 replies each: the one-letter merge ('GTA V. Podés…', 'vitamina D. Yo…') and the 'No.' merge ('No.
  Ser chiquita…', 'No. Si les explico…'). The others have no hits.
  - Abbreviations outside the list split ('Bs. As.', 'Sta. Fe', German 'z.B.').
  - Spanish 'No.' before a capital merges with the next sentence, because upstream's English 'No.' (number) is in the
    list: '¿Vas a venir? No. Mañana tal vez.' gives '¿Vas a venir?' and 'No. Mañana tal vez.'.
  - Numbered list markers split like sentence ends: the marker ends the previous item ('Opciones: 1.'), or, right
    after a sentence end, is an item of its own ('1.').
  - A one-letter word before ? ! . or 。 merges ('¿Y? Contame.', 'vitamina D. Hay', '好。我们').
  - Marks without whitespace merge ('¡Hola!¿Qué?').
  - Devanagari '।' and Arabic '؟' are not boundaries.
  - Pathological cost: 20,000 glued marks take 3.5 s; 8,000 skipped boundaries take 1.8 s.

**Duration.** The per-sentence callable is exact. Measured alternatives on the stream design's pieces:

| option | sentences exact | laugh piece |
|---|---|---|
| total split by weight | 75.0 % | −15.2 % |
| `speed = R/X` | 68.9 % | −29.4 % |
| no duration | 21.7 % | −29.6 % |

Per-sentence frames sum to a median 0.994× of the whole-reply rule. `speed` stays a per-sentence float.

**Joins and post-processing.** Each item gets upstream's post-processing, unchanged.
- Joins are a median 0.42 s, the same as KohAI's natural sentence pause.
- The level step between items is 0.85 dB, against 0.78 dB natural.
- `pad_duration` and `fade_duration` are the only join knobs, and they stay caller knobs.
- A sentence above `audio_chunk_threshold` is one item, through `generate()`'s chunked path, delivered after the
  whole sentence. KohAI's longest sentence is 17.1 s, so it never happens there.

**Laziness and errors**
- Nothing runs before the first `next()`.
- These raise at the first `next()`, before any TN:
  - a number for `duration` (TypeError);
  - an unknown language (ValueError);
  - an unknown or non-config keyword such as `normalize_text=True` (TypeError);
  - non-str text (TypeError, from the splitter);
  - raw text with nothing to speak (ValueError): `''`, `'   '`, `'...'`, `'😀'`, `'¿?'`.
- Other errors (TN, OOM) come out of the `next()` that hits them. Earlier items stay valid, and the stream is then
  finished.
- `close()` before the first `next()`, `close()` at a yield, or dropping the stream does no further work.

**Inference mode.** There is no wrapper. The hooks run outside inference mode, `generate()` runs inside, and the
caller is outside between yields. Iterating inside the caller's own `inference_mode` or `no_grad` also works. Items
are plain numpy arrays.

**Threads**
- Step and close a stream from one thread. `close()` from another thread during `next()` raises
  `ValueError('generator already executing')`.
- Between yields that thread may call `generate()`.
- One call at a time per model instance.
- One `TextNormalizer` per TTS thread, because NeMo keeps parser state (§8).

**RNG**
- The caller's generator, or the global RNG, is drawn in sentence order and only inside `next()`. Each item depends
  on the items yielded before it; a skipped sentence draws nothing.
- With `generator=None`, seed right before the first `next()`.
- A private generator leaves the global RNG untouched.

**Global state.** None. Nothing is printed at fd level, at import or at call time. The module data is a compiled
regex and a frozenset.

**Latency and GPU time (2026-10-20 without the LLM; 2026-10-21 on lib-like builds, alone and next to it; G3c, G3d).**
- **Setup.** RTX 5090, fp16, batch 1, 32 steps, upstream code @ 08be0b4. The stream is emulated exactly: one
  `generate()` per normalized sentence, in order, one shared RNG. A critic re-ran M0, M1 and M2 in fresh processes
  with independent code and reproduced them within ±3.5 %. Results and scripts: `rtf/FINAL.md` in the evidence
  directory.
- **Each call has a floor.** Both backends are launch-bound at short lengths, so a call costs the same whatever the
  sentence length, up to a threshold.
  - sdpa: 7.8-8.1 ms per step, flat up to L = 320 tokens (rule duration ≈ 10.6 s), so about 0.26 s per call. Past
    it, cuBLAS switches GEMM tiles: 10.0 ms per step, rising to 14.5 ms at 21 s.
  - FlashInfer: about 3.1 ms per step up to a packed length P = L + T ≈ 320 (≈ 5.7 s), so about 0.10 s per call;
    4.8 ms per step at 10 s.
  - RTF on the 12 bench lines: 0.0559 (sdpa) and 0.0243 (FlashInfer). On the lib-like builds (decode-only codec,
    `use_cache=False`, CPU-first; bench2, quiet session): 0.0530 (sdpa) and 0.0233-0.0239 (FlashInfer); with int8,
    0.0572 and 0.0311-0.0312 (§10).
- **The stream against the whole reply,** for multi-sentence KohAI replies. The figures are corpus-weighted: a
  measured-curve model over 10,135 v8 replies, whose median error on the 77 measured calls is −2.0 % (sdpa) and −0.1 %
  (FlashInfer).

  | | sdpa | FlashInfer |
  |---|---|---|
  | first audio, whole reply (median / p90) | 261 / 265 ms | 108 / 157 ms |
  | first audio, stream (median / p90) | 257 / 260 ms | 104 / 106 ms |
  | gain (median / p90) | 3.5 / 6.3 ms | 4.2 / 51 ms |
  | gain counting TN (median / p90) | 13.8 / 23.4 ms | 16.2 / 64.9 ms |
  | replies that gain more than 20 ms | 3.0 % | 29.8 % |
  | TTS time, stream / whole (median) | 1.98× | 1.94× |

  - On the measured, length-stratified sample of 20 real replies, the 7 replies past sdpa's knee gain 162 ms median
    (70-229 ms).
  - FlashInfer's whole reply (104-268 ms up to 17.5 s) arrives no later than sdpa's first streamed sentence (257-270
    ms). On the lib-like builds this holds up to 10.5 s (190 against 264 ms), but not at 17.5 s (288 against 259 ms;
    bench2 perf M2).
  - On the lib-like builds (bench2, quiet session), the medians of the first two rows (whole reply / stream) are
    243 / 240 ms (sdpa) and 108 / 103 ms (FlashInfer).
- **Playback never underruns.** There were 0 underruns in 20 replies × 5 reps per backend, also with TN. The minimum
  slack is 1.02 s (sdpa) and 1.18 s (FlashInfer).
- **A sentence in flight** finishes within 0.28 s (sdpa) or about 0.13 s (FlashInfer) for sentences up to 7.2 s.
  Longer ones take more: 0.17 s on FlashInfer at 9.8 s, and at 32 steps about 0.43 / 0.24 s at 15 s and 0.49 / 0.30 s
  at 20 s (sdpa / FlashInfer), so about 0.46 s on sdpa for KohAI's longest sentence (17.1 s). There is no per-step
  cancel in v0.1; add one when a measured barge-in needs it.
- **Next to the LLM** (G3d with llama-server running kohai-v3's model, argv and request shape; bench2, with C and D
  re-run by the critic).
  - While llama-server decodes back to back, one call takes 1.12× as long on sdpa and 1.23× on FlashInfer (median).
    With int8: 1.31× on FlashInfer with §10's forward, and 1.16× (sdpa) and 1.34× (FlashInfer) with slim.py's.
  - Meanwhile the LLM keeps 34 % (sdpa) and 31 % (FlashInfer) of its decode speed, and 23-24 % with int8.
  - The contention is on the GPU. The TTS thread's CPU dispatch stays within 2 % of alone on fp16 at 1.5-5 s (A and
    C). On D, and on B and C at 7 s, it grows along with the GPU time, because launches block once the GPU falls
    behind. The two processes most likely time-slice the GPU (inferred; no MPS daemon ran).
  - The real pipeline ran 20 KohAI replies through `_split_sentences` on the token stream, with one TTS thread and
    no TN. First audio came at 579 ms (sdpa) and 428 ms (FlashInfer) median, p90 681 / 510 ms.
  - That first audio is the LLM's first sentence (≈ 0.30 s) plus one TTS call; the handoff takes 0.1 ms.
  - There were 0 underruns in 41 later sentences, with a minimum slack of 0.69-0.85 s.
- **Single-sentence replies.** The item is bit-identical to `generate()` (50/50 reps per backend) and takes the same
  time.
- **TN** with the pickled grammar: the first sentence takes 7.6 ms, against 17.2 ms for a whole 2-sentence reply. A
  process that built its own grammar runs TN about 4× slower (cause not found), hence §8's advice to prebuild in a
  separate process.
- **Cold start.** The first `generate()` takes 0.55-0.56 s (sdpa) and 0.65-0.89 s (FlashInfer). One warm-up call at
  startup removes it.
- **What it means.**
  - Within one text, sentence streaming brings the first audio forward only for long texts. For typical KohAI replies
    it gains 4-16 ms and costs about 2× the TTS time. FlashInfer + int8, GPU-bound almost from the start, is the
    exception: it gains 30 ms median, yet its first sentence (119 ms) still comes after FlashInfer fp16's whole reply
    (108 ms).
  - What matters more is the backend (FlashInfer is about 2.5× faster on short sentences) and overlapping TTS with
    the LLM. Feeding each sentence as the LLM finishes it, as Pipecat's sentence aggregation does, makes each call
    usually one item. The README says so. Measured next to the LLM (above), that overlap brings the first audio in
    before the LLM alone would finish the median reply (0.66 s).
- **Not measured.**
  - The library's own code (G3a, G3b).
  - TN on the TTS thread while the LLM decodes (G-TN6), and close or cancel under contention (G3e).
  - FlashInfer CUDA-graph buckets, and 16 steps on real replies.

**CPU proof** (stream critic)
- The stream against upstream `generate()`: 840 cases identical.
- The stream against the stream design's `generate()`: 2,080 cases identical.
- Semantics: 36 checks. Inference-mode edges: 5 checks.
- Global state: 0 bytes of output, nothing changed.
- With the real `TextNormalizer`: 42 replies.
- `generate()` against upstream: 20/20 identical with defaults, 10/10 single-pass.
- The final splitter (F1) and the config check (F2): 68/68 cases, fuzz, 0 corpus changes, 12/12 identical items.
- The skip of TN-emptied sentences was added with part 4 and is covered by §18's tests.

**Not in v0.1:** a per-step cancel, an inference_mode wrapper, a public splitter, a minimum-length merge, a Result
type, LLM-delta input.

## 8. Text normalization (`TextNormalizer`)

**What and where**
- **Engine.** Spanish TN is NeMo's `Normalizer(input_case="cased", lang="es")` from nemo-text-processing 1.2.0,
  wrapped by `es_rules.normalize(text, nemo)`. It runs on the text between `[...]` spans, using upstream's protection
  helpers.
- **Location.** Upstream's own TN slot, `omnivoice_tts/utils/text.py`, plus `omnivoice_tts/utils/es_rules.py`.
- **A step the caller runs.** The caller normalizes before `generate()`, or passes `normalize=tn.normalize` to
  `generate_streaming()`. There is no `normalize_text=` flag, because with a duration the caller computes, the flag
  is a trap. On the 485 held-out number sentences, `dur(raw)/dur(normalized)` has a median of 0.938, a p10 of 0.807
  and a minimum of 0.568; 33.8 % are more than 10 % short.
- **Spanish only.** `None` and every other language pass through unchanged. Upstream's zh/en WeTextProcessing path and
  its num2words fallback are deleted:
  - WeTextProcessing prints INFO lines through its own handler and writes FSTs into site-packages;
  - num2words is unusable for Argentine Spanish ('$tres.quinientos', 'ciento uno personas').
- **No module-level state, no output.** The library writes 0 bytes to fd 1/2. It touches NeMo's logger only for the
  length of its own NeMo calls, and puts it back afterwards.

**API**
```python
tn = TextNormalizer(cache_dir=None)    # nothing is imported or built yet
spoken = tn.normalize(text, language)  # upstream's normalize_text(text, language) call shape
```
- `language` is required; without it the call raises TypeError. It takes the values `generate()` takes.
  - `"es"`, `"ES"`, `" es "`, `"Spanish"` and `"spanish"` are normalized, through `LANG_NAME_TO_ID`.
  - `None`, `"en"`, `"spa"`, `"es-AR"`, `"es_419"` and unknown strings come back unchanged, without importing NeMo.
  - Whitespace-only text also comes back unchanged.
- **Errors.**
  - ImportError on every Spanish call while nemo-text-processing is missing; the message gives the install command.
  - OSError when an explicit `cache_dir` cannot be created or written. It is raised before the ~20 s grammar build.
  - Anything else propagates from NeMo or es_rules. Upstream and PR #6 swallowed it with a warning and spoke the raw
    text. No input raised in about 41,000 measured calls.
- **Contract.**
  - One call at a time per instance. One instance shared by 4 threads raised 1 ValueError in 1,200 calls (measured).
  - pynini never releases the GIL (`_pynini.so` has no `PyEval_SaveThread`). Call TN from the TTS or worker thread,
    never from an event loop.
  - Two instances may run in two threads. Outputs stay correct, but see the logging ceiling below.

**Files and provenance**

| File | Source | Lines |
|---|---|---|
| `omnivoice_tts/utils/es_rules.py` | `git show <PR #6 merge sha>:indextts25/utils/es_rules.py`, byte for byte. Never edited here. Its `<word\|pron>` branch (R:1156-1177) is inert unless the text contains IndexTTS `<x\|y>` syntax; then it keeps the annotation verbatim, digits included. | 1,177 |
| `omnivoice_tts/utils/text.py` | Upstream with the §3 cuts; the TN section is the code below. | ~332 (TN section 140, of which 78 code; upstream's is 202/117) |
| `tests/test_es_normalization.py` | PR #6's test file. From `def check` on it is byte-identical; only the header differs. | 870 |
| `tests/test_text_normalizer.py` | New (§18). | ~130 |

- The pinned es_rules copy is the file at PR #6's merge commit, which carries the license header (§14). Its sha256 is
  recorded in `test_es_rules_is_the_indextts25_copy` when it is copied. The verification below ran on dfb3286f (sha256
  `59702b506a9173bd7db08fb91d4cbe056552ca83863d231d3e0c41266c314691`). Adding the two header lines changes no
  behavior.
- Per T6-A the es_rules task waits for that merge, so TN is the last implementation task. If the user wants TN
  sooner, the task copies the PR #6 commit that adds the header, pins its sha256, and re-copies and re-pins at the
  merge commit.
- The test file's header changes are:
  - the docstring;
  - an added `import os`;
  - its two imports, which become `omnivoice_tts.utils.es_rules._spell` and `TextNormalizer`;
  - one added line: `normalize_text = TextNormalizer(cache_dir=os.environ.get("OMNIVOICE_TN_CACHE")).normalize`.
- Reused from upstream text.py:
  - `_BRACKET_TAG_RE` (T:246), verbatim;
  - `_normalize_segment` (T:338-357), without its try and `logger.warning`;
  - `_apply_with_protection` (T:360-388), without pinyin and without the span merge. Its output equals upstream's own
    helper on 148 inputs (measured).
  - T:247-336 and T:391-429 are deleted.
- Reused from indextts25's nemo_tn.py:
  - the pickle name `es-cased-nemo{v}-pynini{v}.pkl`;
  - the try-load-then-rebuild flow, and the tmp file plus `os.replace`;
  - the Spanish hook (:181-182), verbatim.
- Not copied from nemo_tn.py:
  - the singleton (:192-205);
  - `setLevel(WARNING)` (:141-143). It does not work anyway: NeMo's import resets the level to INFO, and PR #6's own
    cold build printed the 97 B INFO line (measured).
  - the warnings, the service-code maps and the XDG default directory.
- **Import cost.** es_rules compiles its regexes at import: 14.8 ms, paid by every `import omnivoice_tts`. Importing
  utils/text does not import NeMo (measured).

**Code: the TN section of `utils/text.py`.** The header imports gain `contextlib`, `os`, `pickle`, `tempfile`,
`from importlib.metadata import version`, `from omnivoice_tts.utils import es_rules` and
`from omnivoice_tts.utils.lang_map import LANG_NAME_TO_ID`; `logging` stays. Upstream's `typing` import goes, since
annotations use `X | None` and built-in generics (§13).

```python
# ---------------------------------------------------------------------------
# Optional text normalization: call TextNormalizer.normalize() before generate()
# ---------------------------------------------------------------------------
#
# Spanish only: NeMo's grammar wrapped by ``es_rules``, a byte-identical copy of
# nicokim/indextts25-inference indextts25/utils/es_rules.py (fix it there first,
# then copy it). Upstream's zh/en (WeTextProcessing) and num2words paths are gone.
#
# The OmniVoice inline control syntax must survive normalization: every ``[...]``
# span (non-verbal tags such as ``[laughter]``; CMU pronunciation overrides such
# as ``[B EY1 S]``, whose stress digit would otherwise be read as a number) is
# held out and re-inserted verbatim.

_BRACKET_TAG_RE = re.compile(r"\[[^\[\]]*\]")

_TN_INSTALL_MSG = (
    "Spanish text normalization needs nemo-text-processing: "
    "pip install 'omnivoice-tts-inference[tn]' (Linux x86_64, CPython 3.10-3.13, a C compiler). "
    "Elsewhere, install pynini from conda-forge first, then nemo-text-processing."
)


class TextNormalizer:
    """Spell out numbers, money, dates, times... before ``generate()``.

    ``normalize(text, language)`` takes the ``language`` you pass to ``generate()``.
    Only Spanish (``"es"``, ``"Spanish"``) is normalized: NeMo's grammar plus the
    rules in ``es_rules``, which follow Argentine/Latin American usage (``$`` is
    pesos, ``US$``/``USD`` are dollars, ``21.30`` is a time). Any other language,
    or ``None``, comes back unchanged. ``[...]`` spans (non-verbal tags, CMU
    pronunciations) pass through byte for byte; numbers do not agree across them
    (``1 [laughter] perro`` -> ``uno``).

    The Spanish grammar is built on the first Spanish call: about 20 s and 500 MB
    of RAM. With ``cache_dir``, the built grammar is pickled there (about 60 MB,
    mode 0600) and later instances load it in about 0.5 s; only a regular file
    owned by this user and not writable by group or others is unpickled.

    One call at a time per instance: NeMo's parser keeps state. pynini holds the
    GIL while it runs (up to ~30 ms per sentence), so keep it off event loops.
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir
        self._es = None

    def normalize(self, text: str, language: str | None) -> str:
        code = (language or "").strip().lower()
        if LANG_NAME_TO_ID.get(code, code) != "es" or not text.strip():
            return text
        if self._es is None:
            self._es = _spanish_nemo(self.cache_dir)
        with _quiet_nemo():
            return _apply_with_protection(text, self._spanish)

    def _spanish(self, text: str) -> str:
        return es_rules.normalize(text, lambda t: self._es.normalize(t, verbose=False))


def _spanish_nemo(cache_dir: str | None):
    """NeMo's Spanish Normalizer, from ``cache_dir`` when it holds our own pickle."""
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer
    except ImportError as e:
        raise ImportError(_TN_INSTALL_MSG) from e
    if cache_dir is None:
        with _quiet_nemo():
            return Normalizer(input_case="cased", lang="es")
    # indextts25's nemo_tn.py cache. NeMo's own cache_dir is not used: its FAR
    # export optimizes again and reads some decimals differently ("1,80").
    path = os.path.join(
        cache_dir,
        f"es-cased-nemo{version('nemo_text_processing')}-pynini{version('pynini')}.pkl",
    )
    try:
        # Unpickling runs code: only a regular file (not a symlink) that this user owns and
        # nobody else can write. ponytail: POSIX only (os.getuid, O_NOFOLLOW); elsewhere
        # the cache is never trusted and every start rebuilds.
        with open(path, "rb", opener=lambda p, fl: os.open(p, fl | os.O_NOFOLLOW)) as f:
            st = os.fstat(f.fileno())
            if st.st_uid == os.getuid() and not st.st_mode & 0o022:
                return pickle.load(f)
    except Exception:  # missing, corrupt, or written by another version: rebuild it
        pass
    os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    # mode 0600 whatever the umask; an unwritable cache_dir fails here, before the ~20 s build
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            with _quiet_nemo():
                normalizer = Normalizer(input_case="cased", lang="es")
            pickle.dump(normalizer, f)
        os.replace(tmp, path)  # atomic: a process starting meanwhile never reads half a file
    except BaseException:
        os.unlink(tmp)
        raise
    return normalizer


@contextlib.contextmanager
def _quiet_nemo():
    """NeMo logs through its own stderr handler and resets its logger to INFO on
    every normalize(). For the length of our call, its records go to our logger at
    DEBUG instead, and its level is put back afterwards: the net change is none.
    ponytail: not coordinated across threads; two instances normalizing at the same
    time can leave NeMo's logger at INFO (measured). A module-level lock fixes it."""
    nemo_logger = logging.getLogger("NeMo-text-processing")
    level = nemo_logger.level

    def to_debug(record):
        logger.debug("NeMo: %s", record.getMessage())
        return False

    nemo_logger.addFilter(to_debug)
    try:
        yield
    finally:
        nemo_logger.removeFilter(to_debug)
        nemo_logger.setLevel(level)


def _normalize_segment(fn, segment: str) -> str:
    """Normalize one segment, keeping its outer whitespace: the normalizer strips
    it, which would glue words to an adjacent span (``the [B EY1 S] guitar``)."""
    if not segment.strip():
        return segment
    lead = segment[: len(segment) - len(segment.lstrip())]
    trail = segment[len(segment.rstrip()) :]
    return lead + fn(segment.strip()) + trail


def _apply_with_protection(text: str, fn) -> str:
    """Run ``fn`` on the text between ``[...]`` spans; the spans pass through verbatim."""
    out: list[str] = []
    last = 0
    for m in _BRACKET_TAG_RE.finditer(text):
        out += [_normalize_segment(fn, text[last : m.start()]), m.group()]
        last = m.end()
    out.append(_normalize_segment(fn, text[last:]))
    return "".join(out)
```

**Tags and pronunciation overrides.** Every `[...]` span passes through verbatim. That covers the 13
`_NONVERBAL_PATTERN` tags (M:1651), CMU overrides such as `[B EY1 S]`, and any other bracketed text, digits included
(`[2]`, `[4.000]`), as upstream does. The README says to use brackets only for tags and overrides.

| Check, on an independent set of 109 probes | Result |
|---|---|
| `[...]` spans changed | 0 |
| Non-verbal tag list changed | 0 |
| Non-idempotent outputs | 0 |
| Bytes written | 0 |
| Digits left outside brackets | only inside IndexTTS `<x\|y>` annotations |

- Without the wrapper (PR #6 on the whole text), 7 of the 109 probes change a span (`[B EY uno S]`, `[dos]`,
  `[laughter ]` trimmed), and 28 keep digits glued to a tag (`Son las 21:30[laughter]`). With the wrapper:
  `Son las veintiuna y media[laughter]`.
- Number agreement across a tag is lost, with or without the wrapper: PR #6 also gives `Tengo 21 [laughter] años.` →
  `veintiuno`, because es_rules treats the tag as a word.
- Real KohAI tagged text is untouched: the 1,086 tagged kohai_r2 training labels are fixed points of both
  implementations.

**Locale.** es_rules follows Argentine and Latin American usage. This is documented in the docstring and the README,
with no option.

| Convention | es_rules.py |
|---|---|
| `$` is pesos; `US$`, `U$S` and `USD` are dollars | R:11-14; `_money` R:365-389 |
| `21.30` and `hs` are times | R:505-534 |
| Heights read "uno setenta" | R:115-118 |
| `911` reads "nueve once" | R:1055 |
| `<3` and `:3` are dropped | R:834-838, R:956 |
| Chat spellings (`tod@s`, `100pre`) | R:977, R:1058 |
| Voseo decides agreement | R:147-151, R:240-248 |
| "Mar del Plata" is a place | R:818 |

`$` as pesos is wrong for Spain and the dollarized varieties (es-ES, EC, SV, PA, PR, es-US). For them the README says:
write US$ or USD, pre-normalize your own text, or skip TN.

**Grammar cache**
- **Location.** Only the caller's `cache_dir`. `None`, the default, means no disk I/O: no write and no load from a
  default location. The file name carries the nemo and pynini versions, so an upgrade writes a new file; old files
  are the caller's to clean up. Measured sizes: 59,701,778 B with pynini 2.1.6.post1, 48,378,776 B with 2.1.7.
- **Write.**
  1. `makedirs(mode=0o700)`. The mode applies only to a directory the library creates; an existing directory keeps
     its permissions.
  2. `mkstemp`, which creates the file 0600 whatever the umask, before the build. An unwritable `cache_dir` fails in
     0.2 s instead of after the build.
  3. `os.replace`. It is atomic, and the last writer wins.
  4. The tmp file is unlinked on any failure.
- **Load.** The file is unpickled only when it opens with `O_NOFOLLOW` (a symlink is refused), its `fstat` owner is
  `os.getuid()`, and it has no group or other write bit. Otherwise the grammar is rebuilt and the file replaced.

  | Planted file | Unpickled? | What happens |
  |---|---|---|
  | 0664 payload | no | rebuilt, replaced by 0600 |
  | 0646 payload | no | rebuilt, replaced by 0600 |
  | Symlink to a 0600 payload the user owns | no | refused |
  | 0644 payload the user owns | yes | the same-uid trust boundary, by design |

- **Why not NeMo's own `cache_dir`.** Its FAR export optimizes again and changes some readings ("1,80"). pynini's
  export also calls the root `logging.info`, which installs a root handler through an implicit `basicConfig`.
- **Timing** (measured).
  - Cold build: 24.9 s on py3.12 and 16.8 s on py3.13; peak RSS 670 MB.
  - First call after a pickle load: 0.40-0.61 s.
  - On py3.12 with pynini 2.1.6.post1, the process that did the cold build stays about 4× slower per call for its
    whole life (21.8 against 5.4 ms median on the 630 cases; 48.2 against 11.9 ms on real replies; not GC, cause not
    found). So prebuild in a separate process, or restart after the first build. On py3.13 with pynini 2.1.7 a call
    takes 3.1 ms, even in the cold-build process.
  - With a pickled grammar, TN of a reply's first sentence takes 7.6 ms median, against 17.2 ms for a whole
    2-sentence reply (§7).
- **Prebuild:**
  `python -c "from omnivoice_tts.utils.text import TextNormalizer as T; T('/srv/kohai/tn').normalize('1', 'es')"`

**Logging and global state** (details in §11)
- `_quiet_nemo` is the library's only logging call besides `logger.debug`. For the length of each build or call,
  NeMo's records go to the library's logger at DEBUG; afterwards the filter is removed and NeMo's level restored
  (NeMo's `normalize()` resets it to INFO on every call).
- Measured with a wide snapshot (loggers with handlers and filters, the logging manager, `os.environ`, the recursion
  limit, `warnings.filters`, atexit, threads, excepthooks, `copyreg`, locale, cwd, umask, `sys.path`, `meta_path`,
  RNG states):
  - with the app setting NeMo to ERROR after importing it, through warm, cold and no-cache starts, 14 sentences, a
    720-word input, Spanish aliases and other languages: 0 bytes on fd 1/2 and an empty diff, on py3.12 and py3.13;
  - an app that ran `basicConfig(DEBUG)` itself receives NeMo's records as DEBUG from `omnivoice_tts.utils.text`,
    and the fds stay empty.
- **Ceiling.** Two instances normalizing in two threads at the same time leave the app's NeMo level at INFO instead
  of ERROR (measured). Outputs stay correct and nothing is printed. A module-level RLock would fix it (+3 lines), but
  it is module-level state; kohai-v3 runs TN on one thread.
- NeMo's own "Failed text" and ">500 words" warnings go to DEBUG. A caller that must never speak digits can check for
  digits outside the brackets.

**Duration, chunking, streaming**
- The caller computes any duration on the normalized text. Raw text weighs a median 0.931 of the hand-reviewed spoken
  form; normalized text weighs 1.000 (measured).
- Normalized Spanish has no digits left on the KohAI corpus (0 on 34,090 sentences), so there `generate()`'s >30 s
  chunker cannot split inside `21:30` or `1,000`. On the 485 held-out number sentences, 2 keep digits: NeMo returns
  `localhost:5000` unchanged, and `10²` stays.
- `generate_streaming(normalize=tn.normalize)` normalizes each raw sentence, computes the duration on the spoken text
  and yields the raw sentence as the subtitle (§7). It skips a sentence that TN empties.
- With `generate()`, the caller checks the normalized text before calling: `generate()` raises ValueError on text with
  nothing to speak.

**Keeping es_rules.py in sync**
- **Ownership.** indextts25-inference owns the file while PR #6 is open and kohai-v3 still runs on it. Fixes land
  there with their test case. The merged file is then copied here with `git show <sha>:indextts25/utils/es_rules.py`,
  and the test body is copied from `def check` on.
- **The pin.** The sha256 test fails on any local edit.
- **When to revisit.** Once kohai-v3 stops using indextts25-inference. Then either this repository becomes the owner
  and the pin is dropped, or, if a fix has had to land in both repositories twice, a shared rules package is made
  (T6-B).
- **Why not depend on indextts25-inference.** No release contains PR #6 yet, and it requires librosa, which brings
  scikit-learn and with it the KMP env vars and filters listed in §11 as residual effects. §12 drops librosa, so they
  do not occur.

**Validation** (critic, CPU, measured on dfb3286f)

| Check | Result |
|---|---|
| 34,090 corpus sentences, proposed against PR #6 | 0 diffs, 0 exceptions, 0 bytes, 0 non-idempotent, 0 digits left; median 14.1 ms with 6 workers |
| PR #6's 630 cases | proposed == PR #6 == expected on py3.12 and py3.13 |
| Ported test file | 647 passed (py3.12, py3.13) |
| kohai_5h training labels | 5,350/5,350 reproduced (249 changed by TN) |
| Tag probes | 109: 0 span or tag changes; the helper equals upstream's on 148 |
| Tagged kohai_r2 labels | 1,086 fixed points |
| Adversarial inputs | 39: 0 exceptions, 0 bytes |
| Silence and state | 0 B and an empty diff: warm, cold and no cache (py3.12); cold (py3.13) |
| Two threads | NeMo level lost without a lock, kept with it |
| Pickle guard | refusals as in the table above |
| Unwritable cache dir | fails in 0.2 s, before the build |

**Known ceilings** (documented)
- **Drift.** The two es_rules copies can drift while PR #6 changes. The pin catches only local edits; copy again at
  the merge commit and after each later fix.
- **Two-thread logging,** above.
- **Same-uid trust.** Any pickle the user owns and nobody else can write is trusted. On Windows the cache is never
  trusted, so every start rebuilds.
- **Hidden NeMo failures.** NeMo's internal failures return the raw segment, and their warning goes to DEBUG. One was
  seen in G-TN1: a sentence containing `localhost:5000` comes back unchanged, while "apunte a 5000" alone is
  normalized.
- **Locale.** The es-AR conventions apply to all 'es' text.
- **Audio eval (G-TN1, passed narrowly).** 485 held-out number sentences were rendered on v5r2b_576 with the KohAI
  duration rule on the normalized text.
  - TN scores CER 0.029, 82 % ok ≤ 5 % and dur/ref 0.99. Hand-written words score 0.013 / 93 %, and raw digits
    0.065 / 53 %.
  - The model reads TN's text with CER 0.008, so the gap is in the text.
  - 88 sentences miss ok ≤ 5 %, 34 of them with hand-written words too. Of the other 54, 34 hit real TN errors or
    gaps (hand labels): symbols such as `>`, `¥` and `√`, `N+`, `00:00`, m:ss timestamps, units such as mAh and
    kcal, abbreviations such as lvl and s. XVIII, and Roman numerals. At least 10 more hide among the shared misses,
    such as codes read as cardinals, "5° C", Roman numerals and `localhost:5000`.
  - Per the ownership rule above, those fixes land in indextts25-inference first.

## 9. FlashInfer (`models/omnivoice_flashinfer.py`)

- `apply_flashinfer(model) -> model`: no kwargs, idempotent, never imported by `omnivoice_tts/__init__.py`. The
  module-level `import flashinfer` is wrapped; the ImportError's message gives the install command.
- State lives on the model: a `PackedAttnRunner` in `model._fi_runner` (wrapper, pos_ids, last planned doc_lens). The
  128 MiB workspace and FlashInfer's 8 MiB int buffers are allocated on the first plan, on the model's device. Plan
  dtype follows the model (fp16/bf16; fp32 rejected).
- No attention registry and no `set_attn_implementation`: the patched attention calls the wrapper directly;
  `attention_mask={'full_attention': None}` skips mask building; `use_cache=False` is passed at the call site.
- Fusion: bias-free `nn.Linear` `qkv_proj` and `gate_up_proj` built on meta from `torch.cat` weights, computed under
  `torch.no_grad()`; the originals are deleted (frees 587,202,560 B). With an autograd graph, the fused tensor
  (`CatBackward0`) keeps q/k/v/gate/up alive and the deletion frees 0 B. That is upstream's case, measured.
- Deleted from upstream: CUDA graphs and buckets, the doc_slots SDPA branch, the stats hook, the unfused branch, every
  kwarg.
- RNG: `_generate_iterative_packed(self, task, gen_config, generator=None)`, same signature as sdpa.
- Guards before patching:
  - CUDA available and fp16/bf16, else ValueError;
  - `q_proj` an `nn.Linear`, else TypeError ("call it before apply_int8"), where upstream fails with
    `AttributeError: 'Int8Linear' object has no attribute 'weight'`;
  - non-empty `flashinfer.jit.current_compilation_context.TARGET_CUDA_ARCHS`, else a RuntimeError naming
    `FLASHINFER_CUDA_ARCH_LIST=12.0a` (cu128) or `12.0f` (cu129). The library never sets env vars.
- Contract: after patching only `generate()`/`generate_streaming()` may drive the LLM. Output is not bit-identical
  to sdpa; it is judged by CER/SIM. Judged on the lib-like emulation, it is undetectable against sdpa on the 124-line
  paired eval, on 13 metrics. Its 5 flagged endings against 2 are not detectable (p 0.25). In the blind listening
  (block 3, A against C) the user heard no difference in the 9 pairs answered.
- Header: keep NVIDIA's SPDX lines and add the modification notice.
- Install (README, no pip extra): `pip install flashinfer-python==0.6.15.post1
  'flashinfer-jit-cache==0.6.15.post1+cu128' --extra-index-url https://flashinfer.ai/whl/cu128/` (cu129 analogous), a
  `[[tool.uv.index]]` snippet, and FLASHINFER_CUDA_ARCH_LIST for RTX 50 cards in the launch environment.
- Recipe: `from_pretrained(repo, dtype=torch.float16)` on CPU, `apply_flashinfer(model).to("cuda:0")`, one warm-up
  `generate()` (loads the .so modules, JIT-compiles the 2 CuTe-DSL RMSNorm kernels).
- kohai-v3 stack: cu128 + `12.0a` if GPU check G0 passes, otherwise cu129 + `12.0f` (as in venv_fi).
- Measured on upstream's FlashInfer (eager, cu129; §7):
  - RTF 0.0243 on the 12 bench lines, and about 3.1 ms per step up to P ≈ 320 (≈ 5.7 s);
  - 2.57 GiB after load and patch, and 2.92 GiB peak allocated on real replies, against 1.89 / 2.24 GiB for sdpa with
    upstream's loader. Upstream keeps the original q/k/v and gate/up weights next to the fused copies; the vendored
    version deletes them (G6).
  - Upstream's own per-length CUDA graphs cut a 1.5 s call from 103 to 61 ms. Each new length costs a 0.26-0.39 s
    capture call, though, so without buckets they lose on unique sentences. They stay out, as decided in part 5.
- Measured on the lib-like emulation (bench2: CPU-first, deletion, decode-only codec, `use_cache=False`; cu129 +
  12.0f).
  - Memory: 1.358 GiB with the runner (sdpa's 1.225 plus its 136 MiB). The emulation allocated the runner at build;
    the library allocates it on the first plan, so expect 1.225 GiB after the build and 1.366 GiB after the warm-up
    call. The peak is 1.464 GiB on the bench lines (G6's estimate holds) and 1.693 GiB at a 20 s text. nvidia-smi
    shows 2,154 MiB after M0.
  - Speed: RTF 0.0233-0.0239, and the cold first call takes 0.605-0.614 s.
  - Patching on the GPU instead leaves 0.57 GiB of reserved holes that `empty_cache` cannot release: 1.932 GiB
    reserved against 1.366 GiB allocated, and 2,608 against 2,090 MiB of nvidia-smi. Hence the CPU-first recipe, and
    a runner that allocates on the first plan.

## 10. int8 (`models/omnivoice_int8.py`)

- `Int8Linear`: symmetric int8 weight, one scale per output channel, computed in fp32 and stored in the weight's
  dtype. With `w = weight.float()`: `s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) * (1 / 127)`,
  `qweight = torch.round(w / s).to(torch.int8)` and `scale = s.to(weight.dtype)`.
  - Computed this way, CPU and CUDA quantize identically and equal slim.py's `int8_layers` run on CUDA: 0 of
    440,401,920 values differ.
  - slim.py writes `/ 127`, which CUDA runs as the multiplication above. On the CPU it is a true division, and 3,209
    values differ.
- `forward` is `F.linear(x, self.qweight * self.scale, self.bias)`, as one expression, never slim.py's
  `self.q.to(x.dtype) * self.scale`.
  - Both give bit-identical PCM.
  - slim.py's form launches two kernels per Linear instead of one: +392 instead of +196 kernels per `model.llm`
    forward on sdpa.
  - It costs +15 % instead of +8 % RTF on sdpa, and +42-46 % instead of +31-34 % on FlashInfer.
- `apply_int8(model) -> model`: converts every `nn.Linear` under `model.llm.layers` (196 for Qwen3); audio heads,
  embeddings, norms, rotary buffers and codec untouched; idempotent; no knobs. After FlashInfer it converts the fused
  layers too: 112 modules (qkv, o, gate_up and down × 28), each called at every step.
  - That is 3,584 calls per 32-step `generate()`, and 0 into an `nn.Linear`.
  - Quantize-then-fuse equals fuse-then-quantize: CPU-verified, and the per-layer sha256 is the same for both layouts.
  - This relies on the vendored layout. Upstream FlashInfer followed by slim.py reports 196 Int8Linear, yet two thirds
    of the weights it reads stay fp16. So checks count forward calls, not modules.
- Recipe: `from_pretrained(dtype=fp16)` on CPU, `apply_int8(model)`, `.to("cuda:0")`. It needs no `empty_cache`, and
  the load has no transient.
  - Memory: 0.816 GiB resident. The peak is 0.922 GiB for texts up to ~6 s and 1.151 GiB at 20 s (nvidia-smi
    1,612 / 1,872 MiB).
  - Against fp16 it saves 0.41 GiB (402-422 MiB of nvidia-smi, 382 at a 20 s text). It costs +8 % RTF (0.0572
    against 0.0530) and +27 ms on a typical first sentence (267.5 against 240.2 ms).
  - With FlashInfer: 0.949 GiB resident with the runner (§9) and a 1.055 GiB peak (1.284 GiB at 20 s). RTF is
    0.0311-0.0312 against 0.0233-0.0239 (+31-34 %), still 1.70× faster than sdpa fp16.
- Not bit-identical to fp16, yet no longer experimental: both checks passed (decision #23).
  - The paired fixed-seed eval passed for B and D: 124 lines, one seed, undetectable on 13 metrics.
  - The blind listening passed: no difference heard in 19 of 20 pairs, and int8 preferred in the other (G8 for B,
    G9 for D).
  - A quantized model cannot be saved.
- No fused int8 GEMM in v0.1: `torch._weight_int8pack_mm` raises NotImplementedError on CUDA in torch 2.8.0. A GEMM
  that skips the fp16 weight needs torchao or a newer torch.

## 11. Process-global state contract

**Removed from upstream**
- the 3 warnings filters and `logging.disable` during load;
- the AutoConfig/AutoModel registration and the flex `AttentionInterface.register`;
- FlashInfer's module `_CTX`, import-time registration, `set_attn_implementation`, `llm.config.use_cache` mutation and
  graph caches;
- transformers' default-dtype flips and "Loading weights" bars, and the snapshot bars;
- stderr warnings for unknown languages and clone prompts, and the need for the `_INSTRUCT_ALL_VALID` mutation;
- global-only Gumbel noise;
- upstream TN's module-level `_ZH_NORMALIZER`/`_EN_NORMALIZER` (T:263-264), and its `logger.warning`, which reached
  stderr through `logging.lastResort` (T:352-355);
- the class-level `@lru_cache` on `RuleDurationEstimator._get_char_weight` (D:165).

**The library never** writes environment variables, adds warnings filters, keeps module-level caches or locks, calls
`torch.cuda.empty_cache` or sets `torch.backends` flags. Its only logging calls besides `logger.debug` are in
`_quiet_nemo` (§8). Around the library's own NeMo calls, `_quiet_nemo` adds a filter to NeMo's logger, then removes it
and restores the level that NeMo's `normalize()` resets to INFO. The net change is none when calls do not overlap in
time; two instances normalizing concurrently can leave NeMo's logger at INFO (measured).

Neighbors on the same machine are the caller's to tune (decision #21): the library sets no thread counts and creates
no CUDA streams. Its GPU work runs on the caller's current stream, so §15's high-priority stream and separate
processes apply as they are.

**Residual effects of dependencies** (README "Global state" list)
- numpy/torch warnings filters at import.
- scikit-learn via transformers' generation utils, when installed (KMP variables, filters).
- pydub SyntaxWarnings on first compile, and a RuntimeWarning without ffmpeg.
- flashinfer/tvm-ffi import effects.
- CUDA_MODULE_LOADING=LAZY from torch.
- The first Spanish `TextNormalizer.normalize()` imports nemo_text_processing. That import:
  - raises `sys.setrecursionlimit` to 3000;
  - adds a stderr StreamHandler at INFO to the 'NeMo-text-processing' logger. A level set before that import is reset,
    while `.disabled = True` survives. Applications should configure NeMo's logger after that call, or set
    `.disabled = True`.
  - makes joblib set KMP_INIT_AT_FORK=FALSE, unless something already set it;
  - registers one atexit callback;
  - registers a copyreg reducer for `_regex.Pattern`, through the `regex` module;
  - adds numpy's warnings filters if numpy was not imported yet.
- With `cache_dir`: one 0600 pickle (59.7 MB; 48.4 MB with pynini 2.1.7), in a 0700 directory if the library created
  it.

## 12. Dependencies and packaging

- `requires-python = ">=3.10"`.
- Runtime: `torch>=2.8`, `torchaudio>=2.8`, `transformers>=5.3`, `huggingface_hub>=1.3`, `safetensors`, `numpy`,
  `pydub`, `audioop-lts; python_version >= '3.13'`.
- One extra, for Spanish TN:
  ```toml
  [project.optional-dependencies]
  # Spanish TN. nemo pins pynini==2.1.6.post1: PyPI wheels only for manylinux x86_64, CPython 3.8-3.13.
  tn = ["nemo-text-processing~=1.2.0; sys_platform == 'linux' and platform_machine == 'x86_64' and python_version < '3.14'"]
  ```
  - The pin is `~=1.2.0` because es_rules is tuned to 1.2.0's output.
  - Resolution on linux x86_64 (uv pip compile): py3.12 and py3.13 get nemo 1.2.0 and pynini 2.1.6.post1; on py3.14
    the marker drops nemo. `uv pip check` passes in the omnivoice-kohai venv.
  - Installing the extra always compiles cdifflib, which ships no Linux wheels, and on 3.13 also editdistance. It
    therefore needs a C/C++ compiler and Python headers on every Python version, and it adds about 0.7 GB (pynini
    takes 650 MiB).
  - Python 3.14: the consumer adds nemo-text-processing itself, with a uv `override-dependencies = ["pynini>=2.1.7"]`
    (a library cannot declare one). Not run.
  - macOS, Windows and aarch64: install pynini from conda-forge first (untested).
- Groups: `dev = ["ruff"]`, `test = ["pytest", "pytest-cov"]`.
- Dropped: accelerate, gradio, tensorboardX, webdataset, soundfile, librosa, peft, WeTextProcessing, num2words, the
  eval stack, the uv cu128 constraints.
- kohai-v3 adoption:
  - it drops its `transformers==4.52.1`, `indextts25-inference[flash]` and `flash-attn` pins;
  - it declares `omnivoice-tts-inference[tn]`;
  - for FlashInfer fp16 (decision #19), it adds §9's two pinned FlashInfer wheels from FlashInfer's index and sets
    `FLASHINFER_CUDA_ARCH_LIST` at launch, since there is no pip extra;
  - its `pynini>=2.1.7` override is optional, and its comment is wrong, since 2.1.6.post1 has cp313 wheels. Keeping it
    gives the faster stack it runs today; dropping it silences `uv pip check`.

## 13. Repository, CI and release

- hatchling.
- ruff as in indextts25-inference: select E, F, W, I, UP; ignore E501, E402; line-length 120; target py310. The
  kept upstream code is converted once with `ruff check --fix` (PEP 604/585 annotations, `collections.abc` imports),
  and an ambiguous `l` (E741) that survives is renamed by hand. The spec's code blocks already use that style.
- CI: lint (`uv run --only-group dev ruff check`) and CPU tests with coverage.
  - The test job installs the extra, so the PR #6 suite and the TN regression tests run on every PR:
    `uv run --extra tn --group test pytest --cov=omnivoice_tts`.
  - It runs on ubuntu-latest, with Python pinned by `.python-version` (3.12) as in indextts25-inference. The `*` TN
    tests block or fake NeMo themselves.
- release-please (manifest, CHANGELOG, `__version__` extra-file with `# x-release-please-version`) and `release.yml`
  publishing to PyPI with trusted publishing (environment `pypi`).
- Default branch `main` (rename from `master` at implementation). Spanish conventional commit messages, author nicokim,
  `Co-Authored-By` trailer; English conventional PR titles, squash merges.

## 14. Licenses

- Code: Apache-2.0 as a derivative work; LICENSE byte-identical; modification notice per modified file; the FlashInfer
  file keeps NVIDIA's SPDX header.
- README: base weights CC-BY-NC (k2-fsa model card), codec under the Boson Higgs Audio 2 Community License.
- Copied Spanish rules. `es_rules.py` and the ported test file are the user's own code from indextts25-inference,
  whose LICENSE is the bilibili Model Use License; es_rules.py has no header of its own. Before the copy, the user adds
  `# Copyright 2026 Kimox` and `# SPDX-License-Identifier: Apache-2.0` to es_rules.py in PR #6, so both copies stay
  byte-identical. This is a prerequisite of the es_rules task, and only the user does it.

## 15. README outline

- Install and quick start.
- Generation parameters.
- Voice design (unknown items pass through).
- Non-verbal tags and pronunciation.
- Text normalization (Spanish):
  - TextNormalizer; install and platforms, including the compiler and the ~0.7 GB;
  - PR #6's 'Spanish numbers' paragraph (README.md:71-88 at dfb3286f, with the path changed);
  - the locale table, and what es-ES and the dollarized varieties should do;
  - tags, and the fact that any `[...]` passes through, digits included;
  - the cache, its security and prebuilding;
  - threads and the GIL;
  - "compute any duration on the normalized text": with the rule on the raw text, KohAI speaks 6.5 % faster (20 %
    on a third of number sentences), and the most skewed sentence becomes unintelligible;
  - what G-TN1 measured, and the known gaps: symbols, `N+`, `00:00`, timestamps, units, abbreviations, Roman
    numerals and codes read as cardinals. Rephrase or pre-normalize such text; `localhost:5000` and `10²` keep their
    digits.
- Streaming:
  - the item contract and the player-thread and asyncio patterns: one worker thread, and `TTS.submit(stream.close)` in
    `finally`;
  - what it buys, from §7's measurements. With the whole reply in hand at sdpa, the first audio of a typical KohAI
    reply comes only ≈ 4 ms (≈ 14 ms with TN) before `generate()`'s, at ≈ 2× the TTS time. Items still give
    subtitles, a cancel point at each sentence and overlap with the LLM when the caller feeds sentences. When latency
    matters, use FlashInfer: first audio in ≈ 0.10 s, and a whole reply of up to ≈ 10 s arrives before sdpa's first
    sentence. When the caller feeds sentences from a streaming LLM as they finish, as kohai-v3 does, the first audio
    comes at the LLM's first sentence plus one call: 0.43 s (FlashInfer) and 0.58 s (sdpa), median, measured next
    to the LLM;
  - put laughs at the start of a sentence;
  - strip markdown and list markers from LLM output before streaming ("1. Pizza" splits like a sentence);
  - base-model auto and designed voices vary between sentences (#255);
  - Pipecat users can keep SpokenSentences, so each call is usually one item;
  - `pad_duration` and `fade_duration` are the only join knobs;
  - the difference from PR #166, which yields cumulative audio plus a status; this stream yields each sentence's audio
    plus its text.
- FlashInfer and int8 recipes, plus the warm-up call at startup. Their table is marked "measured on an emulation of
  the library, to re-measure" until G4-G9 run on the library code (bench2, RTX 5090, 32 steps; resident includes
  the FlashInfer runner):

  | | resident GiB | peak GiB, ≤ 6 s / 20 s text | nvidia-smi MiB | RTF | first sentence of a typical reply |
  |---|---|---|---|---|---|
  | sdpa fp16 | 1.23 | 1.33 / 1.56 | 2,014 | 0.053 | 240 ms |
  | sdpa int8 | 0.82 | 0.92 / 1.15 | 1,612 | 0.057 | 268 ms |
  | FlashInfer fp16 | 1.36 | 1.46 / 1.69 | 2,154 | 0.023-0.024 | 103 ms |
  | FlashInfer + int8 | 0.95 | 1.05 / 1.28 | 1,754 | 0.031 | 119 ms |

  - When to use which:
    - FlashInfer fp16 is the fastest, and the best neighbor for an LLM on the same GPU.
    - int8 trades 0.41 GiB for +8 % (sdpa) or +31-34 % (FlashInfer) time per call, and costs a co-located LLM more
      of its decode. Use it only when GPU memory is the constraint.
  - Both int8 configs passed a blind listening on KohAI: no difference heard in 19 of 20 pairs, and int8 preferred
    in the other.
- Next to an LLM and faster-whisper on the same GPU (kohai-v3's case):
  - The processes share the GPU by time-slicing (inferred). A TTS call takes 1.12-1.34× as long while the LLM
    decodes, and the LLM keeps 23-34 % of its decode speed while the TTS runs. FlashInfer fp16 costs the LLM the
    least per sentence.
  - Run faster-whisper in its own process. Under back-to-back transcription with the LLM idle, a TTS call then
    takes 1.11-1.21× as long, against 2.5-2.9× on a thread of the TTS process; next to a decoding LLM this was not
    measured.
  - If it must share the TTS process, a sentence that overlaps an utterance takes 1.2-1.5× as long with default
    settings. Under back-to-back transcription, `OPENBLAS_NUM_THREADS=1` in the launch environment or a
    high-priority CUDA stream for the TTS brings 2.5-2.9× down to 2.0-2.4× (sdpa and FlashInfer fp16, each alone;
    the two were not measured together).
  - Keep CPU-heavy Python work out of the TTS process.
  - These are caller settings; the library sets none (§11).
- Host memory: the codec decoder's cuDNN execution plans grow host RSS by about 0.7 MiB per new target length.
  - The growth is bounded, at about 320 MiB over every length from 1 to 20 s.
  - `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT`, a caller setting for the launch environment, caps it: with 270, it costs
    +0.02-0.21 ms per call.
- Threads and the global-state contract (§6, §7, §8, §11).
- "Never .half()".
- Numbered differences from upstream, including:
  - `normalize_text=` and `omnivoice.utils.text.normalize_text()` become `TextNormalizer(cache_dir).normalize(text,
    language)`, called before `generate()`; `language` is required, and `None` no longer guesses zh or en;
  - TN covers Spanish only, with NeMo plus es_rules; the zh/en WeTextProcessing path and num2words are gone;
  - TN errors propagate.
- KohAI example as caller config. First prebuild the grammar once, in its own process:
  `python -c "import os; from omnivoice_tts.utils.text import TextNormalizer as T; T(os.path.expanduser('~/.cache/kohai/omnivoice-tn')).normalize('1', 'es')"`

  ```python
  import os
  import re

  import torch

  from omnivoice_tts import OmniVoice
  from omnivoice_tts.utils.text import TextNormalizer

  model = OmniVoice.from_pretrained("kohai-channel/kohai-omni-tts", device_map="cuda:0", dtype=torch.float16)
  # sdpa keeps this short. Next to an LLM on the same GPU, use FlashInfer fp16 (the recipe above), as kohai-v3 does.
  tn = TextNormalizer(cache_dir=os.path.expanduser("~/.cache/kohai/omnivoice-tn"))  # one per TTS thread
  # Load the prebuilt grammar now, not in a reply. A process that builds it itself stays ~4x slower at TN
  # (py3.12, pynini 2.1.6.post1).
  tn.normalize("Son las 21:30.", "es")

  # KohAI's config, not the library's: duration-rule weight per second of KohAI speech, per emotion,
  # fitted on spoken-form (normalized) labels, so it applies to normalized text only.
  RATES = {"happy": 14.11, "neutral": 13.14, "angry": 15.09, "sad": 12.92, "afraid": 14.73,
           "surprised": 12.34, "confused": 12.89, "shy": 12.84, "melancholic": 15.4}
  TAGS = re.compile(r"\[[a-z-]+\]")
  weight = model.duration_estimator.calculate_total_weight


  def kohai_duration(spoken, emotion):
      return max(1.0, weight(spoken) / RATES[emotion] + 1.2 * spoken.count("[laughter]"))


  def kohai_stream(reply, emotion="neutral", generator=None):
      """(audio, subtitle) for each sentence of one LLM reply (its <emotion> tag already removed)."""
      for audio, sentence in model.generate_streaming(
              reply, "es", instruct=None if emotion == "neutral" else emotion,
              normalize=tn.normalize,                                # "21:30" -> "veintiuna y media"
              duration=lambda spoken: kohai_duration(spoken, emotion),
              generator=generator):
          yield audio, " ".join(TAGS.sub("", sentence).split())     # what the LLM wrote, without tags


  def kohai(text, emotion="neutral"):
      """One take for a whole text, or None when normalization leaves nothing to say ("<3")."""
      spoken = tn.normalize(text, "es")
      if not any(c.isalnum() for c in spoken):
          return None
      return model.generate(spoken, "es", instruct=None if emotion == "neutral" else emotion,
                            duration=kohai_duration(spoken, emotion))[0]
  ```
- Licenses, tips, citation, disclaimer.

## 16. Bug fixes in place

1. Newlines glue words (M:1706): a space.
2. Near-silent clip crashes (M:901): empty array.
3. Unknown config fields dropped silently; kwargs ignored with a generation_config (M:191-195, 677-681):
   `dataclasses.replace`.
4. Unknown language becomes None with a warning: ValueError.
5. `audio_chunk_duration <= 0` still splits (M:210-214 vs doc M:657): single pass.
6. Wrong `postprocess_output` docstring (M:652).
7. FA2/flex flags allow unsupported attention (M:279-280): deleted; `attention_mask` required.
8. revision/token not forwarded; codec fallback unpinned (M:270-275, 345, 354-357): §5.
9. Unspeakable text runs anyway ('' → 1 frame, '...' → 18, '' with duration > 30 s → ZeroDivisionError at M:932):
   ValueError.
10. Speed applied twice to chunks ≥ 1 on text above 30 s with duration/speed: chunks ≥ 1 use speed 1.0. Bit-identical
    when duration and speed are None.
11. Upstream TN is unusable for Argentine Spanish: with num2words it gives "$tres.quinientos" and "ciento uno
    personas", without num2words it does nothing, and language=None goes to English. Fix: es goes through NeMo plus
    es_rules, and every other language passes through. Tests: `test_only_spanish_is_normalized` and the PR #6 suite.
12. Upstream TN's failure warning reaches stderr through `logging.lastResort` (T:352-355). Fix: errors propagate.
    Test: `test_tn_is_silent_and_global_free`.

**Not fixed in v0.1** (documented)
- `generate()`'s >30 s chunker can split inside '21:30', '3.5' or '1,000' at a chunk boundary: upstream's
  `chunk_text_punctuation` splits at every . , ; : ! ? without a whitespace check. `generate_streaming`'s splitter
  never does, though a stream sentence above 30 s still goes through this chunker. Spanish text normalized first has
  no digits left to split on the KohAI corpus; 2 of the 485 held-out number sentences keep them (`localhost:5000`,
  `10²`; §8).
- 'Sra.' is missing from `ABBREVIATIONS`, which `generate()`'s >30 s chunker uses. The stream splitter has its own
  Spanish list (§7).
- A short, boosted chunk 0 inflates later chunks when `audio_chunk_duration` is lowered.
- The splitter ceilings listed in §7.

**Kept on purpose** for bit-identity: the int16 pydub round trip and the always-on 0.5 peak normalization.

## 17. Bit-identity

- Identical to upstream `generate()` (sdpa, `from_pretrained(dtype=fp16)` with fp32 inv_freq, fp32 codec, default
  config, global RNG) for any single speakable text without CR/LF.
- `generate_streaming(text)` on text without a sentence break yields exactly `generate(text)[0]`. With several
  sentences, the k-th item is `generate()` of the k-th sentence that is not skipped, with the RNG left by the items
  before it.
- TN is a step the caller runs, so `generate()`'s bit-identity is untouched. TextNormalizer's output:
  - equals indextts25-inference PR #6 (dfb3286f) on all 34,090 KohAI sentences and on its 630 test cases, on py3.12 +
    pynini 2.1.6.post1 and on py3.13 + 2.1.7;
  - reproduces all 5,350 kohai_5h training labels;
  - leaves the 1,086 tagged kohai_r2 labels unchanged;
  - on text with `[...]` spans, equals upstream's own `_apply_with_protection` on 148 probes.
- Pending GPU confirmation: private generator seeded s equals the global seed s (G2); vendored FlashInfer equals
  upstream FlashInfer in eager mode (G5).
- Different by design:
  - CR/LF texts;
  - inputs upstream rejected or crashed on;
  - text with nothing to speak (fix 9), and language strings upstream mapped to None with a warning, such as 'ES' or
    unknown names (fix 4);
  - `audio_chunk_duration <= 0`;
  - kwargs with a generation_config;
  - chunked text with duration or speed (fix 10). This includes a short text whose duration exceeds 30 s, which
    upstream also chunks;
  - FlashInfer against sdpa, and int8 against fp16 (the two int8 forward forms of §10 are bit-identical to each
    other on the 12 bench lines);
  - streaming of multi-sentence text (§7).

## 18. Tests (CPU, in the repo; upstream outputs frozen as fixtures)

Files: `tests/test_loader.py` (loader), `tests/test_generate.py` (call sequences, RNG, bug regressions),
`tests/test_split_sentences.py`, `tests/test_streaming.py`, `tests/test_text_normalizer.py` (every TN test except the
PR #6 suite), `tests/test_es_normalization.py` (the PR #6 suite), `tests/test_int8.py`, `tests/test_flashinfer_cpu.py`.
Fixtures go in `tests/data/`, copied from the evidence directory: the splitter's 68 hand cases
(`stream/scratch/segmenter/scripts/cases.py`), the 109 tag probes (`tn/scratch/tn-critic/out/inputs_probes.json`) and
int8's per-layer sha256 list (`bench2/perf/raw/listen_B.json` `int8.per_layer_sha256`, 28 layers × 8 hashes).

**Loader**
- `test_loader_matches_transformers_from_pretrained`: a tiny Qwen3 OmniVoice; parameters, buffers and the 2-row CFG
  logits are bit-identical; inv_freq is fp32; `.half()` differs.
- `test_load_is_silent_and_global_free`: capfd at fd level; a spy on `set_default_dtype`; the RNG, os.environ,
  warnings.filters, the logging disable level and CONFIG_MAPPING are unchanged; the snapshot is taken after importing
  the dependencies.
- `test_decode_only_codec_matches_full`, skipped unless eustlb@528e871c is cached.
- `test_tokenizer_needs_no_auto_registry`.

**Call sequences and RNG:** `test_generate_matches_upstream_call_sequence`, `test_generator_reaches_every_chunk`,
`test_private_generator_matches_global_seed`, `test_generator_leaves_global_rng_untouched`.

**Bug regressions:** `test_newline_becomes_space`, `test_combine_text_unchanged_without_newlines`,
`test_near_silent_output_returns_empty`, `test_unknown_generate_kwarg_raises_typeerror`,
`test_kwargs_override_generation_config`, `test_unknown_language_raises`, `test_language_id_case_insensitive`,
`test_text_without_speech_raises`, `test_list_text_raises_typeerror`, `test_positional_after_language_raises`,
`test_audio_chunk_duration_zero_disables_split`, `test_from_pretrained_rejects_upstream_kwargs`,
`test_instruct_normalised_like_upstream_and_never_rejected`, `test_later_chunks_not_sped_up_twice`,
`test_hub_kwargs_forwarded_and_codec_pinned` (fix 8: a spy on snapshot_download sees revision, token, cache_dir and
local_files_only reach the model repo, and the codec fallback uses revision
528e871c2a26c4f0f7773b9754e2e1acae20899d), `test_forward_requires_attention_mask` (fix 7), and
`test_generate_chunked_matches_upstream` (identical calls with speed and duration None; fix 10 asserted with
duration=45 s). `test_from_pretrained_rejects_upstream_kwargs` includes `attn_implementation` (fix 7).

**Splitter** (`tests/test_split_sentences.py`)
- `cases`: the 68 hand cases plus three critic cases, every case lossless:
  - 'Bueno... ¿qué querés que te diga?' stays one piece;
  - 'Bueno... "no sé" dijo.' stays one piece;
  - '¿Qué hacés? ¡Hola! Todo bien.' gives 3 pieces.
- `single_sentence_unchanged`: '  Hola  ', '', '...', '[laughter] jaja' and the like come back as `[text]`.
- `fuzz`: 20,000 strings, seed 20261020; the parts include '¿qué', '¡no', '"sí' and '(y'.
- `upstream_chunker_untouched`: frozen fixtures.

**Streaming** (`tests/test_streaming.py`; `_generate_iterative` and decoding are stubbed)
- `one_sentence_is_generate`: the same calls; the item equals `generate(text)[0]`, and the text matches byte for byte.
- `sentences_are_independent_generate_calls`: with the global seed and with one shared generator; item 1 differs from
  a freshly seeded `generate(s_1)`.
- `hooks_order`: normalize gets (the stripped raw sentence, 'es' when 'Spanish' is passed); duration gets normalize's
  output; frames = `max(1, int(d*25))`; the yielded text stays raw.
- `duration_must_be_callable`: 3.0, 3, np.float64(3.0) and True each raise TypeError before any call.
- `config_fields_checked_first`: `num_steps=8` and `normalize_text=True` raise TypeError before the split or
  normalize.
- `lazy_and_closes`: creation runs nothing; `close()` before or after the first item; dropping the stream.
- `error_mid_stream`: an error in sentence 2 comes out of the second `next()`; item 1 stays valid; the stream is
  finished.
- `without_speech_raises`: '', '...', '😀' and '¿?' raise ValueError at the first `next()`, before normalize is
  called.
- `lone_tag_is_spoken`: '[laughter]' gives one item equal to `generate('[laughter]')[0]`.
- `skips_sentences_tn_empties`: a stub normalize that returns '' for the middle one of three sentences gives two items
  with the first and third raw sentences; generate and duration run twice. When it empties every sentence, the stream
  yields nothing and generate never runs.
- `rejects_non_str`.
- `caller_outside_inference_mode`.
- `normalize_keeps_raw_subtitles`: skipped without nemo; HOME and XDG_CACHE_HOME in tmp.

**Text normalization** (skipped without nemo-text-processing unless marked `*`)
- `tests/test_es_normalization.py`: 630 normalization cases plus 17 `_spell` cases, with idempotence on every case;
  optional `OMNIVOICE_TN_CACHE`.
- `test_tags_survive_byte_for_byte`, parametrized over the 109-probe set. For each probe, the list of `[...]` spans
  and the list of non-verbal tags are unchanged, no digits are left outside `[...]` spans and IndexTTS `<x|y>`
  annotations, and the output is idempotent.
  Exact strings:
  - `'Ese [B EY1 S] cuesta $450.'` → `'Ese [B EY1 S] cuesta cuatrocientos cincuenta pesos.'`
  - `'[laughter]21:30 arrancamos.'` → `'[laughter]veintiuna y media arrancamos.'`
  - `'  Hola [laughter]  21 años  '` → `'  Hola [laughter]  veintiún años  '`
- `test_only_spanish_is_normalized`: 'en', 'fr', `None`, 'spa' and 'es-AR' come back unchanged; 'ES', 'Spanish' and
  ' es ' are normalized; no language raises TypeError.
- `*test_non_spanish_needs_no_nemo`: with `sys.modules['nemo_text_processing'] = None`, 'en' and `None` come back
  unchanged, and 'es' raises an ImportError whose message contains `omnivoice-tts-inference[tn]`.
- `*test_cache_loads_only_own_private_pickle`, with a fake Normalizer in `sys.modules` and
  `omnivoice_tts.utils.text.version` monkeypatched to a constant (the pickle name reads both distributions' metadata,
  which a fake module lacks):
  - planted 0664 and 0646 payloads and a symlink to a payload are never unpickled, and each is replaced by a regular
    0600 file;
  - a corrupt 0600 file is rebuilt;
  - our own 0600 cache loads without a rebuild;
  - a library-created directory is 0700;
  - an unwritable `cache_dir` raises OSError before any build (the fake build is never called);
  - no `.tmp` file is left behind.
- `test_tn_is_silent_and_global_free`: a `python -I` subprocess with a cold, then a warm temporary `cache_dir`.
  - fd 1/2 stay empty.
  - After the app sets NeMo to ERROR, the snapshot is unchanged: NeMo's level, handlers and filters, the root
    handlers, `os.environ`, the recursion limit and `warnings.filters`. The snapshot is taken after NeMo's import.
  - The pickle is 0600.
- `*test_importing_text_does_not_import_nemo`.
- `*test_es_rules_is_the_indextts25_copy`: the sha256 pin.

**Optional modules**
- `test_int8.py`: round-trip bound ≤ 0.63·scale (0.561 measured on KohAI), scope, idempotence, the formula, equals
  slim.py.
  - On a tiny model, `qweight` and `scale` equal `round(w.float() / s)` and `s.to(dtype)`, with `s` computed in fp32
    as §10 says.
  - "Equals slim.py" means slim.py run on CUDA. slim.py's `int8_layers` on the CPU is not a valid reference: it
    differs in 3,209 values on KohAI.
  - When the KohAI checkpoint is cached, the per-layer sha256 list equals bench2's `perf/raw/listen_B.json`
    `int8.per_layer_sha256` (top-level d8e4b4726f65…, hashed as in bench2's `perf/perf.py`).
  - Do not use f0dfc9af…: that is slim.py run on CPU.
- `test_flashinfer_cpu.py`: skipped without flashinfer; HOME/TMPDIR in a temp dir; fusion frees the originals and the
  fused weights have no `grad_fn`, 4 Linears per layer, the guards leave the model untouched, the ImportError text,
  FlashInfer then int8 (4 Int8Linear per layer, and no fused weight left as a plain tensor attribute; G9 counts the
  forward calls on the GPU).

**GPU** end-to-end tests are gated by environment variables; `scripts/benchmark.py`.

**One-off TN scripts** stay out of CI, in the evidence directory's `tn/scratch/tn-critic/scripts/`. Re-run them at PR
#6's merge commit: run_impl.py with compare.py (corpus, PR #6 cases, probes, fuzz, tagged labels), silence.py,
cache_guard.py, errors_check.py and duration_check.py. Also time `TextNormalizer.normalize` after the library's own
cache load, in a fresh process. It should match the pickled-grammar timing of §8: 11.9 ms median over the 30 measured
replies and their 57 sentences; for 2-sentence replies, 7.6 ms for the first sentence and 17.2 ms for the whole reply.

## 19. GPU checks (only with the user's OK; the GPU is shared)

Rule for every timing check: torch.profiler runs last or in its own process. One CUDA profiler session slows every
later call in the same process (+26 % on sdpa, +14 % on FlashInfer, measured).

**Core**
- G0 FlashInfer arch on cu128 + 12.0a and on cu129 + 12.0f; without the env var, the arch RuntimeError before
  patching.
- G1 default parity: int16 PCM sha256 against upstream, step by step (loader, codec, global-state removals, str-only
  plumbing, generator threading, MODEL_MAPPING), on the 12 bench lines (KohAI duration rule), 2 newline-free texts over
  30 s (duration=None) and 1 CJK text. The reference is upstream run on the KohAI export (`kohai-omni-tts-fp16`), never
  the model card's eval wavs. Those come from an in-memory fp16 merge that differs from the export in 196 tensors (by
  at most 0.00195), so 0/124 match.
- G2 private generator == global seed; CUDA RNG untouched with a private generator; a CPU generator raises.
- G4 sdpa memory and latency on the library: the load peak without the 806 MB transient; ~1.33 GiB at 32 steps;
  reserved; nvidia-smi next to llama-server and faster-whisper; load time; RTF at 32 and 16 steps. Upstream's loader
  measured 1.89 GiB after load and 2.24 GiB peak on real replies, and RTF 0.0559 at 32 steps (§7) with the KV cache
  on. With `use_cache=False` and the decode-only codec, the lib-like emulation measures 0.0530-0.0535 in three fresh
  processes, the library's expected figure. G10's 0.0514 came from one 3-pass process.

  Emulation done (bench2):
  - 1.3313 GiB M0 peak and 1.561 GiB at 20 s, with no load transient;
  - nvidia-smi 2,014 MiB alone, and a 24,211 MiB GPU total next to llama-server and faster-whisper;
  - load to ready 0.87 s, cold call 0.54 s.

  Open: the library code.
- G5 FlashInfer parity with upstream FlashInfer (eager) and isolation (two FI instances alternating, plus one sdpa).
  Emulation done (bench2): C built CPU-first equals upstream FlashInfer patched on the GPU 12/12, and C in the
  vendored module layout equals C 12/12. Open: the library code and the isolation check.
- G6 FlashInfer memory with the CPU-first recipe. Expect a peak of about 1.46 GiB allocated, against 2.01 GiB for
  upstream FlashInfer with a decode-only codec (INSTRUCTIVO.md) and a 2.92 GiB peak (2.57 GiB after load and patch)
  with upstream's full loader (§9). Emulation done (bench2): 1.4641 GiB on the bench lines (the estimate holds) and
  1.693 GiB at 20 s. Patching on the GPU instead leaves 1.932 GiB reserved against 1.366 GiB allocated. Open: the
  library code.
- G7 FlashInfer CER/SIM against sdpa, RTF on the library (upstream: 0.0243 at 32 steps), and cold start (CuTe-DSL
  RMSNorm JIT; upstream's first call took 0.65-0.89 s). Emulation done (bench2): undetectable against sdpa on the
  124-line eval (13 metrics); RTF 0.0233-0.0239; cold call 0.605-0.614 s. Open: the library code.
- G8 int8: the checks are
  - equality with slim.py's `int8_layers` run on CUDA: 196 modules, 392 qweight/scale tensors, the per-layer
    sha256 list named in §18 and the same PCM hash;
  - forward calls counted, not modules: 196 per `model.llm` forward;
  - VRAM and RTF;
  - the paired fixed-seed eval on 124 lines;
  - the blind listening (the adoption gate).

  Emulation done (bench2): 6,272 calls per 32-step `generate()`, a 0.922 GiB peak, RTF 0.0572 (§10's forward), and
  an eval that is undetectable on 13 metrics with one seed.

  Blind listening done (block 1, A vs B, in the listening gate below): no difference in 9 of 10 pairs, and int8
  preferred in the other.

  Open: a second seed, and the library code.
- G9 FlashInfer then int8: VRAM (~1.05 GiB for texts up to ~6 s), RTF, CER/SIM, and 112 Int8Linear calls per
  `model.llm` forward.

  Emulation done (bench2): 1.0545 GiB peak (1.284 GiB at 20 s), RTF 0.0311-0.0312, undetectable against A and
  against C, and 3,584 calls per 32-step `generate()`.

  Blind listening done (block 2, C vs D, in the listening gate below): no difference in 10 of 10 pairs.

  Open: a second seed, and the library code.
- G10 `use_cache=False` in the sdpa forward. Measured on upstream code: int16 PCM sha256 identical 36/36 (12 bench
  lines × 3 passes), RTF 0.0551 → 0.0514, peak allocated unchanged at 2.00 GiB. Adopted (§6). Still to do: compare
  against G1's library hashes. The lib-like A equals upstream sdpa 12/12, with the KV cache on and off.
- G11 silence and global state on GPU (fd-level capture; env, filters, logging, default dtype, CUDA RNG, torch thread
  counts).
- G12 concurrency with faster-whisper on another thread. Emulation done (bench2 contention). The TTS call slows by:
  - 1.21-1.51× with one transcription per call;
  - 2.49-2.89× with transcriptions back to back;
  - 3.31-3.51× with the LLM decoding as well;
  - 1.11-1.21× with faster-whisper in its own process.

  The cost is on the CPU side: dispatch runs 1.9-2.7× slower. The outcome is README advice (§15), not library
  code.
- G13, by ear before the release: `generate()` on 2-3 KohAI texts over 30 s with the KohAI duration rule (fix 10
  changes nothing when duration and speed are None). Render each with fix 10 and with upstream's timing for chunks ≥ 1
  (`speed=speed`); chunks ≥ 1 are prompted. If the ear prefers upstream's timing, D6 goes back to the user.
- G14 soak: 1,000 calls per config. Emulation done (bench2 soak): no GPU memory growth, no drift, 100/100
  determinism and 0 errors. Host RSS grows +157-160 MiB from cuDNN plans (bounded, §15). Open: the library code, an
  hours-long run and a soak next to the LLM.

**Streaming** (G3 is split into G3a-G3e)
- G3a single-sentence parity on the library code, on sdpa fp16 and on FlashInfer (about 1 min).
  - For the 11 bench lines that `_split_sentences` keeps whole, the one item of
    `generate_streaming(t, 'es', instruct=e, duration=dur)` has the same int16 PCM sha256 as
    `generate(t, 'es', instruct=e, duration=dur(t))[0]`. Check it with `torch.manual_seed` and with a private
    `torch.Generator('cuda:0')`.
  - The 9th line ('¿En serio me regalaste una suscripción? No te puedo creer, sos un divino.') yields two items; it
    goes under G3b.
  - Parity already holds at the `generate()` level on upstream code (50/50 reps per backend).
- G3b multi-sentence parity. On 5 real KohAI replies of 2-3 sentences, each item's hash equals the matching output of
  a `generate()` loop over the stripped sentences, with one shared generator. With `normalize=tn.normalize`, no digit
  reaches the model.
- G3c step cost against length, sdpa and FlashInfer. Done (§7): flat per-step cost, launch-bound at short lengths;
  the knee is at L > 320 on sdpa and at P ≈ 320 on FlashInfer.
- G3d time to first item and GPU time per reply on real replies, sdpa and FlashInfer, with the TN grammar warm. Done
  without the LLM (§7) and next to llama-server (bench2, with no TN in the pipeline): the TTS takes 1.12-1.34× per
  call (1.31× for D with §10's forward), the LLM keeps 23-34 % of its decode, the real pipeline's first audio comes
  at 428-638 ms (medians, A-D; B and D ran slim.py's int8 forward), and there are 0 underruns. Open: TN on the TTS
  thread next to the LLM (G-TN6), and the library code.
- G3e close and cancel on the real model. After `close()` following the first item, the next stream reproduces its
  solo hashes. The asyncio pattern (one worker thread, `close()` queued on it) works under `task.cancel()`.
- Listening, a release gate approved with part 3: the user listens before the release.
  - The sets: 5 multi-sentence KohAI replies, the stream against `generate()` on the whole reply, and 3 lone short
    sentences with the 1.0 s floor; optionally `pad_duration` 0.05 against 0.1.
  - The first two sets are rendered on both backends: 13 wavs each, in `rtf/scratch/listen/{sdpa,fi}/` of the evidence
    directory and on a private listening page.
  - New sets (bench2), on the private listening page:
    - the int8 takes of the same replies and short sentences (`bench2/perf/listen/{sdpa_int8,fi_int8}/`, 13 wavs
      each);
    - the blind set (`bench2/quality/blind/`): block 1 A vs B, block 2 C vs D and block 3 A vs C, 10 pairs each. The
      set's own `key.json` puts D in the `_x.wav` file of 8 of block 2's 10 pairs. The page re-draws the sides so
      that in each block 5 of the 10 pairs play the block's reference (A, C, A) first; its key is
      `bench2/blind_page/key_page.json`, and the page saves the picks itself. Done 2026-10-21: no difference in 28
      of the 29 pairs answered (one pair of block 3 left blank), and sdpa int8 preferred in the other;
    - G-TN3/4's 18 takes, and G-TN2's five fastest pairs.
  - Done 2026-10-21 (decision #24): everything sounded right to the user, so it ships as designed. The README does
    not recommend `pad_duration=0.05`, and the minimum-length merge stays under §7's "Not in v0.1".

**Text normalization**
- G-TN1 (main). Normalize the 485 held-out number sentences (work/numbers/test.jsonl) with
  `TextNormalizer(cache_dir).normalize(digits, 'es')` and render them through v5r2b_576 with the existing num_eval.sh /
  render_eval.py / score_asr_sim.py / num_score.py pipeline, with the KohAI duration rule on the normalized text. Pass
  if CER and ok ≤ 5 % are close to the 'palabras' arm (0.013, 93 %, dur/ref 1.00) and far from 'dígitos' (0.065,
  53 %, 0.93). Done (bench2 tn, with PR #6's TN, whose output equals TextNormalizer's on these 485). It passed by the
  rule of decision #22: CER 0.0287 and ok ≤ 5 % 81.9 % lie 0.30 and 0.28 of the way from 'palabras' to 'dígitos'
  (the limit is 1/3), and dur/ref is 0.990.
- G-TN2 (duration basis). Render the same normalized text with dur(raw) and with dur(normalized), to confirm in audio
  the 0.938 median timing skew behind the "no flag" decision. Done: the raw-text rule gives audio 0.942× as long and
  speech 6.5 % faster (20 % in the third of sentences more than 10 % short). SIM drops by 0.0075 and CER is
  unchanged; the most skewed sentence breaks (CER 0.425).
- G-TN3 (listening: tags next to numbers). '[laughter]21:30 arrancamos.', 'Costó $2.800[sigh], ¿viste?', 'Tengo 21
  [laughter] años.' (read as veintiuno) and '¿En serio?[question-ah] ¡Tenés 31 años!': each tag is voiced and the
  number sounds natural. Rendered (3 seeds each): in Whisper's transcripts the numbers are right 12/12, and
  `[question-ah]` is voiced 3/3. The user heard the laughs and the sigh right (2026-10-21).
- G-TN4 (pronunciation overrides after TN). 'Ese [B EY1 S] cuesta $450.' and '[HH AH0 L OW1] chat, son las 9:05.':
  the override still applies and the number is spoken. Rendered (3 seeds each): in Whisper's transcripts the numbers
  are right 6/6, and both overrides apply 3/3. The user heard them right (2026-10-21).
- G-TN5 (long text). A Spanish text over 30 s with many numbers: normalized and then through `generate()`, and raw
  through `generate_streaming(normalize=tn.normalize)`. No chunk boundary falls inside a number reading, and the stream
  items carry the raw sentences.
- G-TN6 (kohai-v3, after adoption). First-audio latency with TN on the TTS thread (about 3-22 ms per sentence on CPU;
  pynini holds the GIL). The Pipecat event loop does not stall, and the warm-up in `load()` pays the grammar load.

## 20. Risks and limitations

- The loader relies on transformers internals (`llm.rotary_emb` and its constructor, Higgs submodule names); parity is
  proven on transformers 5.20 CPU only. The tiny-model parity test runs in CI on the floor and latest transformers.
- The KohAI LoRA never trained with prompts (prompt_ratio_range [0,0]); prompted chunks are unvalidated (G13). Stream
  sentences are never prompted.
- The cu128 FlashInfer route is inferred from cubins and code reading (G0).
- int8 quality rests on one seed of the 124-line eval and one blind listening of 20 pairs (G8, G9); a second seed is
  open. int8 is also slower on every call (+8 % on sdpa, +31-34 % on FlashInfer, with §10's forward), and it costs a
  co-located LLM more of its decode speed.
- Sharded checkpoints are unsupported.
- Typos in stock voice-design words reach the model silently (user decision).
- Base-model auto and designed voices can change between stream sentences (#255); KohAI's fine-tuned voice is fixed.
- Up to about 10.6 s (sdpa) or 5.7 s (FlashInfer) of audio, an fp16 call costs a fixed floor set by CPU dispatch.
  With int8 the GPU becomes the limit sooner: past about 7.6 s on sdpa, and almost from the start on FlashInfer.
  - Next to llama-server that floor holds: dispatch stays within 2 % of alone (A and C, 1.5-5 s).
  - The two processes most likely time-slice the GPU instead (inferred; no MPS daemon ran). A TTS call takes
    1.12-1.34× as long, and the LLM keeps 23-34 % of its decode speed while they overlap (G3d, measured).
  - The CPU-side risk is inside the TTS process. faster-whisper on a thread costs 1.2-1.5× per overlapping call and
    2.5-2.9× back to back (G12), and a busy pure-Python thread was once seen starving it (not verified).
- Host RSS grows with each new target length (cuDNN plans for the codec decoder), bounded at about 320 MiB (§15).
- GPU-bound timings depend on the session: the desktop's graphics load moved them by up to 9 % between bench2
  sessions. Ratios within one session hold.
- The two es_rules copies can drift until indextts25-inference stops owning the file (§8).
- Spanish TN follows es-AR conventions for all 'es' text. In audio it passes G-TN1 narrowly: CER 0.029, against
  0.013 for hand-written words. At least 44 of 485 held-out number sentences hit real TN errors or gaps, and 2 keep
  digits (§8).
- The `[tn]` extra needs a C compiler and adds about 0.7 GB; it is Linux x86_64 and CPython < 3.14 only.
