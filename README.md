# omnivoice-tts-inference

Inference-only [OmniVoice](https://github.com/k2-fsa/OmniVoice), k2-fsa's massively multilingual text-to-speech
model, packaged from k2-fsa/OmniVoice at `08be0b4`. It speaks with an auto voice or a designed one (`instruct`),
streams sentence by sentence with each sentence's text, normalizes Spanish text, and has optional FlashInfer and int8
modes.

It keeps OmniVoice's own API, trimmed to inference: `OmniVoice.from_pretrained()` and `generate()`, plus a new
`generate_streaming()`. Voice cloning (so the audio codec loads decode-only), batch input, ASR, training, the CLI and
the LoRA utilities are left out. Upstream bugs are fixed in place; otherwise, with default settings, `generate()` gives
upstream's audio bit for bit (see [Differences from upstream](#differences-from-upstream)). Settings that belong to one
model, such as a fine-tune's emotions or speaking rates, are the caller's config: the library never checks them.

## Install and quick start

```bash
pip install omnivoice-tts-inference
```

Python 3.10 or newer. FlashInfer is a separate install ([FlashInfer and int8](#flashinfer-and-int8)).

```python
import torch

from omnivoice_tts import OmniVoice

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
model.generate("Warm-up.", "en")  # once at startup

audio = model.generate("Hello, this is a test of OmniVoice.", "en")[0]
```

`generate()` returns a list with one 1-D float32 array at `model.sampling_rate` (24 kHz). Save it with, for example,
soundfile (not a dependency): `soundfile.write("out.wav", audio, model.sampling_rate)`.

- **One warm-up call at startup.** The first `generate()` of a process takes 0.55-0.56 s on sdpa and 0.65-0.89 s on
  FlashInfer (RTX 5090, measured on upstream code). One call at startup removes that cost from the first request.
- **Never call `.half()` or `.to(dtype)` after loading:** they also cast RoPE's fp32 `inv_freq` buffers and the
  codec. Pass `dtype=` instead. Moving with `.to("cuda:0")` is fine.
- `device_map` is one device (`"cuda:0"`, `0` or a `torch.device`). `None` means CPU; `"auto"`, the other
  multi-device maps and dicts raise `ValueError`. `dtype=None` keeps the config's dtype, which is float32 for
  k2-fsa/OmniVoice, so pass `dtype=torch.float16`.
- The path is a local directory, used as is, or a Hub repo id. Besides `device_map` and `dtype`, `from_pretrained()`
  takes only `revision`, `token`, `cache_dir` and `local_files_only`. They also reach the codec download that a
  checkpoint without `audio_tokenizer/` falls back to (`eustlb/higgs-audio-v2-tokenizer`, at a pinned revision). Any
  other keyword raises `TypeError`. The checkpoint must be a single `model.safetensors`.
- `language` is an ID (`"es"`, `"ES"`) or a name (`"Spanish"`) from `model.supported_language_ids()` or
  `model.supported_language_names()`, matched case-insensitively, or `None` for none. An unknown one raises
  `ValueError`. Upstream notes that naming the language works slightly better.
- Text with nothing to speak (no letter or digit, such as `""` or `"..."`) raises `ValueError`. `"[laughter]"` alone
  is speech.

## Generation parameters

```python
from omnivoice_tts import OmniVoiceGenerationConfig

audio = model.generate(text, "en", num_step=16)[0]
config = OmniVoiceGenerationConfig(num_step=16)
audio = model.generate(text, "en", generation_config=config, postprocess_output=False)[0]
```

`generate()` and `generate_streaming()` take the fields of `OmniVoiceGenerationConfig` as keywords or as a
`generation_config`. Keywords override the fields of a passed config (on a copy), and an unknown name raises
`TypeError`. The fields and their defaults are `num_step` 32 (16 is faster), `guidance_scale` 2.0, `t_shift` 0.1,
`layer_penalty_factor` 5.0, `position_temperature` 5.0, `class_temperature` 0.0, `denoise` True, `postprocess_output`
True, `audio_chunk_duration` 15.0, `audio_chunk_threshold` 30.0, `pad_duration` 0.1 and `fade_duration` 0.1.
[docs/generation-parameters.md](docs/generation-parameters.md) describes each.

- `postprocess_output` shortens pauses longer than 1 s to 1 s and trims leading and trailing silence to 100 ms. Peak
  normalization to 0.5 and the fade/pad always run.
- `duration` (seconds) and `speed` (above 1 is faster) are arguments of `generate()`, not config fields. `duration`
  overrides `speed`.
- Text whose estimated audio is longer than `audio_chunk_threshold` seconds is generated in chunks of about
  `audio_chunk_duration` seconds. The first chunk is generated on its own, and each later one is prompted with the
  first chunk's text and audio. `audio_chunk_duration <= 0` disables chunking.

## Voice design

```python
audio = model.generate("This is a test of voice design.", "en", instruct="female, low pitch, british accent")[0]
```

`instruct` describes the speaker with comma-separated attributes: gender, age, pitch, whisper, an English accent or a
Chinese dialect, in English or Chinese ([docs/voice-design.md](docs/voice-design.md) lists them). Without it, the model
picks a voice itself.

The library normalizes `instruct` as upstream does: it splits on `,` or `，`, lowercases each item and unifies English
and Chinese items to one language. It never rejects it, though. Unknown items pass through lowercased, and typos get
no suggestion (`"femal"` reaches the model as it is).

Voice design is trained on Chinese and English data only. Upstream notes that it can generalize to other languages,
but may be unstable for some low-resource languages or edge cases.

## Non-verbal tags and pronunciation

Thirteen non-verbal tags can go inline: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`,
`[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`,
`[surprise-yo]` and `[dissatisfaction-hnn]`. Each tag is tokenized on its own, whatever the language around it.

```python
audio = model.generate("[laughter] You really got me. I didn't see that coming at all.", "en")[0]
```

English pronunciation can be overridden with [CMU dictionary](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a)
phonemes in brackets (uppercase, with stress digits), and Chinese pronunciation with pinyin and a tone number:

```python
model.generate("He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.", "en")
model.generate("这批货物打ZHE2出售后他严重SHE2本了，再也经不起ZHE1腾了。", "zh")
```

Use brackets only for tags and CMU overrides. Text normalization passes every `[...]` span through untouched, but
pinyin overrides are not protected: in text normalized as Spanish, the tone digit of `ZHE2` is exposed like any other
digit. Text in any other language passes through normalization unchanged.

<!-- TODO(TN): the "Text normalization (Spanish)" section goes here. -->

## Streaming

`generate_streaming()` speaks a text one sentence at a time and yields `(audio, sentence)` as soon as each sentence is
ready. It takes one sentence or a whole LLM reply, and needs no framework.

```python
for audio, sentence in model.generate_streaming("Hola, ¿cómo va? Gracias por pasar por el stream.", "es"):
    play(audio)     # 1-D float32 at model.sampling_rate
    show(sentence)  # the raw sentence, for subtitles
```

**Items and arguments**

- `audio` is exactly `generate(spoken, ...)[0]` for that sentence alone. Text without a sentence break yields exactly
  `generate(text, ...)[0]`.
- `sentence` is the raw slice of the text that `audio` speaks: before normalization, with its tags. With one sentence it
  is your string, byte for byte; with several, each one is stripped.
- `normalize` is `None` or a callable `(sentence, language id) -> spoken text`, run on each sentence.
  `TextNormalizer(...).normalize` fits as it is. A sentence it leaves with nothing to say (`"<3"` becomes `""`) is
  skipped: it yields no item and draws no random numbers. If every sentence is skipped, the stream is empty.
- `duration` is `None` (upstream's estimate for each sentence, including its boost below 2 s) or a callable
  `(spoken text) -> seconds`, called once per sentence on the spoken text. A number raises `TypeError`: one float
  applied to every sentence would make each sentence a median 106 % too long.
- The other arguments are `generate()`'s and apply to every sentence. The language is resolved and the config fields
  checked once, before the split.
- Sentences are independent `generate()` calls: no sentence is prompted with earlier ones. The maintainer advises
  against prompting with earlier segments ([k2-fsa/OmniVoice#241](https://github.com/k2-fsa/OmniVoice/issues/241)).
- Nothing runs before the first `next()`. That call raises for a number as `duration` (`TypeError`), an unknown
  language (`ValueError`), an unknown or non-config keyword (`TypeError`), text that is not a `str`
  (`TypeError`) and text with nothing to speak (`ValueError`). Later errors, such as an exception from `normalize` or
  an out-of-memory error, come out of the `next()` that hits them. Earlier items stay valid, and the stream is then
  finished.
- `close()` stops after the sentence in flight. `close()` before the first `next()`, `close()` at a yield, or dropping
  the stream does no further work. There is no per-step cancel. Measured on upstream code (RTX 5090), a sentence in
  flight finishes within 0.28 s (sdpa) or about 0.13 s (FlashInfer) for sentences up to 7.2 s long.
- The caller's `generator`, or the global RNG, is drawn in sentence order and only inside `next()`. With
  `generator=None`, seed right before the first `next()`. A private generator leaves the global RNG untouched.

**One worker thread.** Step and close a stream from the thread that owns the model. In the player-thread pattern,
that thread generates while a second one plays. Measured on upstream code with KohAI replies, the next item was always
ready before the current one finished playing: 0 underruns in 20 replies × 5 repetitions per backend.

```python
import queue
import threading

items = queue.Queue()


def player():
    while (item := items.get()) is not None:
        audio, sentence = item
        show(sentence)
        play(audio)  # blocks while it plays; the next sentence is generated meanwhile


threading.Thread(target=player, daemon=True).start()
for item in model.generate_streaming(reply, "es"):  # this thread owns the model
    items.put(item)
items.put(None)
```

With asyncio, run every model call, every `next()` and the `close()` on one worker thread. `normalize` then also runs
there, off the event loop.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

TTS = ThreadPoolExecutor(max_workers=1)  # the one thread that calls the model


async def speak(reply):
    loop = asyncio.get_running_loop()
    stream = model.generate_streaming(reply, "es")
    try:
        while (item := await loop.run_in_executor(TTS, next, stream, None)) is not None:
            audio, sentence = item
            await play(audio, sentence)
    finally:
        TTS.submit(stream.close)  # queued behind the sentence in flight, on the same thread
```

`close()` from another thread during `next()` raises `ValueError('generator already executing')`, which is why the
`close()` goes to the worker too. Under `task.cancel()`, the worker finishes the sentence in flight and then closes.

**What it buys.** These figures come from KohAI replies (RTX 5090, fp16, 32 steps), measured on upstream code and on
an emulation of the library, to re-measure on the library itself.

- With the whole reply in hand at sdpa, the first audio of a typical KohAI reply comes only ≈ 4 ms (≈ 14 ms with text
  normalization) before `generate()`'s, at ≈ 2× the TTS time.
- Items still give subtitles, a cancel point at each sentence and overlap with the LLM when the caller feeds
  sentences.
- When latency matters, use FlashInfer: first audio in ≈ 0.10 s, and a whole reply of up to ≈ 10 s arrives before
  sdpa's first sentence.
- When the caller feeds sentences from a streaming LLM as they finish, as kohai-v3 does, the first audio comes at the
  LLM's first sentence plus one call: 0.43 s (FlashInfer) and 0.58 s (sdpa), median, measured next to the LLM.

**Advice**

- Put laughs at the start of a sentence. A tag after a sentence end opens the next item (`"Hola. [laughter] ¿Qué
  tal?"` gives `"Hola."` and `"[laughter] ¿Qué tal?"`), and a tag at the very end stays with the last sentence
  (`"Qué bueno. [laughter]"` is one item).
- Strip markdown and list markers from LLM output before streaming: `"1. Pizza"` splits like a sentence.
- Base-model auto voices and designed voices can change between sentences
  ([k2-fsa/OmniVoice#255](https://github.com/k2-fsa/OmniVoice/issues/255)). The fix is a fixed anchor, that is voice
  cloning, which is not in v0.1. A fine-tuned single-voice model such as KohAI keeps its voice.
- With Pipecat, keep its sentence aggregation in front of the TTS (kohai-v3 uses a `SimpleTextAggregator` subclass,
  `SpokenSentences`). Each call then gets one sentence and usually yields one item.
- Each item gets upstream's post-processing, unchanged. Joins are a median 0.42 s, the same as KohAI's natural
  sentence pause. `pad_duration` and `fade_duration` are the only join knobs.
- A sentence longer than `audio_chunk_threshold` is one item, made by `generate()`'s chunked path and delivered after
  the whole sentence.
- Upstream's [PR #166](https://github.com/k2-fsa/OmniVoice/pull/166) yields cumulative audio plus a status. This
  stream yields each sentence's audio plus its text.

**How text is split.** A sentence ends after a run of `. ! ? …`, plus any closing quotes or brackets, followed by
whitespace, the end of the text or a non-verbal tag, and after a run of `。！？`. There is no split inside `3.5`,
`21:30`, `1.500` or URLs (no whitespace follows the mark), after a listed abbreviation (upstream's English list plus
17 Spanish ones such as `Sra.` and `Lic.`) or an initial, or before a lowercase letter, also behind `¿`, `¡`, quotes
or `(` (`"Bueno... ¿qué querés que te diga?"` stays whole). A piece with nothing to say but tags, punctuation or emoji
joins the next piece, or the previous one at the end, so `"[laughter]"` alone is one item.

**Known ceilings of the splitter.** Two of them occur in 23,881 KohAI replies, in 2 replies each: the one-letter
merge (`"GTA V. Podés…"`, `"vitamina D. Yo…"`) and the `No.` merge (`"No. Ser chiquita…"`, `"No. Si les
explico…"`). The others had no hits.

- Abbreviations outside the list split (`Bs. As.`, `Sta. Fe`, German `z.B.`).
- Spanish `No.` before a capital merges with the next sentence, because upstream's English `No.` (number) is in the
  list: `"¿Vas a venir? No. Mañana tal vez."` gives `"¿Vas a venir?"` and `"No. Mañana tal vez."`.
- Numbered list markers split like sentence ends: the marker ends the previous item (`"Opciones: 1."`), or, right
  after a sentence end, is an item of its own (`"1."`).
- A one-letter word before `? ! .` or `。` merges (`"¿Y? Contame."`, `"vitamina D. Hay"`, `"好。我们"`).
- Marks without whitespace merge (`"¡Hola!¿Qué?"`).
- Devanagari `।` and Arabic `؟` are not boundaries.
- Pathological cost: 20,000 glued marks take 3.5 s, and 8,000 skipped boundaries take 1.8 s.

## FlashInfer and int8

Both patch a model loaded on the CPU, which is then moved to the GPU once, and both want one warm-up call:

```python
import torch

from omnivoice_tts import OmniVoice
from omnivoice_tts.models.omnivoice_flashinfer import apply_flashinfer

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", dtype=torch.float16)  # on the CPU
apply_flashinfer(model).to("cuda:0")
model.generate("Warm-up.", "en")  # loads FlashInfer's modules, JIT-compiles its two CuTe-DSL RMSNorm kernels
```

int8 is `apply_int8(model).to("cuda:0")`, with `from omnivoice_tts.models.omnivoice_int8 import apply_int8`, and
FlashInfer + int8 is `apply_int8(apply_flashinfer(model)).to("cuda:0")`. Then make the warm-up call.

- **Patch on the CPU, then move.** Patching a model already on the GPU leaves about 0.57 GiB of reserved memory that
  `torch.cuda.empty_cache()` cannot release (measured with FlashInfer). Quantizing on the CPU keeps the fp16 layer
  weights off the GPU, so the load has no transient and needs no `empty_cache()`.
- Both functions patch in place, return the model, take no options and are idempotent. FlashInfer goes first:
  `apply_flashinfer` after `apply_int8` raises `TypeError`. `apply_flashinfer` raises `ValueError` without CUDA or
  with fp32 weights. `import omnivoice_tts` imports neither module.
- After FlashInfer, only `generate()` and `generate_streaming()` may drive the model, one call at a time per instance.
  Its output is not bit-identical to sdpa's. Judged on KohAI with the emulation below, it was undetectable against
  sdpa on a 124-line paired evaluation (13 metrics), and no difference was heard in 9 blind-listening pairs.
- int8 stores every `nn.Linear` of the LLM's decoder layers as an int8 weight with one scale per output channel, and
  dequantizes it on each call. Embeddings, audio heads, norms and the codec stay as loaded. Its output is not
  bit-identical to fp16, and a quantized model cannot be saved.

**Installing FlashInfer** (NVIDIA GPUs; there is no pip extra). Pick the wheel that matches your torch's CUDA,
`cu128` or `cu129` (`python -c "import torch; print(torch.version.cuda)"`):

```bash
pip install flashinfer-python==0.6.15.post1 'flashinfer-jit-cache==0.6.15.post1+cu128' \
    --extra-index-url https://flashinfer.ai/whl/cu128/
```

With uv, in your project's `pyproject.toml`:

```toml
[project]
dependencies = [
  "omnivoice-tts-inference",
  "flashinfer-python==0.6.15.post1",
  "flashinfer-jit-cache==0.6.15.post1+cu128",
]

[tool.uv.sources]
flashinfer-jit-cache = { index = "flashinfer-cu128" }

[[tool.uv.index]]
name = "flashinfer-cu128"
url = "https://flashinfer.ai/whl/cu128/"
explicit = true
```

On GPUs with compute capability 12.x, such as the RTX 50 series, start Python with
`FLASHINFER_CUDA_ARCH_LIST=12.0a` (the `+cu128` wheel) or `FLASHINFER_CUDA_ARCH_LIST=12.0f` (`+cu129`) in its
environment. When FlashInfer finds no arch it can build or load for the GPU, `apply_flashinfer` raises a
`RuntimeError` that names the variable, before it patches anything. The library never sets environment variables.
The measured stack is cu129 with `12.0f`; the cu128 route with `12.0a` is inferred from FlashInfer's shipped kernels
and code.

**Measured on an emulation of the library, to re-measure.** The emulation is upstream's code with this library's
changes applied (decode-only codec, `use_cache=False`, CPU-first patching), on an RTX 5090 with fp16 weights, 32 steps
and batch 1. "Resident" is the allocated memory after loading and the warm-up call, and includes the FlashInfer
runner. "Peak" is the allocated peak of a call on a text of up to about 6 s and on a 20 s text. "nvidia-smi" is the
process after the short texts, "RTF" is over 12 short KohAI lines, and the last column is the time to the first
stream item of a typical KohAI reply.

| | resident GiB | peak GiB, ≤ 6 s / 20 s text | nvidia-smi MiB | RTF | first sentence of a typical reply |
|---|---|---|---|---|---|
| sdpa fp16 | 1.23 | 1.33 / 1.56 | 2,014 | 0.053 | 240 ms |
| sdpa int8 | 0.82 | 0.92 / 1.15 | 1,612 | 0.057 | 268 ms |
| FlashInfer fp16 | 1.36 | 1.46 / 1.69 | 2,154 | 0.023-0.024 | 103 ms |
| FlashInfer + int8 | 0.95 | 1.05 / 1.28 | 1,754 | 0.031 | 119 ms |

`scripts/benchmark.py`, in the repository but not the package, measures memory, load time, the cold call, the first
stream item and a 29 s paragraph's RTF on the library (`--backend sdpa|flashinfer`, `--int8`).

When to use which:

- FlashInfer fp16 is the fastest, and the best neighbor for an LLM on the same GPU.
- int8 trades 0.41 GiB for +8 % (sdpa) or +31-34 % (FlashInfer) time per call, and costs a co-located LLM more of its
  decode. Use it only when GPU memory is the constraint.

Both int8 configs passed a blind listening on KohAI: no difference heard in 19 of 20 pairs, and int8 preferred in the
other.

## Next to an LLM and faster-whisper

This is kohai-v3's case: the TTS, an LLM server and faster-whisper on one GPU. The figures were measured on the
emulation of the library above, with llama-server and faster-whisper on the same GPU.

- The processes share the GPU by time-slicing (inferred). A TTS call takes 1.12-1.34× as long while the LLM decodes,
  and the LLM keeps 23-34 % of its decode speed while the TTS runs. FlashInfer fp16 costs the LLM the least per
  sentence.
- Run faster-whisper in its own process. Under back-to-back transcription with the LLM idle, a TTS call then takes
  1.11-1.21× as long, against 2.5-2.9× on a thread of the TTS process; next to a decoding LLM this was not measured.
- If it must share the TTS process, a sentence that overlaps an utterance takes 1.2-1.5× as long with default
  settings. Under back-to-back transcription, `OPENBLAS_NUM_THREADS=1` in the launch environment or a high-priority
  CUDA stream for the TTS brings 2.5-2.9× down to 2.0-2.4× (sdpa and FlashInfer fp16, each alone; the two were not
  measured together).
- Keep CPU-heavy Python work out of the TTS process.
- These are caller settings; the library sets none.

The library's GPU work runs on the caller's current stream, which is per thread. A high-priority stream looks like
this, in the TTS thread:

```python
tts_stream = torch.cuda.Stream(priority=-1)  # a lower number is a higher priority

with torch.cuda.stream(tts_stream):
    audio = model.generate(text, "es")[0]
```

## Host memory

The codec decoder's cuDNN execution plans grow host RSS by about 0.7 MiB per new target length.

- The growth is bounded, at about 320 MiB over every length from 1 to 20 s.
- `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT`, a caller setting for the launch environment, caps it: with 270, it costs
  +0.02-0.21 ms per call.

## Threads and global state

**Threads**

- One call at a time per model instance, also with FlashInfer.
- Step and close a stream from one thread. `close()` from another thread during `next()` raises
  `ValueError('generator already executing')`.
- Between yields that thread may call `generate()`.
- One `TextNormalizer` per TTS thread, because NeMo keeps parser state.

**Global state.** Library code prints nothing to stdout or stderr. The library never writes environment variables,
adds warnings filters, keeps module-level caches or locks, calls `torch.cuda.empty_cache` or sets `torch.backends`
flags. Its only logging calls besides `logger.debug` are in `_quiet_nemo`. Around the library's own NeMo calls,
`_quiet_nemo` adds a filter to NeMo's logger, then removes it and restores the level that NeMo's `normalize()` resets
to INFO. The net change is none when calls do not overlap in time; two instances normalizing concurrently can leave
NeMo's logger at INFO (measured).

Neighbors on the same machine are the caller's to tune: the library sets no thread counts and creates no CUDA streams.
Its GPU work runs on the caller's current stream, so the high-priority stream and the separate processes above apply
as they are.

Residual effects of dependencies:

- numpy/torch warnings filters at import.
- scikit-learn via transformers' generation utils, when installed (KMP variables, filters).
- pydub SyntaxWarnings on first compile, and a RuntimeWarning without ffmpeg.
- flashinfer/tvm-ffi import effects.
- CUDA_MODULE_LOADING=LAZY from torch.
- The first Spanish `TextNormalizer.normalize()` imports nemo_text_processing. That import:
  - raises `sys.setrecursionlimit` to 3000;
  - adds a stderr StreamHandler at INFO to the 'NeMo-text-processing' logger. A level set before that import is
    reset, while `.disabled = True` survives. Applications should configure NeMo's logger after that call, or set
    `.disabled = True`.
  - makes joblib set KMP_INIT_AT_FORK=FALSE, unless something already set it;
  - registers one atexit callback;
  - registers a copyreg reducer for `_regex.Pattern`, through the `regex` module;
  - adds numpy's warnings filters if numpy was not imported yet.
- With `cache_dir`: one 0600 pickle (59.7 MB; 48.4 MB with pynini 2.1.7), in a 0700 directory if the library created
  it.

## Differences from upstream

**Upstream bugs fixed in place.** Each one has a CPU regression test, except fix 6, which only corrects a docstring.

1. **Newlines.** A run of CR/LF becomes a space. Upstream deleted it, so `"a.\nb"` was read `"a.b"`.
2. **Near-silent output.** A clip that post-processing trims to nothing (all of it below −50 dBFS) comes back as an
   empty array; upstream crashed on it.
3. **Config fields.** An unknown keyword raises `TypeError`, and keywords override the fields of a passed
   `generation_config` (built with `dataclasses.replace`). Upstream dropped both silently.
4. **Languages.** An unknown language raises `ValueError` instead of becoming `None` with a warning, and IDs match
   case-insensitively (`"ES"` was unknown upstream). Names were already case-insensitive.
5. **`audio_chunk_duration <= 0`** generates in a single pass, as upstream's docstring said; upstream still split.
6. **`postprocess_output`'s docstring** says what it does.
7. **Attention.** The FlashAttention-2 and flex-attention flags, which allowed attention the model does not support,
   are deleted, and `forward()` requires `attention_mask`.
8. **Hub keywords.** `revision`, `token`, `cache_dir` and `local_files_only` reach the download, which ignored them
   upstream, and the codec fallback is pinned to `eustlb/higgs-audio-v2-tokenizer` at `528e871c` instead of following
   its main branch.
9. **Nothing to speak.** Text without a letter or digit raises `ValueError`. Upstream ran it anyway: `""` gave 1 frame
   and `"..."` 18, and `""` with a duration over 30 s raised `ZeroDivisionError`.
10. **Speed in chunked text.** On text above 30 s with `duration` or `speed`, upstream applied the speed twice to every
    chunk after the first. Chunk 0's length already carries it, so the later chunks use 1.0. The output is unchanged
    when `duration` and `speed` are both `None`.

**Removed and changed**

11. **Removed:** `VoiceClonePrompt`, `create_voice_clone_prompt`, `load_asr_model`, `transcribe`, the
    `ref_text`/`ref_audio`/`voice_clone_prompt` parameters, the `train`/`load_asr`/`asr_*` keywords,
    `set_input_embeddings`, the loss, the flex/FA2 flags and `feature_extractor`. Also removed: the `normalize_text`
    parameter and the module function `normalize_text()`. Text normalization is a step the caller runs:
    `TextNormalizer.normalize()` before `generate()`, or `generate_streaming(normalize=tn.normalize)`.
12. **`OmniVoiceGenerationConfig`** keeps upstream's fields and defaults, without `preprocess_prompt` and
    `from_dict`.
13. **Not packaged:** the CLI (`omnivoice-demo`, `omnivoice-infer`, `omnivoice-infer-batch`), training, data and
    evaluation code, and the LoRA utilities. Dropped dependencies: accelerate, gradio, tensorboardX, webdataset,
    soundfile, librosa, peft, WeTextProcessing and num2words.
14. **`text` is `str` only;** a list raises `TypeError`.
15. **Everything after `language` is keyword-only.** Upstream's third positional argument was `ref_text`, so a stale
    call raises instead of binding to `instruct`.
16. **`generator: torch.Generator | None`.** A generator on the model's device drives every draw, chunk by chunk, in
    order. `None` keeps upstream's global RNG with the same op.
17. **`instruct`** is normalized like upstream (split, lowercase, EN↔ZH) and never rejected. Upstream raised on
    unknown items (with a suggestion), on a dialect mixed with an accent and on two items of one category, and custom
    items needed a change to its `_INSTRUCT_ALL_VALID` set.
18. **`generate_streaming()`** is new. Its `duration` is a callable `(spoken text) -> seconds`, and its `normalize` is
    a callable `(sentence, language id) -> spoken text` ([Streaming](#streaming)).
19. **`from_pretrained()`** is the library's own loader: keyword-only arguments, one device, the codec decode-only,
    and the process's default dtype left alone.
20. **`OmniVoiceConfig`** drops `audio_codebook_weights` (training only; the config.json key still loads as an
    attribute) and keeps `model_type = "omnivoice"`. There is no Auto registration.
21. **FlashInfer** is vendored without CUDA graphs or global state. `apply_flashinfer(model)` takes no keywords
    (upstream's `enable_cuda_graph` is gone) and keeps its state on the model. Once it fuses the q/k/v and gate/up
    weights, it deletes the originals, which upstream kept (560 MiB).
22. **`apply_int8()`** is new ([FlashInfer and int8](#flashinfer-and-int8)).
23. **No process-global changes.** Gone from upstream:
    - the 3 warnings filters and `logging.disable` during load;
    - the AutoConfig/AutoModel registration and the flex `AttentionInterface.register`;
    - FlashInfer's module `_CTX`, import-time registration, `set_attn_implementation`, `llm.config.use_cache`
      mutation and graph caches;
    - transformers' default-dtype flips and "Loading weights" bars, and the snapshot bars;
    - stderr warnings for unknown languages and clone prompts;
    - global-only Gumbel noise;
    - upstream TN's module-level `_ZH_NORMALIZER`/`_EN_NORMALIZER`, and its `logger.warning`, which reached stderr
      through `logging.lastResort`;
    - the class-level `@lru_cache` on `RuleDurationEstimator._get_char_weight`.

<!-- TODO(TN): the text-normalization differences go here. -->

**Bit-identity.** With defaults (sdpa, `from_pretrained(dtype=torch.float16)` with fp32 `inv_freq`, an fp32 codec,
the default config and the global RNG), `generate(text)[0]` is bit-identical (int16 PCM sha256) to upstream's
`generate(text)[0]` for any single speakable text without CR/LF, short or chunked. `generate_streaming(text)` on text
without a sentence break yields that same array. With several sentences, the k-th item is `generate()` of the k-th
sentence that is not skipped, with the RNG left by the items before it. Different by design:

- CR/LF texts;
- inputs upstream rejected or crashed on;
- text with nothing to speak (fix 9), and language strings upstream mapped to `None` with a warning, such as `"ES"`
  or unknown names (fix 4);
- `audio_chunk_duration <= 0`;
- keywords with a `generation_config`;
- chunked text with `duration` or `speed` (fix 10). This includes a short text whose `duration` exceeds 30 s, which
  upstream also chunks;
- FlashInfer against sdpa, and int8 against fp16;
- streaming of multi-sentence text.

## What v0.1 does not fix

- `generate()`'s chunker for text above 30 s can split inside `21:30`, `3.5` or `1,000` at a chunk boundary: upstream's
  `chunk_text_punctuation` splits at every `. , ; : ! ?` without a whitespace check. `generate_streaming()`'s splitter
  never does, though a stream sentence above 30 s still goes through this chunker. Spanish text normalized first has
  no digits left to split on in the KohAI corpus; 2 of the 485 held-out number sentences keep them
  (`localhost:5000`, `10²`).
- `Sra.` is missing from `ABBREVIATIONS`, which `generate()`'s chunker uses. The stream splitter has its own Spanish
  list.
- A short, boosted chunk 0 inflates later chunks when `audio_chunk_duration` is lowered.
- The splitter's known ceilings, listed under [Streaming](#streaming).

Kept on purpose for bit-identity: the int16 pydub round trip and the always-on 0.5 peak normalization.

<!-- TODO(TN): the KohAI example goes here. -->

## Licenses

- The code is Apache-2.0 ([LICENSE](LICENSE), upstream's, unchanged), as a derivative work of k2-fsa/OmniVoice. Each
  modified upstream file carries a modification notice, and the FlashInfer module keeps NVIDIA's SPDX header.
- The library ships no weights. k2-fsa's pre-trained OmniVoice weights are licensed CC-BY-NC, because of their
  training data (see the [model card](https://huggingface.co/k2-fsa/OmniVoice)).
- The audio codec, Higgs Audio v2's tokenizer (a checkpoint's `audio_tokenizer/`, or the
  `eustlb/higgs-audio-v2-tokenizer` fallback), is under the Boson Higgs Audio 2 Community License.

## Tips

From upstream's `docs/tips.md`:

- **Short audio generation.** The model may not reliably generate short audio clips (e.g., 1–2 seconds) without
  reference audio. Upstream's remedy, a reference clip, needs voice cloning, which is not in v0.1. Sentence streaming
  makes such clips; with the KohAI voice, lone short sentences sounded right in a listening check.
- **Min Nan Chinese (Hokkien) input format.** Min Nan Chinese (闽南语, also known as Hokkien) can only be synthesized
  using [Tai-lo romanization](https://en.wikipedia.org/wiki/T%C3%A2i-l%C3%B4) as input; Chinese characters are not
  supported for Min Nan Chinese in the current model version.

## Citation

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```

## Disclaimer

Users are strictly prohibited from using this model for unauthorized voice cloning, voice impersonation, fraud, scams, or any other illegal or unethical activities. All users shall ensure full compliance with applicable local laws, regulations, and ethical standards. The developers assume no liability for any misuse of this model and advocate for responsible AI development and use, encouraging the community to uphold safety and ethical principles in AI research and applications.
