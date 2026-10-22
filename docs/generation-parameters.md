<!-- Modified by omnivoice-tts-inference from k2-fsa/OmniVoice 08be0b4 -->
# Generation Parameters

Parameters can be passed as keyword arguments to `model.generate(...)` or via the `OmniVoiceGenerationConfig` dataclass. See below for the full list and which category each belongs to.

```python
# 1) Direct keyword arguments
audio = model.generate(text="Hello world", num_step=32, guidance_scale=2.0)

# 2) Via OmniVoiceGenerationConfig dataclass
from omnivoice_tts import OmniVoiceGenerationConfig

config = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)
audio = model.generate(text="Hello world", generation_config=config)
```

## Decoding

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_step` | int | 32 | Number of iterative unmasking steps. Higher values improve quality but slow down generation. Use 16 for faster inference. |
| `denoise` | bool | True | Prepend the `<\|denoise\|>` token to the input, which signals the model to produce cleaner speech. Only used with a reference prompt: here, the chunks after the first of a long text. |
| `guidance_scale` | float | 2.0 | Classifier-free guidance scale.|
| `t_shift` | float | 0.1 | Time-step shift for the noise schedule. Smaller values emphasise earlier steps in decoding. |

## Sampling

| Parameter | Type | Default | Description |
|---|---|---|---|
| `position_temperature` | float | 5.0 | Temperature for mask-position selection. 0 = greedy (deterministic). Higher values increase randomness. |
| `class_temperature` | float | 0.0 | Temperature for token sampling at each step. 0 = greedy (deterministic). Higher values increase randomness. |
| `layer_penalty_factor` | float | 5.0 | Penalty applied to deeper codebook layers, encouraging earlier (lower) layers to unmask first. |

## Duration & Speed

`duration` and `speed` are arguments of `generate()`, not config fields:

```python
# Fixed 10-second output
audio = model.generate(text="Hello, this is a test of duration control", duration=10.0)

# Faster speech (1.2x faster than estimated)
audio = model.generate(text="Hello, this is a test of duration control", speed=1.2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `duration` | float \| None | None | Fixed output duration in seconds. Overrides `speed` when set. |
| `speed` | float \| None | None | Speed factor. Values > 1.0 produce shorter audio (faster); values < 1.0 produce longer audio (slower). Ignored when `duration` is set. Defaults to 1.0 when both are None. |

Priority: `duration` > `speed`.

> **Note:** When using `duration`, the default post-processing step may shorten long pauses and trim leading and trailing silence, causing the actual output to be shorter than the requested duration, and `pad_duration` adds silence on both sides. If you need the output duration to match the specified value (rounded down to the codec's 40 ms frames), set `postprocess_output=False` and `pad_duration=0`.

## Pre/Post Processing

| Parameter | Type | Default | Description |
|---|---|---|---|
| `postprocess_output` | bool | True | Shorten pauses longer than 1 s to 1 s and trim leading and trailing silence to 100 ms. Peak normalization to 0.5 and the fade/pad always run. |
| `pad_duration` | float | 0.1 | Silence padding duration per side in seconds. Set to 0 to disable. |
| `fade_duration` | float | 0.1 | Fade-in/out curve duration in seconds. Set to 0 to disable. |

## Long-Form Generation

To support stable long-form speech generation with low VRAM consumption, the text is automatically split into smaller segments when the estimated duration of the generated speech exceeds `audio_chunk_threshold`, with each segment producing approximately `audio_chunk_duration` seconds of audio. This approach allows the model to accept arbitrarily long text and generate arbitrarily long speech with near-constant VRAM consumption.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_chunk_duration` | float | 15.0 | Target chunk duration (seconds) when splitting long text. A value ≤ 0 disables chunking. |
| `audio_chunk_threshold` | float | 30.0 | Estimated audio duration (seconds) above which chunking is activated. |
