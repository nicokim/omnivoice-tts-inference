# omnivoice-tts-inference v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `omnivoice_tts`, an inference-only shrink of k2-fsa/OmniVoice @ 08be0b4. It provides `generate()`,
sentence-level `generate_streaming()`, a Spanish `TextNormalizer`, and opt-in FlashInfer and int8. On the default path
it stays bit-identical to upstream.

**Architecture:**
- Copy upstream's kept modules verbatim, then cut and fix them in place, one reviewable change at a time.
- Freeze upstream's behavior once, from the upstream clone, into a JSON fixture. The CPU tests compare against it,
  using a weight-free stub model and a tiny 2-layer model.
- FlashInfer and int8 are separate modules that patch a loaded model. TN is a class the caller runs.

**Tech Stack:**
- Python ≥ 3.10 (3.12 in dev and CI).
- Runtime: torch ≥ 2.8, transformers ≥ 5.3, huggingface_hub ≥ 1.3, safetensors, numpy, pydub.
- Optional: nemo-text-processing 1.2 and flashinfer 0.6.15.post1.
- Tooling: hatchling, uv, ruff, pytest, GitHub Actions, release-please.

**Spec:** `docs/superpowers/specs/2026-10-20-omnivoice-tts-inference-design.md`, approved by the user on 2026-10-21.
- `§n` is a section of the spec.
- `M:`, `T:` and `D:` are its upstream line references: `omnivoice/models/omnivoice.py`, `omnivoice/utils/text.py`
  and `omnivoice/utils/duration.py` at 08be0b4.

## Global Constraints

**Names**
- Import `omnivoice_tts`; distribution `omnivoice-tts-inference`.
- The version line is `__version__ = "0.1.0"  # x-release-please-version`, with two spaces before `#`.

**Paths.** These are the variables of `.plan-env.sh` (see Shell setup).
- `REV=08be0b4ccbac3e13e374e86fbfead4b4cac343e2` is upstream k2-fsa/OmniVoice.
- `UP=/home/kimox/Projects/omnivoice-kohai/src/OmniVoice` is its local clone.
  - Read it only with `git -C "$UP" show`, `git -C "$UP" rev-parse` and `git --no-optional-locks -C "$UP" status`.
  - Never edit, check out, fetch into or stash anything under `/home/kimox/Projects`.
- `EV=/home/kimox/.claude/projects/-home-kimox-orca-workspaces-omnivoice-tts-inference-feat-initial-implementation/afd1a81d-ce0d-4c3f-b201-4690f4fd694d`
  is the design session's evidence directory: prototypes, inputs and measurements. It is read-only, except for two
  output directories: GPU results go under `$EV/gpu-lib/`, and Task 11's re-run TN checks under `$EV/tn-lib/`.
- `KV=/home/kimox/Projects/omnivoice-kohai/venv/bin/python` is the reference venv: py3.12.3, torch 2.8.0+cu128,
  transformers 5.20.0, nemo 1.2.0 and flashinfer 0.6.15.post1, with upstream installed editable.
  - Run it with `-B` and never install into it.
  - Extra packages go in a `--target` directory under `$SP`, the scratchpad of the session that runs the plan.
- `FV=/home/kimox/Projects/omnivoice-kohai/venv_fi/bin/python` is the second reference venv (cu129 and flashinfer), for
  G0's `12.0f` check.
- `KOHAI=/home/kimox/Projects/omnivoice-kohai/hf_export/kohai-omni-tts-fp16` is the KohAI fp16 export. Its config says
  float32, so always pass `dtype=torch.float16`.

**Shell setup.** Shell state does not persist between tool calls, so every shell block in this plan starts with
`. ./.plan-env.sh`, run from the repo root.
- Before Task 1, the controller writes `.plan-env.sh` at the repo root. It is listed in `.gitignore` (Task 1) and never
  committed, which keeps the author's address out of the repo.
- It exports `REV`, `UP`, `EV`, `KV`, `FV`, `KOHAI` and `SP` as above, plus `HF_HUB_OFFLINE=1` and an empty
  `CUDA_VISIBLE_DEVICES`. Child processes such as `scripts/freeze_upstream.py` read them from the environment. Tasks 12
  and 13 set `CUDA_VISIBLE_DEVICES` themselves.
- It defines this function, with the user's address in place of `<address>`:
  `commit() { git commit --author="nicokim <address>" -m "$1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; }`.

**Dependencies (§12)**
- Runtime, exactly: `torch>=2.8`, `torchaudio>=2.8`, `transformers>=5.3`, `huggingface_hub>=1.3`, `safetensors`,
  `numpy`, `pydub`, `audioop-lts; python_version >= '3.13'`.
- The extra, exactly: `tn = ["nemo-text-processing~=1.2.0; sys_platform == 'linux' and platform_machine == 'x86_64'
  and python_version < '3.14'"]`.
- Groups: `dev = ["ruff"]` and `test = ["pytest", "pytest-cov"]`.

**Lint.** ruff with line-length 120 and target py310; select E, F, W, I, UP; ignore E501, E402. Every task ends with
`uv run --only-group dev ruff check` passing.

**License notices**
- Every non-empty upstream file kept in the package carries `# Modified by omnivoice-tts-inference from
  k2-fsa/OmniVoice 08be0b4`.
  - In files with a two-line upstream header (Xiaomi's copyright, or NVIDIA's SPDX pair) it is line 3.
  - In `omnivoice_tts/__init__.py` it is line 1.
- LICENSE stays byte-identical: sha256 `c843a20ede4d1e7595c2736b88fc62cb7483a8c082eb7092a3772248849c59e2`.

**Global state (§11).** Library code never:
- prints;
- writes env vars or adds warnings filters;
- keeps module-level caches, locks or mutable state;
- calls `torch.cuda.empty_cache`;
- sets `torch.backends` flags or thread counts;
- creates CUDA streams.

Logging is `logger.debug` only, plus `_quiet_nemo` (§8).

**Behavior**
- No model-specific config in code: no KohAI emotions, rates or tags. `instruct` is never validated (decision #3).
- The default path stays bit-identical to upstream (§1 criterion 1, §17). A change that moves a frozen fixture is a
  bug, unless this plan names it as a fix.

**Tests.** They run on CPU and offline, with no network and no GPU, except the tests that are explicitly gated.

**Commits**
- Subjects are Spanish conventional commits.
- Always commit with `.plan-env.sh`'s `commit` function, after an explicit `git add <paths>`. The worktree's git
  config holds another identity, so a bare `git commit` is wrong.
- No push, no remote, no bare `git stash`.

**GPU.** GPU work happens in Tasks 12 and 13 only, and needs the user's explicit OK first. Check `nvidia-smi` before
each heavy run, and never kill another process.

**Secrets and files**
- Never print or copy an HF token.
- Never write the listening-page URL, secrets or emails into the repo.
- `omnivoice_tts/utils/es_rules.py` is never edited here (§8).

## Decisions this plan makes where the spec is silent

They are numbered P1-P16, so that they cannot be confused with the spec's decisions (#1-#24) or with the design
session's questions D1-D6.

- **P1.** `normalized_audio_codebook_weights` (M:316-319) is deleted with the loss, because it reads the dropped
  config field.
- **P2.** `forward` returns `OmniVoiceModelOutput(logits=...)`. The dataclass keeps only `logits`.
- **P3.** `device_map` set to `"auto"`, `"balanced"`, `"balanced_low_0"`, `"sequential"` or a dict raises ValueError
  before any I/O. torch itself would raise RuntimeError or TypeError.
- **P4.** The codec fallback downloads only `config.json` and `model.safetensors`, at the pinned revision. It also gets
  the caller's `token`, `cache_dir` and `local_files_only` (Review Focus 1).
- **P5.** The `logger.debug` at M:943 stays in `_generate_chunked`, as a lazy `%`-format call.
- **P6.** With `dtype=None` and no `dtype` in config.json, tensors keep their checkpoint dtype.
- **P7.** `_resolve_language` does not strip whitespace, as upstream does not, so `generate(t, " es ")` raises
  ValueError. `TextNormalizer` still accepts `" es "` (§8).
- **P8.** FlashInfer's guards run in §9's order: CUDA and dtype, then `q_proj`, then the arch.
  - Without CUDA, `apply_flashinfer` after `apply_int8` raises the ValueError.
  - The tests fake CUDA to reach the TypeError.
- **P9.** In `tests/test_flashinfer_cpu.py`, the ImportError test runs everywhere, with flashinfer blocked in
  `sys.modules`. The rest skips without flashinfer, as §18 says. CI therefore skips it, and Task 9 runs it in the
  reference venv.
- **P10.** The KohAI per-layer int8 hash test and the GPU tests are gated by `OMNIVOICE_TEST_KOHAI_DIR`, a local KohAI
  fp16 export (here `$KOHAI`).
- **P11.** §20 says the parity test runs "on the floor and latest transformers". That becomes two CI jobs, `floor` and
  `latest` (Task 6); §13's jobs stay as written. Codecov is dropped: there is no token, and it fails silently in
  indextts25-inference.
- **P12.** release-please starts from manifest `0.0.0`, with `bootstrap-sha` set to this repo's initial commit
  `c055cfd52576fd234dff0bfe73892ea5750db4da`. The first `feat:` release is then 0.1.0.
- **P13.** The splitter fixture has 73 cases:
  - the 68 hand cases;
  - the spec's two critic cases that are missing from them;
  - three more that the critic fixed: `Hola… ¿cómo andás?`, `Esperá… (ya vuelvo).` and `¡Hola! ¿Qué tal?`.
- **P14.** Files new to this repo (int8, tests, scripts) get no license header. Only modified upstream files carry the
  notice.
- **P15.** The `master` → `main` rename (§13: "at implementation") is left to the user. `master` is checked out in the
  main worktree under `/home/kimox/Projects`. CI and release-please run on `main` only after the rename.
- **P16.** The sdist holds only the package, as upstream's does (`[tool.hatch.build.targets.sdist] include =
  ["omnivoice_tts"]`). Without it, `uv build` would ship this plan, the spec (which has local paths) and the CI files
  to PyPI, and a PyPI upload cannot be replaced.

## Review Focus

1. **Offline load without `audio_tokenizer/`.** KohAI's checkpoint has no `audio_tokenizer/`, so it uses the codec
   fallback.
   - `local_files_only`, `cache_dir` and `token` must also reach the fallback's `snapshot_download`, with the revision
     still pinned.
   - Otherwise an offline kohai-v3 boot fails, or the codec fills the default cache.
   - Test: Task 6, `test_hub_kwargs_forwarded_and_codec_pinned`.
2. **Moving a patched model.** The CPU-first recipe patches the model, then moves it with `.to(device)`.
   - After `apply_int8`, `apply_flashinfer` or both, every tensor must be a registered parameter or buffer, so nothing
     stays behind.
   - Tests: Task 8 `test_int8_model_moves_whole` and Task 9 `test_patched_model_moves_whole`.
3. **Two models in one process.** This covers two voices, or FlashInfer next to sdpa. Patching one model must leave the
   other and the module untouched. Test: Task 9 `test_two_models_share_no_state`.
4. **A long stream sentence.** A sentence longer than `audio_chunk_threshold` comes out as one item, through the
   chunked path, with the raw sentence as its text. Test: Task 7 `test_long_sentence_is_one_chunked_item`.
5. **`instruct` edge strings.** Inputs such as `""`, `"   "`, `"female,"`, `"，男，"`, mixed case, conflicting items,
   and a dialect plus an accent never raise. Wherever upstream accepted them, they normalize as upstream did.
   - Task 2 adds them to the frozen set.
   - Test: Task 3, `test_instruct_normalised_like_upstream_and_never_rejected`.

---

### Task 1: Bootstrap the repository from upstream 08be0b4

**Files:**
- Create (verbatim, imports renamed): `LICENSE`, `omnivoice_tts/__init__.py`,
  `omnivoice_tts/models/{__init__,omnivoice,omnivoice_flashinfer}.py`,
  `omnivoice_tts/utils/{__init__,audio,duration,lang_map,text,voice_design}.py`, `docs/generation-parameters.md`,
  `docs/voice-design.md`
- Create: `pyproject.toml`, `README.md` (a placeholder), `.python-version`, `.gitignore`, `uv.lock`,
  `release-please-config.json`, `.release-please-manifest.json`, `.github/workflows/{ci,release-please,release}.yml`

**Interfaces:**
- Produces: the package tree with upstream's content, ruff-converted; a working `uv sync`; `ruff check` clean.

- [ ] **Step 1: Copy upstream**

```bash
. ./.plan-env.sh
mkdir -p omnivoice_tts/models omnivoice_tts/utils docs
git -C "$UP" show "$REV:LICENSE" > LICENSE
for f in __init__.py models/__init__.py models/omnivoice.py models/omnivoice_flashinfer.py utils/__init__.py \
         utils/audio.py utils/duration.py utils/lang_map.py utils/text.py utils/voice_design.py; do
  git -C "$UP" show "$REV:omnivoice/$f" | sed -E 's/\bomnivoice\.(models|utils)\b/omnivoice_tts.\1/g' > "omnivoice_tts/$f"
done
for f in generation-parameters.md voice-design.md; do git -C "$UP" show "$REV:docs/$f" > "docs/$f"; done
```

- [ ] **Step 2: Verify that the copy is upstream apart from the rename**

```bash
. ./.plan-env.sh
sha256sum LICENSE
for f in $(cd omnivoice_tts && find . -name '*.py'); do
  diff <(sed -E 's/\bomnivoice_tts\./omnivoice./g' "omnivoice_tts/$f") <(git -C "$UP" show "$REV:omnivoice/${f#./}") \
    || echo "DIFF $f"
done
```
Expected: LICENSE's hash is `c843a20e…` and no `DIFF` line is printed.

- [ ] **Step 3: Commit the verbatim copy**

```bash
. ./.plan-env.sh
git add LICENSE omnivoice_tts docs/generation-parameters.md docs/voice-design.md
commit "chore: copiar OmniVoice 08be0b4 con los imports renombrados"
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "omnivoice-tts-inference"
version = "0.1.0"
description = "OmniVoice TTS inference library: generate, sentence streaming, Spanish text normalization, FlashInfer and int8"
readme = "README.md"
authors = [{ name = "k2-fsa/OmniVoice authors" }]
maintainers = [{ name = "Kimox" }]
license = "Apache-2.0"
license-files = ["LICEN[CS]E*"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Developers",
  "Topic :: Multimedia :: Sound/Audio :: Speech",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Operating System :: POSIX :: Linux",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
]
requires-python = ">=3.10"
dependencies = [
  "torch>=2.8",
  "torchaudio>=2.8",  # transformers' Higgs codec imports it
  "transformers>=5.3",
  "huggingface_hub>=1.3",
  "safetensors",
  "numpy",
  "pydub",
  "audioop-lts; python_version >= '3.13'",  # pydub needs audioop, which left the stdlib in 3.13
]

[project.optional-dependencies]
# Spanish TN. nemo pins pynini==2.1.6.post1: PyPI wheels only for manylinux x86_64, CPython 3.8-3.13.
tn = ["nemo-text-processing~=1.2.0; sys_platform == 'linux' and platform_machine == 'x86_64' and python_version < '3.14'"]

[dependency-groups]
dev = ["ruff"]
test = ["pytest", "pytest-cov"]

[project.urls]
Upstream = "https://github.com/k2-fsa/OmniVoice"

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.sdist]
include = ["omnivoice_tts"]  # the package only, as upstream's sdist

[tool.hatch.build.targets.wheel]
packages = ["omnivoice_tts"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP"]
ignore = ["E501", "E402"]
```

- [ ] **Step 5: Write the other scaffold files**
- **`README.md`.** Two lines, `# omnivoice-tts-inference` and `Inference-only OmniVoice (k2-fsa/OmniVoice 08be0b4).
  Work in progress.` hatchling refuses a missing readme. Task 10 replaces this file.
- **`.python-version`.** `3.12`.
- **`.gitignore`.** indextts25-inference's list (`__pycache__/`, `*.py[cod]`, `.venv/`, `dist/`, `build/`,
  `*.egg-info/`, `checkpoints/`, `*.wav`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `coverage.xml`), plus
  `.plan-env.sh`.
- **`release-please-config.json`:**
  ```json
  {
    "$schema": "https://raw.githubusercontent.com/googleapis/release-please/main/schemas/config.json",
    "bootstrap-sha": "c055cfd52576fd234dff0bfe73892ea5750db4da",
    "packages": {
      ".": {
        "release-type": "python",
        "package-name": "omnivoice-tts-inference",
        "include-component-in-tag": false,
        "extra-files": ["omnivoice_tts/__init__.py"]
      }
    }
  }
  ```
- **`.release-please-manifest.json`.** `{".": "0.0.0"}` (P12).
- **`.github/workflows/ci.yml`.** Two jobs on `runs-on: ubuntu-latest`, each with `actions/checkout@v4` and
  `astral-sh/setup-uv@v6` (`enable-cache: true`), running on `push` to `main` and on `pull_request`:
  - `lint` runs `uv run --only-group dev ruff check`;
  - `test` runs `uv run --extra tn --group test pytest --cov=omnivoice_tts`.
- **`release-please.yml` and `release.yml`.** Byte-for-byte indextts25-inference's (`git -C
  /home/kimox/Projects/indextts25-inference show origin/main:.github/workflows/<name>`).
  - release-please-action@v4 with `secrets.RELEASE_PLEASE_TOKEN`.
  - On tags `v*`: environment `pypi`, `id-token: write`, `uv build`, then `pypa/gh-action-pypi-publish@release/v1`.

- [ ] **Step 6: Lock and sync**

Run: `uv lock && uv sync --group dev --group test`
Expected: `uv.lock` is created, and the project installs editable without errors.
- The dry run resolved 94 packages, including torch 2.14.1 and transformers 5.20.2.
- No cdifflib build is needed, because the tn extra is not synced here.

- [ ] **Step 7: Convert to the §13 style once (decision #16), rename the ambiguous `l` names, add the notices**

```bash
. ./.plan-env.sh
uv run --only-group dev ruff check --fix omnivoice_tts   # "Found 96 errors (92 fixed, 4 remaining)": 4 E741
```
- Rename `l` to `n` by hand in the 4 remaining sites:
  - `GenerationTask.get_indices` in `models/omnivoice.py` (2 sites);
  - the doc-lens pack loop in `models/omnivoice_flashinfer.py` (`for l in doc_lens[:-1]`);
  - the list comprehension over `doc_lens_key` in the same file.
- Add the notice with these commands:
  ```bash
  . ./.plan-env.sh
  N='# Modified by omnivoice-tts-inference from k2-fsa/OmniVoice 08be0b4'
  for f in models/omnivoice.py models/omnivoice_flashinfer.py utils/audio.py utils/duration.py utils/lang_map.py \
           utils/text.py utils/voice_design.py; do sed -i "2a $N" "omnivoice_tts/$f"; done
  sed -i "1i $N" omnivoice_tts/__init__.py
  ```
- Check `head -3` of every file: line 3 (line 1 in `__init__.py`) is the notice, right under the two header lines.

- [ ] **Step 8: Verify**

Run: `uv run --only-group dev ruff check`
Expected: `All checks passed!`

- [ ] **Step 9: Commit**

```bash
. ./.plan-env.sh
git add pyproject.toml README.md .python-version .gitignore uv.lock release-please-config.json \
        .release-please-manifest.json .github omnivoice_tts
commit "chore: andamiaje del repo, conversión con ruff y avisos de modificación"
```
Then run `. ./.plan-env.sh && uv build --out-dir "$SP/dist" && tar tzf "$SP"/dist/*.tar.gz`. Expected: only
`omnivoice_tts/`, `LICENSE`, `README.md`, `pyproject.toml`, `PKG-INFO` and `.gitignore` (P16).

---

### Task 2: Freeze upstream's behavior as fixtures

**Files:**
- Create: `tests/stubs.py`, `scripts/freeze_upstream.py`, `tests/data/upstream_08be0b4.json`

**Interfaces:**
- Produces, in `tests/stubs.py`:
  - `FRAME_RATE = 25`, `HOP = 960` and `SAMPLING_RATE = 24000`;
  - `class Recorder(draw: bool = False)`, a callable used as `_generate_iterative(task, gen_config, generator=None)`.
    It appends `[batch_size, texts, target_lens, langs, instructs, ref_texts, ref_token_lens]` (all lists) to
    `.calls`, the generator to `.gens`, `gen_config` to `.configs` and `torch.is_inference_mode_enabled()` to
    `.inference`;
  - `fake_decode(tokens) -> SimpleNamespace(audio_values=Tensor(1, 1, T*960))`;
  - `stub_model(cls, estimator, recorder)`, a weight-free instance of `cls`.
- Produces the fixture `tests/data/upstream_08be0b4.json`. It is one object with these keys:
  - `meta`: `{upstream, python, torch, transformers, numpy}`;
  - `calls`: 28 cases;
  - `chunked`: 9 cases;
  - `chunker`: `{split_punctuation, closing_marks, abbreviations, cases: [{text, chunk_len, out}]}`;
  - `duration`: `[[text, weight, estimate]]`;
  - `instruct`: `[[input, use_zh, output, rejected]]`;
  - `language`: `[[input, output, accepted]]`;
  - `combine_text`: `[[text, ref_text, out]]`.

  Each `calls` and `chunked` case is `{text, kwargs, calls, pcm_sha256, n_samples}`.

- [ ] **Step 1: Write `tests/stubs.py`**

The stub's numbers must not depend on torch's RNG or on vectorized transcendental functions. That way the frozen
upstream values stay valid on CI's newer torch.

```python
"""Weight-free stubs shared by the tests and scripts/freeze_upstream.py.

Parity tokens come from a counter, not an RNG, and fake_decode uses exact float ops only, so the
upstream outputs frozen in tests/data stay valid on any torch version.
"""

from types import SimpleNamespace

import torch

FRAME_RATE, HOP, SAMPLING_RATE = 25, 960, 24000


class Recorder:
    def __init__(self, draw=False):
        self.draw = draw  # True: tokens drawn from the generator (RNG tests); False: from a counter (parity)
        self.calls, self.gens, self.configs, self.inference = [], [], [], []

    def __call__(self, task, gen_config, generator=None):
        self.calls.append([task.batch_size, list(task.texts), list(task.target_lens), list(task.langs),
                           list(task.instructs), list(task.ref_texts),
                           [None if t is None else t.size(-1) for t in task.ref_audio_tokens]])
        self.gens.append(generator)
        self.configs.append(gen_config)
        self.inference.append(torch.is_inference_mode_enabled())
        if self.draw:
            u = torch.empty(task.batch_size, 8, max(task.target_lens)).uniform_(generator=generator)
            return [(u[i, :, :n] * 1023).long() for i, n in enumerate(task.target_lens)]
        k = len(self.calls)
        return [torch.full((8, n), (97 * k + i) % 1024, dtype=torch.long) for i, n in enumerate(task.target_lens)]


def fake_decode(tokens):
    """(1, 8, T) tokens -> a triangle wave whose level follows the tokens, silent for 15 of every 40 frames
    (600 ms gaps and leading silence), so remove_silence and the fades have work to do."""
    frames = tokens.shape[-1]
    level = tokens.float().mean(1)[0] / 1023.0
    level = torch.where(torch.arange(frames) % 40 < 15, torch.zeros_like(level), level)
    tri = ((torch.arange(HOP) % 48) - 24).abs() / 24.0 - 0.5
    return SimpleNamespace(audio_values=(level[:, None] * tri[None, :]).reshape(1, 1, -1) * 0.8)


def stub_model(cls, estimator, recorder):
    """An instance of ``cls`` without weights: the real generate() and post-processing, a recorded
    _generate_iterative and fake_decode."""
    m = cls.__new__(cls)
    torch.nn.Module.__init__(m)
    m.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=FRAME_RATE), device="cpu",
                                        decode=fake_decode)
    m.text_tokenizer, m.duration_estimator, m.sampling_rate = object(), estimator, SAMPLING_RATE
    m._generate_iterative = recorder
    return m
```

- [ ] **Step 2: Write `scripts/freeze_upstream.py`**

Write `main(out_path: str) -> None`. It runs upstream 08be0b4, never `omnivoice_tts`, and reads `UP` and `REV` from
the environment.

**Checks before it starts.** Any failure aborts the script.
- `omnivoice.__file__` is under `$UP`.
- `git -C "$UP" rev-parse HEAD` equals `$REV`.
- `git --no-optional-locks -C "$UP" status --porcelain -- omnivoice` is empty. A plain `git status` may rewrite
  the clone's index.

**Repo paths.** It puts `<repo>/tests` on `sys.path` to import `stubs`.

**Ordering.** Every set it reads is sorted before use. `-I` ignores `PYTHONHASHSEED`, and Step 4 proves that the
output does not depend on it.

**What it records**
- **`calls`.** For each of `TEXTS_CALLS × KWARGS`: build `stub_model(upstream OmniVoice, RuleDurationEstimator(),
  Recorder())`, then call `out = m.generate(text=t, **kw)[0]`. Store `calls` and `n_samples = len(out)`. Store
  `pcm_sha256 = sha256(np.ascontiguousarray(out, dtype="<f4").tobytes())`.
- **`chunked`.** The same, for each of `LONG_TEXTS × CHUNKED_KWARGS`, always with `language="es"`, which is stored in
  `kwargs`.
- **`chunker`.** `sorted()` of `SPLIT_PUNCTUATION`, `CLOSING_MARKS` and `ABBREVIATIONS`, plus
  `chunk_text_punctuation(t, cl, 3)` for `CHUNK_TEXTS × (0, 20, 1000)` and `LONG_TEXTS × (100, 200, 400)`.
- **`duration`.** For each of `TEXTS_CALLS + LONG_TEXTS + CHUNK_TEXTS + ["", "a", "1234", "[laughter]", "你好",
  "Hello world.", "¿Qué?"]`: `calculate_total_weight(t)` and `estimate_duration(t, "Nice to meet you.", 25)`.
- **`instruct`.** Build the inputs with the recipe at `$EV/design2/scripts/shrink-map/scripts/equiv_check.py:24-33`,
  with `sorted()` on sets.
  - Those lines already add `KOHAI` and `["Happy", "happy, female", "shy，男"]`.
  - They also call `vd._INSTRUCT_ALL_VALID.update(KOHAI)`, which lets upstream show how it normalizes an unknown word.
  - Append only `EDGE`.
  - For each input × `use_zh in (False, True)`, store `[s, zh, upstream_output, False]`, or `[s, zh, None, True]` when
    upstream raises ValueError.
- **`language`.** The inputs are:
  - `sorted(LANG_IDS)`, `sorted(LANG_NAME_TO_ID)`, `[n.title() for n in sorted(LANG_NAME_TO_ID)]`, plus `None`,
    `"None"` and `"none"`;
  - the probes `["ES", "Es", "SPANISH", " es ", "es-AR", "castellano", "Klingon"]`.

  Store `[x, upstream_output, accepted]`, where `accepted` means the output is not None or `x` is one of the three
  None spellings.
- **`combine_text`.** Call `random.seed(0)`, then build 1000 strings, each `"".join(random.choice(ALPHABET) for _ in
  range(random.randint(0, 40)))`. Pair each string with the next one as its `ref_text`, and the last one with `None`.

```python
SENT = "Ayer fuimos al centro con mi hermana a comprar regalos para toda la familia, y la verdad es que caminamos muchísimo. "
TEXTS_CALLS = ["  Hola che, ¿todo bien?  ", "[laughter] No puedo más, me ganaste otra vez.", (SENT * 6).strip(),
               "你好，世界。" * 40]
KWARGS = [{}, {"duration": 3.0}, {"speed": 1.3}, {"duration": 40.0, "language": "es"},
          {"instruct": "female, low pitch"}, {"num_step": 16}, {"speed": 0.4}]
S = "Hoy arrancamos el stream con todo, chat, y vamos a jugar un rato largo. "
LONG_TEXTS = [S * 12, ("La verdad que no sé qué decirte, " * 30) + "fin.", "Uno, dos, tres; cuatro: cinco. " * 25]
CHUNKED_KWARGS = [{}, {"speed": 1.3}, {"duration": 45.0}]
CHUNK_TEXTS = ["Ya llegó la lluvia, che[laughter]. No puedo más, me ganaste otra vez.",
               "[laughter] No puedo más, me ganaste otra vez. ¿Viste? Te lo dije.", "Hola.[laughter] ¿Qué tal?",
               "Son las 21:30 y cuesta 3.5 pesos, o sea 1,000 centavos.",
               "La Sra. Pérez y el Sr. Gómez llegaron. Dr. House también.", "  Hola che, ¿todo bien?  ",
               "Hola... ¿estás ahí? ¡Sí!"]
KOHAI = ["angry", "happy", "sad", "surprised", "afraid", "melancholic", "shy", "confused"]
EDGE = ["", "   ", ",", "female,", " ,male", "，男，", "Female , Low Pitch", "FEMALE，LOW PITCH", "female, female",
        "male, female", "american accent，河南话", "河南话, female"]
ALPHABET = "abc Hola,. ¿?¡! [laughter] 你好（）\t  "
```

- [ ] **Step 3: Run it**

Run: `. ./.plan-env.sh && "$KV" -I -B scripts/freeze_upstream.py tests/data/upstream_08be0b4.json`

Expected output, one count per key, as measured in the dry run:
- `calls 28` and `chunked 9`;
- `chunker cases 30`;
- `duration 21`;
- `instruct 862`: 802 accepted and 60 rejected. That is the recipe's 838 pairs (784/54, as in the shrink-map run)
  plus EDGE's 24 pairs (18/6);
- `language 1948`: 1,942 accepted (the 1,941 base inputs plus the probe `"SPANISH"` → `"es"`) and 6 rejected probes;
- `combine_text 1000`.

The JSON is about 373 KB.

- [ ] **Step 4: Prove the output is deterministic**

Run the same command with `"$SP/upstream_again.json"` as the output, then
`. ./.plan-env.sh && cmp tests/data/upstream_08be0b4.json "$SP/upstream_again.json"`.
Expected: no output, which means the files are identical.

- [ ] **Step 5: Commit**

```bash
. ./.plan-env.sh
git add tests/stubs.py scripts/freeze_upstream.py tests/data/upstream_08be0b4.json
commit "test: congelar el comportamiento de upstream 08be0b4"
```

---

### Task 3: Shrink the package to inference, keeping upstream's behavior

This task makes the package importable and str-only, and deletes what v0.1 drops. Behavior stays upstream's: the bug
fixes come in Task 4, each with its own test.

**Files:**
- Modify: `omnivoice_tts/__init__.py`, `omnivoice_tts/models/omnivoice.py`,
  `omnivoice_tts/utils/{audio,duration,lang_map,text,voice_design}.py`
- Create: `tests/conftest.py`, `tests/test_generate.py`, `tests/test_split_sentences.py`, `tests/test_package.py`

**Interfaces:**
- Consumes: `tests/stubs.py` (`Recorder`, `stub_model`, `fake_decode`) and `tests/data/upstream_08be0b4.json`.
- Produces: `omnivoice_tts/__init__.py`, which holds the notice, the import of `OmniVoice`, `OmniVoiceConfig` and
  `OmniVoiceGenerationConfig`, `__version__ = "0.1.0"  # x-release-please-version` and `__all__` with the three names.
- Produces in `models/omnivoice.py`:
  - `OmniVoice.generate(self, text: str, language: str | None = None, *, instruct: str | None = None, duration: float
    | None = None, speed: float | None = None, generation_config: OmniVoiceGenerationConfig | None = None, **kwargs) ->
    list[np.ndarray]`;
  - `OmniVoice._preprocess_all(self, text: str, language: str | None = None, instruct: str | None = None, duration:
    float | None = None, speed: float | None = None) -> GenerationTask`;
  - `OmniVoice._generate_chunked(self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig) ->
    list[torch.Tensor]`;
  - `OmniVoice._decode_and_post_process(self, tokens: torch.Tensor | list[torch.Tensor], gen_config) -> np.ndarray`;
  - `OmniVoice._post_process_audio(self, generated_audio: np.ndarray, gen_config) -> np.ndarray`;
  - `OmniVoice.forward(self, input_ids, audio_mask, attention_mask=None) -> OmniVoiceModelOutput`;
  - `_resolve_instruct(instruct: str | None, use_zh: bool = False) -> str | None`;
  - `GenerationTask(batch_size, texts, target_lens, langs, instructs, ref_texts, ref_audio_tokens, speed: list[float] |
    None = None)`;
  - `OmniVoiceGenerationConfig`, with upstream's fields minus `preprocess_prompt`; `from_dict` stays until Task 4;
  - `OmniVoiceModelOutput(logits)` (P2).
- Produces in `tests/conftest.py`:
  - `TINY_LLM`, `TINY_CODEC` and `SPECIALS`;
  - `tiny_tokenizer() -> PreTrainedTokenizerFast`;
  - `tiny_codec() -> HiggsAudioV2TokenizerModel`;
  - `tiny_model(dtype=torch.float32) -> OmniVoice`, with the tokenizer, the codec, `RuleDurationEstimator()` and
    `sampling_rate` attached;
  - `cfg_batch() -> (input_ids, audio_mask, attention_mask)`, the 2-row CFG batch of
    `$EV/design2/scripts/shrink-map/scripts/loader_parity.py:87-93` (`S, T = 40, 20`, seed 1).

- [ ] **Step 1: Write `tests/conftest.py`**

Use these values; they are checked in `$EV/plan/understand/scratch/tests-map/`:
```python
TINY_LLM = dict(model_type="qwen3", hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, vocab_size=1100, max_position_embeddings=512, rope_theta=1e6,
                rms_norm_eps=1e-6, tie_word_embeddings=True)  # vocab >= 1025: the text embedding also sees audio ids
TINY_CODEC = dict(
    acoustic_model_config=dict(encoder_hidden_size=8, decoder_hidden_size=64, hidden_size=8,
                               downsampling_ratios=[8, 5, 4, 2, 3], upsampling_ratios=[8, 5, 4, 2, 3]),
    semantic_model_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                               conv_dim=[16] * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2))
SPECIALS = ["<|endoftext|>", "<|im_end|>", "<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
            "<|instruct_end|>", "<|text_start|>", "<|text_end|>"]
```
- **`tiny_tokenizer()`.** A `tokenizers` WordLevel tokenizer with the vocabulary `SPECIALS + ["[UNK]", "es", "None",
  "hola", "happy", "[laughter]"]` and a Whitespace pre-tokenizer. Wrap it in `PreTrainedTokenizerFast(unk_token="[UNK]",
  pad_token="<|endoftext|>", eos_token="<|im_end|>", additional_special_tokens=SPECIALS[2:])`.
- **`tiny_codec()` and `tiny_model()`.** Each calls `torch.manual_seed(0)` before building.
  - `tiny_codec()` builds `HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig(**TINY_CODEC))`.
    - A fresh codec's codebooks are all zero, so its decode would not depend on the codes. Fill them with `for q in
      codec.quantizer.quantizers: q.codebook.embed.normal_()`.
    - Then return `codec.eval()`.
  - `tiny_model()` builds `OmniVoice(OmniVoiceConfig(llm_config=TINY_LLM))` and casts its parameters with `p.data =
    p.data.to(dtype)`.
    - The cast happens before the codec is attached, so the codec stays fp32.
    - Buffers keep their dtype, so `inv_freq` stays fp32.
- **Silence.** The tiny codec decodes at about -51 dBFS, which is under `remove_silence`'s -50 dBFS threshold. So the
  end-to-end tests on tiny models pass `postprocess_output=False`; the stub tests cover post-processing.
- **Recipe.** `$EV/plan/understand/scratch/tests-map/tiny_e2e.py` ran upstream end to end. Its codec was built without
  reseeding, so it was louder by chance.

- [ ] **Step 2: Write the failing tests**

`tests/test_generate.py`, with `UP = json.loads((Path(__file__).parent / "data/upstream_08be0b4.json").read_text(encoding="utf-8"))`
and `sha(a) = hashlib.sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()`:
```python
def run(text, kwargs, rec=None):
    rec = rec or Recorder()
    return rec, stub_model(OmniVoice, RuleDurationEstimator(), rec).generate(text, **kwargs)[0]

@pytest.mark.parametrize("case", UP["calls"] + UP["chunked"])
def test_generate_matches_upstream_call_sequence(case):
    rec, out = run(case["text"], case["kwargs"])
    assert rec.calls == case["calls"] and sha(out) == case["pcm_sha256"] and len(out) == case["n_samples"]

def test_duration_estimator_matches_upstream():
    est = RuleDurationEstimator()
    for text, weight, dur in UP["duration"]:
        assert est.calculate_total_weight(text) == weight and est.estimate_duration(text, "Nice to meet you.", 25) == dur

def test_combine_text_unchanged_without_newlines():
    assert all(ov._combine_text(t, r) == out for t, r, out in UP["combine_text"])

def test_instruct_normalised_like_upstream_and_never_rejected():
    for s, zh, out, rejected in UP["instruct"]:
        got = ov._resolve_instruct(s, use_zh=zh)  # never raises
        assert got is None or isinstance(got, str)
        if not rejected:
            assert got == out
    assert ov._resolve_instruct("Melancholic") == "melancholic"  # upstream rejected it without the KohAI hack

def test_language_accepted_like_upstream():
    assert all(ov._resolve_language(x) == out for x, out, accepted in UP["language"] if accepted)

def test_list_text_raises_typeerror():  # bytes, None and 3 too; the recorder stays empty
def test_positional_after_language_raises():  # m.generate("Hola.", "es", "female") -> TypeError
def test_generate_needs_from_pretrained():  # stub with text_tokenizer=None -> RuntimeError, no calls

def test_generate_end_to_end_on_tiny_model():
    m, outs = tiny_model(), []
    for _ in range(2):
        torch.manual_seed(1)
        outs.append(m.generate("hola [laughter] hola.", "es", num_step=4, postprocess_output=False))
    a, b = outs
    assert len(a) == 1 and a[0].dtype == np.float32 and a[0].shape == (45120,)  # 42 frames x 960 + 2 x 2400 pad
    assert np.array_equal(a[0], b[0])
```
- `tests/test_split_sentences.py::test_upstream_chunker_untouched`:
  - the sorted `SPLIT_PUNCTUATION`, `CLOSING_MARKS` and `ABBREVIATIONS` equal the fixture's lists;
  - `chunk_text_punctuation(text, chunk_len, 3)` equals `out` for every fixture case.
- `tests/test_package.py::test_import_adds_no_filters_and_prints_nothing` runs a `[sys.executable, "-I", "-W",
  "ignore", "-c", ...]` subprocess:
  - **Dependencies first.** It imports numpy, torch, pydub, `pydub.silence`, safetensors, huggingface_hub and `from
    transformers import AutoTokenizer, HiggsAudioV2TokenizerModel, PreTrainedModel`.
    - transformers' first modeling import pulls in torch._dynamo, which imports sympy and adds a filter, and scipy's
      filters too when scipy is installed.
    - Those are §11 residual effects, not the library's.
  - **Snapshot and check.** Only then does it snapshot `list(warnings.filters)`. It imports `omnivoice_tts`, asserts the
    filters are unchanged, and prints `__version__`.
  - **Expected.** stdout `"0.1.0\n"` and empty stderr. Upstream's `__init__` added 3 filters.

- [ ] **Step 3: Run the tests to make sure they fail**

Run: `uv run --group test pytest tests -q`
Expected: pytest stops with exit code 4, printing `ImportError while loading conftest 'tests/conftest.py'` and
`ModuleNotFoundError: No module named 'soundfile'`, raised from `omnivoice_tts/utils/audio.py`. The verbatim
`utils/audio.py` imports `soundfile`, which is not a dependency.

- [ ] **Step 4: Cut the utils (§3)**

Delete by symbol. Line numbers moved in Task 1, so use upstream's as orientation only.
- **`utils/audio.py`:**
  - delete the imports `io`, `logging`, `soundfile`, `torch` and `torchaudio`, and `detect_nonsilent`;
  - delete the unused `logger`, the loading banner, `load_waveform`, `load_audio`, `load_audio_bytes` and
    `trim_long_audio`;
  - reword the module docstring: it now covers silence removal, fades and cross-fades.
  - Kept: `numpy_to_audiosegment`, `audiosegment_to_numpy`, `remove_silence`, `remove_silence_edges`,
    `fade_and_pad_audio` and `cross_fade_chunks`. About 203 lines.
- **`utils/duration.py`.** Delete `from functools import lru_cache`, the `@lru_cache(maxsize=4096)` on
  `_get_char_weight` and the `__main__` demo. About 247 lines.
- **`utils/lang_map.py`.** Delete `_TITLE_EXCEPTIONS` and `lang_display_name`. Point the docstring's
  `docs/lang_id_name_map.tsv` at upstream's URL at 08be0b4.
- **`utils/text.py`:**
  - delete `END_PUNCTUATION`, `add_punctuation` and the whole TN section (T:228-429; Task 11 adds the new one);
  - remove the stale docstring bullets.
  - Kept: `SPLIT_PUNCTUATION`, `CLOSING_MARKS`, `ABBREVIATIONS` and `chunk_text_punctuation`. About 177 lines; ruff
    removes the now-unused `import re`.
- **`utils/voice_design.py`:**
  - keep `_ZH_RE`, the bilingual dicts and `_INSTRUCT_EN_TO_ZH`/`_INSTRUCT_ZH_TO_EN`, which are built by the loop;
  - delete the accent and dialect sets, `_INSTRUCT_MUTUALLY_EXCLUSIVE`, `_INSTRUCT_ALL_VALID`, `_INSTRUCT_VALID_EN` and
    `_INSTRUCT_VALID_ZH`;
  - reword the comments that mention validation. About 55 lines.
- **`__init__.py`.** As in Interfaces: no warnings filters, no `importlib.metadata`, no `VoiceClonePrompt`.

- [ ] **Step 5: Cut `models/omnivoice.py`**

The design maps are `$EV/plan/understand/upstream-model.md` (section A) and `$EV/design2/shrink-map.md`.

**Delete:**
- `_AUTOCAST_FLEX_ATTENTION` and `_autocast_flex_attention`, with the flex import try block;
- `_VOICE_CLONE_PROMPT_FORMAT_VERSION`, `VoiceClonePrompt`, `create_voice_clone_prompt`, `load_asr_model`,
  `transcribe` and `set_input_embeddings`;
- in `__init__`: the `llm=` parameter (keep the `AutoModel.from_config` line), the flex registration, the
  `normalized_audio_codebook_weights` lines (P1) and the ASR attributes;
- in `from_pretrained`: the `load_asr`/`asr_*` pops and the ASR branch;
- `_ensure_list`, `_get_packed_mask`, `_mask_mod_packed`, `GenerationTask.get_indices`, `GenerationTask.slice_task`,
  `GenerationTask.ref_rms` and `OmniVoiceGenerationConfig.preprocess_prompt`;
- `normalize_text` everywhere, and the imports that only deleted code used: `difflib`, `functools.partial`,
  `torchaudio`, `ALL_ATTENTION_FUNCTIONS`, `AttentionInterface`, `load_audio`, `trim_long_audio`, `add_punctuation`,
  `normalize_text` and the `_INSTRUCT_VALID*`/`_INSTRUCT_ALL_VALID`/`_INSTRUCT_MUTUALLY_EXCLUSIVE` imports.

**Rewrite.** Each keeps upstream's arithmetic and is pinned by the fixture.
- **`generate()`.** Use the Interfaces signature, decorated with `@torch.inference_mode()`.
  1. The tokenizer guard, with upstream's RuntimeError.
  2. `gen_config = generation_config if generation_config is not None else OmniVoiceGenerationConfig.from_dict(kwargs)`.
  3. `self.eval()`.
  4. `task = self._preprocess_all(text, language, instruct, duration, speed)`.
  5. If `task.target_lens[0] > int(gen_config.audio_chunk_threshold * self.audio_tokenizer.config.frame_rate)`, then
     `tokens = self._generate_chunked(task, gen_config)`; otherwise `tokens = self._generate_iterative(task,
     gen_config)[0]`.
  6. `return [self._decode_and_post_process(tokens, gen_config)]`.

  The docstring covers the str-only `text`, keyword-only arguments, one returned float32 array at `sampling_rate`, and
  saving "e.g. with soundfile (not a dependency)".
- **`_preprocess_all`.** A str-only rewrite of M:1024-1167.
  1. `TypeError` unless `isinstance(text, str)`.
  2. `language = _resolve_language(language)`.
  3. `instruct = None if instruct is None else _resolve_instruct(instruct, use_zh=bool(_ZH_RE.search(text)))`.
  4. `est = self._estimate_target_tokens(text, None, None, speed=1.0 if duration is not None else (float(speed) if
     speed is not None else 1.0))`.
  5. With a duration: `target = max(1, int(float(duration) * frame_rate))`, `speed_list = [est / target]` and `est =
     target`. Otherwise `speed_list = [float(speed)]` if a speed is given, else `None`.
  6. Return `GenerationTask(1, [text], [est], [language], [instruct], [None], [None], speed_list)`.
- **`_generate_chunked`.** Upstream's no-reference branch (M:913-1022), reduced to one item. The `calls` and `chunked`
  fixtures pin everything else.
  - Chunking: `chunk_text_punctuation(text=…, chunk_len=…, min_chunk_len=3)` at upstream's `text_chunk_len`.
  - Prompting: chunk 0 is unprompted; each later chunk is prompted with chunk 0's tokens and text.
  - Speed: `speed = task.speed[0] if task.speed else 1.0` goes to every chunk's `_estimate_target_tokens` (Task 4 fixes
    that).
  - Each chunk is one `GenerationTask(batch_size=1, …)` through `self._generate_iterative(sub_task, gen_config)[0]`.
  - M:943's `logger.debug` stays as a lazy `%` call (P5).
  - It returns the list of chunk token tensors.
- **`_decode_and_post_process` and `_post_process_audio`.**
  - Drop the `rms`/`ref_rms` parameter and the `ref_rms` branch.
  - Keep `peak = np.abs(generated_audio).max()` and `if peak > 1e-6`.
- **`forward`.**
  - Drop `labels`, `document_ids`, `position_ids`, the flex branch and the loss.
  - Keep `attention_mask=None` and the `self.llm(inputs_embeds=…, attention_mask=…, return_dict=True)` call.
  - Return `OmniVoiceModelOutput(logits=audio_logits)`.
- **`_resolve_instruct`.** It only normalizes:
  ```python
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
  ```
- **Module docstring.** Rewrite it: inference only; training, cloning and ASR are removed.

- [ ] **Step 6: Run the tests to make sure they pass**

Run: `uv run --group test pytest tests -q`
Expected: every test passes, including all 37 call-sequence cases.

- [ ] **Step 7: Lint and commit**

```bash
. ./.plan-env.sh
uv run --only-group dev ruff check
git add omnivoice_tts tests
commit "refactor: recortar OmniVoice a inferencia sin cambiar su comportamiento"
```

---

### Task 4: Fix upstream's bugs in place, one test each (§16 fixes 1-7, 9 and 10)

Each fix follows the same cycle:
1. Write its test.
2. Run `uv run --group test pytest tests/test_generate.py -q -k <name>` and see it FAIL.
3. Make the change.
4. See it pass, run the whole suite, and commit.

**Files:**
- Modify: `omnivoice_tts/models/omnivoice.py`, `tests/test_generate.py`

**Interfaces:**
- Produces:
  - `generate()` builds its config with `replace(generation_config or OmniVoiceGenerationConfig(), **kwargs)`, and
    `from_dict` is deleted;
  - `_resolve_language(language: str | None) -> str | None` raises ValueError on unknown values;
  - `forward(self, input_ids, audio_mask, attention_mask)` requires the mask;
  - `_preprocess_all` raises `ValueError(f"text has nothing to speak: {text!r}")`.

- [ ] **Step 1: Fix 1, newlines (M:1706)**
  - Test `test_newline_becomes_space`:
    - `_combine_text("Hola che.\nTodo bien?") == "Hola che. Todo bien?"`;
    - `_combine_text("a\r\n\r\nb") == "a b"`;
    - `_combine_text("你好\n世界") == "你好世界"`.
  - Change: `re.sub(r"[\r\n]+", " ", full_text)`, and update its comment.
  - Commit `fix: los saltos de línea pasan a ser un espacio`.
- [ ] **Step 2: Fix 2, near-silent clip (M:901)**
  - Test `test_near_silent_output_returns_empty`:
    - `quiet = (np.random.default_rng(0).standard_normal((1, 24000)) * 1e-4).astype(np.float32)`;
    - `stub._post_process_audio(quiet, OmniVoiceGenerationConfig()).shape == (1, 0)`;
    - with `stub.audio_tokenizer.decode = lambda t: SimpleNamespace(audio_values=torch.from_numpy(quiet)[None])`,
      `stub.generate("Hola.")[0].shape == (0,)`.
  - Change: `peak = np.abs(generated_audio).max(initial=0.0)`.
  - Commit `fix: un clip casi mudo devuelve audio vacío`.
- [ ] **Step 3: Fix 3, config kwargs (M:191-195, 677-681)**
  - Tests:
    - `test_unknown_generate_kwarg_raises_typeerror`: `num_steps=8` raises TypeError, with and without a
      `generation_config`, and `rec.calls == []`.
    - `test_kwargs_override_generation_config`: with `cfg = OmniVoiceGenerationConfig(num_step=32)` and
      `generate("Hola.", generation_config=cfg, num_step=16)`, `rec.configs[0].num_step == 16` and `cfg.num_step ==
      32`.
  - Change: `gen_config = replace(generation_config or OmniVoiceGenerationConfig(), **kwargs)`, then delete
    `from_dict`. The import becomes `from dataclasses import dataclass, replace`. Fix the docstring's
    "takes precedence" sentence.
  - Commit `fix: los kwargs pisan generation_config y un campo desconocido falla`.
- [ ] **Step 4: Fix 4, unknown language**
  - Tests:
    - `test_unknown_language_raises`: `"castellano"`, `"Klingon"`, `"es-AR"` and `" es "` (P7) raise ValueError
      matching `supported_language_ids`.
    - `test_language_id_case_insensitive`: `"ES"`, `"es"` and `"Spanish"` all give `"es"`, and every accepted fixture
      pair still matches.
  - Change: §6's version.
    1. `None`, or a value whose `.lower()` is `"none"`, gives None, as upstream.
    2. Otherwise `key = language.lower()`; IDs first, then names.
    3. Anything else raises a ValueError naming `supported_language_ids()`/`supported_language_names()`.

    `LANG_NAME_TO_ID` becomes a module-level import.
  - Commit `fix: un idioma desconocido falla y los IDs no distinguen mayúsculas`.
- [ ] **Step 5: Fix 5, `audio_chunk_duration <= 0` (M:210-214)**
  - Test `test_audio_chunk_duration_zero_disables_split`: on `UP["chunked"][0]["text"]`, `audio_chunk_duration=0.0`
    gives exactly 1 recorded call, and the default gives more than 1.
  - Change: the dispatch condition gains `gen_config.audio_chunk_duration > 0 and`.
  - Commit `fix: audio_chunk_duration <= 0 no divide el texto`.
- [ ] **Step 6: Fix 7 and decision #14, forward**
  - Test `test_forward_requires_attention_mask`, on `tiny_model()` and `cfg_batch()`:
    - without the mask, TypeError;
    - with it, `out.logits.shape == (2, 8, 40, 1025)`;
    - a spy on `m.llm.forward`, an instance attribute that wraps the original, sees `use_cache=False`.
  - Change: `forward(self, input_ids, audio_mask, attention_mask)` calls `self.llm(inputs_embeds=…,
    attention_mask=attention_mask, return_dict=True, use_cache=False)`. Delete `_supports_flex_attn` and
    `_supports_flash_attn_2`.
  - Commit `fix: forward exige attention_mask y no arma caché`.
- [ ] **Step 7: Fix 9, text with nothing to speak**
  - Test `test_text_without_speech_raises`:
    - `""`, `"   "`, `"..."`, `"😀"`, `"¿?"`, and `""` with `duration=40.0`, raise ValueError matching
      `nothing to speak`;
    - `rec.calls == []`;
    - `generate("[laughter]")` still returns one item.
  - Change: right after the str check, add `if not any(c.isalnum() for c in text): raise ValueError(...)`.
  - Commit `fix: un texto sin nada que decir falla`.
- [ ] **Step 8: Fix 10, speed applied twice**
  - In `test_generate_matches_upstream_call_sequence`, parametrize over `PARITY = [c for c in UP["calls"] +
    UP["chunked"] if not (len(c["calls"]) > 1 and {"speed", "duration"} & c["kwargs"].keys())]`.
  - Shared helper `_assert_fix10(case)`, which runs the case and checks:
    1. `rec.calls[0] == case["calls"][0]`.
    2. The chunk texts equal the frozen ones.
    3. With `t0, n0 = rec.calls[0][1][0], rec.calls[0][2][0]`, every later call has `target_lens ==
       [m._estimate_target_tokens(call[1][0], t0, n0, speed=1.0)]`.
    4. The later targets differ from the frozen ones.
  - Test `test_later_chunks_not_sped_up_twice`: `_assert_fix10` on the `speed=1.3` chunked cases and on the chunked
    `calls` cases that have a speed.
  - Test `test_generate_chunked_matches_upstream`: frozen equality when speed and duration are None, and
    `_assert_fix10` on the `duration=45.0` cases and the chunked `calls` cases that have a duration.
  - Change: `speed=1.0 if tokens else speed` in `_generate_chunked`.
  - Commit `fix: los trozos siguientes no aplican speed dos veces`.
- [ ] **Step 9: Fix 6, docstring (M:652).** In `generate()`'s docstring, `postprocess_output` now reads: "remove
  silences longer than 500 ms and trim the edges to 100 ms; peak normalization to 0.5 and the fade/pad always run".
  Commit `docs: corregir la descripción de postprocess_output`.
- [ ] **Step 10: Verify.** `uv run --group test pytest tests -q` passes and ruff is clean.

---

### Task 5: Private generator through every draw

**Files:**
- Modify: `omnivoice_tts/models/omnivoice.py`, `tests/test_generate.py`

**Interfaces:**
- Produces these signatures:
  - `generate(..., generation_config=None, generator: torch.Generator | None = None, **kwargs)`;
  - `_generate_chunked(self, task, gen_config, generator=None)`;
  - `_generate_iterative(self, task, gen_config, generator=None)`;
  - `_predict_tokens_with_scoring(self, c_logits, u_logits, gen_config, generator=None)`;
  - `_gumbel_sample(logits, temperature, generator=None)`.

  The FlashInfer module (Task 9) relies on the 4th and 3rd positional parameters.

- [ ] **Step 1: Write the failing tests**
```python
def test_gumbel_sample_matches_rand_like():
    logits = torch.randn(1, 8, 30, 1025)
    for lg in (logits, logits.permute(0, 2, 1, 3)):
        torch.manual_seed(3)
        s = lg / 0.7
        u = torch.rand_like(s)
        want = s + -torch.log(-torch.log(u + 1e-10) + 1e-10)  # upstream's formula
        torch.manual_seed(3)
        assert torch.equal(ov._gumbel_sample(lg, 0.7), want)
        assert torch.equal(ov._gumbel_sample(lg, 0.7, torch.Generator().manual_seed(3)), want)

def test_generator_reaches_every_chunk():  # Recorder(draw=True), long text, generator=g: >1 call, all gens are g

def test_private_generator_matches_global_seed():
    m, seen = tiny_model(), []
    m._decode_and_post_process = lambda tokens, gen_config: seen.append(tokens) or np.zeros(1, np.float32)
    kw = dict(num_step=4, class_temperature=1.0)
    torch.manual_seed(7)
    m.generate("hola [laughter] hola.", "es", **kw)
    m.generate("hola [laughter] hola.", "es", generator=torch.Generator().manual_seed(7), **kw)
    torch.manual_seed(8)
    m.generate("hola [laughter] hola.", "es", **kw)
    assert torch.equal(seen[0], seen[1]) and not torch.equal(seen[0], seen[2])  # tokens: tiny audio hides the RNG

def test_generator_leaves_global_rng_untouched():  # torch.get_rng_state() equal before and after a generator= call
```
- [ ] **Step 2: Run them and see them fail.** Expected: TypeError (unexpected keyword `generator`).
- [ ] **Step 3: Implement**
  - Thread `generator` through every signature above, and pass it on at the M:1403-1405, M:1410 and M:1444-1446 call
    sites.
  - `_gumbel_sample` draws `u = torch.empty_like(scaled_logits).uniform_(generator=generator)`.
  - The docstring adds: "`generator`: a `torch.Generator` on the model's device; `None` uses the global RNG".
- [ ] **Step 4: Run the whole suite.** Everything passes; the frozen parity cases are unchanged.
- [ ] **Step 5: Commit.** `feat: generator= privado en generate()`.

---

### Task 6: Own loader with a decode-only codec (§5, fix 8)

**Files:**
- Modify: `omnivoice_tts/models/omnivoice.py`, `tests/conftest.py`, `.github/workflows/ci.yml`
- Create: `tests/test_loader.py`

**Interfaces:**
- Produces:
  - `OmniVoice.from_pretrained(cls, pretrained_model_name_or_path: str, *, device_map: str | int | torch.device | None =
    None, dtype: torch.dtype | None = None, revision: str | None = None, token: str | bool | None = None, cache_dir: str
    | None = None, local_files_only: bool = False) -> OmniVoice`;
  - `OmniVoice.__init__(self, config: OmniVoiceConfig)`, which builds `self.llm =
    MODEL_MAPPING[type(config.llm_config)](config.llm_config)`;
  - `OmniVoiceConfig.__init__(self, audio_vocab_size=1025, audio_mask_id=1024, num_audio_codebook=8, llm_config=None,
    **kwargs)`, with `model_type = "omnivoice"` and `sub_configs` kept;
  - module-level `_MODEL_FILES = ["*.json", "*.txt", "model.safetensors", "audio_tokenizer/*"]`;
  - `_CODEC_REPO = "eustlb/higgs-audio-v2-tokenizer"` and `_CODEC_REVISION =
    "528e871c2a26c4f0f7773b9754e2e1acae20899d"`;
  - `_CODEC_ENCODER = ("acoustic_encoder", "encoder_semantic", "decoder_semantic", "semantic_model", "fc", "fc1")` and
    `_CODEC_DECODER = ("quantizer.", "fc2.", "acoustic_decoder.")`;
  - `class _NoBar(huggingface_hub.utils.tqdm)`, which forces `disable=True`;
  - `_resolve_model_path(name_or_path: str, allow_patterns: list[str], revision: str | None = None, **hub) -> str`;
  - `_load_safetensors(module, path: str, device: torch.device, dtype: torch.dtype | None = None, prefixes:
    tuple[str, ...] | None = None) -> None`;
  - `_load_decode_only_codec(path: str, device: torch.device) -> HiggsAudioV2TokenizerModel`.
- Produces in `tests/conftest.py`: a session fixture `tiny_dir`. It holds `config.json` and `model.safetensors` (from
  `OmniVoice(OmniVoiceConfig(llm_config=TINY_LLM)).save_pretrained`, with seed 0 and no codec attached), the tokenizer
  files and `audio_tokenizer/` (`tiny_codec().save_pretrained`).

- [ ] **Step 1: Write the failing tests (`tests/test_loader.py`)**
  - **`test_loader_matches_transformers_from_pretrained(tiny_dir)`:**
    - the reference is `PreTrainedModel.from_pretrained.__func__(OmniVoice, tiny_dir, dtype=torch.float16)`, and the
      library model is `OmniVoice.from_pretrained(str(tiny_dir), dtype=torch.float16)`;
    - leaving out `audio_tokenizer.*`, the parameter and buffer names, dtypes and `torch.equal` values match;
    - `llm.rotary_emb.inv_freq` is fp32;
    - the `cfg_batch()` logits are `torch.equal`;
    - `OmniVoice.from_pretrained(str(tiny_dir), dtype=torch.float32).half()` has an fp16 `inv_freq`, and its logits
      differ.
  - **`test_load_is_silent_and_global_free(tiny_dir, capfd, monkeypatch)`:**
    - load once first, so transformers' lazy imports happen before the snapshot;
    - snapshot `torch.get_rng_state()`, `dict(os.environ)`, `list(warnings.filters)`, `logging.root.manager.disable`,
      `torch.get_default_dtype()` and `"omnivoice" in CONFIG_MAPPING`;
    - spy on `torch.set_default_dtype`, then `capfd.readouterr()`, then load;
    - expected: `readouterr()` is `("", "")`, the snapshot is unchanged and the spy saw no call.
  - **`test_decode_only_codec_matches_full_tiny(tiny_dir)`:**
    - `HiggsAudioV2TokenizerModel.from_pretrained(tiny_dir / "audio_tokenizer").eval()` and
      `_load_decode_only_codec(...)` decode `torch.randint(0, 1024, (1, 8, 6),
      generator=torch.Generator().manual_seed(3))` to `torch.equal` audio;
    - the slim codec has no `acoustic_encoder`.
  - **`test_decode_only_codec_matches_full`:** the same on eustlb@528e871c, with codes `(1, 8, 60)`, and
    `len(slim.state_dict()) == 176`. It skips unless `snapshot_download(_CODEC_REPO, revision=_CODEC_REVISION,
    local_files_only=True)` succeeds.
  - **`test_tokenizer_needs_no_auto_registry(tiny_dir, capfd)`:**
    - `AutoTokenizer.from_pretrained(tiny_dir, config=OmniVoiceConfig.from_pretrained(tiny_dir))` leaves `err == ""`;
    - `convert_tokens_to_ids("<|denoise|>") == 2`;
    - `"omnivoice" not in CONFIG_MAPPING`.
  - **`test_hub_kwargs_forwarded_and_codec_pinned(tiny_dir, tmp_path, monkeypatch)`:**
    - patch `omnivoice_tts.models.omnivoice.snapshot_download` with a spy;
    - for `"org/model"` it returns a copy of `tiny_dir` without `audio_tokenizer/`; for the codec repo, `tiny_dir /
      "audio_tokenizer"`;
    - call `OmniVoice.from_pretrained("org/model", revision="r1", token="t", cache_dir=str(tmp_path / "c"),
      local_files_only=True)`;
    - assert exactly two calls:
      1. `"org/model"`, with `revision="r1"`, `token="t"`, that `cache_dir`, `local_files_only=True`,
         `allow_patterns=_MODEL_FILES` and `tqdm_class is _NoBar`;
      2. `_CODEC_REPO`, with `revision=_CODEC_REVISION`, `allow_patterns=["config.json", "model.safetensors"]`, and
         the same `token`, `cache_dir` and `local_files_only` (Review Focus 1).
  - **`test_from_pretrained_rejects_upstream_kwargs`:** parametrized over `train`, `load_asr`, `asr_model_name`,
    `asr_device`, `attn_implementation`, `torch_dtype` and `trust_remote_code`; each raises TypeError.
  - **`test_device_map_rejects_auto_and_dicts(monkeypatch)`:**
    - `OmniVoice.from_pretrained("org/model", device_map=…)` raises ValueError for `"auto"`, `"balanced"`,
      `"sequential"` and `{"": "cpu"}`.
    - `"org/model"` is a Hub id, not a directory, so a `snapshot_download` that fails if called proves that the check
      runs before any I/O.
  - **`test_bad_checkpoints_raise(tiny_dir, tmp_path)`:**
    - a copy without `model.safetensors` raises FileNotFoundError;
    - a copy whose safetensors lacks one key (rewritten with `safetensors.torch.save_file`) raises RuntimeError.
  - **`test_config_keeps_unknown_keys`:** `OmniVoiceConfig(audio_codebook_weights=[8, 8, 6, 6, 4, 4, 2, 2])` loads
    the key as an attribute, and `OmniVoiceConfig().to_dict()` has no such key.
  - **`test_loaded_model_generates(tiny_dir)`:** `OmniVoice.from_pretrained(str(tiny_dir)).generate("hola.", "es",
    num_step=2, postprocess_output=False)[0]` has dtype float32 and shape `(30720,)`, i.e. 27 frames × 960 plus 2 ×
    2400 of padding.
- [ ] **Step 2: Run them and see them fail.** Expected: TypeError on the keyword-only signature, missing names, and
  hub-call mismatches.
- [ ] **Step 3: Implement §5**
  - The reference code is `$EV/design2/shrink-map.md:80-135`. Fix these points against it:
    - `_MODEL_FILES` is the list above;
    - the P3 check is the first statement;
    - `__init__` uses MODEL_MAPPING;
    - the codec fallback forwards `**hub` (P4);
    - the dtype handling is `dtype if dtype is not None else config.dtype`, and it may stay None (P6);
    - the codec build lives in `_load_decode_only_codec`;
    - `codec_config.sample_rate` becomes `model.sampling_rate`;
    - the codec goes on CPU when the model is on MPS.
  - Delete:
    - upstream's `from_pretrained` body, `logging.disable`, `AutoFeatureExtractor`, the `train` path,
      `audio_codebook_weights` with its default, and the `AutoConfig.register`/`AutoModel.register` lines;
    - the `AutoModel` import.
  - Keep `_supports_sdpa`, `config_class`, `get_input_embeddings`, `post_init()` and the `None` attributes, which the
    guard needs.
- [ ] **Step 4: Run the whole suite.** Everything passes; the parity fixtures are unchanged.
- [ ] **Step 5: Add the CI jobs `floor` and `latest` (P11) and run them locally**
  - Add two jobs to `ci.yml`, with the same `runs-on`, checkout and setup-uv steps as the others. Each is a single
    command:
    - **`floor`:** `uv run --group test --with "torch==2.8.0" --with "torchaudio==2.8.0" --with
      "transformers==5.3.0" pytest tests/test_loader.py tests/test_generate.py`
    - **`latest`:** `uv run --upgrade-package transformers --group test pytest tests/test_loader.py
      tests/test_generate.py`. `uv.lock` pins transformers, so without this the CI never sees a newer release.
  - Run both commands locally. Expected: both pass.
  - If transformers 5.3.0 or torch 2.8.0 fails, stop and report it to the user instead of changing a floor.
- [ ] **Step 6: Commit.** `feat: loader propio con codec solo decodificador y kwargs del Hub`.

---

### Task 7: Sentence streaming (§7)

**Files:**
- Modify: `omnivoice_tts/models/omnivoice.py`, `tests/test_split_sentences.py`
- Create: `tests/data/split_cases.json`, `tests/test_streaming.py`

**Interfaces:**
- Produces:
  - `OmniVoice.generate_streaming(...) -> Iterator[tuple[np.ndarray, str]]`. It is §7's code block, from `def
    generate_streaming(` through `generation_config=gen_config, generator=generator)[0], segment`.
  - The module-level `_SENTENCE_END`, `_SENTENCE_ABBREVIATIONS` and `_split_sentences(text: str) -> list[str]`, placed
    next to `_combine_text`. They are the same code block, from `# ---- module level` through `return out`.
- Use content anchors, not spec line numbers: those move whenever the spec changes.
- The imports gain `from collections.abc import Callable, Iterator` and `ABBREVIATIONS`.

- [ ] **Step 1: Build the splitter fixture (P13)**

```bash
. ./.plan-env.sh
"$KV" -I -B -c '
import json, sys
sys.path.insert(0, "/home/kimox/.claude/projects/-home-kimox-orca-workspaces-omnivoice-tts-inference-feat-initial-implementation/afd1a81d-ce0d-4c3f-b201-4690f4fd694d/stream/scratch/segmenter/scripts")
from cases import CASES
extra = [("ell", "Bueno... ¿qué querés que te diga?", ["Bueno... ¿qué querés que te diga?"]),
         ("quote", "Bueno... \"no sé\" dijo.", ["Bueno... \"no sé\" dijo."]),
         ("ell", "Hola… ¿cómo andás?", ["Hola… ¿cómo andás?"]),
         ("ell", "Esperá… (ya vuelvo).", ["Esperá… (ya vuelvo)."]),
         ("es", "¡Hola! ¿Qué tal?", ["¡Hola!", "¿Qué tal?"])]
json.dump([list(c) for c in CASES + extra], open("tests/data/split_cases.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(len(CASES) + len(extra))'
```
Expected: `73`.

- [ ] **Step 2: Write the failing splitter tests**
  - **`test_cases`.** For every `[cat, text, gold]`:
    - `"".join(out) == text`;
    - `got = out if len(out) == 1 else [p.strip() for p in out]`;
    - `got == (gold if cat == "ident" else [g.strip() for g in gold])`.
  - **`test_single_sentence_unchanged`.** `"Hola, ¿qué tal?"`, `"  Hola  "`, `"Son las 21:30 y llegué."`,
    `"[laughter] jaja"`, `"..."` and `""` each give `[t]`.
  - **`test_fuzz`.** The parts and the 4 invariants come from
    `$EV/stream/scratch/stream-critic/scripts/fixed_variant_check.py:54-75`: 20,000 strings with
    `random.Random(20261020)`, using the library's `_NONVERBAL_PATTERN`.
- [ ] **Step 3: Write the failing streaming tests** (`tests/test_streaming.py`; stubs from `tests/stubs.py`)
  - **`test_one_sentence_is_generate`.**
    - `"Hola che, ¿todo bien?"` gives one item, whose array and calls equal a fresh stub's `generate(text, "es")[0]`;
    - the item's text `is` the input string; `"  Hola  "` keeps its whitespace.
  - **`test_sentences_are_independent_generate_calls`.** With `Recorder(draw=True)` and `"Hola. ¿Qué tal? Bien."`:
    - the stream after `torch.manual_seed(5)` equals a loop of `generate()` over the stripped sentences after
      `torch.manual_seed(5)`;
    - the same holds with two generators seeded 5;
    - item 1 differs from `generate("¿Qué tal?")` with a fresh seed of 5.
  - **`test_hooks_order`.**
    - With `language="Spanish"`, normalize gets `(stripped sentence, "es")`.
    - normalize returns `s + " uno"`, and duration gets that output.
    - With duration returning 1.234, the recorded target is `[30]`, i.e. `max(1, int(1.234 * 25))`.
    - The recorded text is the normalized one, and the yielded text stays raw.
  - **`test_duration_must_be_callable`.** For `3.0`, `3`, `np.float64(3.0)` and `True`:
    - creation records nothing;
    - the first `next()` raises TypeError;
    - the next `next()` raises StopIteration.
  - **`test_config_fields_checked_first`.**
    - `num_steps=8` and `normalize_text=True` raise TypeError at the first `next()`;
    - a wrapper around the module's `_split_sentences` and a normalize spy record no call.
  - **`test_lazy_and_closes`.**
    - creating the stream runs nothing;
    - `close()` before the first item ends the stream with no call;
    - `close()` or `del` plus `gc.collect()` after the first item adds no call.
  - **`test_error_mid_stream`.**
    - normalize raises `RuntimeError("tn")` on sentence 2;
    - `next()` #1 returns item 1, and `next()` #2 raises it;
    - item 1's array is unchanged; then StopIteration.
  - **`test_without_speech_raises`.** `""`, `"   "`, `"..."`, `"😀"` and `"¿?"` raise ValueError at the first `next()`,
    and normalize is never called.
  - **`test_lone_tag_is_spoken`.** `"[laughter]"` gives one item, equal to `generate("[laughter]")[0]`, with text
    `"[laughter]"`.
  - **`test_skips_sentences_tn_empties`.**
    - `"Hola. <3. Chau."`, with normalize returning `""` for `"<3."`, gives the texts `["Hola.", "Chau."]`, with 2
      generate calls and 2 duration calls;
    - with a normalize that always returns `""`, `list(stream) == []` and there is no call.
  - **`test_rejects_non_str`.** `["Hola."]`, `b"Hola."`, `None` and `3` raise TypeError at the first `next()`.
  - **`test_caller_outside_inference_mode`.**
    - the hooks see `torch.is_inference_mode_enabled()` False, and `Recorder.inference` holds only True;
    - the caller sees False between yields;
    - iterating inside `torch.inference_mode()` also works, and the items are `np.ndarray`.
  - **`test_long_sentence_is_one_chunked_item`** (Review Focus 4). Take `LONG = ("La verdad que no sé qué decirte, " *
    30) + "fin."`.
    - It is one sentence, about 57 s by the rule (1,437 frames), and the stub records 5 calls for it.
    - `LONG_TEXTS[0]` and `[2]` would not do: they hold many sentences.

    The text `LONG + " ¿Y ahora?"` gives 2 items:
    - item 0 has the text `LONG` and comes from 5 recorded calls;
    - item 1 has the text `"¿Y ahora?"`.
- [ ] **Step 4: Run them and see them fail.** Expected: AttributeError (no `generate_streaming`, no `_split_sentences`).
- [ ] **Step 5: Implement.** Copy both §7 code blocks named in Interfaces verbatim. They already use §13's
  annotations.
- [ ] **Step 6: Run the whole suite.** Everything passes.
- [ ] **Step 7: Commit.** `feat: generate_streaming por oración con normalize y duration por oración`.

---

### Task 8: int8 (§10)

**Files:**
- Create: `omnivoice_tts/models/omnivoice_int8.py`, `tests/test_int8.py`, `tests/data/int8_kohai_per_layer_sha256.json`

**Interfaces:**
- Produces:
  - `class Int8Linear(nn.Module)`: `__init__(self, linear: nn.Linear)`, with the buffers `qweight` (int8, out×in) and
    `scale` (the weight's dtype, out×1); `forward(self, x) -> torch.Tensor`;
  - `apply_int8(model) -> model`.

- [ ] **Step 1: Copy the fixture**

```bash
. ./.plan-env.sh
"$KV" -I -B -c 'import json; d = json.load(open("'"$EV"'/bench2/perf/raw/listen_B.json"))["int8"]
json.dump(d["per_layer_sha256"], open("tests/data/int8_kohai_per_layer_sha256.json", "w"), indent=1)'
```

- [ ] **Step 2: Write the failing tests**
  - **`test_roundtrip_error_bound[float16, float32]`.** For an `nn.Linear(64, 96)` with an outlier row and a zero
    row:
    - `qweight` is int8 within ±127, and `scale` has the weight's dtype and shape `(96, 1)`;
    - `(qweight.float() * scale.float() - w.float()).abs() <= 0.63 * scale.float() + 1e-12`.
  - **`test_quantization_formula`.**
    - With `s = w.float().abs().amax(1, keepdim=True).clamp_min(1e-8) * (1 / 127)`, `torch.equal(qweight,
      torch.round(w.float() / s).to(torch.int8))` and `torch.equal(scale, s.to(w.dtype))`.
    - The fp16 row `[[0.0106658935546875, 0.00533294677734375]]` quantizes to `[[127, 64]]`. slim.py's CPU `/ 127`
      would give `[[127, 63]]`.
  - **`test_forward_is_one_expression`.** `torch.equal(m(x), F.linear(x, m.qweight * m.scale, m.bias))`, in fp16 and
    in fp32.
  - **`test_apply_int8_scope_and_idempotence`.** On `tiny_model(torch.float16)`:
    - 7 `Int8Linear` per layer (14 in all) and no `nn.Linear` left in `llm.layers`;
    - `audio_heads` is still an `nn.Linear`; embeddings and norms are fp16; `inv_freq` is fp32;
    - the codec modules are the same objects;
    - a second `apply_int8` changes nothing and returns the model;
    - the `cfg_batch()` logits' relative error against fp16 is below 0.05.
  - **`test_int8_calls_per_llm_forward`.** Forward hooks count 14 `Int8Linear` calls and 0 `nn.Linear` calls inside
    the layers during one `model.llm` forward. This is the CPU version of G8's count.
  - **`test_int8_model_moves_whole`** (Review Focus 2):
    - no module holds a plain tensor attribute: `[k for m in model.modules() for k, v in vars(m).items() if
      isinstance(v, torch.Tensor)] == []`;
    - after `model.llm.to("meta")`, every parameter and buffer `is_meta`.
  - **`test_kohai_fixture_integrity`.** The fixture has 28 rows, and `sha256(json.dumps(rows).encode()).hexdigest()
    == "d8e4b4726f653815be095faa661c3ca3ffee8f35404bea4100771f3b3f0d5011"`.
  - **`test_kohai_per_layer_sha256`.**
    - It skips unless `OMNIVOICE_TEST_KOHAI_DIR` is set.
    - It runs `from_pretrained(dir, dtype=torch.float16)` on CPU, then `apply_int8`.
    - The rows must equal the fixture. Compute them with bench2's recipe, verbatim from
      `$EV/plan/understand/flashinfer-int8.md` section D:
      - `GROUPS` maps `qkv`, `o`, `gate_up` and `down`, each to the separate or the fused module names;
      - each row holds one sha256 per group and kind, over `.detach().cpu().contiguous().numpy().tobytes()` of
        `qweight` and `scale`.
- [ ] **Step 3: Run them and see them fail.** Expected: ModuleNotFoundError (`omnivoice_tts.models.omnivoice_int8`).
- [ ] **Step 4: Implement**
  - Copy the whole 50-line `$EV/design2/scripts/int8/omnivoice_int8.py`: docstring, imports, `Int8Linear` and
    `apply_int8`. Then edit it:
    - delete the sketch comment on line 1;
    - delete the docstring's "Cannot be combined with ``apply_flashinfer``" (line 14);
    - delete the `_fi_runner` guard (lines 43-44), since FlashInfer then int8 is allowed (§10, decision #8);
    - update the cost comment (lines 36-37) to bench2's figures: RTF +8 % on sdpa (0.0530 → 0.0572) and +31-34 % on
      FlashInfer.
  - Keep the module docstring short: what it converts, that it is idempotent, and "a quantized model cannot be saved".
- [ ] **Step 5: Run the whole suite.** Everything passes. Also run `. ./.plan-env.sh &&
  OMNIVOICE_TEST_KOHAI_DIR="$KOHAI" uv run --group test pytest tests/test_int8.py -q`. `HF_HUB_OFFLINE=1` comes from
  `.plan-env.sh`, so the codec fallback reads the HF cache. Expected: the KohAI hash test passes, in about 10 s.
- [ ] **Step 6: Commit.** `feat: modo int8 por canal con escala en fp32`.

---

### Task 9: Vendored FlashInfer without global state (§9)

**Files:**
- Modify (replace): `omnivoice_tts/models/omnivoice_flashinfer.py`
- Create: `tests/test_flashinfer_cpu.py`

**Interfaces:**
- Consumes:
  - `GenerationTask`, `OmniVoiceGenerationConfig` and `_get_time_steps`;
  - `_gumbel_sample(logits, temperature, generator)`;
  - `_prepare_inference_inputs`, `_prepare_embed_inputs` and `_predict_tokens_with_scoring(c, u, gen_config,
    generator)`;
  - `Int8Linear` (in the tests).
- Produces:
  - `apply_flashinfer(model) -> model`;
  - `PackedAttnRunner(num_qo_heads, num_kv_heads, head_dim, dtype)`, with `.wrapper`, `.pos_ids` and `.plan(doc_lens,
    device)`;
  - `model._fi_runner`;
  - `_generate_iterative_packed(self, task, gen_config, generator=None)`, bound over `_generate_iterative`.

- [ ] **Step 1: Write the failing tests**
  - **`test_import_error_names_the_install_command`.** It runs everywhere.
    - `monkeypatch.setitem(sys.modules, "flashinfer", None)`;
    - `monkeypatch.delitem(sys.modules, "omnivoice_tts.models.omnivoice_flashinfer", raising=False)`;
    - importing the module raises ImportError matching `re.escape("flashinfer-jit-cache==0.6.15.post1+cu128")`.
  - **The other tests** skip with `skipif(importlib.util.find_spec("flashinfer") is None)`, so collection never
    imports flashinfer.
    - A module fixture imports the module under `pytest.MonkeyPatch.context()`, with `HOME` and `TMPDIR` set to a temp
      dir.
    - A `fake_cuda` fixture sets `torch.cuda.is_available` to True, `torch.cuda.get_device_capability` to `(12, 0)`
      and `flashinfer.jit.current_compilation_context.TARGET_CUDA_ARCHS` to `{(12, "0a")}`.
    - The model is `tiny_model(torch.float16)`.
  - **`test_fusion_frees_originals_and_keeps_4_linears_per_layer`.** After `apply_flashinfer`, without wrapping it in
    `no_grad`:
    - weakrefs to the q/k/v/gate/up weights are dead after `gc.collect()`;
    - `qkv_proj.weight` equals `torch.cat` of the old q/k/v (taken before), and `gate_up_proj.weight` likewise;
    - each layer has exactly 4 `nn.Linear`, the fused weights have `requires_grad` False and `grad_fn is None`, and
      `sum(p.numel() for p in model.parameters())` is unchanged;
    - `model._fi_runner.wrapper is None`, i.e. the runner is lazy.
  - **`test_guards_leave_the_model_untouched`.**
    - The cases:
      - no CUDA: ValueError;
      - fp32 weights: ValueError;
      - after `apply_int8`, with CUDA faked: TypeError matching `apply_int8`;
      - an empty arch set: RuntimeError matching `FLASHINFER_CUDA_ARCH_LIST`.
    - After each one, `_fi_runner` is absent, `q_proj` is present and `"_generate_iterative" not in vars(model)`.
  - **`test_flashinfer_then_int8`.** 4 `Int8Linear` per layer, 0 `nn.Linear` in the layers, and no plain tensor
    attributes.
  - **`test_apply_flashinfer_is_idempotent`.** A second call returns the same model, with the same runner object.
  - **`test_patched_model_moves_whole`** (Review Focus 2). The same check as in Task 8, after `apply_flashinfer` and
    again after `apply_int8`.
  - **`test_two_models_share_no_state`** (Review Focus 3).
    - Patching model a leaves model b without `_fi_runner`, with its class `_generate_iterative`, and with its
      original modules.
    - The module has no `_CTX` attribute and no module-level tensors.
    - Patching b gives a runner that `is not` a's.
- [ ] **Step 2: Run them and see them fail.** In the repo venv: `uv run --group test pytest tests/test_flashinfer_cpu.py
  -q`. Expected: the ImportError test fails, because the verbatim module imports flashinfer unwrapped, and the rest
  skips.
- [ ] **Step 3: Implement**
  - Replace the file with `$EV/design2/scripts/flashinfer/sketch/omnivoice_flashinfer.py` (402 lines; its CPU checks
    pass).
  - Then apply these edits:
    1. Line 3 is the exact notice: `Modified by`, not `Modified for`.
    2. The TypeError reads `apply_flashinfer: q_proj is not an nn.Linear; call it before apply_int8`.
    3. Split the ValueError: "needs CUDA" and "needs fp16 or bf16 weights, got …".
    4. Convert `List` and `Optional` to the §13 style, and rename the `l` at SK:214 to `n`.
    5. The docstring names both `generate()` and `generate_streaming()`.
    6. Add `# ponytail: the runner's buffers are allocated on the first plan, on the model's device at that time; move
       the model before the first generate()`.
  - Keep the ImportError message, NVIDIA's two SPDX lines, the `@torch.no_grad()` on `apply_flashinfer`, and every
    deletion the sketch already made.
- [ ] **Step 4: Run in the reference venv, which has flashinfer**

```bash
. ./.plan-env.sh
uv pip install --target "$SP/pytest-site" pytest   # once
PYTHONPATH="$PWD:$SP/pytest-site" "$KV" -B -m pytest tests/test_flashinfer_cpu.py tests/test_int8.py -q
```
Expected: everything passes, with none skipped. Also run `uv run --group test pytest tests -q` in the repo venv:
everything passes there, and the FlashInfer tests skip except the ImportError one.
- [ ] **Step 5: Commit.** `feat: FlashInfer vendorizado, sin grafos ni estado global`.

---

### Task 10: README, trimmed docs and the benchmark script (§3, §14, §15)

**Files:**
- Modify: `README.md`, `docs/generation-parameters.md`, `docs/voice-design.md`
- Create: `scripts/benchmark.py` (repo root, not in the wheel). It is written here, on CPU, because §3 lists it as a
  repo file; Task 13 runs it on the GPU.

The TN section and the KohAI example come with Task 11.

- [ ] **Step 1: Write README.md** with §15's outline, minus the TN section and the KohAI example.
  - **Install and quick start:**
    - `pip install omnivoice-tts-inference`;
    - `OmniVoice.from_pretrained(..., device_map="cuda:0", dtype=torch.float16)` and `generate()`;
    - one warm-up call at startup;
    - "never `.half()` or `.to(dtype)` after loading".
  - **Generation parameters and voice design.** Unknown items pass through lowercased, and typos get no suggestion.
  - **Non-verbal tags and pronunciation.** The 13 tags, CMU `[...]` overrides, and that pinyin overrides are not
    protected by TN.
  - **Streaming**, every bullet of §15's streaming list:
    - the item contract;
    - the player-thread and asyncio patterns (one worker thread, `submit(stream.close)` in `finally`);
    - the measured gains and costs (≈ 4 ms at sdpa, ≈ 2× TTS time; FlashInfer first audio ≈ 0.10 s; whole reply before
      sdpa's first sentence up to ≈ 10 s; 0.43 s and 0.58 s next to the LLM);
    - laughs at sentence starts; strip markdown and list markers;
    - voices vary between sentences (#255);
    - Pipecat's SpokenSentences;
    - the join knobs;
    - the difference from PR #166;
    - the splitter's known ceilings, from §7's list.
  - **FlashInfer and int8 recipes.**
    - The CPU-first recipe, the §9 wheels, the `[[tool.uv.index]]` snippet and `FLASHINFER_CUDA_ARCH_LIST`
      (`12.0a`/`12.0f`).
    - §15's table verbatim, labelled "measured on an emulation of the library, to re-measure".
    - "When to use which", and the blind-listening sentence.
  - **Next to an LLM and faster-whisper**, every bullet of that §15 list:
    - time-slicing;
    - faster-whisper in its own process;
    - `OPENBLAS_NUM_THREADS=1` or a high-priority stream;
    - CPU-heavy work out of the TTS process;
    - "the library sets none".
  - **Host memory.** cuDNN plans and `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT`.
  - **Threads and the global-state contract.** All verbatim:
    - §6 and §7's thread rules;
    - §11's "The library never…" paragraph and its neighbors paragraph;
    - §11's "Residual effects of dependencies" list.
  - **Numbered differences from upstream.** §16's fixes 1-10, §4's removals and signature changes, and §17's
    "different by design" list.
  - **What v0.1 does not fix.** §16's "Not fixed in v0.1" list.
  - **Licenses** (§14). Code under Apache-2.0; base weights CC-BY-NC; the codec under the Boson Higgs Audio 2
    Community License.
  - **Tips.** Upstream's `docs/tips.md` tips 2 and 3.
  - **Citation and disclaimer.** Upstream's README:408-423.
- [ ] **Step 2: Trim the docs**
  - **Both files:**
    - add `<!-- Modified by omnivoice-tts-inference from k2-fsa/OmniVoice 08be0b4 -->` as line 1;
    - the import becomes `from omnivoice_tts import ...`.
  - **`generation-parameters.md`:**
    - drop the `preprocess_prompt` row and the batch-list wording;
    - types become `float | None`;
    - fix line 65: chunking starts above `audio_chunk_threshold`;
    - add "a value ≤ 0 disables chunking" (fix 5).
  - **`voice-design.md`.** Reword "only one attribute at a time" as advice, and add "unknown items pass through,
    typos get no suggestion".
- [ ] **Step 3: Write `scripts/benchmark.py`.** Follow the shape of indextts25-inference's (`git -C
  /home/kimox/Projects/indextts25-inference show origin/main:scripts/benchmark.py`, 74 lines).
  - **Arguments.** `--model`, `--backend sdpa|flashinfer`, `--int8`, `--steps 32` and `--reps 5`.
  - **Recipe.** For FlashInfer and int8 it uses the CPU-first recipe: load on CPU, patch, then `.to("cuda:0")`, then
    one warm-up call.
  - **Measures** (G4 and §15's table), printed as medians:
    - load time, with CUDA sync;
    - the load peak, with `torch.cuda.reset_peak_memory_stats()` before the load and `max_memory_allocated()` after
      it. Expect no 806 MB transient;
    - `memory_allocated()` after the load and after the warm-up (resident), and `memory_reserved()`;
    - this process's `nvidia-smi` used MiB, read right after the ≤ 6 s text. §15's figure is bench2's "after M0";
    - the cold first `generate()`;
    - the peak on a text of ≤ 6 s and on a 20 s text;
    - time to first item of `generate_streaming()` on a typical KohAI reply, on `SHORT` and on `PARAGRAPH`;
    - `generate()` RTF on `PARAGRAPH`. That text is about 29 s by the rule, past sdpa's L = 320 knee, so this is not
      §15's RTF. Task 13 measures that one separately.
  - **Profiling.** It never runs `torch.profiler` (§19's rule).
  - **Smoke check.** Run `uv run python scripts/benchmark.py --help`; expected: the usage, with no CUDA needed.
- [ ] **Step 4: Verify**
  - `grep -n "normalize_text\|ref_audio\|voice_clone\|omnivoice-demo\|soundfile" README.md docs/*.md` prints only
    deliberate mentions: the "differences" list, and soundfile as "not a dependency".
  - Every number in the tables matches §15.
  - No URL of the listening page appears.
- [ ] **Step 5: Commit.** Stage `README.md`, `docs/generation-parameters.md`, `docs/voice-design.md` and
  `scripts/benchmark.py`, with the message `docs: README de la librería, docs recortados y script de benchmark`.

---

### Task 11: Spanish text normalization (§8; the last implementation task, decision #17)

**Prerequisite (the user's step).** PR #6 of nicokim/indextts25-inference must be merged, with `# Copyright 2026
Kimox` and `# SPDX-License-Identifier: Apache-2.0` as the first two lines of `indextts25/utils/es_rules.py`.

Check with `gh pr view 6 --repo nicokim/indextts25-inference --json state,mergeCommit --jq '.state + " " +
.mergeCommit.oid'`.
- Expected: `MERGED <sha>`.
- Otherwise stop and ask the user. §8's alternative is to copy the PR #6 commit that adds the header, pin it, and
  re-copy at the merge.

**Files:**
- Create: `omnivoice_tts/utils/es_rules.py`, `tests/test_text_normalizer.py`, `tests/test_es_normalization.py`,
  `tests/data/tn_probes.json`
- Modify: `omnivoice_tts/utils/text.py`, `tests/conftest.py`, `tests/test_streaming.py`, `.github/workflows/ci.yml`,
  `README.md`

**Interfaces:**
- Produces:
  - `TextNormalizer(cache_dir: str | None = None)`, with `.normalize(text: str, language: str | None) -> str`;
  - `_spanish_nemo(cache_dir)`, `_quiet_nemo()`, `_normalize_segment(fn, segment)` and `_apply_with_protection(text,
    fn)`;
  - `_BRACKET_TAG_RE` and `_TN_INSTALL_MSG`.

  All of them are the code block under §8's "Code: the TN section of `utils/text.py`". Locate it by that heading, not by
  line numbers.
- Produces in `tests/conftest.py`: a session fixture `tn_cache` (`$OMNIVOICE_TN_CACHE`, or a session temp dir) and a
  session fixture `tn` (`TextNormalizer(cache_dir=tn_cache)`), so NeMo's ~20 s build runs once per session.

- [ ] **Step 1: Copy the rules and the PR #6 suite**

```bash
. ./.plan-env.sh
MERGE=<the merge commit's sha, from the prerequisite>
gh api -H "Accept: application/vnd.github.raw" \
  "repos/nicokim/indextts25-inference/contents/indextts25/utils/es_rules.py?ref=$MERGE" > omnivoice_tts/utils/es_rules.py
head -2 omnivoice_tts/utils/es_rules.py; sha256sum omnivoice_tts/utils/es_rules.py
{ sed -n '1,19p' "$EV/tn/scratch/tn-critic/tests/test_es_normalization.py"
  gh api -H "Accept: application/vnd.github.raw" \
    "repos/nicokim/indextts25-inference/contents/tests/test_es_normalization.py?ref=$MERGE" | sed -n '/^def check/,$p'
} > tests/test_es_normalization.py
cp "$EV/tn/scratch/tn-critic/out/inputs_probes.json" tests/data/tn_probes.json   # 109 strings
```
Expected: the two header lines, and a sha256 to pin.

- [ ] **Step 2: Write the failing tests** (`tests/test_text_normalizer.py`; a `*` test needs no NeMo, the rest skip
  without `nemo_text_processing`)
  - **`test_tags_survive_byte_for_byte`**, parametrized over the 109 probes, with the session `tn`:
    - the list of `[...]` spans and the list of `_NONVERBAL_PATTERN` tags are unchanged;
    - no digit is left outside `[...]` spans and `<x|y>` annotations; strip those with es_rules' `_ANNOTATION` regex
      first;
    - the output is idempotent.
    - Exact strings:
      - `'Ese [B EY1 S] cuesta $450.'` → `'Ese [B EY1 S] cuesta cuatrocientos cincuenta pesos.'`
      - `'[laughter]21:30 arrancamos.'` → `'[laughter]veintiuna y media arrancamos.'`
      - `'  Hola [laughter]  21 años  '` → `'  Hola [laughter]  veintiún años  '`
  - **`test_only_spanish_is_normalized`.**
    - `"Tengo 2 gatos."` comes back unchanged for `"en"`, `"fr"`, `None`, `"spa"` and `"es-AR"`;
    - it becomes `"Tengo dos gatos."` for `"ES"`, `"Spanish"` and `" es "`;
    - `tn.normalize("x")` raises TypeError.
  - **`*test_non_spanish_needs_no_nemo`.** In a `python -I` subprocess, with every `nemo_text_processing*` key set
    to `None` in `sys.modules`:
    - `"en"` and `None` come back unchanged;
    - `"es"` raises ImportError, twice, whose message contains `omnivoice-tts-inference[tn]`.
  - **`*test_cache_loads_only_own_private_pickle`.** Use a fake
    `nemo_text_processing.text_normalization.normalize.Normalizer` in `sys.modules`, with a build counter, and set
    `omnivoice_tts.utils.text.version = lambda _: "0"`.
    - Planted 0664 and 0646 payloads, and a symlink to a payload, are never unpickled. Each payload's `__reduce__`
      would write a marker file. Each one is replaced by a regular 0600 file.
    - A corrupt 0600 file is rebuilt.
    - Our own 0600 cache loads without a build.
    - A directory the library creates is 0700.
    - An unwritable `cache_dir` raises OSError, and the build counter stays 0. Skip this case when `os.geteuid() ==
      0`.
    - No `.tmp` file is left behind.
  - **`test_tn_is_silent_and_global_free`.** A `python -I` subprocess runs twice, with a cold and then a warm temp
    `cache_dir`, and `HOME`, `XDG_CACHE_HOME` and `TMPDIR` in tmp. In each run:
    - the app imports NeMo and sets its logger to ERROR, then snapshots: NeMo's level, handlers and filters, the root
      handlers, `os.environ`, `sys.getrecursionlimit()` and `warnings.filters`;
    - it then normalizes `'[laughter]21:30 arrancamos.'`;
    - expected: empty fd 1 and fd 2, an unchanged snapshot, and a 0600 pickle.
  - **`*test_importing_text_does_not_import_nemo`.** In a subprocess, after `import omnivoice_tts.utils.text`,
    `"nemo_text_processing" not in sys.modules`.
  - **`*test_es_rules_is_the_indextts25_copy`.** The file's sha256 equals the pinned value, and its first two lines are
    the header.
- **`tests/test_streaming.py::test_normalize_keeps_raw_subtitles`.** It skips without NeMo, uses the session `tn`, and
  sets `HOME` and `XDG_CACHE_HOME` in tmp.
  - `'Son las 21:30. Me donaron $1.500.'` gives 2 items with the raw texts.
  - The recorded texts contain no digit.
- [ ] **Step 3: Run them and see them fail**

Run: `. ./.plan-env.sh && OMNIVOICE_TN_CACHE="$SP/tn-cache" uv run --extra tn --group test pytest tests -q`
Expected: ImportError (`cannot import name 'TextNormalizer'`).
- [ ] **Step 4: Implement**
  - **Imports.** In `utils/text.py`, add back `import re`: Task 3's ruff pass removed it, and `_BRACKET_TAG_RE` needs
    it. Then add §8's imports, from the paragraph "The header imports gain…". The block ends as:
    - `contextlib`, `logging`, `os`, `pickle`, `re` and `tempfile`;
    - `from importlib.metadata import version`;
    - `from omnivoice_tts.utils import es_rules`;
    - `from omnivoice_tts.utils.lang_map import LANG_NAME_TO_ID`.
  - **TN section.** Add the code block under §8's "Code: the TN section of `utils/text.py`" heading, verbatim.
  - **Docstring.** Add the module-docstring bullet for `TextNormalizer`.
- [ ] **Step 5: Run the whole suite**

Run: `. ./.plan-env.sh && OMNIVOICE_TN_CACHE="$SP/tn-cache" uv run --extra tn --group test pytest tests -q`
Expected: everything passes, including the 647 cases of the PR #6 suite.
- [ ] **Step 6: Update CI and the README**
  - `ci.yml`'s test step gets `env: OMNIVOICE_TN_CACHE: ${{ runner.temp }}/omnivoice-tn`.
  - In the README, add §15's TN section:
    - install and platforms, with the compiler and the ~0.7 GB;
    - PR #6's "Spanish numbers" paragraph, from README.md:71-88 at dfb3286f. Re-read it at the prerequisite's merge
      sha with `gh api -H "Accept: application/vnd.github.raw"
      "repos/nicokim/indextts25-inference/contents/README.md?ref=<sha>"`. It is verbatim, except that
      `indextts25/utils/es_rules.py` becomes `omnivoice_tts/utils/es_rules.py`;
    - the locale table, and what es-ES and the dollarized varieties should do;
    - tags and `[...]`;
    - the cache, its security and prebuilding;
    - threads and the GIL, and §8's contract:
      - one call at a time per instance;
      - two instances in two threads keep correct outputs, but can leave NeMo's logger at INFO. This is the one
        exception of §1 criterion 2;
      - NeMo's hidden failures return the raw segment, with the warning at DEBUG. A caller that must never speak
        digits checks for digits outside the brackets;
    - "compute any duration on the normalized text";
    - G-TN1's result and the known gaps.
  - Also add the KohAI example, which is §15's "KohAI example as caller config" bullet with its prebuild command and
    code block, and the TN items of the numbered differences.
- [ ] **Step 7: Re-run the one-off TN checks at the merge commit (§18).** They stay out of CI.
  - **Scripts.** Run `$EV/tn/scratch/tn-critic/scripts/` `run_impl.py` with `compare.py` (corpus, PR #6 cases, probes,
    fuzz, tagged labels), plus `silence.py`, `cache_guard.py`, `errors_check.py` and `duration_check.py`.
    - Use `$KV` with `PYTHONPATH=$PWD`.
    - Point their package path at this repo.
  - **Timing.** Time `TextNormalizer.normalize` after the library's own cache load, in a fresh process. Expected: about
    11.9 ms median over the 30 replies; 7.6 ms for a 2-sentence reply's first sentence and 17.2 ms for the whole reply.
  - **Recording.** Write the outputs to `$EV/tn-lib/` and report any difference from §8's table.
- [ ] **Step 8: Check the size (§1 criterion 4)**

Run: `find omnivoice_tts -name '*.py' ! -name es_rules.py | xargs wc -l | tail -1`
Expected: about 3.0k lines. Report the figure.
- [ ] **Step 9: Commit.** `feat: normalización de texto en español (NeMo más las reglas de indextts25)`.

---

### Task 12: GPU parity and contract checks on the library (§19; only with the user's OK)

**Prerequisite.** The user's explicit OK, and `nvidia-smi` showing room. Never stop another process.

**Environment**
- `. ./.plan-env.sh`, then `CUDA_VISIBLE_DEVICES=0`. The env file sets it empty for the CPU tasks.
- `$KV` with `PYTHONPATH="$PWD:$SP/pytest-site"`, so upstream and the library run with the same torch and transformers.
- `HF_HUB_OFFLINE=1` (from the env file), `XDG_CACHE_HOME=$SP/xdg` and `FLASHINFER_CUDA_ARCH_LIST=12.0a` (cu128).
- §19's rule for every timing: `torch.profiler` runs last or in its own process. One CUDA profiler session slows every
  later call in that process, by +26 % on sdpa and +14 % on FlashInfer.
- The bench lines and the KohAI rule come from `$EV/bench2/perf/perf.py:55-77` (`LINES`, `rule`) and `pcm_hash`
  (:79).
- Results go to `$EV/gpu-lib/`.

**Files:**
- Create: `tests/test_gpu.py`, gated by `OMNIVOICE_TEST_KOHAI_DIR` and `torch.cuda.is_available()`
- Create (evidence, not in the repo): `$EV/gpu-lib/{g1_parity,g3_stream,g5_isolation,g11_state}.py` and
  `$EV/gpu-lib/RESULTS.md`

- [ ] **Step 1: `tests/test_gpu.py`**
  - **`test_private_generator_matches_global_seed_gpu`** (G2):
    - tokens are equal under `torch.manual_seed(s)` and under `torch.Generator("cuda:0").manual_seed(s)`;
    - `torch.cuda.get_rng_state()` is unchanged with the private generator;
    - a CPU generator raises.
  - **`test_stream_single_sentence_parity_gpu`** (G3a), on sdpa and, when flashinfer is installed, on FlashInfer: for
    the 11 bench lines the splitter keeps whole, the one item's int16 PCM sha256 equals `generate(t, "es",
    instruct=e, duration=dur(t))[0]`'s, with a global seed and with a private generator.
  - Run: `. ./.plan-env.sh && CUDA_VISIBLE_DEVICES=0 OMNIVOICE_TEST_KOHAI_DIR="$KOHAI"
    PYTHONPATH="$PWD:$SP/pytest-site" "$KV" -B -m pytest tests/test_gpu.py -q`. Expected: everything passes.
- [ ] **Step 2: G0.** Arch check on `$KV` (cu128 + `12.0a`) and on `$FV` (cu129 + `12.0f`). Without the env var,
  `apply_flashinfer` raises the RuntimeError before patching.
- [ ] **Step 3: G1 and G10.** `g1_parity.py` runs upstream's `OmniVoice` and the library in one process, sdpa fp16,
  default config, with the global seed set per text.
  - **Texts.** The 12 bench lines (KohAI rule), 2 newline-free texts over 30 s (`duration=None`) and 1 CJK text.
  - **Stages.** It compares step by step: loader tensors, then codec decode, then the int16 PCM sha256.
  - **Pass.** 15/15 hashes equal.
  - **If one differs.** Bisect with §19's list: loader, codec, global-state removals, str-only plumbing, generator,
    `MODEL_MAPPING`, `use_cache`.
- [ ] **Step 4: G3b and G3e** (`g3_stream.py`).
  - **G3b.** The set is 5 KohAI replies of 2-3 sentences, plus the 9th bench line, which `_split_sentences` splits in
    two (§19 G3a).
    - Each item's hash equals that of a `generate()` loop over the stripped sentences with one shared generator.
    - With `normalize=tn.normalize`, no digit reaches the model.
  - **G3e.** After `close()` following the first item, the next stream reproduces its solo hashes. The asyncio pattern
    survives `task.cancel()`.
- [ ] **Step 5: G5.** FlashInfer, built CPU-first.
  - Its 12 hashes equal upstream FlashInfer's, run eager.
  - Isolation: two FlashInfer instances alternating, plus one sdpa, each reproduce their solo hashes.
- [ ] **Step 6: G11** (`g11_state.py`).
  - **Capture.** fd-level, around load, generate, stream, FlashInfer and int8.
  - **Unchanged afterwards.** env, warnings filters, logging, default dtype, the CUDA RNG with a private generator, and
    `torch.get_num_threads()`/`get_num_interop_threads()`.
  - **Pass.** Nothing is printed, and nothing changes.
- [ ] **Step 7: Record and commit**
  - Write `RESULTS.md` (checks, pass/fail, hashes) in the evidence dir.
  - In the repo, commit only `tests/test_gpu.py`: `test: chequeos de paridad en GPU`.

---

### Task 13: GPU performance, memory and soak on the library (§19; only with the user's OK)

**Prerequisite and environment.** The same as Task 12, including §19's profiler rule. Count Int8Linear calls with
forward hooks, never with the profiler.

**Files:**
- Modify: `README.md` (the measured table). Also `scripts/benchmark.py`, but only if Step 1 has to fix it, in its own
  commit.
- Create (evidence): `$EV/gpu-lib/perf/`

- [ ] **Step 1: Check the benchmark.** Run Task 10's `scripts/benchmark.py` once on A, with `--reps 1`, to check that
  it runs on the GPU. If it does not, fix it and commit it on its own (`fix: benchmark en GPU`).
- [ ] **Step 2: G4, G6, G7, G8 and G9.** Run it per config: A sdpa fp16, B sdpa int8, C FlashInfer fp16, D FlashInfer
  + int8. Expected, within the session noise of §20:

  | Config | Resident GiB | Peak GiB (≤ 6 s / 20 s) | nvidia-smi MiB | RTF | Cold first call | First sentence |
  |---|---|---|---|---|---|---|
  | A | ≈ 1.23 | ≈ 1.33 / 1.56 | ≈ 2,014 | ≈ 0.053 | ≈ 0.54 s | ≈ 240 ms |
  | B | ≈ 0.82 | ≈ 0.92 / 1.15 | ≈ 1,612 | ≈ 0.057 | | ≈ 268 ms |
  | C | 1.225 after build, 1.366 after warm-up | ≈ 1.46 / 1.69 | ≈ 2,154 | 0.0233-0.0239 | 0.60-0.62 s | ≈ 103 ms |
  | D | ≈ 0.95 | ≈ 1.05 / 1.28 | ≈ 1,754 | ≈ 0.031 | | ≈ 119 ms |

  Also:
  - **RTF column.** §15's RTF is M0's: the 12 bench lines of `$EV/bench2/perf/perf.py` (`LINES`, with the KohAI
    `rule`), median of 5, in a fresh process. Measure it that way with a script in `$EV/gpu-lib/perf/`, not with the
    benchmark's `PARAGRAPH`.
  - **A at 16 steps (G4).** Run A once more with `--steps 16`.
  - **Neighbors (G4).** If the user starts llama-server and faster-whisper, record A's nvidia-smi next to them (the
    emulation measured 24,211 MiB of GPU in total).
  - **G8 and G9 counts.** 196 (B) and 112 (D) `Int8Linear` calls per `model.llm` forward, with 0 `nn.Linear`.
  - **G8.** The per-layer hashes equal slim.py run on CUDA (`/home/kimox/Projects/omnivoice-kohai/work/bench/slim.py`
    `int8_layers`), and so does the PCM hash.
- [ ] **Step 3: The paired eval on the library (G7, G8, G9).** Run bench2's 124-line paired eval with
  `$EV/bench2/quality/{render,score,analyze}.py`:
  - C against A (G7);
  - B and D against A and C, with a second seed (G8, G9).

  Expected: undetectable on the 13 metrics, as in the emulation.
- [ ] **Step 4: G3d on the library.**
  - **What.** Time to first item, and GPU time per reply, on the 20 measured KohAI replies, with sdpa and FlashInfer,
    with the TN grammar warm.
  - **How.** `$EV/rtf/bench_stream.py` is the template.
  - **Expected.** It matches §7's lib-like medians within the session noise: 240 and 103 ms to the first item.
- [ ] **Step 5: G14 soak.** 1,000 calls per config on the library.
  - Expected: no GPU memory growth, 100/100 determinism, 0 errors, and host RSS bounded (≤ about 320 MiB of cuDNN
    plans).
  - An hours-long run and a soak next to the LLM stay open unless the user asks for them.
- [ ] **Step 6: G13 and G-TN5 renders for the user**
  - **G13.** 2-3 KohAI texts over 30 s with the KohAI rule. Render each with fix 10, and with upstream's timing for
    chunks ≥ 1 (`speed=speed`).
  - **G-TN5.** A long Spanish text with numbers: normalized then `generate()`, and raw through
    `generate_streaming(normalize=tn.normalize)`.
    - **Pass.** No chunk of `generate()`'s chunker (as `_generate_chunked`'s `logger.debug` reports it) starts or ends
      inside a number reading, and every stream item's text is its raw sentence, digits kept.
    - Record both results in RESULTS.md.
  - Put the wavs in `$EV/gpu-lib/listen/`.
  - Offer to add them to the private listening page; republishing it needs the user's OK.
  - If the ear prefers upstream's timing, the fix-10 decision (decision #10, the design session's question D6) goes
    back to the user (§19 G13).
- [ ] **Step 7: Update the README table**
  - Replace §15's emulation figures with the library's.
  - Drop "to re-measure" once G4-G9 have run on the library.
  - Commit `README.md` with the message `docs: medidas de la librería en GPU`.

---

## After the plan: the user's steps (outward-facing, not done by an agent)

- Create `nicokim/omnivoice-tts-inference` on GitHub and rename the local `master` to `main` (§13).
- Set squash-only merges (PR_TITLE / COMMIT_MESSAGES) and delete-branch-on-merge.
- Add the `RELEASE_PLEASE_TOKEN` secret, the `pypi` environment and the PyPI trusted publisher.
- Open the PR with an English conventional title, e.g. `feat: omnivoice-tts-inference v0.1`.
- After the release, update the KohAI model card on the Hub.
- Adopt it in kohai-v3 as §12 lists:
  - drop the `transformers==4.52.1`, `indextts25-inference[flash]` and `flash-attn` pins;
  - declare `omnivoice-tts-inference[tn]`;
  - add §9's two FlashInfer wheels and set `FLASHINFER_CUDA_ARCH_LIST` at launch;
  - then run G-TN6.
