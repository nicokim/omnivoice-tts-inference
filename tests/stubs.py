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
