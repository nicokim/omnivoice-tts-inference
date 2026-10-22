"""Measures what the README's FlashInfer and int8 table reports, on the library and one CUDA GPU.

    python scripts/benchmark.py --model k2-fsa/OmniVoice --backend flashinfer --int8

sdpa fp16 loads straight onto the GPU; FlashInfer and int8 use the CPU-first recipe (load on the CPU, patch, then
.to("cuda:0")). Every config then makes one warm-up call, timed as the cold first generate(). "load" is
from_pretrained plus the patches and the move, after the imports and the CUDA context. Repeated measures print the
median of --reps. It never runs torch.profiler: one profiler session slows every later call in the process.
"""

import argparse
import os
import statistics
import subprocess
import time

SHORT = "Hola, ¿cómo va? Gracias por pasar por el stream."  # a typical two-sentence reply, 2.8 s by the rule
TEXT_6S = "La inteligencia artificial cambió la forma en que trabajamos, y nadie sabe qué viene después."  # 5.7 s
TEXT_20S = (  # 20.1 s
    "Hace apenas unos años, nadie imaginaba que una computadora pudiera mantener una conversación fluida, escribir "
    "código o componer música. Hoy, el 14 de marzo de 2026 a las 18:45, más de 3.500 personas siguieron la "
    "transmisión en vivo, y el récord anterior era de 2.100 espectadores."
)
PARAGRAPH = (  # 28.9 s: one pass, under the 30 s chunk threshold but past sdpa's L = 320 knee (about 10.6 s)
    "La inteligencia artificial cambió la forma en que trabajamos. Hace apenas unos años, "
    "nadie imaginaba que una computadora pudiera mantener una conversación fluida, escribir "
    "código o componer música. Hoy, el 14 de marzo de 2026 a las 18:45, más de 3.500 personas "
    "siguieron la transmisión en vivo, y el récord anterior era de 2.100 espectadores. "
    "¿Qué viene después? Nadie lo sabe con certeza, pero vale la pena prestar atención."
)


def smi_mib():
    """This process's GPU memory as nvidia-smi reports it, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = (line.split(",") for line in out.splitlines() if line.strip())
    return next((int(mib) for pid, mib in rows if int(pid) == os.getpid()), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="k2-fsa/OmniVoice", help="Hub repo id or local directory")
    ap.add_argument("--backend", choices=["sdpa", "flashinfer"], default="sdpa")
    ap.add_argument("--int8", action="store_true", help="apply_int8 (after apply_flashinfer with --backend flashinfer)")
    ap.add_argument("--steps", type=int, default=32, help="num_step of every call")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    import torch  # after argparse: --help needs no CUDA

    from omnivoice_tts import OmniVoice, __version__

    gen = {"num_step": a.steps}
    torch.cuda.synchronize()  # creates the CUDA context outside the timed load
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if a.backend == "sdpa" and not a.int8:
        model = OmniVoice.from_pretrained(a.model, device_map="cuda:0", dtype=torch.float16)
    else:
        model = OmniVoice.from_pretrained(a.model, dtype=torch.float16)
        if a.backend == "flashinfer":
            from omnivoice_tts.models.omnivoice_flashinfer import apply_flashinfer

            apply_flashinfer(model)
        if a.int8:
            from omnivoice_tts.models.omnivoice_int8 import apply_int8

            apply_int8(model)
        model.to("cuda:0")
    torch.cuda.synchronize()
    load = time.perf_counter() - t0
    load_peak, loaded = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
    loaded_reserved = torch.cuda.memory_reserved()

    t0 = time.perf_counter()
    model.generate(SHORT, "es", **gen)  # the warm-up call
    cold = time.perf_counter() - t0
    resident, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

    def peak(text):
        torch.cuda.reset_peak_memory_stats()
        model.generate(text, "es", **gen)
        return torch.cuda.max_memory_allocated()

    def first_item(text):
        t0 = time.perf_counter()
        stream = model.generate_streaming(text, "es", **gen)
        next(stream)  # a numpy array, so its GPU work is done
        elapsed = time.perf_counter() - t0
        stream.close()
        return elapsed

    def rtf(text):
        t0 = time.perf_counter()
        audio = model.generate(text, "es", **gen)[0]
        return (time.perf_counter() - t0) / (len(audio) / model.sampling_rate)

    def median(measure, text):
        values = []
        for r in range(a.reps):
            torch.manual_seed(r)
            values.append(measure(text))
        return statistics.median(values)

    peak_6s = median(peak, TEXT_6S)
    smi = smi_mib()  # right after the <= 6 s text, as the README's nvidia-smi column
    first_short, first_paragraph = median(first_item, SHORT), median(first_item, PARAGRAPH)
    peak_20s = median(peak, TEXT_20S)
    paragraph_rtf = median(rtf, PARAGRAPH)

    gib = 2**30
    print(f"{torch.cuda.get_device_name()} | backend={a.backend} int8={a.int8} steps={a.steps} reps={a.reps} | "
          f"omnivoice_tts {__version__}, torch {torch.__version__}")
    print(f"load {load:.2f} s | load peak {load_peak / gib:.3f} GiB | allocated {loaded / gib:.3f} GiB | "
          f"reserved {loaded_reserved / gib:.3f} GiB")
    print(f"cold first generate() {cold:.3f} s | resident {resident / gib:.3f} GiB | reserved {reserved / gib:.3f} GiB")
    print(f"peak {peak_6s / gib:.3f} GiB at 5.7 s, {peak_20s / gib:.3f} GiB at 20 s | "
          f"nvidia-smi {'n/a' if smi is None else smi} MiB after the 5.7 s text")
    print(f"first stream item: SHORT {first_short * 1e3:.0f} ms | PARAGRAPH {first_paragraph * 1e3:.0f} ms")
    print(f"generate() RTF on PARAGRAPH (28.9 s by the rule) {paragraph_rtf:.4f}")


if __name__ == "__main__":
    main()
