import subprocess
import sys

# Dependencies first: transformers' first modeling import pulls in torch._dynamo, whose sympy import adds a warnings
# filter (scipy adds its own when installed). Those are §11 residual effects, not the library's.
CODE = """
import warnings
import huggingface_hub, numpy, pydub, pydub.silence, safetensors, torch
from transformers import AutoTokenizer, HiggsAudioV2TokenizerModel, PreTrainedModel
before = list(warnings.filters)
import omnivoice_tts
assert list(warnings.filters) == before, warnings.filters
print(omnivoice_tts.__version__)
"""


def test_import_adds_no_filters_and_prints_nothing():
    r = subprocess.run([sys.executable, "-I", "-W", "ignore", "-c", CODE], capture_output=True, text=True)
    assert (r.stdout, r.stderr) == ("0.1.0\n", "")  # upstream's __init__ added 3 filters
