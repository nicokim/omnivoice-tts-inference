import subprocess
import sys

import omnivoice_tts

# Dependencies first: transformers' first modeling import pulls in torch._dynamo, whose sympy import adds a warnings
# filter (scipy adds its own when installed). Those are §11 residual effects, not the library's.
CODE = """
import sys, warnings
import huggingface_hub, numpy, pydub, pydub.silence, safetensors, torch
from transformers import AutoTokenizer, HiggsAudioV2TokenizerModel, PreTrainedModel
before = list(warnings.filters)
import omnivoice_tts
assert list(warnings.filters) == before, warnings.filters
assert "omnivoice_tts.models.omnivoice_int8" not in sys.modules  # both modes are opt-in imports
assert "omnivoice_tts.models.omnivoice_flashinfer" not in sys.modules
print(omnivoice_tts.__version__)
"""


def test_import_adds_no_filters_and_prints_nothing():
    r = subprocess.run([sys.executable, "-I", "-W", "ignore", "-c", CODE], capture_output=True, text=True)
    assert (r.stdout, r.stderr) == (f"{omnivoice_tts.__version__}\n", "")  # upstream's __init__ added 3 filters
