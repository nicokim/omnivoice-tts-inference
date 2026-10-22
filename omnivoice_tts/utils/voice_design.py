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

"""Voice-design instruct constants for TTS inference.

Defines the bilingual speaker attribute tags (gender, age, pitch, whisper)
and the English <-> Chinese maps that unify an instruct to one language.
Used by ``OmniVoice.generate()`` for voice design mode.
"""

import re

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")

# One {english: chinese} dict per category. Accents (English-only) and dialects
# (Chinese-only) have no counterpart, so they pass through unmapped.
_INSTRUCT_CATEGORIES = [
    {"male": "男", "female": "女"},
    {
        "child": "儿童",
        "teenager": "少年",
        "young adult": "青年",
        "middle-aged": "中年",
        "elderly": "老年",
    },
    {
        "very low pitch": "极低音调",
        "low pitch": "低音调",
        "moderate pitch": "中音调",
        "high pitch": "高音调",
        "very high pitch": "极高音调",
    },
    {"whisper": "耳语"},
]

_INSTRUCT_EN_TO_ZH = {}
_INSTRUCT_ZH_TO_EN = {}
for _cat in _INSTRUCT_CATEGORIES:
    _INSTRUCT_EN_TO_ZH.update(_cat)
    _INSTRUCT_ZH_TO_EN.update({v: k for k, v in _cat.items()})
