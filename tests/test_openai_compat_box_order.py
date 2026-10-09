# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL emits box_2d as [x1, y1, x2, y2]; the openai_compat backend converts it to Gemini's order."""
import json

from simfoundry.models.vlm import convert_box_2d_to_gemini_order

QWEN_REPLY = """```json
[
{"label": "bowl", "box_2d": [296, 275, 635, 721]},
{"label": "cup", "box_2d":[625,130,725,368]},
{"label": "pen", "box_2d": [778, 518, 877, 756]}
]
```"""


def test_xyxy_is_swapped_to_yxyx():
    out = convert_box_2d_to_gemini_order(QWEN_REPLY, box_order="xyxy")
    boxes = json.loads(out.strip("`json\n"))
    assert boxes[0]["box_2d"] == [275, 296, 721, 635]
    assert boxes[1]["box_2d"] == [130, 625, 368, 725]
    assert boxes[2]["box_2d"] == [518, 778, 756, 877]


def test_yxyx_is_untouched():
    assert convert_box_2d_to_gemini_order(QWEN_REPLY, box_order="yxyx") == QWEN_REPLY


def test_env_default_is_xyxy(monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_OPENAI_COMPAT_BOX_ORDER", raising=False)
    assert '"box_2d": [275, 296, 721, 635]' in convert_box_2d_to_gemini_order(QWEN_REPLY)
    monkeypatch.setenv("SIMFOUNDRY_OPENAI_COMPAT_BOX_ORDER", "yxyx")
    assert convert_box_2d_to_gemini_order(QWEN_REPLY) == QWEN_REPLY
