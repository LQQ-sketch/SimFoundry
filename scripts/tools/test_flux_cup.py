#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Re-generate the stage-6 object image of the cup with FLUX under different prompts, to see whether
the lid FLUX invented (stage 6 turned an open-top paper cup into a lidded take-away cup) can be
avoided by prompt wording alone. Uses the same FLUX call as stage 6 (guidance 2.5, 20 steps,
max_sequence_length 512) and the stage-6 input crop, so only the prompt and seed differ.

Run inside the `simfoundry` env (FLUX loads from the local weights, sequential offload ~2.4 GiB):
    python scripts/tools/test_flux_cup.py \
        --input /data/SimFoundry/Data/put_cup_test/s6_upsample/padded/iter_2.png \
        --out /data/projects/simfoundry/flux_cup --seeds 0 1 2

Writes <out>/input.png, <out>/<prompt>_seed<k>.png and <out>/compare.png (input | outputs).
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("SIMFOUNDRY_FLUX_KONTEXT_PATH", "/data/models/FLUX.1-Kontext-dev")
os.environ.setdefault("SIMFOUNDRY_FLUX_OFFLOAD", "sequential")

import cv2
import numpy as np
import torch
from PIL import Image

from simfoundry.models.vlm import FLUX1, PREFERRED_KONTEXT_RESOLUTIONS
from simfoundry.utils.prompt_utils import prompt_flux_object_completion_and_upsample_preserve

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="/data/SimFoundry/Data/put_cup_test/s6_upsample/padded/iter_2.png")
ap.add_argument("--out", default="/data/projects/simfoundry/flux_cup")
ap.add_argument("--phrase", default="white paper cup / disposable cup / printed paper cup")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

phrase = args.phrase
PROMPTS = {
    # exactly what stage 6 sends today
    "orig": prompt_flux_object_completion_and_upsample_preserve(phrase),
    # carry over the Gemini prompt's "keep lids / parts in the state shown" instruction
    "keepstate": (
        f"Show a single {phrase} in the foreground and a natural flat surface in the background. "
        f"Preserve the {phrase} visual appearance, but fix the occluded parts. If the object has lids or other "
        "movable parts, preserve their exact state from the input image: do not add, close or change them. "
        "Ultra-detailed, 8K, shot on a Canon EOS R5, F16, 89mm, sharp details."
    ),
    # state the facts about this particular cup
    "opentop": (
        f"Show a single {phrase} in the foreground and a natural flat surface in the background. "
        "It is an open-top paper cup with NO lid: keep the top open with the rim visible. "
        "Preserve the printed text on the cup and the green ring at its base. Fix only the occluded parts. "
        "Do not add a lid, a sleeve or a straw. Ultra-detailed, 8K, sharp details."
    ),
}

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# FLUX1.__call__ only accepts the resolutions Kontext prefers; pick the closest aspect ratio.
src = Image.open(args.input).convert("RGB")
w, h = src.size
if (w, h) not in set(PREFERRED_KONTEXT_RESOLUTIONS):
    tw, th = min(PREFERRED_KONTEXT_RESOLUTIONS, key=lambda r: abs(r[0] / r[1] - w / h))
    src = src.resize((tw, th), Image.LANCZOS)
    print(f"input {w}x{h} -> {tw}x{th}")
in_path = out / "input.png"
src.save(in_path)

model = FLUX1(model="FLUX.1-Kontext-dev", dtype=torch.bfloat16, device="cuda")

tiles = [("input", np.array(src))]
for name, prompt in PROMPTS.items():
    for seed in args.seeds:
        t0 = time.time()
        img = model(image_path=str(in_path), prompt=prompt, guidance_scale=2.5,
                    num_inference_steps=20, max_sequence_length=512, seed=seed)
        img.save(out / f"{name}_seed{seed}.png")
        tiles.append((f"{name} s{seed}", np.array(img.convert("RGB"))))
        print(f"{name} seed {seed}: {time.time() - t0:.0f}s", flush=True)

# contact sheet: every tile scaled to the same height, labelled
H = 480
row = []
for label, a in tiles:
    s = H / a.shape[0]
    a = cv2.resize(a, (int(a.shape[1] * s), H))
    cv2.putText(a, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    row.append(a)
Image.fromarray(np.concatenate(row, axis=1)).save(out / "compare.png")
print("wrote", out / "compare.png")
