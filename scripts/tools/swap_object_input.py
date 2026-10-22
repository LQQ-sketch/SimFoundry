#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Replace one object's stage-6 result with a hand-picked source image, so stage 7 builds its mesh from
that image instead of from the FLUX completion.

Stage 7 reads <scene>/s6_upsample/upsampled/iter_<N>_transparent.png. FLUX can invent structure that
the video does not show (it put a lid on an open paper cup), and a mesh built from that image carries
the lid into collision and physics. When the object is fully visible in the video, the crop FLUX
received is already a better stage-7 input.

Existing iter_<N>.png and iter_<N>_transparent.png are copied to upsampled_backup/ first (never
overwritten there), then the background is removed with the same helper and rembg model stage 6 uses.

Run in the `simfoundry` env:
    python scripts/tools/swap_object_input.py --scene-dir Data/put_cup_test --index 2 \
        --source /data/projects/simfoundry/flux_cup/input.png

then rebuild only that mesh:
    bash scripts/pipeline/A_reconstruction/run.sh --scene-name put_cup_test --include 7 \
        -- s7_mesh.object_indices=[2] s7_mesh.low_vram=true
"""
import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from simfoundry.utils.processing_utils import remove_background

REMBG_CPU_PROVIDERS = ["CPUExecutionProvider"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True, help="Data/<scene_name>")
    ap.add_argument("--index", type=int, required=True, help="object index N in iter_<N>")
    ap.add_argument("--source", required=True, help="RGB image to use as the object's stage-7 input")
    ap.add_argument("--max-side", type=int, default=1024, help="downscale cap before rembg, as in stage 6")
    args = ap.parse_args()

    up = Path(args.scene_dir) / "s6_upsample" / "upsampled"
    if not up.is_dir():
        sys.exit(f"not found: {up}")
    src = Path(args.source)
    if not src.is_file():
        sys.exit(f"not found: {src}")

    name = f"iter_{args.index}"
    backup = up.parent / "upsampled_backup"
    backup.mkdir(exist_ok=True)
    for f in (up / f"{name}.png", up / f"{name}_transparent.png"):
        if f.exists():
            dst = backup / f.name
            if dst.exists():
                print(f"backup already holds {dst.name}; keeping it")
            else:
                shutil.copy2(f, dst)
                print(f"backed up {f.name} -> {dst}")

    # Keep iter_<N>.png consistent with the transparent one, for anything that reads the plain image.
    Image.open(src).convert("RGB").save(up / f"{name}.png")

    from rembg import new_session
    session = new_session("bria-rmbg", providers=REMBG_CPU_PROVIDERS)
    out = up / f"{name}_transparent.png"
    remove_background(src, out, session=session, max_side=args.max_side)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
