#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Build a one-object scene directory that stage 7 and stage 9 accept, from a single photo.

The photo path skips stages 1-6 and 8: there is no video, so no poses and no scale. Stage 9 only
reads three things per object, and this lays them out where it looks:

    s6_upsample/upsampled/iter_0.png              the photo (stage 9 passes it to the articulation workflow)
    s6_upsample/upsampled/iter_0_transparent.png  the cut-out (stage 7's Hunyuan3D input)
    s5_scene/obj_cat_list/iter_0.json             the phrase the VLM sees as the object's name
    s8_pose/canonical_mesh/iter_0.glb             step ``mesh``: stage 7's output, copied

Run from the repo root, in two steps around stage 7::

    python scripts/tools/make_articulation_scene.py setup --scene-name drawer_test \\
        --rgb /data/projects/simfoundry/drawer_in/iter_0.png \\
        --cutout /data/projects/simfoundry/drawer_in/iter_0_transparent.png \\
        --phrase "bedside table with three drawers"
    bash scripts/pipeline/A_reconstruction/run.sh --scene-name drawer_test --video-fpath <any file> --include 7
    python scripts/tools/make_articulation_scene.py mesh --scene-name drawer_test

Stage 8 normally bakes an upright, z-up orientation into ``canonical_mesh``; this copies stage 7's raw
mesh instead, so check the rendered views under ``s9_articulate_objects/.../s1_render`` before trusting
any joint axis.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "Data"


def setup(scene_dir: Path, rgb: Path, cutout: Path, phrase: str) -> None:
    for path in (rgb, cutout):
        if not path.is_file():
            raise SystemExit(f"missing input image: {path}")
    upsampled = scene_dir / "s6_upsample/upsampled"
    cat_dir = scene_dir / "s5_scene/obj_cat_list"
    upsampled.mkdir(parents=True, exist_ok=True)
    cat_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(rgb, upsampled / "iter_0.png")
    shutil.copy(cutout, upsampled / "iter_0_transparent.png")
    (cat_dir / "iter_0.json").write_text(
        json.dumps({"removed_obj_phrase": phrase, "is_valid_removed_obj": True}, indent=2)
    )
    print(f"scene laid out under {scene_dir}")
    for path in sorted(scene_dir.rglob("*")):
        if path.is_file():
            print("  ", path.relative_to(scene_dir))


def copy_mesh(scene_dir: Path) -> None:
    source = scene_dir / "s7_mesh/textured_mesh/hunyuan/iter_0_mesh.glb"
    if not source.is_file():
        raise SystemExit(f"stage 7 has not produced {source}; run it first")
    target = scene_dir / "s8_pose/canonical_mesh/iter_0.glb"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    print(f"copied {source.name} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["setup", "mesh"])
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rgb", type=Path, help="setup: the photo")
    parser.add_argument("--cutout", type=Path, help="setup: the photo with its background made transparent")
    parser.add_argument("--phrase", help="setup: what the object is called, e.g. 'bedside table with three drawers'")
    args = parser.parse_args(argv)

    scene_dir = args.root_dir / args.scene_name
    if args.step == "setup":
        if not (args.rgb and args.cutout and args.phrase):
            parser.error("setup needs --rgb, --cutout and --phrase")
        setup(scene_dir, args.rgb, args.cutout, args.phrase)
    else:
        copy_mesh(scene_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
