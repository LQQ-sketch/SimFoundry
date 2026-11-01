#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Replace the VLM's part grouping for one stage-9 object with a grouping a person chose.

When the articulation workflow's VLM steps get the structure wrong (stage-9 steps 2 and 4: how many
movable parts there are, and which P3-SAM segments belong to them), this writes the corrected tree and
segment assignment where the workflow reads its caches, backs up everything it replaces, and clears the
later steps so they run again. Steps 3 (segmentation) and 5 (the VLM writing the joint code, then the
simulation) are not touched, so what comes out is "the VLM's joint code on a hand-corrected grouping",
and must be described that way.

    python scripts/tools/set_articulation_grouping.py --object-dir \\
        Data/drawer2_test/s9_articulate_objects/drawer2_test/chest_of_drawers --movable 8 10 12 1

Without ``--apply`` it only prints the plan. Every segment id not listed in ``--movable`` goes to the fixed
part: the workflow drops faces that no part claims. All listed segments become ONE mesh and ONE link
(``export_mesh_parts`` writes one mesh per part name).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPLACED_STEPS = ("s4_merge_mesh_parts", "s5_articulate", "results")


def segment_ids(object_dir: Path) -> list[int]:
    parts = object_dir / "s3_segment_mesh/rendered_parts"
    ids_path = parts / "face_ids.npy"
    if not ids_path.is_file():
        raise SystemExit(f"missing {ids_path}; run step 3 first")
    face_ids = np.load(ids_path)
    ids = sorted(int(i) for i in np.unique(face_ids) if i >= 0)
    mask_path = parts / "label_to_faces_mask.npz"
    if mask_path.is_file():
        stored = np.load(mask_path)
        mask = stored[stored.files[0]]
        if mask.ndim == 2 and mask.shape[1] == len(face_ids):
            bad = [i for i in ids if i >= mask.shape[0] or set(mask[i].nonzero()[0]) != set(np.nonzero(face_ids == i)[0])]
            if bad:
                print(f"WARNING: face_ids.npy and label_to_faces_mask.npz disagree on segments {bad}; "
                      "the ids you read off a picture may not be the ones the workflow uses.")
    return ids


def fixed_part_name(object_dir: Path) -> str:
    recognised = object_dir / "s2_generate_articulation_tree/result_recognize_parts.json"
    if not recognised.is_file():
        raise SystemExit(f"missing {recognised}; run step 2 first (it names the fixed part)")
    return json.loads(recognised.read_text())["fixed_part_name"]


def build(fixed: str, movable_ids: list[int], all_ids: list[int], part: str, joint: str, description: str):
    base_link, moving_link = f"{fixed}_link", f"{part}_1_link"
    recognise = {"part_list": [{"part_name": part, "description": description}], "fixed_part_name": fixed}
    tree = {
        "parts": [{"part_name": part}, {"part_name": fixed}],
        "links": [{"link_name": base_link}, {"link_name": moving_link}],
        "joints": [{"joint_name": f"{part}_1_joint", "joint_type": joint,
                    "parent_link": base_link, "child_link": moving_link}],
        # The workflow adds this key when it saves its own tree, and merge.py reads it back from the
        # cache (articulation_tree_dict['fixed_part_name']); the VLM's raw reply does not carry it.
        "fixed_part_name": fixed,
    }
    others = [i for i in all_ids if i not in movable_ids]
    merge = {
        "analysis": {"note": "Set by hand with scripts/tools/set_articulation_grouping.py; not a VLM result."},
        "part_segment_dict": {part: sorted(movable_ids), fixed: others},
    }
    return recognise, tree, merge


def write_json_and_text(path: Path, payload: dict):
    text = json.dumps(payload, indent=4)
    path.write_text(text)
    path.with_suffix(".txt").write_text("```json\n" + text + "\n```")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--object-dir", required=True, type=Path)
    parser.add_argument("--movable", required=True, type=int, nargs="+", help="segment ids that move together as one part")
    parser.add_argument("--part-name", default="drawer")
    parser.add_argument("--joint-type", default="prismatic", choices=["prismatic", "revolute"])
    parser.add_argument("--description", default="a drawer that slides out of the front of the base; it has a handle.")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    object_dir = args.object_dir.resolve()
    ids = segment_ids(object_dir)
    unknown = sorted(set(args.movable) - set(ids))
    if unknown:
        raise SystemExit(f"segment ids {unknown} do not exist; the object has {ids}")
    fixed = fixed_part_name(object_dir)
    recognise, tree, merge = build(fixed, args.movable, ids, args.part_name, args.joint_type, args.description)

    print(f"object: {object_dir}")
    print(f"movable part '{args.part_name}' ({args.joint_type}), one link: segments {sorted(args.movable)}")
    print(f"fixed part '{fixed}': segments {merge['part_segment_dict'][fixed]}")
    if not args.apply:
        print("\n(dry run; add --apply to write this)")
        return 0

    backup = object_dir / f"_vlm_run_{time.strftime('%Y%m%d_%H%M%S')}"
    backup.mkdir()
    s2 = object_dir / "s2_generate_articulation_tree"
    shutil.copytree(s2, backup / s2.name)
    for step in REPLACED_STEPS:
        if (object_dir / step).exists():
            shutil.move(str(object_dir / step), str(backup / step))
    write_json_and_text(s2 / "result_recognize_parts.json", recognise)
    write_json_and_text(s2 / "result_generate_articulation_tree.json", tree)
    (object_dir / "s4_merge_mesh_parts").mkdir()
    write_json_and_text(object_dir / "s4_merge_mesh_parts/merge_result.json", merge)
    print(f"\nwritten. The VLM's version of everything replaced is in {backup}")
    print("Now re-run stage 9; steps 4 and 5 run again, step 2 and 3 are reused.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
