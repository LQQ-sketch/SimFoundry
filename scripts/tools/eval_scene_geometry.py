#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Score a reconstructed scene's posed meshes against the video it came from.

Reads stage 8's canonical meshes and poses, stage 2's depth / intrinsics / extrinsics, and asks SAM 3
to segment each object on every one of the 15 frames. The metrics, and why the held-out frames are
the ones to trust, are explained in ``simfoundry/pipeline/scene_eval.py``. Run from the repo root in
the ``simfoundry`` env, on a machine with a GPU for SAM 3:

    export SIMFOUNDRY_SAM3_CHECKPOINT=/data/models/sam3/sam3.pt HF_HUB_DISABLE_XET=1 HF_HOME=/data/cache/huggingface
    python scripts/tools/eval_scene_geometry.py --scene-dir /data/SimFoundry/Data/put_cup_test

Writes ``<scene-dir>/s16_scene_eval/``: ``report.json``, ``report.md``, ``sam_masks.npz`` (so a rerun
with ``--reuse-sam`` needs no GPU) and an overlay sheet per object (SAM mask in green, the mesh's
rendered silhouette outlined in red, the canonical frame boxed in yellow).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simfoundry.pipeline import scene_eval as se  # noqa: E402

# Below this, the canonical-frame IoU says the pose was read the wrong way, not that it fits badly.
CONVENTION_CHECK_MIN_IOU = 0.3


def load_mesh(path):
    """Vertices and faces of a GLB, however many nodes it holds (stage 8 treats Scenes the same way)."""
    import trimesh

    loaded = trimesh.load(str(path))
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def canonical_frame(scene_dir: Path) -> int:
    found = {}
    for rel, key in (("s3_ground/frame_selection.json", "selected_idx"), ("s5_scene/decomposition_frame.json", "img_idx")):
        path = scene_dir / rel
        if path.is_file():
            found[rel] = int(json.loads(path.read_text())[key])
    if not found:
        raise SystemExit(f"no frame_selection.json / decomposition_frame.json under {scene_dir}")
    if len(set(found.values())) > 1:
        print(f"WARNING: the two files disagree on the canonical frame: {found}; using stage 3's.")
    return found.get("s3_ground/frame_selection.json", next(iter(found.values())))


def load_objects(scene_dir: Path, only: list[str] | None):
    objects, phrases = {}, {}
    for info_path in sorted((scene_dir / "s8_pose/info").glob("iter_*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
        name = info_path.stem
        if only and name not in only:
            continue
        pose = json.loads(info_path.read_text())["z_up"]
        cat = scene_dir / "s5_scene/obj_cat_list" / f"{name}.json"
        phrase = json.loads(cat.read_text())["removed_obj_phrase"].strip() if cat.is_file() else ""
        mesh = scene_dir / "s8_pose/canonical_mesh" / f"{name}.glb"
        if not phrase or not mesh.is_file():
            print(f"skipping {name}: no phrase ({cat}) or no mesh ({mesh})")
            continue
        vertices, faces = load_mesh(mesh)
        objects[name] = se.PosedMesh(vertices, faces, np.asarray(pose["scale"]), np.asarray(pose["rot"]), np.asarray(pose["trans"]))
        phrases[name] = phrase
    if not objects:
        raise SystemExit(f"no objects found under {scene_dir / 's8_pose/info'}")
    return objects, phrases


def run_sam3(images, phrases, confidence):
    """name -> frame -> boolean mask (``None`` when SAM 3 finds nothing), plus the winning score."""
    from simfoundry.models.sam_v3 import SAM3

    sam = SAM3(confidence_threshold=confidence, device="cuda", video=False)
    masks = {name: {} for name in phrases}
    scores = {name: {} for name in phrases}
    h, w = images.shape[1:3]
    for name, phrase in phrases.items():
        for f, image in enumerate(images):
            found, _boxes, conf = sam.predict_segmentation(pil_img=Image.fromarray(image), text_prompt=phrase)
            if len(found) == 0:
                masks[name][f], scores[name][f] = None, None
                continue
            best = int(np.argmax(conf))
            mask = np.asarray(found[best])[0].astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks[name][f], scores[name][f] = mask.astype(bool), float(conf[best])
    return masks, scores


def save_masks(path: Path, masks):
    np.savez_compressed(path, **{f"{n}__{f}": m for n, per in masks.items() for f, m in per.items() if m is not None})


def load_masks(path: Path, names, n_frames):
    stored = np.load(path)
    return {n: {f: (stored[f"{n}__{f}"] if f"{n}__{f}" in stored.files else None) for f in range(n_frames)} for n in names}


def overlay(image, sam_mask, rendered, canonical):
    """The frame, SAM's mask tinted green, the mesh silhouette outlined red."""
    out = image.copy()
    if sam_mask is not None:
        out[sam_mask] = (0.55 * out[sam_mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(rendered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 0, 0), 1)
    if canonical:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (255, 255, 0), 3)
    return out


def write_sheets(out_dir: Path, images, masks, renders, canonical_idx, report):
    sheets = out_dir / "overlays"
    sheets.mkdir(exist_ok=True)
    h, w = images.shape[1:3]
    for name, per_frame in renders.items():
        tiles = []
        for f in range(len(images)):
            tile = overlay(images[f], masks[name].get(f), per_frame[f], f == canonical_idx)
            iou = next((r.get("iou") for r in _rows(report, name) if r["frame"] == f and r["status"] == "scored"), None)
            label = f"f{f}" + (f" IoU {iou:.2f}" if iou is not None else " -")
            cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        cols = 3
        while len(tiles) % cols:
            tiles.append(np.zeros((h, w, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
        Image.fromarray(grid).save(sheets / f"{name}_sheet.png")
        Image.fromarray(cv2.resize(overlay(images[canonical_idx], masks[name].get(canonical_idx), per_frame[canonical_idx], True),
                                   (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)).save(sheets / f"{name}_canonical.png")


def _rows(report, name):
    obj = report["objects"][name]
    return [obj["same_frame"]] + obj["held_out"]["per_frame"]


def git_head(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene-dir", required=True, type=Path, help="Data/<scene_name>")
    parser.add_argument("--out-dir", type=Path, help="default: <scene-dir>/s16_scene_eval")
    parser.add_argument("--objects", nargs="*", help="only these (e.g. iter_2)")
    parser.add_argument("--sam-conf", type=float, default=0.5)
    parser.add_argument("--reuse-sam", action="store_true", help="read sam_masks.npz instead of running SAM 3")
    parser.add_argument("--n-surface", type=int, default=se.N_SURFACE_SAMPLES)
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args(argv)

    scene_dir = args.scene_dir.resolve()
    out_dir = (args.out_dir or scene_dir / "s16_scene_eval").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = np.load(scene_dir / "s2_da/da/exports/npz/results.npz")
    images, depth, k, ext = results["image"], results["depth"], results["intrinsics"], results["extrinsics"]
    n_frames, h, w = depth.shape
    canonical_idx = canonical_frame(scene_dir)
    objects, phrases = load_objects(scene_dir, args.objects)
    print(f"{len(objects)} objects {phrases}; {n_frames} frames at {w}x{h}; canonical frame {canonical_idx}")

    sam_path = out_dir / "sam_masks.npz"
    if args.reuse_sam:
        masks, scores = load_masks(sam_path, objects, n_frames), None
    else:
        masks, scores = run_sam3(images, phrases, args.sam_conf)
        save_masks(sam_path, masks)

    report = se.evaluate_scene(objects, depth, k, ext, canonical_idx, masks, n_surface=args.n_surface, keep_renders=True)
    renders = report.pop("_renders")
    report["meta"] = {
        "scene_dir": str(scene_dir),
        "image_size_wh": [int(w), int(h)],
        "phrases": phrases,
        "sam_confidence_threshold": args.sam_conf,
        "sam_scores": scores,
        "n_surface_samples": args.n_surface,
        "supersample": se.SUPERSAMPLE,
        "git_head": git_head(Path(__file__).resolve().parents[2]),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    header = f"场景 `{scene_dir.name}`,{n_frames} 帧,分辨率 {w}×{h},提交 {report['meta']['git_head']}。\n\n"
    text = se.format_report(report, phrases)
    (out_dir / "report.md").write_text(text.replace("\n\n", "\n\n" + header, 1))
    if not args.no_overlays:
        write_sheets(out_dir, images, masks, renders, canonical_idx, report)
    print(text)

    same = {n: o["same_frame"] for n, o in report["objects"].items()}
    unsegmented = [n for n, s in same.items() if s["status"] == "no_sam_mask"]
    if unsegmented:
        print(f"\nSAM 3 found nothing on the canonical frame for {unsegmented}; their rows say nothing about the pose.")
    # A mesh that is invisible, or barely overlaps the mask, on the very frame it was fitted to means the
    # pose is being applied the wrong way (or SAM 3 picked another object), not that the fit is imperfect.
    broken = [n for n, s in same.items() if s["status"] == "not_visible" or s.get("iou", 1.0) < CONVENTION_CHECK_MIN_IOU]
    if broken:
        print(
            f"\nCONVENTION CHECK FAILED for {broken}: the mesh does not land on the frame it was fitted to. "
            f"Look at {out_dir / 'overlays'} before trusting any number above."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
