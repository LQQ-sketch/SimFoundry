# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scene-evaluation CLI end to end on a synthetic scene directory (no GPU, no SAM 3, no GLB)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_scene_eval import _scene

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/eval_scene_geometry.py"


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("eval_scene_geometry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_scene(root: Path, pose_rot, pose_trans):
    mesh, depth, ks, ext, canonical, masks = _scene()
    frames = depth.shape[0]
    (root / "s2_da/da/exports/npz").mkdir(parents=True)
    np.savez(
        root / "s2_da/da/exports/npz/results.npz",
        image=np.random.default_rng(0).integers(0, 255, (frames, *depth.shape[1:], 3), dtype=np.uint8),
        depth=depth, intrinsics=ks, extrinsics=ext,
    )
    for rel, body in (
        ("s3_ground/frame_selection.json", {"selected_idx": canonical}),
        ("s8_pose/info/iter_0.json", {"z_up": {"trans": list(pose_trans), "rot": np.asarray(pose_rot).tolist(), "scale": [1.0, 1.0, 1.0]}}),
        ("s5_scene/obj_cat_list/iter_0.json", {"removed_obj_phrase": "ball"}),
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(body))
    (root / "s8_pose/canonical_mesh").mkdir(parents=True)
    (root / "s8_pose/canonical_mesh/iter_0.glb").write_bytes(b"")  # load_mesh is patched out
    return mesh, masks


def test_cli_writes_the_report_and_overlays_for_a_correct_pose(cli, tmp_path, monkeypatch):
    mesh, masks = _write_scene(tmp_path, np.eye(3), [0.0, 0.0, 1.0])
    monkeypatch.setattr(cli, "load_mesh", lambda path: (mesh.vertices, mesh.faces))
    out = tmp_path / "s16_scene_eval"
    out.mkdir()
    cli.save_masks(out / "sam_masks.npz", {"iter_0": masks["s"]})

    assert cli.main(["--scene-dir", str(tmp_path), "--reuse-sam"]) == 0

    report = json.loads((out / "report.json").read_text())
    obj = report["objects"]["iter_0"]
    assert report["canonical_frame"] == 1 and report["meta"]["phrases"] == {"iter_0": "ball"}
    assert obj["same_frame"]["iou"] > 0.95 and obj["held_out"]["iou"]["min"] > 0.9
    assert "ball" in (out / "report.md").read_text()
    assert (out / "overlays/iter_0_sheet.png").is_file() and (out / "overlays/iter_0_canonical.png").is_file()


def test_cli_fails_loudly_when_the_pose_is_applied_the_wrong_way(cli, tmp_path, monkeypatch, capsys):
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    mesh, masks = _write_scene(tmp_path, rot, [0.0, 0.0, 1.0])
    monkeypatch.setattr(cli, "load_mesh", lambda path: (mesh.vertices + [0.2, 0.0, 0.0], mesh.faces))  # 20 cm off-centre: rotating it swings it clear of the mask
    out = tmp_path / "s16_scene_eval"
    out.mkdir()
    cli.save_masks(out / "sam_masks.npz", {"iter_0": masks["s"]})

    assert cli.main(["--scene-dir", str(tmp_path), "--reuse-sam", "--no-overlays"]) == 1
    assert "CONVENTION CHECK FAILED" in capsys.readouterr().out


def test_a_frame_with_no_segmentation_is_reported_not_treated_as_a_failure(cli, tmp_path, monkeypatch, capsys):
    mesh, masks = _write_scene(tmp_path, np.eye(3), [0.0, 0.0, 1.0])
    masks["s"][1] = None  # the canonical frame itself
    monkeypatch.setattr(cli, "load_mesh", lambda path: (mesh.vertices, mesh.faces))
    out = tmp_path / "s16_scene_eval"
    out.mkdir()
    cli.save_masks(out / "sam_masks.npz", {"iter_0": masks["s"]})

    assert cli.main(["--scene-dir", str(tmp_path), "--reuse-sam", "--no-overlays"]) == 0
    assert "found nothing on the canonical frame" in capsys.readouterr().out
