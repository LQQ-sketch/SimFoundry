# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-consistency metrics, checked against geometry whose answer is known analytically."""

import numpy as np
import pytest

from simfoundry.pipeline.scene_eval import (
    PosedMesh,
    apply_pose,
    backproject,
    depth_relative_error,
    evaluate_scene,
    format_report,
    mask_scores,
    one_sided_chamfer,
    relative_transform,
    render_visible,
    sample_surface,
    splat_depth,
    transform_points,
)

SHAPE = (252, 448)
K = np.array([[300.0, 0.0, 224.0], [0.0, 300.0, 126.0], [0.0, 0.0, 1.0]])


def uv_sphere(radius=0.05, n=48):
    lat = np.linspace(0.0, np.pi, n)
    lon = np.linspace(0.0, 2 * np.pi, 2 * n, endpoint=False)
    verts = [[0, 0, -radius], [0, 0, radius]]
    for a in lat[1:-1]:
        for b in lon:
            verts.append([radius * np.sin(a) * np.cos(b), radius * np.sin(a) * np.sin(b), -radius * np.cos(a)])
    verts = np.array(verts)
    m = len(lon)
    ring = lambda i, j: 2 + i * m + (j % m)
    faces = []
    for j in range(m):
        faces.append([0, ring(0, j + 1), ring(0, j)])
        faces.append([1, ring(n - 3, j), ring(n - 3, j + 1)])
    for i in range(n - 3):
        for j in range(m):
            faces.append([ring(i, j), ring(i, j + 1), ring(i + 1, j)])
            faces.append([ring(i, j + 1), ring(i + 1, j + 1), ring(i + 1, j)])
    return verts, np.array(faces)


def extrinsic(yaw_deg, translation):
    a = np.radians(yaw_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return np.hstack([rot, np.asarray(translation, float)[:, None]])


def test_relative_transform_matches_a_world_point():
    e_c, e_f = extrinsic(5, [0.1, 0.0, 0.2]), extrinsic(-12, [-0.3, 0.05, 0.4])
    world = np.array([[0.3, -0.2, 1.5]])
    in_c = transform_points(world, np.vstack([e_c, [0, 0, 0, 1]]))
    in_f = transform_points(world, np.vstack([e_f, [0, 0, 0, 1]]))
    np.testing.assert_allclose(transform_points(in_c, relative_transform(e_c, e_f)), in_f, atol=1e-9)


def test_sample_surface_stays_on_a_cube_and_is_area_weighted():
    v = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], float)
    faces = []
    for axis in range(3):
        for sign in (-.5, .5):
            idx = [i for i, p in enumerate(v) if p[axis] == sign]
            a, b, c, d = sorted(idx, key=lambda i: tuple(v[i][[(axis + 1) % 3, (axis + 2) % 3]]))
            faces += [[a, b, c], [b, d, c]]
    pts = sample_surface(v, np.array(faces), 60_000)
    assert np.all(np.isclose(np.abs(pts).max(axis=1), 0.5))  # every point on some face
    counts = [np.sum(np.isclose(pts[:, axis], s)) for axis in range(3) for s in (-.5, .5)]
    assert max(counts) / min(counts) < 1.1  # equal faces get equal share


def test_rendered_sphere_matches_the_analytic_silhouette_and_depth():
    verts, faces = uv_sphere(0.05)
    pts = sample_surface(verts, faces, 150_000) + [0, 0, 1.0]
    mask, depth = render_visible({"s": pts}, K, SHAPE)["s"]
    radius_px = K[0, 0] * 0.05 / np.sqrt(1.0**2 - 0.05**2)
    assert mask.sum() == pytest.approx(np.pi * radius_px**2, rel=0.04)
    assert depth[int(K[1, 2]), int(K[0, 2])] == pytest.approx(1.0 - 0.05, abs=1e-3)


def test_front_object_occludes_the_one_behind_it():
    verts, faces = uv_sphere(0.05)
    base = sample_surface(verts, faces, 150_000)
    out = render_visible({"front": base + [0, 0, 1.0], "back": base + [0.02, 0, 1.2]}, K, SHAPE)
    both = out["front"][0] & out["back"][0]
    assert both.sum() == 0  # a pixel belongs to one object
    solo = render_visible({"back": base + [0.02, 0, 1.2]}, K, SHAPE)["back"][0]
    assert out["back"][0].sum() < 0.8 * solo.sum()


def test_mask_scores_on_known_rectangles():
    a, b = np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)
    a[10:20, 10:30] = True  # 200 px
    b[10:20, 20:40] = True  # 200 px, overlap 100
    s = mask_scores(a, b)
    assert s["iou"] == pytest.approx(100 / 300)
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(0.5)


def test_chamfer_reads_the_offset_between_two_planes():
    # The surface must be sampled much more finely than the offset being measured: the distance is
    # to the nearest sample, so a spacing of a few centimetres would be read as a centimetre of error.
    rng = np.random.default_rng(0)
    surface = np.column_stack([rng.uniform(-.1, .1, 40_000), rng.uniform(-.1, .1, 40_000), np.zeros(40_000)])
    observed = np.column_stack([rng.uniform(-.05, .05, 400), rng.uniform(-.05, .05, 400), np.full(400, 0.005)])
    assert one_sided_chamfer(observed, surface)["chamfer_median_m"] == pytest.approx(0.005, abs=0.0007)


def _scene(frames=4, radius=0.05):
    """A sphere seen from several cameras, with masks built independently of `relative_transform`."""
    verts, faces = uv_sphere(radius)
    ext = np.stack([extrinsic(8 * i - 10, [0.05 * i, 0.0, 0.03 * i]) for i in range(frames)])
    canonical = 1
    center_c = np.array([0.0, 0.0, 1.0])
    world = np.linalg.inv(np.vstack([ext[canonical], [0, 0, 0, 1]])) @ np.append(center_c, 1.0)
    base = sample_surface(verts, faces, 150_000)
    masks, depth = {"s": {}}, np.full((frames, *SHAPE), 2.0, np.float32)
    for f in range(frames):
        center_f = (np.vstack([ext[f], [0, 0, 0, 1]]) @ world)[:3]  # straight through the extrinsic
        m, d = render_visible({"s": base + center_f}, K, SHAPE)["s"]
        masks["s"][f] = m
        depth[f] = np.where(m, d, 2.0)
    mesh = PosedMesh(verts, faces, 1.0, np.eye(3), center_c)
    return mesh, depth, np.stack([K] * frames), ext, canonical, masks


def test_exact_pose_scores_near_one_on_every_frame():
    mesh, depth, ks, ext, c, masks = _scene()
    r = evaluate_scene({"s": mesh}, depth, ks, ext, c, masks)["objects"]["s"]
    assert r["same_frame"]["iou"] > 0.95
    assert r["same_frame"]["chamfer_median_m"] < 0.003
    assert r["held_out"]["n_scored"] == 3
    assert r["held_out"]["iou"]["min"] > 0.9
    assert r["held_out"]["depth_rel_median"]["median"] < 0.01


def test_a_three_centimetre_pose_error_is_caught_on_held_out_frames():
    mesh, depth, ks, ext, c, masks = _scene()
    off = PosedMesh(mesh.vertices, mesh.faces, 1.0, np.eye(3), mesh.trans + [0.03, 0.0, 0.0])
    r = evaluate_scene({"s": off}, depth, ks, ext, c, masks)["objects"]["s"]
    assert r["held_out"]["iou"]["median"] < 0.7


def test_missing_segmentation_and_invisible_meshes_are_not_scored_as_zero():
    mesh, depth, ks, ext, c, masks = _scene()
    masks["s"][0] = None
    behind = PosedMesh(mesh.vertices, mesh.faces, 1.0, np.eye(3), np.array([0.0, 0.0, -1.0]))
    held = evaluate_scene({"s": behind}, depth, ks, ext, c, masks)["objects"]["s"]["held_out"]
    assert held["n_scored"] == 0 and held["iou"]["n"] == 0  # nothing to average, not an average of zeros
    kept = evaluate_scene({"s": mesh}, depth, ks, ext, c, masks)["objects"]["s"]["held_out"]
    assert kept["n_no_sam_mask"] == 1 and kept["n_scored"] == 2


def test_a_pose_applied_the_wrong_way_fails_the_same_frame_check():
    mesh, depth, ks, ext, c, masks = _scene()
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    wrong = PosedMesh(mesh.vertices + [0.04, 0.0, 0.0], mesh.faces, 1.0, rot, mesh.trans)  # off-centre mesh, rotated
    r = evaluate_scene({"s": wrong}, depth, ks, ext, c, masks)["objects"]["s"]
    assert r["same_frame"]["iou"] < 0.7


def test_report_labels_the_same_frame_number_as_a_fit_residual():
    mesh, depth, ks, ext, c, masks = _scene()
    text = format_report(evaluate_scene({"s": mesh}, depth, ks, ext, c, masks), {"s": "ball"})
    assert "拟合残差" in text and "不是相对真实场景的精度" in text and "ball" in text


def test_backproject_inverts_the_pinhole_model():
    depth = np.full(SHAPE, 1.5, np.float32)
    mask = np.zeros(SHAPE, bool)
    mask[100, 300] = True
    np.testing.assert_allclose(backproject(depth, K, mask)[0], [(300 - 224) * 1.5 / 300, (100 - 126) * 1.5 / 300, 1.5])
