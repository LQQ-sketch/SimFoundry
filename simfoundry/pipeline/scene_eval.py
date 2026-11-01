# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consistency of a reconstructed scene with the video it came from.

There is no ground-truth geometry for a captured video, so none of these numbers is "accuracy"
against the real scene. What they measure is how well the posed meshes agree with observations,
and the two kinds of observation are not equally trustworthy:

* **Same frame.** Stage 8 fits each mesh to the canonical frame's depth and mask. Scoring the fit
  on that same frame mostly reads back the fit's own residual, so it is reported separately and
  labelled as such. It is still useful as a convention check: a pose applied the wrong way scores
  near zero here.
* **Held-out frames.** The other frames of the video were never used by the fit. Moving a mesh
  into one of them needs only the depth model's camera extrinsics, and its silhouette and depth
  can be compared with a fresh segmentation and the depth model's own estimate for that frame.
  This is the closest thing to an independent signal available, with the limit that the extrinsics
  are estimated too.

Everything here is numpy (plus OpenCV for a morphological close), so it runs without a GPU.
Loading meshes and running SAM 3 live in ``scripts/tools/eval_scene_geometry.py``.

Conventions match stage 8: a pose maps canonical-mesh vertices into the canonical frame's camera as
``p_cam = scale * p @ rot.T + trans``; cameras are OpenCV (x right, y down, z forward); extrinsics
are world-to-camera, 3x4, in one shared reference (stage 2 runs a single chunk for <= 220 frames).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

# A mesh seen at this size or smaller is not scored: its IoU would be quantisation noise.
MIN_VISIBLE_PIXELS = 30
# Surface samples per mesh. At the depth model's 448 px width a pixel is a few millimetres, so
# this leaves no gaps larger than the 3x3 close below fills.
N_SURFACE_SAMPLES = 150_000
# Silhouettes are drawn at this many times the image size and averaged down; see render_visible.
SUPERSAMPLE = 4
# Observed points are matched to a subsample of the surface samples. The sample spacing adds to
# every nearest-neighbour distance (about 1 mm on a cup, a few mm on a large bowl).
CHAMFER_SURFACE_SAMPLES = 50_000


@dataclass
class PosedMesh:
    """A canonical-frame mesh and the pose stage 8 stored for it (``z_up`` entry)."""

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3)
    scale: np.ndarray | float
    rot: np.ndarray  # (3, 3)
    trans: np.ndarray  # (3,)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Area-weighted random points on a triangle mesh."""
    tri = vertices[faces]  # (F, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not area.sum() > 0:
        raise ValueError("mesh has no surface area")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1.0  # fold the parallelogram back onto the triangle
    r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
    a, b, c = tri[pick, 0], tri[pick, 1], tri[pick, 2]
    return a + r1[:, None] * (b - a) + r2[:, None] * (c - a)


def apply_pose(points: np.ndarray, scale, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Stage 8's ``scale * p @ rot.T + t`` (``scale`` may be a scalar or per-axis)."""
    return np.asarray(scale) * points @ np.asarray(rot).T + np.asarray(trans)


def _homogeneous(extrinsic_3x4: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :4] = extrinsic_3x4
    return out


def relative_transform(ext_from: np.ndarray, ext_to: np.ndarray) -> np.ndarray:
    """4x4 taking points in camera ``from`` to camera ``to``: ``E_to @ inv(E_from)``."""
    return _homogeneous(ext_to) @ np.linalg.inv(_homogeneous(ext_from))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def backproject(depth: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Camera-frame points of the masked pixels (pinhole, pixel index = coordinate)."""
    v, u = np.nonzero(mask)
    z = depth[v, u]
    good = np.isfinite(z) & (z > 0)
    u, v, z = u[good], v[good], z[good]
    return np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=1)


def splat_depth(points_cam: np.ndarray, k: np.ndarray, shape: tuple[int, int], close_px: int = 3) -> np.ndarray:
    """Z-buffer of dense surface samples; ``inf`` where nothing projects.

    Splatting points is much cheaper than rasterising triangles that are smaller than a pixel, and a
    small morphological close fills the pinholes between samples.
    """
    h, w = shape
    z = points_cam[:, 2]
    p = points_cam[z > 1e-6]
    z = p[:, 2]
    u = np.round(k[0, 0] * p[:, 0] / z + k[0, 2]).astype(np.int64)
    v = np.round(k[1, 1] * p[:, 1] / z + k[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(depth, v[inside] * w + u[inside], z[inside].astype(np.float32))
    depth = depth.reshape(h, w)
    if close_px > 0:
        kernel = np.ones((close_px, close_px), np.uint8)
        seen = np.isfinite(depth)
        closed = cv2.morphologyEx(seen.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        nearest = cv2.erode(np.where(seen, depth, 1e9).astype(np.float32), kernel)
        nearest[nearest >= 1e9] = np.inf
        depth = np.where(seen, depth, np.where(closed, nearest, np.inf)).astype(np.float32)
    return depth


def render_visible(
    objects_cam: dict[str, np.ndarray], k: np.ndarray, shape: tuple[int, int], supersample: int = SUPERSAMPLE
):
    """Per-object visible mask and depth, with the objects occluding each other.

    Returns ``{name: (mask (H, W) bool, depth (H, W) float32 with nan outside the mask)}``.

    Splatting marks a pixel whenever any sample falls inside its cell, which grows every silhouette
    by about half a pixel -- a 9% area error on a 15 px disc and far worse on a pen a few pixels wide,
    and it would lower every IoU for a reason that has nothing to do with the reconstruction. So the
    scene is drawn ``supersample`` times larger and a pixel counts as covered when at least half of
    its sub-pixels are, which is the same rule a segmentation mask follows.
    """
    s = supersample
    h, w = shape
    # Pixel centres sit on integer coordinates, so pixel i spans [i - .5, i + .5).
    k_hi = np.array(k, dtype=np.float64)
    k_hi[:2, :] *= s
    k_hi[0, 2] = s * (k[0, 2] + 0.5) - 0.5
    k_hi[1, 2] = s * (k[1, 2] + 0.5) - 0.5
    depths = {name: splat_depth(pts, k_hi, (h * s, w * s)) for name, pts in objects_cam.items()}
    nearest = np.min(np.stack(list(depths.values())), axis=0)
    cover, mean_depth = {}, {}
    for name, d in depths.items():
        # Tolerance for samples of two surfaces landing in the same pixel at almost equal depth.
        visible = np.isfinite(d) & (d <= nearest + 2e-3 + 1e-3 * np.where(np.isfinite(nearest), nearest, 0.0))
        cover[name] = visible.reshape(h, s, w, s).mean(axis=(1, 3))
        with warnings.catch_warnings():  # an empty block is expected wherever the object is absent
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_depth[name] = np.nanmean(np.where(visible, d, np.nan).reshape(h, s, w, s), axis=(1, 3))

    # Two objects can each cover exactly half of a boundary pixel. A pixel belongs to one object:
    # the one covering more of it, and on a tie the nearer one.
    names = list(depths)
    score = np.stack([
        cover[n] - 1e-6 * np.where(np.isfinite(mean_depth[n]), mean_depth[n], 1e3) for n in names
    ])
    winner = score.argmax(axis=0)
    out = {}
    for i, name in enumerate(names):
        mask = (cover[name] >= 0.5) & (winner == i)
        out[name] = (mask, np.where(mask, mean_depth[name], np.nan).astype(np.float32))
    return out


def mask_scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    inter = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    pred_px, ref_px = int(pred.sum()), int(ref.sum())
    return {
        "iou": inter / union if union else 0.0,
        "precision": inter / pred_px if pred_px else 0.0,
        "recall": inter / ref_px if ref_px else 0.0,
        "rendered_px": pred_px,
        "sam_px": ref_px,
    }


def depth_relative_error(rendered: np.ndarray, observed: np.ndarray, overlap: np.ndarray) -> dict:
    """|rendered - observed| / observed over the pixels both the mesh and the segmentation cover."""
    sel = overlap & np.isfinite(rendered) & np.isfinite(observed) & (observed > 0)
    if not sel.any():
        return {"depth_rel_median": None, "depth_rel_p90": None}
    err = np.abs(rendered[sel] - observed[sel]) / observed[sel]
    return {"depth_rel_median": float(np.median(err)), "depth_rel_p90": float(np.percentile(err, 90))}


def one_sided_chamfer(observed: np.ndarray, surface: np.ndarray, max_observed: int = 3000, seed: int = 0) -> dict:
    """Distance from each observed point to the nearest mesh surface sample (metres).

    One-sided because a single view sees part of the object while the mesh is complete; the other
    direction would charge the mesh for the far side nobody observed. The distance is to the nearest
    surface *sample*, so it carries the sample spacing as a floor.
    """
    if len(observed) == 0:
        return {"chamfer_median_m": None, "chamfer_p90_m": None, "chamfer_n": 0}
    if len(observed) > max_observed:
        observed = observed[np.random.default_rng(seed).choice(len(observed), max_observed, replace=False)]
    surface = surface[:: max(1, len(surface) // CHAMFER_SURFACE_SAMPLES)]
    nearest = np.empty(len(observed))
    s2 = (surface**2).sum(1)
    for i in range(0, len(observed), 256):
        chunk = observed[i : i + 256]
        d2 = (chunk**2).sum(1)[:, None] - 2.0 * chunk @ surface.T + s2[None, :]
        nearest[i : i + 256] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return {
        "chamfer_median_m": float(np.median(nearest)),
        "chamfer_p90_m": float(np.percentile(nearest, 90)),
        "chamfer_n": int(len(observed)),
    }


def _aggregate(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"median": None, "mean": None, "min": None, "max": None, "n": 0}
    return {
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": len(vals),
    }


def evaluate_scene(
    objects: dict[str, PosedMesh],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    canonical_idx: int,
    masks: dict[str, dict[int, np.ndarray | None]],
    n_surface: int = N_SURFACE_SAMPLES,
    min_pixels: int = MIN_VISIBLE_PIXELS,
    seed: int = 0,
    keep_renders: bool = False,
) -> dict:
    """Score every posed mesh on the canonical frame and on every other frame.

    Args:
        objects: name -> posed mesh (stage 8's canonical mesh and ``z_up`` pose).
        depth: (F, H, W) depth model output, metres.
        intrinsics: (F, 3, 3). extrinsics: (F, 3, 4) world-to-camera.
        canonical_idx: the frame stage 8 fitted against.
        masks: name -> frame index -> segmentation mask of that object (``None`` if none was found).
        keep_renders: also return the rendered masks, for overlay figures.
    """
    n_frames, h, w = depth.shape
    surface_c = {
        name: apply_pose(sample_surface(m.vertices, m.faces, n_surface, seed), m.scale, m.rot, m.trans)
        for name, m in objects.items()
    }

    rows: dict[str, list[dict]] = {name: [] for name in objects}
    renders: dict[str, dict[int, np.ndarray]] = {name: {} for name in objects}
    for f in range(n_frames):
        t_cf = relative_transform(extrinsics[canonical_idx], extrinsics[f])
        shown = render_visible({n: transform_points(p, t_cf) for n, p in surface_c.items()}, intrinsics[f], (h, w))
        for name, (rmask, rdepth) in shown.items():
            row = {"frame": f, "canonical": f == canonical_idx}
            ref = masks.get(name, {}).get(f)
            if keep_renders:
                renders[name][f] = rmask
            if ref is None:
                row["status"] = "no_sam_mask"
            elif int(rmask.sum()) < min_pixels:
                row["status"] = "not_visible"
                row["rendered_px"] = int(rmask.sum())
            else:
                row["status"] = "scored"
                row.update(mask_scores(rmask, ref))
                row.update(depth_relative_error(rdepth, depth[f], rmask & ref))
                if f == canonical_idx:
                    obs = backproject(depth[f], intrinsics[f], ref)
                    row.update(one_sided_chamfer(obs, surface_c[name], seed=seed))
            rows[name].append(row)

    report = {"canonical_frame": int(canonical_idx), "n_frames": int(n_frames), "objects": {}}
    for name, all_rows in rows.items():
        same = next(r for r in all_rows if r["canonical"])
        held = [r for r in all_rows if not r["canonical"]]
        scored = [r for r in held if r["status"] == "scored"]
        report["objects"][name] = {
            "same_frame": same,
            "held_out": {
                "n_scored": len(scored),
                "n_no_sam_mask": sum(r["status"] == "no_sam_mask" for r in held),
                "n_not_visible": sum(r["status"] == "not_visible" for r in held),
                "iou": _aggregate(scored, "iou"),
                "depth_rel_median": _aggregate(scored, "depth_rel_median"),
                "per_frame": held,
            },
        }
    if keep_renders:
        report["_renders"] = renders
    return report


def format_report(report: dict, phrases: dict[str, str] | None = None) -> str:
    """Markdown summary. The wording keeps the same-frame number labelled as a fit residual."""
    phrases = phrases or {}
    c = report["canonical_frame"]
    lines = [
        "# 场景重建一致性报告",
        "",
        f"规范帧:第 {c} 帧(共 {report['n_frames']} 帧)。**这些数字衡量的是网格与观测的一致程度,不是相对真实场景的精度。**",
        "",
        "## 同一帧(拟合残差,不独立)",
        "",
        "第 8 步就是对这一帧的深度和掩码拟合的,所以这里的数字主要是拟合残差,同时用来检验位姿约定有没有用反。",
        "",
        "| 物体 | 提示词 | 掩码 IoU | 深度相对误差(中位数) | 单向 Chamfer 中位数 (mm) | 状态 |",
        "|---|---|---|---|---|---|",
    ]
    for name, obj in report["objects"].items():
        s = obj["same_frame"]
        chamfer = s.get("chamfer_median_m")
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                name, phrases.get(name, ""),
                f"{s['iou']:.3f}" if "iou" in s else "-",
                f"{s['depth_rel_median']:.3f}" if s.get("depth_rel_median") is not None else "-",
                f"{chamfer * 1000:.1f}" if chamfer is not None else "-",
                s["status"],
            )
        )
    lines += [
        "",
        "## 留出视角(重投影一致性)",
        "",
        "其余帧没有参与拟合。网格借助深度模型估计的相机外参变到该帧,与该帧上 SAM 3 的分割、深度模型自己的深度比较。外参本身也是估计的,所以这不是真值。",
        "",
        "| 物体 | 提示词 | 评分帧数 | 无分割 | 不可见 | IoU 中位数 | IoU 最小-最大 | 深度相对误差中位数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, obj in report["objects"].items():
        h = obj["held_out"]
        iou, dep = h["iou"], h["depth_rel_median"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, phrases.get(name, ""), h["n_scored"], h["n_no_sam_mask"], h["n_not_visible"],
                f"{iou['median']:.3f}" if iou["median"] is not None else "-",
                f"{iou['min']:.3f} - {iou['max']:.3f}" if iou["median"] is not None else "-",
                f"{dep['median']:.3f}" if dep["median"] is not None else "-",
            )
        )
    lines += [
        "",
        "## 解读时要注意",
        "",
        "- 输入深度分辨率低(见报告元数据),小物体只有几十个像素,IoU 的量化误差大。",
        "- 网格是完整的而分割只看到可见部分;遮挡和出画的帧会被标成不可见,不计入分数。",
        "- SAM 3 找不到物体的帧单独计数,不当作 0 分。",
        "- 网格若带有原物体没有的部分(例如被补全出来的杯盖),轮廓会偏大,IoU 会偏低。",
    ]
    return "\n".join(lines) + "\n"
