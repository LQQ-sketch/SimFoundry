#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Print what each P3-SAM segment id of a stage-9 object actually is, in plain numbers.

The VLM merge step (stage 9, step 4) decides which segment ids are drawers, lids or base from
coloured, numbered renders. When it gets that wrong, a person needs the same ids in a form that
can be read without the renders: where each segment sits in the object, how big it is, and which
way it faces. Read-only; run in any env that has numpy and trimesh::

    python scripts/tools/inspect_segments.py --object-dir \\
        Data/drawer2_test/s9_articulate_objects/drawer2_test/chest_of_drawers

Position columns are fractions of the object's own bounding box (0 = min side, 1 = max side) along
the mesh's x, y and z; ``thin`` is the axis the segment is thinnest along (a panel's normal), and the
extents are the segment's size as a fraction of the whole object's.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def load_faces(object_dir: Path):
    parts = object_dir / "s3_segment_mesh/rendered_parts"
    mesh_path, ids_path = parts / "original_mesh.glb", parts / "face_ids.npy"
    for path in (mesh_path, ids_path):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    loaded = trimesh.load(str(mesh_path))
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
    else:
        mesh = loaded
    face_ids = np.load(ids_path)
    if len(face_ids) != len(mesh.faces):
        raise SystemExit(f"face_ids has {len(face_ids)} entries but the mesh has {len(mesh.faces)} faces")
    return mesh, face_ids


def describe(mesh, face_ids):
    lo, hi = mesh.bounds
    span = np.maximum(hi - lo, 1e-9)
    tri = mesh.triangles
    areas = mesh.area_faces
    centers = tri.mean(axis=1)
    rows = []
    for seg in np.unique(face_ids):
        sel = face_ids == seg
        weight = areas[sel]
        if weight.sum() <= 0:
            continue
        centroid = (centers[sel] * weight[:, None]).sum(axis=0) / weight.sum()
        frac = (centroid - lo) / span
        sub = tri[sel].reshape(-1, 3)
        extent = (sub.max(axis=0) - sub.min(axis=0)) / span
        # A panel's normal is the axis it is thinnest along; the area-weighted normal sum is no use
        # on a closed piece, where it cancels to zero.
        thin = "xyz"[int(np.argmin(extent))]
        rows.append((int(seg), int(sel.sum()), weight.sum(), frac, thin, extent))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--object-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    mesh, face_ids = load_faces(args.object_dir.resolve())
    print(f"mesh: {len(mesh.faces)} faces, bounds {np.round(mesh.bounds[0], 3)} to {np.round(mesh.bounds[1], 3)}")
    total = mesh.area_faces.sum()
    print(f"{'id':>3} {'faces':>6} {'area%':>6}  {'pos x':>5} {'pos y':>5} {'pos z':>5}  {'thin':>4}  {'ext x':>5} {'ext y':>5} {'ext z':>5}")
    for seg, n, area, frac, thin, ext in sorted(describe(mesh, face_ids), key=lambda r: -r[2]):
        label = " -1" if seg < 0 else f"{seg:>3}"
        print(f"{label} {n:>6} {100 * area / total:>6.1f}  {frac[0]:>5.2f} {frac[1]:>5.2f} {frac[2]:>5.2f}  {thin:>4}  {ext[0]:>5.2f} {ext[1]:>5.2f} {ext[2]:>5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
