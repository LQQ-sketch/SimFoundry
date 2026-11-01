#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Translate chosen P3-SAM segments of a stage-9 object by a fixed offset, by hand.

A single-photo mesh is built from the photo's pose: if the photo shows a drawer half open, the mesh has
it half open, and the articulation then starts from "open" instead of "closed". This moves the named
segments (e.g. the whole drawer: front plate, side board, inner board, floor) in the mesh's own units so
that joint position 0 is the closed pose. It is a hand correction of geometry, not something the
pipeline recovered, and must be described that way::

    python scripts/tools/shift_segments.py --object-dir \\
        Data/drawer2_test/s9_articulate_objects/drawer2_test/chest_of_drawers \\
        --segments 1 8 10 12 --delta 0 0 -0.484

Without ``--apply`` it writes ``shifted_preview.glb`` next to the real mesh and prints each segment's
bounding box before and after; render it with ``render_segment_map.py --mesh-path``. With ``--apply`` it
backs the original up as ``original_mesh.unshifted.glb`` (and refuses to run again while that backup
exists, so an offset is never applied twice) and overwrites ``original_mesh.glb``, which step 4 reads.

Face order and ``face_ids.npy`` are untouched: the shifted faces get their own copies of their
vertices (and UVs), so neighbouring segments that shared an edge are not dragged along.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals

PARTS = "s3_segment_mesh/rendered_parts"


def load_mesh(path: Path):
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
    return loaded


def shift_faces(mesh, selected: np.ndarray, delta: np.ndarray):
    """Return a copy of ``mesh`` whose selected faces use translated copies of their vertices."""
    faces = np.asarray(mesh.faces)
    used = np.unique(faces[selected])
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = len(mesh.vertices) + np.arange(len(used))
    new_faces = faces.copy()
    new_faces[selected] = remap[faces[selected]]
    vertices = np.vstack([mesh.vertices, mesh.vertices[used] + delta])

    visual = None
    if isinstance(mesh.visual, TextureVisuals) and mesh.visual.uv is not None:
        uv = np.asarray(mesh.visual.uv)
        visual = TextureVisuals(uv=np.vstack([uv, uv[used]]), material=mesh.visual.material)
    else:
        print("WARNING: mesh has no UV texture; the shifted copy is written without colours.")
    return trimesh.Trimesh(vertices=vertices, faces=new_faces, visual=visual, process=False)


def boxes(mesh, face_ids, segments):
    out = {}
    for seg in segments:
        pts = mesh.triangles[face_ids == seg].reshape(-1, 3)
        out[seg] = (pts.min(axis=0), pts.max(axis=0))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--object-dir", required=True, type=Path)
    parser.add_argument("--segments", required=True, type=int, nargs="+")
    parser.add_argument("--delta", required=True, type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                        help="offset in the mesh's own units")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    parts = args.object_dir.resolve() / PARTS
    mesh_path, ids_path = parts / "original_mesh.glb", parts / "face_ids.npy"
    backup = parts / "original_mesh.unshifted.glb"
    for path in (mesh_path, ids_path):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    if args.apply and backup.exists():
        raise SystemExit(f"{backup} exists: a shift was already applied. Restore it first if you want another.")

    mesh = load_mesh(mesh_path)
    face_ids = np.load(ids_path)
    if len(face_ids) != len(mesh.faces):
        raise SystemExit(f"face_ids has {len(face_ids)} entries but the mesh has {len(mesh.faces)} faces")
    unknown = sorted(set(args.segments) - set(int(i) for i in np.unique(face_ids)))
    if unknown:
        raise SystemExit(f"segment ids {unknown} do not exist")

    delta = np.array(args.delta, dtype=float)
    shifted = shift_faces(mesh, np.isin(face_ids, args.segments), delta)

    before, after = boxes(mesh, face_ids, args.segments), boxes(shifted, face_ids, args.segments)
    print(f"offset {delta.tolist()} applied to segments {sorted(args.segments)}")
    for seg in args.segments:
        b, a = before[seg], after[seg]
        print(f"  seg {seg:>3}  min {np.round(b[0], 3)} -> {np.round(a[0], 3)}   max {np.round(b[1], 3)} -> {np.round(a[1], 3)}")

    out = mesh_path if args.apply else parts / "shifted_preview.glb"
    if args.apply:
        shutil.copy(mesh_path, backup)
    shifted.export(str(out))

    # Read it back the way step 4 will: same file, same face count, segments where we put them.
    check = load_mesh(out)
    ok = len(check.faces) == len(face_ids) and all(
        np.allclose(boxes(check, face_ids, [s])[s][0], after[s][0], atol=1e-4) for s in args.segments)
    print(f"\nwrote {out}; read back: {'face count and boxes match' if ok else 'MISMATCH, do not use'}")
    if not args.apply:
        print("(preview only; the real mesh is unchanged. Add --apply to use it.)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
