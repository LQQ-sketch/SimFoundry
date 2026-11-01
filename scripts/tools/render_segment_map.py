#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Draw a stage-9 object whole, one colour per P3-SAM segment, with each segment's id written on it.

The renders the VLM merge step sees are exploded and crowded; this is the picture a person marks up
("this id is the right side panel, keep it fixed") so the ids can be corrected by hand. Orthographic,
painter's algorithm, no GPU, no display. Run in any env with numpy, trimesh and Pillow::

    python scripts/tools/render_segment_map.py --object-dir \\
        Data/drawer2_test/s9_articulate_objects/drawer2_test/chest_of_drawers \\
        --out-dir /data/projects/simfoundry/segment_map

``--assign`` takes a JSON file ``{"base": [3, 7], "drawer_1": [4]}``; segments it names are drawn in the
group's colour and everything else in grey, so a proposed grouping can be checked by eye before it is used.
The mesh's own axes are not known to be z-up, so six axis views and two oblique ones are written; the
file name says where the camera is.
"""
import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

SIZE = 1000
MARGIN = 70


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
    face_ids = np.load(parts / "face_ids.npy")
    if len(face_ids) != len(mesh.faces):
        raise SystemExit(f"face_ids has {len(face_ids)} entries but the mesh has {len(mesh.faces)} faces")
    return mesh, face_ids


def views():
    """name -> (direction from the object to the camera, up vector)."""
    z, y = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    out = {
        "from_plus_x": (np.array([1.0, 0, 0]), z), "from_minus_x": (np.array([-1.0, 0, 0]), z),
        "from_plus_y": (np.array([0.0, 1, 0]), z), "from_minus_y": (np.array([0.0, -1, 0]), z),
        "from_plus_z": (z, y), "from_minus_z": (-z, y),
    }
    for name, az in (("oblique_a", 40.0), ("oblique_b", 220.0)):
        a, e = np.radians(az), np.radians(25.0)
        out[name] = (np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]), z)
    return out


def palette(ids):
    colours = {}
    for k, seg in enumerate(sorted(i for i in ids if i >= 0)):
        r, g, b = colorsys.hsv_to_rgb((k * 0.381966) % 1.0, 0.65, 0.95)
        colours[seg] = (int(255 * r), int(255 * g), int(255 * b))
    colours[-1] = (170, 170, 170)
    return colours


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def render(tri, face_ids, direction, up, colours, labels, title):
    d = direction / np.linalg.norm(direction)
    right = np.cross(-d, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, -d)
    pts = tri.reshape(-1, 3)
    xy = np.stack([pts @ right, pts @ up], axis=1)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    scale = (SIZE - 2 * MARGIN) / max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
    px = (xy - lo) * scale + MARGIN
    px[:, 1] = SIZE - px[:, 1]  # image y grows downward
    px = px.reshape(-1, 3, 2)
    depth = (tri @ d).mean(axis=1)

    colour_img = Image.new("RGB", (SIZE, SIZE), (245, 245, 245))
    id_img = Image.new("I", (SIZE, SIZE), 0)
    cd, idd = ImageDraw.Draw(colour_img), ImageDraw.Draw(id_img)
    for f in np.argsort(depth):  # far to near
        poly = [tuple(p) for p in px[f]]
        cd.polygon(poly, fill=colours[int(face_ids[f])], outline=(60, 60, 60))
        idd.polygon(poly, fill=int(face_ids[f]) + 2)  # +2 keeps -1 distinct from the empty background (0)

    id_map = np.asarray(id_img)
    draw = ImageDraw.Draw(colour_img)
    small = font(26)
    for seg in sorted(set(int(i) for i in np.unique(face_ids))):
        ys, xs = np.nonzero(id_map == seg + 2)
        if len(xs) < 40:
            continue  # sliver, or hidden from this side
        cx, cy = np.median(xs), np.median(ys)
        k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))  # a pixel that really belongs to the segment
        x, y = int(xs[k]), int(ys[k])
        text = labels.get(seg, str(seg))
        box = draw.textbbox((x, y), text, font=small, anchor="mm")
        draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((x, y), text, font=small, fill=(0, 0, 0), anchor="mm")
    draw.text((12, 10), title, font=font(30), fill=(0, 0, 0))
    return colour_img


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--object-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--assign", type=Path, help="JSON {group: [segment ids]}; colours by group")
    args = parser.parse_args(argv)

    mesh, face_ids = load_faces(args.object_dir.resolve())
    tri = mesh.triangles
    ids = set(int(i) for i in np.unique(face_ids))
    colours, labels = palette(ids), {}
    if args.assign:
        groups = json.loads(args.assign.read_text())
        grey = {i: (190, 190, 190) for i in ids}
        group_colours = palette(set(range(len(groups))))
        for k, (name, members) in enumerate(groups.items()):
            for seg in members:
                grey[int(seg)] = group_colours[k]
                labels[int(seg)] = f"{seg} {name}"
        colours = {**colours, **grey}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sheet = []
    for name, (direction, up) in views().items():
        image = render(tri, face_ids, direction, up, colours, labels, name.replace("_", " "))
        image.save(args.out_dir / f"{name}.png")
        sheet.append(image.resize((500, 500)))
    grid = Image.new("RGB", (2000, 1000), (255, 255, 255))
    for k, tile in enumerate(sheet):
        grid.paste(tile, ((k % 4) * 500, (k // 4) * 500))
    grid.save(args.out_dir / "all_views.png")
    print(f"wrote {len(sheet)} views and all_views.png to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
