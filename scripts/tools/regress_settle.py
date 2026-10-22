#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Regression check for the settle gate (scripts/interactive/light_editor/settle.py).

A gate is only worth trusting if it is shown to catch faults it should catch. This injects known
faults into copies of a stable scene, runs settle.py on each copy, and asserts the faulty props are
reported as unsettled while a clean copy reports nothing:

  baseline          unmodified scene                      -> nothing flagged (false-alarm check)
  float_and_sink    one prop lifted, one sunk into floor  -> both flagged, the control prop is not
  overlap           one prop moved into another prop's    -> the mover is flagged
                    wall

Every faulty case is also run with --no-wake, which reproduces the old behaviour (rigid bodies left
asleep after restore, so nothing ever moves). That gives the before/after detection count.

Run it in the `simfoundry` env on a GPU machine (each settle run boots Isaac Sim, roughly a minute
or two; five runs by default):

    python scripts/tools/regress_settle.py --scene /abs/path/to/<scene>_scene_state_settled_....json

The scene must already pass settle.py, otherwise the baseline fails and the faults cannot be
attributed. The copies are written next to the scene (usd paths in the JSON are relative to it)
and removed afterwards unless --keep. Exit status is 0 only if the wake-enabled gate passes.
"""
import argparse
import json
import sys
import time
import subprocess
from pathlib import Path

SETTLE = Path(__file__).resolve().parents[1] / "interactive" / "light_editor" / "settle.py"


def registry(doc):
    return doc["state"]["registry"]["object_registry"]


def get_pos(doc, name):
    return list(registry(doc)[name]["root_link"]["pos"])


def place(doc, name, pos):
    """Move an object and zero its velocities (saved states carry live velocities)."""
    link = registry(doc)[name]["root_link"]
    link["pos"] = [float(v) for v in pos]
    link["lin_vel"] = [0.0, 0.0, 0.0]
    link["ang_vel"] = [0.0, 0.0, 0.0]


def build_cases(args):
    """Return [(case_name, mutate(doc) or None, expected_flagged, allowed_to_move)]."""

    def float_and_sink(doc):
        p = get_pos(doc, args.lift)
        p[2] += args.lift_dz
        place(doc, args.lift, p)
        q = get_pos(doc, args.sink)
        q[2] -= args.sink_dz
        place(doc, args.sink, q)

    def overlap(doc):
        target = get_pos(doc, args.target)
        mover = get_pos(doc, args.mover)
        place(doc, args.mover, [target[0] + args.overlap_offset, target[1], mover[2]])

    return [
        ("baseline", None, set(), set()),
        ("float_and_sink", float_and_sink, {args.lift, args.sink}, set()),
        # The target may be shoved by the mover; that is not a false alarm.
        ("overlap", overlap, {args.mover}, {args.target}),
    ]


def run_settle(scene, report_path, log_path, steps, no_wake, timeout):
    cmd = [sys.executable, str(SETTLE), "--scene", str(scene), "--report", str(report_path),
           "--steps", str(steps)]
    if no_wake:
        cmd.append("--no-wake")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    report = None
    if Path(report_path).exists():
        try:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        except ValueError:
            report = None
    return proc.returncode, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="a scene state JSON that already passes settle.py")
    ap.add_argument("--out-dir", default="regress_settle_out")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per settle run")
    ap.add_argument("--lift", default="iter_2", help="prop to float (default: the cup)")
    ap.add_argument("--lift-dz", type=float, default=0.05)
    ap.add_argument("--sink", default="iter_1", help="prop to push into the floor (default: the pen)")
    ap.add_argument("--sink-dz", type=float, default=0.03)
    ap.add_argument("--mover", default="iter_2", help="prop moved into another prop (default: the cup)")
    ap.add_argument("--target", default="iter_0", help="prop it is moved into (default: the bowl)")
    ap.add_argument("--overlap-offset", type=float, default=0.13,
                    help="x offset from the target's centre, about the bowl's radius so the mover "
                         "lands in its wall rather than its cavity")
    ap.add_argument("--keep", action="store_true", help="keep the injected copies and settled outputs")
    ap.add_argument("--dry-run", action="store_true", help="write the injected copies and print the "
                    "changes, but do not run settle.py (no simulator needed)")
    args = ap.parse_args()

    scene = Path(args.scene).resolve()
    base = json.loads(scene.read_text(encoding="utf-8"))
    names = set(registry(base))
    for label, n in (("--lift", args.lift), ("--sink", args.sink), ("--mover", args.mover),
                     ("--target", args.target)):
        if n not in names:
            sys.exit(f"{label} {n!r} is not in the scene. Objects: {', '.join(sorted(names))}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = scene.stem
    prefix = stem.split("_scene_state_")[0] if "_scene_state_" in stem else stem

    cases = build_cases(args)
    results, created = [], []
    for case, mutate, expected, allowed in cases:
        doc = json.loads(json.dumps(base))
        if mutate:
            mutate(doc)
        # Named like a scene state so settle.py derives its output names the usual way.
        copy_path = scene.parent / f"{prefix}_scene_state_regress{case}.json"
        copy_path.write_text(json.dumps(doc, indent=4), encoding="utf-8")
        created.append(copy_path)
        if mutate:
            for n in sorted(expected | allowed):
                print(f"[regress] {case}: {n} {get_pos(base, n)} -> {get_pos(doc, n)}")
        if args.dry_run:
            continue

        modes = [("wake", False)] if case == "baseline" else [("wake", False), ("no_wake", True)]
        for mode, no_wake in modes:
            tag = f"{case}_{mode}"
            print(f"[regress] {time.strftime('%H:%M:%S')} running {tag} ...", flush=True)
            code, report = run_settle(copy_path, out_dir / f"{tag}.json", out_dir / f"{tag}.log",
                                      args.steps, no_wake, args.timeout)
            if report is None or report.get("error"):
                results.append(dict(case=case, mode=mode, error=(report or {}).get("error") or
                                    f"no report (exit {code}); see {tag}.log"))
                continue
            flagged = {m["name"] for m in report.get("moved", [])}
            results.append(dict(
                case=case, mode=mode, exit=code, woken=report.get("woken"),
                flagged=sorted(flagged), expected=sorted(expected),
                detected=sorted(expected & flagged), missed=sorted(expected - flagged),
                unexpected=sorted(flagged - expected - allowed),
                moves={e["name"]: round(e["delta"], 4) for e in report.get("all", [])},
            ))
            if report.get("settled_path"):
                created.append(Path(report["settled_path"]))

    if not args.dry_run:
        (out_dir / "regress_settle_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not args.keep:
        for p in created:
            if p.exists():
                p.unlink()

    if args.dry_run:
        print("[regress] dry run: injected copies written and removed; settle.py was not run")
        return 0

    print("\n=== results ===")
    for r in results:
        if "error" in r:
            print(f"{r['case']:16s} {r['mode']:8s} ERROR  {r['error']}")
            continue
        print(f"{r['case']:16s} {r['mode']:8s} detected {r['detected']} missed {r['missed']} "
              f"unexpected {r['unexpected']} woken={r['woken']} moves={r['moves']}")

    faults = [r for r in results if r["case"] != "baseline" and "error" not in r]
    total = sum(len(r["expected"]) for r in faults if r["mode"] == "wake")
    new = sum(len(r["detected"]) for r in faults if r["mode"] == "wake")
    old = sum(len(r["detected"]) for r in faults if r["mode"] == "no_wake")
    print(f"\ninjected faults: {total}   caught with wake: {new}/{total}   caught with --no-wake (old): {old}/{total}")

    wake = [r for r in results if r["mode"] == "wake"]
    ok = (bool(wake) and len(wake) == 3
          and all("error" not in r and not r["missed"] and not r["unexpected"] for r in wake))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
