#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Collect the eval_results.json files that run_eval_matrix.sh produced (one results directory per
condition, named by `s15_eval.checkpoint`) into one table: success rate with a Wilson 95%
interval, the share of episodes reaching each milestone, mean milestone progress and mean max
reward.

    python scripts/tools/summarize_eval.py [--root Data/put_cup_test/s15_eval/gr00t]
        [--conditions base floor ...] [--csv out.csv]

With several results directories for one condition, the newest one is used.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion; (0, 0) when n == 0."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newest_results(cond_dir: Path):
    runs = sorted(cond_dir.glob("results_*/eval_results.json"))
    return runs[-1] if runs else None


def summarize(path: Path):
    data = json.loads(path.read_text())
    eps = data.get("episodes", [])
    n = len(eps)
    k = sum(1 for e in eps if e.get("success"))
    lo, hi = wilson(k, n)
    names = []
    for e in eps:
        for m in (e.get("milestones") or {}):
            if m not in names:
                names.append(m)
    ms_rate = {
        m: sum(1 for e in eps if (e.get("milestones") or {}).get(m)) / n if n else 0.0 for m in names
    }
    return {
        "n": n,
        "k": k,
        "lo": lo,
        "hi": hi,
        "progress": sum(e.get("milestone_progress", 0.0) for e in eps) / n if n else 0.0,
        "reward": sum(e.get("episode_reward", 0.0) for e in eps) / n if n else 0.0,
        "milestones": ms_rate,
        "prompt": data.get("prompt", ""),
        "path": str(path.parent),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Data/put_cup_test/s15_eval/gr00t")
    ap.add_argument("--conditions", nargs="*", default=None, help="order/subset; default: every directory")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    conds = args.conditions or sorted(d.name for d in root.iterdir() if d.is_dir())
    rows = {}
    for c in conds:
        res = newest_results(root / c)
        if res is None:
            print(f"# no results for '{c}' under {root / c}")
            continue
        rows[c] = summarize(res)
    if not rows:
        raise SystemExit("no eval_results.json found")

    ms_names = []
    for r in rows.values():
        for m in r["milestones"]:
            if m not in ms_names:
                ms_names.append(m)

    header = ["condition", "n", "success", "95% CI", *ms_names, "milestone progress", "mean max reward"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for c, r in rows.items():
        cells = [
            c,
            str(r["n"]),
            f"{r['k']}/{r['n']} ({100 * r['k'] / r['n']:.0f}%)" if r["n"] else "-",
            f"[{100 * r['lo']:.0f}%, {100 * r['hi']:.0f}%]",
            *[f"{100 * r['milestones'].get(m, 0.0):.0f}%" for m in ms_names],
            f"{100 * r['progress']:.0f}%",
            f"{r['reward']:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    print("\n".join(lines))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["condition", "n", "successes", "ci_lo", "ci_hi", *ms_names, "milestone_progress", "mean_max_reward", "prompt"])
            for c, r in rows.items():
                w.writerow([c, r["n"], r["k"], f"{r['lo']:.4f}", f"{r['hi']:.4f}",
                            *[f"{r['milestones'].get(m, 0.0):.4f}" for m in ms_names],
                            f"{r['progress']:.4f}", f"{r['reward']:.4f}", r["prompt"]])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
