#!/usr/bin/env python3
"""Compare two training-metrics JSONL files, record by record.

Used for the paired old-path/new-path validation run: the two runs are
deterministic (fixed-seed weight init, timestep and noise draws derived from the
step counter), so every number should agree to within the floating-point
re-association the batched reductions introduce — and nothing else.

What it prints, in the order that matters for a diffusion model:

  - `train_loss`: the headline trajectory;
  - `train_probe`: the loss per timestep bucket, which is the diagnostic that
    actually says whether the model uses `t` (INSIGHTS_TRAINING.md: a batch loss
    that goes down is not enough);
  - everything else, so a divergence hiding in a field nobody looks at still
    shows up.

Usage: compare_metrics.py old.jsonl new.jsonl
"""

import json
import math
import sys
from collections import defaultdict


def load(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, sub, out)
    elif isinstance(value, list):
        for index, sub in enumerate(value):
            flatten(f"{prefix}[{index}]", sub, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def main():
    old_records, new_records = load(sys.argv[1]), load(sys.argv[2])
    print(f"records: old {len(old_records)}, new {len(new_records)}")
    if len(old_records) != len(new_records):
        print("!! record counts differ — the two runs did not do the same work")

    worst_abs = ("", 0.0)
    worst_rel = ("", 0.0)
    per_kind = defaultdict(lambda: [0.0, 0.0])  # kind -> [max abs, max rel]
    non_finite = 0
    shape_mismatch = 0

    for old, new in zip(old_records, new_records):
        kind = old.get("kind") or old.get("type") or "?"
        if (new.get("kind") or new.get("type") or "?") != kind:
            shape_mismatch += 1
            continue
        old_flat, new_flat = {}, {}
        flatten("", old, old_flat)
        flatten("", new, new_flat)
        if old_flat.keys() != new_flat.keys():
            shape_mismatch += 1
        for key in old_flat.keys() & new_flat.keys():
            a, b = old_flat[key], new_flat[key]
            if not (math.isfinite(a) and math.isfinite(b)):
                non_finite += 1
                continue
            diff = abs(a - b)
            if diff > worst_abs[1]:
                worst_abs = (f"{kind}.{key}", diff)
            per_kind[kind][0] = max(per_kind[kind][0], diff)
            # Relative differences are meaningless on values at the rounding
            # floor; the 1e-3 gate is the same one PERF_CONVOLUTION.md used.
            if max(abs(a), abs(b)) > 1e-3:
                rel = diff / max(abs(a), abs(b))
                if rel > worst_rel[1]:
                    worst_rel = (f"{kind}.{key}", rel)
                per_kind[kind][1] = max(per_kind[kind][1], rel)

    print(f"\nworst absolute diff : {worst_abs[1]:.3e}  ({worst_abs[0]})")
    print(f"worst relative diff : {worst_rel[1]:.3e}  ({worst_rel[0]})  [|v| > 1e-3]")
    print(f"non-finite values   : {non_finite}")
    print(f"structure mismatches: {shape_mismatch}")

    print("\nper record kind:      max abs      max rel")
    for kind, (a, r) in sorted(per_kind.items()):
        print(f"  {kind:<16} {a:11.3e}  {r:11.3e}")

    # The trajectory itself, side by side.
    print("\ntrain_loss trajectory (old | new):")
    losses = [
        (o, n)
        for o, n in zip(old_records, new_records)
        if (o.get("kind") or o.get("type")) == "train_loss"
    ]
    for old, new in losses[:: max(1, len(losses) // 12)]:
        step = old.get("step")
        a, b = old.get("loss"), new.get("loss")
        print(f"  step {step:>4}  {a:.8f} | {b:.8f}")


if __name__ == "__main__":
    main()
