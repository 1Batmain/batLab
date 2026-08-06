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
    """Parse a metrics JSONL, REPORTING what it could not parse.

    A malformed line is not skipped quietly. During this mission a 600-step
    validation run was silently corrupted because two `--headless-train` runs of
    the same model wrote to the same default metrics path and interleaved; the
    only reason it was noticed is that the file stopped parsing. A loader that
    swallowed the bad line would have compared 539 records against 822 and
    printed a confident, meaningless answer.
    """
    records, bad = [], []
    with open(path, errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                bad.append((number, len(line), str(err)))
    if bad:
        print(f"!! {path}: {len(bad)} unparseable line(s) — the file is CORRUPT")
        for number, length, err in bad[:5]:
            print(f"   line {number} ({length} chars): {err}")
        print("   most likely two runs shared a metrics path; pass a distinct --out")
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

    # Aligned on (kind, step), not zipped by position: one dropped or extra
    # record would otherwise shift every later comparison and turn a single
    # missing line into thousands of spurious differences.
    # The key must identify a record by its OWN coordinates, not by its
    # position in the file. Two reasons, both met in practice:
    #
    #   - (kind, step) is not unique: a `sample` step emits one `denoise_step`
    #     per point of the reverse chain, all sharing the same step. Keying on
    #     the pair alone collapsed 256 of them onto one and compared diffusion
    #     step 0 against 255 — a "255.0 divergence" that was pure bookkeeping;
    #   - falling back to position within the group is no better as soon as one
    #     record is missing (a corrupt file, an interrupted run): everything
    #     after it shifts and every later comparison is between unrelated pairs.
    #
    # So the key is built from whichever identity fields a record carries.
    # `denoise_step` uses `train_step`/`step_index`, the others use `step`.
    IDENTITY = ("step", "train_step", "step_index", "diffusion_step", "seed", "bucket")

    def key_of(record):
        kind = record.get("kind") or record.get("type") or "?"
        return (kind,) + tuple(record.get(field) for field in IDENTITY)

    old_keys = [key_of(record) for record in old_records]
    new_by_key = {key_of(record): record for record in new_records}
    assert len(new_by_key) == len(new_records), "identity key is not unique — widen IDENTITY"

    unmatched = 0
    for old, key in zip(old_records, old_keys):
        new = new_by_key.get(key)
        if new is None:
            unmatched += 1
            continue
        kind = key[0]
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
    print(f"unmatched (kind, step): {unmatched}")

    print("\nper record kind:      max abs      max rel")
    for kind, (a, r) in sorted(per_kind.items()):
        print(f"  {kind:<16} {a:11.3e}  {r:11.3e}")

    # The trajectory itself, side by side.
    print("\ntrain_loss trajectory (old | new):")
    losses = [
        (o, new_by_key[k])
        for o, k in zip(old_records, old_keys)
        if (o.get("kind") or o.get("type")) == "train_loss" and k in new_by_key
    ]
    for old, new in losses[:: max(1, len(losses) // 12)]:
        step = old.get("step")
        a, b = old.get("loss"), new.get("loss")
        print(f"  step {step:>4}  {a:.8f} | {b:.8f}")


if __name__ == "__main__":
    main()
