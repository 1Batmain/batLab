#!/usr/bin/env python3
"""Compare paired training runs from their *_metrics.jsonl.

Every run in the comparison sees byte-identical data, diffusion timesteps and
noise: `main.rs` derives the per-step seed as `(step as u64) << 32` and the probe
seed as `0x50B0_1234 ^ step`, both pure functions of the step index. So a
difference between two runs at the same step is an optimiser/init effect and
nothing else. The empirical check of that claim is `--check-pairing`: the step-0
batch loss must be bit-identical across arms (it is measured before any update).

Reads:
  train_loss  — batch loss on the training batch (noisy, 16 samples)
  train_probe — the diagnostic that matters: loss per bucket of t, on a FIXED
                probe set with a per-step-deterministic noise draw, so it is
                directly comparable across runs.
"""
import argparse
import json
import math
import os
import sys


def load(path):
    train, probe = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["kind"] == "train_loss":
                train.append((d["step"], d["loss"]))
            elif d["kind"] == "train_probe":
                probe.append(
                    (
                        d["step"],
                        [(b["t_lo"], b["t_hi"], b["loss"]) for b in d["buckets"]],
                        [(b["eps_hat"]["std"], b["eps_target"]["std"]) for b in d["buckets"]],
                        sum(b["eps_hat"]["non_finite"] for b in d["buckets"]),
                    )
                )
    train.sort()
    probe.sort()
    return {"train": train, "probe": probe}


def bucket_series(run, idx):
    """(step, loss) for bucket `idx`; idx -1 means the mean over all buckets."""
    out = []
    for step, buckets, _, _ in run["probe"]:
        if idx < 0:
            out.append((step, sum(b[2] for b in buckets) / len(buckets)))
        else:
            out.append((step, buckets[idx][2]))
    return out


def steps_to(series, threshold):
    """First step whose loss is <= threshold AND stays <= it for the rest of the
    run. A single lucky dip below a threshold is not 'reaching' it."""
    for i, (step, value) in enumerate(series):
        if value <= threshold and all(v <= threshold for _, v in series[i:]):
            return step
    return None


def tail(series, n=5):
    vals = [v for _, v in series[-n:]]
    return sum(vals) / len(vals) if vals else float("nan")


def sparkline(values, lo=None, hi=None):
    chars = "▁▂▃▄▅▆▇█"
    vals = [math.log10(max(v, 1e-6)) for v in values]
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if hi - lo < 1e-12:
        return chars[0] * len(vals)
    return "".join(chars[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="tag=path/to/metrics.jsonl")
    ap.add_argument("--thresholds", default="0.2,0.1,0.05,0.02,0.01")
    ap.add_argument("--check-pairing", action="store_true")
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        tag, _, path = spec.partition("=")
        if not path:
            path, tag = tag, os.path.basename(tag).replace("_metrics.jsonl", "")
        if not os.path.exists(path):
            print(f"!! missing: {path}", file=sys.stderr)
            continue
        runs[tag] = load(path)
    if not runs:
        sys.exit("no runs loaded")

    for tag, run in runs.items():
        last = run["probe"][-1][0] if run["probe"] else -1
        print(f"{tag:24s} probes={len(run['probe']):4d} last_step={last}")
    print()

    if args.check_pairing:
        print("== pairing check: step-0 batch loss must be bit-identical ==")
        zero = {}
        for tag, run in runs.items():
            v = next((l for s, l in run["train"] if s == 0), None)
            zero[tag] = v
            print(f"  {tag:24s} {v!r}")
        distinct = {repr(v) for v in zero.values() if v is not None}
        print(f"  -> {'IDENTICAL' if len(distinct) == 1 else 'DIVERGENT'}\n")

    buckets = runs[next(iter(runs))]["probe"][0][1]
    labels = [f"t[{lo},{hi})" for lo, hi, _ in buckets] + ["mean"]
    indices = list(range(len(buckets))) + [-1]

    print("== probe loss, mean of the last 5 probes ==")
    head = "run".ljust(24) + "".join(l.rjust(14) for l in labels)
    print(head)
    for tag, run in runs.items():
        row = tag.ljust(24)
        for idx in indices:
            row += f"{tail(bucket_series(run, idx)):14.5f}"
        print(row)
    print()

    print("== non-finite ε̂ over the whole run (0 = healthy) ==")
    for tag, run in runs.items():
        print(f"  {tag:24s} {sum(p[3] for p in run['probe'])}")
    print()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    for idx, label in zip(indices, labels):
        print(f"== steps to reach (and hold) a probe loss on {label} ==")
        head = "run".ljust(24) + "".join(f"<={t:g}".rjust(10) for t in thresholds)
        print(head)
        for tag, run in runs.items():
            series = bucket_series(run, idx)
            row = tag.ljust(24)
            for t in thresholds:
                s = steps_to(series, t)
                row += ("never" if s is None else str(s)).rjust(10)
            print(row)
        print()

    print("== probe loss trajectory (log scale, common per bucket) ==")
    for idx, label in zip(indices, labels):
        allv = [v for run in runs.values() for _, v in bucket_series(run, idx)]
        lo, hi = math.log10(max(min(allv), 1e-6)), math.log10(max(allv))
        print(f"  {label}  [{min(allv):.4f} .. {max(allv):.4f}]")
        for tag, run in runs.items():
            s = bucket_series(run, idx)
            print(f"    {tag:22s} {sparkline([v for _, v in s], lo, hi)}")
        print()

    print("== probe loss at checkpoints ==")
    marks = [0, 100, 250, 500, 750, 1000, 1250, 1499]
    for idx, label in zip(indices, labels):
        if idx != -1 and idx != len(buckets) - 1:
            continue
        print(f"  {label}")
        print("    " + "run".ljust(22) + "".join(f"s{m}".rjust(11) for m in marks))
        for tag, run in runs.items():
            s = dict(bucket_series(run, idx))
            row = "    " + tag.ljust(22)
            for m in marks:
                near = min(s, key=lambda k: abs(k - m)) if s else None
                row += ("-" if near is None or abs(near - m) > 30 else f"{s[near]:.5f}").rjust(11)
            print(row)
        print()


if __name__ == "__main__":
    main()
