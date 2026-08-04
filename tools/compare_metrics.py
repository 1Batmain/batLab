#!/usr/bin/env python3
"""Compare deux runs via leurs `*_metrics.jsonl` (loss par tranche de t, ε̂).

La loss batch ne suffit pas : le signal utile est la loss par tranche de `t`
(cf. INSIGHTS_TRAINING §2.1), en particulier la tranche basse t[0-64) qui
plafonne à ~0,61 sur le petit modèle.

Usage :
  python3 tools/compare_metrics.py baseline:<a.jsonl> L:<b.jsonl>
"""
import json
import sys


def load(path):
    probes, losses = [], []
    with open(path) as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("kind") == "train_probe":
                probes.append(o)
            elif o.get("kind") == "train_loss":
                losses.append(o)
    probes.sort(key=lambda o: o["step"])
    losses.sort(key=lambda o: o["step"])
    return probes, losses


def at(probes, step):
    """Dernier probe dont le step <= `step` (None si le run est plus court)."""
    hit = [p for p in probes if p["step"] <= step]
    return hit[-1] if hit else None


def main():
    runs = []
    for arg in sys.argv[1:]:
        label, path = arg.split(":", 1)
        probes, losses = load(path)
        runs.append((label, probes, losses))
    if not runs:
        sys.exit(__doc__)

    marks = [0, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
    header = f"{'step':>6} | {'run':<10} | " + " | ".join(
        f"{n:>9}" for n in ("t0-64", "t64-128", "t128-192", "t192-256")
    ) + " | eps_hat.std(hi) | batch_loss"
    print(header)
    print("-" * len(header))
    for step in marks:
        for label, probes, losses in runs:
            p = at(probes, step)
            if p is None or (step > 0 and p["step"] < step - 200):
                continue
            b = p["buckets"]
            bl = [x for x in losses if x["step"] <= step]
            print(
                f"{p['step']:>6} | {label:<10} | "
                + " | ".join(f"{x['loss']:9.4f}" for x in b)
                + f" | {b[-1]['eps_hat']['std']:15.3f}"
                + (f" | {bl[-1]['loss']:.4f}" if bl else "")
            )
        print()


if __name__ == "__main__":
    main()
