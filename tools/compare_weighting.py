#!/usr/bin/env python3
"""Comparatif apparié de deux bras de pondération de loss (uniform vs snr).

Lit les `*_metrics.jsonl` des deux bras et sort, par tranche de t :
la loss de sonde finale (moyenne des N derniers points), le meilleur point,
le rapport entre bras, et la comparaison au prédicteur trivial ε̂ = x_t
(0,0004 en t[192-256), cf. SCALE_UNET §5.2).

Usage :
  python3 tools/compare_weighting.py uniform:runs/uniform_metrics.jsonl \
                                     snr:runs/snr_g1_metrics.jsonl
"""
import json
import sys

# Prédicteur trivial ε̂ = x_t, mesuré sur 256 vraies images (tools/trivial_baselines.py).
TRIVIAL = [0.8735, 0.1402, 0.0114, 0.0004]
BUCKETS = ["t[0-64)", "t[64-128)", "t[128-192)", "t[192-256)"]
TAIL = 5


def load(path):
    probes = []
    with open(path) as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("kind") == "train_probe":
                probes.append(o)
    probes.sort(key=lambda o: o["step"])
    return probes


def losses(probes):
    """[step] -> [loss par tranche]"""
    out = []
    for p in probes:
        row = [b["loss"] for b in p["buckets"]]
        out.append((p["step"], row))
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    arms = {}
    for spec in sys.argv[1:]:
        label, _, path = spec.partition(":")
        arms[label] = losses(load(path))

    names = list(arms)
    n_buckets = len(arms[names[0]][0][1])

    print(f"points de sonde : " + ", ".join(f"{k}={len(v)}" for k, v in arms.items()))
    print()
    header = f"{'tranche':<12}" + "".join(f"{n:>14}" for n in names)
    if len(names) == 2:
        header += f"{'rapport':>10}"
    header += f"{'trivial':>10}{'verdict':>12}"
    print(header)
    print("-" * len(header))

    finals = {}
    for b in range(n_buckets):
        cells = []
        for n in names:
            tail = [row[b] for _, row in arms[n][-TAIL:]]
            finals.setdefault(n, []).append(sum(tail) / len(tail))
            cells.append(finals[n][b])
        line = f"{BUCKETS[b] if b < len(BUCKETS) else b:<12}"
        line += "".join(f"{c:>14.5f}" for c in cells)
        if len(names) == 2:
            a, c = cells
            line += f"{(a / c if c else float('inf')):>10.2f}"
        triv = TRIVIAL[b] if b < len(TRIVIAL) else float("nan")
        line += f"{triv:>10.4f}"
        best = min(cells)
        line += f"{'SOUS trivial' if best < triv else 'au-dessus':>12}"
        print(line)

    print()
    print("meilleur point atteint sur tout le run (min sur les pas) :")
    for n in names:
        mins = [min(row[b] for _, row in arms[n]) for b in range(n_buckets)]
        print(f"  {n:<12}" + "".join(f"{m:>14.5f}" for m in mins))

    print()
    print("trajectoire haut-t (t[192-256)), un point sur 4 :")
    for n in names:
        traj = [row[-1] for _, row in arms[n]][::4]
        print(f"  {n:<12}" + " ".join(f"{v:.4f}" for v in traj[:20]))


if __name__ == "__main__":
    main()
