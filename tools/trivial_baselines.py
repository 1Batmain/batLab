#!/usr/bin/env python3
"""Loss ε de prédicteurs TRIVIAUX (zéro paramètre), tranche par tranche.

Un modèle entraîné doit au minimum battre ces trois-là. Ils sont calculés sur
les vraies images du dataset, avec le schedule de production, et selon le même
protocole de tirage que `probe_diffusion` (t = t_lo et t_lo + span/2).

  A. ε̂ = 0        — ne rien prédire.
  B. ε̂ = x_t      — recopier l'entrée. À `t` élevé, x_t ≈ ε, donc c'est
                     presque exact SANS RIEN APPRENDRE.
  C. ε̂ = (x_t - √ᾱ·x̄₀)/√(1-ᾱ)  où x̄₀ est l'image MOYENNE du dataset —
                     « le modèle qui a appris la moyenne », exactement le
                     comportement qu'on reproche au petit U-Net.

Enjeu : si un modèle entraîné ne bat pas B à `t` élevé, alors sa « loss
quasi-parfaite » en haut de schedule ne démontre aucun apprentissage. Et s'il
n'est pas nettement meilleur que C, il n'a effectivement appris que la moyenne.

Usage : python3 tools/trivial_baselines.py [--n 256]
"""
import argparse
import struct

import numpy as np

BETA_REFERENCE_STEPS = 1000.0
MAX_BETA = 0.999


def schedule(num_steps=256, beta_start=1e-4, beta_end=2e-2):
    scale = BETA_REFERENCE_STEPS / num_steps
    s0, s1 = min(beta_start * scale, MAX_BETA), min(beta_end * scale, MAX_BETA)
    t = np.arange(num_steps) / max(num_steps - 1, 1)
    betas = np.minimum(s0 + (s1 - s0) * t, MAX_BETA)
    return np.cumprod(1.0 - betas, dtype=np.float64)


def load(path, count, offset=0):
    with open(path, "rb") as fh:
        header = fh.read(24)
        magic = header[:8].rstrip(b"\x00").decode("ascii", "replace")
        _, w, h, c = struct.unpack("<IIII", header[8:24])
        per = w * h * c
        fh.seek(24 + per * offset * 4)
        raw = np.frombuffer(fh.read(per * count * 4), dtype="<f4")
    # payload f32 (main.rs:941) ; BATRAW1 en [0,1] -> v*2-1 comme la production
    imgs = raw.reshape(-1, per).astype(np.float64)
    return magic, (imgs * 2.0 - 1.0 if magic == "BATRAW1" else imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/cifar10_grey.batraw")
    ap.add_argument("--n", type=int, default=256, help="images de test")
    ap.add_argument("--mean-from", type=int, default=20000, help="images pour la moyenne")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ab = schedule()
    steps = len(ab)
    magic, pool = load(args.dataset, args.mean_from)
    mean_img = pool.mean(axis=0)
    _, test = load(args.dataset, args.n, offset=args.mean_from + 1000)
    rng = np.random.default_rng(args.seed)

    print(f"dataset {magic} | {args.n} images de test | moyenne sur {args.mean_from}")
    print(f"E[x0²] = {float((test**2).mean()):.4f}\n")
    print(f"{'tranche':>12} {'t tirés':>12} | {'ε̂=0':>8} {'ε̂=x_t':>8} {'ε̂ via moyenne':>15}")
    print("-" * 62)

    totals = {}
    for b in range(4):
        lo = b * steps // 4
        hi = max((b + 1) * steps // 4, lo + 1)
        ts = [lo, lo + (hi - lo) // 2]
        errs = {"zero": [], "copy": [], "mean": []}
        for t in ts:
            a = ab[t]
            s, n = np.sqrt(a), np.sqrt(1.0 - a)
            eps = rng.standard_normal(test.shape)
            xt = s * test + n * eps
            errs["zero"].append(float(((0.0 - eps) ** 2).mean()))
            errs["copy"].append(float(((xt - eps) ** 2).mean()))
            eps_mean = (xt - s * mean_img) / n
            errs["mean"].append(float(((eps_mean - eps) ** 2).mean()))
        row = {k: float(np.mean(v)) for k, v in errs.items()}
        totals[f"t[{lo}-{hi})"] = row
        print(f"{f't[{lo}-{hi})':>12} {str(ts):>12} | {row['zero']:8.4f} "
              f"{row['copy']:8.4f} {row['mean']:15.4f}")

    print("\nÀ comparer aux modèles entraînés (loss finale à 10 000 pas) :")
    obs = {
        "baseline": [0.6103, 0.0644, 0.0219, 0.0142],
        "L (grand)": [0.6211, 0.0696, 0.0276, 0.0209],
    }
    names = list(totals)
    print(f"{'modèle':>12} " + " ".join(f"{n:>12}" for n in names))
    for label, vals in obs.items():
        print(f"{label:>12} " + " ".join(f"{v:12.4f}" for v in vals))
    for triv, key in (("ε̂=x_t", "copy"), ("ε̂ moyenne", "mean")):
        print(f"{triv:>12} " + " ".join(f"{totals[n][key]:12.4f}" for n in names))

    # --- le rapport qui explique tout -------------------------------------
    # Dans la cible ε, la part qui porte le CONTENU de l'image vaut
    # √ᾱ·x₀/√(1-ᾱ) : c'est tout ce qu'un modèle peut extraire à ce t. Si son
    # erreur dépasse cette amplitude, sa sortie ne contient aucune information
    # exploitable sur l'image — et à `t` élevé, c'est là que le contenu global
    # de l'échantillon se décide (la génération part de t=255).
    print("\nSignal de contenu présent dans ε, vs erreur des modèles :")
    print(f"{'tranche':>12} {'MSE signal':>12} {'RMS signal':>11} | "
          + " ".join(f"{k:>22}" for k in obs))
    ex2 = float((test**2).mean())
    for b, name in enumerate(names):
        lo = b * steps // 4
        hi = max((b + 1) * steps // 4, lo + 1)
        ts = [lo, lo + (hi - lo) // 2]
        sig = float(np.mean([ab[t] / (1.0 - ab[t]) * ex2 for t in ts]))
        cells = []
        for label, vals in obs.items():
            ratio = np.sqrt(vals[b] / sig)
            cells.append(f"erreur = {ratio:6.2f}x signal")
        print(f"{name:>12} {sig:12.5f} {np.sqrt(sig):11.4f} | " + " ".join(f"{c:>22}" for c in cells))
    print("\n  (>1 = le modèle n'apporte aucune information exploitable sur le contenu)")


if __name__ == "__main__":
    main()
