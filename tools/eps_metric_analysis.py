#!/usr/bin/env python3
"""Ce que la loss ε mesure vraiment par tranche de t (facteur d'amplification).

L'identité exacte, sans approximation : si un modèle reconstruit le signal
propre avec une erreur `x̂₀ - x₀`, alors comme

    ε   = (x_t - √ᾱ·x₀) / √(1-ᾱ)      et      ε̂ = (x_t - √ᾱ·x̂₀) / √(1-ᾱ)
    ε̂ - ε = √ᾱ·(x₀ - x̂₀) / √(1-ᾱ)

la loss rapportée vaut

    MSE(ε̂, ε) = [ ᾱ / (1-ᾱ) ] · MSE(x̂₀, x₀)

Le facteur `ᾱ/(1-ᾱ)` explose quand t → 0 (le bruit devient invisible). Une même
qualité de reconstruction produit donc des loss ε radicalement différentes selon
`t` : la tranche basse n'est pas « plus dure à apprendre », elle est
**mesurée avec une loupe**.

Le probe (metrics.rs:213-221) tire `t_per_bucket` valeurs par tranche, à
`t = t_lo + k·span/t_per_bucket` — pour la tranche 0 avec 2 tirages : t=0 et
t=32. La moyenne de la tranche est donc dominée par le tirage t=0.

Usage : python3 tools/eps_metric_analysis.py [--losses 0.6103 0.0644 ...]
"""
import argparse

import numpy as np

BETA_REFERENCE_STEPS = 1000.0
MAX_BETA = 0.999


def schedule(num_steps=256, beta_start=1e-4, beta_end=2e-2):
    scale = BETA_REFERENCE_STEPS / num_steps
    s0, s1 = min(beta_start * scale, MAX_BETA), min(beta_end * scale, MAX_BETA)
    t = np.arange(num_steps) / max(num_steps - 1, 1)
    betas = np.minimum(s0 + (s1 - s0) * t, MAX_BETA)
    return np.cumprod(1.0 - betas, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, default=4)
    ap.add_argument("--t-per-bucket", type=int, default=2)
    ap.add_argument("--losses", type=float, nargs="*",
                    help="loss observée par tranche, pour la convertir en RMSE sur x0")
    ap.add_argument("--label", default="modèle")
    args = ap.parse_args()

    ab = schedule()
    steps = len(ab)
    amp = ab / (1.0 - ab)  # facteur d'amplification par t

    print("Facteur d'amplification  MSE(ε) / MSE(x₀)  =  ᾱ/(1-ᾱ)\n")
    print(f"{'t':>4} {'ᾱ':>10} {'√(1-ᾱ)':>10} {'amplification':>14}")
    for t in (0, 8, 16, 32, 64, 96, 128, 192, 255):
        print(f"{t:>4} {ab[t]:10.6f} {np.sqrt(1-ab[t]):10.4f} {amp[t]:14.1f}")

    print(f"\nTirages du probe ({args.buckets} tranches, {args.t_per_bucket} t/tranche) :")
    names, draws = [], []
    for b in range(args.buckets):
        lo = b * steps // args.buckets
        hi = max((b + 1) * steps // args.buckets, lo + 1)
        span = hi - lo
        ts = [lo + (k * span) // args.t_per_bucket for k in range(args.t_per_bucket)]
        names.append(f"t[{lo}-{hi})")
        draws.append(ts)
        mean_amp = float(np.mean([amp[t] for t in ts]))
        print(f"  {names[b]:<12} t = {ts}   amplification moyenne = {mean_amp:10.1f}"
              f"   (dont t={ts[0]} : {amp[ts[0]]:.1f})")

    if args.losses:
        print(f"\nConversion des loss observées ({args.label}) en précision sur x₀ :")
        print(f"{'tranche':>12} {'loss ε':>9} {'RMSE(x₀) implicite':>20} {'% de la plage [-1,1]':>21}")
        for name, ts, loss in zip(names, draws, args.losses):
            mean_amp = float(np.mean([amp[t] for t in ts]))
            rmse = float(np.sqrt(loss / mean_amp))
            print(f"{name:>12} {loss:9.4f} {rmse:20.4f} {100*rmse/2:20.2f}%")


if __name__ == "__main__":
    main()
