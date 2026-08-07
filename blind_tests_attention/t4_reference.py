#!/usr/bin/env python3
"""P4 — référence f64 indépendante de la couche d'attention.

Compare la sortie brute du réseau, lue en boîte noire (`harness.eps_hat`), à
`reference.attention`, écrite depuis la seule formule de la spec.

Le modèle sonde est `[Convolution 1×1 (gain g · identité), Attention]`. Le gain
vaut `g = a/b` : le pas inverse calcule `x0_hat = a·x_pre − b·ε̂`, donc avec
`ε̂ ≈ g·x_pre + δ` les deux termes en `x_pre` s'annulent et TOUTE la fenêtre non
écrêtée de `x0_hat` est disponible pour `δ`, la contribution de l'attention.
C'est ce qui permet de faire tourner la couche sur une entrée d'amplitude ~1 —
donc une softmax franchement non uniforme — sans que le sampler écrête.

Trois formes, dont deux au-dessus des 64 threads d'un workgroup :
8×8 = 64 positions, 9×8 = 72, 16×16 = 256.
"""
import sys

import numpy as np

import harness as H
import reference as R

TOL = 3e-5          # tolérance relative : la lecture de ε̂ transite par des f32
results = []


def probe_weights(c, seed, scale_o=0.10):
    rng = np.random.default_rng(seed)
    Wq, Wk, Wv = (rng.normal(0, 0.6, (c, c)) for _ in range(3))
    Wo = rng.normal(0, scale_o, (c, c))
    bq, bk, bv = (rng.normal(0, 0.2, c) for _ in range(3))
    bo = rng.normal(0, 0.02, c)
    return Wq, Wk, Wv, Wo, bq, bk, bv, bo


def observe(w, h, c, seed, tag, scale_o=0.10):
    """Lance la sonde et rend (ε̂ observé, masque, entrée x de l'attention, poids)."""
    name = H.ensure_model(f"Probe_{w}x{h}x{c}", w, h, c, with_attention=True)
    base = H.ensure_model(f"Probe_{w}x{h}x{c}_noattn", w, h, c, with_attention=False)
    a, b, t = H.calibrate(base, c, seed=seed)
    g = np.float32(a / b)                       # gain d'annulation, arrondi comme le moteur
    P = probe_weights(c, seed, scale_o)
    wpk, bpk = R.pack(*P)
    fr = H.perpetual(name, [{"index": 0, "w": g * H.identity_conv(c), "b": np.zeros(c)},
                            {"index": 1, "w": wpk, "b": bpk}],
                     tag, seed=seed, magnitude=1.0)
    eps, mask, x_pre = H.eps_hat(fr, a, b)
    x = float(g) * x_pre                        # entrée de l'attention
    return eps, mask, x, P, t


def one(w, h, c, seed):
    eps, mask, x, P, t = observe(w, h, c, seed, f"ref_{w}x{h}x{c}")
    xs = x.reshape(-1, c)
    ref = R.attention(xs, *P).reshape(x.shape)
    probs = _probs(xs, *P)
    n = int(mask.sum())
    rel = float((np.abs(eps - ref)[mask] / np.maximum(np.abs(ref)[mask], 0.05)).max())
    moved = float(np.abs(ref - x)[mask].max())
    ok = rel < TOL and n > 0.5 * mask.size and moved > 1e-3
    results.append((f"{w}×{h}×{c} — N={w*h} positions, C={c}", ok,
                    f"t={t} n={n}/{mask.size} rel_max={rel:.2e} |y-x|max={moved:.3f} "
                    f"softmax p_max/p_min={probs.max()/probs.min():.1f} (uniforme=1)"))


def _probs(xs, Wq, Wk, Wv, Wo, bq, bk, bv, bo):
    q, k = xs @ Wq.T + bq, xs @ Wk.T + bk
    return R.softmax(q @ k.T / np.sqrt(xs.shape[1]), axis=1)


def sensitivity(w, h, c, seed):
    """La référence doit REFUSER les variantes fautives de la même formule."""
    eps, mask, x, P, t = observe(w, h, c, seed, f"mut_{w}x{h}x{c}")
    xs = x.reshape(-1, c)
    Wq, Wk, Wv, Wo, bq, bk, bv, bo = P

    def rel(y):
        y = y.reshape(x.shape)
        return float((np.abs(eps - y)[mask] / np.maximum(np.abs(y)[mask], 0.05)).max())

    good = R.attention(xs, *P)
    variants = {
        "Q/K échangés": R.attention(xs, Wk, Wq, Wv, Wo, bk, bq, bv, bo),
        "sans 1/√d": _variant(xs, P, scale=1.0),
        "softmax sur le mauvais axe": _variant(xs, P, axis=0),
        "sans résiduel": good - xs,
        "sans b_o": good - bo,
        "W_o transposé": R.attention(xs, Wq, Wk, Wv, Wo.T, bq, bk, bv, bo),
        "W_v transposé": R.attention(xs, Wq, Wk, Wv.T, Wo, bq, bk, bv, bo),
        # la seule dégénérescence plausible : échanger les rôles Q/K ET l'axe de
        # la softmax. Si elle passait, l'ordre d'empaquetage Q,K resterait
        # indécidable de l'extérieur — elle ne passe pas.
        "Q/K échangés + axe inversé": _variant(xs, (Wk, Wq, Wv, Wo, bk, bq, bv, bo), axis=0),
    }
    ok = rel(good) < TOL and all(rel(v) > 100 * TOL for v in variants.values())
    detail = f"correct={rel(good):.1e} | " + "  ".join(f"{k}={rel(v):.1e}"
                                                       for k, v in variants.items())
    results.append((f"sensibilité de la référence ({w}×{h})", ok, detail))


def _variant(xs, P, scale=None, axis=1):
    Wq, Wk, Wv, Wo, bq, bk, bv, bo = P
    d = xs.shape[1]
    q, k, v = xs @ Wq.T + bq, xs @ Wk.T + bk, xs @ Wv.T + bv
    s = q @ k.T / (np.sqrt(d) if scale is None else scale)
    return xs + R.softmax(s, axis=axis) @ v @ Wo.T + bo


if __name__ == "__main__":
    one(8, 8, 3, seed=7)
    one(9, 8, 3, seed=11)
    one(16, 16, 3, seed=13)
    one(8, 8, 8, seed=17)      # C = 8 : le facteur 1/√d suit bien d = C
    sensitivity(8, 8, 3, seed=7)
    bad = 0
    for label, ok, detail in results:
        print(("PASS  " if ok else "FAIL  ") + label + "\n        " + detail)
        bad += not ok
    sys.exit(1 if bad else 0)
