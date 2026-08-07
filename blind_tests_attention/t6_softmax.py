#!/usr/bin/env python3
"""Stabilité numérique de la softmax — propriété adjacente, testée indépendamment.

`docs/reports/ATTENTION.md` §4.5 range « softmax sans soustraction du max » parmi
les 16 mutations attrapées, par « débordement exp ». On le vérifie de l'extérieur :
on pousse les logits `qᵀk/√d` bien au-delà de ln(f32_max) ≈ 88, là où un
`exp(score)` non recentré rend `inf` puis `inf/inf = NaN`.

La référence f64 (recentrée par le max) reste finie ; la sortie observée doit la
suivre. Deux régimes : ~10² (le premier où f32 déborde) et ~10⁴.
"""
import sys

import numpy as np

import harness as H
import reference as R

results = []
TOL = 3e-5


def run(gain_qk, tag, w=8, h=8, c=3, seed=21):
    name = H.ensure_model(f"Probe_{w}x{h}x{c}", w, h, c, with_attention=True)
    base = H.ensure_model(f"Probe_{w}x{h}x{c}_noattn", w, h, c, with_attention=False)
    a, b, t = H.calibrate(base, c)
    g = np.float32(a / b)
    rng = np.random.default_rng(seed)
    Wq = rng.normal(0, 0.6, (c, c)) * gain_qk
    Wk = rng.normal(0, 0.6, (c, c)) * gain_qk
    Wv = rng.normal(0, 0.6, (c, c))
    Wo = rng.normal(0, 0.10, (c, c))
    z = np.zeros(c)
    P = (Wq, Wk, Wv, Wo, z, z, z, z)
    wpk, bpk = R.pack(*P)
    fr = H.perpetual(name, [{"index": 0, "w": g * H.identity_conv(c), "b": np.zeros(c)},
                            {"index": 1, "w": wpk, "b": bpk}], tag, magnitude=1.0)
    eps, mask, x_pre = H.eps_hat(fr, a, b)
    x = float(g) * x_pre
    xs = x.reshape(-1, c)
    ref = R.attention(xs, *P).reshape(x.shape)
    q, k = xs @ Wq.T, xs @ Wk.T
    logit = float(np.abs(q @ k.T / np.sqrt(c)).max())
    finite = bool(np.isfinite(eps[mask]).all())
    rel = float((np.abs(eps - ref)[mask] / np.maximum(np.abs(ref)[mask], 0.05)).max())
    ok = finite and rel < TOL and logit > 88
    results.append((f"logits |qᵀk/√d| jusqu'à {logit:.3g} — pas de débordement de exp", ok,
                    "sortie finie=%s ; écart relatif à la référence f64 = %.2e ; "
                    "n=%d/%d" % (finite, rel, mask.sum(), mask.size)))


if __name__ == "__main__":
    run(8.0, "sm_1e2")
    run(80.0, "sm_1e4")
    n = 0
    for label, ok, detail in results:
        print(("PASS  " if ok else "FAIL  ") + label + "\n        " + detail)
        n += not ok
    sys.exit(1 if n else 0)
