#!/usr/bin/env python3
"""La grille de dispatch au-delà de 65 535 workgroups — franchie de l'extérieur.

`docs/reports/ATTENTION.md` §4.6 : les passes de lignes sont dispatchées « un
workgroup par (échantillon, ligne) », donc elles franchissent la limite de
65 535 quand `seq · batch > 65 535` ; le premier repli tenté était une COURSE
(la softmax fait du read-modify-write sur la ligne).

Au goulot 8×8 (seq = 64), la frontière tombe à batch 1 024. Le seul chemin
batché observable est l'entraînement, et le seul témoin de la couche y est W_o
après un pas (cf. t5_batch.py). Deux propriétés y sont testables de part et
d'autre de la frontière :

  * DÉTERMINISME — une course entre deux workgroups sur la même ligne donnerait
    un résultat qui bouge d'un run à l'autre ;
  * ADDITIVITÉ par slot — une ligne corrompue casserait la décomposition.
"""
import os
import subprocess
import sys

import numpy as np

import batraw
import ckpt as ckptio
import harness as H

MODEL = "Probe_8x8x3"
DIR = os.path.join(H.WORK, "grid")
results = []


def wo(batch, tag, slot_overrides=()):
    os.makedirs(DIR, exist_ok=True)
    rng = np.random.default_rng(3)                    # même fond pour tous les datasets
    data = np.clip(rng.normal(0, 0.5, (batch, 8, 8, 3)), -1, 1)
    for slot, val in slot_overrides:
        data[slot] = val
    ds = os.path.join(DIR, tag + ".batraw")
    batraw.write(ds, data)
    out = os.path.join(DIR, tag + ".ckpt")
    r = subprocess.run([H.BIN, "--headless-train", MODEL, "--steps", "0", "--lr", "1e-3",
                        "--batch", str(batch), "--dataset", ds, "--out", out],
                       env=dict(os.environ, BATLAB_ROOT=H.WORK), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = [x for x in ckptio.read(out)["records"] if x["index"] == 1][0]
    return rec["w"].reshape(4, 3, 3)[3]


if __name__ == "__main__":
    H.ensure_model(MODEL, 8, 8, 3, with_attention=True)
    HI, LO = np.full((8, 8, 3), 0.9), np.full((8, 8, 3), -0.9)

    for batch in (1023, 1024, 1025):
        wg = batch * 64
        a = wo(batch, f"det_{batch}_a")
        b = wo(batch, f"det_{batch}_b")
        results.append((f"batch {batch} — {wg} workgroups par passe de ligne : déterministe",
                        np.array_equal(a, b) and np.isfinite(a).all(),
                        "%s la limite de 65 535 ; |W_o|max=%.2e ; bit à bit=%s"
                        % ("au-delà de" if wg > 65535 else "sous", np.abs(a).max(),
                           np.array_equal(a, b))))

    batch = 1024
    last = batch - 1
    f = {}
    for i, s0 in enumerate((HI, LO)):
        for j, s1 in enumerate((HI, LO)):
            f[(i, j)] = wo(batch, f"add_{i}{j}", [(0, s0), (last, s1)])
    scale = max(np.abs(v).max() for v in f.values())
    resid = np.abs(f[(0, 0)] + f[(1, 1)] - f[(0, 1)] - f[(1, 0)]).max() / scale
    sig_first = np.abs(f[(0, 0)] - f[(1, 0)]).max() / scale
    sig_last = np.abs(f[(0, 0)] - f[(0, 1)]).max() / scale
    # à batch 1 024, un slot ne pèse qu'un millième du gradient : le seuil est
    # relatif au signal, pas absolu — le résidu doit rester deux ordres de
    # grandeur SOUS l'effet mesuré du plus discret des deux slots.
    weakest = min(sig_first, sig_last)
    results.append(("batch 1024 (65 536 workgroups) : le premier ET le dernier "
                    "échantillon comptent, et sans terme croisé",
                    weakest > 0 and resid < weakest / 100,
                    "résidu relatif %.2e ; sensibilité slot 0 = %.2e, slot %d = %.2e "
                    "(rapport signal/résidu %.0f×)"
                    % (resid, sig_first, last, sig_last, weakest / resid)))

    n = 0
    for label, ok, detail in results:
        print(("PASS  " if ok else "FAIL  ") + label + "\n        " + detail)
        n += not ok
    sys.exit(1 if n else 0)
