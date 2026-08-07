#!/usr/bin/env python3
"""P3 — déterminisme et round-trip observable du checkpoint.

Spec : « `--headless-sample` sur ce checkpoint, deux fois même seed → PNG octet
pour octet identiques (déterminisme) ; et si tu peux charger/resauver via une
commande, le second sample reste identique. »

Le binaire n'offre pas de commande « charger puis resauver ». Le round-trip est
donc fait à travers le conteneur PUBLIÉ, avec le codec écrit en boîte noire
(`ckpt.py`) : on relit les 14,3 Mo du checkpoint, on les réécrit octet par
octet, et on vérifie que la génération ne bouge pas. Une variante réécrit le
tenseur d'attention par le chemin (4, C, C) → aplati : elle prouve que les
quatre projections traversent l'aller-retour dans le bon ordre et sans décalage.
"""
import hashlib
import os
import subprocess
import sys

import numpy as np

import ckpt as ckptio
import harness as H

XL = "Color_Diffusion_XL"
CKPT = os.path.join(H.WORK, "Models", XL, "pretrained_weights", "night_run.ckpt")
OUT = os.path.join(H.WORK, "out", "rt")
results = []


def sha_sample(ckpt, tag, seed=4, paths=2, magnitude=1.0):
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, tag + ".png")
    r = subprocess.run([H.BIN, "--headless-sample", XL, "--checkpoint", ckpt,
                        "--seed", str(seed), "--paths", str(paths),
                        "--magnitude", str(magnitude), "--out", png],
                       env=dict(os.environ, BATLAB_ROOT=H.WORK), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return hashlib.sha256(open(png, "rb").read()).hexdigest()


if __name__ == "__main__":
    a = sha_sample(CKPT, "run_a")
    b = sha_sample(CKPT, "run_b")
    results.append(("même graine deux fois ⇒ PNG octet pour octet identiques",
                    a == b, f"sha256 {a[:16]} / {b[:16]}"))

    seeds = [sha_sample(CKPT, f"seed{s}", seed=s) for s in (4, 5)]
    results.append(("graines différentes ⇒ PNG différents (le test précédent a du mordant)",
                    seeds[0] != seeds[1], f"{seeds[0][:16]} vs {seeds[1][:16]}"))

    c = ckptio.read(CKPT)
    rt = os.path.join(OUT, "roundtrip.ckpt")
    ckptio.write(rt, c["records"])          # sans l'état d'optimiseur, non utilisé en inférence
    results.append(("round-trip du conteneur ⇒ même génération",
                    sha_sample(rt, "run_rt") == a, "relu puis réécrit par ckpt.py"))

    rec = [r for r in c["records"] if r["index"] == 14][0]
    W = rec["w"].reshape(4, 192, 192)
    B = rec["b"].reshape(4, 192)
    repack = [r if r["index"] != 14 else
              {"index": 14, "w": np.concatenate([W[i].ravel() for i in range(4)]),
               "b": np.concatenate([B[i] for i in range(4)])} for r in c["records"]]
    rt2 = os.path.join(OUT, "roundtrip_attn.ckpt")
    ckptio.write(rt2, repack)
    results.append(("les 4 projections survivent au dépaquetage/repaquetage (4, C, C)",
                    sha_sample(rt2, "run_rt2") == a, "W_o reste le bloc 3"))

    # contrôle : permuter deux projections DOIT changer la génération
    Wp = W.copy(); Wp[[0, 1]] = Wp[[1, 0]]
    perm = [r if r["index"] != 14 else
            {"index": 14, "w": Wp.ravel(), "b": rec["b"]} for r in c["records"]]
    rt3 = os.path.join(OUT, "perm.ckpt")
    ckptio.write(rt3, perm)
    results.append(("contrôle : Q et K permutés ⇒ génération différente",
                    sha_sample(rt3, "run_perm") != a, "sinon l'aller-retour ne prouverait rien"))

    n = 0
    for label, ok, detail in results:
        print(("PASS  " if ok else "FAIL  ") + label + "\n        " + detail)
        n += not ok
    sys.exit(1 if n else 0)
