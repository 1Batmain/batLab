#!/usr/bin/env python3
"""P5 — anti-fuite inter-échantillons, dans les limites de ce qui est observable.

Spec : « si tu peux exercer un batch > 1 par un chemin observable, vérifie qu'un
échantillon n'influence pas la sortie d'un autre. Sinon, consigne que ce n'est
pas observable de l'extérieur. »

Ce qui a été cherché, et ce qui a été trouvé :

  * `--headless-sample --paths N` N'EST PAS un axe de batch : le temps mural est
    linéaire en N (0,83 / 3,15 / 12,2 / 55,2 s pour N = 1 / 4 / 16 / 64 sur
    Color_Diffusion_XL) — les chemins sont déroulés en séquence.
  * `--headless-perpetual` travaille sur un seul latent.
  * Le seul axe de batch atteignable de l'extérieur est `--headless-train
    --batch B`. Mais cet entrée-là force l'init from scratch (elle ne charge
    jamais de poids) : au pas 0, W_o = 0, donc la couche est l'identité et sa
    sortie ne peut rien contaminer en aval.

Reste une prise, et elle vise exactement le risque : avec W_o = 0, le gradient
de W_o vaut `Σ_n (∂L/∂y_n) ⊗ ctx_n` — et `ctx` EST le tampon où un échantillon
lirait les k/v d'un autre. Un pas d'optimiseur écrit ce gradient dans le
checkpoint, donc `ctx` devient observable.

Test : le gradient batché doit se DÉCOMPOSER par slot. Avec `f(u,v)` = W_o après
un pas sur le batch (slot0=u, slot1=v), l'absence de terme croisé impose

        f(x,y) + f(z,w) == f(x,w) + f(z,y)          (à tirage de t/ε fixé par slot)

Cette forme est immunisée au fait — mesuré ici — que chaque slot tire SON t et
SON bruit : ceux-ci ne dépendent que de l'indice de slot, pas du contenu.
"""
import os
import subprocess
import sys

import numpy as np

import batraw
import ckpt as ckptio
import harness as H

results = []
MODEL = "Probe_8x8x3"
DIR = os.path.join(H.WORK, "batch")


def wo(pair, tag, lr=1e-3):
    os.makedirs(DIR, exist_ok=True)
    ds = os.path.join(DIR, tag + ".batraw")
    batraw.write(ds, np.stack(pair))
    out = os.path.join(DIR, tag + ".ckpt")
    r = subprocess.run([H.BIN, "--headless-train", MODEL, "--steps", "0", "--lr", str(lr),
                        "--batch", str(len(pair)), "--dataset", ds, "--out", out],
                       env=dict(os.environ, BATLAB_ROOT=H.WORK), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = [x for x in ckptio.read(out)["records"] if x["index"] == 1][0]
    return rec["w"].reshape(4, 3, 3), rec["b"].reshape(4, 3)


if __name__ == "__main__":
    H.ensure_model(MODEL, 8, 8, 3, with_attention=True)
    yy, xx = np.mgrid[0:8, 0:8]
    # Deux paires très contrastées : le test n'a de valeur que si CHAQUE slot
    # pèse visiblement sur W_o. Des motifs voisins (deux rampes) rendraient la
    # sensibilité nulle et l'additivité vraie pour de mauvaises raisons.
    P = {
        "x": np.stack([((xx + yy) % 2) * 1.8 - 0.9] * 3, -1),           # damier
        "z": np.full((8, 8, 3), 0.9),                                   # constante haute
        "y": np.full((8, 8, 3), -0.9),                                  # constante basse
        "w": np.stack([xx / 3.5 - 1.0] * 3, -1),                        # rampe
    }

    f = {}
    for a in "xz":
        for b in "yw":
            f[a + b] = wo([P[a], P[b]], "f_" + a + b)[0][3]
    # ∂L/∂Q = ∂L/∂K = ∂L/∂V = 0 tant que W_o = 0 : conséquence exacte de la
    # formule (tout le chemin passe par W_o). Un pas ne doit donc bouger que W_o.
    Wf, Bf = wo([P["x"], P["y"]], "f_xy")
    W0, B0 = wo([P["x"], P["y"]], "f_xy_lr0", lr=0)
    results.append(("un pas d'optimiseur n'écrit QUE W_o (∂L/∂Q,K,V = 0 quand W_o = 0)",
                    all(np.array_equal(Wf[i], W0[i]) and np.array_equal(Bf[i], B0[i])
                        for i in range(3)) and np.abs(Wf[3]).max() > 0,
                    "Q/K/V inchangés bit à bit depuis l'init ; |W_o|max=%.2e "
                    "(nul avant le pas : %s)" % (np.abs(Wf[3]).max(), (W0[3] == 0).all())))

    same = wo([P["x"], P["y"]], "f_xy_bis")[0][3]
    results.append(("le pas est déterministe (même dataset ⇒ même W_o)",
                    np.array_equal(same, f["xy"]), "bit à bit"))

    scale = max(np.abs(v).max() for v in f.values())
    # amplitude du signal : ce que change le contenu d'un slot
    sig1 = np.abs(f["xy"] - f["xw"]).max() / scale     # slot 1 : y → w
    sig0 = np.abs(f["xy"] - f["zy"]).max() / scale     # slot 0 : x → z
    resid = np.abs(f["xy"] + f["zw"] - f["xw"] - f["zy"]).max() / scale
    results.append(("gradient batché sans terme croisé : f(x,y)+f(z,w) == f(x,w)+f(z,y)",
                    resid < 1e-5 and min(sig0, sig1) > 1e-2,
                    "résidu relatif %.2e ; sensibilité slot0=%.3f slot1=%.3f "
                    "(rapport signal/résidu %.0f×)" % (resid, sig0, sig1, min(sig0, sig1) / resid)))

    order = wo([P["y"], P["x"]], "f_yx")[0][3]
    results.append(("chaque slot tire son propre t/ε (f(x,y) ≠ f(y,x)) — la forme "
                    "additive du test est donc la bonne",
                    not np.allclose(order, f["xy"], rtol=0, atol=1e-4 * scale),
                    "écart relatif %.3f" % (np.abs(order - f["xy"]).max() / scale)))

    # batch 4 : carré latin sur deux slots, les deux autres tenus fixes
    g = {}
    for a in "xz":
        for b in "yw":
            g[a + b] = wo([P[a], P[b], P["x"], P["w"]], "g_" + a + b)[0][3]
    s4 = max(np.abs(v).max() for v in g.values())
    r4 = np.abs(g["xy"] + g["zw"] - g["xw"] - g["zy"]).max() / s4
    v4 = np.abs(g["xy"] - g["xw"]).max() / s4
    results.append(("idem à batch 4", r4 < 1e-5 and v4 > 1e-2,
                    "résidu relatif %.2e ; sensibilité %.3f" % (r4, v4)))

    n = 0
    for label, ok, detail in results:
        print(("PASS  " if ok else "FAIL  ") + label + "\n        " + detail)
        n += not ok
    print("\nNOTE — couverture partielle : ce chemin n'observe la fuite que dans le "
          "tampon `ctx` du FORWARD, à W_o = 0. Une fuite dans les passes BACKWARD "
          "de q/k/v reste hors de portée : leurs gradients sont exactement nuls tant "
          "que W_o = 0, et aucune entrée headless ne sait charger des poids entraînés "
          "dans un entraînement batché.")
    sys.exit(1 if n else 0)
