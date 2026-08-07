#!/usr/bin/env python3
"""P1 — la couche d'attention fraîchement construite est l'IDENTITÉ.

Spec : « `W_o` est initialisé à zéro → une couche d'attention fraîchement
construite est l'IDENTITÉ (sortie == entrée). Observable : un modèle contenant
une attention non entraînée doit produire exactement les mêmes échantillons que
le même modèle sans la couche d'attention (à poids Q/K/V identiques). »

Trois niveaux :
  a) structure — dans le checkpoint d'un modèle à 0 pas, le bloc W_o (poids ET
     biais) est EXACTEMENT nul et Q/K/V ne le sont pas. Une couche toute à zéro
     serait « identité » et morte : les deux moitiés comptent.
  b) sortie de couche — la sonde ε̂ (cf. harness.py) rend la sortie brute du
     réseau ; avec W_o=0 elle doit être l'entrée, à la précision f32 de la lecture.
  c) génération — `Color_Diffusion_XL` (30 couches) contre le MÊME modèle privé
     de sa seule couche Attention (29 couches, les deux GroupNorm conservées).
     Les checkpoints sont écrits par nous : tous les poids partagés sont
     bit-à-bit identiques, il n'y a donc aucun décalage de tirage d'init à
     confondre avec l'effet de la couche. Les PNG doivent être identiques
     OCTET POUR OCTET.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

import batraw
import ckpt as ckptio
import harness as H
import reference as R

results = []
XL = "Color_Diffusion_XL"
ATTN_IDX = 14                      # indice de la couche Attention dans la config XL
OUT = os.path.join(H.WORK, "out")


def add(label, ok, detail):
    results.append((label, ok, detail))


def fresh_checkpoint(model, out, dataset):
    """Poids d'initialisation : un « entraînement » à lr=0 ne les bouge pas."""
    r = subprocess.run([H.BIN, "--headless-train", model, "--steps", "0", "--lr", "0",
                        "--dataset", dataset, "--out", out],
                       env=dict(os.environ, BATLAB_ROOT=H.WORK),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return ckptio.read(out)


def sample_png(model, records, tag, seed=3, paths=2):
    os.makedirs(OUT, exist_ok=True)
    cp, png = os.path.join(OUT, tag + ".ckpt"), os.path.join(OUT, tag + ".png")
    ckptio.write(cp, records)
    r = subprocess.run([H.BIN, "--headless-sample", model, "--checkpoint", cp,
                        "--seed", str(seed), "--paths", str(paths), "--out", png],
                       env=dict(os.environ, BATLAB_ROOT=H.WORK),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return hashlib.sha256(open(png, "rb").read()).hexdigest()


# ---------------------------------------------------------------- a) structure
def structure():
    H.adopt_repo_model(XL)
    ds = os.path.join(H.WORK, "datasets", "tiny8.batraw")
    os.makedirs(os.path.dirname(ds), exist_ok=True)
    rng = np.random.default_rng(1234)
    batraw.write(ds, np.clip(rng.normal(0, 0.4, (64, 8, 8, 3)), -1, 1))
    H.ensure_model("Probe_8x8x3", 8, 8, 3, with_attention=True)
    c = fresh_checkpoint("Probe_8x8x3", os.path.join(H.WORK, "t1_toy.ckpt"), ds)
    _blocks("jouet 8×8×3", c, 1, 3)

    ds32 = os.path.join(H.WORK, "datasets", "rgb32.batraw")
    batraw.write(ds32, np.clip(rng.normal(0, 0.4, (32, 32, 32, 3)), -1, 1))
    c = fresh_checkpoint(XL, os.path.join(H.WORK, "t1_xl.ckpt"), ds32)
    _blocks("Color_Diffusion_XL (C=192)", c, ATTN_IDX, 192)
    return c


def _blocks(label, c, index, dim):
    rec = [r for r in c["records"] if r["index"] == index][0]
    W = rec["w"].reshape(4, dim, dim)
    B = rec["b"].reshape(4, dim)
    zero_w = [bool((W[i] == 0).all()) for i in range(4)]
    zero_b = [bool((B[i] == 0).all()) for i in range(4)]
    ok = zero_w == [False, False, False, True] and zero_b[3]
    add(f"init — un seul des 4 blocs est nul, et c'est le dernier ({label})", ok,
        "poids nuls par bloc " + str(zero_w) + " ; biais nuls " + str(zero_b) +
        " ; |Q|max=%.3f |K|max=%.3f |V|max=%.3f" % tuple(np.abs(W[i]).max() for i in range(3)))


# ------------------------------------------------------- b) sortie de la couche
def layer_output():
    for (w, h) in ((8, 8), (9, 8), (16, 16)):
        c = 3
        name = H.ensure_model(f"Probe_{w}x{h}x{c}", w, h, c, with_attention=True)
        base = H.ensure_model(f"Probe_{w}x{h}x{c}_noattn", w, h, c, with_attention=False)
        a, b, t = H.calibrate(base, c)
        rng = np.random.default_rng(5)
        P = (rng.normal(0, .6, (c, c)), rng.normal(0, .6, (c, c)), rng.normal(0, .6, (c, c)),
             np.zeros((c, c)), rng.normal(0, .2, c), rng.normal(0, .2, c),
             rng.normal(0, .2, c), np.zeros(c))
        wpk, bpk = R.pack(*P)
        g = np.float32(a / b)
        fr = H.perpetual(name, [{"index": 0, "w": g * H.identity_conv(c), "b": np.zeros(c)},
                                {"index": 1, "w": wpk, "b": bpk}],
                         f"id_{w}x{h}", magnitude=1.0)
        eps, mask, x_pre = H.eps_hat(fr, a, b)
        x = float(g) * x_pre
        err = float(np.abs(eps - x)[mask].max())
        add(f"W_o=0 ⇒ sortie == entrée, Q/K/V non nuls ({w}×{h}, N={w*h})",
            err < 3e-5 * max(1.0, np.abs(x)[mask].max()),
            "max|ε̂ − x| = %.2e sur %d positions, |x|max=%.2f" % (err, mask.sum(), np.abs(x).max()))


# ------------------------------------------------------------- c) génération XL
def generation(xl_fresh):
    cfg = json.load(open(os.path.join(H.WORK, "Models", XL, "config_file")))
    assert "Attention" in cfg["layers"][ATTN_IDX], cfg["layers"][ATTN_IDX]

    # même modèle, couche Attention RETIRÉE (les deux GroupNorm restent)
    noattn = dict(cfg, model_name=XL + "_noattn")
    noattn["layers"] = cfg["layers"][:ATTN_IDX] + cfg["layers"][ATTN_IDX + 1:]
    d = os.path.join(H.WORK, "Models", XL + "_noattn")
    os.makedirs(os.path.join(d, "pretrained_weights"), exist_ok=True)
    json.dump(noattn, open(os.path.join(d, "config_file"), "w"), indent=2)

    recs = xl_fresh["records"]
    shifted = [{"index": r["index"] - (1 if r["index"] > ATTN_IDX else 0),
                "w": r["w"], "b": r["b"]}
               for r in recs if r["index"] != ATTN_IDX]
    h_with = sample_png(XL, recs, "xl_attn_id")
    h_without = sample_png(XL + "_noattn", shifted, "xl_noattn")
    add("génération XL : avec attention neuve == sans la couche (PNG octet pour octet)",
        h_with == h_without, f"sha256 {h_with[:16]} / {h_without[:16]}")

    # Q/K/V arbitraires, W_o toujours nul : la génération ne doit pas bouger d'un octet
    rng = np.random.default_rng(9)
    attn = [r for r in recs if r["index"] == ATTN_IDX][0]
    W = attn["w"].reshape(4, 192, 192).copy()
    B = attn["b"].reshape(4, 192).copy()
    W[:3] = rng.normal(0, 0.3, (3, 192, 192)); B[:3] = rng.normal(0, 0.3, (3, 192))
    other = [r if r["index"] != ATTN_IDX else
             {"index": ATTN_IDX, "w": W.ravel(), "b": B.ravel()} for r in recs]
    h_other = sample_png(XL, other, "xl_attn_qkv")
    add("génération XL : Q/K/V arbitraires, W_o=0 ⇒ même PNG octet pour octet",
        h_other == h_with, f"sha256 {h_other[:16]}")

    # variante littérale de l'ordre de mission : GroupNorm + Attention retirées
    lit = dict(cfg, model_name=XL + "_litnoattn")
    lit["layers"] = cfg["layers"][:ATTN_IDX - 1] + cfg["layers"][ATTN_IDX + 1:]
    d = os.path.join(H.WORK, "Models", XL + "_litnoattn")
    os.makedirs(os.path.join(d, "pretrained_weights"), exist_ok=True)
    json.dump(lit, open(os.path.join(d, "config_file"), "w"), indent=2)
    lit_recs = [{"index": r["index"] - (2 if r["index"] > ATTN_IDX else 0),
                 "w": r["w"], "b": r["b"]}
                for r in recs if r["index"] not in (ATTN_IDX - 1, ATTN_IDX)]
    h_lit = sample_png(XL + "_litnoattn", lit_recs, "xl_litnoattn")
    add("variante littérale (GroupNorm+Attention retirées) — informative, non décisive",
        True, ("identique" if h_lit == h_with else
               "diffère : une GroupNorm de moins dans la chaîne, pas l'attention"))


if __name__ == "__main__":
    xl = structure()
    layer_output()
    generation(xl)
    bad = 0
    for label, ok, detail in results:
        print(("PASS  " if ok else "FAIL  ") + label + "\n        " + detail)
        bad += not ok
    sys.exit(1 if bad else 0)
