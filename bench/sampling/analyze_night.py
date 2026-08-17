#!/usr/bin/env python3
"""Agrège le balayage night.ckpt et classe par distance au VECTEUR COMPLET des
stats du dataset. Contrôle de non-vacuité : du bruit blanc pur et un aplat gris
doivent tous deux obtenir un MAUVAIS score — sinon la métrique est truquable."""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = "/Users/bat/development/lab/batLab/worktrees/sampling"
sys.path.insert(0, os.path.join(REPO, "tools"))
import batraw  # noqa: E402
from sample_diversity import stats as diversity_stats  # noqa: E402

DS = "/Users/bat/development/lab/batLab/datasets/elephants256.batraw"
SWEEP = sys.argv[1]

def grad_energy(imgs):
    dh = np.diff(imgs, axis=-2); dv = np.diff(imgs, axis=-3)
    return float(np.sqrt((dh**2).mean() + (dv**2).mean()))

def lap_var(imgs):
    lum = imgs.mean(axis=-1)
    lap = (-4*lum[:,1:-1,1:-1] + lum[:,:-2,1:-1] + lum[:,2:,1:-1]
           + lum[:,1:-1,:-2] + lum[:,1:-1,2:])
    return float(lap.reshape(lap.shape[0], -1).var(axis=1).mean())

def measure(imgs):
    d = diversity_stats(imgs, "x")
    return {
        "intra_image_std": d["intra_image_std"],
        "grad_energy": grad_energy(imgs),
        "laplacian_var": lap_var(imgs),
        "saturation": d.get("chroma_std", 0.0),
        "pixel_mean": d["pixel_mean"],
        "banding_ratio": d["banding_ratio"],
        "inter_seed_std": d["inter_seed_std"],
    }

def load_pngs(paths):
    return np.stack([np.asarray(Image.open(p).convert("RGB"), np.float64)/255.0 for p in paths])

payload, count, w, h, c, _ = batraw.read(DS)
ds_imgs = np.frombuffer(payload, np.uint8).reshape(count, h, w, c).astype(np.float64)/255.0
TARGET = measure(ds_imgs)

# Le VECTEUR COMPLET qui définit la distance (erreur relative moyenne).
KEYS = ["intra_image_std", "grad_energy", "laplacian_var", "saturation",
        "pixel_mean", "banding_ratio"]

def distance(row):
    return float(np.mean([abs(row[k]-TARGET[k])/TARGET[k] for k in KEYS]))

# --- contrôle de non-vacuité : deux extrêmes doivent mal scorer ---
rng = np.random.default_rng(0)
white = rng.random((6, 32, 32, 3))                       # bruit blanc uniforme
gray = np.full((6, 32, 32, 3), TARGET["pixel_mean"])     # aplat à la bonne moyenne
CONTROLS = {
    "WHITE-NOISE (doit mal scorer)": measure(white),
    "FLAT-GRAY (doit mal scorer)": measure(gray),
}

SEEDS = [101,202,303,404,505,606]
MAGS = ["0.3","0.5","0.7","0.85","1.0","1.2","1.5"]
VARS = ["beta","posterior"]
WEIGHTS = ["ema","raw"]

rows = []
for wt in WEIGHTS:
    for v in VARS:
        for m in MAGS:
            files = [f"{SWEEP}/m{m}_{v}_{wt}_p1_s{s}.png" for s in SEEDS]
            files = [f for f in files if os.path.exists(f)]
            if len(files) < 6:
                continue
            row = measure(load_pngs(files))
            row.update({"mag": m, "var": v, "w": wt, "dist": distance(row)})
            rows.append(row)

rows.sort(key=lambda r: r["dist"])

def fmt(tag, r):
    return (f"{tag:<26} dist={r['dist']:.3f}  intra={r['intra_image_std']:.3f} "
            f"grad={r['grad_energy']:.3f} lap={r['laplacian_var']:.4f} "
            f"sat={r['saturation']:.3f} mean={r['pixel_mean']:.3f} "
            f"band={r['banding_ratio']:.2f}")

print("=== TARGET (256 vraies images) ===")
print(fmt("dataset", {**TARGET, "dist": distance(TARGET)}))
print("\n=== CONTRÔLE DE NON-VACUITÉ ===")
for name, r in CONTROLS.items():
    print(fmt(name, {**r, "dist": distance(r)}))
worst_real = max((r["dist"] for r in rows), default=0)
best_ctrl = min(distance(r) for r in CONTROLS.values())
print(f"  -> pire config réelle dist={worst_real:.3f} ; meilleur contrôle dist={best_ctrl:.3f} ; "
      f"{'OK métrique non-vacue' if best_ctrl > worst_real else 'ATTENTION un contrôle bat une config'}")

print(f"\n=== {len(rows)} configs (paths=1), classées ===")
for r in rows:
    print(fmt(f"m{r['mag']} {r['var']} {r['w']}", r))

with open(f"{SWEEP}/results.json", "w") as fh:
    json.dump({"target": TARGET, "controls": {k: v for k, v in CONTROLS.items()},
               "rows": rows}, fh, indent=2)
print(f"\nwrote {SWEEP}/results.json")
