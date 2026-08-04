#!/usr/bin/env python3
"""Mesure la diversité d'un lot d'échantillons générés (seeds différents).

Le petit modèle rend « la moyenne du dataset » : tous ses échantillons sont
quasi identiques, donc l'écart-type pixel-à-pixel ENTRE seeds est ~0. C'est la
mesure que ce script produit, plus deux repères :

- `inter_seed_std`   : moyenne sur les pixels de l'écart-type entre seeds.
- `mean_pairwise_rmse` : RMSE moyenne entre paires d'échantillons distincts.
- `dataset_std`      : même mesure sur N images réelles du dataset (plafond
                       atteignable — un modèle parfait s'en approche).
- `banding_ratio`    : énergie des différences verticales (lignes) rapportée aux
                       différences horizontales. ~1 = isotrope ; ≫1 = banding
                       horizontal (des lignes qui changent d'une rangée à l'autre).

Usage :
  python3 tools/sample_diversity.py --glob 'out/L_seed*.png' [--label L]
  python3 tools/sample_diversity.py --dataset datasets/cifar10_grey.batraw --n 16
"""
import argparse
import glob as globmod
import json
import struct
import sys

import numpy as np
from PIL import Image


def load_images(pattern):
    paths = sorted(globmod.glob(pattern))
    if not paths:
        sys.exit(f"aucune image pour le motif {pattern!r}")
    arrs = []
    for p in paths:
        img = Image.open(p).convert("L")
        arrs.append(np.asarray(img, dtype=np.float64) / 255.0)
    return paths, np.stack(arrs)


def load_dataset(path, n, size=32):
    """Lit les n premières images d'un .batraw (magic BATRAW1/2)."""
    with open(path, "rb") as fh:
        header = fh.read(24)
        magic = header[:8].rstrip(b"\x00").decode("ascii", "replace")
        count, width, height, channels = struct.unpack("<IIII", header[8:24])
        per = width * height * channels
        raw = np.frombuffer(fh.read(per * n), dtype=np.uint8)
    imgs = raw.reshape(n, height, width, channels)[..., 0].astype(np.float64) / 255.0
    return magic, count, imgs


def banding_ratio(imgs):
    """Différences verticales (entre rangées) vs horizontales (entre colonnes)."""
    dv = np.diff(imgs, axis=-2)  # rangée -> rangée
    dh = np.diff(imgs, axis=-1)  # colonne -> colonne
    v = float(np.sqrt((dv**2).mean()))
    h = float(np.sqrt((dh**2).mean()))
    return v / h if h > 0 else float("inf"), v, h


def stats(imgs, label):
    n = imgs.shape[0]
    inter = imgs.std(axis=0, ddof=1) if n > 1 else np.zeros_like(imgs[0])
    pair = []
    for i in range(n):
        for j in range(i + 1, n):
            pair.append(float(np.sqrt(((imgs[i] - imgs[j]) ** 2).mean())))
    ratio, v, h = banding_ratio(imgs)
    return {
        "label": label,
        "n": n,
        "inter_seed_std": float(inter.mean()),
        "inter_seed_std_max": float(inter.max()),
        "mean_pairwise_rmse": float(np.mean(pair)) if pair else 0.0,
        "min_pairwise_rmse": float(np.min(pair)) if pair else 0.0,
        "intra_image_std": float(imgs.std(axis=(-2, -1)).mean()),
        "pixel_mean": float(imgs.mean()),
        "banding_ratio": ratio,
        "row_diff_rms": v,
        "col_diff_rms": h,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob")
    ap.add_argument("--label", default="samples")
    ap.add_argument("--dataset")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    if args.dataset:
        magic, count, imgs = load_dataset(args.dataset, args.n)
        out = stats(imgs, f"dataset({magic}, {count} imgs)")
    else:
        paths, imgs = load_images(args.glob)
        out = stats(imgs, args.label)
        out["files"] = len(paths)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
