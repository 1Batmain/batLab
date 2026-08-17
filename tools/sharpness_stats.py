#!/usr/bin/env python3
"""Statistiques de netteté d'un lot d'images, ancrées sur le dataset réel.

La question de la campagne d'échantillonnage n'est pas « est-ce joli » mais
« les statistiques de la génération approchent-elles celles des VRAIES photos ».
Ce script produit donc le même vecteur de mesures sur deux sources — un motif de
PNG générés, ou un `.batraw` (les vraies images) — pour qu'on les compare
terme à terme. La cible, c'est le dataset ; trop lisse ET trop bruité s'en
écartent des deux côtés.

Ce qu'il mesure (par image puis moyenné, sauf mention) :

- ``intra_image_std``   : écart-type des pixels d'une image. Un aplat de couleur
                          tombe vers 0 ; du bruit ou du contraste le montent. À
                          approcher **par le bas** (le dépasser = bruit ajouté).
- ``grad_energy``       : énergie moyenne du gradient (différences voisines au
                          carré), √ pour rester en unités de pixel. La « quantité
                          de bord ». **Le bruit la maximise** — ne pas la lire
                          seule, la comparer à la cible.
- ``laplacian_var``     : variance du laplacien 4-voisins, la mesure de netteté
                          classique (variance of Laplacian). Même piège que
                          ``grad_energy`` : le bruit la gonfle.
- ``saturation``        : ``chroma_std`` — écart-type entre canaux d'un même pixel,
                          moyenné. 0 = image grise ; la cible dit combien de
                          couleur portent les vraies photos.
- ``intra_image_std`` / ``banding_ratio`` viennent de ``sample_diversity`` : on
  s'en sert plutôt que de les réécrire.
- ``inter_seed_std`` / ``mean_pairwise_rmse`` : diversité entre images (utile
  quand la source est un lot de seeds distinctes).

Usage :
  python3 tools/sharpness_stats.py --glob 'out/*.png' [--label L] [--json]
  python3 tools/sharpness_stats.py --dataset datasets/elephants256.batraw [--n N]

`--json` imprime une ligne JSON (pour agréger un balayage) au lieu du tableau.
"""
import argparse
import glob as globmod
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batraw  # noqa: E402  (le décodeur .batraw, source unique)
# On réutilise les mesures existantes plutôt que de les réécrire.
from sample_diversity import banding_ratio, stats as diversity_stats  # noqa: E402


def load_png_stack(pattern):
    paths = sorted(globmod.glob(pattern))
    if not paths:
        sys.exit(f"aucune image pour le motif {pattern!r}")
    arrs = []
    for p in paths:
        # RGB, jamais "L" : convertir en luminance mélangerait les canaux avant
        # la mesure (même raison que sample_diversity).
        arrs.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0)
    return np.stack(arrs), len(paths)


def load_dataset_stack(path, n):
    """Les n premières images d'un `.batraw`, en [0,1], via le décodeur unique.

    ``batraw.read`` rend toujours des octets u8 quel que soit le format (BATRAW3
    u8 comme BATRAW1/2 f32), donc /255 suffit pour retomber en [0,1] comme un PNG
    décodé.
    """
    payload, count, w, h, c, _ = batraw.read(path)
    take = min(n, count) if n else count
    flat = np.frombuffer(payload, dtype=np.uint8)[: take * w * h * c]
    imgs = flat.reshape(take, h, w, c).astype(np.float64) / 255.0
    return imgs, count


def gradient_energy(imgs):
    """RMS du gradient : √ moyenne(dh² + dv²) sur tout le lot.

    `imgs` est (n, H, W, C). Le bruit la maximise ; c'est pour ça qu'elle se lit
    contre la cible du dataset, pas en valeur absolue.
    """
    dh = np.diff(imgs, axis=-2)
    dv = np.diff(imgs, axis=-3)
    return float(np.sqrt((dh ** 2).mean() + (dv ** 2).mean()))


def laplacian_variance(imgs):
    """Variance du laplacien 4-voisins sur l'intérieur (mesure de netteté).

    Noyau [[0,1,0],[1,-4,1],[0,1,0]] appliqué par décalages numpy, luminance
    moyenne des canaux. Bords exclus pour ne pas mesurer le padding.
    """
    lum = imgs.mean(axis=-1)  # (n, H, W)
    lap = (
        -4.0 * lum[:, 1:-1, 1:-1]
        + lum[:, :-2, 1:-1]
        + lum[:, 2:, 1:-1]
        + lum[:, 1:-1, :-2]
        + lum[:, 1:-1, 2:]
    )
    # Variance par image puis moyenne : une image nette a un laplacien qui
    # « bouge » beaucoup, un aplat un laplacien plat.
    return float(lap.reshape(lap.shape[0], -1).var(axis=1).mean())


def measure(imgs, label):
    div = diversity_stats(imgs, label)  # intra_image_std, banding_ratio, chroma…
    out = {
        "label": label,
        "n": int(imgs.shape[0]),
        "intra_image_std": div["intra_image_std"],
        "grad_energy": gradient_energy(imgs),
        "laplacian_var": laplacian_variance(imgs),
        "pixel_mean": div["pixel_mean"],
        "banding_ratio": div["banding_ratio"],
        "inter_seed_std": div["inter_seed_std"],
        "mean_pairwise_rmse": div["mean_pairwise_rmse"],
    }
    # chroma_std n'existe que pour un lot RGB ; le dataset et les PNG le sont.
    out["saturation"] = div.get("chroma_std", 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob")
    ap.add_argument("--label", default="samples")
    ap.add_argument("--dataset")
    ap.add_argument("--n", type=int, default=0, help="0 = toutes les images du dataset")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.dataset:
        imgs, count = load_dataset_stack(args.dataset, args.n)
        out = measure(imgs, f"{os.path.basename(args.dataset)} ({imgs.shape[0]}/{count})")
    elif args.glob:
        imgs, nfiles = load_png_stack(args.glob)
        out = measure(imgs, args.label)
        out["files"] = nfiles
    else:
        sys.exit("--glob ou --dataset requis")

    if args.json:
        print(json.dumps(out))
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
