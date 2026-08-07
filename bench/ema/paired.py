#!/usr/bin/env python3
"""Comparaison APPARIÉE des deux bras : même graine à gauche et à droite.

`tools/sample_diversity.py` résume chaque bras séparément — deux nombres qu'on
compare à l'œil, sans savoir si leur écart tient au jeu de poids ou au tirage
des seeds. Ici les deux bras partagent leurs graines, donc chaque seed est sa
propre paire et la différence se lit par paire :

- `mean_rmse_per_seed` : à quel point les deux jeux de poids divergent sur une
  MÊME trajectoire. 0 = les deux images sont le même fichier (l'EMA n'a rien
  changé) ; un chiffre du même ordre que `mean_pairwise_rmse` intra-bras = les
  deux jeux sont aussi éloignés l'un de l'autre que deux seeds le sont.
- les deltas signés par seed sur `intra_image_std` : combien de seeds sur N
  l'EMA améliore, pas seulement la moyenne (une moyenne peut bouger sur un seul
  outlier).
"""
import argparse
import glob as globmod
import json
import os
import sys

import numpy as np
from PIL import Image

# Le plafond du dataset, mesuré par tools/sample_diversity.py sur CIFAR-10 gris.
DATASET_INTRA_STD = 0.206


def load(dirpath):
    paths = sorted(globmod.glob(os.path.join(dirpath, "seed_*.png")))
    if not paths:
        sys.exit(f"aucune image dans {dirpath!r}")
    keys = [os.path.basename(p) for p in paths]
    arrs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0 for p in paths]
    return keys, np.stack(arrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dossier du bras A (EMA)")
    ap.add_argument("--b", required=True, help="dossier du bras B (poids bruts)")
    args = ap.parse_args()

    ka, a = load(args.a)
    kb, b = load(args.b)
    if ka != kb:
        sys.exit(f"les deux bras n'ont pas les mêmes graines : {ka} vs {kb}")

    per_seed_rmse = np.sqrt(((a - b) ** 2).mean(axis=(1, 2, 3)))
    std_a = a.std(axis=(1, 2, 3))
    std_b = b.std(axis=(1, 2, 3))
    # « Plus proche du dataset » et non « plus grand » : l'écart-type intra-image
    # doit APPROCHER 0,206 par le bas, pas le dépasser (CLAUDE.md).
    closer = np.abs(std_a - DATASET_INTRA_STD) < np.abs(std_b - DATASET_INTRA_STD)

    print(json.dumps({
        "n": len(ka),
        "identical_files": bool(np.all(per_seed_rmse == 0.0)),
        "mean_rmse_per_seed": float(per_seed_rmse.mean()),
        "max_rmse_per_seed": float(per_seed_rmse.max()),
        "min_rmse_per_seed": float(per_seed_rmse.min()),
        "intra_std_a": [round(float(v), 4) for v in std_a],
        "intra_std_b": [round(float(v), 4) for v in std_b],
        "seeds_where_a_is_closer_to_dataset": int(closer.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
