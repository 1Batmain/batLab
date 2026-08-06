#!/usr/bin/env python3
"""Compose une planche comparative : une rangée par modèle, une colonne par seed.

Une rangée optionnelle d'images réelles du dataset sert de repère visuel (ce que
le modèle devrait finir par produire).

Usage :
  python3 tools/compare_plate.py --out plate.png \
      --row 'baseline 10k:eval/baseline/seed_*.png' \
      --row 'L 10k:eval/L/seed_*.png' \
      --dataset-row 'CIFAR-10 (réel):datasets/cifar10_grey.batraw'
"""
import argparse
import glob as globmod
import struct

import numpy as np
from PIL import Image, ImageDraw

SCALE = 4
PAD = 6
LABEL_W = 150
TITLE_H = 22


def load_row(pattern, limit):
    paths = sorted(globmod.glob(pattern))[:limit]
    # "RGB" et non "L" : une planche censee montrer un modele couleur ne doit pas
    # decolorer ses rangees. Un PNG gris passe en trois canaux egaux, inchange.
    return [Image.open(p).convert("RGB") for p in paths]


def load_dataset_row(path, limit, offset=0):
    with open(path, "rb") as fh:
        header = fh.read(24)
        magic = header[:8].rstrip(b"\x00").decode("ascii", "replace")
        _, width, height, channels = struct.unpack("<IIII", header[8:24])
        per = width * height * channels
        fh.seek(24 + per * offset * 4)
        raw = np.frombuffer(fh.read(per * limit * 4), dtype="<f4")
    # payload f32 (main.rs:941) ; BATRAW1 est en [0,1], BATRAW2 en [-1,1]
    # Tous les canaux : `[..., 0]` affichait le seul ROUGE d'un dataset RGB.
    imgs = raw.reshape(-1, height, width, channels).astype(np.float64)
    if magic != "BATRAW1":
        imgs = (imgs + 1.0) / 2.0
    out = []
    for a in imgs:
        a = np.clip(a * 255, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(a[..., 0] if channels == 1 else a[..., :3]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", action="append", default=[], help="'label:glob'")
    ap.add_argument("--dataset-row", help="'label:path.batraw'")
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    rows = []
    for spec in args.row:
        label, pattern = spec.split(":", 1)
        rows.append((label, load_row(pattern, args.cols)))
    if args.dataset_row:
        label, path = args.dataset_row.split(":", 1)
        rows.append((label, load_dataset_row(path, args.cols)))

    if not rows:
        raise SystemExit("aucune rangée")
    tile = rows[0][1][0].size[0] * SCALE
    width = LABEL_W + args.cols * (tile + PAD) + PAD
    height = TITLE_H + len(rows) * (tile + PAD) + PAD

    canvas = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    for c in range(args.cols):
        x = LABEL_W + c * (tile + PAD)
        draw.text((x + 4, 6), f"seed {c + 1}", fill=(150, 150, 160))

    for r, (label, imgs) in enumerate(rows):
        y = TITLE_H + r * (tile + PAD)
        draw.text((6, y + tile // 2 - 6), label, fill=(230, 230, 235))
        for c, img in enumerate(imgs[: args.cols]):
            up = img.resize((tile, tile), Image.NEAREST).convert("RGB")
            canvas.paste(up, (LABEL_W + c * (tile + PAD), y))

    canvas.save(args.out)
    print(f"écrit {args.out} ({width}x{height})")


if __name__ == "__main__":
    main()
