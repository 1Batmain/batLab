#!/usr/bin/env python3
"""Planches comparatives : une rangée par config, les 6 graines en colonnes,
upscalées (le natif est 32x32). Chaque rangée porte une étiquette.

  python3 planche.py <out.png> <scale> "TEMPLATE::Label" ...
où TEMPLATE contient {s} pour la graine, p.ex.
  "/dir/m1.2_beta_raw_p1_s{s}.png::mag1.2 beta raw"
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

SEEDS = [101, 202, 303, 404, 505, 606]
out, scale = sys.argv[1], int(sys.argv[2])
rows_spec = sys.argv[3:]

CELL = 32 * scale
LABEL_W = 250
HEAD = 30
GAP = 6
cols = len(SEEDS)
rows = len(rows_spec)
W = LABEL_W + cols * (CELL + GAP) + GAP
H = HEAD + rows * (CELL + GAP) + GAP
canvas = Image.new("RGB", (W, H), (245, 245, 245))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    fhead = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
except Exception:
    font = fhead = ImageFont.load_default()

for j, s in enumerate(SEEDS):
    x = LABEL_W + j * (CELL + GAP) + GAP
    draw.text((x + CELL // 2 - 18, 8), f"seed {s}", fill=(30, 30, 30), font=fhead)

for i, spec in enumerate(rows_spec):
    tmpl, _, label = spec.partition("::")
    y = HEAD + i * (CELL + GAP) + GAP
    draw.text((8, y + CELL // 2 - 8), label, fill=(20, 20, 20), font=font)
    for j, s in enumerate(SEEDS):
        p = tmpl.format(s=s)
        x = LABEL_W + j * (CELL + GAP) + GAP
        if os.path.exists(p):
            img = Image.open(p).convert("RGB").resize((CELL, CELL), Image.NEAREST)
            canvas.paste(img, (x, y))
        else:
            draw.rectangle([x, y, x + CELL, y + CELL], fill=(200, 120, 120))

canvas.save(out)
print(f"wrote {out}  ({W}x{H})")
