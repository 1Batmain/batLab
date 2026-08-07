#!/usr/bin/env python3
"""Quelle distance sépare la moyenne de l'itéré, DANS le checkpoint.

Une planche d'images qui ne bouge pas admet deux lectures : « l'EMA n'apporte
rien à cette échelle » ou « les deux jeux de poids sont quasiment le même jeu ».
Elles n'ont pas les mêmes conséquences, et seul le fichier les départage — d'où
cette lecture directe de la remorque EMA (`BBCKPT3`), sans GPU.

Sortie, par couche puis global : ‖ema − w‖ / ‖w‖.

Format (crates/batlab-core/src/model/model.rs, `checkpoint_bytes`) :

    magic 7 o ("BBCKPT1|2|3")
    u32 nb_entrées
      par entrée : u32 index_couche, u32 n_w, f32*n_w, u32 n_b, f32*n_b
    remorque optimiseur (V2+) : u32 tag ; si tag==1 (Adam) :
      u64 t, u32 nb_couches, par couche : u32 index, u32 nb_buffers,
      par buffer : u32 len, f32*len
    remorque EMA (V3) : u32 tag ; si tag==1 :
      f32 decay, u32 nb_couches, même forme que ci-dessus, les buffers étant
      [ema_weights, ema_bias]
"""
import argparse
import json
import struct
import sys

import numpy as np


class Reader:
    def __init__(self, data):
        self.d = data
        self.o = 0

    def u32(self):
        v = struct.unpack_from("<I", self.d, self.o)[0]
        self.o += 4
        return v

    def u64(self):
        v = struct.unpack_from("<Q", self.d, self.o)[0]
        self.o += 8
        return v

    def f32(self):
        v = struct.unpack_from("<f", self.d, self.o)[0]
        self.o += 4
        return v

    def vec(self, n):
        a = np.frombuffer(self.d, dtype="<f4", count=n, offset=self.o).astype(np.float64)
        self.o += 4 * n
        return a


def sections(r):
    """`nb_couches` puis, par couche, ses buffers — la forme des deux remorques."""
    out = {}
    for _ in range(r.u32()):
        idx = r.u32()
        out[idx] = [r.vec(r.u32()) for _ in range(r.u32())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    with open(args.ckpt, "rb") as fh:
        r = Reader(fh.read())

    magic = bytes(r.d[:7]).decode("ascii", "replace")
    r.o = 7
    if magic not in ("BBCKPT1", "BBCKPT2", "BBCKPT3"):
        sys.exit(f"magic inattendu : {magic!r}")

    raw = {}
    for _ in range(r.u32()):
        idx = r.u32()
        w = r.vec(r.u32())
        b = r.vec(r.u32())
        raw[idx] = [w, b]

    if magic in ("BBCKPT2", "BBCKPT3"):
        if r.u32() == 1:  # Adam
            r.u64()
            sections(r)

    if magic != "BBCKPT3":
        sys.exit(f"{args.ckpt} est un {magic} : aucune moyenne à comparer")
    if r.u32() != 1:
        sys.exit("remorque EMA marquée absente dans un BBCKPT3")
    decay = r.f32()
    ema = sections(r)
    if r.o != len(r.d):
        sys.exit(f"octets résiduels : {len(r.d) - r.o}")

    per_layer, num, den, nw = [], 0.0, 0.0, 0
    for idx in sorted(ema):
        # [0] = poids, [1] = biais ; le rapport se lit sur les poids, les biais
        # sont trois ordres de grandeur moins nombreux.
        w, e = raw[idx][0], ema[idx][0]
        d = float(np.linalg.norm(e - w))
        n = float(np.linalg.norm(w))
        per_layer.append({
            "layer": idx,
            "n_weights": int(w.size),
            "rel_gap": d / n if n > 0 else float("nan"),
        })
        num += d * d
        den += n * n
        nw += w.size

    print(json.dumps({
        "file": args.ckpt,
        "magic": magic,
        "ema_decay": decay,
        "trainable_layers": len(per_layer),
        "n_weights": nw,
        "global_rel_gap": (num ** 0.5) / (den ** 0.5),
        "per_layer": per_layer,
    }, indent=2))


if __name__ == "__main__":
    main()
