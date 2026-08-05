"""Lecteur du dump binaire produit par --headless-perpetual --dump.

Format déduit par observation (boîte noire) — voir BLIND_TEST_FLUX.md, §format :
  en-tête : magic 8 octets ("BATFLUX1") + u32 width + u32 height + u32 channels
  puis, par frame : 1 octet (toujours 0 observé) + u32 little-endian (compteur/t)
                    + width*height*channels f32 (x_t) + width*height*channels f32 (x0_hat)
"""

import numpy as np

HEADER = 20


def read_dump(path):
    raw = open(path, "rb").read()
    magic = raw[:8]
    w, h, c = np.frombuffer(raw[8:HEADER], dtype=np.uint32)
    w, h, c = int(w), int(h), int(c)
    npix = w * h * c
    rec = 5 + 2 * npix * 4
    body = len(raw) - HEADER
    if body % rec != 0:
        raise ValueError(f"{path}: corps {body} non multiple de {rec}")
    n = body // rec
    xt = np.empty((n, h, w), dtype=np.float32)
    x0 = np.empty((n, h, w), dtype=np.float32)
    tag = np.empty(n, dtype=np.uint8)
    meta = np.empty(n, dtype=np.uint32)
    for i in range(n):
        o = HEADER + i * rec
        tag[i] = raw[o]
        meta[i] = int.from_bytes(raw[o + 1 : o + 5], "little")
        f = np.frombuffer(raw, dtype=np.float32, count=2 * npix, offset=o + 5)
        xt[i] = f[:npix].reshape(h, w)
        x0[i] = f[npix:].reshape(h, w)
    return dict(magic=magic, w=w, h=h, c=c, n=n, tag=tag, meta=meta, xt=xt, x0=x0)


def corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")
