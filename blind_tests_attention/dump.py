"""Lecture du dump `--dump` de `--headless-perpetual` (format publié par --help).

    "BATFLUX1" u32 w u32 h u32 c
    par frame : u8 phase, u32 t, f32[w*h*c] x_t, f32[w*h*c] x0_hat
"""
import struct
import numpy as np


def read(path):
    b = open(path, "rb").read()
    assert b[:8] == b"BATFLUX1", b[:8]
    w, h, c = struct.unpack_from("<III", b, 8)
    n = w * h * c
    stride = 1 + 4 + 8 * n
    off, frames = 20, []
    while off + stride <= len(b):
        phase = b[off]
        (t,) = struct.unpack_from("<I", b, off + 1)
        xt = np.frombuffer(b, "<f4", n, off + 5).astype(np.float64).reshape(h, w, c)
        x0 = np.frombuffer(b, "<f4", n, off + 5 + 4 * n).astype(np.float64).reshape(h, w, c)
        frames.append({"phase": phase, "t": t, "x_t": xt, "x0_hat": x0})
        off += stride
    return {"w": w, "h": h, "c": c, "frames": frames}
