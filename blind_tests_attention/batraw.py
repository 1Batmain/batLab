"""Écriture d'un dataset `.batraw` (format documenté dans tools/sample_diversity.py) :
magic 8 o, u32 count/width/height/channels, puis count*w*h*c f32 LE en [-1,1]."""
import struct
import numpy as np


def write(path, samples):
    a = np.asarray(samples, dtype="<f4")           # (count, h, w, c)
    n, h, w, c = a.shape
    with open(path, "wb") as fh:
        fh.write(b"BATRAW2\x00")
        fh.write(struct.pack("<IIII", n, w, h, c))
        fh.write(a.tobytes())
    return path
