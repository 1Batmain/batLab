"""Lecteur du dump binaire produit par --headless-perpetual --dump.

Le format est désormais publié par `main --help` (il ne l'était pas lors de la première
passe : je l'avais reconstitué par dichotomie sur la taille du fichier — cf. défaut D4).
Citation du contrat :

    "BATFLUX1"  u32 width  u32 height  u32 channels
    then per frame:  u8 phase (0 descent, 1 climb, 2 flux)
                     u32 t (the level this frame landed on)
                     f32[w*h*c] x_t        f32[w*h*c] x0_hat

    The frame count is left to be inferred from the file size, so a run killed
    mid-write still parses up to its last whole frame.

Ce lecteur est celui de la première passe, inchangé quant au décodage : la structure
publiée confirme exactement ce qui avait été déduit à l'aveugle. Seul le nom du premier
octet change (« tag toujours 0 » → « phase »), et le dernier enregistrement partiel est
désormais toléré comme le contrat l'autorise.
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
    n = body // rec
    partial = body % rec  # toléré par le contrat : run tué en cours d'écriture
    xt = np.empty((n, h, w), dtype=np.float32)
    x0 = np.empty((n, h, w), dtype=np.float32)
    phase = np.empty(n, dtype=np.uint8)
    meta = np.empty(n, dtype=np.uint32)
    for i in range(n):
        o = HEADER + i * rec
        phase[i] = raw[o]
        meta[i] = int.from_bytes(raw[o + 1 : o + 5], "little")
        f = np.frombuffer(raw, dtype=np.float32, count=2 * npix, offset=o + 5)
        xt[i] = f[:npix].reshape(h, w)
        x0[i] = f[npix:].reshape(h, w)
    return dict(magic=magic, w=w, h=h, c=c, n=n, phase=phase, tag=phase,
                meta=meta, xt=xt, x0=x0, partial=partial)


def corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")
