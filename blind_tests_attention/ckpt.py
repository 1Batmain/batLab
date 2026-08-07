"""Lecture/écriture du conteneur `.ckpt` de batlab.

Format déduit en BOÎTE NOIRE — hexdump, comptage des paramètres de configs dont
je connais la géométrie, et vérification que le parseur tombe EXACTEMENT sur la
fin du fichier. Aucune source du moteur n'a été lue.

    b"BBCKPT2"                     7 octets de magie
    u32  n_records                 nb de couches PARAMÉTRÉES (les autres absentes)
    n_records fois :
        u32 layer_index            indice dans la liste `layers` de la config
        u32 n_w   f32[n_w]         tenseur de poids
        u32 n_b   f32[n_b]         tenseur de biais
    u32  has_optimizer             0 = fin du fichier
    si has_optimizer :
        u64 step
        u32 n_records
        n_records fois :
            u32 layer_index  u32 k(=4)
            k fois : u32 len  f32[len]     (m_w, v_w, m_b, v_b)

Vérifié à l'octet près sur Tiny (SGD, sans optimiseur) et sur
`Color_Diffusion_XL_attn_5700.ckpt` (Adam, 20 couches paramétrées, 14 311 503 o).
"""
import struct
import numpy as np

MAGIC = b"BBCKPT2"


def read(path):
    b = open(path, "rb").read()
    assert b[:7] == MAGIC, b[:8]
    (n,) = struct.unpack_from("<I", b, 7)
    off = 11
    recs = []
    for _ in range(n):
        idx, nw = struct.unpack_from("<II", b, off); off += 8
        w = np.frombuffer(b, "<f4", nw, off).astype(np.float64); off += 4 * nw
        (nb,) = struct.unpack_from("<I", b, off); off += 4
        bi = np.frombuffer(b, "<f4", nb, off).astype(np.float64); off += 4 * nb
        recs.append({"index": idx, "w": w, "b": bi})
    (has_opt,) = struct.unpack_from("<I", b, off); off += 4
    out = {"records": recs, "has_optimizer": has_opt, "step": None, "moments": []}
    if has_opt:
        (out["step"],) = struct.unpack_from("<Q", b, off); off += 8
        (m,) = struct.unpack_from("<I", b, off); off += 4
        for _ in range(m):
            idx, k = struct.unpack_from("<II", b, off); off += 8
            ts = []
            for _ in range(k):
                (L,) = struct.unpack_from("<I", b, off); off += 4
                ts.append(np.frombuffer(b, "<f4", L, off).astype(np.float64)); off += 4 * L
            out["moments"].append({"index": idx, "tensors": ts})
    assert off == len(b), (off, len(b))
    return out


def write(path, records, has_optimizer=0):
    """records : liste de {"index": i, "w": array, "b": array} — sans optimiseur."""
    out = bytearray(MAGIC) + struct.pack("<I", len(records))
    for r in records:
        w = np.asarray(r["w"], dtype="<f4").ravel()
        bi = np.asarray(r["b"], dtype="<f4").ravel()
        out += struct.pack("<II", int(r["index"]), w.size) + w.tobytes()
        out += struct.pack("<I", bi.size) + bi.tobytes()
    out += struct.pack("<I", has_optimizer)
    open(path, "wb").write(bytes(out))
    return len(out)
