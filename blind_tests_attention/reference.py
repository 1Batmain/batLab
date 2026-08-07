"""Référence NumPy f64 de la couche d'attention, écrite DEPUIS LA FORMULE.

Spec (`docs/reports/ATTENTION.md` §1, `MISSION_BLIND_ATTENTION.md`) :

    y = x + W_o · softmax(qᵀk / √d) · v

où la séquence est faite des H·W positions spatiales, la dimension de features
est le nombre de canaux C, et Q, K, V, W_o sont quatre projections C×C (1×1),
chacune avec son biais. Single-head, d = C.

Aucune source du moteur n'a été lue pour l'écrire.
"""
import warnings

import numpy as np

# numpy 2.0.2 + Accelerate émet des RuntimeWarning « divide by zero / overflow »
# sur des `matmul` dont les entrées ET les sorties sont pourtant toutes finies
# (vérifié). Bruit de plateforme, pas un signal — on le tait ici seulement.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)   # stabilisation par le max
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(x, Wq, Wk, Wv, Wo, bq, bk, bv, bo):
    """x : (N, C) f64 — N positions, C canaux. Wi : (C, C) en [sortie, entrée]."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    q = x @ np.asarray(Wq, float).T + bq          # (N, C)
    k = x @ np.asarray(Wk, float).T + bk
    v = x @ np.asarray(Wv, float).T + bv
    scores = q @ k.T / np.sqrt(d)                  # (N, N) : ligne n = position n
    probs = softmax(scores, axis=1)                # normalisée sur les positions lues
    ctx = probs @ v                                # (N, C)
    return x + ctx @ np.asarray(Wo, float).T + bo


def pack(Wq, Wk, Wv, Wo, bq, bk, bv, bo):
    """Empaquette dans le tenseur unique 4·C·C / 4·C du checkpoint."""
    w = np.concatenate([np.asarray(m, float).ravel() for m in (Wq, Wk, Wv, Wo)])
    b = np.concatenate([np.asarray(v, float).ravel() for v in (bq, bk, bv, bo)])
    return w, b
