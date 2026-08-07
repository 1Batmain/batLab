"""Sonde : lire la sortie brute du réseau (ε̂) d'un modèle jouet, en boîte noire.

Voir `probe.py` pour la calibration affine. Ce module assemble le tout :
construit une config jouet `[Convolution 1×1 identité, Attention]`, écrit un
checkpoint choisi, lance `--headless-perpetual --dump`, et rend ε̂.

La couche 0 est une convolution 1×1 dont les poids sont l'IDENTITÉ sur les trois
canaux image et ZÉRO sur le(s) canal(aux) d'embedding temporel : l'entrée de
l'attention est donc exactement le `x_t` publié par le dump, sans avoir à
supposer quoi que ce soit du schedule ni de l'embedding.
"""
import json
import os
import subprocess

import numpy as np

import ckpt as ckptio
import dump as dumpio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(HERE, "work")
TMP = os.path.join(WORK, "probe")
BIN = os.path.join(ROOT, "target/release/batlab")
SAT = 0.9999            # |x0_hat| au-delà duquel le sampler a écrêté : lecture invalide
FRAME = 160             # frame de lecture pendant la descente (t = 94)


def ensure_model(name, w, h, c, with_attention):
    layers = [{"Convolution": {"dim_input": [w, h, c + 1], "nb_kernel": c,
                               "dim_kernel": [1, 1, c + 1], "stride": 1,
                               "padding": "Same", "save_key": None}}]
    if with_attention:
        layers.append({"Attention": {"dim_input": [w, h, c], "save_key": None}})
    cfg = {"model_name": name, "input_size": [w, h, c + 1], "layers": layers,
           "inference": {"random_seed": False, "seed": 390, "denoising_paths": 1,
                         "denoise_magnitude": 1.0},
           "run": {"mode": "Infer"}}
    d = os.path.join(WORK, "Models", name)
    os.makedirs(os.path.join(d, "pretrained_weights"), exist_ok=True)
    with open(os.path.join(d, "config_file"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    return name


def identity_conv(c):
    m = np.zeros((c, c + 1))
    m[:, :c] = np.eye(c)
    return m.ravel()


def perpetual(model, records, tag, seed=7, actions=FRAME + 1, magnitude=0.1):
    os.makedirs(TMP, exist_ok=True)
    cp, dp = os.path.join(TMP, tag + ".ckpt"), os.path.join(TMP, tag + ".f32")
    ckptio.write(cp, records)
    cmd = [BIN, "--headless-perpetual", model, "--checkpoint", cp, "--regime", "flux",
           "--actions", str(actions), "--t-star", "1", "--seed", str(seed),
           "--magnitude", str(magnitude), "--dump", dp]
    r = subprocess.run(cmd, env=dict(os.environ, BATLAB_ROOT=WORK),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return dumpio.read(dp)["frames"]


def calibrate(model, c, seed=7):
    """Rend (a, b, t) du pas inverse à la frame FRAME : x0_hat = a·x_pre − b·ε̂."""
    nw = c * (c + 1)
    fz = perpetual(model, [{"index": 0, "w": np.zeros(nw), "b": np.zeros(c)}],
                   "cal_zero", seed=seed)
    x0 = fz[FRAME]["x0_hat"]; m = np.abs(x0) < SAT
    a = float((x0[m] / fz[FRAME - 1]["x_t"][m]).mean())
    const = np.linspace(0.4, -0.25, c)
    fb = perpetual(model, [{"index": 0, "w": np.zeros(nw), "b": const}],
                   "cal_bias", seed=seed)
    x0b = fb[FRAME]["x0_hat"]; mb = np.abs(x0b) < SAT
    cc = np.broadcast_to(const, x0b.shape)
    b = float(((a * fb[FRAME - 1]["x_t"] - x0b) / cc)[mb].mean())
    return a, b, int(fz[FRAME]["t"])


def eps_hat(frames, a, b):
    """Rend (ε̂ observé, masque des positions non écrêtées, entrée x)."""
    x0 = frames[FRAME]["x0_hat"]
    x = frames[FRAME - 1]["x_t"]
    return (a * x - x0) / b, np.abs(x0) < SAT, x
