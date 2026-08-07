"""Sonde boîte noire : lire la sortie brute du réseau (ε̂) depuis le dump perpétuel.

Le binaire n'expose aucune couche isolée. Mais `--headless-perpetual --dump`
publie, pour chaque frame, `t`, `x_t` et `x0_hat` en f32. Le pas inverse d'un
DDPM est AFFINE en la sortie du réseau :

        x0_hat(t) = a(t) · x_t  −  b(t) · eps_hat

Deux runs de calibration suffisent à fixer a et b sans rien savoir du schedule :

  * poids ET biais du réseau à zéro  → eps_hat ≡ 0 → a(t) = x0_hat / x_t
  * poids à zéro, biais = c connu    → eps_hat ≡ c → b(t) = (a·x_t − x0_hat)/c

La constance de a et de b sur les 192 valeurs de la frame est elle-même le test
de l'hypothèse affine — si le sampler écrêtait ou n'était pas affine, elle
tomberait. Ensuite, pour n'importe quels poids, eps_hat = (a·x_t − x0_hat)/b.

Seule la frame 0 est utilisée : son `x_t` est le bruit initial, donc identique
d'un run à l'autre à graine égale (vérifié) et INDÉPENDANT du modèle.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt as ckptio
import dump as dumpio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(HERE, "work")
BIN = os.path.join(ROOT, "target/release/batlab")


def run_perpetual(model, ckpt_path, out_dump, seed=7, actions=1, t_star=100, magnitude=1.0):
    env = dict(os.environ, BATLAB_ROOT=WORK)
    cmd = [BIN, "--headless-perpetual", model, "--checkpoint", ckpt_path,
           "--regime", "flux", "--actions", str(actions), "--t-star", str(t_star),
           "--seed", str(seed), "--magnitude", str(magnitude), "--dump", out_dump]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return dumpio.read(out_dump)


def frame0(model, records, tag, seed=7, tmp=None):
    tmp = tmp or os.path.join(WORK, "probe")
    os.makedirs(tmp, exist_ok=True)
    cp = os.path.join(tmp, f"{tag}.ckpt")
    dp = os.path.join(tmp, f"{tag}.f32")
    ckptio.write(cp, records)
    d = run_perpetual(model, cp, dp, seed=seed)
    return d["frames"][0]
