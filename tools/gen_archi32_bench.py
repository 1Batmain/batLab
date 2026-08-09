#!/usr/bin/env python3
"""Génère le banc d'architectures de la mission archi32 (item 4), écrit sous
Models/ comme configs COMMITTÉES et reproductibles.

Quatre candidats 32×32 RGB (entrée [32,32,7] = 3 canaux signal + 4 canaux
d'embedding temporel, sortie 3), à entraîner from scratch sur elephants_all,
mêmes pas / lr / seed, puis à juger par `--eval` :

  A = Archi32_A_Baseline  — l'archi actuelle (équivalent Color_Diffusion_XL,
      attention au goulot 8×8). C'est l'archi d'Elephants_XL.
  B = Archi32_B_TimeBias  — A + TimeBias dans les blocs profonds (item 3).
  C = Archi32_C_Wide      — A avec largeur ×1.5 (48/96/192 → 72/144/288).
  D = Archi32_D_Residual  — A + un bloc résiduel de plus par étage (item 2, Add).

Usage : python3 tools/gen_archi32_bench.py [Models_dir]
"""
import json
import os
import sys

MODELS = sys.argv[1] if len(sys.argv) > 1 else "Models"
SIGNAL = 3           # canaux image (RGB)
T_CH = 4             # canaux d'embedding temporel ; entrée = SIGNAL + T_CH = 7
OFFSET = SIGNAL      # les canaux temporels commencent après le signal


class Net:
    """Empile des couches en propageant (h, w, z) — le config stocke les vraies
    dims comme le fait le TUI, sinon le chargeur ne les infère pas."""

    def __init__(self):
        self.h, self.w, self.z = 32, 32, SIGNAL + T_CH
        self.layers = []

    def act(self, save=None, method="Silu"):
        self.layers.append({"Activation": {"dim_input": [self.h, self.w, self.z],
                                            "method": method, "save_key": save}})

    def gn(self, groups=8):
        # groups doit diviser z ; on retombe sur le plus grand diviseur ≤ 8.
        g = groups
        while self.z % g != 0:
            g -= 1
        self.layers.append({"GroupNorm": {"dim_input": [self.h, self.w, self.z],
                                          "num_groups": g, "save_key": None}})

    def conv(self, nk, stride=1, save=None):
        self.layers.append({"Convolution": {"dim_input": [self.h, self.w, self.z],
                                            "nb_kernel": nk, "dim_kernel": [3, 3, self.z],
                                            "stride": stride, "padding": "Same", "save_key": save}})
        if stride == 2:
            self.h = (self.h + 1) // 2
            self.w = (self.w + 1) // 2
        self.z = nk

    def attn(self):
        self.layers.append({"Attention": {"dim_input": [self.h, self.w, self.z], "save_key": None}})

    def up(self, nk):
        self.layers.append({"UpsampleConv": {"dim_input": [self.h, self.w, self.z],
                                             "scale_factor": 2, "nb_kernel": nk,
                                             "dim_kernel": [3, 3, self.z], "padding": "Same", "save_key": None}})
        self.h *= 2
        self.w *= 2
        self.z = nk

    def cat(self, key, skipz):
        self.layers.append({"Concat": {"dim_input": [self.h, self.w, self.z],
                                       "dim_skip": [self.h, self.w, skipz], "skip_key": key, "save_key": None}})
        self.z += skipz

    def tb(self):
        self.layers.append({"TimeBias": {"dim_input": [self.h, self.w, self.z],
                                         "time_key": "tin", "embed_offset": OFFSET,
                                         "embed_channels": T_CH, "save_key": None}})

    def res_add(self, key):
        """Add(clé) : sortie = entrée courante + tenseur sauvé sous `key`."""
        self.layers.append({"Add": {"dim_input": [self.h, self.w, self.z],
                                    "dim_skip": [self.h, self.w, self.z], "skip_key": key, "save_key": None}})


def build(variant, c0=48, c1=96, c2=192):
    """Le U-Net de A, paramétré par les trois largeurs. `variant` ∈
    {'A','B','C','D'} ajoute TimeBias (B) ou des blocs résiduels (D)."""
    n = Net()
    if variant == "B":
        n.act(save="tin", method="Linear")     # sauve l'entrée pour TimeBias
    # -- encodeur 32×32 --
    n.conv(c0); n.gn(); n.act(save="skip32")    # 32×32×c0
    if variant == "D":                          # bloc résiduel de plus (32×32)
        n.act(save="r32", method="Linear"); n.conv(c0); n.gn(); n.act(); n.res_add("r32"); n.gn(); n.act()
    # -- descente 16×16 --
    n.conv(c1, stride=2); n.gn(); n.act()       # 16×16×c1
    if variant == "B": n.tb()
    n.conv(c1); n.gn(); n.act(save="skip16")    # 16×16×c1
    if variant == "D":
        n.act(save="r16", method="Linear"); n.conv(c1); n.gn(); n.act(); n.res_add("r16"); n.gn(); n.act()
    # -- goulot 8×8 --
    n.conv(c2, stride=2); n.gn(); n.act()       # 8×8×c2
    if variant == "B": n.tb()
    n.conv(c2); n.gn()
    n.attn()
    if variant == "D":
        n.act(save="r8", method="Linear"); n.conv(c2); n.gn(); n.act(); n.res_add("r8"); n.gn()
    n.act()
    if variant == "B": n.tb()
    # -- remontée --
    n.up(c1); n.cat("skip16", c1); n.gn(); n.act()   # 16×16×(2*c1)
    if variant == "B": n.tb()
    n.conv(c1)
    n.up(c0); n.cat("skip32", c0); n.gn(); n.act()   # 32×32×(2*c0)
    if variant == "B": n.tb()
    n.conv(c0); n.gn(); n.act()
    n.conv(SIGNAL)                                    # sortie 32×32×3
    return n.layers


SPECS = {
    "Archi32_A_Baseline": dict(variant="A"),
    "Archi32_B_TimeBias": dict(variant="B"),
    "Archi32_C_Wide": dict(variant="C", c0=72, c1=144, c2=288),
    "Archi32_D_Residual": dict(variant="D"),
}

for name, kw in SPECS.items():
    layers = build(**kw)
    cfg = {
        "model_name": name,
        "input_size": [32, 32, SIGNAL + T_CH],
        "layers": layers,
        "inference": {"random_seed": False, "seed": 42, "denoising_paths": 3, "denoise_magnitude": 0.3},
        "run": {"mode": "Infer"},
    }
    d = os.path.join(MODELS, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config_file"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"{name}: {len(layers)} couches")
