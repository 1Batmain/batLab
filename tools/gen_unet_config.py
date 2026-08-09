#!/usr/bin/env python3
"""Génère le config_file d'un U-Net de diffusion en calculant les dims.

Les dims du JSON sont recalculées ici (et non écrites à la main) : `Layer::new`
propage de toute façon la sortie de la couche précédente comme `dim_input`, mais
`dim_kernel.z` DOIT égaler `dim_input.z` — la convolution le rejette au build
(`KernelChannelMismatch`), l'UpsampleConv NON (corruption silencieuse : son
shader indexe les poids avec `IC = dim_input.z`). Ce script rend l'erreur
impossible en dérivant `dim_kernel.z` de la dim courante.

Usage : python3 tools/gen_unet_config.py > Models/<name>/config_file
        python3 tools/gen_unet_config.py --name Color_Diffusion_L --signal-channels 3 \
                                         --widths 32 64 128 > Models/Color_Diffusion_L/config_file
        # un U-Net 64x64 à quatre étages (64 -> 32 -> 16 -> 8) :
        python3 tools/gen_unet_config.py --name Color_Diffusion_64 --signal-channels 3 \
                                         --size 64 --widths 48 96 192 384 --attention \
                                         > Models/Color_Diffusion_64/config_file

Le nombre d'étages est LIBRE : `--widths` prend autant de largeurs qu'on veut
(une par étage, la dernière étant le goulot), et `--size` fixe la résolution
d'entrée. Chaque `//2` de descente doit tomber sur un entier, donc `size` doit
être divisible par `2**(len(widths)-1)`. Les dims restent toutes DÉRIVÉES.

Les valeurs par défaut reproduisent `Greyscale_Diffusion_L` à l'octet près
(taille 32, largeurs 32/64/128). Un garde-fou le vérifie dans
`tools/test_dataset_tools.py`.
"""
import argparse
import json
import sys

DEFAULT_NAME = "Greyscale_Diffusion_L"
DEFAULT_SIGNAL_CHANNELS = 1  # niveaux de gris ; 3 pour RGB
TIME_CHANNELS = 4  # canaux d'embedding temporel (input.z - output.z)

# Résolution d'entrée par défaut, carrée.
DEFAULT_SIZE = 32
# Largeurs par étage : 32x32 -> 16x16 -> 8x8 (le dernier étage est le goulot).
DEFAULT_WIDTHS = (32, 64, 128)
GROUPS = 8


class Builder:
    def __init__(self, size, channels):
        self.dim = [size, size, channels]
        self.layers = []
        self.skips = {}

    def conv(self, nb_kernel, stride=1, k=3, save_key=None):
        dim_in = list(self.dim)
        self.layers.append({
            "Convolution": {
                "dim_input": dim_in,
                "nb_kernel": nb_kernel,
                # profondeur du noyau == canaux d'entrée (garde-fou du build)
                "dim_kernel": [k, k, dim_in[2]],
                "stride": stride,
                "padding": "Same",
                "save_key": save_key,
            }
        })
        # padding Same : out = ceil(in / stride)
        self.dim = [-(-dim_in[0] // stride), -(-dim_in[1] // stride), nb_kernel]
        if save_key:
            self.skips[save_key] = list(self.dim)
        return self

    def upsample(self, nb_kernel, scale=2, k=3, save_key=None):
        dim_in = list(self.dim)
        self.layers.append({
            "UpsampleConv": {
                "dim_input": dim_in,
                "scale_factor": scale,
                "nb_kernel": nb_kernel,
                "dim_kernel": [k, k, dim_in[2]],
                "padding": "Same",
                "save_key": save_key,
            }
        })
        self.dim = [dim_in[0] * scale, dim_in[1] * scale, nb_kernel]
        if save_key:
            self.skips[save_key] = list(self.dim)
        return self

    def norm(self, groups=GROUPS, save_key=None):
        assert self.dim[2] % groups == 0, (
            f"GroupNorm : {self.dim[2]} canaux non divisibles par {groups} groupes")
        self.layers.append({
            "GroupNorm": {
                "dim_input": list(self.dim),
                "num_groups": groups,
                "save_key": save_key,
            }
        })
        if save_key:
            self.skips[save_key] = list(self.dim)
        return self

    def silu(self, save_key=None):
        self.layers.append({
            "Activation": {
                "dim_input": list(self.dim),
                "method": "Silu",
                "save_key": save_key,
            }
        })
        if save_key:
            self.skips[save_key] = list(self.dim)
        return self

    def attention(self, save_key=None):
        """Self-attention spatiale, résiduel inclus dans la couche.

        Shape-preserving : elle n'a aucune dimension propre, la séquence est la
        grille spatiale et la dimension de features le nombre de canaux. Le
        scratch `probs` est en N² par échantillon — 64 positions au goulot 8x8
        font 4 096 flottants, la MÊME couche en 32x32 en ferait 1 048 576.
        C'est une couche de goulot, pas une couche à mettre partout.
        """
        self.layers.append({
            "Attention": {
                "dim_input": list(self.dim),
                "save_key": save_key,
            }
        })
        if save_key:
            self.skips[save_key] = list(self.dim)
        return self

    def concat(self, skip_key, save_key=None):
        skip = self.skips[skip_key]
        assert skip[0] == self.dim[0] and skip[1] == self.dim[1], (
            f"concat spatial mismatch {self.dim} vs {skip}")
        self.layers.append({
            "Concat": {
                "dim_input": list(self.dim),
                "dim_skip": list(skip),
                "skip_key": skip_key,
                "save_key": save_key,
            }
        })
        self.dim = [self.dim[0], self.dim[1], self.dim[2] + skip[2]]
        return self

    # bloc "norm -> silu -> conv" (pré-activation, comme la baseline)
    def block(self, nb_kernel, stride=1, groups=GROUPS):
        return self.norm(groups).silu().conv(nb_kernel, stride)


def build(signal_channels=DEFAULT_SIGNAL_CHANNELS, widths=DEFAULT_WIDTHS,
          size=DEFAULT_SIZE, attention=False):
    """U-Net symétrique à `len(widths)` étages, résolution d'entrée `size`.

    Chaque étage divise la résolution par deux (padding Same, stride 2). Le
    DERNIER étage est le goulot : il ne sauve pas de skip et reçoit l'attention
    optionnelle. Le décodeur remonte symétriquement, en concaténant à chaque
    palier le skip de même résolution pris à la descente.

    Le nom du skip est sa résolution (`skip64`, `skip32`, …) : c'est ce qui rend
    la valeur par défaut identique à l'ancienne (skip32/skip16) au caractère près.
    """
    widths = list(widths)
    n = len(widths)
    if n < 2:
        raise ValueError("il faut au moins deux étages (un descendant + le goulot)")
    downs = n - 1
    if size % (1 << downs) != 0:
        raise ValueError(
            f"size={size} n'est pas divisible par 2**{downs}={1 << downs} : "
            f"une descente tomberait sur une résolution non entière")

    b = Builder(size, signal_channels + TIME_CHANNELS)
    res = size

    # --- stem : pleine résolution, widths[0] --------------------------------
    b.conv(widths[0])
    b.norm().silu(save_key=f"skip{res}")   # skip pris après activation

    # --- étages descendants intermédiaires (skip sauvé à chaque palier) -----
    for k in range(1, n - 1):
        res //= 2
        b.conv(widths[k], stride=2)
        b.block(widths[k])                 # norm/silu/conv
        b.norm().silu(save_key=f"skip{res}")

    # --- goulot : dernier étage, pas de skip --------------------------------
    res //= 2
    b.conv(widths[-1], stride=2)
    b.block(widths[-1])
    if attention:
        # GroupNorm PUIS attention : la pré-normalisation est la couche
        # GroupNorm existante placée devant, pas une normalisation interne à
        # l'attention. C'est ici que le contenu global se décide — les positions
        # du goulot se voient toutes, ce qui manque au modèle pour composer un
        # objet plutôt qu'une texture. Le scratch `probs` étant en N² par
        # échantillon, garder le goulot à 8x8 (64 positions) : le monter à 16x16
        # quadruplerait les positions et x16 le scratch.
        b.norm().attention()
    b.norm().silu()

    # --- décodeur : remonte jusqu'à chaque résolution sauvée ----------------
    for k in range(n - 2, -1, -1):
        res *= 2
        b.upsample(widths[k])
        b.concat(f"skip{res}")
        b.block(widths[k])

    # --- tête : retour au signal -------------------------------------------
    b.norm().silu()
    b.conv(signal_channels)

    assert b.dim == [size, size, signal_channels], b.dim
    return b


def macs(layers):
    """MACs par échantillon (forward), pour dimensionner le coût."""
    total = 0
    for layer in layers:
        if "Convolution" in layer:
            c = layer["Convolution"]
            di, k, s = c["dim_input"], c["dim_kernel"], c["stride"]
            ox, oy = -(-di[0] // s), -(-di[1] // s)
            total += ox * oy * c["nb_kernel"] * k[0] * k[1] * k[2]
        elif "UpsampleConv" in layer:
            c = layer["UpsampleConv"]
            di, k, f = c["dim_input"], c["dim_kernel"], c["scale_factor"]
            ox, oy = di[0] * f, di[1] * f
            total += ox * oy * c["nb_kernel"] * k[0] * k[1] * k[2]
    return total


def build_config(name=DEFAULT_NAME, signal_channels=DEFAULT_SIGNAL_CHANNELS,
                 widths=DEFAULT_WIDTHS, size=DEFAULT_SIZE, attention=False):
    """Le dict config complet — la seule source de l'octet écrit sur disque.

    Isolé de `__main__` pour qu'un test puisse le comparer au `config_file`
    d'un modèle de référence (cf. `tools/test_dataset_tools.py`).
    """
    b = build(signal_channels, widths, size, attention)
    return b, {
        "model_name": name,
        "input_size": [size, size, signal_channels + TIME_CHANNELS],
        "layers": b.layers,
        "inference": {
            "random_seed": False,
            "seed": 390,
            "denoising_paths": 3,
            "denoise_magnitude": 0.3,
        },
        "run": {"mode": "Infer"},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=DEFAULT_NAME, help="model_name du config")
    p.add_argument("--signal-channels", type=int, default=DEFAULT_SIGNAL_CHANNELS,
                   help="canaux image : 1 (grey) ou 3 (RGB)")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE,
                   help=f"résolution d'entrée carrée (défaut {DEFAULT_SIZE}) ; "
                        f"doit être divisible par 2**(len(widths)-1)")
    p.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS),
                   metavar="C",
                   help="largeur par étage (nombre libre) ; la dernière est le "
                        "goulot. Défaut 32 64 128 (trois étages 32/16/8)")
    p.add_argument("--attention", action="store_true",
                   help="insère GroupNorm + Attention au goulot")
    args = p.parse_args()

    b, config = build_config(args.name, args.signal_channels, args.widths,
                             args.size, args.attention)
    attn = sum(1 for l in b.layers if "Attention" in l)
    print(f"# {len(b.layers)} couches ({attn} attention), "
          f"{macs(b.layers)/1e6:.1f} MMACs/échantillon (convolutions seules)",
          file=sys.stderr)
    json.dump(config, sys.stdout, indent=2)
    print()
