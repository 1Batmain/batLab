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

Les valeurs par défaut reproduisent `Greyscale_Diffusion_L` à l'octet près.
"""
import argparse
import json
import sys

DEFAULT_NAME = "Greyscale_Diffusion_L"
DEFAULT_SIGNAL_CHANNELS = 1  # niveaux de gris ; 3 pour RGB
TIME_CHANNELS = 4  # canaux d'embedding temporel (input.z - output.z)

# Largeurs par étage : 32x32 -> 16x16 -> 8x8
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


def build(signal_channels=DEFAULT_SIGNAL_CHANNELS, widths=DEFAULT_WIDTHS):
    C1, C2, C3 = widths
    b = Builder(32, signal_channels + TIME_CHANNELS)

    # --- stem : 32x32, C1 ---------------------------------------------------
    b.conv(C1)                        # [32,32,32]
    b.norm().silu(save_key="skip32")  # skip pris après activation

    # --- down 32 -> 16, C2 --------------------------------------------------
    b.conv(C2, stride=2)              # [16,16,64]
    b.block(C2)                       # norm/silu/conv -> [16,16,64]
    b.norm().silu(save_key="skip16")

    # --- down 16 -> 8, C3 (bottleneck) --------------------------------------
    b.conv(C3, stride=2)              # [8,8,128]
    b.block(C3)                       # [8,8,128]
    b.norm().silu()

    # --- up 8 -> 16 ---------------------------------------------------------
    b.upsample(C2)                    # [16,16,64]
    b.concat("skip16")                # [16,16,128]
    b.block(C2)                       # [16,16,64]

    # --- up 16 -> 32 --------------------------------------------------------
    b.upsample(C1)                    # [32,32,32]
    b.concat("skip32")                # [32,32,64]
    b.block(C1)                       # [32,32,32]

    # --- tête : retour au signal -------------------------------------------
    b.norm().silu()
    b.conv(signal_channels)           # [32,32,C_signal]

    assert b.dim == [32, 32, signal_channels], b.dim
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=DEFAULT_NAME, help="model_name du config")
    p.add_argument("--signal-channels", type=int, default=DEFAULT_SIGNAL_CHANNELS,
                   help="canaux image : 1 (grey) ou 3 (RGB)")
    p.add_argument("--widths", type=int, nargs=3, default=list(DEFAULT_WIDTHS),
                   metavar=("C1", "C2", "C3"),
                   help="largeurs des trois étages 32x32 / 16x16 / 8x8")
    args = p.parse_args()

    b = build(args.signal_channels, args.widths)
    config = {
        "model_name": args.name,
        "input_size": [32, 32, args.signal_channels + TIME_CHANNELS],
        "layers": b.layers,
        "inference": {
            "random_seed": False,
            "seed": 390,
            "denoising_paths": 3,
            "denoise_magnitude": 0.3,
        },
        "run": {"mode": "Infer"},
    }
    print(f"# {len(b.layers)} couches, {macs(b.layers)/1e6:.1f} MMACs/échantillon",
          file=sys.stderr)
    json.dump(config, sys.stdout, indent=2)
    print()
