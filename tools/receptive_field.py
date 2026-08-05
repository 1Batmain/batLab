#!/usr/bin/env python3
"""Calcule le champ réceptif (en pixels d'entrée) le long d'un config_file.

Récurrence standard : pour une couche de noyau `k` et de stride `s`,
    RF_out = RF_in + (k - 1) * jump_in       jump_out = jump_in * s
Un `UpsampleConv` de facteur `f` suivi d'une conv `k` divise d'abord le jump
par `f` (interpolation) puis applique la conv :
    jump' = jump / f                          RF_out = RF_in + (k - 1) * jump'

Enjeu : un modèle de diffusion dont le champ réceptif de sortie est plus petit
que l'image ne peut pas choisir un contenu *global* — chaque pixel n'est décidé
que par son voisinage, ce qui pousse la sortie vers la moyenne du dataset.

Usage : python3 tools/receptive_field.py Models/<name>/config_file
"""
import json
import sys


def walk(config):
    rf, jump = 1.0, 1.0
    rows = []
    for i, layer in enumerate(config["layers"]):
        kind = next(iter(layer))
        spec = layer[kind]
        if kind == "Convolution":
            k, s = spec["dim_kernel"][0], spec["stride"]
            rf += (k - 1) * jump
            jump *= s
        elif kind == "UpsampleConv":
            k, f = spec["dim_kernel"][0], spec["scale_factor"]
            jump /= f
            rf += (k - 1) * jump
        else:
            # GroupNorm / Activation / Concat : pointwise ou fusion, RF inchangé
            # (le Concat réinjecte un skip dont le RF est plus petit, donc le RF
            # maximal du chemin principal reste la borne)
            continue
        rows.append((i, kind, spec.get("nb_kernel"), rf, jump))
    return rows


def main():
    for path in sys.argv[1:]:
        config = json.load(open(path))
        rows = walk(config)
        size = config["input_size"][0]
        print(f"== {config.get('model_name', path)} (entrée {size}x{size}) ==")
        for i, kind, nb, rf, jump in rows:
            print(f"  L{i:<3} {kind:<14} nb={nb!s:<5} RF={rf:6.1f} px  jump={jump:g}")
        final = rows[-1][3]
        print(f"  -> champ réceptif de sortie : {final:.0f} px "
              f"({'couvre' if final >= size else 'NE COUVRE PAS'} l'image {size} px)\n")


if __name__ == "__main__":
    main()
