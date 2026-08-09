#!/usr/bin/env python3
"""« Downsampled ImageNet » (32×32 **ou** 64×64) → un `.batraw` BATRAW3.

Une fondation plus large que CIFAR-10 : 1 281 160 images contre 50 000, même
géométrie de plans. C'est le jeu à partir duquel fine-tuner vers un corpus
étroit (voir `docs/reports/CUSTOM_DATASET.md`).

    python3 tools/imagenet_to_raw.py ~/Downloads/Imagenet64_train \\
        --out datasets/imagenet64_rgb.batraw

    # Un sous-ensemble, pour un premier essai qui ne coûte pas des Go :
    python3 tools/imagenet_to_raw.py ~/Downloads/Imagenet64_train \\
        --out datasets/imagenet64_small.batraw --limit 50000

**La résolution est déduite, pas demandée.** Les deux releases officielles —
32×32 et 64×64 — ont le **même format** : seul le nombre de valeurs par ligne
change (3072 = 32²·3, ou 12288 = 64²·3). Le script lit `data.shape[1]`, en tire
le côté (`√(cols/3)`), et refuse une ligne qui n'est pas trois plans carrés.
Toutes les batches d'un dossier doivent porter la même taille.

**Ce script ne télécharge rien.** Il attend, dans le dossier donné, les fichiers
de l'archive officielle :

    train_data_batch_1 … train_data_batch_10      (SANS extension)

Ce sont des **pickles** Python, pas des `.npz` : chacun est un dict portant
`data` — un `ndarray` uint8 de forme (n, 3·côté²) — et `labels`, dont ce projet
n'a que faire (la diffusion ici est inconditionnelle). Les clés sont parfois des
`str`, parfois des `bytes` selon la façon dont l'archive a été produite ; les
deux sont acceptées. Si les fichiers manquent, le script dit lesquels et
s'arrête — il n'invente rien.

Le piège, le même que pour CIFAR
--------------------------------
Les octets d'une image sont **trois plans** (R entier, puis G entier, puis B
entier), pas des pixels entrelacés. Le moteur attend l'inverse : canaux
contigus, pixel par pixel. Lu tel quel, le dataset donne des images dont le tiers
haut est rouge, le tiers médian vert et le tiers bas bleu — et l'entraînement ne
lève aucune erreur. La conversion se fait ici, et le test
`test_the_three_planes_become_pixels` la tient.

La taille, et pourquoi BATRAW3
------------------------------
En 32×32, 3,94 Go en u8 contre 15,7 Go en f32 ; en 64×64, quatre fois plus. Ce
n'est pas qu'une question de disque : le f32 doit aussi tenir en RAM hôte pendant
tout le run, alors que l'u8 tient en plus **entièrement dans un tampon GPU
résident** sous `--gpu-limits native` (le binding de cette machine plafonne à
4 Gio) — donc zéro transfert hôte→GPU par pas. C'est la configuration cible pour
un run de fondation. (ImageNet 64×64 en u8 fait ~15,7 Go : au-delà du binding, il
sera streamé par chunks — regarder la bannière du run.)

Dépendance : numpy.
"""

import argparse
import math
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batraw  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("numpy est requis : pip install numpy", file=sys.stderr)
    raise

PLANES = 3

#: Les noms attendus. Nommés en clair pour que l'erreur soit actionnable.
BATCH_PATTERN = re.compile(r"^train_data_batch_(\d+)$")


def side_from_row(cols):
    """Le côté carré d'une ligne de `cols` valeurs en trois plans.

    Refuse ce qui n'est pas exactement `3·côté²` : une ligne de la mauvaise
    longueur n'est pas un « Downsampled ImageNet », et deviner produirait des
    images tordues sans erreur.
    """
    if cols % PLANES != 0:
        raise ValueError(f"{cols} valeurs/ligne non divisibles par {PLANES} plans")
    pixels = cols // PLANES
    side = int(round(math.isqrt(pixels)))
    if side * side != pixels:
        raise ValueError(
            f"{cols} valeurs/ligne → {pixels} pixels, qui n'est pas un carré "
            f"(attendu 3·côté², p. ex. 3072=32² ou 12288=64²)")
    return side

#: Images traitées d'un coup. Un batch entier en float64 pour la luminance
#: coûterait ~1 Go de pic ; par blocs, le pic est négligeable et la progression
#: s'imprime.
BLOCK = 16384


def collect_batches(directory):
    """Les batches du dossier, triés **numériquement**.

    Un tri lexicographique placerait `train_data_batch_10` entre `_1` et `_2` :
    l'ordre du dataset dépendrait alors du nombre de fichiers présents, et un
    run repris sur une archive complétée ne verrait plus les mêmes images aux
    mêmes indices.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"'{directory}' n'est pas un dossier. Attendu : le dossier contenant "
            f"train_data_batch_1 … train_data_batch_10 (fichiers SANS extension)."
        )
    found = []
    for name in os.listdir(directory):
        match = BATCH_PATTERN.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(directory, name)))
    if not found:
        listing = sorted(os.listdir(directory))[:6]
        raise FileNotFoundError(
            f"aucun train_data_batch_N dans '{directory}'. "
            f"Attendu : train_data_batch_1 … train_data_batch_10, sans extension "
            f"(ce sont des pickles, pas des .npz). "
            f"Trouvé à la place : {listing or 'rien'}"
        )
    return [path for _, path in sorted(found)]


def load_batch(path, expected_cols=None):
    """Le tableau `(n, cols)` uint8 d'un batch.

    `expected_cols`, s'il est donné, impose que cette batche porte la même taille
    d'image que les précédentes : mélanger du 32×32 et du 64×64 dans un dossier
    produirait un `.batraw` incohérent.
    """
    with open(path, "rb") as fh:
        # `encoding='bytes'` pour les pickles produits sous Python 2 ; les clés
        # ressortent alors en bytes sur certaines archives et en str sur
        # d'autres, d'où la recherche sur les deux.
        payload = pickle.load(fh, encoding="bytes")

    data = payload.get("data", payload.get(b"data"))
    if data is None:
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in payload]
        raise ValueError(f"{path} : pas de clé 'data' (clés présentes : {keys})")

    data = np.asarray(data, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError(
            f"{path} : data a la forme {data.shape}, attendu (n, 3·côté²) — "
            f"est-ce bien un « Downsampled ImageNet » ?")
    side_from_row(data.shape[1])  # rejette une ligne qui n'est pas 3 plans carrés
    if expected_cols is not None and data.shape[1] != expected_cols:
        raise ValueError(
            f"{path} : {data.shape[1]} valeurs/ligne, mais une batche précédente "
            f"en portait {expected_cols} — un dossier ne peut pas mêler deux tailles")
    return data


def to_samples(block, mode):
    """Un bloc `(n, 3·côté²)` en plans → les octets attendus par le moteur.

    C'est ici que les trois plans deviennent des pixels.
    """
    pixels = block.shape[1] // PLANES
    planes = block.reshape(-1, PLANES, pixels)
    if mode == "grey":
        # float64 puis `rint`, exactement comme `cifar_to_raw.py` : les deux
        # convertisseurs doivent donner le même gris, sans quoi un fine-tune de
        # l'un vers l'autre change la définition en cours de route.
        lum = (
            0.299 * planes[:, 0].astype(np.float64)
            + 0.587 * planes[:, 1].astype(np.float64)
            + 0.114 * planes[:, 2].astype(np.float64)
        )
        return np.rint(lum).astype(np.uint8).tobytes()
    return np.ascontiguousarray(planes.transpose(0, 2, 1)).tobytes()


def convert(directory, out_path, mode="rgb", limit=0, version=batraw.DEFAULT_VERSION,
            verbose=False):
    """Convertit et rend un rapport chiffré. La résolution est déduite du 1er batch."""
    batches = collect_batches(directory)
    channels = 1 if mode == "grey" else PLANES

    # La taille vient des données, pas d'un flag : lire une ligne du premier
    # batch, en tirer le côté, et l'imposer à tous les suivants.
    first = load_batch(batches[0])
    side = side_from_row(first.shape[1])
    expected_cols = first.shape[1]
    report = {"batches": len(batches), "written": 0, "side": side}

    with batraw.Writer(out_path, side, side, channels, version=version) as out:
        for i, path in enumerate(batches):
            if limit and out.count >= limit:
                break
            data = first if i == 0 else load_batch(path, expected_cols)
            first = None
            if verbose:
                print(f"  {os.path.basename(path)} : {len(data)} images "
                      f"({side}×{side})", flush=True)
            for start in range(0, len(data), BLOCK):
                if limit and out.count >= limit:
                    break
                block = data[start : start + BLOCK]
                if limit:
                    block = block[: limit - out.count]
                out.append(to_samples(block, mode))
            # Le batch pèse plusieurs centaines de Mo : le lâcher avant d'ouvrir
            # le suivant garde le pic à un batch, pas à deux.
            del data
        report["written"] = out.count
        report["bytes"] = None

    report["bytes"] = os.path.getsize(out_path)
    if report["written"] == 0:
        raise ValueError(f"aucune image lue depuis '{directory}'")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Convertit « Downsampled ImageNet » (32×32 ou 64×64) en .batraw pour batLab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "directory",
        help="dossier contenant train_data_batch_1 … train_data_batch_10 (rien n'est téléchargé)",
    )
    parser.add_argument("--out", required=True, help="fichier .batraw à écrire")
    parser.add_argument(
        "--mode",
        choices=["rgb", "grey"],
        default="rgb",
        help="rgb (3 canaux, défaut, exact) ou grey (1 canal, luminance BT.601 arrondie)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="s'arrête après N images (0 = tout). Un sous-ensemble pour un premier essai",
    )
    batraw.add_format_argument(parser)
    args = parser.parse_args()

    try:
        report = convert(
            args.directory,
            args.out,
            mode=args.mode,
            limit=args.limit,
            version=batraw.version_from_choice(args.format),
            verbose=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)

    channels = 1 if args.mode == "grey" else PLANES
    side = report["side"]
    print(
        f"{report['written']} image(s) ({side}×{side}×{channels}, {args.format}) "
        f"depuis {report['batches']} batch(es) → {args.out}"
        f"  [{report['bytes'] / 1e9:.2f} Go]"
    )
    print(
        "  Regarder la planche avant d'entraîner :\n"
        f"    python3 -c \"import sys; sys.path.insert(0,'tools'); import images_to_raw; \"\\\n"
        f"      \"images_to_raw.contact_sheet('{args.out}', 'imagenet_sheet.png')\""
    )


if __name__ == "__main__":
    main()
