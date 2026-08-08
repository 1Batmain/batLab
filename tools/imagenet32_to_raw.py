#!/usr/bin/env python3
"""« Downsampled ImageNet 32×32 » → un `.batraw` BATRAW3.

Une fondation plus large que CIFAR-10 : 1 281 160 images contre 50 000, mêmes
32×32, même géométrie de modèle. C'est le jeu à partir duquel fine-tuner vers un
corpus étroit (voir `docs/reports/CUSTOM_DATASET.md`).

    python3 tools/imagenet32_to_raw.py ~/Downloads/Imagenet32_train \\
        --out datasets/imagenet32_rgb.batraw

    # Un sous-ensemble, pour un premier essai qui ne coûte pas 4 Go :
    python3 tools/imagenet32_to_raw.py ~/Downloads/Imagenet32_train \\
        --out datasets/imagenet32_small.batraw --limit 50000

**Ce script ne télécharge rien.** Il attend, dans le dossier donné, les fichiers
de l'archive officielle :

    train_data_batch_1 … train_data_batch_10      (SANS extension)

Ce sont des **pickles** Python, pas des `.npz` : chacun est un dict portant
`data` — un `ndarray` uint8 de forme (n, 3072) — et `labels`, dont ce projet
n'a que faire (la diffusion ici est inconditionnelle). Les clés sont parfois des
`str`, parfois des `bytes` selon la façon dont l'archive a été produite ; les
deux sont acceptées. Si les fichiers manquent, le script dit lesquels et
s'arrête — il n'invente rien.

Le piège, le même que pour CIFAR
--------------------------------
Les 3072 octets d'une image sont **trois plans** de 1024 (R entier, puis G
entier, puis B entier), pas des pixels entrelacés. Le moteur attend l'inverse :
canaux contigus, pixel par pixel. Lu tel quel, le dataset donne des images dont
le tiers haut est rouge, le tiers médian vert et le tiers bas bleu — et
l'entraînement ne lève aucune erreur. La conversion se fait ici, et le test
`test_batches_are_read_in_numeric_order_and_interleaved` la tient.

La taille, et pourquoi BATRAW3
------------------------------
3,94 Go en u8 contre 15,7 Go en f32. Ce n'est pas qu'une question de disque :
15,7 Go de f32 doivent aussi tenir en RAM hôte pendant tout le run, alors que
3,94 Go tiennent en plus **entièrement dans un tampon GPU résident** sous
`--gpu-limits native` (le binding de cette machine plafonne à 4 Gio) — donc zéro
transfert hôte→GPU par pas. C'est la configuration cible pour un run de
fondation.

Dépendance : numpy.
"""

import argparse
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

WIDTH = HEIGHT = 32
PIXELS = WIDTH * HEIGHT
PLANES = 3
SAMPLE_VALUES = PIXELS * PLANES  # 3072

#: Les noms attendus. Nommés en clair pour que l'erreur soit actionnable.
BATCH_PATTERN = re.compile(r"^train_data_batch_(\d+)$")

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


def load_batch(path):
    """Le tableau `(n, 3072)` uint8 d'un batch."""
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
    if data.ndim != 2 or data.shape[1] != SAMPLE_VALUES:
        raise ValueError(
            f"{path} : data a la forme {data.shape}, attendu (n, {SAMPLE_VALUES}) — "
            f"est-ce bien du « Downsampled ImageNet 32×32 » ?"
        )
    return data


def to_samples(block, mode):
    """Un bloc `(n, 3072)` en plans → les octets attendus par le moteur.

    C'est ici que les trois plans deviennent des pixels.
    """
    planes = block.reshape(-1, PLANES, PIXELS)
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
    """Convertit et rend un rapport chiffré."""
    batches = collect_batches(directory)
    channels = 1 if mode == "grey" else PLANES
    report = {"batches": len(batches), "written": 0}

    with batraw.Writer(out_path, WIDTH, HEIGHT, channels, version=version) as out:
        for path in batches:
            if limit and out.count >= limit:
                break
            data = load_batch(path)
            if verbose:
                print(f"  {os.path.basename(path)} : {len(data)} images", flush=True)
            for start in range(0, len(data), BLOCK):
                if limit and out.count >= limit:
                    break
                block = data[start : start + BLOCK]
                if limit:
                    block = block[: limit - out.count]
                out.append(to_samples(block, mode))
            # Le batch pèse ~400 Mo : le lâcher avant d'ouvrir le suivant garde
            # le pic à un batch, pas à deux.
            del data
        report["written"] = out.count
        report["bytes"] = None

    report["bytes"] = os.path.getsize(out_path)
    if report["written"] == 0:
        raise ValueError(f"aucune image lue depuis '{directory}'")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Convertit « Downsampled ImageNet 32×32 » en .batraw pour batLab.",
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
    print(
        f"{report['written']} image(s) ({WIDTH}×{HEIGHT}×{channels}, {args.format}) "
        f"depuis {report['batches']} batch(es) → {args.out}"
        f"  [{report['bytes'] / 1e9:.2f} Go]"
    )
    print(
        "  Regarder la planche avant d'entraîner :\n"
        f"    python3 -c \"import sys; sys.path.insert(0,'tools'); import images_to_raw; \"\\\n"
        f"      \"images_to_raw.contact_sheet('{args.out}', 'imagenet32_sheet.png')\""
    )


if __name__ == "__main__":
    main()
