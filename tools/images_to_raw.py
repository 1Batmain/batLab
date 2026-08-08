#!/usr/bin/env python3
"""Un dossier d'images hétérogènes → un `.batraw` entraînable.

C'est la porte d'entrée pour entraîner batLab sur SES images : on pointe un
dossier, on obtient un dataset au format du dépôt (`tools/batraw.py`).

    python3 tools/images_to_raw.py mes_photos/ --out datasets/elephants.batraw \\
        --size 32 --mode rgb --min-size 32 --dedup \\
        --contact-sheet docs/gallery/elephants.png

**Toujours regarder la planche.** `--contact-sheet` est la seule façon de voir ce
sur quoi on va réellement entraîner, après recadrage et redimensionnement — et
c'est là que se voient les erreurs qui ne lèvent aucune exception : un dossier
qui contenait surtout des vignettes, un `--crop center` qui décapite tous les
sujets, un `--mode grey` demandé par erreur. Un dataset qu'on n'a pas regardé
coûte une nuit d'entraînement pour l'apprendre.

Ce que l'outil fait de chaque image
-----------------------------------
1. **orientation EXIF** appliquée (une photo de téléphone en portrait est
   stockée en paysage avec un tag ; l'ignorer entraîne sur des images
   couchées) ;
2. **carré** — `--crop center` prend le plus grand carré centré (défaut : un
   dataset de diffusion veut des cadrages serrés), `--crop fit` met l'image
   entière dans le carré et complète en gris moyen (l'octet 128, c'est-à-dire
   0 dans la plage [-1,1] du modèle : la zone ajoutée est neutre) ;
3. **redimensionnement** à `--size` (Lanczos) ;
4. **mode** `rgb` (3 canaux) ou `grey` (1 canal, luminance BT.601 — les mêmes
   poids que `cifar_to_raw.py`).

Un scrape, ça, et donc de la robustesse
---------------------------------------
- fichier illisible, tronqué, ou qui n'est pas une image : **sauté**, et compté ;
- `--min-size N` écarte les images dont le plus petit côté est sous `N`.
  Agrandir du 20×15 en 32×32 donne de la bouillie que le modèle apprendra
  consciencieusement. Par défaut aucune n'est écartée, mais le compte des
  images agrandies est imprimé : s'il est gros, poser `--min-size <size>` ;
- `--dedup` écarte les doublons **sur l'échantillon produit**, pas sur le
  fichier source : deux ré-encodages JPEG du même cliché sont deux fichiers
  différents et un seul échantillon. C'est le doublon qui compte, parce que
  c'est celui que le modèle verrait deux fois ;
- l'ordre est **déterministe** : tri par chemin relatif. Deux exécutions sur le
  même dossier donnent le même fichier, octet pour octet.

Dépendances : Pillow. numpy est optionnel (il n'accélère que les formats f32).
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batraw  # noqa: E402

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    print(
        "Pillow est requis : pip install pillow (ou l'ajouter à /etc/nix-darwin/flake.nix)",
        file=sys.stderr,
    )
    raise

#: Ce qu'on tente d'ouvrir. La liste ne fait qu'éviter d'ouvrir un .zip de
#: 3 Go pour rien — un fichier de cette liste qui n'est pas une image est
#: sauté comme les autres, pas une erreur.
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

#: L'octet neutre : `decode_u8(128) ≈ 0`, le milieu de la plage du modèle.
NEUTRAL = 128


def collect_paths(root):
    """Tous les fichiers d'images sous `root`, triés par chemin relatif.

    Le tri est sur le chemin RELATIF : déplacer le dossier ne doit pas changer
    l'ordre, donc pas le fichier produit.
    """
    if os.path.isfile(root):
        return [root]
    found = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if os.path.splitext(name)[1].lower() in SUFFIXES:
                found.append(os.path.join(directory, name))
    return sorted(found, key=lambda p: os.path.relpath(p, root).replace(os.sep, "/"))


def to_square(image, crop):
    """Rend une image carrée, par recadrage centré ou par ajout de marges."""
    width, height = image.size
    if width == height:
        return image

    if crop == "center":
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))

    # fit : l'image entière, centrée dans un carré rempli de gris neutre.
    side = max(width, height)
    fill = (NEUTRAL,) * len(image.getbands())
    canvas = Image.new(image.mode, (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def prepare(image, size, mode, crop):
    """Une image ouverte → les octets de son échantillon."""
    # Le tag EXIF d'abord : recadrer avant de redresser recadrerait le mauvais
    # côté d'une photo prise en portrait.
    image = ImageOps.exif_transpose(image)

    # La transparence sur du noir donne des halos noirs sur tout ce qui vient
    # d'un PNG détouré ; sur le gris neutre elle disparaît dans le fond.
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
        flat = Image.new("RGBA", image.size, (NEUTRAL, NEUTRAL, NEUTRAL, 255))
        image = Image.alpha_composite(flat, image)

    image = image.convert("L" if mode == "grey" else "RGB")
    image = to_square(image, crop)
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    return image.tobytes()


def convert(root, out_path, size, mode, crop, min_size, dedup, version, verbose=True):
    """Convertit le dossier et rend un rapport chiffré de ce qui s'est passé."""
    channels = 1 if mode == "grey" else 3
    sample_bytes = size * size * channels

    payload = bytearray()
    seen = set()
    report = {
        "found": 0,
        "written": 0,
        "unreadable": 0,
        "too_small": 0,
        "duplicates": 0,
        "upscaled": 0,
    }

    paths = collect_paths(root)
    report["found"] = len(paths)
    for path in paths:
        try:
            with Image.open(path) as image:
                # `load()` sous le with : c'est là que Pillow lit vraiment les
                # pixels, et donc là qu'un JPEG tronqué se signale. Ouvrir seul
                # ne lit que l'en-tête et laisserait passer les fichiers coupés.
                image.load()
                width, height = ImageOps.exif_transpose(image).size
                if min(width, height) < min_size:
                    report["too_small"] += 1
                    continue
                if min(width, height) < size:
                    report["upscaled"] += 1
                sample = prepare(image, size, mode, crop)
        except Exception as exc:  # scrape : tout ce qui casse est sauté, compté
            report["unreadable"] += 1
            if verbose:
                print(f"  ignoré {path} : {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        if len(sample) != sample_bytes:
            report["unreadable"] += 1
            continue

        if dedup:
            digest = hashlib.sha256(sample).digest()
            if digest in seen:
                report["duplicates"] += 1
                continue
            seen.add(digest)

        payload.extend(sample)
        report["written"] += 1

    if report["written"] == 0:
        raise ValueError(
            f"aucune image utilisable sous '{root}' "
            f"({report['found']} fichier(s) candidat(s), "
            f"{report['unreadable']} illisible(s), {report['too_small']} trop petite(s))"
        )

    size_on_disk = batraw.write(
        out_path, payload, report["written"], size, size, channels, version=version
    )
    report["bytes"] = size_on_disk
    return report


def contact_sheet(batraw_path, png_path, columns=8, cell=64, rows=8):
    """Une planche des échantillons du dataset produit.

    Les images sont prises **régulièrement réparties** sur tout le dataset, pas
    en tête : les 64 premiers fichiers d'un scrape trié par chemin viennent tous
    du même sous-dossier, et une planche qui ne montre qu'eux ne dit rien du
    reste.
    """
    payload, count, width, height, channels, _ = batraw.read(batraw_path)
    sample_bytes = width * height * channels
    wanted = min(count, columns * rows)

    mode = "L" if channels == 1 else "RGB"
    sheet = Image.new(mode, (columns * cell, ((wanted + columns - 1) // columns) * cell), NEUTRAL)
    for slot in range(wanted):
        # Bornes INCLUSES : la première et la dernière image du dataset sont
        # toujours sur la planche. Avec un simple `slot * count / wanted` la
        # dernière ne l'est jamais — et c'est précisément celle qu'on veut voir,
        # puisque la fin d'un scrape trié par chemin est la partie qu'on a le
        # moins regardée en le constituant.
        index = 0 if wanted == 1 else round(slot * (count - 1) / (wanted - 1))
        chunk = payload[index * sample_bytes : (index + 1) * sample_bytes]
        tile = Image.frombytes(mode, (width, height), bytes(chunk))
        # NEAREST : à cette échelle on veut voir les pixels du dataset, pas une
        # version lissée qui ferait paraître nette une source qui ne l'est pas.
        tile = tile.resize((cell, cell), Image.NEAREST)
        sheet.paste(tile, ((slot % columns) * cell, (slot // columns) * cell))

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    sheet.save(png_path)
    return wanted


def main():
    parser = argparse.ArgumentParser(
        description="Convertit un dossier d'images en dataset .batraw pour batLab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("root", help="dossier d'images, parcouru récursivement")
    parser.add_argument("--out", required=True, help="fichier .batraw à écrire")
    parser.add_argument(
        "--size", type=int, default=32, help="côté de l'image produite (défaut : 32)"
    )
    parser.add_argument(
        "--mode",
        choices=["rgb", "grey"],
        default="rgb",
        help="rgb (3 canaux, défaut) ou grey (1 canal, luminance BT.601)",
    )
    parser.add_argument(
        "--crop",
        choices=["center", "fit"],
        default="center",
        help=(
            "center : plus grand carré centré, l'image est coupée (défaut) ; "
            "fit : image entière, marges en gris neutre"
        ),
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        help=(
            "écarte les images dont le plus petit côté est sous N. "
            "Poser --min-size <size> évite tout agrandissement"
        ),
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="écarte les doublons, comparés sur l'échantillon produit",
    )
    parser.add_argument(
        "--contact-sheet",
        metavar="PNG",
        default=None,
        help="écrit une planche d'échantillons du dataset produit — À FAIRE À CHAQUE FOIS",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="ne pas lister les fichiers sautés"
    )
    batraw.add_format_argument(parser)

    args = parser.parse_args()
    if not os.path.exists(args.root):
        print(f"Erreur : '{args.root}' n'existe pas.", file=sys.stderr)
        sys.exit(1)
    if args.size < 1:
        print("Erreur : --size doit être >= 1.", file=sys.stderr)
        sys.exit(1)

    try:
        report = convert(
            args.root,
            args.out,
            args.size,
            args.mode,
            args.crop,
            args.min_size,
            args.dedup,
            batraw.version_from_choice(args.format),
            verbose=not args.quiet,
        )
    except ValueError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)

    channels = 1 if args.mode == "grey" else 3
    print(
        f"{report['written']} image(s) écrite(s) "
        f"({args.size}×{args.size}×{channels}, {args.format}) → {args.out}"
        f"  [{report['bytes'] / (1024 * 1024):.1f} MiB]"
    )
    print(
        f"  {report['found']} candidat(s) · {report['unreadable']} illisible(s) · "
        f"{report['too_small']} trop petite(s) · {report['duplicates']} doublon(s)"
    )
    if report["upscaled"]:
        print(
            f"  ATTENTION : {report['upscaled']} image(s) agrandie(s) vers {args.size}px — "
            f"envisager --min-size {args.size}"
        )
    if report["written"] < 1000:
        print(
            f"  NOTE : {report['written']} images, c'est sous le seuil (~1000) où le modèle "
            "se met à mémoriser plutôt qu'à généraliser. Voir docs/reports/CUSTOM_DATASET.md."
        )

    if args.contact_sheet:
        shown = contact_sheet(args.out, args.contact_sheet)
        print(f"  planche : {shown} échantillon(s) → {args.contact_sheet}")
    else:
        print("  pas de planche demandée — --contact-sheet <png> est fortement conseillé.")


if __name__ == "__main__":
    main()
