#!/usr/bin/env python3
"""Extrait un petit sous-ensemble d'images d'un `.batraw` en `seeds.bin`.

La page web a besoin d'images de départ pour la dérive (« errance »), mais
embarquer les 3,99 Mo du dataset complet sur une page portfolio serait absurde :
quelques dizaines d'images suffisent. Ce script lit un fichier `.batraw`
(BATRAW3, payload u8) et en écrit les `--count` premières dans le format compact
que `SeedImages::parse` attend côté Rust (`crates/batlab-web/src/lib.rs`) :

    Offset  Taille  Type    Description
    ------  ------  ----    -----------
    0       4 o     u32le   nombre d'images (count)
    4       4 o     u32le   largeur (W)
    8       4 o     u32le   hauteur (H)
    12      4 o     u32le   canaux (C)
    16      …       u8      count*W*H*C octets, canaux entrelacés (R,G,B)

Le payload u8 est recopié tel quel : c'est exactement celui d'un `.batraw`, que
le moteur re-décode en [-1, 1] par `decode_u8` — l'image de départ du navigateur
est donc au bit près celle dont partirait la dérive native.

L'en-tête `.batraw` (24 octets) est défini une seule fois dans `tools/batraw.py`
et lu ici sans le réécrire.
"""

import argparse
import struct
import sys
from pathlib import Path

# tools/batraw.py vit à la racine du dépôt, deux niveaux au-dessus de ce fichier
# (web/tools/make_seeds.py). On l'importe pour ne pas recopier l'en-tête.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import batraw  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="le fichier .batraw source")
    parser.add_argument("output", type=Path, help="le seeds.bin à écrire")
    parser.add_argument(
        "--count",
        type=int,
        default=32,
        help="nombre d'images à embarquer (défaut 32)",
    )
    args = parser.parse_args()

    raw = args.source.read_bytes()
    if len(raw) < batraw.HEADER_BYTES:
        print(f"{args.source}: trop court pour un en-tête .batraw", file=sys.stderr)
        return 1

    magic = raw[:8]
    version = next((v for v, m in batraw.MAGIC.items() if m == magic), None)
    if version is None:
        print(f"{args.source}: magic inconnu {magic!r}", file=sys.stderr)
        return 1
    if version != 3:
        print(
            f"{args.source}: seul BATRAW3 (u8) est géré ici, pas BATRAW{version}",
            file=sys.stderr,
        )
        return 1

    n, w, h, c = struct.unpack_from("<IIII", raw, 8)
    per_image = w * h * c
    payload = raw[batraw.HEADER_BYTES :]
    available = len(payload) // per_image
    count = min(args.count, available, n)
    if count <= 0:
        print(f"{args.source}: aucune image à extraire", file=sys.stderr)
        return 1

    subset = payload[: count * per_image]
    header = struct.pack("<IIII", count, w, h, c)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + subset)

    size_kb = (len(header) + len(subset)) / 1024
    print(
        f"{args.output}: {count} images {w}x{h}x{c} depuis {args.source.name} "
        f"({size_kb:.1f} Kio)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
