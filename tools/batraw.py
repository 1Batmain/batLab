#!/usr/bin/env python3
"""Le format `.batraw`, en un seul endroit.

Trois convertisseurs écrivent ce format (`datasets/cifar_to_raw.py`,
`tools/images_to_raw.py`, `tools/imagenet32_to_raw.py`) et le moteur Rust le lit
(`try_load_raw_dataset` dans `crates/batlab/src/main.rs`). Une définition
recopiée dans chaque script, c'est trois occasions de diverger sur un octet
d'en-tête — donc elle est ici, et les trois l'importent.

En-tête, identique aux trois versions
-------------------------------------
Offset  Taille  Type    Description
------  ------  ----    -----------
0       8 o     bytes   magic (voir plus bas)
8       4 o     u32le   nombre d'images (N)
12      4 o     u32le   largeur (W)
16      4 o     u32le   hauteur (H)
20      4 o     u32le   canaux par pixel (C) — 1 en niveaux de gris, 3 en RGB
24      …               payload, N*W*H*C valeurs, ligne par ligne, canaux
                        contigus (R,G,B pour un pixel, puis pixel suivant)

Les trois payloads
------------------
- ``BATRAW3`` — **u8** dans [0, 255], élargi en [-1, 1] **sur le GPU**. C'est le
  défaut. Toutes les images de ce projet sont 8 bits à la source : les stocker en
  f32 quadruplait le fichier, la RAM hôte et surtout les transferts hôte→GPU,
  pour des valeurs dont trois octets sur quatre étaient déductibles du premier.
- ``BATRAW2`` — f32 déjà en [-1, 1].
- ``BATRAW1`` — f32 en [0, 1], rééchelonné au chargement. Historique.

L'élargissement ``v/127.5 - 1`` est **exact** : 256 entrées, 256 f32, aucun
arrondi. Un fichier BATRAW3 et le BATRAW2 issu des mêmes images se décodent donc
au bit près (`an_8_bit_file_decodes_to_the_same_bits_as_the_f32_one_it_replaces`).

Une seule dépendance optionnelle : numpy, et seulement pour accélérer les
formats f32. Écrire du BATRAW3 est une concaténation d'octets — c'est aussi
pourquoi c'est le format qui rend ImageNet 32×32 praticable.
"""

import os
import struct

try:
    import numpy as _np
except ImportError:  # pragma: no cover - le chemin lent reste correct
    _np = None

HEADER_BYTES = 24

MAGIC = {
    1: b"BATRAW1\0",
    2: b"BATRAW2\0",
    3: b"BATRAW3\0",
}

#: Octets par valeur, par version — le nombre dont dépend toute la taille.
VALUE_BYTES = {1: 4, 2: 4, 3: 1}

DEFAULT_VERSION = 3

VERSION_CHOICES = ["batraw1", "batraw2", "batraw3"]


def version_from_choice(choice):
    """``"batraw3"`` → ``3``. Le nom que porte l'option ``--format``."""
    return int(str(choice).lower().removeprefix("batraw"))


def decode_u8(value):
    """L'élargissement [0, 255] → [-1, 1], le même que `decode_u8` côté Rust."""
    return value / 127.5 - 1.0


def encode_u8(value):
    """Son inverse, pour relire un f32 vers l'octet dont il vient."""
    return int(round((min(max(value, -1.0), 1.0) + 1.0) * 127.5))


def _payload(samples_u8, version):
    """Le payload encodé, depuis des octets.

    `samples_u8` est un `bytes`/`bytearray` plat, ou un tableau numpy uint8 —
    dans les deux cas N*W*H*C octets, images à la suite.
    """
    if version == 3:
        if _np is not None and hasattr(samples_u8, "tobytes"):
            return samples_u8.tobytes()
        return bytes(samples_u8)

    if _np is not None:
        flat = _np.frombuffer(bytes(samples_u8), dtype=_np.uint8)
        if version == 2:
            # En float64 puis abaissé, pour tomber sur les mêmes octets que
            # `struct.pack('<f', v/127.5 - 1)` du chemin pur Python.
            values = (flat.astype(_np.float64) / 127.5 - 1.0).astype(_np.float32)
        else:
            values = (flat.astype(_np.float64) / 255.0).astype(_np.float32)
        return values.tobytes()

    convert = decode_u8 if version == 2 else (lambda v: v / 255.0)
    flat = bytes(samples_u8)
    return struct.pack(f"<{len(flat)}f", *(convert(v) for v in flat))


def write(path, samples_u8, count, width, height, channels, version=DEFAULT_VERSION):
    """Écrit un `.batraw`. `samples_u8` est toujours en octets, quel que soit le format.

    Prendre l'entrée en 8 bits même pour BATRAW1/2 n'est pas une restriction : la
    source EST 8 bits, et c'est ce qui garantit que les trois versions décrivent
    exactement les mêmes images.
    """
    if version not in MAGIC:
        raise ValueError(f"version .batraw inconnue : {version}")
    expected = count * width * height * channels
    if len(samples_u8) != expected:
        raise ValueError(
            f"payload de {len(samples_u8)} octets pour {count}×{width}×{height}×{channels} "
            f"= {expected} attendus"
        )

    with open(path, "wb") as fh:
        fh.write(MAGIC[version])
        fh.write(struct.pack("<IIII", count, width, height, channels))
        fh.write(_payload(samples_u8, version))

    return os.path.getsize(path)


def read(path):
    """Relit un `.batraw` → ``(octets, count, width, height, channels, version)``.

    Rend le payload **en octets** quelle que soit la version : c'est la forme
    commune, et pour BATRAW1/2 elle passe par `encode_u8`, l'inverse exact de ce
    que `write` a fait. Un aller-retour write→read rend donc les octets de
    départ, sur les trois versions — ce que vérifient les tests.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path} : trop court pour un en-tête .batraw")
    magic = raw[:8]
    version = next((v for v, m in MAGIC.items() if m == magic), None)
    if version is None:
        raise ValueError(f"{path} : magic inconnu {magic!r}")

    count, width, height, channels = struct.unpack("<IIII", raw[8:HEADER_BYTES])
    values = count * width * height * channels
    payload = raw[HEADER_BYTES:]
    if len(payload) != values * VALUE_BYTES[version]:
        raise ValueError(
            f"{path} : payload de {len(payload)} octets, "
            f"{values * VALUE_BYTES[version]} attendus"
        )

    if version == 3:
        return payload, count, width, height, channels, version

    if _np is not None:
        floats = _np.frombuffer(payload, dtype="<f4")
        if version == 1:
            floats = floats * 2.0 - 1.0
        bytes_out = _np.rint((_np.clip(floats, -1.0, 1.0) + 1.0) * 127.5).astype(_np.uint8)
        return bytes_out.tobytes(), count, width, height, channels, version

    floats = struct.unpack(f"<{values}f", payload)
    if version == 1:
        floats = [v * 2.0 - 1.0 for v in floats]
    return bytes(encode_u8(v) for v in floats), count, width, height, channels, version


def add_format_argument(parser):
    """L'option `--format`, écrite une fois pour les trois convertisseurs."""
    parser.add_argument(
        "--format",
        choices=VERSION_CHOICES,
        default="batraw3",
        help=(
            "encodage du payload : batraw3 (u8, défaut — quatre fois plus petit "
            "sur disque ET en transfert GPU), batraw2 (f32 en [-1,1]), "
            "batraw1 (f32 en [0,1], historique)"
        ),
    )
