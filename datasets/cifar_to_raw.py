#!/usr/bin/env python3
"""Convert CIFAR-10 binary batches to the batLab raw dataset format.

The output `.batraw` file can be loaded directly by the Rust training crate
without any image-library dependencies. All dataset-specific parsing logic lives
here so the Rust side only needs to read a flat array of pixel values.

The file layout, the three payload encodings and the reasons for them live in
`tools/batraw.py`, which this script imports — one definition of the format, read
by every converter.

**BATRAW3 is the default**, and for CIFAR-10 it is exactly lossless in RGB: the
archive holds one byte per channel, and that byte is what gets written.

Greyscale is the one place a byte is lost: BT.601 luminance of three bytes is
not a byte, and it is now rounded to one. That applies to **every** format this
script writes, including `--format batraw2` — the pipeline's common form is
bytes, so quantising once here rather than differently per format is what keeps
the three outputs describing the same images. A `cifar10_grey.batraw` produced
today therefore differs from one produced before BATRAW3 by at most 1/255 per
pixel, on a payload the model reads in [-1, 1]. Existing files keep loading
unchanged; nothing needs reconverting.

Usage
-----
    # Toutes les batches d'entraînement en niveaux de gris (défaut) :
    python cifar_to_raw.py --cifar-dir cifar-10-batches-bin

    # RGB, le cas courant :
    python cifar_to_raw.py --cifar-dir cifar-10-batches-bin --mode rgb

    # L'ancien format f32, si un outil tiers en dépend :
    python cifar_to_raw.py --cifar-dir cifar-10-batches-bin --format batraw2

    # Ajouter la batch de test et nommer la sortie :
    python cifar_to_raw.py --cifar-dir cifar-10-batches-bin --include-test --out cifar_rgb.batraw

Le script ne demande aucune bibliothèque tierce — numpy est utilisé s'il est là,
seulement pour aller vite.
"""

import argparse
import os
import sys

# Le format vit dans tools/, à côté des deux autres convertisseurs. Ce script
# est dans datasets/ depuis toujours (les chemins de la doc et des bancs le
# citent), d'où l'ajout de chemin plutôt qu'un déplacement.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import batraw  # noqa: E402

try:  # chemin rapide : 50000×32×32×3 en pur Python coûte des minutes
    import numpy as _np
except ImportError:
    _np = None

CIFAR_RECORD_BYTES = 3073  # 1 label byte + 3*1024 channel bytes
CIFAR_WIDTH = 32
CIFAR_HEIGHT = 32
CIFAR_RGB_CHANNELS = 3
CIFAR_PIXELS = CIFAR_WIDTH * CIFAR_HEIGHT


def _luminance(r: int, g: int, b: int) -> float:
    """Convert an sRGB pixel to a greyscale value in [0, 255] using BT.601 weights."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def _parse_cifar_batch(data: bytes, mode: str) -> bytearray:
    """Parse a CIFAR-10 binary batch into a flat `bytearray` of pixel bytes.

    Everything downstream takes bytes, whatever the output format: the archive
    is 8-bit, so bytes are the common form all three encodings agree on.
    """
    if len(data) % CIFAR_RECORD_BYTES != 0:
        raise ValueError(
            f"Unexpected batch size: {len(data)} bytes "
            f"(not a multiple of {CIFAR_RECORD_BYTES})"
        )

    out = bytearray()
    for record in _chunks(data, CIFAR_RECORD_BYTES):
        # record[0] is the class label – not used for unconditional diffusion.
        payload = record[1:]  # 3072 bytes: R plane, G plane, B plane (1024 each)
        r_plane = payload[0:1024]
        g_plane = payload[1024:2048]
        b_plane = payload[2048:3072]

        if mode == "grey":
            out.extend(
                round(_luminance(r_plane[i], g_plane[i], b_plane[i]))
                for i in range(CIFAR_PIXELS)
            )
        else:  # rgb : plan → pixel, canaux contigus
            for i in range(CIFAR_PIXELS):
                out.append(r_plane[i])
                out.append(g_plane[i])
                out.append(b_plane[i])

    return out


def _parse_cifar_batch_np(data: bytes, mode: str):
    """Version vectorisée de `_parse_cifar_batch`, même résultat."""
    if len(data) % CIFAR_RECORD_BYTES != 0:
        raise ValueError(
            f"Unexpected batch size: {len(data)} bytes "
            f"(not a multiple of {CIFAR_RECORD_BYTES})"
        )

    records = _np.frombuffer(data, dtype=_np.uint8).reshape(-1, CIFAR_RECORD_BYTES)
    # record[0] = label (inutilisé) ; puis 3 plans de 1024 octets (R, G, B)
    planes = records[:, 1:].reshape(-1, CIFAR_RGB_CHANNELS, CIFAR_PIXELS)

    if mode == "grey":
        lum = (
            0.299 * planes[:, 0].astype(_np.float64)
            + 0.587 * planes[:, 1].astype(_np.float64)
            + 0.114 * planes[:, 2].astype(_np.float64)
        )
        # `round()` de Python, comme le chemin lent : arrondi au pair le plus
        # proche. Les deux chemins doivent rendre le MÊME fichier.
        out = _np.rint(lum).astype(_np.uint8)
    else:  # rgb : entrelacement plan → pixel
        out = planes.transpose(0, 2, 1).reshape(-1, CIFAR_PIXELS * CIFAR_RGB_CHANNELS)

    return _np.ascontiguousarray(out, dtype=_np.uint8)


def _chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def _collect_batch_files(cifar_dir: str, include_test: bool) -> list:
    """Return sorted training batch paths, optionally including the test batch."""
    batch_files = sorted(
        os.path.join(cifar_dir, name)
        for name in os.listdir(cifar_dir)
        if name.startswith("data_batch_") and name.endswith(".bin")
    )

    if include_test:
        test_path = os.path.join(cifar_dir, "test_batch.bin")
        if os.path.isfile(test_path):
            batch_files.append(test_path)

    if not batch_files:
        raise FileNotFoundError(
            f"No CIFAR batch files found in '{cifar_dir}'. "
            "Make sure the directory contains data_batch_*.bin files."
        )

    return batch_files


def convert(cifar_dir: str, out_path: str, mode: str, include_test: bool, version: int) -> None:
    """Convert CIFAR-10 batches to a .batraw file."""
    batch_files = _collect_batch_files(cifar_dir, include_test)
    channels = 1 if mode == "grey" else CIFAR_RGB_CHANNELS

    payload = bytearray()
    for batch_file in batch_files:
        print(f"  Reading {batch_file} …", flush=True)
        with open(batch_file, "rb") as fh:
            data = fh.read()
        if _np is not None:
            payload.extend(_parse_cifar_batch_np(data, mode).tobytes())
        else:
            payload.extend(_parse_cifar_batch(data, mode))

    sample_bytes = CIFAR_WIDTH * CIFAR_HEIGHT * channels
    count = len(payload) // sample_bytes

    print(
        f"  Writing {count} samples ({CIFAR_WIDTH}×{CIFAR_HEIGHT}×{channels}, "
        f"{batraw.MAGIC[version][:7].decode()}) → {out_path}",
        flush=True,
    )
    size = batraw.write(
        out_path, payload, count, CIFAR_WIDTH, CIFAR_HEIGHT, channels, version=version
    )
    print(f"  Done – {size / (1024 * 1024):.1f} MiB written to {out_path}")


def _default_output_name(mode: str) -> str:
    return f"cifar10_{mode}.batraw"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CIFAR-10 binary batches to the batLab .batraw format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cifar-dir",
        default="cifar-10-batches-bin",
        help="Path to the cifar-10-batches-bin directory (default: ./cifar-10-batches-bin)",
    )
    parser.add_argument(
        "--mode",
        choices=["grey", "rgb"],
        default="grey",
        help=(
            "Output colour mode: 'rgb' (3 channels, byte-exact) or 'grey' "
            "(1 channel, BT.601 luminance ROUNDED to a byte — the archive has no "
            "greyscale plane, so this one is a quantisation, ±1/255). Default: grey"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file path. Defaults to cifar10_<mode>.batraw in the current directory.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also include test_batch.bin in addition to the training batches.",
    )
    batraw.add_format_argument(parser)

    args = parser.parse_args()

    cifar_dir = args.cifar_dir
    if not os.path.isdir(cifar_dir):
        print(f"Error: '{cifar_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or _default_output_name(args.mode)

    try:
        convert(
            cifar_dir,
            out_path,
            args.mode,
            args.include_test,
            batraw.version_from_choice(args.format),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
