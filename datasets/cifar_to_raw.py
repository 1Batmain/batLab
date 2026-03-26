#!/usr/bin/env python3
"""Convert CIFAR-10 binary batches to the batLab raw dataset format.

The output `.batraw` file can be loaded directly by the Rust training crate without
any image-library dependencies.  All dataset-specific parsing logic lives here so the
Rust side only needs to read a flat array of normalised f32 values.

Raw file layout
---------------
Offset  Size   Type    Description
------  ----   ----    -----------
0       8 B    bytes   Magic: b"BATRAW1\\0"
8       4 B    u32le   Number of samples (N)
12      4 B    u32le   Image width  (W)
16      4 B    u32le   Image height (H)
20      4 B    u32le   Channels per pixel (C)  – 1 for greyscale, 3 for RGB
24      N*W*H*C*4 B    f32le   Pixel values normalised to [0, 1], stored in
                               row-major order; for RGB the channel order is R,G,B.

Usage
-----
    # Convert all training batches to greyscale (default):
    python cifar_to_raw.py --cifar-dir cifar-10-batches-bin

    # Explicit greyscale output:
    python cifar_to_raw.py --cifar-dir cifar-10-batches-bin --mode grey

    # RGB output:
    python cifar_to_raw.py --cifar-dir cifar-10-batches-bin --mode rgb

    # Include the test batch and specify an output file name:
    python cifar_to_raw.py --cifar-dir cifar-10-batches-bin --include-test --out cifar_rgb.batraw

The script requires no third-party libraries – only the Python standard library.
"""

import argparse
import struct
import os
import sys

MAGIC = b"BATRAW1\0"
CIFAR_RECORD_BYTES = 3073  # 1 label byte + 3*1024 channel bytes
CIFAR_WIDTH = 32
CIFAR_HEIGHT = 32
CIFAR_RGB_CHANNELS = 3
CIFAR_PIXELS = CIFAR_WIDTH * CIFAR_HEIGHT


def _luminance(r: int, g: int, b: int) -> float:
    """Convert an sRGB pixel to a normalised greyscale value using BT.601 weights."""
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def _parse_cifar_batch(data: bytes, mode: str) -> list:
    """Parse a CIFAR-10 binary batch and return a list of flat f32 sample lists."""
    if len(data) % CIFAR_RECORD_BYTES != 0:
        raise ValueError(
            f"Unexpected batch size: {len(data)} bytes "
            f"(not a multiple of {CIFAR_RECORD_BYTES})"
        )

    samples = []
    for record in _chunks(data, CIFAR_RECORD_BYTES):
        # record[0] is the class label – not used for unconditional diffusion.
        payload = record[1:]  # 3072 bytes: R plane, G plane, B plane (1024 each)
        r_plane = payload[0:1024]
        g_plane = payload[1024:2048]
        b_plane = payload[2048:3072]

        if mode == "grey":
            floats = [
                _luminance(r_plane[i], g_plane[i], b_plane[i])
                for i in range(CIFAR_PIXELS)
            ]
        else:  # rgb
            floats = []
            for i in range(CIFAR_PIXELS):
                floats.append(r_plane[i] / 255.0)
                floats.append(g_plane[i] / 255.0)
                floats.append(b_plane[i] / 255.0)

        samples.append(floats)

    return samples


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


def convert(cifar_dir: str, out_path: str, mode: str, include_test: bool) -> None:
    """Convert CIFAR-10 batches to a .batraw file."""
    batch_files = _collect_batch_files(cifar_dir, include_test)
    channels = 1 if mode == "grey" else CIFAR_RGB_CHANNELS

    all_samples: list = []
    for batch_file in batch_files:
        print(f"  Reading {batch_file} …", flush=True)
        with open(batch_file, "rb") as fh:
            data = fh.read()
        all_samples.extend(_parse_cifar_batch(data, mode))

    count = len(all_samples)
    sample_floats = CIFAR_WIDTH * CIFAR_HEIGHT * channels

    print(
        f"  Writing {count} samples ({CIFAR_WIDTH}×{CIFAR_HEIGHT}×{channels}) → {out_path}",
        flush=True,
    )

    with open(out_path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<IIII", count, CIFAR_WIDTH, CIFAR_HEIGHT, channels))
        for sample in all_samples:
            fh.write(struct.pack(f"<{sample_floats}f", *sample))

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Done – {size_mb:.1f} MiB written to {out_path}")


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
        help="Output colour mode: 'grey' (1 channel, BT.601) or 'rgb' (3 channels). Default: grey",
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

    args = parser.parse_args()

    cifar_dir = args.cifar_dir
    if not os.path.isdir(cifar_dir):
        print(f"Error: '{cifar_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or _default_output_name(args.mode)

    try:
        convert(cifar_dir, out_path, args.mode, args.include_test)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
