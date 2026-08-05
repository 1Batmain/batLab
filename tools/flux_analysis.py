#!/usr/bin/env python3
"""Reads a `--dump` file from `--headless-perpetual` and measures the three
things the flux regime is judged on: that no frame ever jumps, that the level it
holds does not drift, and whether what is left over is a drift or a shiver.

The dump holds both panes of every frame as raw f32 — see `FrameDump` in
`main/src/main.rs`. That matters: in flux the frame-to-frame change of the x̂₀
pane is of the order of three 8-bit levels, so measuring off PNGs would be
measuring the quantiser (`CLIMB_COHERENCE.md` §6).

    python3 tools/flux_analysis.py stats   flux.bin [--floor flux_seed8.bin]
    python3 tools/flux_analysis.py compare flux.bin wander.bin breathe.bin
    python3 tools/flux_analysis.py plate   flux.bin out.png --start 200 --count 12
    python3 tools/flux_analysis.py plate   flux.bin out.png --count 12 --stride 25

`--floor` is the honest zero of the correlation columns: two runs under
different seeds share the model's idea of a picture, so their frames correlate
without being related at all. A lag whose correlation has reached that number is
decorrelated, whatever the number happens to be.
"""

import struct
import sys

import numpy as np

PHASES = {0: "descente", 1: "remontée", 2: "flux"}
# Levels of the 8-bit image one unit of the [-1, 1] tensors is worth.
LEVELS = 127.5
# What counts as a jolt: a frame whose change is this many times the run's own
# median. Set against the eye, not against a distribution — the complaint the
# regime answers is "it snags between the steps", and a step that moves several
# times the usual amount is exactly what that is.
JOLT = 3.0


def read_dump(path):
    """-> (phases, levels, x_t, x0, (w, h, c)), the tensors as (frames, len)."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:8] != b"BATFLUX1":
        raise SystemExit(f"{path}: not a frame dump")
    w, h, c = struct.unpack("<III", blob[8:20])
    length = w * h * c
    record = 1 + 4 + 2 * length * 4
    frames = (len(blob) - 20) // record
    phases, levels = np.empty(frames, np.uint8), np.empty(frames, np.uint32)
    x_t = np.empty((frames, length), np.float32)
    x0 = np.empty((frames, length), np.float32)
    for i in range(frames):
        base = 20 + i * record
        phases[i] = blob[base]
        (levels[i],) = struct.unpack("<I", blob[base + 1 : base + 5])
        body = np.frombuffer(blob, np.float32, 2 * length, base + 5)
        x_t[i], x0[i] = body[:length], body[length:]
    return phases, levels, x_t, x0, (w, h, c)


def deltas(frames, reduce="mean"):
    """Change between consecutive frames, in 8-bit levels, one number a frame.

    The tensors live in [-1, 1] and the visualiser maps that onto 0..255, so one
    unit here is one grey level of the image actually displayed. `reduce="max"`
    takes the worst pixel of the frame instead of the average one: a handful of
    pixels blinking hard is invisible in a mean and very visible on screen.
    """
    change = np.abs(np.diff(frames, axis=0))
    return (change.max(axis=1) if reduce == "max" else change.mean(axis=1)) * LEVELS


def correlate(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def spread(series):
    """The shape of a |Δ| distribution, as the anti-jump claim needs it read.

    The claim is not about the average frame, it is about the worst one: a
    regime with no turning points has a max of the same order as its median.
    `jolts` counts the frames a viewer would actually catch.
    """
    median = float(np.median(series))
    return {
        "n": len(series),
        "p50": median,
        "p95": float(np.percentile(series, 95)),
        "p99": float(np.percentile(series, 99)),
        "max": float(series.max()),
        "ratio": float(series.max() / median) if median else float("inf"),
        "jolts": int((series > JOLT * median).sum()) if median else len(series),
    }


def msd(frames, tau):
    """Mean squared displacement over `tau` frames."""
    index = np.arange(0, len(frames) - tau, max(1, tau // 2))
    moved = frames[index + tau] - frames[index]
    return float((moved * moved).mean())


def travel(frames):
    """How the walk accumulates: `α` in `MSD(τ) ∝ τ^α`, and what it keeps.

    The two numbers the Ornstein-Uhlenbeck question turns on.

    `α` is fitted on short lags only (τ ≤ 16). Both panes are *stationary* — the
    level of x_t is held and the image stays an image — so their MSD must flatten
    at long lags whatever the short-range behaviour is; fitting through that
    plateau would report confinement as if it were a shiver.

    `kept` is the fraction of one frame's motion still standing 25 frames later,
    against what independent increments would keep. 1.0 is a memoryless walk;
    near 0 is a walk that spends its time undoing itself — motion the eye reads
    as scintillation because the picture goes nowhere.
    """
    short = [(tau, msd(frames, tau)) for tau in (1, 2, 4, 8, 16) if tau < len(frames)]
    if len(short) < 3:
        return float("nan"), float("nan")
    alpha = float(
        np.polyfit(np.log([tau for tau, _ in short]), np.log([value for _, value in short]), 1)[0]
    )
    free = 25 * short[0][1]
    kept = msd(frames, 25) / free if len(frames) > 25 and free else float("nan")
    return alpha, kept


def increment_autocorrelation(frames):
    """Correlation between one frame's step and the next one's.

    Zero means the steps are drawn independently — the walk is memoryless, and
    correlated (Ornstein-Uhlenbeck) noise would be the way to give it memory.
    Negative means consecutive steps undo each other: a pixel pulled one way and
    pushed back, which reads on screen as scintillation.
    """
    steps = np.diff(frames, axis=0)
    if len(steps) < 3:
        return float("nan")
    # A pair where either step is exactly zero is dropped rather than counted:
    # in the cycling regimes x̂₀ is not re-estimated during a climb, so its step
    # is a true zero and its correlation with the next one is undefined. Kept in,
    # a single one of those turns the whole average into a NaN.
    pairs = [
        correlate(steps[i], steps[i + 1])
        for i in range(0, len(steps) - 1, 7)
        if steps[i].any() and steps[i + 1].any()
    ]
    return float(np.mean(pairs)) if pairs else float("nan")


def measure(path, floor_path=None):
    phases, levels, x_t, x0, dims = read_dump(path)
    # Every regime opens on the same walk down from pure noise, and that walk
    # holds the biggest |Δ| any run will ever show. Comparing regimes with it
    # included compares their shared prologue, so it goes.
    settled = np.flatnonzero(phases != 0)
    opening = int(settled[0]) if len(settled) else 0
    phases, levels = phases[opening:], levels[opening:]
    x_t, x0 = x_t[opening:], x0[opening:]

    panes = {"x_t": x_t, "x̂₀": x0}
    changes = {name: deltas(pane) for name, pane in panes.items()}
    worst = {name: deltas(pane, "max") for name, pane in panes.items()}
    # A change is attributed to the phase of the frame it lands on.
    tags = phases[1:]

    result = {
        "path": path,
        "dims": dims,
        "frames": len(phases),
        "opening": opening,
        "levels": (int(levels.min()), int(levels.max())),
        "phases": {},
        "whole": {name: spread(series) for name, series in changes.items()},
        "worst": {name: spread(series) for name, series in worst.items()},
        # Frames that are bit-for-bit the previous one. A pane that holds still
        # for a while and then moves is the shape of a jump, and it is how the
        # cycling regimes treat x̂₀: the model is not called during a climb, so
        # the estimate is stale until the next descent re-establishes it.
        "frozen": {
            name: float((series == 0).mean()) for name, series in changes.items()
        },
    }
    for tag in sorted(set(tags.tolist())):
        keep = tags == tag
        if keep.sum() < 4:
            continue
        result["phases"][PHASES[tag]] = {
            name: spread(series[keep]) for name, series in changes.items()
        }

    # The frames where the run changes its mind — a descent turning into a climb
    # or the other way about — against every other frame. This is the user's
    # complaint stated as a number: "it snags *between the steps*". A regime
    # without phases has no such frame at all, and the split is then empty.
    turn = phases[:-1] != phases[1:]
    result["turns"] = int(turn.sum())
    if turn.any() and (~turn).any():
        result["turning"] = {
            name: {"turn": spread(series[turn]), "within": spread(series[~turn])}
            for name, series in changes.items()
        }

    # The window every stationarity and correlation figure is read on: the flux
    # frames if there are any, the whole settled run otherwise. Its first
    # quarter goes — in flux that is the approach walking down to t*, and in the
    # cycling regimes it is one cycle's worth of warm-up.
    steady = np.flatnonzero(phases == 2)
    if len(steady) < 8:
        steady = np.arange(len(phases))
    window = steady[len(steady) // 4 :]
    result["window"] = (int(window[0]), int(window[-1]), len(window))
    result["corridor"] = {
        name: [float(chunk.mean()) for chunk in np.array_split(pane[steady].std(axis=1), 5)]
        for name, pane in panes.items()
    }
    # The corridor read frame by frame rather than in fifths: a regime built out
    # of phases sweeps its level by design, and averaging fifths of it hides how
    # wide the sweep is.
    result["drift"] = {
        name: (
            float(pane[window].std(axis=1).mean()),
            float(pane[window].std(axis=1).std()),
            float(pane[window].std(axis=1).min()),
            float(pane[window].std(axis=1).max()),
        )
        for name, pane in panes.items()
    }
    result["corr"] = {}
    for lag in (1, 25, 100, 300):
        if len(window) <= lag:
            continue
        result["corr"][lag] = {
            name: float(
                np.mean(
                    [
                        correlate(pane[window[i]], pane[window[i + lag]])
                        for i in range(0, len(window) - lag, 17)
                    ]
                )
            )
            for name, pane in panes.items()
        }
    # How much of the noisy pane is still the picture. The live window shows
    # `x_t | x̂₀` side by side, so this is what says whether the left half reads
    # as an image at all at the level being held — it falls off with `t*` and is
    # the second thing the dial trades.
    result["panes_corr"] = float(
        np.mean([correlate(x_t[i], x0[i]) for i in window[::17]])
    )
    result["travel"] = {name: travel(pane[window]) for name, pane in panes.items()}
    result["increment_corr"] = {
        name: increment_autocorrelation(pane[window]) for name, pane in panes.items()
    }

    if floor_path:
        _, _, other_xt, other_x0, _ = read_dump(floor_path)
        others = {"x_t": other_xt, "x̂₀": other_x0}
        # Frames of a run under another seed, taken from its second half so its
        # own opening descent is out of the way. Nothing links a pair, so their
        # correlation is what "unrelated" is worth on this model.
        result["floor"] = {
            name: float(
                np.mean(
                    [
                        correlate(pane[here], others[name][there])
                        for here, there in zip(
                            window[::17],
                            np.linspace(
                                len(others[name]) // 2,
                                len(others[name]) - 1,
                                len(window[::17]),
                            ).astype(int),
                        )
                    ]
                )
            )
            for name, pane in panes.items()
        }
    return result


def show_spread(label, name, block):
    print(
        f"  {label:<9} |Δ{name}|  n={block['n']:<5} "
        f"médiane {block['p50']:6.2f}  p95 {block['p95']:6.2f}  p99 {block['p99']:6.2f}  "
        f"max {block['max']:7.2f}  max/médiane {block['ratio']:6.2f}  "
        f"secousses(>{JOLT:g}×méd) {block['jolts']}"
    )


def stats(path, floor_path=None):
    result = measure(path, floor_path)
    w, h, c = result["dims"]
    print(
        f"\n=== {path}   {w}×{h}×{c}, {result['frames']} frames "
        f"(prologue de {result['opening']} écarté), t ∈ [{result['levels'][0]}, "
        f"{result['levels'][1]}]"
    )
    for label, block in result["phases"].items():
        for name in ("x_t", "x̂₀"):
            show_spread(label, name, block[name])
    # Whole-run figures: the phase transitions are exactly what a per-phase
    # split hides, and they are the thing the user feels.
    for name in ("x_t", "x̂₀"):
        show_spread("TOUT", name, result["whole"][name])
    for name in ("x_t", "x̂₀"):
        show_spread("PIRE PX", name, result["worst"][name])
    for name in ("x_t", "x̂₀"):
        for label, block in result.get("turning", {}).get(name, {}).items():
            show_spread("↳ " + label, name, block)

    start, end, count = result["window"]
    print(f"  fenêtre de mesure : frames {start}..{end} ({count})")
    for name in ("x_t", "x̂₀"):
        mean, deviation, low, high = result["drift"][name]
        corridor = "  ".join(f"{value:.4f}" for value in result["corridor"][name])
        print(
            f"  couloir std({name})  {mean:.4f} ± {deviation:.4f}  [{low:.4f}, {high:.4f}]"
            f"   par cinquième  {corridor}"
        )
    for name in ("x_t", "x̂₀"):
        if result["frozen"][name]:
            print(f"  {name} figé (|Δ| nul) sur {result['frozen'][name]:6.1%} des frames")
    print(f"  corr des deux panes x_t↔x̂₀   {result['panes_corr']:+.3f}")
    for lag, block in result["corr"].items():
        print(f"  corr décalage {lag:>3}   x_t {block['x_t']:+.3f}   x̂₀ {block['x̂₀']:+.3f}")
    if "floor" in result:
        print(
            f"  plancher (runs indépendants)   x_t {result['floor']['x_t']:+.3f}   "
            f"x̂₀ {result['floor']['x̂₀']:+.3f}"
        )
    for name in ("x_t", "x̂₀"):
        alpha, kept = result["travel"][name]
        print(
            f"  {name}  exposant MSD α {alpha:+.2f}   mouvement conservé à 25 frames "
            f"{kept:.2f}   corr des incréments {result['increment_corr'][name]:+.3f}"
        )
    return result


def compare(paths):
    """The flux-versus-cycles table the report is built on."""
    results = [measure(path) for path in paths]
    names = [
        result["path"].rsplit("/", 1)[-1].removesuffix(".bin") for result in results
    ]
    width = max(15, max(len(name) for name in names) + 2)

    def row(label, values):
        print(f"  {label:<28}" + "".join(f"{value:>{width}}" for value in values))

    print("\n=== anti-saut : distribution de |Δ| par frame, en niveaux de gris 8 bits")
    for name in ("x_t", "x̂₀"):
        print(f"\n  · pane {name}")
        blocks = [result["whole"][name] for result in results]
        row("", names)
        row("médiane", [f"{block['p50']:.2f}" for block in blocks])
        row("p95", [f"{block['p95']:.2f}" for block in blocks])
        row("p99", [f"{block['p99']:.2f}" for block in blocks])
        row("max", [f"{block['max']:.2f}" for block in blocks])
        row("max / médiane", [f"{block['ratio']:.2f}" for block in blocks])
        row(
            f"secousses > {JOLT:g}× médiane",
            [f"{block['jolts']}/{block['n']}" for block in blocks],
        )
        pixels = [result["worst"][name] for result in results]
        row("pire pixel, médiane", [f"{block['p50']:.2f}" for block in pixels])
        row("pire pixel, max", [f"{block['max']:.2f}" for block in pixels])
        turning = [result.get("turning", {}).get(name) for result in results]
        if any(turning):
            # Median *and* max on both sides: a turn is only half the story if
            # the two kinds of turning point differ, and here they do — one
            # freezes the pane and the other unfreezes it with a jolt.
            for side, label in (("within", "hors rebroussement"), ("turn", "AU rebroussement")):
                row(
                    f"{label} (méd / max)",
                    [
                        f"{block[side]['p50']:.2f} / {block[side]['max']:.2f}" if block else "—"
                        for block in turning
                    ],
                )
            row(
                "rebroussements",
                [f"{result['turns']}" if block else "0" for result, block in zip(results, turning)],
            )
        # Per-phase medians say the other half of it: a regime made of phases
        # changes its *rate* at every turning point, and a rate that jumps is
        # felt even when no single frame does.
        for phase in ("descente", "remontée", "flux"):
            present = [result["phases"].get(phase, {}).get(name) for result in results]
            if any(present):
                row(
                    f"médiane en {phase}",
                    [f"{block['p50']:.2f}" if block else "—" for block in present],
                )

    print("\n=== stationnarité et dérive (fenêtre de mesure de chaque run)")
    for name in ("x_t", "x̂₀"):
        row(f"std({name}) moyen", [f"{result['drift'][name][0]:.4f}" for result in results])
        row(
            f"std({name}) min→max par frame",
            [f"{result['drift'][name][2]:.3f}→{result['drift'][name][3]:.3f}" for result in results],
        )
        row(
            f"std({name}) 1er → 5e cinquième",
            [
                f"{result['corridor'][name][0]:.3f}→{result['corridor'][name][-1]:.3f}"
                for result in results
            ],
        )
        row(f"{name} figé", [f"{result['frozen'][name]:.1%}" for result in results])
    row("corr des panes x_t↔x̂₀", [f"{result['panes_corr']:+.3f}" for result in results])
    for lag in (1, 25, 100, 300):
        for name in ("x_t", "x̂₀"):
            if all(lag in result["corr"] for result in results):
                row(
                    f"corr {name} décalage {lag}",
                    [f"{result['corr'][lag][name]:+.3f}" for result in results],
                )
    for name in ("x_t", "x̂₀"):
        row(f"exposant MSD α {name}", [f"{result['travel'][name][0]:+.2f}" for result in results])
        row(
            f"conservé à 25 frames {name}",
            [f"{result['travel'][name][1]:.2f}" for result in results],
        )
        row(
            f"corr des incréments {name}",
            [f"{result['increment_corr'][name]:+.3f}" for result in results],
        )


def plate(path, out, start, count, stride, pane, phase):
    from PIL import Image

    phases, levels, x_t, x0, (w, h, c) = read_dump(path)
    source = x_t if pane == "xt" else x0
    wanted = {"descente": 0, "remontée": 1, "flux": 2}.get(phase)
    if wanted is None:
        eligible = np.arange(len(phases))
    else:
        eligible = np.flatnonzero(phases == wanted)
        if not len(eligible):
            raise SystemExit(f"{path}: aucune frame en phase {phase}")
    picks = eligible[start : start + count * stride : stride]
    scale = 4
    sheet = Image.new("L", (len(picks) * (w * scale + 2), h * scale), 40)
    for i, index in enumerate(picks):
        tile = np.clip((source[index].reshape(h, w) + 1) * LEVELS, 0, 255).astype(np.uint8)
        image = Image.fromarray(tile).resize((w * scale, h * scale), Image.NEAREST)
        sheet.paste(image, (i * (w * scale + 2), 0))
    sheet.save(out)
    print(
        f"{out}: {len(picks)} vignettes, pane {pane}, frames {picks[0]}..{picks[-1]} "
        f"(t={levels[picks[0]]}..{levels[picks[-1]]}, pas de {stride})"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    command, args = sys.argv[1], sys.argv[2:]

    def flag(name, default):
        return type(default)(args[args.index(name) + 1]) if name in args else default

    positional = []
    skip = False
    for i, value in enumerate(args):
        if skip:
            skip = False
        elif value.startswith("--"):
            skip = True
        else:
            positional.append(value)

    if command == "stats":
        floor = flag("--floor", "") or None
        for path in positional:
            stats(path, floor)
    elif command == "compare":
        compare(positional)
    elif command == "plate":
        plate(
            positional[0],
            positional[1],
            flag("--start", 0),
            flag("--count", 12),
            flag("--stride", 1),
            flag("--pane", "xt"),
            flag("--phase", "flux"),
        )
    else:
        raise SystemExit(__doc__)
