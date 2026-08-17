//! File purpose: Application entry point that orchestrates training, inference, and TUI workflows.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, SystemTime};

use batlab_ui::storage::{self, SeedSource};
use batlab_ui::tui::{
    self, LayerDraft, ModelConfig, MonitorOutcome, PerpetualConfig,
    RunMode, TrainingConfig,
};
use batlab_core::{
    CheckpointWeights, DEFAULT_SNR_GAMMA, DatasetPayload, DenoiseFrame, DiffusionTask, DriftAction,
    DriftWalk, EmaConfig, EvalConfig, EvalReport, GpuContext, GpuDataset, GpuLimitsProfile,
    LinearNoiseSchedule, LiveFrame,
    LossMethod as PLoss, LossWeighting, MetricsLogger, Model, OptimizerKind, PerpetualDrift,
    PosteriorVariance, ProbeConfig, Stats, Trainer, WeightInit, compose_live_frame_view, evaluate,
    log_probe,
    log_train_loss, log_trajectory, model::Training, probe_diffusion, sample_diffusion,
};
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};

// Magic headers for the raw binary dataset format produced by the pre-processing
// scripts. Format: magic(8) | count | width | height | channels | f32 data…
/// Legacy raw-dataset magic — payload stored in `[0, 1]`. Converted to the
/// model's `[-1, 1]` convention at load time.
const RAW_DATASET_MAGIC_UNIT: &[u8; 8] = b"BATRAW1\0";
/// Legacy raw-dataset magic — f32 payload already stored in `[-1, 1]`.
const RAW_DATASET_MAGIC_SIGNED: &[u8; 8] = b"BATRAW2\0";
/// Current raw-dataset magic — **u8** payload in `[0, 255]`, widened to
/// `[-1, 1]` on the GPU. A quarter of the file, a quarter of the host RAM and a
/// quarter of the chunk traffic, for exactly the same values: the sources are
/// 8-bit, so the f32 encoding stored four bytes of which three were always
/// derivable from the first.
const RAW_DATASET_MAGIC_BYTES: &[u8; 8] = b"BATRAW3\0";

// The schedule constants now live in the engine (one definition, shared with
// the web build); these aliases keep every call site below unchanged.
const DIFFUSION_SCHEDULE_STEPS: usize = batlab_core::DIFFUSION_SCHEDULE_STEPS;
const DIFFUSION_BETA_START: f32 = batlab_core::DIFFUSION_BETA_START;
const DIFFUSION_BETA_END: f32 = batlab_core::DIFFUSION_BETA_END;
const LOSS_REPORT_INTERVAL_STEPS: usize = 25;
const INFERENCE_RUNTIME_LR: f32 = 0.01;
const INFERENCE_RUNTIME_BATCH_SIZE: u32 = 1;
/// What `--help` prints. The headless entry points are the whole scriptable
/// surface of this binary, so this text *is* their contract: an agent, a CI job
/// or a black-box tester has nothing else to read, and a flag that is not here
/// does not exist. The dump layout is spelled out for the same reason — a reader
/// written from a prose description of "raw f32" reads it shifted by five bytes.
const HELP: &str = "\
batlab — deep-learning framework (Rust + wgpu). Run with no arguments for
the interactive TUI. The flags below are the DEV/CI headless entry points; they
are not reachable from the TUI and never write back a model's config_file.

  --headless-train <model> --steps N --dataset <path>
      [--lr F] [--batch N] [--out <ckpt>] [--optimizer sgd|adam]
      [--weight-init uniform|he] [--loss-weighting uniform|snr] [--snr-gamma F]
      [--ema F] [--resume <ckpt>] [--checkpoint-every N]

  --headless-sample <model> [--checkpoint <path>] [--seed N] [--paths N]
      [--magnitude F] [--variance beta|posterior] [--out <img>] [--log <jsonl>] [--raw-weights]

  --headless-perpetual <model> [--checkpoint <path>] [--regime wander|breathe|flux]
      [--t-r K | --t-star K | --depth K] [--seed N] [--magnitude F] [--dump <path>]
      [--frames N] [--actions N] [--window] [--climb-frames N] [--out <dir>]
      [--raw-weights]

  --eval <model> --ckpt <path> [--ckpt <path> ...] --dataset <path>
      [--samples N] [--buckets N] [--t-per-bucket N] [--seed N] [--raw-weights]

  --export-weights <model> --ckpt <path> --out <path> [--raw-weights] [--quantize]

  --resources <model> [--batch N] [--dataset <path>] [--measure] [--steps N]
      [--inference] [--vram GiB] [--device <name>] [--no-gpu] [--width N]

  --profile-step <model> --dataset <path> [--batch N] [--optimizer sgd|adam]
      [--lr F] [--rounds N] [--warmup N] [--top N] [--ema]

Where a training step's time actually goes:
  --profile-step    run ONE training step and report the GPU time of EVERY
                    compute pass it encodes, named by layer and entry point,
                    sorted by cost. The numbers come from the GPU's own clock
                    (`TIMESTAMP_QUERY` written at each pass's beginning and
                    end), not from host timing around a submit.
                    Under the table is the budget, and the budget is the point:
                    the sum of the passes, the GPU SPAN of the whole step (last
                    end minus first begin, so barriers and pass setup are inside
                    it), and the host wall clock. Span minus sum is what the step
                    spends BETWEEN its passes — the per-pass floor that
                    PERF_CONVOLUTION.md §5.4 hypothesised and could not measure.
                    Runs the same production call the trainer runs and nothing
                    else: no loss readback, no probe, no periodic sampling, all
                    of which submit GPU work of their own and would be profiled
                    as if they were part of a step.
  --rounds N        armed steps (default 5). The estimator is the MINIMUM over
                    them, the same rule the rest of this project's benchmarks
                    use: contention can only add time.
  --warmup N        unarmed steps first (default 3). On Metal the first dispatch
                    of a pipeline pays its compilation, which would otherwise be
                    charged to whichever pass ran first.
  --top N           show only the N most expensive passes; the rest are summed
                    into one line rather than dropped.
  --ema             also encode the weight-average passes, as `--ema` does.
                    Requires a device with TIMESTAMP_QUERY; without one the
                    command says so and stops rather than substitute host-side
                    estimates for GPU measurements.

What the model costs on the GPU:
  --resources       print the itemised GPU inventory: weights, gradients, Adam
                    moments, EMA, activations, attention's N² scratch, the
                    diffusion prepass, the resident dataset chunk — each against
                    this machine's real limits, with the batch ceiling that
                    follows. Read-only: it opens an adapter to read its limits
                    and writes nothing anywhere. The TUI shows the same page on
                    `[R]` from a model's action menu.
  --batch N         size the inventory for N samples instead of the config's
                    batch. The activation posts scale with it, the parameter
                    posts do not, and the page says which is which.
  --dataset <path>  the dataset to account for; defaults to the one the model's
                    config trains on. Only its header is read.
  --measure         also RUN: build the model, take a few training steps and a
                    few reverse steps, and report the counters' difference —
                    bytes up, bytes down, round trips and submissions per step.
                    Without it no traffic table is printed at all, rather than a
                    plausible zero. Needs a real GPU.
  --steps N         steps to measure (default 5). The first is a warmup and is
                    excluded: it uploads the first dataset chunk.
  --inference       size the inference graph (forward only, batch 1) instead of
                    the training one.
  --vram GiB        state a memory budget — 'would this fit on an 8 GiB card?'.
                    No API reports one, so without this flag the verdict is
                    about per-binding limits only.
  --device <name>   answer for a machine that is not here, at the WebGPU default
                    limits (128 MiB storage bindings, 256 MiB buffers) — which
                    is what a browser grants the visitor's GPU.
  --no-gpu          same, unnamed: never opens an adapter.

How good a checkpoint actually is (architecture arbitration):
  --eval            score one or more checkpoints on a HELD-OUT, DETERMINISTIC
                    slice — the same images, timesteps and noise fields every
                    time and for every checkpoint, so a difference in the numbers
                    is a difference in the weights and nothing else. Reports
                    MSE(ε̂, ε) per timestep bucket AND the same error carried into
                    image space (x₀-MSE = (1-ᾱ)/ᾱ · ε-MSE), because ε-MSE alone
                    inverts the schedule's importance: at high t a copy of the
                    input is already near-exact. Each row is set against the three
                    zero-parameter baselines of tools/trivial_baselines.py
                    (ε̂=0, ε̂=x_t, ε̂=mean) — a model that does not beat ε̂=mean in
                    x₀ has only learned the dataset average.
  --ckpt <path>     a checkpoint to score. Repeatable: pass several to get one
                    comparison table across them (this is how two architectures
                    are compared after paired training).
  --dataset <path>  the .batraw the held-out images and the mean image come from.
  --samples N       held-out images (default 256), taken from the TAIL of the
                    dataset; the mean image is built from the head, so the two do
                    not overlap.
  --buckets N       timestep buckets the schedule is split into (default 4).
  --t-per-bucket N  timesteps drawn inside each bucket (default 4).
  --seed N          base seed for the noise draws (default 7). Fixed across
                    checkpoints — that is what makes them comparable.
  --raw-weights     score the last iterate; by default the average is used when
                    the checkpoint carries one, exactly as sampling does.

Which GPU limits the run asks for:
  --gpu-limits native|web
                    `request_device` hands out what you ASK for, and asking for
                    nothing means the WebGPU baseline: 256 MiB per buffer,
                    128 MiB per storage binding. That is right for inference —
                    the engine's target is the visitor's browser — and wrong for
                    training on this machine, whose adapter allows 28 GiB.
                    native (DEFAULT) asks the adapter for its own limits: a
                    higher batch ceiling, and a dataset that fits stays RESIDENT
                    instead of being streamed in chunks (CIFAR-10 RGB in 8-bit
                    is 147 MiB — one upload for the whole run, then no host→GPU
                    traffic per step at all).
                    web asks for the baseline: the way to check a model would
                    still run in a browser. Inference and perpetual work under
                    both. `--resources --no-gpu` answers the browser question
                    without opening an adapter at all, whatever this flag says.
                    Every run banner prints the profile it was GRANTED — if the
                    adapter refuses its own limits the run falls back to the
                    baseline and says so.

Weight averaging (EMA):
  --ema F           keep an exponential moving average of the weights,
                    `ema <- d*ema + (1-d)*w` after every optimiser step, with
                    d = min(F, (1+t)/(10+t)) — a warmup ramp — and the shadow
                    seeded with the weights, never with zeros. F must be
                    strictly between 0 and 1; 0.999 is the usual choice.
                    ABSENT MEANS NO AVERAGING: the run is bit-for-bit what it
                    was before this flag existed, down to the checkpoint bytes.
                    A run that keeps an average writes a BBCKPT3 checkpoint
                    holding BOTH weight sets (BBCKPT2 = raw only, still read
                    and still written by runs without --ema).
  --raw-weights     generate from the LAST ITERATE instead of the average.
                    Sampling and perpetual use the average whenever the
                    checkpoint carries one; this is the other arm of the
                    comparison. Both paths print which set they loaded.

Shipping a checkpoint (a lighter file to download):
  --export-weights  read a checkpoint and write one that holds ONLY the weights
                    inference generates from — no Adam moments, no EMA trailer.
                    A training checkpoint is four times heavier: the two optimiser
                    moments and (for a V3 file) the averaged copy each weigh as
                    much as the weights again. This drops them, keeping by default
                    the AVERAGED set the EMA exists to be sampled from, exactly as
                    sampling would select it. The result is an ordinary checkpoint:
                    inference, the TUI weight selector and --resume all read it.
                    It samples bit-for-bit like the file it came from on the same
                    seed — that is the point, and it is a test, not a claim.
  --ckpt <path>     the checkpoint to strip.
  --out <path>      where to write the stripped checkpoint. Required: this mode
                    only exists to produce a file, so it never guesses the name.
  --raw-weights     carry the last iterate instead of the average, the same
                    choice sampling's --raw-weights makes.
  --quantize        code each weight to 8 bits (per-tensor affine: one
                    (min, scale) pair and one byte per weight, dequantised on
                    load through a 256-entry table). A quarter of the file again,
                    at the cost of the quantisation error — measure it with
                    --eval before shipping it, exactly as the dataset's own 8-bit
                    move was measured. The file (magic BBCKPTQ) is read by
                    inference and the web engine; it is NOT resumable.

Where a run's weights land:
  Every run writes its OWN file, `run-<YYYY-MM-DD_HHMM>.ckpt`, and then points
  `latest.ckpt` at it (a hard link beside it — same bytes, no second copy).
  Nothing overwrites the previous run any more: yesterday's weights are still
  in `pretrained_weights/` this morning, under the date they were trained.
  The stamp is local time and zero-padded, so sorting those names by name IS
  sorting them by run; a second run in the same minute gets an `_02` suffix.
  --out <ckpt>      write exactly there, exactly under that name: no date, no
                    `latest.ckpt` beside it. Naming the file is the way to opt
                    out of the convention. WITHOUT it a headless run writes to
                    scratch — `$TMPDIR/batlab-<model>/run-<stamp>.ckpt` — and
                    never into the model's saved weights.

Resuming and partial checkpoints:
  --resume <ckpt>   continue a run from <ckpt>: weights, Adam moments, the
                    global step counter t, and the EMA if the file has one.
                    Writes to --out, so it never overwrites what it resumed
                    from unless told to. A checkpoint of another architecture
                    is refused, not truncated. Without it a headless run still
                    starts from scratch.
  --checkpoint-every N
                    write a partial checkpoint every N steps to
                    `<out stem>.partial.ckpt` — ONE file, rotated, written via
                    a temp file and an atomic rename so a kill mid-write can
                    never leave a truncated .ckpt. It is a normal checkpoint:
                    --resume and the TUI weight selector both see it. Unless
                    --out named the file, `latest.ckpt` follows each rotation,
                    so a run killed at hour nine leaves its partial as the
                    model's newest weights. Default: only at the end of the
                    run, as before.

Perpetual notes:
  --regime          wander (errance) | breathe (respiration) | flux. French
                    spellings are accepted too.
  --t-r/--t-star    the level dial, one flag under three spellings. It is a
                    renoise depth in wander/breathe and the level held in flux,
                    which is why it answers to both names.
  --frames N        stops after N *closed cycles*, and writes one PNG per cycle.
                    Flux closes none: it writes no PNG and REFUSES this bound
                    rather than running forever.
  --actions N       stops after N drift actions (frames). The only bound flux
                    accepts, and how two regimes are compared over equal frames.
                    Overrides --frames when both are given.

--dump <path> writes every frame of the run, little-endian throughout:

    \"BATFLUX1\"  u32 width  u32 height  u32 channels
    then per frame:  u8 phase (0 descent, 1 climb, 2 flux)
                     u32 t (the level this frame landed on)
                     f32[w*h*c] x_t        f32[w*h*c] x0_hat

The phase names what the frame DID, so in flux the opening approach is
T-1-t* frames of 0 and every frame from the first churn on is 2 — cut a
prologue on that byte rather than on a count.

The frame count is left to be inferred from the file size, so a run killed
mid-write still parses up to its last whole frame. Read it with
`tools/flux_analysis.py`.
";

/// Rejects any flag the parser does not know, before anything expensive runs.
///
/// A headless run is driven by scripts and agents that cannot see a typo. While
/// unknown flags were dropped in silence, `--t-star` — a flag named in the
/// public contract but never parsed — was indistinguishable from a flag that
/// worked: a whole black-box campaign passed it, got byte-identical dumps, and
/// only caught it by diffing against a deliberately invented flag.
///
/// `valued` flags consume the token after them, so a path or a negative number
/// can never be mistaken for a flag of its own.
fn reject_unknown_flags(args: &[String], valued: &[&str], bare: &[&str]) -> Result<(), String> {
    let mut index = 1;
    while index < args.len() {
        let arg = args[index].as_str();
        if arg.starts_with("--") || (arg.starts_with('-') && arg.len() > 1) {
            if valued.contains(&arg) {
                index += 2;
                continue;
            }
            if !bare.contains(&arg) {
                let mut known: Vec<&str> = valued.iter().chain(bare.iter()).copied().collect();
                known.sort_unstable();
                return Err(format!(
                    "unknown flag `{arg}`. Known flags here: {}. See --help.",
                    known.join(" ")
                ));
            }
        }
        index += 1;
    }
    Ok(())
}

/// Reads `--gpu-limits` and makes it the profile every device request uses.
///
/// Applied before any adapter is opened and never consulted again — see
/// [`GpuLimitsProfile::set_process_default`] for why this is a process setting
/// rather than a parameter. Absent, the default stands (native).
fn apply_gpu_limits_flag(args: &[String]) -> Result<(), String> {
    let Some(value) = args
        .iter()
        .position(|arg| arg == "--gpu-limits")
        .map(|index| args.get(index + 1))
    else {
        return Ok(());
    };
    let value = value.ok_or_else(|| {
        "--gpu-limits takes a value: native (the adapter's own limits, default) \
         or web (the WebGPU baseline)"
            .to_string()
    })?;
    GpuLimitsProfile::parse(value)?.set_process_default();
    Ok(())
}

/// The level dial of a perpetual run, under any of its three spellings.
///
/// One field, three names, because the field means two things: `t_r`, the depth
/// a cycle re-noises back to, and `t*`, the level flux holds. The public
/// contract named `--t-r` and `--t-star` while only `--depth` was parsed, so
/// both were accepted by the shell, dropped by the binary, and the run went
/// ahead at the default level — a blind test campaign passed `--t-star` for a
/// whole day against dumps that were byte-identical to no flag at all.
///
/// `Ok(None)` means no dial was given; the caller supplies the default.
fn dial_level(args: &[String]) -> Result<Option<usize>, String> {
    let named = |name: &'static str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|i| args.get(i + 1))
            .map(|value| (name, value.as_str()))
    };
    match named("--t-star")
        .or_else(|| named("--t-r"))
        .or_else(|| named("--depth"))
    {
        Some((name, value)) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|_| format!("{name} takes a level (an integer), got `{value}`")),
        None => Ok(None),
    }
}

/// How often (in steps) the training loop generates an instrumented sample and
/// logs the full per-step denoising trajectory. Coarser than the loss/probe
/// interval because a full sample runs `schedule.len()` forward passes.
const SAMPLE_INTERVAL_STEPS: usize = 200;

#[derive(Debug, Clone)]
struct ImageSample {
    target: Vec<f32>,
}

fn main() {
    // winit's event loop must own the process main thread (a hard AppKit
    // requirement on macOS), so the TUI and the training/inference loop run on a
    // worker thread instead. `run_on_main_thread` returns once that worker does.
    // The headless path also runs inside it: the training path can warm the
    // visualiser, which needs the event loop to be available.
    batlab_ui::visualiser::run_on_main_thread(|| {
        // -----------------------------------------------------------------
        // DEV/CI ONLY — headless training entry point.
        //
        // The normal entry point is the interactive TUI below. This branch
        // exists so a training run can be driven from a script (validating
        // pipeline fixes, regression runs) without a terminal. It reuses
        // `run_training` unchanged, so it exercises exactly the production
        // path and cannot drift from it.
        //
        //   cargo run --release -- --headless-train <model> --steps N \
        //       --dataset <path> [--lr F] [--batch N] [--out <ckpt path>] \
        //       [--optimizer sgd|adam] [--weight-init uniform|he] \
        //       [--loss-weighting uniform|snr] [--snr-gamma F]
        //
        // It never writes back to the model's config_file and defaults its
        // checkpoint to a scratch path, so it cannot clobber saved weights.
        // -----------------------------------------------------------------
        let args: Vec<String> = std::env::args().collect();
        // Before anything else: the binary has to be able to say what it takes.
        // Without this the only way to discover a flag was to try it, and an
        // unknown flag used to be ignored in silence — so trying it proved
        // nothing either.
        if args.iter().any(|arg| arg == "--help" || arg == "-h") {
            print!("{HELP}");
            return;
        }
        // The limits a device is opened with cannot be changed afterwards, and
        // every path below — headless, TUI, visualiser — opens one. So the
        // choice is made here, once, before the first adapter request. It is
        // read again nowhere: `GpuContext::new_headless` picks it up.
        if let Err(err) = apply_gpu_limits_flag(&args) {
            eprintln!("{err}");
            std::process::exit(1);
        }
        // -----------------------------------------------------------------
        // DEV/CI ONLY — where a training step's 4.4 seconds go.
        //
        //   cargo run --release -p batlab -- --profile-step <model> \
        //       --dataset <path> [--batch N] [--rounds N] [--top N]
        // -----------------------------------------------------------------
        if args.iter().any(|arg| arg == "--profile-step") {
            if let Err(err) = run_profile_step(&args) {
                eprintln!("profile: {err}");
                std::process::exit(1);
            }
            return;
        }
        if args.iter().any(|arg| arg == "--headless-train") {
            if let Err(err) = run_headless_train(&args) {
                eprintln!("headless training failed: {err}");
                std::process::exit(1);
            }
            return;
        }

        // -----------------------------------------------------------------
        // DEV/CI ONLY — headless sampling entry point.
        //
        // Loads a checkpoint and generates images WITHOUT training, writing a
        // JSONL log of per-step denoising stats (latent + ε̂ min/max/mean/σ) so
        // the point where the latent diverges is visible. Never writes back the
        // model config and never touches saved weights.
        //
        //   cargo run --release -- --headless-sample <model> \
        //       --checkpoint <path> [--seed N] [--paths N] \
        //       [--magnitude F] [--out <img>] [--log <jsonl>]
        // -----------------------------------------------------------------
        if args.iter().any(|arg| arg == "--headless-sample") {
            if let Err(err) = run_headless_sample(&args) {
                eprintln!("headless sampling failed: {err}");
                std::process::exit(1);
            }
            return;
        }

        // -----------------------------------------------------------------
        // DEV/CI ONLY — headless perpetual time-lapse.
        //
        // Walks the same drift the TUI's Perpetual mode walks and writes one
        // PNG per completed cycle, so the wandering can be inspected as a
        // sequence of stills without a window.
        //
        //   cargo run --release -p main -- --headless-perpetual <model> \
        //       --frames N [--checkpoint <path>] [--depth T] \
        //       [--regime wander|breathe] [--seed N] [--magnitude F] \
        //       [--window] [--climb-frames N] [--out <dir>]
        // -----------------------------------------------------------------
        // -----------------------------------------------------------------
        // What this model costs on the GPU, and what crosses the boundary.
        //
        // Read-only: it opens an adapter to read its limits, and with
        // `--measure` it builds the model and runs a handful of steps. It never
        // writes a config, a checkpoint or an image.
        //
        //   cargo run -p batlab -- --resources <model> [--batch N] [--measure]
        // -----------------------------------------------------------------
        if args.iter().any(|arg| arg == "--resources") {
            if let Err(err) = run_resources(&args) {
                eprintln!("resources: {err}");
                std::process::exit(1);
            }
            return;
        }

        if args.iter().any(|arg| arg == "--headless-perpetual") {
            if let Err(err) = run_headless_perpetual(&args) {
                eprintln!("headless perpetual failed: {err}");
                std::process::exit(1);
            }
            return;
        }

        // DEV/CI ONLY — held-out ε/x₀ evaluation of one or more checkpoints.
        if args.iter().any(|arg| arg == "--eval") {
            if let Err(err) = run_eval(&args) {
                eprintln!("eval failed: {err}");
                std::process::exit(1);
            }
            return;
        }

        // DEV/CI ONLY — strip a checkpoint down to the weights inference uses.
        if args.iter().any(|arg| arg == "--export-weights") {
            if let Err(err) = run_export_weights(&args) {
                eprintln!("export-weights failed: {err}");
                std::process::exit(1);
            }
            return;
        }

        let config = match tui::run() {
            Ok(c) => c,
            Err(_) => return,
        };

        run_execution_loop(config);
    });
}

/// Where a perpetual run's opening picture comes from.
///
/// The dataset is loaded once, on the CPU, and kept as plain tensors: a run
/// asks it for one `x₀` when it starts and one more on every `[r]`. That is
/// nothing next to a single model call, so there is no GPU dataset here and no
/// chunk juggling — [`GpuDataset`] exists for training, which streams thousands
/// of samples a minute.
///
/// **One function provides the picture** — [`SeedImages::provide_x0`]. Adding a
/// chooser later (an index typed into the form, a file dropped on the window)
/// is a second constructor beside `at_random`, not a new call site: the two
/// perpetual paths only ever call `provide_x0`.
struct SeedImages {
    samples: Dataset,
    path: PathBuf,
    source: SeedSource,
}

impl SeedImages {
    /// Loads the dataset a model drifts away from — **the one entry point every
    /// path that starts a drift goes through**.
    ///
    /// It takes the whole `config`, not a pre-chosen path, and that is the
    /// point: a caller cannot forget to consult the model's own
    /// `seed_dataset` the way `--headless-perpetual` did for an entire
    /// campaign, drifting `Elephants_XL` away from CIFAR trucks. The ranking
    /// itself — flag, then config, then the channel convention — lives once, in
    /// [`storage::Storage::resolve_seed_dataset`].
    ///
    /// Three outcomes, and they differ by source:
    ///
    /// - `Ok(Some(_))` — a dataset was found and it fits the model.
    /// - `Ok(None)` — nothing to load, and nothing was promised: no convention
    ///   for this geometry, or the derived file is not on this machine. A
    ///   missing dataset is not a reason to refuse to run, it is a reason to
    ///   fall back on pure noise and say so.
    /// - `Err(_)` — something *named* it (flag or config) and it is missing,
    ///   empty, or has the wrong channels. Falling back to noise there would
    ///   look exactly like the setting being ignored, which is the bug this
    ///   whole function exists to close.
    fn resolve(
        flag: Option<&str>,
        config: &ModelConfig,
        output_size: (u32, u32, u32),
    ) -> Result<Option<Self>, String> {
        let Some(choice) =
            storage::resolve_seed_dataset(flag, config.seed_dataset.as_deref(), output_size.2)
        else {
            return Ok(None);
        };
        // What to say when it goes wrong, naming the source that asked for it:
        // "--seed-dataset x: …" and "config_file seed_dataset x: …" send the
        // reader to two different places.
        let blame = |reason: &str| {
            format!(
                "{} {}: {reason}",
                choice.source.label(),
                choice.path.display()
            )
        };
        let named = choice.source != SeedSource::Convention;
        if !choice.path.exists() {
            return match named {
                true => Err(blame("no such file")),
                false => Ok(None),
            };
        }
        // Refused, not resized. Width and height are resampled on the host on
        // the way in, which is a visible thing to do to a picture; channels are
        // not — a greyscale file handed to a colour model is replicated across
        // R, G and B and *succeeds*, which is what makes it worth refusing. The
        // convention cannot land here (it is derived from these very channels),
        // so this only ever fires on something a human named.
        if let Some((_, _, _, channels, _)) = storage::read_batraw_header(&choice.path) {
            if channels != output_size.2 {
                return Err(blame(&format!(
                    "{channels} channel(s) for a model that emits {} — it would be \
                     replicated or flattened without a word, not resized",
                    output_size.2
                )));
            }
        }
        let samples = load_dataset(&choice.path.to_string_lossy(), output_size)?;
        if samples.is_empty() {
            return match named {
                true => Err(blame("holds no images")),
                false => Ok(None),
            };
        }
        Ok(Some(Self {
            samples,
            path: choice.path,
            source: choice.source,
        }))
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn path(&self) -> &Path {
        &self.path
    }

    /// Which of the three sources won. Printed, never re-derived: the banner
    /// has to report what the run actually resolved.
    fn source(&self) -> SeedSource {
        self.source
    }

    /// The `x₀` a run sets out from: one image of the dataset, drawn from
    /// `seed`.
    ///
    /// The index is **avalanched out of the seed, never taken modulo it**.
    /// Seeds here come from a clock (`random_seed`) or from a form, so
    /// `seed % len` would walk consecutive images on consecutive re-seeds and
    /// correlate the run with whatever order the dataset happens to be in. Same
    /// discipline as `gaussian_at` — mix first, index second (`ANISOTROPY_HUNT.md`).
    fn provide_x0(&self, seed: u64) -> Vec<f32> {
        self.samples.sample(self.at_random(seed))
    }

    fn at_random(&self, seed: u64) -> usize {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) % self.samples.len() as u64) as usize
    }
}

/// The model, wrapped as the one question a perpetual walk asks it: "what noise
/// do you see in this latent, at this level?".
///
/// The walk itself lives in the engine and holds no model, so this adapter is
/// the whole of the coupling — and it is the same one on both perpetual paths,
/// the TUI run and the headless one, which is why they cannot drift apart.
struct ModelNoise<'a, State> {
    model: &'a mut Model<State>,
    schedule: &'a LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
}

impl<State> batlab_core::NoisePredictor for ModelNoise<'_, State> {
    fn predict_noise(&mut self, latent: &[f32], diffusion_step: usize) -> Vec<f32> {
        batlab_core::predict_epsilon(
            self.model,
            self.schedule,
            self.input_channels,
            self.signal_channels,
            latent,
            diffusion_step,
        )
    }
}

/// Which weight set a headless generating run asks for.
///
/// The default is the average, because a checkpoint that has one was written by
/// a run that wanted to be sampled from it. `--raw-weights` is what makes the
/// comparison possible — and it is a *bare* flag, so it cannot swallow the
/// argument after it.
fn weights_source(args: &[String]) -> CheckpointWeights {
    if args.iter().any(|arg| arg == "--raw-weights") {
        CheckpointWeights::Raw
    } else {
        CheckpointWeights::Ema
    }
}

/// Load a checkpoint for **generating**, and say which of its two weight sets
/// was used.
///
/// Every sampling path goes through here — the two headless entry points and
/// the two TUI ones — so "generation uses the average when the file has one"
/// is one decision in one place rather than four that can drift.
///
/// The line it prints is the contract's observable half: an EMA comparison is
/// worthless if the two arms might silently have loaded the same weights, and
/// `carries_ema` vs `used_ema` is exactly what tells them apart.
fn load_sampling_checkpoint(
    model: &mut Model<Training>,
    path: &Path,
    source: CheckpointWeights,
) -> Result<(), String> {
    let report = model
        .load_checkpoint_with(path, source)
        .map_err(|err| format!("failed to load checkpoint {}: {err}", path.display()))?;
    let which = match (report.carries_ema, report.used_ema) {
        (true, true) => format!(
            "EMA weights (decay {:.5})",
            report.ema_decay.unwrap_or_default()
        ),
        (true, false) => "raw weights (the file also carries an EMA)".to_string(),
        (false, _) => "raw weights (the file carries no EMA)".to_string(),
    };
    println!("[weights] {} → {which}", path.display());
    Ok(())
}

/// The bytes an export carries.
///
/// Load `source` with the inference selection rule — the average when the file
/// carries one, the raw iterate under `--raw-weights` — into a fresh inference
/// model, then re-serialise only the weights that model would sample from. No
/// optimiser moments, no EMA trailer: four fifths of a training checkpoint that
/// inference never reads.
///
/// The selection is *not* re-implemented here — it is the one `CheckpointWeights`
/// enum every sampling path already funnels through. The stripping is not a new
/// serialiser either: the model is built with SGD (stateless, no optimiser
/// trailer) and no EMA shadow, so [`Model::checkpoint_bytes`] emits exactly the
/// weights-only V2 file it has always written for such a model — the layout
/// `the_parameter_count_matches_the_scalars_a_checkpoint_holds` pins down.
async fn stripped_checkpoint(
    config: &ModelConfig,
    source: &[u8],
    weights: CheckpointWeights,
    quantize: bool,
) -> Result<Vec<u8>, String> {
    let (_gpu, mut model) = build_execution_model(
        config,
        INFERENCE_RUNTIME_LR,
        INFERENCE_RUNTIME_BATCH_SIZE,
        OptimizerKind::default(),
        WeightInit::default(),
        None,
    )
    .await?;
    let report = model
        .load_checkpoint_bytes_with(source, weights)
        .map_err(|err| format!("failed to load source checkpoint: {err}"))?;
    let which = match (report.carries_ema, report.used_ema) {
        (true, true) => format!(
            "EMA weights (decay {:.5})",
            report.ema_decay.unwrap_or_default()
        ),
        (true, false) => "raw weights (the file also carries an EMA)".to_string(),
        (false, _) => "raw weights (the file carries no EMA)".to_string(),
    };
    println!("[weights] source → {which}");
    if quantize {
        model.checkpoint_bytes_quantized().map_err(|err| err.to_string())
    } else {
        model.checkpoint_bytes().map_err(|err| err.to_string())
    }
}

/// See the DEV/CI note in `main`. Not reachable from the TUI.
///
/// Writes a checkpoint carrying only the weights inference would generate from —
/// the file a browser downloads. A training checkpoint is four times heavier: it
/// also holds Adam's two moments and, for a V3 file, the EMA trailer. This drops
/// both, keeping (by default) the averaged set the EMA exists to be sampled from.
fn run_export_weights(args: &[String]) -> Result<(), String> {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    reject_unknown_flags(
        args,
        &["--export-weights", "--ckpt", "--out", "--gpu-limits"],
        &["--raw-weights", "--quantize"],
    )?;

    let model_name = flag("--export-weights")
        .ok_or_else(|| "--export-weights requires a model name".to_string())?;
    let ckpt = flag("--ckpt").ok_or_else(|| "--ckpt <path to .ckpt> is required".to_string())?;
    let ckpt_path = PathBuf::from(&ckpt);
    if !ckpt_path.exists() {
        return Err(format!("checkpoint does not exist: {ckpt}"));
    }
    let out = flag("--out").ok_or_else(|| "--out <path to .ckpt> is required".to_string())?;
    // The average when the file has one, the raw iterate under `--raw-weights` —
    // the same rule sampling applies, so the export carries what the page shows.
    let weights = weights_source(args);
    // 8-bit codes instead of f32: a quarter of the weights-only file again, at
    // the cost of the quantisation error — a trade the run measures, not assumes.
    let quantize = args.iter().any(|arg| arg == "--quantize");

    let config_path = storage::model_config_path(&model_name)
        .map_err(|err| format!("failed to resolve config path: {err}"))?;
    let config = storage::load_model_config(&config_path)
        .map_err(|err| format!("failed to load {}: {err}", config_path.display()))?;

    let source = fs::read(&ckpt_path).map_err(|err| format!("failed to read {ckpt}: {err}"))?;

    let rt = tokio::runtime::Runtime::new().map_err(|err| format!("tokio runtime: {err}"))?;
    let bytes = rt.block_on(stripped_checkpoint(&config, &source, weights, quantize))?;

    let out_path = PathBuf::from(&out);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create output dir: {err}"))?;
        }
    }
    fs::write(&out_path, &bytes).map_err(|err| format!("failed to write {out}: {err}"))?;

    // The banner re-emits the parsed configuration so a scripted run can prove
    // the flag was understood — the same contract every headless mode keeps.
    let which = if matches!(weights, CheckpointWeights::Raw) {
        "raw"
    } else {
        "ema"
    };
    let encoding = if quantize { "u8" } else { "f32" };
    println!(
        "export-weights '{model_name}': ckpt={ckpt}, weights={which}, encoding={encoding}, \
         out={out}, bytes={} (source {} bytes)",
        bytes.len(),
        source.len()
    );
    Ok(())
}

/// See the DEV/CI note in `main`.
fn run_headless_train(args: &[String]) -> Result<(), String> {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    reject_unknown_flags(
        args,
        &[
            "--headless-train",
            "--steps",
            "--dataset",
            "--lr",
            "--batch",
            "--out",
            "--optimizer",
            "--weight-init",
            "--loss-weighting",
            "--snr-gamma",
            "--ema",
            "--resume",
            "--checkpoint-every",
            "--gpu-limits",
        ],
        &[],
    )?;
    let parse = |name: &str, fallback: f32| -> Result<f32, String> {
        match flag(name) {
            Some(v) => v
                .parse()
                .map_err(|_| format!("invalid value for {name}: {v}")),
            None => Ok(fallback),
        }
    };

    let model_name = flag("--headless-train")
        .ok_or_else(|| "--headless-train requires a model name".to_string())?;
    let steps = parse("--steps", 500.0)? as usize;
    let lr = parse("--lr", 1e-3)?;
    let batch_size = parse("--batch", 16.0)? as u32;
    let optimizer = match flag("--optimizer") {
        Some(value) => OptimizerKind::parse(&value)
            .ok_or_else(|| format!("invalid value for --optimizer: {value} (want sgd|adam)"))?,
        None => OptimizerKind::default(),
    };
    let weight_init = match flag("--weight-init") {
        Some(value) => WeightInit::parse(&value)
            .ok_or_else(|| format!("invalid value for --weight-init: {value} (want uniform|he)"))?,
        None => WeightInit::default(),
    };
    // Absent means no averaging at all, bit for bit the run this binary did
    // before the flag existed — not "averaging with some default decay".
    let ema = match flag("--ema") {
        Some(value) => Some(EmaConfig::parse(&value).ok_or_else(|| {
            format!("invalid value for --ema: {value} (want a decay strictly between 0 and 1, e.g. 0.999)")
        })?),
        None => None,
    };
    let resume_from = flag("--resume").map(PathBuf::from);
    let checkpoint_every = match flag("--checkpoint-every") {
        Some(value) => {
            let steps: usize = value
                .parse()
                .map_err(|_| format!("invalid value for --checkpoint-every: {value}"))?;
            if steps == 0 {
                return Err("--checkpoint-every takes a number of steps > 0".to_string());
            }
            Some(steps)
        }
        None => None,
    };
    let snr_gamma = parse("--snr-gamma", DEFAULT_SNR_GAMMA)?;
    let loss_weighting = match flag("--loss-weighting") {
        Some(value) => LossWeighting::parse(&value, snr_gamma).ok_or_else(|| {
            format!("invalid value for --loss-weighting: {value} (want uniform|snr)")
        })?,
        None => LossWeighting::default(),
    };

    let config_path = storage::model_config_path(&model_name)
        .map_err(|err| format!("failed to resolve config path: {err}"))?;
    let mut config = storage::load_model_config(&config_path)
        .map_err(|err| format!("failed to load {}: {err}", config_path.display()))?;

    let dataset_path = flag("--dataset")
        .ok_or_else(|| "--dataset <path to .batraw or image dir> is required".to_string())?;
    let (checkpoint_path, maintain_latest) =
        headless_checkpoint_target(flag("--out"), &model_name, SystemTime::now())?;

    let train_cfg = TrainingConfig {
        lr,
        batch_size,
        steps,
        dataset_path,
        loss: match &config.run.mode {
            RunMode::Train(existing) => existing.loss.clone(),
            RunMode::Infer | RunMode::Perpetual(_) => tui::LossMethod::MeanSquared,
        },
        checkpoint_path: Some(checkpoint_path.to_string_lossy().to_string()),
        // Fixes #1 and #4 changed the convolution operator and the data range,
        // so any pre-existing checkpoint is meaningless. Always start fresh —
        // unless `--resume` names a file, which is the explicit opt-in.
        load_checkpoint: false,
        optimizer,
        weight_init,
        loss_weighting,
        ema_decay: ema.map(|e| e.decay),
    };
    config.run.mode = RunMode::Train(train_cfg.clone());

    // The banner re-emits the parsed configuration: `bench/optimizer/lib.sh`
    // asserts on it, and it is the only way a scripted run can prove a flag
    // was understood rather than dropped.
    println!(
        "headless training '{model_name}': {steps} steps, lr={lr}, batch={batch_size}, \
         optimizer={}, init={}, loss-weighting={}, ema={}, resume={}, checkpoint-every={}, \
         out={}, dataset={}",
        optimizer.label(),
        weight_init.label(),
        loss_weighting.label(),
        match ema {
            Some(config) => config.decay.to_string(),
            None => "off".to_string(),
        },
        match resume_from.as_ref() {
            Some(path) => path.display().to_string(),
            None => "none".to_string(),
        },
        match checkpoint_every {
            Some(steps) => steps.to_string(),
            None => "final only".to_string(),
        },
        checkpoint_path.display(),
        train_cfg.dataset_path
    );

    let (tx, rx) = std::sync::mpsc::channel::<tui::TrainingEvent>();
    let worker = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let options = RunOptions {
                resume_from,
                checkpoint_every,
                // The path above is already the run's own, and `--resume` is
                // how a headless run names what it reads: nothing to redirect.
                load_from: None,
                write_to: None,
                maintain_latest,
            };
            if let Err(message) =
                run_training(config, train_cfg, options, &tx, std::sync::mpsc::channel().1).await
            {
                let _ = tx.send(tui::TrainingEvent::Error { message });
            }
            let _ = tx.send(tui::TrainingEvent::Done);
        });
    });

    let mut failure = None;
    while let Ok(event) = rx.recv() {
        match event {
            tui::TrainingEvent::Step {
                step,
                loss: Some(loss),
                ..
            } => println!("step {step}\tloss {loss:.6}"),
            tui::TrainingEvent::Error { message } => {
                eprintln!("error: {message}");
                failure = Some(message);
            }
            tui::TrainingEvent::Done => break,
            _ => {}
        }
    }
    let _ = worker.join();

    match failure {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// See the DEV/CI note in `main`. Not reachable from the TUI.
fn run_headless_sample(args: &[String]) -> Result<(), String> {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    reject_unknown_flags(
        args,
        &[
            "--headless-sample",
            "--checkpoint",
            "--seed",
            "--paths",
            "--magnitude",
            "--variance",
            "--out",
            "--log",
            "--gpu-limits",
        ],
        &["--raw-weights"],
    )?;

    let model_name = flag("--headless-sample")
        .ok_or_else(|| "--headless-sample requires a model name".to_string())?;
    let checkpoint = flag("--checkpoint")
        .ok_or_else(|| "--checkpoint <path to .ckpt> is required".to_string())?;
    let checkpoint_path = PathBuf::from(&checkpoint);
    if !checkpoint_path.exists() {
        return Err(format!("checkpoint does not exist: {checkpoint}"));
    }
    let seed = flag("--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let paths = flag("--paths")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let magnitude = flag("--magnitude")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);
    // Generating uses the average when the checkpoint carries one, because
    // that is the set the average exists to be sampled from. `--raw-weights`
    // asks for the last iterate instead — the other arm of the comparison.
    let weights = weights_source(args);

    let config_path = storage::model_config_path(&model_name)
        .map_err(|err| format!("failed to resolve config path: {err}"))?;
    let config = storage::load_model_config(&config_path)
        .map_err(|err| format!("failed to load {}: {err}", config_path.display()))?;

    // The reverse-step variance. Defaults to the config's choice (Beta unless the
    // config_file overrode it), and `--variance beta|posterior` forces one.
    let variance = match flag("--variance") {
        Some(spec) => PosteriorVariance::from_cli(&spec)
            .ok_or_else(|| format!("--variance must be 'beta' or 'posterior', got '{spec}'"))?,
        None => config.inference.posterior_variance,
    };

    let out_path = flag("--out").unwrap_or_else(|| {
        std::env::temp_dir()
            .join(format!("{model_name}_sample_{seed}.png"))
            .to_string_lossy()
            .to_string()
    });
    let log_path = flag("--log").unwrap_or_else(|| {
        std::env::temp_dir()
            .join(format!("{model_name}_sample_{seed}_metrics.jsonl"))
            .to_string_lossy()
            .to_string()
    });

    let rt = tokio::runtime::Runtime::new().map_err(|err| format!("tokio runtime: {err}"))?;
    rt.block_on(async {
        let (_gpu, mut model) = build_execution_model(
            &config,
            INFERENCE_RUNTIME_LR,
            INFERENCE_RUNTIME_BATCH_SIZE,
            OptimizerKind::default(),
            WeightInit::default(),
            None,
        )
        .await?;
        load_sampling_checkpoint(&mut model, &checkpoint_path, weights)?;

        let input_dims = model
            .input_dim()
            .ok_or_else(|| "model has no input dimensions".to_string())?;
        let output_dims = model
            .output_dim()
            .ok_or_else(|| "model has no output dimensions".to_string())?;
        let output_size = (output_dims.x, output_dims.y, output_dims.z);
        let output_len = (output_size.0 * output_size.1 * output_size.2) as usize;

        let schedule = LinearNoiseSchedule::new_linear(
            DIFFUSION_SCHEDULE_STEPS,
            DIFFUSION_BETA_START,
            DIFFUSION_BETA_END,
        );

        let mut metrics = MetricsLogger::create(Path::new(&log_path), true);
        let mut trajectory = Vec::new();
        let image = sample_diffusion(
            &mut model,
            input_dims.z as usize,
            output_dims.z as usize,
            output_len,
            &schedule,
            seed,
            paths,
            magnitude,
            variance,
            Some(&mut trajectory),
            None,
            |_, _| {},
        );
        log_trajectory(&mut metrics, 0, seed, &trajectory, &Stats::of(&image));

        let out = PathBuf::from(&out_path);
        let sample_dir = out.parent().map(Path::to_path_buf).unwrap_or_default();
        if !sample_dir.as_os_str().is_empty() {
            fs::create_dir_all(&sample_dir)
                .map_err(|err| format!("failed to create output dir: {err}"))?;
        }
        // Reuse the shared encoder so the saved file matches training previews.
        let saved = save_tensor_as_image(&image, output_size, &sample_dir, 0)?;
        // Rename the deterministic `step_0000.png` to the requested path.
        if saved != out {
            fs::rename(&saved, &out)
                .map_err(|err| format!("failed to move sample to {out_path}: {err}"))?;
        }

        let img_stats = Stats::of(&image);
        println!(
            "headless sample '{model_name}': seed={seed} paths={paths} magnitude={magnitude} \
             variance={}\n\
             image → {out_path}\nmetrics → {log_path}\n\
             final image stats: min={:.4} max={:.4} mean={:.4} std={:.4}",
            variance.as_str(), img_stats.min, img_stats.max, img_stats.mean, img_stats.std
        );
        Ok::<(), String>(())
    })
}

/// See the DEV/CI note in `main`. Not reachable from the TUI.
///
/// The instrument the architecture mission is built on: a way to say *this
/// checkpoint is better than that one* that is not "look at the picture". It
/// scores one or more checkpoints on a held-out, deterministic slice — the same
/// images, the same timesteps, the same noise fields for every checkpoint — so a
/// difference in the numbers is a difference in the weights and nothing else.
///
/// Reports MSE(ε̂, ε) per timestep bucket and the same error in image space
/// (x₀), against the three trivial baselines of `tools/trivial_baselines.py`.
/// Every arbitration later in the mission goes through this, exactly because the
/// EMA one could not — the repo had no ε evaluator, so a 4.9 % weight change
/// could only be judged by eye (`docs/reports/EMA.md`).
fn run_eval(args: &[String]) -> Result<(), String> {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    reject_unknown_flags(
        args,
        &[
            "--eval",
            "--ckpt",
            "--dataset",
            "--samples",
            "--buckets",
            "--t-per-bucket",
            "--seed",
            "--gpu-limits",
        ],
        &["--raw-weights"],
    )?;

    let model_name = flag("--eval").ok_or_else(|| "--eval requires a model name".to_string())?;

    // Every `--ckpt <path>`, in order — this is the flag that makes the table a
    // comparison rather than a single score.
    let checkpoints: Vec<PathBuf> = args
        .iter()
        .enumerate()
        .filter(|(_, arg)| arg.as_str() == "--ckpt")
        .filter_map(|(i, _)| args.get(i + 1))
        .map(PathBuf::from)
        .collect();
    if checkpoints.is_empty() {
        return Err("--eval requires at least one --ckpt <path>".to_string());
    }
    for ckpt in &checkpoints {
        if !ckpt.exists() {
            return Err(format!("checkpoint does not exist: {}", ckpt.display()));
        }
    }

    let dataset_path =
        flag("--dataset").ok_or_else(|| "--dataset <path to .batraw> is required".to_string())?;
    let requested_samples = flag("--samples")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256)
        .max(1);
    let buckets = flag("--buckets")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let t_per_bucket = flag("--t-per-bucket")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let seed = flag("--seed").and_then(|v| v.parse::<u64>().ok()).unwrap_or(7);
    let weights = weights_source(args);

    let config_path = storage::model_config_path(&model_name)
        .map_err(|err| format!("failed to resolve config path: {err}"))?;
    let config = storage::load_model_config(&config_path)
        .map_err(|err| format!("failed to load {}: {err}", config_path.display()))?;

    let rt = tokio::runtime::Runtime::new().map_err(|err| format!("tokio runtime: {err}"))?;
    rt.block_on(async {
        let (_gpu, mut model) = build_execution_model(
            &config,
            INFERENCE_RUNTIME_LR,
            INFERENCE_RUNTIME_BATCH_SIZE,
            OptimizerKind::default(),
            WeightInit::default(),
            None,
        )
        .await?;

        let input_dims = model
            .input_dim()
            .ok_or_else(|| "model has no input dimensions".to_string())?;
        let output_dims = model
            .output_dim()
            .ok_or_else(|| "model has no output dimensions".to_string())?;
        let output_size = (output_dims.x, output_dims.y, output_dims.z);
        let output_len = (output_size.0 * output_size.1 * output_size.2) as usize;

        let dataset = load_dataset(&dataset_path, output_size)?;
        let total = dataset.len();
        if total == 0 {
            return Err(format!("dataset {dataset_path} is empty"));
        }
        let eval_count = requested_samples.min(total);
        // Held-out images from the TAIL; the mean image from the HEAD, so the
        // two do not overlap. When the dataset is smaller than the ask, the head
        // is empty and the mean falls back to the whole set (said out loud).
        let held_start = total - eval_count;
        let held: Vec<Vec<f32>> = (0..eval_count)
            .map(|k| dataset.sample(held_start + k))
            .collect();
        let mean_count = if held_start == 0 {
            total
        } else {
            held_start.min(20_000)
        };
        let mut mean_image = vec![0.0f64; output_len];
        for index in 0..mean_count {
            for (acc, value) in mean_image.iter_mut().zip(dataset.sample(index).iter()) {
                *acc += *value as f64;
            }
        }
        let mean_image: Vec<f32> = mean_image
            .iter()
            .map(|v| (*v / mean_count as f64) as f32)
            .collect();
        if held_start == 0 {
            eprintln!(
                "[eval] note: dataset has {total} images but {requested_samples} were asked for; \
                 the held-out set and the mean-image pool overlap."
            );
        }

        let schedule = LinearNoiseSchedule::new_linear(
            DIFFUSION_SCHEDULE_STEPS,
            DIFFUSION_BETA_START,
            DIFFUSION_BETA_END,
        );
        let cfg = EvalConfig {
            buckets,
            t_per_bucket,
            seed,
        };

        let mut reports: Vec<(PathBuf, EvalReport)> = Vec::with_capacity(checkpoints.len());
        for ckpt in &checkpoints {
            load_sampling_checkpoint(&mut model, ckpt, weights)?;
            let report = evaluate(
                &schedule,
                &held,
                &mean_image,
                input_dims.z as usize,
                output_dims.z as usize,
                &cfg,
                |input| model.predict(input),
            );
            reports.push((ckpt.clone(), report));
        }

        print_eval_table(
            &model_name,
            &dataset_path,
            eval_count,
            mean_count,
            &cfg,
            weights,
            &reports,
        );
        Ok::<(), String>(())
    })
}

/// The comparison table `--eval` prints. Presentation only: every number comes
/// from the engine's [`evaluate`], so the host decides layout, never arithmetic.
///
/// Two blocks, both against the same baselines (which are identical across
/// checkpoints — same draws — so they are read off the first report):
///   - ε-MSE per bucket, the training objective resolved per timestep range;
///   - x₀-RMSE per bucket, the error in image units [-1, 1], where the high-t
///     buckets are where the global content of a sample is actually decided.
/// The verdict a reader wants is the last line: does the checkpoint beat the
/// "learned only the mean" baseline in x₀ at high t.
fn print_eval_table(
    model_name: &str,
    dataset_path: &str,
    eval_count: usize,
    mean_count: usize,
    cfg: &EvalConfig,
    weights: CheckpointWeights,
    reports: &[(PathBuf, EvalReport)],
) {
    let weight_label = match weights {
        CheckpointWeights::Ema => "average when present (--raw-weights to force the iterate)",
        CheckpointWeights::Raw => "raw iterate (--raw-weights)",
    };
    println!(
        "\neval '{model_name}': {eval_count} held-out images (tail), mean image over {mean_count} \
         (head)\n  dataset={dataset_path}  buckets={}  t/bucket={}  seed={}  weights={weight_label}",
        cfg.buckets, cfg.t_per_bucket, cfg.seed
    );

    let Some((_, first)) = reports.first() else {
        return;
    };
    let bucket_headers: Vec<String> = first
        .model
        .iter()
        .map(|b| format!("t[{}-{})", b.t_lo, b.t_hi))
        .collect();

    // A run label short enough for a table: the file stem, trimmed on the left
    // (the date/suffix that distinguishes two runs lives at the end).
    let label_of = |path: &Path| -> String {
        let stem = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<ckpt>")
            .to_string();
        if stem.len() > 22 {
            format!("…{}", &stem[stem.len() - 21..])
        } else {
            stem
        }
    };

    let name_w = reports
        .iter()
        .map(|(p, _)| label_of(p).len())
        .chain(std::iter::once("ε̂=mean".len()))
        .max()
        .unwrap_or(12)
        .max(12);
    let col_w = 11usize;
    let header_cell = |s: &str| format!("{s:>col_w$}");

    // ---- ε-MSE ---------------------------------------------------------
    println!("\nε-MSE per timestep bucket (lower is better; the training objective):");
    let mut head = format!("{:<name_w$}", "checkpoint");
    for h in &bucket_headers {
        head.push_str(&header_cell(h));
    }
    head.push_str(&header_cell("all"));
    println!("{head}");
    for (path, report) in reports {
        let mut row = format!("{:<name_w$}", label_of(path));
        for b in &report.model {
            row.push_str(&header_cell(&format!("{:.4}", b.eps_mse())));
        }
        row.push_str(&header_cell(&format!("{:.4}", report.total_eps_mse())));
        println!("{row}");
    }
    println!("{}", "-".repeat(name_w + col_w * (bucket_headers.len() + 1)));
    let baseline_eps_row = |label: &str, pick: &dyn Fn(&batlab_core::BaselineBucket) -> f64| {
        let mut row = format!("{:<name_w$}", label);
        for b in &first.baselines {
            row.push_str(&header_cell(&format!("{:.4}", pick(b))));
        }
        row
    };
    println!("{}", baseline_eps_row("ε̂=x_t", &|b| b.eps_copy()));
    println!("{}", baseline_eps_row("ε̂=mean", &|b| b.eps_mean()));
    println!("{}", baseline_eps_row("ε̂=0", &|b| b.eps_zero()));

    // ---- x₀-RMSE (clipped reconstruction) ------------------------------
    // Element-weighted mean of the per-bucket MSE — every bucket carries the
    // same number of terms, so a plain mean is the whole-schedule error.
    let overall = |vals: &[f64]| -> f64 {
        if vals.is_empty() {
            0.0
        } else {
            (vals.iter().sum::<f64>() / vals.len() as f64).sqrt()
        }
    };
    println!(
        "\nx₀-RMSE per bucket (clipped reconstruction, image units [-1,1]; lower is better):"
    );
    let mut head = format!("{:<name_w$}", "checkpoint");
    for h in &bucket_headers {
        head.push_str(&header_cell(h));
    }
    head.push_str(&header_cell("all"));
    println!("{head}");
    for (path, report) in reports {
        let mut row = format!("{:<name_w$}", label_of(path));
        let per: Vec<f64> = report.model.iter().map(|b| b.x0_mse()).collect();
        for m in &per {
            row.push_str(&header_cell(&format!("{:.4}", m.sqrt())));
        }
        row.push_str(&header_cell(&format!("{:.4}", overall(&per))));
        println!("{row}");
    }
    println!("{}", "-".repeat(name_w + col_w * (bucket_headers.len() + 1)));
    let baseline_x0_row = |label: &str, pick: &dyn Fn(&batlab_core::BaselineBucket) -> f64| {
        let mut row = format!("{:<name_w$}", label);
        let per: Vec<f64> = first.baselines.iter().map(|b| pick(b)).collect();
        for v in &per {
            row.push_str(&header_cell(&format!("{:.4}", v.sqrt())));
        }
        row.push_str(&header_cell(&format!("{:.4}", overall(&per))));
        row
    };
    println!("{}", baseline_x0_row("ε̂=mean", &|b| b.x0_mean()));
    println!("{}", baseline_x0_row("ε̂=x_t", &|b| b.x0_copy()));
    println!("{}", baseline_x0_row("ε̂=0", &|b| b.x0_zero()));

    // ---- verdict -------------------------------------------------------
    // Two honest facts, no single pass/fail. The PRIMARY rank is whole-schedule
    // ε-MSE, the training objective and what the full multi-step sampler tracks
    // — that is the number to compare two architectures on. The SECONDARY read
    // is per-bucket reconstruction against the "learned only the mean" baseline:
    // a model beats it at the buckets where content is recoverable and cannot at
    // the very top, where a one-shot x̂₀ is unlearnable for ANY model (which is
    // why generation takes 256 steps, not one). An overall x₀ average would let
    // that unlearnable bucket flip the verdict, so it is deliberately not one.
    println!("\nverdict — primary rank: whole-schedule ε-MSE (lower is better):");
    let mut ranked: Vec<(&PathBuf, f64)> = reports
        .iter()
        .map(|(p, r)| (p, r.total_eps_mse()))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (rank, (path, eps)) in ranked.iter().enumerate() {
        let marker = if rank == 0 { "  ← best" } else { "" };
        println!("  {:>8.4}  {}{}", eps, label_of(path), marker);
    }
    println!("\n         reconstruction vs ε̂=mean (x₀-RMSE beaten, per bucket):");
    for (path, report) in reports {
        let beaten: Vec<&str> = report
            .model
            .iter()
            .zip(first.baselines.iter())
            .zip(bucket_headers.iter())
            .filter(|((m, b), _)| m.x0_mse() < b.x0_mean())
            .map(|((_, _), h)| h.as_str())
            .collect();
        let where_beaten = if beaten.is_empty() {
            "none".to_string()
        } else {
            beaten.join(", ")
        };
        println!(
            "  {}: beats the mean at {}/{} buckets ({})",
            label_of(path),
            beaten.len(),
            bucket_headers.len(),
            where_beaten
        );
    }
}

/// Every frame of a headless drift, both panes, as raw `f32`.
///
/// The measurements this exists for are frame-to-frame differences and
/// correlations, and in flux those live at about **one 8-bit level per frame**.
/// Reading them off PNGs would be measuring the quantiser: `CLIMB_COHERENCE.md`
/// §6 records a grain correlation of 1.000 in `f32` that PNGs could not report
/// above 0.87. So the frames come out unquantised, and every analysis of the
/// flux regime is done on this file.
///
/// Layout — little-endian throughout, `len = w * h * c`:
///
/// ```text
///   "BATFLUX1"  u32 w  u32 h  u32 c
///   then per frame:  u8 phase (0 descent, 1 climb, 2 flux)  u32 t
///                    f32[len] x_t        f32[len] x0_hat
/// ```
///
/// The frame count is left for the reader to infer from the file size: a run
/// that is killed mid-write then still parses up to its last whole frame.
struct FrameDump {
    path: PathBuf,
    writer: std::io::BufWriter<std::fs::File>,
    frames: usize,
}

impl FrameDump {
    fn create(path: &Path, size: (u32, u32, u32)) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
        let file = std::fs::File::create(path)
            .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        let mut header = Vec::with_capacity(20);
        header.extend_from_slice(b"BATFLUX1");
        header.extend_from_slice(&size.0.to_le_bytes());
        header.extend_from_slice(&size.1.to_le_bytes());
        header.extend_from_slice(&size.2.to_le_bytes());
        std::io::Write::write_all(&mut writer, &header)
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer,
            frames: 0,
        })
    }

    /// Takes the **action** rather than a phase, so the tag can only ever be the
    /// one of the deed the frame records. Handed a phase, a caller reads it off
    /// `drift.phase()` before stepping and files the frame under the phase it
    /// just left — which is exactly how the first churn frame of every approach
    /// went out labelled `descent` (see [`batlab_core::DriftAction::phase`]).
    fn record(
        &mut self,
        action: batlab_core::DriftAction,
        level: usize,
        latent: &[f32],
        x0: &[f32],
    ) -> Result<(), String> {
        let tag: u8 = match action.phase() {
            batlab_core::DriftPhase::Descent => 0,
            batlab_core::DriftPhase::Climb => 1,
            batlab_core::DriftPhase::Flux => 2,
        };
        let mut head = Vec::with_capacity(5);
        head.push(tag);
        head.extend_from_slice(&(level as u32).to_le_bytes());
        let mut body = Vec::with_capacity((latent.len() + x0.len()) * 4);
        for value in latent.iter().chain(x0.iter()) {
            body.extend_from_slice(&value.to_le_bytes());
        }
        let write = |writer: &mut std::io::BufWriter<std::fs::File>, bytes: &[u8]| {
            std::io::Write::write_all(writer, bytes).map_err(|err| err.to_string())
        };
        write(&mut self.writer, &head).and_then(|_| write(&mut self.writer, &body))
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.frames += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<PathBuf, String> {
        std::io::Write::flush(&mut self.writer)
            .map_err(|err| format!("failed to flush {}: {err}", self.path.display()))?;
        Ok(self.path)
    }
}

/// The `.batraw` header alone: how many samples, how big each one is, and how
/// many bytes one value takes.
///
/// A dataset's GPU footprint is decided by those numbers, and CIFAR-10 is
/// 195 MiB on disk — reading it whole to answer "how many chunks?" would make
/// `--resources` slower than the run it describes. The payload width is part of
/// the answer since BATRAW3: the same 50 000 images are 586 MiB of f32 or
/// 147 MiB of u8, and an inventory that assumed f32 would over-report a
/// BATRAW3 dataset's residency by four.
fn read_batraw_header(path: &Path) -> Result<(u64, u32, u32, u32, u64), String> {
    use std::io::Read;
    let mut file =
        fs::File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut header = [0u8; 24];
    file.read_exact(&mut header)
        .map_err(|err| format!("failed to read the header of {}: {err}", path.display()))?;
    let magic = &header[..8];
    let value_bytes: u64 = if magic == RAW_DATASET_MAGIC_BYTES {
        1
    } else if magic == RAW_DATASET_MAGIC_SIGNED || magic == RAW_DATASET_MAGIC_UNIT {
        4
    } else {
        return Err(format!("invalid magic in {}", path.display()));
    };
    let word = |i: usize| {
        u32::from_le_bytes([
            header[8 + i * 4],
            header[9 + i * 4],
            header[10 + i * 4],
            header[11 + i * 4],
        ])
    };
    Ok((word(0) as u64, word(1), word(2), word(3), value_bytes))
}

/// The dataset a `--resources` question is about: the one named on the command
/// line, else the one the model's config trains on. `None` when neither exists
/// on disk — the inventory then simply has no streamed post, and says so.
fn resources_dataset(
    explicit: Option<String>,
    config: &ModelConfig,
) -> Option<(PathBuf, batlab_core::DatasetSpec)> {
    let named = explicit.or_else(|| match &config.run.mode {
        RunMode::Train(train) => Some(train.dataset_path.clone()),
        _ => None,
    })?;
    let path = PathBuf::from(&named);
    let (count, width, height, channels, value_bytes) = read_batraw_header(&path).ok()?;
    Some((
        path,
        batlab_core::DatasetSpec {
            sample_count: count,
            sample_bytes: width as u64 * height as u64 * channels as u64 * value_bytes,
        },
    ))
}

/// See the DEV/CI note in `main`. Not reachable from the TUI — except that the
/// TUI shows the same page, from the same inventory, on the same key.
///
/// Prints what the model puts on the GPU and what crosses the host boundary.
/// Two questions, and the second is the one nobody could answer before: the
/// model is uploaded once and stays resident, the dataset is streamed one
/// 64 MiB chunk at a time, and the reverse chain pays a full CPU↔GPU round trip
/// on every one of its 256 steps because the latent lives on the CPU.
///
/// `--measure` is what makes those claims measurements instead of assertions:
/// it builds the model, runs real training steps and real reverse steps, and
/// reports the counters' difference. Without it the page is pure prediction and
/// prints no traffic table at all, rather than a plausible zero.
fn run_resources(args: &[String]) -> Result<(), String> {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    reject_unknown_flags(
        args,
        &[
            "--resources",
            "--batch",
            "--dataset",
            "--vram",
            "--device",
            "--width",
            "--steps",
            "--gpu-limits",
        ],
        &["--measure", "--no-gpu", "--inference"],
    )?;

    let model_name =
        flag("--resources").ok_or_else(|| "--resources requires a model name".to_string())?;
    let config_path = storage::model_config_path(&model_name)
        .map_err(|err| format!("failed to resolve config path: {err}"))?;
    let config = storage::load_model_config(&config_path)
        .map_err(|err| format!("failed to load {}: {err}", config_path.display()))?;

    // The batch the config trains at, unless the caller asks about another —
    // which is the whole point of the flag: "what would batch 64 cost?".
    let config_batch = match &config.run.mode {
        RunMode::Train(train) => train.batch_size.max(1),
        _ => 16,
    };
    let batch = match flag("--batch") {
        Some(value) => value
            .parse::<u32>()
            .map_err(|_| format!("--batch takes an integer, got `{value}`"))?
            .max(1),
        None => config_batch,
    };
    let width = flag("--width")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(96);
    let measure = args.iter().any(|a| a == "--measure");
    let measure_steps = flag("--steps")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);

    let dataset = resources_dataset(flag("--dataset"), &config);

    // The device: this machine unless the question is about another one.
    let hypothetical = flag("--device");
    let no_gpu = args.iter().any(|a| a == "--no-gpu");
    let (gpu, mut device) = if hypothetical.is_some() || no_gpu {
        (
            None,
            batlab_core::DeviceProfile::hypothetical(
                hypothetical.unwrap_or_else(|| "unnamed device (WebGPU default limits)".to_string()),
                None,
            ),
        )
    } else {
        let rt = tokio::runtime::Runtime::new().map_err(|err| format!("tokio runtime: {err}"))?;
        let gpu = Arc::new(rt.block_on(GpuContext::new_headless()));
        let profile = batlab_core::DeviceProfile::from_gpu(gpu.as_ref());
        (Some((rt, gpu)), profile)
    };
    if let Some(vram) = flag("--vram") {
        let gib: f64 = vram
            .parse()
            .map_err(|_| format!("--vram takes a size in GiB, got `{vram}`"))?;
        device = device.with_budget((gib * 1024.0 * 1024.0 * 1024.0) as u64);
    }

    let inference_only = args.iter().any(|a| a == "--inference");
    let workload = if inference_only {
        batlab_core::Workload::Inference
    } else {
        match &config.run.mode {
            RunMode::Train(train) => {
                batlab_core::Workload::training(train.optimizer, train.ema_decay.is_some())
            }
            // A model whose config is not a training config still answers the
            // training question — that is what someone sizing a machine asks.
            _ => batlab_core::Workload::training(OptimizerKind::Adam, false),
        }
    };

    let request = batlab_core::InventoryRequest {
        layers: config.layers.clone(),
        input_size: config.input_size,
        batch,
        workload,
        dataset: dataset.as_ref().map(|(_, spec)| *spec),
        live_frame: false,
    };
    let inventory = batlab_core::inventory(&request, &device)
        .map_err(|err| format!("failed to compute the inventory: {err}"))?;

    let measured = match (measure, gpu.as_ref()) {
        (true, Some((rt, gpu))) => Some(rt.block_on(measure_transfers(
            gpu,
            &config,
            batch,
            dataset.as_ref().map(|(path, _)| path.as_path()),
            measure_steps,
        ))?),
        (true, None) => {
            return Err("--measure needs a real GPU: drop --no-gpu/--device".to_string());
        }
        _ => None,
    };

    println!("model '{model_name}' — {}", config_path.display());
    println!();
    for line in batlab_core::report_lines(
        &inventory,
        measured.as_ref(),
        batlab_core::ReportOptions {
            width,
            max_layer_rows: None,
            bars: true,
        },
    ) {
        println!("{line}");
    }

    // The two questions the page exists for, answered last because they are
    // what someone screenshots: where the ceiling is, and what the other
    // workload costs.
    println!();
    match batlab_core::max_batch_that_fits(&request, &device, 4096) {
        Ok(Some(ceiling)) => println!(
            "BATCH CEILING  {ceiling}{}",
            if device.memory_budget.is_none() {
                "  (per-binding limits only — no memory budget was stated; \
                 pass --vram N to bound it)"
            } else {
                ""
            }
        ),
        Ok(None) => println!("BATCH CEILING  none — not even one sample fits"),
        Err(err) => println!("BATCH CEILING  unknown: {err}"),
    }
    let other = batlab_core::inventory(
        &request
            .clone()
            .with_workload(if inference_only {
                batlab_core::Workload::training(OptimizerKind::Adam, false)
            } else {
                batlab_core::Workload::Inference
            })
            .with_dataset(None),
        &device,
    )
    .map_err(|err| format!("failed to compute the paired inventory: {err}"))?;
    println!(
        "{:<14} {} ({})",
        if inference_only {
            "TRAINING"
        } else {
            "INFERENCE"
        },
        batlab_core::format_bytes(other.total_bytes()),
        other.workload.label()
    );
    Ok(())
}

/// Run the real thing and read the counters — the measured half of the page.
///
/// Deliberately runs *production* entry points: `DiffusionTask::train_step_
/// report_batch` for training and `reverse_step` for inference, the same calls
/// the trainer and the sampler make. A measurement of a special path measures
/// the special path.
///
/// The first step is measured separately and thrown away: it uploads the first
/// dataset chunk and warms every pipeline, so folding it into the average would
/// attribute a one-off 64 MiB to every step for ever.
async fn measure_transfers(
    gpu: &Arc<GpuContext>,
    config: &ModelConfig,
    batch: u32,
    dataset_path: Option<&Path>,
    steps: usize,
) -> Result<batlab_core::MeasuredTransfers, String> {
    let output_size = {
        let out = batlab_core::compute_inferred_input(&config.layers, config.input_size);
        (out.0, out.1, out.2)
    };

    let before_build = gpu.transfers();
    let mut model = Model::new_training_with_optimizer(
        Arc::clone(gpu),
        1e-3,
        batch,
        PLoss::MeanSquared,
        OptimizerKind::Adam,
    )
    .await;
    for draft in &config.layers {
        model.add_draft(draft).map_err(|err| err.to_string())?;
    }
    model.build().map_err(|err| err.to_string())?;
    let build_upload_bytes = gpu.transfers().since(before_build).host_to_device_bytes;

    let schedule = LinearNoiseSchedule::new_linear(
        DIFFUSION_SCHEDULE_STEPS,
        DIFFUSION_BETA_START,
        DIFFUSION_BETA_END,
    );
    let mut task = DiffusionTask::new(schedule.clone());

    let training_step = match dataset_path {
        Some(path) => {
            let samples = try_load_raw_dataset(path, output_size)?
                .ok_or_else(|| format!("no .batraw dataset at {}", path.display()))?;
            if samples.is_empty() {
                return Err("the dataset is empty".to_string());
            }
            let sample_len = samples.sample_len;
            let mut dataset = GpuDataset::from_payload(gpu.as_ref(), samples.payload, sample_len)
                .map_err(|err| format!("failed to upload the dataset: {err}"))?;

            // Warmup, then measure.
            task.train_step_report_batch(&mut model, &mut dataset, 0, batch as usize, 7)
                .map_err(|err| format!("training step failed: {err}"))?;
            let before = gpu.transfers();
            for step in 1..=steps {
                task.train_step_report_batch(&mut model, &mut dataset, step, batch as usize, 7)
                    .map_err(|err| format!("training step failed: {err}"))?;
            }
            Some(gpu.transfers().since(before).per(steps as u64))
        }
        None => None,
    };

    // Inference: the reverse chain, one step at a time, on the very function
    // the sampler uses.
    let input_channels = config.input_size.2 as usize;
    let signal_channels = output_size.2 as usize;
    let output_len = (output_size.0 * output_size.1 * output_size.2) as usize;
    let mut latent = schedule.sample_noise(output_len, 11);
    let reverse_steps = steps.min(schedule.len());
    // Warmup outside the measured window, same reason as above.
    latent = batlab_core::reverse_step(
        &mut model,
        &schedule,
        input_channels,
        signal_channels,
        &latent,
        schedule.len() - 1,
        11,
        1.0,
        PosteriorVariance::Beta,
        false,
    )
    .latent;
    let before = gpu.transfers();
    for index in 0..reverse_steps {
        latent = batlab_core::reverse_step(
            &mut model,
            &schedule,
            input_channels,
            signal_channels,
            &latent,
            schedule.len().saturating_sub(2 + index),
            11,
            1.0,
            PosteriorVariance::Beta,
            false,
        )
        .latent;
    }
    let inference_step = Some(gpu.transfers().since(before).per(reverse_steps as u64));

    Ok(batlab_core::MeasuredTransfers {
        training_step,
        inference_step,
        build_upload_bytes: Some(build_upload_bytes),
        steps_per_image: schedule.len(),
    })
}

/// See the DEV/CI note in `main`. Not reachable from the TUI.
///
/// Walks the same drift the TUI mode walks and writes one PNG per completed
/// cycle — a time-lapse of the wandering, without a window. It also reports the
/// mean absolute change between consecutive frames, which is the number that
/// says whether the run is *drifting* or merely redrawing the same image.
/// Spacing between the climb frames `--climb-frames n` writes: enough to land
/// about `n` of them across a climb of `ceiling + 1` increments.
///
/// `None` when nothing should be written, so the caller never divides by zero
/// on a degenerate request.
fn climb_stride(ceiling: usize, wanted: usize) -> Option<usize> {
    if wanted == 0 {
        return None;
    }
    Some((ceiling + 1).div_ceil(wanted).max(1))
}

fn run_headless_perpetual(args: &[String]) -> Result<(), String> {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    reject_unknown_flags(
        args,
        &[
            "--headless-perpetual",
            "--checkpoint",
            "--regime",
            "--t-r",
            "--t-star",
            "--depth",
            "--seed",
            "--magnitude",
            "--frames",
            "--actions",
            "--climb-frames",
            "--dump",
            "--out",
            "--seed-dataset",
            "--gpu-limits",
        ],
        &["--window", "--seed-noise", "--single-view", "--raw-weights"],
    )?;

    let model_name = flag("--headless-perpetual")
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| {
            "--headless-perpetual takes the model name as its value, e.g. \
             `--headless-perpetual Greyscale_Diffusion_L --regime flux`"
                .to_string()
        })?;
    let frames = flag("--frames")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
        .max(1);
    let depth = dial_level(args)?.unwrap_or_else(PerpetualConfig::default_renoise_depth);
    let regime = match flag("--regime") {
        Some(value) => batlab_core::PerpetualRegime::parse(&value)
            .ok_or_else(|| format!("invalid --regime: {value} (want wander|breathe|flux)"))?,
        None => batlab_core::PerpetualRegime::default(),
    };
    let seed = flag("--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let magnitude = flag("--magnitude")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);
    // `--window` also writes the composed x_t | x̂₀ frame the live visualiser
    // would be showing — the layout, out of the same function the window uses.
    let window_frames = args.iter().any(|arg| arg == "--window");
    // `--climb-frames N` writes roughly N composed frames per climb, so the
    // gradual dissolve can be read as a contact sheet instead of being taken on
    // trust. It is the only way to see the climb without a window.
    let climb_frames = flag("--climb-frames").and_then(|v| v.parse::<usize>().ok());
    // `--actions N` stops after N drift actions instead of after N closed
    // cycles. Flux never closes one — that is the point of it — so it is the
    // only way to bound a flux run, and it is also how the two regimes get
    // compared over the same number of frames.
    let action_budget = flag("--actions").and_then(|v| v.parse::<usize>().ok());
    // Refused rather than defaulted: `--frames` counts *closed cycles*, and flux
    // closes none. The old code would have spun on `written < frames` with
    // `written` stuck at zero — a run that never ends and never says why.
    if regime == batlab_core::PerpetualRegime::Flux && action_budget.is_none() {
        return Err(
            "--regime flux never closes a cycle, so --frames cannot bound it: pass --actions N"
                .to_string(),
        );
    }
    // `--dump <path>` writes every frame's two panes as raw f32, which is the
    // only honest substrate for the frame-to-frame measurements: |Δ| in flux is
    // of the order of one 8-bit level, so a PNG would quantise away the very
    // quantity being measured (`CLIMB_COHERENCE.md` §6).
    let dump_path = flag("--dump").map(PathBuf::from);
    // The run drifts away from a real image by default, like the TUI mode.
    // `--seed-noise` asks for the old opening — pure noise at the top of the
    // schedule — which is the only way to reproduce a pre-img2img campaign;
    // `--seed-dataset <path>` names the file to draw from instead of deriving
    // it from the model's output channels.
    let seed_from_noise = args.iter().any(|arg| arg == "--seed-noise");
    // `--window` writes the frame the live visualiser would be showing, so it
    // has to be able to write the layout the visualiser actually defaults to in
    // this mode: x̂₀ alone, square. `--single-view` asks for that one.
    let window_view = match args.iter().any(|arg| arg == "--single-view") {
        true => batlab_core::LiveView::X0Only,
        false => batlab_core::LiveView::Both,
    };
    let seed_dataset = flag("--seed-dataset");
    let weights = weights_source(args);
    let out_dir = flag("--out").map(PathBuf::from).unwrap_or_else(|| {
        storage::project_root()
            .join("perpetual_samples")
            .join(regime.label())
    });

    let config_path = storage::model_config_path(&model_name)
        .map_err(|err| format!("failed to resolve config path: {err}"))?;
    let config = storage::load_model_config(&config_path)
        .map_err(|err| format!("failed to load {}: {err}", config_path.display()))?;
    let checkpoint = flag("--checkpoint");

    let rt = tokio::runtime::Runtime::new().map_err(|err| format!("tokio runtime: {err}"))?;
    rt.block_on(async {
        let (_gpu, mut model) = build_execution_model(
            &config,
            INFERENCE_RUNTIME_LR,
            INFERENCE_RUNTIME_BATCH_SIZE,
            OptimizerKind::default(),
            WeightInit::default(),
            None,
        )
        .await?;
        let checkpoint_path = resolve_sampling_checkpoint_path(&config, checkpoint.as_deref())?;
        load_sampling_checkpoint(&mut model, &checkpoint_path, weights)?;

        let input_dims = model
            .input_dim()
            .ok_or_else(|| "model has no input dimensions".to_string())?;
        let output_dims = model
            .output_dim()
            .ok_or_else(|| "model has no output dimensions".to_string())?;
        let output_size = (output_dims.x, output_dims.y, output_dims.z);
        let output_len = (output_size.0 * output_size.1 * output_size.2) as usize;

        let schedule = LinearNoiseSchedule::new_linear(
            DIFFUSION_SCHEDULE_STEPS,
            DIFFUSION_BETA_START,
            DIFFUSION_BETA_END,
        );
        let seed_images = match seed_from_noise {
            true => None,
            // The model's `config_file` is consulted here too — it is handed
            // over whole, so this path cannot quietly skip it.
            false => SeedImages::resolve(seed_dataset.as_deref(), &config, output_size)?,
        };
        let mut drift = match seed_images.as_ref() {
            Some(_) => PerpetualDrift::from_image(schedule.len(), regime, depth, seed),
            None => PerpetualDrift::new(schedule.len(), regime, depth, seed),
        };
        // Same walk the TUI run uses, so a dump records the frames a window
        // would have shown — including the climb's now-live x̂₀.
        let opening = match seed_images.as_ref() {
            Some(images) => images.provide_x0(seed),
            None => schedule.sample_noise(output_len, drift.initial_noise_seed()),
        };
        let mut last_x0 = match seed_images.as_ref() {
            Some(_) => opening.clone(),
            None => vec![0.0f32; output_len],
        };
        let mut walk = DriftWalk::new(opening, magnitude);
        let mut previous_frame: Option<Vec<f32>> = None;
        let mut mid_captured = false;

        // The banner is the only place a caller can check that what it typed was
        // heard — which is why it reports the dial under the name the *regime*
        // gives it, and reads it back off the drift (post-clamp) rather than off
        // the parse. A banner frozen at `t_r=64` is how `--t-star` went a whole
        // campaign without being parsed.
        println!(
            "headless perpetual '{model_name}': regime={} {}={} bound={} \
             magnitude={magnitude} seed={seed}\nweights → {}",
            drift.regime().label(),
            drift.regime().depth_label(),
            drift.depth(),
            match action_budget {
                Some(budget) => format!("--actions {budget}"),
                None => format!("--frames {frames} (cycles)"),
            },
            checkpoint_path.display(),
        );
        // Which of the three sources won is part of the answer, not a detail:
        // "origine → image du dataset X" alone leaves the one question a
        // surprised reader has — why THAT dataset — unanswered, and that is
        // precisely how a specialised model drifted away from CIFAR trucks
        // without anything on screen looking wrong.
        match seed_images.as_ref() {
            Some(images) => println!(
                "origine → image du dataset {} ({} images) [{}] — dérive img2img",
                images.path().display(),
                images.len(),
                images.source().label()
            ),
            None => println!(
                "origine → bruit pur en haut du schedule ({})",
                match seed_from_noise {
                    true => "--seed-noise",
                    false => "aucun dataset de graine trouvable",
                }
            ),
        }
        // Announced only when something will actually land there. A PNG is
        // written when a cycle closes; flux closes none, so naming a directory
        // it never even creates reads as a run that failed to write.
        if drift.regime() == batlab_core::PerpetualRegime::Flux {
            println!("frames  → no PNG in flux (no cycle ever closes) — use --dump");
        } else {
            println!("frames  → {}", out_dir.display());
        }

        let mut dump = match dump_path.as_ref() {
            Some(path) => Some(FrameDump::create(path, output_size)?),
            None => None,
        };

        let started = std::time::Instant::now();
        let mut steps = 0usize;
        let mut written = 0usize;
        let mut actions = 0usize;
        while match action_budget {
            Some(budget) => actions < budget,
            None => written < frames,
        } {
            actions += 1;
            // Handed whole to `record` below, which names the frame from it —
            // never from `drift.phase()` read beforehand, which `step`
            // reconciles on its way in and so names the phase just *left*.
            let action = drift.step();
            let level = match action {
                DriftAction::Descend { diffusion_step, .. } => diffusion_step.saturating_sub(1),
                DriftAction::Climb { forward_step, .. } => forward_step,
                DriftAction::Flux { diffusion_step, .. } => diffusion_step,
            };

            // A cycle closes on the increment that opens the climb, and what it
            // settled on is what the run holds *before* that increment moves
            // anything: the latent at the floor and the estimate beside it. So
            // the PNGs are written here, ahead of the advance.
            if let DriftAction::Climb {
                forward_step,
                opens_cycle: true,
                ..
            } = action
            {
                let path = write_tensor_png(
                    &last_x0,
                    output_size,
                    &out_dir.join(format!("{:03}.png", written)),
                )?;
                // …and, on request, the frame the live window would be showing
                // at this instant. In errance the cycle closes at t=0, where the
                // sampler's output IS its x̂₀ estimate, so the two panes must
                // land on the same image — the visual half of the check
                // `identical_sources_produce_two_identical_panes` makes on
                // synthetic data.
                if window_frames {
                    write_tensor_png(
                        &compose_live_frame_view(
                            walk.latent(),
                            &last_x0,
                            output_size.0,
                            output_size.1,
                            output_size.2,
                            window_view,
                        ),
                        (
                            window_view.frame_width(output_size.0),
                            output_size.1,
                            output_size.2,
                        ),
                        &out_dir.join(format!("window_{:03}.png", written)),
                    )?;
                }
                let change = previous_frame.as_ref().map(|prev| {
                    prev.iter()
                        .zip(last_x0.iter())
                        .map(|(a, b)| (a - b).abs() as f64)
                        .sum::<f64>()
                        / last_x0.len() as f64
                });
                println!(
                    "  cycle {:>3} → {}  (mean |Δ| vs previous frame: {})",
                    drift.cycle(),
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    change.map_or("—".to_string(), |c| format!("{c:.4}"))
                );
                previous_frame = Some(last_x0.clone());
                written += 1;
                mid_captured = false;
                let _ = forward_step;
            }

            let frame = walk.advance(
                action,
                &schedule,
                &mut ModelNoise {
                    model: &mut model,
                    schedule: &schedule,
                    input_channels: input_dims.z as usize,
                    signal_channels: output_dims.z as usize,
                },
            );
            steps += frame.model_calls;
            let latent = frame.latent;
            last_x0 = frame.x0_hat;

            match action {
                DriftAction::Descend { diffusion_step, .. } => {
                    // One mid-descent frame per cycle: the moment the panes are
                    // most unlike each other — x_t still visibly noisy, x̂₀
                    // already an image. That contrast is what a user reads as
                    // "the two halves are decorrelated", so it is the frame the
                    // rule has to survive.
                    if window_frames && !mid_captured && diffusion_step * 2 <= drift.depth() {
                        write_tensor_png(
                            &compose_live_frame_view(
                                &latent,
                                &last_x0,
                                output_size.0,
                                output_size.1,
                                output_size.2,
                                window_view,
                            ),
                            (
                                window_view.frame_width(output_size.0),
                                output_size.1,
                                output_size.2,
                            ),
                            &out_dir
                                .join(format!("window_{:03}_mid_t{diffusion_step}.png", written)),
                        )?;
                        mid_captured = true;
                    }
                }
                DriftAction::Climb { forward_step, .. } => {
                    // Sampled rather than exhaustive — a t_r = 64 climb is 65
                    // frames and a contact sheet wants a handful. The opening
                    // increment is always written, so the sheet starts on the
                    // first *dissolved* frame rather than on the settled image
                    // (which `window_NNN.png` already holds).
                    if let Some(count) = climb_frames
                        && (forward_step == drift.departure_step()
                            || climb_stride(drift.ceiling(), count)
                                .is_some_and(|stride| forward_step % stride == 0))
                    {
                        write_tensor_png(
                            &compose_live_frame_view(
                                &latent,
                                &last_x0,
                                output_size.0,
                                output_size.1,
                                output_size.2,
                                window_view,
                            ),
                            (
                                window_view.frame_width(output_size.0),
                                output_size.1,
                                output_size.2,
                            ),
                            &out_dir.join(format!("climb_{written:03}_t{forward_step:03}.png")),
                        )?;
                    }
                }
                DriftAction::Flux { .. } => {}
            }

            if let Some(dump) = dump.as_mut() {
                dump.record(action, level, &latent, &last_x0)?;
            }
        }
        if let Some(dump) = dump.take() {
            let path = dump.finish()?;
            println!("  dump → {}", path.display());
        }

        let elapsed = started.elapsed().as_secs_f32();
        // The two counts used to differ, and the difference was the whole cost
        // model: a climb increment was closed-form arithmetic and called no
        // model, so a climbing run cost fewer model calls than actions. It is
        // also exactly what made x̂₀ freeze on the way up. They are now equal by
        // construction — one call per frame, every frame — and both are printed
        // so a run where they diverge again is visible at a glance.
        println!(
            "{actions} actions, {steps} model calls, in {elapsed:.2} s \
             ({:.0} calls/s, unthrottled)",
            steps as f32 / elapsed.max(f32::EPSILON)
        );
        Ok::<(), String>(())
    })
}

fn run_execution_loop(mut config: ModelConfig) {
    loop {
        let prepared = match normalize_config_for_models_layout(&mut config) {
            Ok(prepared) => prepared,
            Err(err) => {
                eprintln!("failed to prepare model persistence: {err}");
                break;
            }
        };

        let (tx, rx) = std::sync::mpsc::channel::<tui::TrainingEvent>();
        let (control_tx, control_rx) = std::sync::mpsc::channel::<tui::TrainingControlCommand>();
        // Both training and perpetual runs are steered while they run; a plain
        // inference has nothing to steer.
        let is_steerable = matches!(&config.run.mode, RunMode::Train(_) | RunMode::Perpetual(_));
        let config_clone = config.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(async {
                let run_result = match config_clone.run.mode.clone() {
                    RunMode::Train(train_cfg) => {
                        // `--resume` and `--checkpoint-every` are DEV/CI flags:
                        // the TUI walks the weight selector for the first and
                        // has `[s]` in the monitor for the second.
                        //
                        // The config now names where this run *writes* (its own
                        // dated file); what the selector picked comes in beside
                        // it as the file to *read*. Training again never writes
                        // back over the weights it continued from.
                        let options = RunOptions {
                            load_from: prepared.load_from.clone(),
                            maintain_latest: prepared.maintain_latest,
                            ..RunOptions::default()
                        };
                        run_training(config_clone, train_cfg, options, &tx, control_rx).await
                    }
                    RunMode::Perpetual(perpetual_cfg) => {
                        run_perpetual(config_clone, perpetual_cfg, &tx, control_rx).await
                    }
                    RunMode::Infer => run_inference(config_clone, &tx).await,
                };
                if let Err(message) = run_result {
                    let _ = tx.send(tui::TrainingEvent::Error { message });
                    let _ = tx.send(tui::TrainingEvent::Done);
                }
            });
        });

        let maybe_control_tx = if is_steerable { Some(control_tx) } else { None };
        match tui::run_monitor(config.clone(), rx, maybe_control_tx) {
            Ok(MonitorOutcome::Restart(new_config)) => {
                config = new_config;
            }
            _ => break,
        }
    }
}

/// What preparing a run resolved that the config itself cannot hold.
#[derive(Debug, Clone, Default)]
struct PreparedRun {
    /// The checkpoint the weight selector pointed at — the file this run
    /// *reads*. It is not written into the `config_file`: which file tonight's
    /// run happened to continue from is not a property of the model, and the
    /// config's own `checkpoint_path` now records where the run *writes*.
    load_from: Option<PathBuf>,
    /// Whether `latest.ckpt` should follow what this run writes. True exactly
    /// when the path below was dated here.
    maintain_latest: bool,
}

fn normalize_config_for_models_layout(config: &mut ModelConfig) -> Result<PreparedRun, String> {
    let model_name = match config.model_name.clone() {
        Some(name) => name,
        None => {
            let generated = storage::next_model_name()
                .map_err(|err| format!("failed to allocate model name: {err}"))?;
            config.model_name = Some(generated.clone());
            generated
        }
    };

    // A training run gets its own dated file, resolved once here — including
    // once per restart from the monitor, so the evening's second run gets a
    // second file instead of reopening the first one's. The path the selector
    // left in the config becomes the run's *load* source; the config is then
    // rewritten naming the file this run will write, so reopening the model
    // tomorrow lands on the weights this run produced and not on the ones it
    // started from.
    let mut prepared = PreparedRun::default();
    if let RunMode::Train(train) = &mut config.run.mode {
        let dated = storage::new_run_checkpoint_path(&model_name, SystemTime::now())
            .map_err(|err| format!("failed to resolve checkpoint path: {err}"))?;
        prepared.load_from = train.checkpoint_path.take().map(PathBuf::from);
        prepared.maintain_latest = true;
        train.checkpoint_path = Some(dated.to_string_lossy().to_string());
    }

    storage::write_model_config(&model_name, config)
        .map_err(|err| format!("failed to write model config for '{model_name}': {err}"))?;
    Ok(prepared)
}

// ---------------------------------------------------------------------------
// DEV/CI ONLY — where a training step's time goes, pass by pass.
// ---------------------------------------------------------------------------

/// `--profile-step <model>`: one training step, timed by the GPU itself.
///
/// Why a subcommand of its own rather than a `--profile` on `--headless-train`:
/// the training loop reports a loss, runs a per-bucket probe and samples an
/// image on a schedule, all of which submit their own GPU work. A profile of
/// "one step" that silently included a 256-step denoising chain every 200 steps
/// would be a profile of something else. This harness runs the *same*
/// production call the trainer runs — `DiffusionTask::train_step_batch` — and
/// nothing else, the way `measure_transfers` does for traffic.
fn run_profile_step(args: &[String]) -> Result<(), String> {
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    reject_unknown_flags(
        args,
        &[
            "--profile-step",
            "--batch",
            "--dataset",
            "--optimizer",
            "--lr",
            "--rounds",
            "--warmup",
            "--top",
            "--gpu-limits",
        ],
        &["--ema"],
    )?;

    let model_name =
        flag("--profile-step").ok_or_else(|| "--profile-step requires a model name".to_string())?;
    let config_path = storage::model_config_path(&model_name)
        .map_err(|err| format!("failed to resolve config path: {err}"))?;
    let config = storage::load_model_config(&config_path)
        .map_err(|err| format!("failed to load {}: {err}", config_path.display()))?;

    let config_batch = match &config.run.mode {
        RunMode::Train(train) => train.batch_size.max(1),
        _ => 16,
    };
    let batch = match flag("--batch") {
        Some(value) => value
            .parse::<u32>()
            .map_err(|_| format!("--batch takes an integer, got `{value}`"))?
            .max(1),
        None => config_batch,
    };
    let optimizer = match flag("--optimizer") {
        Some(value) => OptimizerKind::parse(&value)
            .ok_or_else(|| format!("invalid value for --optimizer: {value} (want sgd|adam)"))?,
        None => OptimizerKind::Adam,
    };
    let lr = match flag("--lr") {
        Some(value) => value
            .parse::<f32>()
            .map_err(|_| format!("--lr takes a number, got `{value}`"))?,
        None => 1e-3,
    };
    let rounds = match flag("--rounds") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| format!("--rounds takes an integer, got `{value}`"))?
            .max(1),
        None => 5,
    };
    // Two warmups is not a ritual: the first step compiles every pipeline (see
    // `PERF_CONVOLUTION.md` §1, where the compile of the first dispatch was
    // charged to the first layer and inverted the whole profile), and the first
    // step is also the one that uploads the dataset chunk.
    let warmup = match flag("--warmup") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| format!("--warmup takes an integer, got `{value}`"))?,
        None => 3,
    };
    let top = match flag("--top") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| format!("--top takes an integer, got `{value}`"))?,
        None => usize::MAX,
    };
    let dataset_path = flag("--dataset")
        .ok_or_else(|| "--profile-step needs --dataset <path to .batraw>".to_string())?;
    let ema = args.iter().any(|arg| arg == "--ema");

    // Before the first adapter request — the feature belongs to the device.
    batlab_core::request_pass_profiling(true);

    let rt = tokio::runtime::Runtime::new().map_err(|err| format!("tokio runtime: {err}"))?;
    rt.block_on(async move {
        let (gpu, mut model) = build_execution_model(
            &config,
            lr,
            batch,
            optimizer,
            WeightInit::default(),
            ema.then(|| EmaConfig::new(0.999).expect("0.999 is a valid decay")),
        )
        .await?;

        let output_dims = model
            .output_dim()
            .ok_or_else(|| "model has no output dimensions".to_string())?;
        let output_size = (output_dims.x, output_dims.y, output_dims.z);
        let dataset = load_dataset(&dataset_path, output_size)?;
        let sample_len = (output_size.0 * output_size.1 * output_size.2) as usize;
        let mut gpu_dataset = GpuDataset::from_payload(gpu.as_ref(), dataset.payload, sample_len)
            .map_err(|err| format!("failed to upload dataset to GPU: {err}"))?;

        let schedule = LinearNoiseSchedule::new_linear(
            DIFFUSION_SCHEDULE_STEPS,
            DIFFUSION_BETA_START,
            DIFFUSION_BETA_END,
        );
        let mut task = DiffusionTask::new(schedule);

        println!(
            "PASS PROFILE  {model_name} · batch {batch} · {} · {rounds} armed steps after \
             {warmup} warmup · limits={} · dataset {}",
            optimizer.label(),
            gpu.limits_profile().label(),
            if gpu_dataset.is_resident() {
                "resident".to_string()
            } else {
                format!("{} chunks", gpu_dataset.chunk_count())
            }
        );
        // A profile taken under a swept knob is a profile of a different
        // machine's settings; say so on the line above the table, or the number
        // pasted into a report will read as the stock one.
        for line in batlab_core::tuning::overrides_in_force() {
            println!("              tuning override · {line}");
        }
        if !gpu.can_profile() {
            return Err(
                "this device has no TIMESTAMP_QUERY, so there is no per-pass GPU time to \
                 report. Rather than print host-side guesses dressed as GPU measurements, \
                 this command stops here: use `--headless-train` and time whole steps."
                    .to_string(),
            );
        }

        let mut step = 0usize;
        for _ in 0..warmup {
            task.train_step_batch(&mut model, &mut gpu_dataset, step, batch as usize, 7)
                .map_err(|err| format!("training step failed: {err}"))?;
            step += 1;
        }
        // The warmups are asynchronous — nothing above waited for the GPU. Drain
        // the queue before the first armed round, or its host wall clock would
        // include three steps of somebody else's work.
        gpu.wait_idle();

        let mut runs = Vec::with_capacity(rounds);
        let mut walls = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            gpu.arm_profiler();
            let started = std::time::Instant::now();
            task.train_step_batch(&mut model, &mut gpu_dataset, step, batch as usize, 7)
                .map_err(|err| format!("training step failed: {err}"))?;
            let encoded = started.elapsed();
            // `collect_profile` blocks on the submission, so the wall clock
            // below is host time around a step that has actually finished.
            let run = gpu
                .collect_profile()
                .ok_or_else(|| "the profiler recorded nothing".to_string())?;
            walls.push((started.elapsed(), encoded));
            runs.push(run);
            step += 1;
        }

        print_pass_profile(&batlab_core::ProfileSummary::reduce(&runs), &walls, top);
        Ok(())
    })
}

/// The table the mission asks for: sorted by cost, with the budget under it.
fn print_pass_profile(
    summary: &batlab_core::ProfileSummary,
    walls: &[(Duration, Duration)],
    top: usize,
) {
    let ms = |nanos: u64| nanos as f64 / 1e6;
    let floor = summary.attributed_floor_nanos();
    println!();
    println!(
        "{:>9}  {:>6}  {:>7}  {:>10}  {:>3}  {}",
        "min ms", "% Σ", "max/min", "workgroups", "n", "pass"
    );
    let shown = summary.passes.len().min(top);
    for pass in summary.passes.iter().take(shown) {
        println!(
            "{:>9.3}  {:>5.1}%  {:>6.2}×  {:>10}  {:>3}  {}{}",
            ms(pass.min_nanos),
            100.0 * pass.min_nanos as f64 / floor.max(1) as f64,
            pass.max_nanos as f64 / pass.min_nanos.max(1) as f64,
            pass.workgroups,
            pass.invocations,
            pass.label,
            // Never silently: a pass the backend refused to time reads 0.000 ms,
            // which is indistinguishable from a fast one unless it says so.
            match pass.rounds_sampled {
                0 => "   ⚠ NEVER TIMED by this backend".to_string(),
                n if (n as usize) < summary.rounds =>
                    format!("   (timed in {n}/{} rounds)", summary.rounds),
                _ => String::new(),
            }
        );
    }
    if shown < summary.passes.len() {
        let rest: u64 = summary.passes[shown..].iter().map(|p| p.min_nanos).sum();
        println!(
            "{:>9.3}  {:>5.1}%  {:>7}  {:>10}  {:>3}  … {} more passes",
            ms(rest),
            100.0 * rest as f64 / floor.max(1) as f64,
            "",
            "",
            summary.passes.len() - shown,
            summary.passes.len() - shown
        );
    }

    // The budget, and the budget is the point.
    //
    // ONE round, not three minima. The estimator everywhere else in this project
    // is the minimum over rounds, and it is the right one for a *single* number
    // — but Σ passes, span and wall are three views of the SAME step, and
    // minimising each independently mixes rounds. It printed "−0.9 ms outside
    // the GPU span" on the first sweep: a host clock that finished before the
    // GPU started, which is not a discovery about latency, it is two different
    // steps subtracted from each other. So the reference round is picked once —
    // the fastest by wall clock, the closest thing to an uncontended step — and
    // all three lines come from it. The spread across rounds is printed beside
    // the span so a reader can see how much that choice was worth.
    let reference = walls
        .iter()
        .enumerate()
        .min_by_key(|(_, (wall, _))| *wall)
        .map(|(index, _)| index)
        .unwrap_or(0);
    let span = summary.round_span.get(reference).copied().unwrap_or(0);
    let attributed = summary.round_attributed.get(reference).copied().unwrap_or(0);
    let (wall, encode) = walls.get(reference).copied().unwrap_or_default();
    let span_lo = summary.round_span.iter().copied().min().unwrap_or(0);
    let span_hi = summary.round_span.iter().copied().max().unwrap_or(0);
    println!();
    println!(
        "BUDGET  (round {} of {}, the fastest by wall clock)",
        reference + 1,
        summary.rounds
    );
    println!(
        "  Σ pass minima              {:>10.1} ms   {} passes timed, over all rounds",
        ms(floor),
        summary.passes.iter().map(|p| p.invocations).sum::<u32>()
    );
    println!("  Σ passes, this round       {:>10.1} ms", ms(attributed));
    println!(
        "  GPU span, this round       {:>10.1} ms   → {:.1} ms ({:.1}%) BETWEEN the passes",
        ms(span),
        ms(span.saturating_sub(attributed)),
        100.0 * span.saturating_sub(attributed) as f64 / span.max(1) as f64,
    );
    println!(
        "  host wall, this round      {:>10.1} ms   → {:.1} ms outside the GPU span",
        wall.as_secs_f64() * 1e3,
        wall.as_secs_f64() * 1e3 - ms(span),
    );
    println!(
        "  of which CPU encoding      {:>10.1} ms   (the step returns here; the GPU is still running)",
        encode.as_secs_f64() * 1e3,
    );
    println!(
        "  span across all rounds     {:>10.1} ms … {:.1} ms  (spread {:.1}%)",
        ms(span_lo),
        ms(span_hi),
        100.0 * (span_hi.saturating_sub(span_lo)) as f64 / span_lo.max(1) as f64,
    );
    if summary.dropped > 0 {
        println!(
            "  ⚠ {} passes could not be timed (query set full) — every total above is a \
             LOWER bound and the gap between them is overstated.",
            summary.dropped
        );
    }
    let unsampled = summary
        .round_unsampled
        .get(reference)
        .copied()
        .unwrap_or(0);
    if unsampled > 0 {
        println!(
            "  ⚠ {unsampled} of this round's passes were NOT sampled by the backend (Metal \n\
             \x20   drops counter samples on some small dispatches). They are excluded from the \n\
             \x20   span and contribute 0 to Σ, so 'BETWEEN the passes' is an OVERestimate here."
        );
    }
}

async fn build_execution_model(
    config: &ModelConfig,
    lr: f32,
    batch_size: u32,
    optimizer: OptimizerKind,
    weight_init: WeightInit,
    ema: Option<EmaConfig>,
) -> Result<(Arc<GpuContext>, Model<Training>), String> {
    let gpu = Arc::new(GpuContext::new_headless().await);
    let mut model = Model::new_training_with_optimizer(
        gpu.clone(),
        lr,
        batch_size,
        PLoss::MeanSquared,
        optimizer,
    )
    .await;
    model.set_weight_init(weight_init);
    // Before build(): the shadow buffers are allocated with the optimiser
    // passes and seeded from the weights that exist then.
    model.set_ema(ema);
    for draft in &config.layers {
        append_layer(&mut model, draft).map_err(|err| err.to_string())?;
    }
    model.build().map_err(|err| err.to_string())?;
    Ok((gpu, model))
}

/// Where `--headless-train` writes when `--out` is absent: a per-model scratch
/// directory under the system temp dir.
///
/// Scratch, and it must stay scratch — a CI run that trained into
/// `Models/<name>/pretrained_weights/` would quietly become the model's weights.
/// One directory per model so the dated names, which are only unique per
/// directory, cannot collide between two models trained in the same minute.
fn headless_scratch_dir(model_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("batlab-{model_name}"))
}

/// The file a headless run writes, and whether `latest.ckpt` follows it.
///
/// `--out` is the user's own filing: written exactly where and exactly as
/// named, with no dating and no `latest.ckpt` dropped beside it — naming the
/// file *is* how one opts out of the convention. Without it the run still
/// writes to scratch, never into a model's saved weights, but under the dated
/// name every run now carries.
fn headless_checkpoint_target(
    explicit_out: Option<String>,
    model_name: &str,
    at: SystemTime,
) -> Result<(PathBuf, bool), String> {
    match explicit_out {
        Some(path) => Ok((PathBuf::from(path), false)),
        None => {
            let dir = headless_scratch_dir(model_name);
            fs::create_dir_all(&dir).map_err(|err| {
                format!("failed to create scratch directory {}: {err}", dir.display())
            })?;
            Ok((storage::new_run_checkpoint_path_in(&dir, at), true))
        }
    }
}

/// What a run does on top of its `TrainingConfig`, and only ever from the
/// headless entry point.
///
/// A separate struct rather than three more fields on `TrainingConfig`: that
/// one is serialised into every model's `config_file`, and neither "the file I
/// happened to resume from tonight" nor "how often to write a partial" is a
/// property of the model.
#[derive(Debug, Clone, Default)]
struct RunOptions {
    /// `--resume <ckpt>`: pick the weights, the Adam moments, the step counter
    /// and the EMA out of this file and carry on.
    resume_from: Option<PathBuf>,
    /// `--checkpoint-every N`: how often a partial checkpoint is written.
    checkpoint_every: Option<usize>,
    /// The checkpoint this run **reads**, when that is not where it writes.
    ///
    /// The TUI's weight selector picks a file to continue from; the run then
    /// writes its own dated one. `None` reads from wherever it writes, which is
    /// what `--resume`-less headless runs and legacy configs do.
    load_from: Option<PathBuf>,
    /// Where this run **writes**, when that is not where it reads.
    ///
    /// A run used to write back over the file it continued from, which is how
    /// an evening of training erased the morning's. The write target is now a
    /// dated file of the run's own (`run-<stamp>.ckpt`), while
    /// `TrainingConfig::checkpoint_path` stays what it always was: the file the
    /// run *loads*. `None` keeps the old behaviour — that is `--out`, where the
    /// user named the file and nothing may rename it.
    write_to: Option<PathBuf>,
    /// Point `latest.ckpt` at every checkpoint this run writes, beside it.
    ///
    /// Set exactly when `write_to` is a name batlab chose: `--out` is the
    /// user's own filing and gets no extra file dropped next to it.
    maintain_latest: bool,
}

async fn run_training(
    config: ModelConfig,
    train_cfg: TrainingConfig,
    options: RunOptions,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
    control_rx: Receiver<tui::TrainingControlCommand>,
) -> Result<(), String> {
    let ema = match train_cfg.ema_decay {
        Some(decay) => Some(EmaConfig::new(decay).ok_or_else(|| {
            format!("invalid EMA decay {decay}: it must be strictly between 0 and 1")
        })?),
        None => None,
    };
    let (gpu, mut model) = build_execution_model(
        &config,
        train_cfg.lr,
        train_cfg.batch_size,
        train_cfg.optimizer,
        train_cfg.weight_init,
        ema,
    )
    .await?;
    let checkpoint_path = match train_cfg.checkpoint_path.as_deref() {
        Some(path) => Some(PathBuf::from(path)),
        None => config
            .model_name
            .as_deref()
            .map(storage::default_model_checkpoint_path)
            .transpose()
            .map_err(|err| format!("failed to resolve checkpoint path: {err}"))?,
    };
    // Read here, written there. Everything below that *saves* — the final
    // checkpoint, the `--checkpoint-every` partials, `[s]` in the monitor — goes
    // to `write_path`, and the metrics journal follows it, so a run's diagnostics
    // sit beside the weights that run produced.
    let write_path = options
        .write_to
        .clone()
        .or_else(|| checkpoint_path.clone());
    let load_path = options
        .load_from
        .clone()
        .or_else(|| checkpoint_path.clone());

    // `--resume` reads from a file of its own and writes to `--out`, so a
    // resumed run never overwrites the checkpoint it came from unless it is
    // asked to. It takes precedence over the config's own load mode: it is an
    // explicit command-line instruction and the config is a default.
    if let Some(resume) = options.resume_from.as_ref() {
        if !resume.exists() {
            return Err(format!("--resume: no such checkpoint: {}", resume.display()));
        }
        // Raw, not Ema: training continues from the iterate the optimiser left
        // — Adam's moments describe that iterate, and the average is not a
        // point the optimiser ever visited.
        let report = model
            .load_checkpoint_with(resume, batlab_core::CheckpointWeights::Raw)
            .map_err(|err| {
                format!(
                    "--resume {}: incompatible checkpoint ({err}). A checkpoint only \
                     loads into the architecture it was written from.",
                    resume.display()
                )
            })?;
        println!(
            "[resume] {} — optimiser step {}{}",
            resume.display(),
            model.optimizer_step(),
            match (report.carries_ema, ema.is_some()) {
                (true, true) => format!(
                    ", EMA restored (file decay {:.5}, this run {:.5})",
                    report.ema_decay.unwrap_or_default(),
                    ema.map(|e| e.decay).unwrap_or_default()
                ),
                (true, false) =>
                    ", WARNING: the file carries an EMA and this run keeps none — \
                     the average will NOT be carried into the checkpoint this run writes"
                        .to_string(),
                (false, true) => ", no EMA in the file: the shadow starts on these weights"
                    .to_string(),
                (false, false) => String::new(),
            }
        );
    } else if let Some(path) = load_path.as_ref() {
        if train_cfg.load_checkpoint {
            if !path.exists() {
                return Err(format!(
                    "selected checkpoint does not exist: {}",
                    path.display()
                ));
            }
            model
                .load_checkpoint(path)
                .map_err(|err| format!("failed to load checkpoint {}: {err}", path.display()))?;
        } else if train_cfg.checkpoint_path.is_none() && path.exists() {
            // Backward compatibility for legacy configs without explicit load mode.
            model
                .load_checkpoint(path)
                .map_err(|err| format!("failed to load checkpoint {}: {err}", path.display()))?;
        }
    }

    let input_dims = model
        .input_dim()
        .ok_or_else(|| "model has no input dimensions".to_string())?;
    let output_dims = model
        .output_dim()
        .ok_or_else(|| "model has no output dimensions".to_string())?;
    let input_size = (input_dims.x, input_dims.y, input_dims.z);
    let output_size = (output_dims.x, output_dims.y, output_dims.z);

    let schedule = LinearNoiseSchedule::new_linear(
        DIFFUSION_SCHEDULE_STEPS,
        DIFFUSION_BETA_START,
        DIFFUSION_BETA_END,
    );
    let mut trainer = Trainer::new(DiffusionTask::new_with_weighting(
        schedule,
        train_cfg.loss_weighting,
    ));
    trainer
        .configure_for_model(&model)
        .map_err(|err| format!("failed to configure diffusion task: {err}"))?;
    let diffusion = trainer.task().schedule().clone();
    {
        // Re-emits the weighting the training loop actually built, and the share
        // of draws it sends to each probe bucket — the number the report reads.
        let mass = trainer.task().timestep_bucket_mass(4);
        let mass = if mass.is_empty() {
            "uniform (0.25 each)".to_string()
        } else {
            mass.iter()
                .map(|m| format!("{m:.4}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "[weighting] {} — timestep mass per bucket: {mass}",
            train_cfg.loss_weighting.label()
        );
    }

    let dataset = load_dataset(&train_cfg.dataset_path, output_size)?;
    let sample_len = (output_size.0 * output_size.1 * output_size.2) as usize;
    // CPU copies of a handful of clean targets kept for the diagnostic probe
    // (the GPU dataset is opaque to CPU-side readback). Widened one by one:
    // an 8-bit dataset is not decoded on the host for anything else.
    let probe_config = ProbeConfig::default();
    let probe_samples: Vec<Vec<f32>> = (0..probe_config.sample_count.min(dataset.len()))
        .map(|index| dataset.sample(index))
        .collect();
    let mut gpu_dataset = GpuDataset::from_payload(gpu.as_ref(), dataset.payload, sample_len)
        .map_err(|err| format!("failed to upload dataset to GPU: {err}"))?;

    // The two numbers that decide what a step costs before it has computed
    // anything, printed together because one explains the other: the limits the
    // device was GRANTED, and whether the dataset fits inside them.
    //
    // A run whose dataset is resident pays its upload once and then nothing;
    // one whose dataset is streamed pays a chunk on most steps. That difference
    // is worth more than any optimiser flag on a large corpus, and it used to
    // be invisible — the engine always asked for the WebGPU baseline, so the
    // answer was always "streamed" and nobody had a reason to ask.
    {
        use batlab_core::format_bytes;
        let limits = gpu.device().limits();
        let residency = if gpu_dataset.is_resident() {
            "ONE resident chunk — uploaded once, no dataset traffic per step".to_string()
        } else {
            format!(
                "{} chunks of {} — streamed, a shuffled batch re-uploads the ones it lands in",
                gpu_dataset.chunk_count(),
                format_bytes(gpu_dataset.gpu_buffer_bytes())
            )
        };
        println!(
            "[gpu] limits={} (buffer {}, storage binding {}) · dataset {} in {residency}",
            gpu.limits_profile().label(),
            format_bytes(limits.max_buffer_size),
            format_bytes(limits.max_storage_buffer_binding_size as u64),
            format_bytes(gpu_dataset.payload_bytes()),
        );
    }

    // Metrics land next to the checkpoint (`<stem>_metrics.jsonl`), or in a
    // temp file when no checkpoint path is configured. Truncated per run so the
    // file always describes the current run only.
    let mut metrics = {
        let metrics_path = write_path
            .as_ref()
            .map(|ckpt| {
                let stem = ckpt
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "run".to_string());
                ckpt.with_file_name(format!("{stem}_metrics.jsonl"))
            })
            .unwrap_or_else(|| std::env::temp_dir().join("diffusion_metrics.jsonl"));
        MetricsLogger::create(&metrics_path, true)
    };
    println!(
        "[metrics] writing training diagnostics to {}",
        metrics.path()
    );
    let limits = gpu.device().limits();
    let estimated_training_bytes = model
        .estimated_gpu_bytes()
        .saturating_add(gpu_dataset.gpu_buffer_bytes())
        .saturating_add(
            trainer
                .task()
                .estimated_prepare_gpu_bytes_for_batch(output_dims, train_cfg.batch_size.max(1)),
        );
    let _ = tx.send(tui::TrainingEvent::ResourceReport {
        max_buffer_bytes: limits.max_buffer_size,
        max_storage_binding_bytes: limits.max_storage_buffer_binding_size as u64,
        estimated_training_bytes,
    });
    let sample_dir = prepare_sample_dir(&train_cfg.dataset_path)?;
    let mut current_lr = train_cfg.lr;
    let mut current_batch_size = train_cfg.batch_size.max(1);
    let mut total_steps = train_cfg.steps.max(1);
    let mut paused = false;
    let _ = tx.send(tui::TrainingEvent::TrainingState {
        paused,
        lr: current_lr,
        batch_size: current_batch_size,
        total_steps,
    });

    if let Some(output_buf) = model.last_output_buffer() {
        let title = format!(
            "Model Output  ({}×{}×{} channels)",
            output_size.0, output_size.1, output_size.2
        );
        tui::register_visualiser_source(
            model.gpu_context(),
            output_buf,
            output_size.0,
            output_size.1,
            output_size.2,
            title,
        );
    } else {
        eprintln!("[visualiser] model has no output buffer yet");
    }

    let mut step = 0usize;
    while step < total_steps {
        while let Ok(command) = control_rx.try_recv() {
            let channel_open = apply_and_publish_training_state(
                command,
                &mut model,
                &mut paused,
                &mut current_lr,
                &mut current_batch_size,
                &mut total_steps,
                write_path.as_deref(),
                options.maintain_latest,
                tx,
            );
            if !channel_open {
                return Ok(());
            }
        }
        if paused {
            match control_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(command) => {
                    let channel_open = apply_and_publish_training_state(
                        command,
                        &mut model,
                        &mut paused,
                        &mut current_lr,
                        &mut current_batch_size,
                        &mut total_steps,
                        write_path.as_deref(),
                        options.maintain_latest,
                        tx,
                    );
                    if !channel_open {
                        return Ok(());
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
            continue;
        }

        let should_report_loss = step % LOSS_REPORT_INTERVAL_STEPS == 0 || step + 1 == total_steps;
        let loss = if should_report_loss {
            trainer
                .task_mut()
                .train_step_report_batch(
                    &mut model,
                    &mut gpu_dataset,
                    step,
                    current_batch_size as usize,
                    (step as u64) << 32,
                )
                .map_err(|err| format!("failed diffusion GPU batch step: {err}"))?
        } else {
            trainer
                .task_mut()
                .train_step_batch(
                    &mut model,
                    &mut gpu_dataset,
                    step,
                    current_batch_size as usize,
                    (step as u64) << 32,
                )
                .map_err(|err| format!("failed diffusion GPU batch step: {err}"))?;
            None
        };

        if should_report_loss {
            if let Some(batch_loss) = loss {
                log_train_loss(
                    &mut metrics,
                    step,
                    batch_loss,
                    current_lr,
                    current_batch_size,
                );
            }
            // Per-timestep-bucket diagnostic: how well ε̂ tracks ε across t.
            let buckets = probe_diffusion(
                &mut model,
                &diffusion,
                &probe_samples,
                input_size.2 as usize,
                output_size.2 as usize,
                &probe_config,
                0x50B0_1234 ^ step as u64,
            );
            log_probe(&mut metrics, step, &buckets);
        }

        // Periodic instrumented sample: capture where along the denoising chain
        // the latent diverges.
        if step > 0 && step % SAMPLE_INTERVAL_STEPS == 0 {
            let mut trajectory = Vec::new();
            let output_len = (output_size.0 * output_size.1 * output_size.2) as usize;
            let image = sample_diffusion(
                &mut model,
                input_size.2 as usize,
                output_size.2 as usize,
                output_len,
                &diffusion,
                step as u64,
                1,
                1.0,
                PosteriorVariance::Beta,
                Some(&mut trajectory),
                None,
                |_, _| {},
            );
            log_trajectory(
                &mut metrics,
                step,
                step as u64,
                &trajectory,
                &Stats::of(&image),
            );
            if let Ok(path) = save_tensor_as_image(&image, output_size, &sample_dir, step) {
                let _ = tx.send(tui::TrainingEvent::Step {
                    step,
                    loss: None,
                    sample_path: Some(path.display().to_string()),
                });
            }
        }

        if tx
            .send(tui::TrainingEvent::Step {
                step,
                loss,
                sample_path: None,
            })
            .is_err()
        {
            return Ok(());
        }

        step += 1;

        // Periodic partial checkpoint. Skipped on the last step, which the
        // final save below covers anyway.
        if let Some(every) = options.checkpoint_every
            && every > 0
            && step % every == 0
            && step < total_steps
        {
            write_partial_checkpoint(
                &model,
                write_path.as_deref(),
                step,
                options.maintain_latest,
            );
        }
    }

    let final_step = step.saturating_sub(1);
    let mut final_trajectory = Vec::new();
    let final_output_len = (output_size.0 * output_size.1 * output_size.2) as usize;
    let output = sample_diffusion(
        &mut model,
        input_size.2 as usize,
        output_size.2 as usize,
        final_output_len,
        &diffusion,
        final_step as u64,
        1,
        1.0,
        PosteriorVariance::Beta,
        Some(&mut final_trajectory),
        None,
        |_, _| {},
    );
    log_trajectory(
        &mut metrics,
        final_step,
        final_step as u64,
        &final_trajectory,
        &Stats::of(&output),
    );
    let sample_path = save_tensor_as_image(&output, output_size, &sample_dir, final_step)
        .map(|path| path.display().to_string())?;
    let _ = tx.send(tui::TrainingEvent::Step {
        step: final_step,
        loss: None,
        sample_path: Some(sample_path),
    });

    if let Some(path) = write_path.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create checkpoint directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        model
            .save_checkpoint(path)
            .map_err(|err| format!("failed to save checkpoint {}: {err}", path.display()))?;
        println!("[checkpoint] final → {}", path.display());
        if options.maintain_latest {
            point_latest_at_reporting(path);
        }
    }

    let _ = tx.send(tui::TrainingEvent::Done);
    Ok(())
}

/// Where `--checkpoint-every` writes: `<out stem>.partial.ckpt`, beside the
/// run's own checkpoint.
///
/// **Rotation, not history.** One file, overwritten every time, because the
/// need this serves is "a run of ten hours that dies at hour nine is not lost",
/// not "keep every intermediate". Seventeen 40 MB files for one night of
/// training would be a different feature, and a worse default.
///
/// It is a plain `.ckpt` and it is *meant* to show up in the weight selector
/// and in `--resume`: the point is to be able to pick a dying run back up.
fn partial_checkpoint_path(checkpoint: &Path) -> PathBuf {
    let stem = checkpoint
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "run".to_string());
    checkpoint.with_file_name(format!("{stem}.partial.ckpt"))
}

/// Write the partial checkpoint, **atomically**: a temporary file beside the
/// target, then a rename.
///
/// A checkpoint is tens of megabytes; a kill in the middle of `fs::write`
/// leaves a truncated file that still ends in `.ckpt`, and the run that
/// resumes from it fails on a length mismatch — at best. `rename` on the same
/// filesystem is atomic, so the partial is always either the previous whole
/// checkpoint or the new whole one.
///
/// Never fatal: a run does not die because a disk filled up at step 5 000. It
/// says so on every failure, which is the loudest thing it can do without
/// throwing the run away.
fn write_partial_checkpoint(
    model: &Model<Training>,
    checkpoint: Option<&Path>,
    step: usize,
    maintain_latest: bool,
) {
    let Some(checkpoint) = checkpoint else {
        eprintln!("[checkpoint] step {step}: no checkpoint path configured, nothing written");
        return;
    };
    let target = partial_checkpoint_path(checkpoint);
    let started = std::time::Instant::now();
    let bytes = match model.checkpoint_bytes() {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("[checkpoint] step {step}: failed to encode checkpoint: {err}");
            return;
        }
    };
    if let Some(parent) = target.parent()
        && let Err(err) = fs::create_dir_all(parent)
    {
        eprintln!("[checkpoint] step {step}: {}: {err}", parent.display());
        return;
    }
    let scratch = target.with_extension("ckpt.tmp");
    if let Err(err) = fs::write(&scratch, &bytes) {
        eprintln!("[checkpoint] step {step}: {}: {err}", scratch.display());
        return;
    }
    if let Err(err) = fs::rename(&scratch, &target) {
        eprintln!("[checkpoint] step {step}: {}: {err}", target.display());
        let _ = fs::remove_file(&scratch);
        return;
    }
    println!(
        "[checkpoint] step {step} → {} ({:.1} MB, {} ms)",
        target.display(),
        bytes.len() as f64 / 1.0e6,
        started.elapsed().as_millis()
    );
    // `latest.ckpt` follows the rotation too: a run killed at hour nine should
    // leave the *partial* as the model's newest weights, which is the whole
    // reason partials exist. It also keeps the link from pinning the previous
    // rotation's bytes on disk — a rename replaces the name, not the inode, so
    // a stale link would keep a full extra checkpoint alive.
    if maintain_latest {
        point_latest_at_reporting(&target);
    }
}

/// Point `latest.ckpt` at what was just written, and say so if it fails.
///
/// Never fatal: the weights are already on disk under their dated name, which
/// is the file that must not be lost. A missing link is a listing that opens on
/// the wrong row, not a lost run.
fn point_latest_at_reporting(written: &Path) {
    if let Err(err) = storage::point_latest_at(written) {
        eprintln!(
            "[checkpoint] could not point latest.ckpt at {}: {err}",
            written.display()
        );
    }
}

fn apply_and_publish_training_state(
    command: tui::TrainingControlCommand,
    model: &mut Model<Training>,
    paused: &mut bool,
    current_lr: &mut f32,
    current_batch_size: &mut u32,
    total_steps: &mut usize,
    checkpoint_path: Option<&Path>,
    maintain_latest: bool,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
) -> bool {
    apply_training_control_command(
        command,
        model,
        paused,
        current_lr,
        current_batch_size,
        total_steps,
        checkpoint_path,
        maintain_latest,
        tx,
    );
    tx.send(tui::TrainingEvent::TrainingState {
        paused: *paused,
        lr: *current_lr,
        batch_size: *current_batch_size,
        total_steps: *total_steps,
    })
    .is_ok()
}

fn apply_training_control_command(
    command: tui::TrainingControlCommand,
    model: &mut Model<Training>,
    paused: &mut bool,
    current_lr: &mut f32,
    current_batch_size: &mut u32,
    total_steps: &mut usize,
    checkpoint_path: Option<&Path>,
    maintain_latest: bool,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
) {
    match command {
        tui::TrainingControlCommand::SetPaused(next) => {
            *paused = next;
        }
        tui::TrainingControlCommand::SaveCheckpoint => {
            let Some(path) = checkpoint_path else {
                let _ = tx.send(tui::TrainingEvent::SaveStatus {
                    message: "cannot save checkpoint: no checkpoint path configured".to_string(),
                    is_error: true,
                });
                return;
            };
            if let Some(parent) = path.parent()
                && let Err(err) = fs::create_dir_all(parent)
            {
                let _ = tx.send(tui::TrainingEvent::SaveStatus {
                    message: format!(
                        "failed to create checkpoint directory {}: {err}",
                        parent.display()
                    ),
                    is_error: true,
                });
                return;
            }
            match model.save_checkpoint(path) {
                Ok(()) => {
                    // The manual save is a save like any other: `latest.ckpt`
                    // has to follow it, or the newest weights on disk stop
                    // being the ones the name promises.
                    if maintain_latest {
                        point_latest_at_reporting(path);
                    }
                    let _ = tx.send(tui::TrainingEvent::SaveStatus {
                        message: format!("checkpoint saved → {}", path.display()),
                        is_error: false,
                    });
                }
                Err(err) => {
                    let _ = tx.send(tui::TrainingEvent::SaveStatus {
                        message: format!("failed to save checkpoint {}: {err}", path.display()),
                        is_error: true,
                    });
                }
            }
        }
        tui::TrainingControlCommand::UpdateParams {
            lr,
            batch_size,
            total_steps: new_total_steps,
        } => {
            *current_lr = lr;
            *current_batch_size = batch_size.max(1);
            *total_steps = new_total_steps.max(1);
            model.set_learning_rate(*current_lr);
            // Rebuilds the graph when the batch actually changes: the batch
            // axis is baked into every activation buffer. Weights, Adam moments
            // and the step counter survive the rebuild (see `resize_batch`).
            if let Err(err) = model.resize_batch(*current_batch_size) {
                eprintln!("[training] could not resize the batch: {err}");
            }
        }
        // Perpetual-only controls. The monitor gates them on the run mode, so
        // reaching one here means a stale command from a previous run's
        // keystroke — ignoring it is the whole handling.
        tui::TrainingControlCommand::NudgeRenoiseDepth(_)
        | tui::TrainingControlCommand::NudgeTempo(_)
        | tui::TrainingControlCommand::Reseed
        | tui::TrainingControlCommand::ToggleRegime
        | tui::TrainingControlCommand::ToggleView
        | tui::TrainingControlCommand::SaveImage => {}
    }
}

async fn run_inference(
    config: ModelConfig,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
) -> Result<(), String> {
    let (gpu, mut model) = build_execution_model(
        &config,
        INFERENCE_RUNTIME_LR,
        INFERENCE_RUNTIME_BATCH_SIZE,
        OptimizerKind::default(),
        WeightInit::default(),
        None,
    )
    .await?;

    let checkpoint_path =
        resolve_sampling_checkpoint_path(&config, config.inference.checkpoint.as_deref())?;
    load_sampling_checkpoint(&mut model, &checkpoint_path, CheckpointWeights::Ema)?;

    let limits = gpu.device().limits();
    let _ = tx.send(tui::TrainingEvent::ResourceReport {
        max_buffer_bytes: limits.max_buffer_size,
        max_storage_binding_bytes: limits.max_storage_buffer_binding_size as u64,
        estimated_training_bytes: model.estimated_gpu_bytes(),
    });

    let input_dims = model
        .input_dim()
        .ok_or_else(|| "model has no input dimensions".to_string())?;
    let output_dims = model
        .output_dim()
        .ok_or_else(|| "model has no output dimensions".to_string())?;
    let input_size = (input_dims.x, input_dims.y, input_dims.z);
    let output_size = (output_dims.x, output_dims.y, output_dims.z);
    let diffusion = LinearNoiseSchedule::new_linear(
        DIFFUSION_SCHEDULE_STEPS,
        DIFFUSION_BETA_START,
        DIFFUSION_BETA_END,
    );

    let inference = &config.inference;
    let seed = if inference.random_seed {
        random_seed()
    } else {
        inference.seed.unwrap_or(0)
    };

    let total_steps = diffusion
        .len()
        .saturating_mul(inference.denoising_paths.max(1));
    let _ = tx.send(tui::TrainingEvent::InferenceProgress {
        label: "Preparing inference path".to_string(),
        current: 0,
        total: total_steps.max(1),
    });

    // The reverse chain keeps its latent on the CPU, so — unlike training, which
    // hands the visualiser the model's own output buffer — inference needs a GPU
    // frame to publish into. It shows x_t next to the model's x0 estimate.
    let mut live = LiveFrame::new(
        model.gpu_context(),
        output_size.0,
        output_size.1,
        output_size.2,
    );
    tui::register_visualiser_source(
        model.gpu_context(),
        live.buffer(),
        live.frame_width(),
        live.frame_height(),
        live.channels(),
        // The window title is the only legend a user sees while watching the
        // frame, so it names the halves rather than merely separating them.
        format!(
            "Denoising  —  gauche: x_t (bruité)  |  droite: x̂₀ (estimation)  —  {}×{}×{}",
            output_size.0, output_size.1, output_size.2
        ),
    );

    let output = sample_diffusion_image_with_controls(
        &mut model,
        input_size,
        output_size,
        &diffusion,
        seed,
        inference.denoising_paths,
        inference.denoise_magnitude,
        inference.posterior_variance,
        Some(&mut |frame: &DenoiseFrame| live.publish(frame.latent, frame.x0_hat)),
        |current, total| {
            let _ = tx.send(tui::TrainingEvent::InferenceProgress {
                label: denoising_progress_label(current, &diffusion, inference.denoising_paths),
                current,
                total,
            });
        },
    );

    // The latent is gone once sampling returns; leaving the window bound to a
    // frozen frame would misrepresent it as still live.
    tui::clear_visualiser_source();
    let pixels = tensor_to_rgb_pixels(&output, output_size)?;

    if tx
        .send(tui::TrainingEvent::InferenceImage {
            width: output_size.0,
            height: output_size.1,
            channels: output_size.2,
            pixels,
            checkpoint_path: checkpoint_path.display().to_string(),
            seed,
        })
        .is_err()
    {
        return Ok(());
    }

    let _ = tx.send(tui::TrainingEvent::Done);
    Ok(())
}

/// An inference that never returns.
///
/// The reverse chain is walked exactly as [`sample_diffusion`] walks it — same
/// [`reverse_step`], same seed derivation — but instead of stopping at `t = 0`
/// and handing back an image, the run re-noises what it just made and descends
/// again. [`PerpetualDrift`] owns the itinerary (which `t`, when to turn round);
/// this function owns the tensors, the pacing and the controls.
///
/// It is the *only* consumer of the run-control channel besides training, and
/// it never sends `Done`: the run ends when the TUI drops the channel.
async fn run_perpetual(
    config: ModelConfig,
    cfg: PerpetualConfig,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
    control_rx: Receiver<tui::TrainingControlCommand>,
) -> Result<(), String> {
    let (gpu, mut model) = build_execution_model(
        &config,
        INFERENCE_RUNTIME_LR,
        INFERENCE_RUNTIME_BATCH_SIZE,
        OptimizerKind::default(),
        WeightInit::default(),
        None,
    )
    .await?;

    let checkpoint_path = resolve_sampling_checkpoint_path(&config, cfg.checkpoint.as_deref())?;
    load_sampling_checkpoint(&mut model, &checkpoint_path, CheckpointWeights::Ema)?;

    let limits = gpu.device().limits();
    let _ = tx.send(tui::TrainingEvent::ResourceReport {
        max_buffer_bytes: limits.max_buffer_size,
        max_storage_binding_bytes: limits.max_storage_buffer_binding_size as u64,
        estimated_training_bytes: model.estimated_gpu_bytes(),
    });

    let input_dims = model
        .input_dim()
        .ok_or_else(|| "model has no input dimensions".to_string())?;
    let output_dims = model
        .output_dim()
        .ok_or_else(|| "model has no output dimensions".to_string())?;
    let output_size = (output_dims.x, output_dims.y, output_dims.z);
    let output_len = (output_size.0 * output_size.1 * output_size.2) as usize;
    let input_channels = input_dims.z as usize;
    let signal_channels = output_dims.z as usize;

    let schedule = LinearNoiseSchedule::new_linear(
        DIFFUSION_SCHEDULE_STEPS,
        DIFFUSION_BETA_START,
        DIFFUSION_BETA_END,
    );

    let seed = if cfg.random_seed {
        random_seed()
    } else {
        cfg.seed.unwrap_or(0)
    };
    // The picture the drift sets out from. A dataset that cannot be found is
    // not fatal: the run falls back on the pure-noise opening perpetual runs
    // always had, and says which one it is doing on the status line.
    // No flag on this path — the TUI has no command line — so the model's own
    // `seed_dataset` is what decides, and the convention behind it.
    let seed_images = SeedImages::resolve(None, &config, output_size)?;
    let mut drift = match seed_images.as_ref() {
        Some(_) => PerpetualDrift::from_image(schedule.len(), cfg.regime, cfg.renoise_depth, seed),
        None => PerpetualDrift::new(schedule.len(), cfg.regime, cfg.renoise_depth, seed),
    };
    // The latent, and the climb departure it has to remember, both live in the
    // walk — engine side, so the frame-by-frame behaviour is unit-testable
    // against an oracle instead of only observable through a window.
    let opening = match seed_images.as_ref() {
        Some(images) => images.provide_x0(seed),
        None => schedule.sample_noise(output_len, drift.initial_noise_seed()),
    };
    // The estimate to show before the first frame reports one. Seeded from an
    // image, that IS the image — which is the honest answer and also what the
    // viewer expects to see for the frame before the drift moves. Seeded from
    // noise, a flat mid-grey: never the pure-noise latent, which `add_noise`
    // would take for a clean image (see the perpetual module's header).
    let mut last_x0 = match seed_images.as_ref() {
        Some(_) => opening.clone(),
        None => vec![0.0f32; output_len],
    };
    let mut walk = DriftWalk::new(opening, cfg.denoise_magnitude);
    if let Some(images) = seed_images.as_ref() {
        let _ = tx.send(tui::TrainingEvent::SaveStatus {
            message: format!(
                "dérive img2img depuis {} ({} images)",
                images.path().display(),
                images.len()
            ),
            is_error: false,
        });
    }

    // A drift is something to look at, so the default is the estimate ALONE,
    // square, at the image's own aspect ratio — "on n'a même pas besoin de la
    // fenêtre de gauche". `[x]` brings the latent back when the question is
    // what the noise is doing.
    let mut view = batlab_core::LiveView::X0Only;
    let mut live = LiveFrame::with_view(
        model.gpu_context(),
        output_size.0,
        output_size.1,
        output_size.2,
        view,
    );
    let show = |live: &LiveFrame, gpu: std::sync::Arc<batlab_core::GpuContext>, view: batlab_core::LiveView| {
        tui::register_visualiser_source(
            gpu,
            live.buffer(),
            live.frame_width(),
            live.frame_height(),
            live.channels(),
            format!(
                "Perpetual  —  {}  —  {}×{}×{}",
                view.caption(),
                output_size.0,
                output_size.1,
                output_size.2
            ),
        );
    };
    show(&live, model.gpu_context(), view);

    let sample_dir = storage::project_root().join("perpetual_samples");
    let mut tempo = cfg
        .tempo
        .clamp(PerpetualConfig::MIN_TEMPO, PerpetualConfig::MAX_TEMPO);
    let mut paused = false;
    let mut steps = 0usize;
    let mut saved = 0usize;
    let mut next_step_at = std::time::Instant::now();
    let mut pace = PaceMeter::new();
    let mut last_published = std::time::Instant::now();

    // Names the file AND the source that chose it. The panel is the only place
    // a TUI run says where its pictures come from, and "image du dataset" alone
    // answered neither which one nor why.
    let origin_label = match seed_images.as_ref() {
        Some(images) => format!(
            "{} [{}]",
            images
                .path()
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| images.path().display().to_string()),
            images.source().label()
        ),
        None => "bruit pur (aucun dataset)".to_string(),
    };
    let publish = |tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
                   drift: &PerpetualDrift,
                   phase: batlab_core::DriftPhase,
                   steps: usize,
                   steps_per_sec: f32,
                   tempo: f32,
                   paused: bool,
                   view: batlab_core::LiveView|
     -> bool {
        tx.send(tui::TrainingEvent::PerpetualState(tui::PerpetualStatus {
            regime: drift.regime().label().to_string(),
            // An image coming apart on screen is the nominal behaviour half the
            // time; unlabelled, it reads as a fault. It names the phase of the
            // action just taken, not the one the drift intends next: `step`
            // reconciles the phase on its way in and can overrule that intent,
            // which used to show as one stray `descente` at the top of every
            // upward approach in flux.
            phase: phase.label().to_string(),
            depth: drift.depth(),
            depth_label: drift.regime().depth_label().to_string(),
            min_depth: drift.min_depth(),
            max_depth: drift.max_depth(),
            cycle: drift.counter().1,
            cycle_label: drift.counter().0.to_string(),
            diffusion_step: drift.current_step(),
            steps,
            steps_per_sec,
            tempo,
            paused,
            view: match view {
                batlab_core::LiveView::X0Only => "x̂₀ seul".to_string(),
                batlab_core::LiveView::Both => "x_t | x̂₀".to_string(),
            },
            origin: origin_label.clone(),
        }))
        .is_ok()
    };
    // Nothing has been stepped yet, so the opening read is the drift's own.
    let mut phase = drift.phase();
    if !publish(tx, &drift, phase, steps, 0.0, tempo, paused, view) {
        tui::clear_visualiser_source();
        return Ok(());
    }

    loop {
        let mut dirty = false;
        loop {
            match control_rx.try_recv() {
                Ok(command) => {
                    dirty = true;
                    match command {
                        tui::TrainingControlCommand::SetPaused(next) => {
                            paused = next;
                            next_step_at = std::time::Instant::now();
                            pace.reset();
                        }
                        tui::TrainingControlCommand::NudgeRenoiseDepth(delta) => {
                            drift.nudge_depth(delta)
                        }
                        tui::TrainingControlCommand::ToggleRegime => drift.toggle_regime(),
                        tui::TrainingControlCommand::Reseed => {
                            let next_seed = random_seed();
                            drift.reseed(next_seed);
                            // `restart_from` drops the climb departure with the
                            // latent: a climb reading a snapshot from before the
                            // re-seed would carry the old image up the schedule.
                            // Whatever the run opened on, `[r]` opens on again
                            // — the drift keeps its origin across a re-seed, so
                            // handing it the other kind of latent would put a
                            // photograph at the top of the schedule.
                            let opening = match seed_images.as_ref() {
                                Some(images) => images.provide_x0(next_seed),
                                None => schedule
                                    .sample_noise(output_len, drift.initial_noise_seed()),
                            };
                            last_x0 = match seed_images.as_ref() {
                                Some(_) => opening.clone(),
                                None => vec![0.0f32; output_len],
                            };
                            walk.restart_from(opening);
                        }
                        tui::TrainingControlCommand::NudgeTempo(delta) => {
                            let factor = PerpetualConfig::TEMPO_FACTOR.powi(delta);
                            tempo = (tempo * factor)
                                .clamp(PerpetualConfig::MIN_TEMPO, PerpetualConfig::MAX_TEMPO);
                            next_step_at = std::time::Instant::now();
                            pace.reset();
                        }
                        tui::TrainingControlCommand::SaveImage => {
                            let name = format!(
                                "{}_c{:03}_{:03}.png",
                                drift.regime().label(),
                                drift.cycle(),
                                saved
                            );
                            let message = match write_tensor_png(
                                &last_x0,
                                output_size,
                                &sample_dir.join(name),
                            ) {
                                Ok(path) => {
                                    saved += 1;
                                    format!("image → {}", path.display())
                                }
                                Err(err) => err,
                            };
                            let is_error = !message.starts_with("image");
                            if tx
                                .send(tui::TrainingEvent::SaveStatus { message, is_error })
                                .is_err()
                            {
                                tui::clear_visualiser_source();
                                return Ok(());
                            }
                        }
                        tui::TrainingControlCommand::ToggleView => {
                            // The buffer is sized for the layout, so a new view
                            // means a new frame and a re-registration: the
                            // window has to be told the new width anyway, and
                            // it comes back at the new aspect ratio.
                            view = view.toggle();
                            live = LiveFrame::with_view(
                                model.gpu_context(),
                                output_size.0,
                                output_size.1,
                                output_size.2,
                                view,
                            );
                            // Painted at once from what is already in hand, so
                            // the new window opens on the picture rather than
                            // on a frame of mid-grey.
                            live.publish(walk.latent(), &last_x0);
                            show(&live, model.gpu_context(), view);
                        }
                        // Training-only commands; a perpetual run has no
                        // optimiser to retune and no weights of its own to save.
                        tui::TrainingControlCommand::SaveCheckpoint
                        | tui::TrainingControlCommand::UpdateParams { .. } => {}
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                // The monitor is gone: the only way this run ever ends.
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    tui::clear_visualiser_source();
                    return Ok(());
                }
            }
        }

        if paused {
            if dirty && !publish(tx, &drift, phase, steps, 0.0, tempo, paused, view) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }

        let action = drift.step();
        phase = action.phase();
        // A climb increment is closed-form arithmetic and used to be free; it
        // now costs the one model call that keeps x̂₀ alive while the image
        // dissolves (`DriftWalk`, and `IMG2IMG_DRIFT.md`). The tempo ratio
        // below therefore paces two comparable frames, not a cheap one and an
        // expensive one.
        let climbing = matches!(action, DriftAction::Climb { .. });
        let opens_cycle = matches!(action, DriftAction::Climb { opens_cycle: true, .. });
        let frame = walk.advance(
            action,
            &schedule,
            &mut ModelNoise {
                model: &mut model,
                schedule: &schedule,
                input_channels,
                signal_channels,
            },
        );
        live.publish(&frame.latent, &frame.x0_hat);
        last_x0 = frame.x0_hat;
        steps += frame.model_calls;
        pace.tick();
        // The turn of the cycle is worth a redraw of the panel; the rest of the
        // climb rides the usual 100 ms refresh.
        dirty |= opens_cycle;

        let now = std::time::Instant::now();
        // A climb increment costs no model call, so its pace is free to differ
        // from the descent's; `CLIMB_TEMPO_RATIO` keeps them equal by default.
        let step_tempo = if climbing {
            tempo * batlab_core::CLIMB_TEMPO_RATIO
        } else {
            tempo
        };
        next_step_at += Duration::from_secs_f32(1.0 / step_tempo.max(f32::EPSILON));
        if next_step_at > now {
            std::thread::sleep(next_step_at - now);
        } else {
            // Fell behind the requested pace (the sampler is the ceiling): drop
            // the debt instead of sprinting to repay it.
            next_step_at = now;
        }

        if dirty || last_published.elapsed() >= Duration::from_millis(100) {
            last_published = std::time::Instant::now();
            if !publish(tx, &drift, phase, steps, pace.per_second(), tempo, paused, view) {
                break;
            }
        }
    }

    tui::clear_visualiser_source();
    Ok(())
}

/// Rolling measurement of how fast the chain is actually being walked, which is
/// not the requested tempo whenever the sampler is the bottleneck.
struct PaceMeter {
    window_start: std::time::Instant,
    ticks: usize,
    last_rate: f32,
}

impl PaceMeter {
    fn new() -> Self {
        Self {
            window_start: std::time::Instant::now(),
            ticks: 0,
            last_rate: 0.0,
        }
    }

    fn reset(&mut self) {
        self.window_start = std::time::Instant::now();
        self.ticks = 0;
    }

    fn tick(&mut self) {
        self.ticks += 1;
        let elapsed = self.window_start.elapsed();
        if elapsed >= Duration::from_millis(500) {
            self.last_rate = self.ticks as f32 / elapsed.as_secs_f32();
            self.reset();
        }
    }

    fn per_second(&self) -> f32 {
        self.last_rate
    }
}

/// Renders "which path, which step, which t" from the sampler's flat work
/// counter.
///
/// The sampler reports a single `current`/`total` across all paths, but the two
/// numbers a user actually watches are the path and the timestep — and t counts
/// *down* while the step counter counts up, so showing the raw counter as "t"
/// would state the schedule backwards.
fn denoising_progress_label(
    current: usize,
    schedule: &LinearNoiseSchedule,
    denoising_paths: usize,
) -> String {
    let steps = schedule.len().max(1);
    let path_count = denoising_paths.max(1);
    // `current` is 1-based (it counts completed steps); step 0 has none done.
    let done = current.saturating_sub(1);
    let path_idx = (done / steps).min(path_count - 1);
    let step_index = done % steps;
    let t = steps - 1 - step_index;

    if path_count > 1 {
        format!(
            "Denoising · path {}/{} · step {}/{} (t={t})",
            path_idx + 1,
            path_count,
            step_index + 1,
            steps
        )
    } else {
        format!("Denoising · step {}/{} (t={t})", step_index + 1, steps)
    }
}

/// Which weights a sampling run loads.
///
/// The weight selector lets the user pick any file in the model's
/// `pretrained_weights/`; honouring that choice here is what makes the screen
/// mean something for inference. `latest.ckpt` stays the fallback, so configs
/// written before the choice was recorded behave exactly as they did.
fn resolve_sampling_checkpoint_path(
    config: &ModelConfig,
    selected: Option<&str>,
) -> Result<PathBuf, String> {
    if let Some(path) = selected.map(PathBuf::from)
        && path.exists()
    {
        return Ok(path);
    }
    let model_name = config
        .model_name
        .as_deref()
        .ok_or_else(|| "inference requires a named model configuration".to_string())?;
    let checkpoint = storage::default_model_checkpoint_path(model_name).map_err(|err| {
        format!("failed to resolve inference checkpoint path for '{model_name}': {err}")
    })?;
    if !checkpoint.exists() {
        return Err(format!(
            "inference checkpoint not found: {} (train the model first or place weights there)",
            checkpoint.display()
        ));
    }
    Ok(checkpoint)
}

/// A loaded dataset, still in the encoding it will cross to the GPU in.
///
/// The point of the type is that it does NOT normalise: a BATRAW3 file arrives
/// as bytes and leaves as bytes, and only the two callers that genuinely need
/// host-side pixels (the metrics probe, the perpetual seed image) pay to widen
/// one sample. Flattening on load would have thrown away three quarters of the
/// format's benefit before it reached the buffer that matters.
struct Dataset {
    payload: DatasetPayload,
    sample_len: usize,
}

impl Dataset {
    fn from_samples(samples: Vec<ImageSample>, sample_len: usize) -> Self {
        Self {
            payload: DatasetPayload::Floats(
                samples.into_iter().map(|sample| sample.target).collect(),
            ),
            sample_len,
        }
    }

    fn len(&self) -> usize {
        self.payload.sample_count(self.sample_len)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One sample as f32. Panics on an out-of-range index, which callers avoid
    /// by taking it modulo `len()`.
    fn sample(&self, index: usize) -> Vec<f32> {
        self.payload
            .sample_f32(index, self.sample_len)
            .unwrap_or_else(|| panic!("sample {index} out of a dataset of {}", self.len()))
    }
}

fn load_dataset(dataset_path: &str, output_size: (u32, u32, u32)) -> Result<Dataset, String> {
    let canonical_path = Path::new(dataset_path)
        .canonicalize()
        .map_err(|err| format!("failed to resolve dataset path '{}': {err}", dataset_path))?;
    let sample_len = (output_size.0 * output_size.1 * output_size.2) as usize;

    if let Some(dataset) = try_load_raw_dataset(&canonical_path, output_size)? {
        return Ok(dataset);
    }

    if let Some(dataset) = try_load_cifar_dataset(&canonical_path, output_size)? {
        return Ok(Dataset::from_samples(dataset, sample_len));
    }

    let mut image_paths = Vec::new();
    let mut visited_dirs = HashSet::new();
    collect_image_paths(&canonical_path, &mut image_paths, &mut visited_dirs)?;
    image_paths.sort();

    if image_paths.is_empty() {
        return Err(format!("no images found at '{}'", dataset_path));
    }

    let samples = image_paths
        .into_iter()
        .map(|path| {
            let image = image::open(&path)
                .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
            Ok(ImageSample {
                target: image_to_tensor(&image, output_size),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Dataset::from_samples(samples, sample_len))
}

/// Tries to load a dataset from a raw binary file (`*.batraw`) or a directory that contains such
/// files.  Returns `Ok(None)` when the path does not look like a raw-binary dataset so the caller
/// can fall back to other loaders.
///
/// # Binary format (produced by the Python pre-processing scripts)
/// ```text
/// [0..8]   magic: b"BATRAW3\0" (or legacy b"BATRAW2\0" / b"BATRAW1\0")
/// [8..12]  count:    u32 LE – number of samples
/// [12..16] width:    u32 LE – image width in pixels
/// [16..20] height:   u32 LE – image height in pixels
/// [20..24] channels: u32 LE – number of channels per pixel
/// [24..]   data:     count * width * height * channels values, in the payload
///                    encoding the magic names
/// ```
///
/// Three magics, one header, three payloads:
///
/// - `BATRAW3` — **u8**, `[0, 255]`, widened to `[-1, 1]` on the GPU. The
///   default the converters write, because every image this project has trained
///   on was 8-bit at the source and storing it as f32 quadrupled the file, the
///   host RAM and the chunk uploads for nothing. The widening is exact, so a
///   BATRAW3 file and the BATRAW2 file converted from the same images decode to
///   the same bits.
/// - `BATRAW2` — f32 already in `[-1, 1]`, the convention the diffusion pipeline
///   expects.
/// - `BATRAW1` — f32 in `[0, 1]`, rescaled on the fly.
///
/// A BATRAW3 file whose geometry already matches the model stays 8-bit all the
/// way to the GPU. Anything else — a mismatched geometry needing a resample —
/// falls back to the f32 path, because the resample happens on the host anyway.
fn try_load_raw_dataset(
    dataset_path: &Path,
    output_size: (u32, u32, u32),
) -> Result<Option<Dataset>, String> {
    // Collect candidate .batraw files.
    let mut raw_files: Vec<PathBuf> = Vec::new();

    if dataset_path.is_file() {
        if dataset_path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("batraw"))
        {
            raw_files.push(dataset_path.to_path_buf());
        } else {
            return Ok(None);
        }
    } else if dataset_path.is_dir() {
        for entry in fs::read_dir(dataset_path)
            .map_err(|err| format!("failed to read directory {}: {err}", dataset_path.display()))?
        {
            let path = entry
                .map_err(|err| {
                    format!(
                        "failed to read directory entry in {}: {err}",
                        dataset_path.display()
                    )
                })?
                .path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("batraw"))
            {
                raw_files.push(path);
            }
        }
        if raw_files.is_empty() {
            return Ok(None);
        }
        raw_files.sort();
    } else {
        return Ok(None);
    }

    let model_sample_len = (output_size.0 * output_size.1 * output_size.2) as usize;
    // Both accumulate; at most one ends up non-empty. A directory holding an
    // 8-bit file next to an f32 one is a mixture the GPU cannot stream as one
    // payload, so it is refused rather than silently widened.
    let mut floats: Vec<ImageSample> = Vec::new();
    let mut eight_bit: Vec<u8> = Vec::new();

    for raw_file in &raw_files {
        let bytes = fs::read(raw_file)
            .map_err(|err| format!("failed to read {}: {err}", raw_file.display()))?;

        if bytes.len() < RAW_DATASET_MAGIC_SIGNED.len() + 16 {
            return Err(format!(
                "raw dataset file too short to contain a valid header: {}",
                raw_file.display()
            ));
        }
        let magic = &bytes[..RAW_DATASET_MAGIC_SIGNED.len()];
        // BATRAW1 payloads predate the [-1, 1] convention and are rescaled below.
        let (value_bytes, needs_unit_rescale) = if magic == RAW_DATASET_MAGIC_BYTES {
            (1usize, false)
        } else if magic == RAW_DATASET_MAGIC_SIGNED {
            (4, false)
        } else if magic == RAW_DATASET_MAGIC_UNIT {
            (4, true)
        } else {
            return Err(format!(
                "invalid magic in raw dataset file: {}",
                raw_file.display()
            ));
        };

        let mut offset = RAW_DATASET_MAGIC_SIGNED.len();
        // count is a u32 value from a validated header produced by our own tooling, so it
        // safely fits in usize on all supported 32- and 64-bit targets.
        let count = read_u32_le_bytes(&bytes, &mut offset)? as usize;
        let width = read_u32_le_bytes(&bytes, &mut offset)?;
        let height = read_u32_le_bytes(&bytes, &mut offset)?;
        let channels = read_u32_le_bytes(&bytes, &mut offset)?;

        let sample_values = (width * height * channels) as usize;
        let expected_bytes = offset + count * sample_values * value_bytes;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "raw dataset file size mismatch in {}: expected {expected_bytes} bytes, got {}",
                raw_file.display(),
                bytes.len()
            ));
        }

        let geometry_matches = (width, height, channels) == output_size;
        // The geometry mismatch below is silently repaired by a u8 round-trip
        // (`raw_floats_to_dynamic_image` + `image_to_tensor`). That is convenient for
        // rescaling, but it also means feeding a 1-channel dataset to a 3-channel model
        // "works": every sample is grey replicated over R, G and B, and a whole overnight
        // run trains on colourless data without a single error. Say it out loud.
        if !geometry_matches {
            eprintln!(
                "[dataset] WARNING {}: file is {width}x{height}x{channels}, model expects \
                 {}x{}x{} — samples are converted through an 8-bit image round-trip\
                 {}",
                raw_file.display(),
                output_size.0,
                output_size.1,
                output_size.2,
                if channels != output_size.2 {
                    " (CHANNEL COUNT DIFFERS: check you passed the right .batraw)"
                } else {
                    ""
                }
            );
        }

        // The fast lane, and the only one that keeps the format's benefit: an
        // 8-bit file the model can eat as it lies. `bytes` is moved wholesale
        // rather than copied sample by sample — on ImageNet that is 3.9 GB not
        // walked.
        if value_bytes == 1 && geometry_matches {
            if !floats.is_empty() {
                return Err(format!(
                    "{}: a directory cannot mix 8-bit and f32 .batraw files",
                    dataset_path.display()
                ));
            }
            let mut payload = bytes;
            payload.drain(..offset);
            eight_bit.extend_from_slice(&payload);
            continue;
        }
        if !eight_bit.is_empty() {
            return Err(format!(
                "{}: a directory cannot mix 8-bit and f32 .batraw files",
                dataset_path.display()
            ));
        }

        for _ in 0..count {
            let raw: Vec<f32> = if value_bytes == 1 {
                bytes[offset..offset + sample_values]
                    .iter()
                    .copied()
                    .map(batlab_core::decode_u8)
                    .collect()
            } else {
                bytes[offset..offset + sample_values * 4]
                    .chunks_exact(4)
                    .map(|b| {
                        let value = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        if needs_unit_rescale {
                            value * 2.0 - 1.0
                        } else {
                            value
                        }
                    })
                    .collect()
            };
            offset += sample_values * value_bytes;

            // Rescale to the model's output dimensions if they differ.
            let target = if geometry_matches {
                raw
            } else {
                let image = raw_floats_to_dynamic_image(&raw, width, height, channels)?;
                image_to_tensor(&image, output_size)
            };
            floats.push(ImageSample { target });
        }
    }

    if !eight_bit.is_empty() {
        return Ok(Some(Dataset {
            payload: DatasetPayload::Bytes(eight_bit),
            sample_len: model_sample_len,
        }));
    }
    Ok(Some(Dataset::from_samples(floats, model_sample_len)))
}

/// Decode a flat `[-1, 1]` f32 slice back into a [`DynamicImage`] for rescaling.
fn raw_floats_to_dynamic_image(
    data: &[f32],
    width: u32,
    height: u32,
    channels: u32,
) -> Result<DynamicImage, String> {
    let pixels: Vec<u8> = data.iter().map(|v| to_u8(*v)).collect();
    match channels {
        1 => {
            let img = GrayImage::from_raw(width, height, pixels)
                .ok_or_else(|| "failed to reconstruct greyscale image from raw data".to_string())?;
            Ok(DynamicImage::ImageLuma8(img))
        }
        3 => {
            let img = RgbImage::from_raw(width, height, pixels)
                .ok_or_else(|| "failed to reconstruct RGB image from raw data".to_string())?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        c => Err(format!(
            "unsupported channel count {c} in raw dataset (expected 1 or 3)"
        )),
    }
}

/// Read a little-endian `u32` from `bytes` at `*offset`, advancing the offset by 4.
fn read_u32_le_bytes(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = *offset + 4;
    if end > bytes.len() {
        return Err(format!(
            "unexpected end of data reading u32 at offset {offset}"
        ));
    }
    let value = u32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    *offset = end;
    Ok(value)
}

fn try_load_cifar_dataset(
    dataset_path: &Path,
    output_size: (u32, u32, u32),
) -> Result<Option<Vec<ImageSample>>, String> {
    let cifar_dir = if dataset_path.join("cifar-10-batches-bin").is_dir() {
        Some(dataset_path.join("cifar-10-batches-bin"))
    } else if dataset_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "cifar-10-batches-bin")
    {
        Some(dataset_path.to_path_buf())
    } else {
        None
    };

    let Some(cifar_dir) = cifar_dir else {
        return Ok(None);
    };

    let mut batch_files = Vec::new();
    for entry in fs::read_dir(&cifar_dir).map_err(|err| {
        format!(
            "failed to read CIFAR directory {}: {err}",
            cifar_dir.display()
        )
    })? {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read directory entry in {}: {err}",
                cifar_dir.display()
            )
        })?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("data_batch_") && name.ends_with(".bin"))
        {
            batch_files.push(path);
        }
    }
    batch_files.sort();

    if batch_files.is_empty() {
        let test_batch = cifar_dir.join("test_batch.bin");
        if test_batch.is_file() {
            batch_files.push(test_batch);
        }
    }

    if batch_files.is_empty() {
        return Err(format!(
            "no CIFAR batch files found in {}",
            cifar_dir.display()
        ));
    }

    let mut dataset = Vec::new();
    for batch_file in batch_files {
        let bytes = fs::read(&batch_file)
            .map_err(|err| format!("failed to read {}: {err}", batch_file.display()))?;
        if bytes.len() % 3073 != 0 {
            return Err(format!(
                "unexpected CIFAR batch size in {}: {} bytes",
                batch_file.display(),
                bytes.len()
            ));
        }

        for record in bytes.chunks_exact(3073) {
            let image = cifar_record_to_image(record)?;
            dataset.push(ImageSample {
                target: image_to_tensor(&image, output_size),
            });
        }
    }

    Ok(Some(dataset))
}

fn cifar_record_to_image(record: &[u8]) -> Result<DynamicImage, String> {
    if record.len() != 3073 {
        return Err(format!(
            "invalid CIFAR record length: expected 3073, got {}",
            record.len()
        ));
    }

    let channels = &record[1..];
    let mut pixels = Vec::with_capacity(32 * 32 * 3);
    for index in 0..1024 {
        pixels.push(channels[index]);
        pixels.push(channels[1024 + index]);
        pixels.push(channels[2048 + index]);
    }
    let image = RgbImage::from_raw(32, 32, pixels)
        .ok_or_else(|| "failed to build CIFAR RGB image".to_string())?;
    Ok(DynamicImage::ImageRgb8(image))
}

fn collect_image_paths(
    path: &Path,
    files: &mut Vec<PathBuf>,
    visited_dirs: &mut HashSet<PathBuf>,
) -> Result<(), String> {
    if path.is_file() {
        if is_supported_image(path) {
            files.push(path.to_path_buf());
            return Ok(());
        }
        return Err(format!(
            "'{}' is not a supported image file",
            path.display()
        ));
    }

    if !path.is_dir() {
        return Err(format!("'{}' is not a file or directory", path.display()));
    }

    let canonical_dir = path
        .canonicalize()
        .map_err(|err| format!("failed to resolve {}: {err}", path.display()))?;
    if !visited_dirs.insert(canonical_dir) {
        return Ok(());
    }

    for entry in fs::read_dir(path)
        .map_err(|err| format!("failed to read directory {}: {err}", path.display()))?
    {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read directory entry in {}: {err}",
                path.display()
            )
        })?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_image_paths(&entry_path, files, visited_dirs)?;
        } else if is_supported_image(&entry_path) {
            files.push(entry_path);
        }
    }

    Ok(())
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp"
            )
        })
        .unwrap_or(false)
}

fn image_to_tensor(image: &DynamicImage, dims: (u32, u32, u32)) -> Vec<f32> {
    let (width, height, channels) = dims;
    if width == 0 || height == 0 || channels == 0 {
        return Vec::new();
    }

    if channels == 1 {
        return image
            .resize_exact(width, height, FilterType::Triangle)
            .to_luma8()
            .pixels()
            .map(|pixel| from_u8(pixel.0[0]))
            .collect();
    }

    let resized = image
        .resize_exact(width, height, FilterType::Triangle)
        .to_rgb8();
    let mut tensor = Vec::with_capacity((width * height * channels) as usize);
    for pixel in resized.pixels() {
        let rgb = pixel.0;
        for channel in 0..channels as usize {
            let value = match channel {
                0 => rgb[0],
                1 => rgb[1],
                2 => rgb[2],
                _ => rgb[2],
            };
            tensor.push(from_u8(value));
        }
    }
    tensor
}

/// Thin wrapper over the library's single diffusion sampler. Kept so the call
/// sites can pass `(u32, u32, u32)` dims; the actual denoising math (and the
/// `[signal | timestep]` input composition) lives in `batlab_core::metrics` so
/// training instrumentation and inference cannot drift apart.
#[allow(clippy::too_many_arguments)]
fn sample_diffusion_image_with_controls<State, F>(
    model: &mut Model<State>,
    input_dims: (u32, u32, u32),
    output_dims: (u32, u32, u32),
    schedule: &LinearNoiseSchedule,
    seed: u64,
    denoising_paths: usize,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    observer: Option<&mut dyn FnMut(&DenoiseFrame)>,
    progress: F,
) -> Vec<f32>
where
    F: FnMut(usize, usize),
{
    let output_len = (output_dims.0 * output_dims.1 * output_dims.2) as usize;
    sample_diffusion(
        model,
        input_dims.2 as usize,
        output_dims.2 as usize,
        output_len,
        schedule,
        seed,
        denoising_paths,
        denoise_magnitude,
        variance,
        None,
        observer,
        progress,
    )
}

fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x5eed_u64);
    nanos ^ (nanos.rotate_left(17)).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn prepare_sample_dir(dataset_path: &str) -> Result<PathBuf, String> {
    let dataset_path = Path::new(dataset_path);
    let sample_dir = if dataset_path.is_dir() {
        dataset_path.join("generated_samples")
    } else {
        dataset_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("generated_samples")
    };

    fs::create_dir_all(&sample_dir).map_err(|err| {
        format!(
            "failed to create sample directory {}: {err}",
            sample_dir.display()
        )
    })?;
    Ok(sample_dir)
}

fn save_tensor_as_image(
    tensor: &[f32],
    dims: (u32, u32, u32),
    sample_dir: &Path,
    step: usize,
) -> Result<PathBuf, String> {
    write_tensor_png(
        tensor,
        dims,
        &sample_dir.join(format!("step_{step:04}.png")),
    )
}

/// Encodes a `[-1, 1]` tensor to a PNG at `path`, creating its directory.
///
/// Same encoder as the training previews and the headless sampler — a saved
/// frame of a perpetual run and a saved sample of a finite one are the same
/// bytes for the same tensor.
fn write_tensor_png(tensor: &[f32], dims: (u32, u32, u32), path: &Path) -> Result<PathBuf, String> {
    let (width, height, channels) = dims;
    let expected_len = (width * height * channels) as usize;
    if tensor.len() != expected_len {
        return Err(format!(
            "output tensor length mismatch: expected {expected_len}, got {}",
            tensor.len()
        ));
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }

    let path = path.to_path_buf();
    if channels == 1 {
        let pixels: Vec<u8> = tensor.iter().map(|value| to_u8(*value)).collect();
        let image = GrayImage::from_raw(width, height, pixels)
            .ok_or_else(|| format!("failed to build grayscale image for {}", path.display()))?;
        image
            .save(&path)
            .map_err(|err| format!("failed to save {}: {err}", path.display()))?;
        return Ok(path);
    }

    let pixels = tensor_to_rgb_pixels(tensor, dims)?;
    let image = RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| format!("failed to build RGB image for {}", path.display()))?;
    image
        .save(&path)
        .map_err(|err| format!("failed to save {}: {err}", path.display()))?;
    Ok(path)
}

/// Encode an 8-bit sample into the model's `[-1, 1]` convention.
///
/// Diffusion's forward process `x_t = sqrt(a_bar)*x_0 + sqrt(1-a_bar)*eps`
/// assumes a zero-centered `x_0`; a `[0, 1]` encoding leaves a mean bias of
/// `0.5*sqrt(a_bar)` at every timestep while the sampler starts from N(0, 1).
///
/// Delegates rather than restates: the same widening is applied by the dataset
/// decode shader, and a second copy of the expression here would be a second
/// thing to keep in step.
fn from_u8(value: u8) -> f32 {
    batlab_core::decode_u8(value)
}

/// Inverse of [`from_u8`].
fn to_u8(value: f32) -> u8 {
    ((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8
}

fn tensor_to_rgb_pixels(tensor: &[f32], dims: (u32, u32, u32)) -> Result<Vec<u8>, String> {
    let (width, height, channels) = dims;
    let expected_len = (width * height * channels) as usize;
    if tensor.len() != expected_len {
        return Err(format!(
            "output tensor length mismatch: expected {expected_len}, got {}",
            tensor.len()
        ));
    }
    if channels == 0 {
        return Err("output channels must be > 0".to_string());
    }

    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for pixel in tensor.chunks(channels as usize) {
        let fallback = *pixel.first().unwrap_or(&0.0);
        pixels.push(to_u8(fallback));
        pixels.push(to_u8(*pixel.get(1).unwrap_or(&fallback)));
        pixels.push(to_u8(*pixel.get(2).unwrap_or(&fallback)));
    }
    Ok(pixels)
}

/// Append the layer a `config_file` entry describes.
///
/// One line, because the translation itself now lives in the engine
/// ([`Model::add_draft`]). It used to live here, spelled out layer by layer —
/// and the resource inventory, which has to build the same stack from the same
/// file without a GPU, could not reach it. Two copies of "what this config
/// means" is exactly the kind of duplication that ends with a page confidently
/// reporting a graph the trainer does not build.
fn append_layer<State>(
    model: &mut Model<State>,
    draft: &LayerDraft,
) -> Result<(), batlab_core::ModelError> {
    model.add_draft(draft)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// The progress label must run t *downwards* while the step counter runs up
    /// — the reverse chain starts at the noisiest timestep. Reading these two the
    /// same way round is the easy mistake, so pin both ends of the schedule.
    #[test]
    fn denoising_label_counts_t_down_while_steps_count_up() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);

        let first = denoising_progress_label(1, &schedule, 1);
        assert!(
            first.contains("step 1/256") && first.contains("t=255"),
            "first step should be step 1 at t=255, got {first:?}"
        );

        let last = denoising_progress_label(256, &schedule, 1);
        assert!(
            last.contains("step 256/256") && last.contains("t=0"),
            "last step should be step 256 at t=0, got {last:?}"
        );
    }

    /// With several paths the counter is flat across all of them; the label has
    /// to split it back into path and step, and must not run past the last path
    /// on the final tick.
    #[test]
    fn denoising_label_splits_flat_counter_into_paths() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);

        let start_of_second = denoising_progress_label(257, &schedule, 4);
        assert!(
            start_of_second.contains("path 2/4") && start_of_second.contains("step 1/256"),
            "unit 257 should open path 2, got {start_of_second:?}"
        );

        let final_unit = denoising_progress_label(1024, &schedule, 4);
        assert!(
            final_unit.contains("path 4/4") && final_unit.contains("t=0"),
            "last unit should close path 4 at t=0, got {final_unit:?}"
        );
    }

    /// `--out` is the way to opt out of the naming convention: the file is
    /// written under the name that was asked for, and nothing else is dropped
    /// beside it. A scripted arm that names `runs/adam_lr1e-3.ckpt` must find
    /// exactly that file — every bench under `bench/` reads its runs back by
    /// the path it passed.
    #[test]
    fn an_explicit_out_is_written_exactly_as_named() {
        let asked = "/tmp/somewhere/adam_lr1e-3.ckpt";
        let (path, maintain_latest) =
            headless_checkpoint_target(Some(asked.to_string()), "Greyscale_Diffusion", now())
                .expect("an explicit --out needs no directory prepared");

        assert_eq!(path, PathBuf::from(asked));
        assert!(
            !maintain_latest,
            "--out must not drop a latest.ckpt beside the file the user named"
        );
    }

    /// Without `--out` the run is dated *and* stays in scratch: a CI run that
    /// wrote into `Models/<name>/pretrained_weights/` would quietly become the
    /// model's weights.
    #[test]
    fn a_headless_run_without_out_is_dated_and_stays_in_scratch() {
        let (path, maintain_latest) =
            headless_checkpoint_target(None, "Greyscale_Diffusion", now())
                .expect("scratch directory");

        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("run-") && name.ends_with(".ckpt"), "{name}");
        assert_eq!(
            path.parent(),
            Some(headless_scratch_dir("Greyscale_Diffusion").as_path()),
            "the scratch run escaped its directory"
        );
        assert!(
            !path.starts_with(storage::project_root().join("Models")),
            "a headless run must never write into a model's saved weights"
        );
        assert!(maintain_latest);
    }

    fn now() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_775_000_000)
    }

    /// The single-path case is the common one; naming a path there is noise.
    #[test]
    fn denoising_label_omits_path_when_there_is_only_one() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        assert!(!denoising_progress_label(10, &schedule, 1).contains("path"));
    }

    fn write_batraw(
        path: &std::path::Path,
        count: u32,
        width: u32,
        height: u32,
        channels: u32,
        samples: &[Vec<f32>],
    ) {
        write_batraw_with_magic(
            path,
            RAW_DATASET_MAGIC_SIGNED,
            count,
            width,
            height,
            channels,
            samples,
        )
    }

    fn write_batraw_with_magic(
        path: &std::path::Path,
        magic: &[u8; 8],
        count: u32,
        width: u32,
        height: u32,
        channels: u32,
        samples: &[Vec<f32>],
    ) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(magic).unwrap();
        for v in [count, width, height, channels] {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        for sample in samples {
            for &v in sample {
                file.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    /// A BATRAW3 file: the same header, an 8-bit payload.
    fn write_batraw3(
        path: &std::path::Path,
        count: u32,
        width: u32,
        height: u32,
        channels: u32,
        samples: &[Vec<u8>],
    ) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(RAW_DATASET_MAGIC_BYTES).unwrap();
        for v in [count, width, height, channels] {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        for sample in samples {
            file.write_all(sample).unwrap();
        }
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("batlab_test_{name}"));
        dir
    }

    /// Reads a `BATFLUX1` dump back as `(phase tag, t)` per frame — the same way
    /// `tools/flux_analysis.py` does, and deliberately not through any of the
    /// writing code, so a mistake in the layout cannot cancel itself out.
    fn read_dump_tags(path: &std::path::Path) -> Vec<(u8, u32)> {
        let bytes = std::fs::read(path).expect("dump should exist");
        assert_eq!(&bytes[..8], b"BATFLUX1", "magic");
        let u32_at = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let (w, h, c) = (u32_at(8), u32_at(12), u32_at(16));
        // One frame: the tag, the level, then x_t and x̂₀ back to back.
        let frame = 1 + 4 + (w * h * c) as usize * 4 * 2;
        let mut tags = Vec::new();
        let mut at = 20;
        while at + frame <= bytes.len() {
            tags.push((bytes[at], u32_at(at + 1)));
            at += frame;
        }
        assert_eq!(at, bytes.len(), "the file should hold whole frames only");
        tags
    }

    /// The frontier a reader of the dump sees between the approach and the churn
    /// must be the frontier the drift actually walks — the first frame *produced
    /// by* the churn is a churn frame.
    ///
    /// This is the check that was missing when the regime shipped: the drift's
    /// own tests all read the phase off the action and were green, while the
    /// dump — the only thing any analysis of flux ever looks at — recorded
    /// `drift.phase()` sampled *before* the step and so ran one frame late. A
    /// black-box run caught it from the outside by amplitude (the frame tagged
    /// `descent` moved like a churn frame, some 40 % above a reverse step at the
    /// same level); this pins it by name, on the bytes themselves.
    #[test]
    fn the_dump_files_the_first_churn_frame_as_flux() {
        let out = tmp_path("flux_phase_frontier.batflux");
        let (steps, t_star) = (32usize, 8usize);
        let mut drift =
            PerpetualDrift::new(steps, batlab_core::PerpetualRegime::Flux, t_star, 7);

        // 2×2×1 frames: the payload is irrelevant here, the header is not.
        let pixels = vec![0.0_f32; 4];
        let mut dump = FrameDump::create(&out, (2, 2, 1)).expect("dump should open");
        for _ in 0..steps {
            let action = drift.step();
            let level = match action {
                DriftAction::Descend { diffusion_step, .. } => diffusion_step.saturating_sub(1),
                DriftAction::Climb { forward_step, .. } => forward_step,
                DriftAction::Flux { diffusion_step, .. } => diffusion_step,
            };
            dump.record(action, level, &pixels, &pixels).expect("record");
        }
        dump.finish().expect("flush");

        let tags = read_dump_tags(&out);
        let _ = std::fs::remove_file(&out);
        assert_eq!(tags.len(), steps, "one frame recorded per action");

        // The descent runs t = steps-1 down to t*+1; the level below t*+1 *is*
        // t*, so the very next action is already the churn.
        let approach = steps - 1 - t_star;
        let phases: Vec<u8> = tags.iter().map(|(tag, _)| *tag).collect();
        assert_eq!(
            phases.iter().filter(|tag| **tag == 0).count(),
            approach,
            "descent frames counted in the dump — one too many means the tag is \
             lagging a frame behind the deed: {phases:?}"
        );
        assert_eq!(phases[approach - 1], 0, "the last approach frame descends");
        assert_eq!(
            phases[approach], 2,
            "and the one after it is churn, not a twenty-fourth descent"
        );
        assert!(
            phases[approach..].iter().all(|tag| *tag == 2),
            "the churn never stops on a fixed dial: {phases:?}"
        );
        assert!(
            tags[approach..].iter().all(|(_, t)| *t as usize == t_star),
            "and it holds t* throughout: {tags:?}"
        );
    }

    /// The number the model list puts in front of a user, checked against the
    /// **real** buffers a built model allocates.
    ///
    /// `summarize_architecture` counts parameters from the config alone — no
    /// GPU, no checkpoint — which is what makes it affordable on every
    /// keystroke and also what makes it easy to get quietly wrong: forget a
    /// bias vector and every model in the list is understated by a few hundred,
    /// with nothing on screen to say so. A checkpoint holds exactly the
    /// trainable scalars and nothing else, so counting them is an independent
    /// oracle rather than the same arithmetic written twice.
    ///
    /// Run on both built-in templates, which between them exercise
    /// convolution, GroupNorm, upsample+conv, activation and concat.
    #[test]
    fn the_parameter_count_matches_the_scalars_a_checkpoint_holds() {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        for template in batlab_core::config::built_in_templates() {
            let config = ModelConfig {
                model_name: Some(template.key.clone()),
                input_size: template.input_size,
                layers: template.layers.clone(),
                inference: batlab_core::InferenceConfig::default(),
                seed_dataset: None,
                run: batlab_core::RunConfig {
                    mode: RunMode::Infer,
                },
            };
            let counted =
                batlab_core::config::summarize_architecture(&config.layers, config.input_size)
                    .parameters;

            let held = rt.block_on(async {
                let (_gpu, model) = build_execution_model(
                    &config,
                    INFERENCE_RUNTIME_LR,
                    INFERENCE_RUNTIME_BATCH_SIZE,
                    OptimizerKind::Sgd,
                    WeightInit::default(),
                    None,
                )
                .await
                .expect("template must build");
                // `BBCKPT2` + entry count, then per entry: layer index, weight
                // count, weights, bias count, biases; then the optimiser
                // trailer, which SGD writes as a bare 4-byte `none` tag. The
                // scalars are read from the declared lengths rather than from
                // the file size, so the trailer cannot be mistaken for weights.
                let bytes = model.checkpoint_bytes().expect("checkpoint");
                let u32_at = |at: usize| {
                    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
                        as usize
                };
                let entries = u32_at(7);
                let mut at = 11;
                let mut scalars = 0u64;
                for _ in 0..entries {
                    at += 4; // layer index
                    let weights = u32_at(at);
                    at += 4 + weights * 4;
                    let bias = u32_at(at);
                    at += 4 + bias * 4;
                    scalars += (weights + bias) as u64;
                }
                assert_eq!(
                    at + 4,
                    bytes.len(),
                    "the walk did not land on the SGD trailer — the checkpoint \
                     layout moved and this oracle is reading noise"
                );
                scalars
            });

            assert_eq!(
                counted, held,
                "template '{}': the panel would say {counted} parameters where the \
                 model holds {held}",
                template.key
            );
        }
    }

    // -- Exporting the weights inference uses --------------------------------

    /// A tiny t-conditioned diffusion model: 8×8, one signal channel and one
    /// time channel in, one channel out. Small enough that a full 256-step
    /// reverse chain is cheap, real enough that `sample_diffusion` runs it end
    /// to end (input depth 2 > output depth 1, the conditioning invariant).
    fn tiny_diffusion_config() -> ModelConfig {
        use batlab_core::config::{ActivationMethod, PaddingMode};
        let layers = vec![
            LayerDraft::Convolution {
                dim_input: (8, 8, 2),
                nb_kernel: 4,
                dim_kernel: (3, 3, 2),
                stride: 1,
                padding: PaddingMode::Same,
                save_key: None,
            },
            LayerDraft::Activation {
                dim_input: (8, 8, 4),
                method: ActivationMethod::Silu,
                save_key: None,
            },
            LayerDraft::Convolution {
                dim_input: (8, 8, 4),
                nb_kernel: 1,
                dim_kernel: (3, 3, 4),
                stride: 1,
                padding: PaddingMode::Same,
                save_key: None,
            },
        ];
        ModelConfig {
            model_name: Some("export-test".to_string()),
            input_size: (8, 8, 2),
            layers,
            inference: batlab_core::InferenceConfig::default(),
            seed_dataset: None,
            run: batlab_core::RunConfig {
                mode: RunMode::Infer,
            },
        }
    }

    /// Train `config` a few steps with Adam and an EMA so the averaged set and
    /// the last iterate genuinely differ, and return its V3 checkpoint bytes.
    async fn trained_v3_checkpoint(config: &ModelConfig) -> Vec<u8> {
        let ema = EmaConfig::new(0.9).expect("decay in range");
        let (_gpu, mut model) = build_execution_model(
            config,
            1e-2,
            1,
            OptimizerKind::Adam,
            WeightInit::default(),
            Some(ema),
        )
        .await
        .expect("tiny model must build");

        let input_len = 8 * 8 * 2;
        let output_len = 8 * 8 * 1;
        for step in 0..8 {
            let input: Vec<f32> = (0..input_len)
                .map(|i| ((i as f32 * 0.37 + step as f32 * 0.11).sin()) * 0.8)
                .collect();
            let target: Vec<f32> = (0..output_len)
                .map(|i| ((i as f32 * 0.21 - step as f32 * 0.17).cos()) * 0.5)
                .collect();
            model.train_step(&input, &target);
        }
        model.checkpoint_bytes().expect("checkpoint")
    }

    /// Generate one image from `bytes`, loaded with `weights`, on a fresh
    /// inference model — the same path `--headless-sample` walks.
    async fn sample_from_checkpoint(
        config: &ModelConfig,
        bytes: &[u8],
        weights: CheckpointWeights,
        seed: u64,
    ) -> Vec<f32> {
        let (_gpu, mut model) = build_execution_model(
            config,
            INFERENCE_RUNTIME_LR,
            INFERENCE_RUNTIME_BATCH_SIZE,
            OptimizerKind::default(),
            WeightInit::default(),
            None,
        )
        .await
        .expect("inference model must build");
        model
            .load_checkpoint_bytes_with(bytes, weights)
            .expect("load checkpoint");

        let input_dims = model.input_dim().expect("input dims");
        let output_dims = model.output_dim().expect("output dims");
        let output_len = (output_dims.x * output_dims.y * output_dims.z) as usize;
        let schedule = LinearNoiseSchedule::new_linear(
            DIFFUSION_SCHEDULE_STEPS,
            DIFFUSION_BETA_START,
            DIFFUSION_BETA_END,
        );
        sample_diffusion(
            &mut model,
            input_dims.z as usize,
            output_dims.z as usize,
            output_len,
            &schedule,
            seed,
            1,
            1.0,
            None,
            None,
            |_, _| {},
        )
    }

    /// The property the whole feature rests on: an export samples **bit for
    /// bit** the same image, on the same seed, as the checkpoint it came from —
    /// selecting the same weight set the page selects (the average by default,
    /// the raw iterate under `--raw-weights`). Proven, not inspected: two full
    /// reverse chains compared value by value.
    ///
    /// The export is also checked to be a smaller, ordinary V2 checkpoint — the
    /// optimiser moments and the EMA trailer are gone, the weights remain — and
    /// the EMA and raw references are checked to differ, so "same image" is not
    /// the vacuous truth of a model whose two weight sets coincide.
    #[test]
    fn an_export_samples_the_same_image_as_the_checkpoint_it_came_from() {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let config = tiny_diffusion_config();
            let source = trained_v3_checkpoint(&config).await;
            assert_eq!(
                &source[..7],
                b"BBCKPT3",
                "a run with an EMA writes a V3 checkpoint"
            );

            let ema_ref = sample_from_checkpoint(&config, &source, CheckpointWeights::Ema, 101).await;
            let raw_ref = sample_from_checkpoint(&config, &source, CheckpointWeights::Raw, 101).await;
            assert_ne!(
                ema_ref, raw_ref,
                "the averaged and raw weight sets sample the same image — the \
                 equivalence below would then be vacuous"
            );

            for (weights, reference) in [
                (CheckpointWeights::Ema, &ema_ref),
                (CheckpointWeights::Raw, &raw_ref),
            ] {
                let exported = stripped_checkpoint(&config, &source, weights, false)
                    .await
                    .expect("export");
                assert_eq!(
                    &exported[..7],
                    b"BBCKPT2",
                    "a weights-only export carries no EMA trailer, so it is a V2 file"
                );
                assert!(
                    exported.len() < source.len(),
                    "the export ({} bytes) is not lighter than the source ({} bytes)",
                    exported.len(),
                    source.len()
                );

                // The export has no EMA trailer, so `Ema` falls back to its raw
                // entries — which are exactly the set that was baked in.
                let sampled =
                    sample_from_checkpoint(&config, &exported, CheckpointWeights::Ema, 101).await;
                assert_eq!(
                    &sampled, reference,
                    "the export sampled a different image from the checkpoint it \
                     came from"
                );
            }
        });
    }

    // -- The img2img seed ----------------------------------------------------

    /// A model config of the given output channels and nothing else — enough
    /// for `SeedImages::resolve`, which reads only `seed_dataset` off it.
    fn a_model_naming(seed_dataset: Option<&str>) -> ModelConfig {
        ModelConfig {
            model_name: Some("seed-test".to_string()),
            input_size: (2, 2, 3),
            layers: Vec::new(),
            inference: batlab_core::InferenceConfig::default(),
            seed_dataset: seed_dataset.map(str::to_string),
            run: batlab_core::RunConfig {
                mode: RunMode::Infer,
            },
        }
    }

    /// **The ranking**: `--seed-dataset` beats the model's `seed_dataset`,
    /// which beats the convention derived from the output channels.
    ///
    /// Each source names a *different* file of distinguishable images, so the
    /// answer says which one won rather than merely that something loaded. The
    /// middle rung is the one that did not exist: `--headless-perpetual` read
    /// the flag or nothing at all, so `Models/Elephants_XL` set out from a
    /// CIFAR truck for an entire campaign.
    #[test]
    fn the_flag_beats_the_config_which_beats_the_convention() {
        let by_flag = tmp_path("seed_rank_flag.batraw");
        let by_config = tmp_path("seed_rank_config.batraw");
        // One constant sample each, its value identifying the file.
        write_batraw(&by_flag, 1, 2, 2, 1, &[vec![0.25; 4]]);
        write_batraw(&by_config, 1, 2, 2, 1, &[vec![-0.75; 4]]);

        let flag = by_flag.to_string_lossy().into_owned();
        let config = a_model_naming(Some(&by_config.to_string_lossy()));

        let both = SeedImages::resolve(Some(&flag), &config, (2, 2, 1))
            .expect("both present")
            .expect("a dataset");
        assert_eq!(both.source(), SeedSource::Flag);
        assert_eq!(both.path(), by_flag, "the flag must win over the config");

        let config_only = SeedImages::resolve(None, &config, (2, 2, 1))
            .expect("config present")
            .expect("a dataset");
        assert_eq!(config_only.source(), SeedSource::Config);
        assert_eq!(
            config_only.path(),
            by_config,
            "the model's own seed_dataset must be honoured when no flag is given — \
             this is the whole defect: it was declared, documented, and never read"
        );

        // Neither: the convention takes over, and it is derived from the output
        // channels. `7` matches no built-in file, so it resolves to nothing at
        // all rather than to one of the other two.
        let neither = a_model_naming(None);
        assert!(
            matches!(SeedImages::resolve(None, &neither, (2, 2, 7)), Ok(None)),
            "with nothing named, the convention decides — and it has no answer \
             for a 7-channel model"
        );

        let _ = std::fs::remove_file(&by_flag);
        let _ = std::fs::remove_file(&by_config);
    }

    /// A seed dataset whose **channels** disagree with the model is refused,
    /// naming both counts — not resized.
    ///
    /// Width and height are resampled on the host, which is a visible thing to
    /// do to a picture. Channels are not: a greyscale file handed to a colour
    /// model is replicated across R, G and B and the run *succeeds*, drifting
    /// away from a grey picture pretending to be colour. That silence has cost
    /// this repository time before, on the training path.
    ///
    /// Checked on both named sources, because both are a human's decision and
    /// both used to be swallowed.
    #[test]
    fn a_seed_dataset_of_the_wrong_channels_is_refused_not_resized() {
        let grey = tmp_path("seed_wrong_channels.batraw");
        let samples: Vec<Vec<f32>> = (0..4).map(|_| vec![0.5; 4]).collect();
        write_batraw(&grey, 4, 2, 2, 1, &samples);
        let named = grey.to_string_lossy().into_owned();

        // The model emits three channels; the file holds one.
        for (flag, config) in [
            (Some(named.as_str()), a_model_naming(None)),
            (None, a_model_naming(Some(&named))),
        ] {
            let err = match SeedImages::resolve(flag, &config, (2, 2, 3)) {
                Err(err) => err,
                Ok(_) => panic!("a 1-channel file must not seed a 3-channel model"),
            };
            assert!(
                err.contains('1') && err.contains('3'),
                "the refusal must name both counts, or it cannot be acted on: {err}"
            );
        }

        // And the same file is fine for the model it actually fits.
        assert!(
            SeedImages::resolve(Some(&named), &a_model_naming(None), (2, 2, 1))
                .expect("matching channels must load")
                .is_some()
        );
        let _ = std::fs::remove_file(&grey);
    }

    /// **The claim of the mode**: the latent a perpetual run sets out from is a
    /// real image of the dataset, bit for bit — not noise, not a resized
    /// approximation of one, not a blend.
    ///
    /// Written against distinct, recognisable samples so the answer identifies
    /// *which* image was drawn: a provider that returned the first sample every
    /// time, or an average, would pass a "not noise" check and fail this.
    #[test]
    fn the_opening_latent_is_a_dataset_image_to_the_bit() {
        let out = tmp_path("seed_images.batraw");
        // Eight 2×2 greyscale samples, each a constant of its own value.
        let samples: Vec<Vec<f32>> = (0..8)
            .map(|s| vec![s as f32 / 8.0 - 0.5; 4])
            .collect();
        write_batraw(&out, 8, 2, 2, 1, &samples);

        let images = SeedImages::resolve(Some(&out.to_string_lossy()), &a_model_naming(None), (2, 2, 1))
            .expect("load should succeed")
            .expect("an explicit path must yield a source");
        assert_eq!(images.len(), 8);

        for seed in 0u64..64 {
            let x0 = images.provide_x0(seed);
            assert!(
                samples.iter().any(|sample| sample
                    .iter()
                    .zip(x0.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6)),
                "seed {seed} produced a latent that is in the dataset nowhere: {x0:?}"
            );
        }

        let _ = std::fs::remove_file(&out);
    }

    /// The draw has to be a draw. `seed % len` on seeds that come from a clock
    /// or from a form walks the dataset in order — consecutive `[r]` presses
    /// would hand out consecutive images, and a run would be correlated with
    /// whatever order the file happens to be in. The index is therefore
    /// avalanched first (same discipline as `gaussian_at`,
    /// `ANISOTROPY_HUNT.md`).
    ///
    /// Checked two ways: consecutive seeds must not give consecutive indices,
    /// and 64 draws over 8 images must reach every one of them.
    #[test]
    fn consecutive_seeds_do_not_draw_consecutive_images() {
        let out = tmp_path("seed_images_spread.batraw");
        let samples: Vec<Vec<f32>> = (0..8).map(|s| vec![s as f32 / 8.0 - 0.5; 4]).collect();
        write_batraw(&out, 8, 2, 2, 1, &samples);
        let images = SeedImages::resolve(Some(&out.to_string_lossy()), &a_model_naming(None), (2, 2, 1))
            .expect("load")
            .expect("source");
        let _ = std::fs::remove_file(&out);

        let drawn: Vec<usize> = (0u64..64).map(|seed| images.at_random(seed)).collect();
        let stepping = drawn
            .windows(2)
            .filter(|pair| (pair[1] + 8 - pair[0]) % 8 == 1)
            .count();
        assert!(
            stepping < 24,
            "{stepping} of 63 consecutive seeds stepped one image forward — the \
             index is following the seed instead of being drawn from it: {drawn:?}"
        );
        let reached: std::collections::HashSet<usize> = drawn.iter().copied().collect();
        assert_eq!(reached.len(), 8, "only {} of 8 images reachable", reached.len());
    }

    /// A dataset named on the command line and missing is an **error**: falling
    /// back on noise would look exactly like the flag being ignored. A dataset
    /// merely *derived* and missing is not — that is the machine having no
    /// CIFAR handy, and the run still has a pure-noise opening to fall back on.
    #[test]
    fn a_named_dataset_that_is_missing_is_an_error_but_a_derived_one_is_not() {
        let missing = tmp_path("no_such_dataset.batraw");
        let _ = std::fs::remove_file(&missing);
        assert!(
            SeedImages::resolve(Some(&missing.to_string_lossy()), &a_model_naming(None), (2, 2, 1))
                .is_err(),
            "a named dataset that is not there must be reported, not swallowed"
        );
        // A geometry no built-in dataset matches: nothing to derive, no error.
        assert!(
            matches!(
                SeedImages::resolve(None, &a_model_naming(None), (2, 2, 7)),
                Ok(None)
            ),
            "a 7-channel model has no default dataset and must not fail for it"
        );
    }

    /// **One resolution, and every drift start goes through it.**
    ///
    /// This is the mechanical half of the fix, and the half that keeps it
    /// fixed. The defect was never that the ranking was wrong — it was that
    /// `--headless-perpetual` resolved its opening picture *its own way* and so
    /// never learned about `seed_dataset`, while the TUI path did. Two ways to
    /// answer one question is how they drifted apart, and a reviewer reading
    /// either one in isolation sees nothing wrong.
    ///
    /// So: any function that opens a drift — one that mentions
    /// `PerpetualDrift::from_image`, the constructor that means "set out from a
    /// picture" — must also call [`SeedImages::resolve`]. A third perpetual
    /// entry point that resolved its own seed would fail here, by name.
    ///
    /// Same discipline as `load_sampling_checkpoint` for weights, and as
    /// `nothing_in_the_engine_opens_an_untimed_pass` for compute passes.
    #[test]
    fn every_path_that_starts_a_drift_resolves_its_seed_the_same_way() {
        let source = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs"),
        )
        .expect("this binary's own source must be readable");

        // Split into top-level items: a line starting at column 0 with `fn`,
        // `async fn` or `pub fn` opens one, and it runs to the next such line.
        let mut blocks: Vec<(String, String)> = Vec::new();
        for line in source.lines() {
            let opens = line.starts_with("fn ")
                || line.starts_with("async fn ")
                || line.starts_with("pub fn ")
                || line.starts_with("pub async fn ");
            if opens {
                let name = line
                    .split(['(', '<'])
                    .next()
                    .unwrap_or(line)
                    .rsplit(' ')
                    .next()
                    .unwrap_or(line)
                    .to_string();
                blocks.push((name, String::new()));
            }
            if let Some(block) = blocks.last_mut() {
                block.1.push('\n');
                block.1.push_str(line);
            }
        }

        // Assembled from halves, so this test is not itself an offender — the
        // detector has to be allowed to name what it detects.
        let starts_a_drift = concat!("PerpetualDrift", "::from_image");
        let the_one_door = concat!("SeedImages", "::resolve");

        let mut offenders: Vec<&str> = Vec::new();
        let mut checked = 0usize;
        for (name, body) in &blocks {
            if !body.contains(starts_a_drift) || name.starts_with("a_")
            // the test module's own helpers
            {
                continue;
            }
            checked += 1;
            if !body.contains(the_one_door) {
                offenders.push(name);
            }
        }

        assert!(
            checked >= 2,
            "only {checked} function(s) open a drift — the scan stopped seeing the \
             two perpetual paths, so it is no longer proving anything"
        );
        assert!(
            offenders.is_empty(),
            "these start a perpetual drift without going through SeedImages::resolve, \
             so they answer 'which pictures?' their own way and will drift from the \
             common ranking exactly as --headless-perpetual did: {offenders:?}"
        );
    }

    #[test]
    fn raw_dataset_greyscale_round_trip() {
        let out = tmp_path("grey.batraw");
        // One 2×2 greyscale sample, in the [-1, 1] convention.
        let sample = vec![-1.0_f32, -0.5, 0.0, 1.0];
        write_batraw(&out, 1, 2, 2, 1, &[sample.clone()]);

        let dataset = try_load_raw_dataset(&out, (2, 2, 1))
            .expect("load should succeed")
            .expect("should detect .batraw file");

        let _ = std::fs::remove_file(&out);
        assert_eq!(dataset.len(), 1);
        for (a, b) in dataset.sample(0).iter().zip(sample.iter()) {
            assert!((a - b).abs() < 1e-6, "value mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn raw_dataset_rgb_round_trip() {
        let out = tmp_path("rgb.batraw");
        // One 2×2 RGB sample (4 pixels × 3 channels), in the [-1, 1] convention.
        let sample: Vec<f32> = (0..12).map(|i| i as f32 / 11.0 * 2.0 - 1.0).collect();
        write_batraw(&out, 1, 2, 2, 3, &[sample.clone()]);

        let dataset = try_load_raw_dataset(&out, (2, 2, 3))
            .expect("load should succeed")
            .expect("should detect .batraw file");

        let _ = std::fs::remove_file(&out);
        assert_eq!(dataset.len(), 1);
        for (a, b) in dataset.sample(0).iter().zip(sample.iter()) {
            assert!((a - b).abs() < 1e-6, "value mismatch: {a} vs {b}");
        }
    }

    /// Colour path — a dataset sample must reach the PNG with its channels intact.
    ///
    /// The whole pipeline is HWC/z-fastest (`convolution.wgsl`: `iy*W*C + ix*C + iz`),
    /// which a 1-channel model can never exercise: with C=1 an interleaved and a planar
    /// layout are the same bytes. This pins the convention on C=3 from the `.batraw`
    /// header all the way to the encoded pixels, so a planar/interleaved slip shows up
    /// as a failing test rather than as plausible-looking garbage in a sample.
    #[test]
    fn rgb_dataset_sample_survives_to_png_with_channels_unswapped() {
        let raw = tmp_path("rgb_png_path.batraw");
        // 2×2, one saturated primary per pixel: red, green, blue, white.
        let sample: Vec<f32> = vec![
            1.0, -1.0, -1.0, // pixel (0,0) rouge
            -1.0, 1.0, -1.0, // pixel (1,0) vert
            -1.0, -1.0, 1.0, // pixel (0,1) bleu
            1.0, 1.0, 1.0, // pixel (1,1) blanc
        ];
        write_batraw(&raw, 1, 2, 2, 3, &[sample]);

        let dataset = try_load_raw_dataset(&raw, (2, 2, 3))
            .expect("load should succeed")
            .expect("should detect .batraw file");
        let _ = std::fs::remove_file(&raw);

        let png = tmp_path("rgb_png_path.png");
        write_tensor_png(&dataset.sample(0), (2, 2, 3), &png).expect("png should be written");
        let decoded = image::open(&png).expect("png should decode").to_rgb8();
        let _ = std::fs::remove_file(&png);

        assert_eq!(decoded.dimensions(), (2, 2));
        let expected = [
            [255u8, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
        ];
        for (pixel, want) in decoded.pixels().zip(expected.iter()) {
            assert_eq!(&pixel.0, want, "channel layout changed on the colour path");
        }
    }

    /// Finding #4 — legacy BATRAW1 payloads are stored in [0, 1] and must be
    /// rescaled to the [-1, 1] convention the diffusion pipeline expects.
    #[test]
    fn raw_dataset_legacy_unit_payload_is_rescaled_to_signed_range() {
        let out = tmp_path("legacy.batraw");
        let stored = vec![0.0_f32, 0.25, 0.5, 1.0];
        let expected = [-1.0_f32, -0.5, 0.0, 1.0];
        write_batraw_with_magic(&out, RAW_DATASET_MAGIC_UNIT, 1, 2, 2, 1, &[stored]);

        let dataset = try_load_raw_dataset(&out, (2, 2, 1))
            .expect("load should succeed")
            .expect("should detect .batraw file");

        let _ = std::fs::remove_file(&out);
        assert_eq!(dataset.len(), 1);
        for (got, want) in dataset.sample(0).iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "value mismatch: {got} vs {want}");
        }
    }

    /// BATRAW3 is a **lossless** re-encoding, and this is the proof the claim
    /// rests on.
    ///
    /// The two files hold the same images: one as the f32 values the old
    /// converter wrote, one as the bytes those values came from. Decoded, they
    /// must be equal **bit for bit** — not close. Anything less and "a quarter
    /// of the size, at no cost" would be a quarter of the size at a cost nobody
    /// could see: a systematic shift of the whole corpus, invisible in a loss
    /// curve and impossible to attribute later.
    ///
    /// (The GPU side of the same equality is
    /// `the_gpu_decode_agrees_with_the_cpu_one` in `training/dataset.rs` — the
    /// widening happens there for training, and here for the probe.)
    #[test]
    fn an_8_bit_file_decodes_to_the_same_bits_as_the_f32_one_it_replaces() {
        // Sixteen bytes spanning the range, including both ends and mid-grey.
        let source: Vec<u8> = vec![0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 191, 200, 252, 253, 254, 255];
        let as_floats: Vec<f32> = source.iter().copied().map(from_u8).collect();

        let old = tmp_path("lossless_f32.batraw");
        let new = tmp_path("lossless_u8.batraw");
        write_batraw(&old, 1, 4, 4, 1, &[as_floats.clone()]);
        write_batraw3(&new, 1, 4, 4, 1, &[source.clone()]);

        let from_old = try_load_raw_dataset(&old, (4, 4, 1)).unwrap().unwrap();
        let from_new = try_load_raw_dataset(&new, (4, 4, 1)).unwrap().unwrap();

        // The reason for all of the above: same images, a quarter of the
        // payload. Measured on the files, header excluded.
        let header = 24;
        let old_payload = std::fs::metadata(&old).unwrap().len() - header;
        let new_payload = std::fs::metadata(&new).unwrap().len() - header;
        assert_eq!(old_payload, new_payload * 4);
        let _ = std::fs::remove_file(&old);
        let _ = std::fs::remove_file(&new);

        assert_eq!(from_old.len(), 1);
        assert_eq!(from_new.len(), 1);
        assert_eq!(
            from_new.sample(0).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            from_old.sample(0).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the 8-bit file decoded to different bits than the f32 file it replaces"
        );
    }

    /// An 8-bit file whose geometry matches must stay 8-bit: widening it on the
    /// host would give the right pixels and throw away the entire point.
    #[test]
    fn a_matching_8_bit_file_reaches_the_gpu_as_bytes() {
        let out = tmp_path("stays_bytes.batraw");
        write_batraw3(&out, 2, 2, 2, 1, &[vec![0, 64, 128, 255], vec![255, 128, 64, 0]]);
        let dataset = try_load_raw_dataset(&out, (2, 2, 1)).unwrap().unwrap();
        let _ = std::fs::remove_file(&out);

        assert!(
            matches!(dataset.payload, DatasetPayload::Bytes(_)),
            "a matching 8-bit dataset was widened on the host"
        );
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.sample(1), vec![1.0, from_u8(128), from_u8(64), -1.0]);
    }

    /// A geometry that does not match still has to be resampled, and that
    /// happens on the host — so the payload falls back to f32 rather than
    /// pretending the bytes are usable as they lie.
    #[test]
    fn a_mismatched_8_bit_file_falls_back_to_the_float_path() {
        let out = tmp_path("resampled_bytes.batraw");
        write_batraw3(&out, 1, 4, 4, 1, &[vec![128u8; 16]]);
        let dataset = try_load_raw_dataset(&out, (2, 2, 1)).unwrap().unwrap();
        let _ = std::fs::remove_file(&out);

        assert!(
            matches!(dataset.payload, DatasetPayload::Floats(_)),
            "a resampled dataset must not claim to be 8-bit"
        );
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.sample(0).len(), 4);
    }

    /// The header reader has to report the payload width, because the resource
    /// inventory multiplies by it. Reading a BATRAW3 file as f32 would predict
    /// four times the residency it will actually get.
    #[test]
    fn the_header_reports_how_wide_a_value_is() {
        let f32_file = tmp_path("header_f32.batraw");
        let u8_file = tmp_path("header_u8.batraw");
        write_batraw(&f32_file, 1, 2, 2, 3, &[vec![0.0; 12]]);
        write_batraw3(&u8_file, 1, 2, 2, 3, &[vec![0u8; 12]]);

        assert_eq!(read_batraw_header(&f32_file).unwrap(), (1, 2, 2, 3, 4));
        assert_eq!(read_batraw_header(&u8_file).unwrap(), (1, 2, 2, 3, 1));
        let _ = std::fs::remove_file(&f32_file);
        let _ = std::fs::remove_file(&u8_file);
    }

    /// A truncated 8-bit file must be refused, not read short. The size check is
    /// the only thing standing between a partial download and a dataset that
    /// silently holds fewer images than its header claims.
    #[test]
    fn a_truncated_8_bit_file_is_refused() {
        let out = tmp_path("truncated.batraw");
        write_batraw3(&out, 4, 2, 2, 1, &[vec![0u8; 4], vec![0u8; 4]]);
        let result = try_load_raw_dataset(&out, (2, 2, 1));
        let _ = std::fs::remove_file(&out);
        assert!(result.is_err(), "a short 8-bit payload was accepted");
    }

    /// Finding #4 — the encode/decode pair must be symmetric, so a sample that
    /// round-trips through an image file comes back unchanged.
    #[test]
    fn u8_encoding_round_trips_through_signed_range() {
        for byte in 0..=255u8 {
            assert_eq!(to_u8(from_u8(byte)), byte, "round trip failed for {byte}");
        }
        assert!((from_u8(0)).abs() - 1.0 < 1e-6);
        assert!((from_u8(255) - 1.0).abs() < 1e-6);
        // Mid-grey must land near zero — that is the whole point of the change.
        assert!(from_u8(128).abs() < 0.01);
    }

    #[test]
    fn raw_dataset_non_batraw_file_returns_none() {
        let out = tmp_path("image.png");
        std::fs::write(&out, b"notabatraw").unwrap();

        let result = try_load_raw_dataset(&out, (32, 32, 1)).unwrap();
        let _ = std::fs::remove_file(&out);
        assert!(result.is_none(), "non-.batraw file should return None");
    }

    #[test]
    fn raw_dataset_wrong_magic_returns_error() {
        let out = tmp_path("bad.batraw");
        // Header with correct structure but wrong magic.
        let mut data = b"WRONGMAG".to_vec();
        for v in [1u32, 2, 2, 1] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..4u32 {
            data.extend_from_slice(&0.5f32.to_le_bytes());
        }
        std::fs::write(&out, &data).unwrap();

        let result = try_load_raw_dataset(&out, (2, 2, 1));
        let _ = std::fs::remove_file(&out);
        assert!(result.is_err(), "wrong magic should return an error");
    }

    fn argv(line: &str) -> Vec<String> {
        std::iter::once("main".to_string())
            .chain(line.split_whitespace().map(str::to_string))
            .collect()
    }

    /// The three spellings of the level dial are one flag.
    ///
    /// Written against the blind test that caught the defect: the public
    /// contract offered `--t-star` and `--t-r`, only `--depth` was parsed, and
    /// nothing anywhere said so — the dumps came out byte-identical to a run
    /// with no flag at all.
    #[test]
    fn every_spelling_of_the_level_dial_is_parsed() {
        for line in [
            "--headless-perpetual M --regime flux --t-star 30",
            "--headless-perpetual M --regime flux --t-r 30",
            "--headless-perpetual M --regime flux --depth 30",
        ] {
            assert_eq!(
                dial_level(&argv(line)).expect("valid level"),
                Some(30),
                "not parsed: {line}"
            );
        }
        assert_eq!(
            dial_level(&argv("--headless-perpetual M --regime flux")).expect("no dial"),
            None,
            "absent dial must fall through to the caller's default"
        );
        assert!(
            dial_level(&argv("--headless-perpetual M --t-star abc")).is_err(),
            "a level that is not a number must be refused, not silently defaulted"
        );
    }

    /// A flag nobody parses has to be an error, not a shrug.
    ///
    /// This is what made the dial defect invisible: passing `--t-star`, passing
    /// `--t-r`, and passing a flag invented on the spot were three ways of
    /// getting the same run, so no experiment could tell "ignored" from
    /// "unimplemented".
    #[test]
    fn an_unknown_flag_is_refused_rather_than_ignored() {
        let valued = ["--headless-perpetual", "--regime", "--t-star"];
        let bare = ["--window"];

        assert!(
            reject_unknown_flags(
                &argv("--headless-perpetual M --regime flux --t-star 30 --window"),
                &valued,
                &bare
            )
            .is_ok()
        );
        let refused = reject_unknown_flags(
            &argv("--headless-perpetual M --flag-inexistant 30"),
            &valued,
            &bare,
        )
        .expect_err("an unknown flag must be refused");
        assert!(
            refused.contains("--flag-inexistant"),
            "the message must name the offending flag, got: {refused}"
        );
        // A value that looks like a flag belongs to the flag before it: paths
        // and negative numbers must not be mistaken for typos.
        assert!(
            reject_unknown_flags(&argv("--headless-perpetual --regime"), &valued, &bare).is_ok(),
            "the token after a known flag is its value, whatever it looks like"
        );
    }
}
