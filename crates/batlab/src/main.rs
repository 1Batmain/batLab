//! File purpose: Application entry point that orchestrates training, inference, and TUI workflows.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use batlab_ui::storage;
use batlab_ui::tui::{
    self, ActivationMethod, LayerDraft, ModelConfig, MonitorOutcome, PaddingMode, PerpetualConfig,
    RunMode, TrainingConfig,
};
use batlab_core::{
    ActivationMethod as PActivation, ActivationType, AttentionType, CheckpointWeights,
    ConvolutionType, DEFAULT_SNR_GAMMA,
    DenoiseFrame, DiffusionTask, Dim3, DriftAction, DriftWalk, EmaConfig, FullyConnectedType,
    GpuContext, GpuDataset,
    GroupNormType, LayerTypes, LinearNoiseSchedule, LiveFrame, LossMethod as PLoss, LossWeighting,
    MetricsLogger, Model, OptimizerKind, PaddingMode as PPadding, PerpetualDrift, ProbeConfig,
    Stats, Trainer, UpsampleConvType, WeightInit, compose_live_frame_view, log_probe,
    log_train_loss, log_trajectory, model::Training, probe_diffusion, sample_diffusion,
};
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};

// Magic headers for the raw binary dataset format produced by the pre-processing
// scripts. Format: magic(8) | count | width | height | channels | f32 data…
/// Legacy raw-dataset magic — payload stored in `[0, 1]`. Converted to the
/// model's `[-1, 1]` convention at load time.
const RAW_DATASET_MAGIC_UNIT: &[u8; 8] = b"BATRAW1\0";
/// Current raw-dataset magic — payload already stored in `[-1, 1]`.
const RAW_DATASET_MAGIC_SIGNED: &[u8; 8] = b"BATRAW2\0";

const DIFFUSION_SCHEDULE_STEPS: usize = 256;
const DIFFUSION_BETA_START: f32 = 1e-4;
const DIFFUSION_BETA_END: f32 = 2e-2;
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
      [--magnitude F] [--out <img>] [--log <jsonl>] [--raw-weights]

  --headless-perpetual <model> [--checkpoint <path>] [--regime wander|breathe|flux]
      [--t-r K | --t-star K | --depth K] [--seed N] [--magnitude F] [--dump <path>]
      [--frames N] [--actions N] [--window] [--climb-frames N] [--out <dir>]
      [--raw-weights]

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
                    --resume and the TUI weight selector both see it. Default:
                    only at the end of the run, as before.

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
        if args.iter().any(|arg| arg == "--headless-perpetual") {
            if let Err(err) = run_headless_perpetual(&args) {
                eprintln!("headless perpetual failed: {err}");
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

/// The dataset a model of `channels` output channels drifts away from, when its
/// config names none.
///
/// Derived from the model's **output** geometry rather than guessed: feeding a
/// greyscale drift the RGB file would not error — `try_load_raw_dataset` resizes
/// — and the run would quietly set out from a mangled picture.
fn default_seed_dataset_name(channels: u32) -> Option<&'static str> {
    match channels {
        1 => Some("cifar10_grey.batraw"),
        3 => Some("cifar10_rgb.batraw"),
        _ => None,
    }
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
    samples: Vec<ImageSample>,
    path: PathBuf,
}

impl SeedImages {
    /// Finds the dataset a model should drift away from.
    ///
    /// `explicit` wins when given (`--seed-dataset`, or `seed_dataset` in the
    /// model's `config_file`). Otherwise the choice follows the model's own
    /// **output** geometry: a 1-channel model wants `cifar10_grey.batraw`, a
    /// 3-channel one `cifar10_rgb.batraw`. Feeding a greyscale drift the RGB
    /// file would not error — `try_load_raw_dataset` would resize and the run
    /// would drift away from a mangled picture — so the default is derived
    /// rather than guessed at.
    ///
    /// `Ok(None)` when there is nothing to load: a missing dataset is not a
    /// reason to refuse to run, it is a reason to fall back on pure noise and
    /// say so.
    fn resolve(
        explicit: Option<&str>,
        output_size: (u32, u32, u32),
    ) -> Result<Option<Self>, String> {
        let candidate = match explicit {
            Some(path) => PathBuf::from(path),
            None => match default_seed_dataset_name(output_size.2) {
                Some(name) => storage::project_root().join("datasets").join(name),
                // A geometry neither file matches: nothing to default to, and
                // inventing one would drift away from the wrong images.
                None => return Ok(None),
            },
        };
        if !candidate.exists() {
            // Named explicitly, a missing file IS an error — silently drifting
            // from noise would look like the flag was ignored.
            return match explicit {
                Some(path) => Err(format!("--seed-dataset {path}: no such file")),
                None => Ok(None),
            };
        }
        let samples = load_dataset(&candidate.to_string_lossy(), output_size)?;
        if samples.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            samples,
            path: candidate,
        }))
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn path(&self) -> &Path {
        &self.path
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
        self.samples[self.at_random(seed)].target.clone()
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
    let checkpoint_path = flag("--out").unwrap_or_else(|| {
        std::env::temp_dir()
            .join(format!("{model_name}_headless.ckpt"))
            .to_string_lossy()
            .to_string()
    });

    let train_cfg = TrainingConfig {
        lr,
        batch_size,
        steps,
        dataset_path,
        loss: match &config.run.mode {
            RunMode::Train(existing) => existing.loss.clone(),
            RunMode::Infer | RunMode::Perpetual(_) => tui::LossMethod::MeanSquared,
        },
        checkpoint_path: Some(checkpoint_path),
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
         dataset={}",
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
        train_cfg.dataset_path
    );

    let (tx, rx) = std::sync::mpsc::channel::<tui::TrainingEvent>();
    let worker = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let options = RunOptions {
                resume_from,
                checkpoint_every,
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
            "--out",
            "--log",
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
            "headless sample '{model_name}': seed={seed} paths={paths} magnitude={magnitude}\n\
             image → {out_path}\nmetrics → {log_path}\n\
             final image stats: min={:.4} max={:.4} mean={:.4} std={:.4}",
            img_stats.min, img_stats.max, img_stats.mean, img_stats.std
        );
        Ok::<(), String>(())
    })
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
            false => SeedImages::resolve(seed_dataset.as_deref(), output_size)?,
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
        match seed_images.as_ref() {
            Some(images) => println!(
                "origine → image du dataset {} ({} images) — dérive img2img",
                images.path().display(),
                images.len()
            ),
            None => println!("origine → bruit pur en haut du schedule (--seed-noise)"),
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
        if let Err(err) = normalize_config_for_models_layout(&mut config) {
            eprintln!("failed to prepare model persistence: {err}");
            break;
        }

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
                        run_training(
                            config_clone,
                            train_cfg,
                            RunOptions::default(),
                            &tx,
                            control_rx,
                        )
                        .await
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

fn normalize_config_for_models_layout(config: &mut ModelConfig) -> Result<(), String> {
    let model_name = match config.model_name.clone() {
        Some(name) => name,
        None => {
            let generated = storage::next_model_name()
                .map_err(|err| format!("failed to allocate model name: {err}"))?;
            config.model_name = Some(generated.clone());
            generated
        }
    };

    if let RunMode::Train(train) = &mut config.run.mode
        && train.checkpoint_path.is_none()
    {
        train.checkpoint_path = Some(
            storage::default_model_checkpoint_path(&model_name)
                .map_err(|err| format!("failed to resolve checkpoint path: {err}"))?
                .to_string_lossy()
                .to_string(),
        );
    }

    storage::write_model_config(&model_name, config)
        .map_err(|err| format!("failed to write model config for '{model_name}': {err}"))?;
    Ok(())
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
    } else if let Some(path) = checkpoint_path.as_ref() {
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
    let gpu_samples: Vec<Vec<f32>> = dataset.into_iter().map(|sample| sample.target).collect();
    // CPU copies of a handful of clean targets kept for the diagnostic probe
    // (the GPU dataset is opaque to CPU-side readback).
    let probe_config = ProbeConfig::default();
    let probe_samples: Vec<Vec<f32>> = gpu_samples
        .iter()
        .take(probe_config.sample_count)
        .cloned()
        .collect();
    let mut gpu_dataset = GpuDataset::from_samples(gpu.as_ref(), gpu_samples, sample_len)
        .map_err(|err| format!("failed to upload dataset to GPU: {err}"))?;

    // Metrics land next to the checkpoint (`<stem>_metrics.jsonl`), or in a
    // temp file when no checkpoint path is configured. Truncated per run so the
    // file always describes the current run only.
    let mut metrics = {
        let metrics_path = checkpoint_path
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
                checkpoint_path.as_deref(),
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
                        checkpoint_path.as_deref(),
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
            write_partial_checkpoint(&model, checkpoint_path.as_deref(), step);
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

    if let Some(path) = checkpoint_path.as_ref() {
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
fn write_partial_checkpoint(model: &Model<Training>, checkpoint: Option<&Path>, step: usize) {
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
}

fn apply_and_publish_training_state(
    command: tui::TrainingControlCommand,
    model: &mut Model<Training>,
    paused: &mut bool,
    current_lr: &mut f32,
    current_batch_size: &mut u32,
    total_steps: &mut usize,
    checkpoint_path: Option<&Path>,
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
    let seed_images = SeedImages::resolve(cfg.seed_dataset.as_deref(), output_size)?;
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

    let origin_label = match seed_images.as_ref() {
        Some(_) => "image du dataset".to_string(),
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

fn load_dataset(
    dataset_path: &str,
    output_size: (u32, u32, u32),
) -> Result<Vec<ImageSample>, String> {
    let canonical_path = Path::new(dataset_path)
        .canonicalize()
        .map_err(|err| format!("failed to resolve dataset path '{}': {err}", dataset_path))?;

    if let Some(dataset) = try_load_raw_dataset(&canonical_path, output_size)? {
        return Ok(dataset);
    }

    if let Some(dataset) = try_load_cifar_dataset(&canonical_path, output_size)? {
        return Ok(dataset);
    }

    let mut image_paths = Vec::new();
    let mut visited_dirs = HashSet::new();
    collect_image_paths(&canonical_path, &mut image_paths, &mut visited_dirs)?;
    image_paths.sort();

    if image_paths.is_empty() {
        return Err(format!("no images found at '{}'", dataset_path));
    }

    image_paths
        .into_iter()
        .map(|path| {
            let image = image::open(&path)
                .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
            Ok(ImageSample {
                target: image_to_tensor(&image, output_size),
            })
        })
        .collect()
}

/// Tries to load a dataset from a raw binary file (`*.batraw`) or a directory that contains such
/// files.  Returns `Ok(None)` when the path does not look like a raw-binary dataset so the caller
/// can fall back to other loaders.
///
/// # Binary format (produced by the Python pre-processing scripts)
/// ```text
/// [0..8]   magic: b"BATRAW2\0" (or legacy b"BATRAW1\0")
/// [8..12]  count:    u32 LE – number of samples
/// [12..16] width:    u32 LE – image width in pixels
/// [16..20] height:   u32 LE – image height in pixels
/// [20..24] channels: u32 LE – number of channels per pixel
/// [24..]   data:     count * width * height * channels × f32 LE values
/// ```
///
/// `BATRAW2` stores values already normalised in `[-1, 1]` — the convention the
/// diffusion pipeline expects. `BATRAW1` files store `[0, 1]` and are rescaled
/// on the fly, so pre-existing datasets keep loading unchanged.
fn try_load_raw_dataset(
    dataset_path: &Path,
    output_size: (u32, u32, u32),
) -> Result<Option<Vec<ImageSample>>, String> {
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

    let mut dataset: Vec<ImageSample> = Vec::new();
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
        let needs_unit_rescale = if magic == RAW_DATASET_MAGIC_SIGNED {
            false
        } else if magic == RAW_DATASET_MAGIC_UNIT {
            true
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

        let sample_floats = (width * height * channels) as usize;
        let expected_bytes = offset + count * sample_floats * 4;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "raw dataset file size mismatch in {}: expected {expected_bytes} bytes, got {}",
                raw_file.display(),
                bytes.len()
            ));
        }

        // The geometry mismatch below is silently repaired by a u8 round-trip
        // (`raw_floats_to_dynamic_image` + `image_to_tensor`). That is convenient for
        // rescaling, but it also means feeding a 1-channel dataset to a 3-channel model
        // "works": every sample is grey replicated over R, G and B, and a whole overnight
        // run trains on colourless data without a single error. Say it out loud.
        if (width, height, channels) != output_size {
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

        for _ in 0..count {
            let raw: Vec<f32> = bytes[offset..offset + sample_floats * 4]
                .chunks_exact(4)
                .map(|b| {
                    let value = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    if needs_unit_rescale {
                        value * 2.0 - 1.0
                    } else {
                        value
                    }
                })
                .collect();
            offset += sample_floats * 4;

            // Rescale to the model's output dimensions if they differ.
            let target = if (width, height, channels) == output_size {
                raw
            } else {
                let image = raw_floats_to_dynamic_image(&raw, width, height, channels)?;
                image_to_tensor(&image, output_size)
            };
            dataset.push(ImageSample { target });
        }
    }

    Ok(Some(dataset))
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
fn from_u8(value: u8) -> f32 {
    value as f32 / 127.5 - 1.0
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

fn append_layer<State>(
    model: &mut Model<State>,
    draft: &LayerDraft,
) -> Result<(), batlab_core::ModelError> {
    match draft {
        LayerDraft::Convolution {
            dim_input,
            nb_kernel,
            dim_kernel,
            stride,
            padding,
            save_key,
        } => {
            model.add_layer(LayerTypes::Convolution(ConvolutionType::new(
                Dim3::new(*dim_input),
                *nb_kernel,
                Dim3::new(*dim_kernel),
                *stride,
                convert_padding(padding),
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::Activation {
            dim_input,
            method,
            save_key,
        } => {
            model.add_layer(LayerTypes::Activation(ActivationType::new(
                convert_activation(method),
                Dim3::new(*dim_input),
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::GroupNorm {
            dim_input,
            num_groups,
            save_key,
        } => {
            model.add_layer(LayerTypes::GroupNorm(GroupNormType::new(
                Dim3::new(*dim_input),
                *num_groups,
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::Attention {
            dim_input,
            save_key,
        } => {
            model.add_layer(LayerTypes::Attention(AttentionType::new(Dim3::new(
                *dim_input,
            ))))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::FullyConnected {
            dim_input,
            nb_neurons,
            method,
            save_key,
            ..
        } => {
            model.add_layer(LayerTypes::FullyConnected(FullyConnectedType::new(
                Dim3::new(*dim_input),
                *nb_neurons,
                convert_activation(method),
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::UpsampleConv {
            dim_input,
            scale_factor,
            nb_kernel,
            dim_kernel,
            padding,
            save_key,
            ..
        } => {
            model.add_layer(LayerTypes::UpsampleConv(UpsampleConvType::new(
                Dim3::new(*dim_input),
                *scale_factor,
                *nb_kernel,
                Dim3::new(*dim_kernel),
                convert_padding(padding),
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::Concat {
            skip_key, save_key, ..
        } => {
            model.add_concat(skip_key.clone())?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
    }
}

fn convert_padding(p: &PaddingMode) -> PPadding {
    match p {
        PaddingMode::Valid => PPadding::Valid,
        PaddingMode::Same => PPadding::Same,
    }
}

fn convert_activation(a: &ActivationMethod) -> PActivation {
    match a {
        ActivationMethod::Relu => PActivation::Relu,
        ActivationMethod::Silu => PActivation::Silu,
        ActivationMethod::Linear => PActivation::Linear,
    }
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

    // -- The img2img seed ----------------------------------------------------

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

        let images = SeedImages::resolve(Some(&out.to_string_lossy()), (2, 2, 1))
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
        let images = SeedImages::resolve(Some(&out.to_string_lossy()), (2, 2, 1))
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
            SeedImages::resolve(Some(&missing.to_string_lossy()), (2, 2, 1)).is_err(),
            "a named dataset that is not there must be reported, not swallowed"
        );
        // A geometry no built-in dataset matches: nothing to derive, no error.
        assert!(
            matches!(SeedImages::resolve(None, (2, 2, 7)), Ok(None)),
            "a 7-channel model has no default dataset and must not fail for it"
        );
    }

    /// The default follows the model's own output channels. Handing a
    /// greyscale drift the RGB file does not error — the loader resizes — so a
    /// wrong default would show up as a drift away from a mangled picture and
    /// nothing else.
    #[test]
    fn the_default_dataset_follows_the_models_output_channels() {
        assert_eq!(default_seed_dataset_name(1), Some("cifar10_grey.batraw"));
        assert_eq!(default_seed_dataset_name(3), Some("cifar10_rgb.batraw"));
        assert_eq!(default_seed_dataset_name(2), None);
        assert_eq!(default_seed_dataset_name(0), None);
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
        for (a, b) in dataset[0].target.iter().zip(sample.iter()) {
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
        for (a, b) in dataset[0].target.iter().zip(sample.iter()) {
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
        write_tensor_png(&dataset[0].target, (2, 2, 3), &png).expect("png should be written");
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
        for (got, want) in dataset[0].target.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "value mismatch: {got} vs {want}");
        }
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
