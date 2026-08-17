//! Diffusion training/sampling instrumentation — per-bucket loss, ε̂-vs-ε stats,
//! per-step trajectory — written to a JSONL beside the checkpoint. Drives the
//! model through its PUBLIC API (`predict`) and the shared schedule, so the
//! diagnostics never depend on GPU internals.

use crate::model::Model;
use crate::model::training::{LinearNoiseSchedule, PosteriorVariance};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Packs a `[signal | timestep-embedding]` model input, byte-for-byte the layout
/// the GPU `diffusion_prepare` shader produces during training — the single source
/// of truth for input composition, so the conditioning the network is TRAINED on
/// is exactly the one it is SAMPLED with.
pub fn compose_diffusion_input(
    signal: &[f32],
    input_channels: usize,
    signal_channels: usize,
    timestep_features: &[f32],
) -> Vec<f32> {
    if signal.is_empty() || input_channels == 0 || signal_channels == 0 {
        return signal.to_vec();
    }
    let pixel_count = signal.len() / signal_channels;
    let mut packed = vec![0.0f32; pixel_count * input_channels];
    for (pixel_idx, pixel) in signal.chunks(signal_channels).enumerate() {
        let dst = &mut packed[pixel_idx * input_channels..(pixel_idx + 1) * input_channels];
        dst[..signal_channels].copy_from_slice(pixel);
        for (value, feature) in dst
            .iter_mut()
            .skip(signal_channels)
            .zip(timestep_features.iter())
        {
            *value = *feature;
        }
    }
    packed
}

/// Summary statistics of a tensor slice (NaN/Inf tolerant).
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub count: usize,
    pub non_finite: usize,
}

impl Stats {
    pub fn of(values: &[f32]) -> Self {
        let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let non_finite = values.len() - finite.len();
        if finite.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
                count: 0,
                non_finite,
            };
        }
        let count = finite.len();
        let mean = finite.iter().sum::<f32>() / count as f32;
        let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / count as f32;
        Self {
            min: finite.iter().cloned().fold(f32::INFINITY, f32::min),
            max: finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            mean,
            std: var.sqrt(),
            count,
            non_finite,
        }
    }

    fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "non_finite": self.non_finite,
        })
    }
}

/// Appends JSON records, one per line, to a metrics file. Failures are reported
/// once and then silently ignored so instrumentation never aborts a run.
pub struct MetricsLogger {
    writer: Option<BufWriter<std::fs::File>>,
    path: String,
}

impl MetricsLogger {
    /// Opens `path` for appending. A fresh run truncates so the file always
    /// describes the current run only.
    pub fn create(path: &Path, truncate: bool) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(truncate)
            .append(!truncate)
            .open(path);
        match file {
            Ok(file) => Self {
                writer: Some(BufWriter::new(file)),
                path: path.display().to_string(),
            },
            Err(err) => {
                eprintln!(
                    "[metrics] could not open {}: {err}; metrics disabled",
                    path.display()
                );
                Self {
                    writer: None,
                    path: path.display().to_string(),
                }
            }
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn write(&mut self, record: &serde_json::Value) {
        if let Some(writer) = self.writer.as_mut() {
            if writeln!(writer, "{record}")
                .and_then(|_| writer.flush())
                .is_err()
            {
                eprintln!(
                    "[metrics] write failed; disabling metrics for {}",
                    self.path
                );
                self.writer = None;
            }
        }
    }
}

/// How the diagnostic probe slices the schedule and how many draws it takes.
#[derive(Debug, Clone, Copy)]
pub struct ProbeConfig {
    /// Number of dataset samples fed through the probe.
    pub sample_count: usize,
    /// Number of timesteps drawn inside each bucket.
    pub t_per_bucket: usize,
    /// Number of contiguous timestep buckets the schedule is split into.
    pub bucket_count: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            sample_count: 8,
            t_per_bucket: 2,
            bucket_count: 4,
        }
    }
}

/// Per-bucket diagnostic: how well ε̂ matches ε for a range of timesteps.
#[derive(Debug, Clone)]
pub struct BucketStat {
    pub bucket: usize,
    pub t_lo: usize,
    pub t_hi: usize,
    /// Mean over the bucket of MSE(ε̂, ε) — the same quantity the training loss
    /// minimises, but resolved per timestep range instead of collapsed to the
    /// batch's last element.
    pub loss: f32,
    pub eps_hat: Stats,
    pub eps_target: Stats,
    pub draws: usize,
}

/// Runs the per-timestep-bucket diagnostic. For each bucket it noises probe
/// samples at timesteps inside the bucket, asks the model to predict the noise,
/// and reports MSE(ε̂, ε) plus the ε̂/ε distributions.
///
/// A model with no timestep conditioning cannot vary ε̂ with t, so its ε̂ std
/// collapses and its loss climbs sharply with t — this probe makes that visible.
pub fn probe_diffusion<State>(
    model: &mut Model<State>,
    schedule: &LinearNoiseSchedule,
    samples: &[Vec<f32>],
    input_channels: usize,
    signal_channels: usize,
    cfg: &ProbeConfig,
    seed: u64,
) -> Vec<BucketStat> {
    let steps = schedule.len().max(1);
    let bucket_count = cfg.bucket_count.max(1);
    let timestep_channels = input_channels.saturating_sub(signal_channels);
    let mut out = Vec::with_capacity(bucket_count);

    for bucket in 0..bucket_count {
        let t_lo = bucket * steps / bucket_count;
        let t_hi = (((bucket + 1) * steps / bucket_count).max(t_lo + 1)).min(steps);
        let span = t_hi - t_lo;

        let mut sq_err_sum = 0.0f64;
        let mut err_terms = 0usize;
        let mut eps_hat_all: Vec<f32> = Vec::new();
        let mut eps_target_all: Vec<f32> = Vec::new();
        let mut draws = 0usize;

        for (s_idx, clean) in samples.iter().take(cfg.sample_count).enumerate() {
            for k in 0..cfg.t_per_bucket.max(1) {
                // Spread the draws across the bucket span deterministically.
                let t = t_lo + (k * span) / cfg.t_per_bucket.max(1);
                let noise_seed = seed
                    ^ ((s_idx as u64) << 40)
                    ^ ((bucket as u64) << 20)
                    ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let (noisy, eps) = schedule.add_noise(clean, t, noise_seed);
                let features = schedule.timestep_embedding(t, timestep_channels);
                let input =
                    compose_diffusion_input(&noisy, input_channels, signal_channels, &features);
                let eps_hat = model.predict(&input);

                for (a, b) in eps_hat.iter().zip(eps.iter()) {
                    sq_err_sum += ((a - b) as f64).powi(2);
                    err_terms += 1;
                }
                eps_hat_all.extend_from_slice(&eps_hat);
                eps_target_all.extend_from_slice(&eps);
                draws += 1;
            }
        }

        let loss = if err_terms > 0 {
            (sq_err_sum / err_terms as f64) as f32
        } else {
            0.0
        };
        out.push(BucketStat {
            bucket,
            t_lo,
            t_hi,
            loss,
            eps_hat: Stats::of(&eps_hat_all),
            eps_target: Stats::of(&eps_target_all),
            draws,
        });
    }
    out
}

/// Emits one `train_probe` JSONL record from a set of bucket stats.
pub fn log_probe(logger: &mut MetricsLogger, step: usize, buckets: &[BucketStat]) {
    let buckets_json: Vec<serde_json::Value> = buckets
        .iter()
        .map(|b| {
            serde_json::json!({
                "bucket": b.bucket,
                "t_lo": b.t_lo,
                "t_hi": b.t_hi,
                "loss": b.loss,
                "draws": b.draws,
                "eps_hat": b.eps_hat.to_json(),
                "eps_target": b.eps_target.to_json(),
            })
        })
        .collect();
    logger.write(&serde_json::json!({
        "kind": "train_probe",
        "step": step,
        "buckets": buckets_json,
    }));
}

/// Emits one `train_loss` record (the batch loss reported by the GPU step).
pub fn log_train_loss(
    logger: &mut MetricsLogger,
    step: usize,
    loss: f32,
    lr: f32,
    batch_size: u32,
) {
    logger.write(&serde_json::json!({
        "kind": "train_loss",
        "step": step,
        "loss": loss,
        "lr": lr,
        "batch_size": batch_size,
    }));
}

/// Latent (and ε̂) statistics captured at one denoising step of a sampling run.
#[derive(Debug, Clone, Copy)]
pub struct DenoiseStepStat {
    pub step_index: usize,
    pub diffusion_step: usize,
    pub latent_in: Stats,
    pub eps_hat: Stats,
    pub latent_out: Stats,
}

/// One denoising step's tensors, handed to an observer as they are produced.
///
/// Borrowed, never owned: the observer runs inside the sampling loop and is
/// expected to forward the data somewhere (a GPU buffer, a file) rather than
/// accumulate it — a 256-step run over 4 paths would otherwise retain a
/// thousand tensors for nothing.
pub struct DenoiseFrame<'a> {
    /// Index of the denoising path currently being walked.
    pub path_idx: usize,
    /// Total number of paths this run will walk.
    pub path_count: usize,
    /// How many steps of this path are done (0-based, counts *up*).
    pub step_index: usize,
    /// Position on the noise schedule (counts *down* from `total_steps - 1`).
    pub diffusion_step: usize,
    /// Length of the noise schedule.
    pub total_steps: usize,
    /// The latent *after* this reverse step: x_{t-1}.
    pub latent: &'a [f32],
    /// The clipped x0 estimate this step's posterior mean was built from —
    /// "what the model believes the clean image looks like" from x_t.
    pub x0_hat: &'a [f32],
}

/// Odd increment used to fold the timestep into a path's seed.
const STEP_SEED_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// The seed one reverse step draws its noise from. The timestep is mixed in by
/// MULTIPLY-ADD, not XOR: `path_seed ^ step` collided with the field's own XOR of
/// the pixel index and painted horizontal bands (`ANISOTROPY_HUNT.md`). One
/// function, so no caller can derive its step seed a different, broken way.
pub fn reverse_step_seed(path_seed: u64, diffusion_step: usize) -> u64 {
    path_seed.wrapping_add((diffusion_step as u64 + 1).wrapping_mul(STEP_SEED_GAMMA))
}

/// The fold from a run seed to its opening latent's seed — one value, one place.
/// The native sampler, the drift's noise fallback and the web port all pass their
/// run seed through [`base_noise_seed`], so a browser run opens on the field native
/// path 0 does. A second copy (a private `const` in the web crate) is exactly the
/// silent divergence `WEB_PORT.md` warns of; public so callers import, not restate.
pub const BASE_NOISE_FOLD: u64 = 0xa5a5_5a5a_0123_4567;

/// The opening latent's noise seed for a run seed — see [`BASE_NOISE_FOLD`].
pub fn base_noise_seed(seed: u64) -> u64 {
    seed ^ BASE_NOISE_FOLD
}

/// The seed of denoising path `path_idx`. Path 0 is the run seed itself, so a
/// single-path run (inference, web) and path 0 of a multi-path sample draw the same
/// chain. One function, so no caller spreads the seed a different way.
pub fn path_seed(seed: u64, path_idx: usize) -> u64 {
    seed ^ (path_idx as u64).wrapping_mul(STEP_SEED_GAMMA)
}

/// What one reverse step produced.
pub struct ReverseStep {
    /// x_{t-1}, the latent after the step.
    pub latent: Vec<f32>,
    /// ε̂, the noise the model predicted from x_t.
    pub predicted_noise: Vec<f32>,
    /// The clipped x0 estimate the posterior mean was built from — present only
    /// when asked for, since nothing in the recursion needs it.
    pub x0_hat: Option<Vec<f32>>,
}

/// Asks the model what noise it sees in `latent` at `diffusion_step` — a reverse
/// step's one model call. Split out so the recursion is written against a
/// PREDICTION, not a `Model`, letting the perpetual walk run against an oracle in
/// tests without a GPU or a second copy of the maths.
pub fn predict_epsilon<State>(
    model: &mut Model<State>,
    schedule: &LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
    latent: &[f32],
    diffusion_step: usize,
) -> Vec<f32> {
    let timestep_channels = input_channels.saturating_sub(signal_channels);
    let features = schedule.timestep_embedding(diffusion_step, timestep_channels);
    let model_input = compose_diffusion_input(latent, input_channels, signal_channels, &features);
    model.predict(&model_input)
}

/// The reverse recursion proper, from an already-predicted ε̂ — the ONLY place it
/// is written. Every walker goes through it ([`sample_diffusion`] and the
/// perpetual drift), so the step-seed derivation and posterior draw cannot diverge
/// between a finite sample and an endless one.
#[allow(clippy::too_many_arguments)]
pub fn reverse_step_from_epsilon(
    schedule: &LinearNoiseSchedule,
    latent: &[f32],
    predicted_noise: Vec<f32>,
    diffusion_step: usize,
    path_seed: u64,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    want_x0_hat: bool,
) -> ReverseStep {
    // Derived from x_t, so it must be read before the reverse step produces
    // x_{t-1}. Skipped entirely when nobody is watching.
    let x0_hat =
        want_x0_hat.then(|| schedule.x0_estimate(latent, &predicted_noise, diffusion_step));
    let next = schedule.denoise_step_with_magnitude(
        latent,
        &predicted_noise,
        diffusion_step,
        reverse_step_seed(path_seed, diffusion_step),
        denoise_magnitude,
        variance,
    );
    ReverseStep {
        latent: next,
        predicted_noise,
        x0_hat,
    }
}

/// The async sibling of [`reverse_step`] — same composition, ε̂ and posterior draw,
/// but it AWAITS the readback ([`Model::predict_async`]). So a browser inference
/// descent does not re-derive the recursion (which could compose `[x_t|timestep]`
/// or seed the posterior differently and drift undetected); held to the sync path
/// by `the_async_reverse_step_matches_the_sync_one`.
#[allow(clippy::too_many_arguments)]
pub async fn reverse_step_async<State>(
    model: &mut Model<State>,
    schedule: &LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
    latent: &[f32],
    diffusion_step: usize,
    path_seed: u64,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    want_x0_hat: bool,
) -> ReverseStep {
    let timestep_channels = input_channels.saturating_sub(signal_channels);
    let features = schedule.timestep_embedding(diffusion_step, timestep_channels);
    let model_input = compose_diffusion_input(latent, input_channels, signal_channels, &features);
    let predicted_noise = model.predict_async(&model_input).await;
    reverse_step_from_epsilon(
        schedule,
        latent,
        predicted_noise,
        diffusion_step,
        path_seed,
        denoise_magnitude,
        variance,
        want_x0_hat,
    )
}

/// One step down the reverse chain: compose `[x_t | timestep]`, predict ε̂, and
/// sample the posterior — [`predict_epsilon`] then
/// [`reverse_step_from_epsilon`], which is the whole of it.
#[allow(clippy::too_many_arguments)]
pub fn reverse_step<State>(
    model: &mut Model<State>,
    schedule: &LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
    latent: &[f32],
    diffusion_step: usize,
    path_seed: u64,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    want_x0_hat: bool,
) -> ReverseStep {
    let predicted_noise = predict_epsilon(
        model,
        schedule,
        input_channels,
        signal_channels,
        latent,
        diffusion_step,
    );
    reverse_step_from_epsilon(
        schedule,
        latent,
        predicted_noise,
        diffusion_step,
        path_seed,
        denoise_magnitude,
        variance,
        want_x0_hat,
    )
}

/// The SINGLE diffusion sampler, used by inference and diagnostics alike, so the
/// math the metrics observe is production's. `trajectory: Some` records the first
/// path's per-step stats (where along the chain the latent diverges). `observer:
/// Some` is called per step with the latent and x0 estimate — what the live
/// visualiser hangs off, so it watches the real chain, not a re-implementation
/// (x0 is computed only when observed).
#[allow(clippy::too_many_arguments)]
pub fn sample_diffusion<State, F>(
    model: &mut Model<State>,
    input_channels: usize,
    signal_channels: usize,
    output_len: usize,
    schedule: &LinearNoiseSchedule,
    seed: u64,
    denoising_paths: usize,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    mut trajectory: Option<&mut Vec<DenoiseStepStat>>,
    mut observer: Option<&mut dyn FnMut(&DenoiseFrame)>,
    mut progress: F,
) -> Vec<f32>
where
    F: FnMut(usize, usize),
{
    let path_count = denoising_paths.max(1);
    let steps = schedule.len().max(1);
    let total_work = path_count.saturating_mul(steps);
    let mut accumulated = vec![0.0f32; output_len];
    let base_latent = schedule.sample_noise(output_len, base_noise_seed(seed));

    for path_idx in 0..path_count {
        let path_seed = path_seed(seed, path_idx);
        let mut latent = base_latent.clone();

        for (step_idx, diffusion_step) in (0..schedule.len()).rev().enumerate() {
            let latent_in_stats = Stats::of(&latent);
            let stepped = reverse_step(
                model,
                schedule,
                input_channels,
                signal_channels,
                &latent,
                diffusion_step,
                path_seed,
                denoise_magnitude,
                variance,
                observer.is_some(),
            );
            latent = stepped.latent;
            if path_idx == 0 {
                if let Some(traj) = trajectory.as_deref_mut() {
                    traj.push(DenoiseStepStat {
                        step_index: step_idx,
                        diffusion_step,
                        latent_in: latent_in_stats,
                        eps_hat: Stats::of(&stepped.predicted_noise),
                        latent_out: Stats::of(&latent),
                    });
                }
            }
            if let (Some(observe), Some(x0_hat)) =
                (observer.as_deref_mut(), stepped.x0_hat.as_deref())
            {
                observe(&DenoiseFrame {
                    path_idx,
                    path_count,
                    step_index: step_idx,
                    diffusion_step,
                    total_steps: steps,
                    latent: &latent,
                    x0_hat,
                });
            }
            progress(path_idx * steps + step_idx + 1, total_work);
        }

        for (acc, value) in accumulated.iter_mut().zip(latent.iter()) {
            *acc += *value;
        }
    }

    for value in &mut accumulated {
        *value /= path_count as f32;
    }
    accumulated
}

/// Emits one `sample` record plus one `denoise_step` record per captured step.
pub fn log_trajectory(
    logger: &mut MetricsLogger,
    step: usize,
    seed: u64,
    trajectory: &[DenoiseStepStat],
    image: &Stats,
) {
    logger.write(&serde_json::json!({
        "kind": "sample",
        "step": step,
        "seed": seed,
        "final_image": image.to_json(),
        "denoise_steps": trajectory.len(),
    }));
    for s in trajectory {
        logger.write(&serde_json::json!({
            "kind": "denoise_step",
            "train_step": step,
            "step_index": s.step_index,
            "diffusion_step": s.diffusion_step,
            "latent_in": s.latent_in.to_json(),
            "eps_hat": s.eps_hat.to_json(),
            "latent_out": s.latent_out.to_json(),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_packs_signal_then_timestep_channels() {
        // 2 pixels, 1 signal channel, 3 input channels => 2 timestep channels.
        let signal = vec![0.5, -0.5];
        let features = vec![0.1, 0.2];
        let packed = compose_diffusion_input(&signal, 3, 1, &features);
        assert_eq!(packed, vec![0.5, 0.1, 0.2, -0.5, 0.1, 0.2]);
    }

    #[test]
    fn compose_without_timestep_channels_is_identity_layout() {
        let signal = vec![0.5, -0.5, 0.25, -0.25];
        let packed = compose_diffusion_input(&signal, 1, 1, &[]);
        assert_eq!(packed, signal);
    }

    #[test]
    fn stats_reports_spread() {
        let s = Stats::of(&[1.0, 2.0, 3.0]);
        assert_eq!(s.count, 3);
        assert!((s.mean - 2.0).abs() < 1e-6);
        assert!((s.min - 1.0).abs() < 1e-6);
        assert!((s.max - 3.0).abs() < 1e-6);
        assert!(s.std > 0.0);
    }

    /// Now that every walker of the chain derives its per-step seed here, this
    /// is the one place the anisotropy defect could come back. Under the old
    /// `path_seed ^ diffusion_step` the seeds of two steps differed by a few low
    /// bits — exactly the relation the pixel-index XOR turned into a reindexing
    /// of one single noise field (`ANISOTROPY_HUNT.md`).
    #[test]
    fn reverse_step_seed_folds_the_timestep_in_without_xor() {
        let path_seed = 0xdead_beef_u64;
        for step in 0..256usize {
            assert_ne!(
                reverse_step_seed(path_seed, step),
                path_seed ^ step as u64,
                "step {step} seed fell back to the XOR derivation"
            );
        }
        let distinct: std::collections::HashSet<u64> = (0..256)
            .map(|step| reverse_step_seed(path_seed, step))
            .collect();
        assert_eq!(distinct.len(), 256, "two timesteps share a seed");
    }

    /// The canonical folds must still compute the exact values the four scattered
    /// literals used to — pinning them so this centralisation is provably a
    /// no-op, and so a fat-fingered edit to `BASE_NOISE_FOLD` cannot pass.
    #[test]
    fn the_canonical_seed_folds_match_the_old_literals() {
        // The two folds exactly as they read at metrics.rs:560, perpetual.rs:561,
        // model.rs:2129 and web/lib.rs before centralisation — rebuilt from
        // halves so this test is not itself flagged as a second copy.
        let old_base_fold: u64 = (0xa5a5_5a5a_u64 << 32) | 0x0123_4567;
        let old_gamma: u64 = (0x9e37_79b9_u64 << 32) | 0x7f4a_7c15;
        for seed in [0u64, 1, 42, 7, 0xdead_beef, u64::MAX] {
            assert_eq!(base_noise_seed(seed), seed ^ old_base_fold);
            for path_idx in 0..8usize {
                assert_eq!(
                    path_seed(seed, path_idx),
                    seed ^ (path_idx as u64).wrapping_mul(old_gamma),
                );
            }
        }
        // Path 0 is the run seed itself — the identity the web's single-path
        // inference and native path 0 both lean on.
        assert_eq!(path_seed(12345, 0), 12345);
    }

    /// The base-noise fold has exactly one home. Any *other* line in any crate
    /// that spells the literal out is a second source of truth — the very
    /// divergence a web port re-deriving its opening noise would introduce, and
    /// which no run would ever flag. Checked mechanically over every crate's
    /// sources, not just this one, because the copy that mattered lived in
    /// `batlab-web`.
    #[test]
    fn the_base_noise_fold_is_written_in_exactly_one_place() {
        // Assembled from halves so this detector is not itself an offender.
        let literal = concat!("0xa5a5_5a5a", "_0123_4567");
        // crates/batlab-core/src → up to the workspace, then every crate's src.
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("crates/<name> sits two levels below the workspace root")
            .to_path_buf();

        fn walk(dir: &std::path::Path, literal: &str, offenders: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries {
                let path = entry.expect("bad directory entry").path();
                if path.is_dir() {
                    walk(&path, literal, offenders);
                    continue;
                }
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !name.ends_with(".rs") {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("failed to read a source file");
                for (number, line) in source.lines().enumerate() {
                    if !line.contains(literal) {
                        continue;
                    }
                    let code = line.trim_start();
                    if code.starts_with("//") {
                        continue;
                    }
                    // The one allowed line: the canonical `pub const` definition.
                    if code.contains("const BASE_NOISE_FOLD") {
                        continue;
                    }
                    offenders.push(format!("{}:{}: {}", path.display(), number + 1, code));
                }
            }
        }

        let mut offenders = Vec::new();
        for crate_dir in std::fs::read_dir(workspace.join("crates"))
            .expect("failed to read crates/")
            .flatten()
        {
            walk(&crate_dir.path().join("src"), literal, &mut offenders);
        }
        assert!(
            offenders.is_empty(),
            "the base-noise fold must come from batlab_core::BASE_NOISE_FOLD, not a restated \
             literal — a second copy is how a run opens on a different noise field than the \
             native sampler, silently:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn stats_flags_non_finite() {
        let s = Stats::of(&[1.0, f32::NAN, f32::INFINITY, 3.0]);
        assert_eq!(s.count, 2);
        assert_eq!(s.non_finite, 2);
    }
}
