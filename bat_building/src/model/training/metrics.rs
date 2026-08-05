//! File purpose: Diffusion training/sampling instrumentation — per-timestep-bucket
//! loss, predicted-noise (ε̂) vs target-noise (ε) statistics, and per-step latent
//! trajectory stats, all written to a JSONL file next to the checkpoint.
//!
//! Everything here drives the model through its *public* API (`predict`) and the
//! shared [`LinearNoiseSchedule`], so the diagnostics never depend on GPU
//! internals and behave identically for a healthy and a degenerate model.

use crate::model::Model;
use crate::model::training::LinearNoiseSchedule;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Packs a `[signal | timestep-embedding]` model input, matching byte-for-byte
/// the layout the GPU `diffusion_prepare` shader produces during training.
///
/// This is the single source of truth for input composition: training's GPU
/// prepare pass, the diagnostic probe, and inference sampling all agree on it,
/// so the timestep conditioning the network is *trained* on is exactly the one
/// it is *sampled* with.
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

/// Core diffusion sampler with optional per-step trajectory capture.
///
/// This is the *single* sampler used by inference and by the diagnostics, so the
/// denoising math the metrics observe is exactly the one production runs use.
/// When `trajectory` is `Some`, the first path's per-step latent/ε̂ stats are
/// recorded — enough to see *where* along the 256-step chain the latent diverges.
///
/// `observer`, when `Some`, is called once per reverse step with the freshly
/// computed latent and x0 estimate; this is what the live inference visualiser
/// hangs off, so it watches the real chain instead of a re-implementation of it.
/// The x0 estimate is only computed when an observer is attached, so the
/// non-observed path allocates exactly as before.
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
    mut trajectory: Option<&mut Vec<DenoiseStepStat>>,
    mut observer: Option<&mut dyn FnMut(&DenoiseFrame)>,
    mut progress: F,
) -> Vec<f32>
where
    F: FnMut(usize, usize),
{
    let timestep_channels = input_channels.saturating_sub(signal_channels);
    let path_count = denoising_paths.max(1);
    let steps = schedule.len().max(1);
    let total_work = path_count.saturating_mul(steps);
    let mut accumulated = vec![0.0f32; output_len];
    let base_noise_seed = seed ^ 0xa5a5_5a5a_0123_4567;
    let base_latent = schedule.sample_noise(output_len, base_noise_seed);

    for path_idx in 0..path_count {
        let path_seed = seed ^ ((path_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut latent = base_latent.clone();

        for (step_idx, diffusion_step) in (0..schedule.len()).rev().enumerate() {
            let features = schedule.timestep_embedding(diffusion_step, timestep_channels);
            let model_input =
                compose_diffusion_input(&latent, input_channels, signal_channels, &features);
            let predicted_noise = model.predict(&model_input);
            let latent_in_stats = Stats::of(&latent);
            // Derived from x_t, so it must be read before the reverse step
            // overwrites `latent`. Skipped entirely when nobody is watching.
            let x0_hat = observer
                .is_some()
                .then(|| schedule.x0_estimate(&latent, &predicted_noise, diffusion_step));
            // The per-step seed mixes the timestep in by multiply-add, not by
            // XOR: `path_seed ^ diffusion_step` collided with the XOR the noise
            // field itself used to fold in the pixel index, so every step of
            // the chain re-drew one single field under an `index ^ step`
            // permutation. See `gaussian_at` in `schedule.rs`. The schedule-side
            // fix already breaks the collision; this keeps the caller from
            // relying on it.
            let step_seed = path_seed
                .wrapping_add((diffusion_step as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            latent = schedule.denoise_step_with_magnitude(
                &latent,
                &predicted_noise,
                diffusion_step,
                step_seed,
                denoise_magnitude,
            );
            if path_idx == 0 {
                if let Some(traj) = trajectory.as_deref_mut() {
                    traj.push(DenoiseStepStat {
                        step_index: step_idx,
                        diffusion_step,
                        latent_in: latent_in_stats,
                        eps_hat: Stats::of(&predicted_noise),
                        latent_out: Stats::of(&latent),
                    });
                }
            }
            if let (Some(observe), Some(x0_hat)) = (observer.as_deref_mut(), x0_hat.as_deref()) {
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

    #[test]
    fn stats_flags_non_finite() {
        let s = Stats::of(&[1.0, f32::NAN, f32::INFINITY, 3.0]);
        assert_eq!(s.count, 2);
        assert_eq!(s.non_finite, 2);
    }
}
