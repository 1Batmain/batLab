//! Diffusion training/sampling instrumentation — per-bucket loss, ε̂-vs-ε stats,
//! per-step trajectory — written to a JSONL beside the checkpoint. Drives the
//! model through its PUBLIC API (`predict`) and the shared schedule, so the
//! diagnostics never depend on GPU internals.

use crate::model::Model;
use crate::model::training::LinearNoiseSchedule;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

// Commit-1 bridge: the inference path moved to `sampler.rs`; re-exporting it here
// keeps every existing `metrics::…` path (and the `mod.rs`/`lib.rs` re-exports
// that lean on it) resolving unchanged, so this move touches no caller. Commit 2
// repoints the callers and removes this line.
pub use super::sampler::*;

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
