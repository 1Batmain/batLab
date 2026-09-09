//! File purpose: deterministic held-out evaluation of a diffusion model — the
//! MSE of ε̂ against ε, resolved per timestep bucket and *also* carried into
//! image (x₀) space, measured against the three zero-parameter baselines of
//! `tools/trivial_baselines.py`.
//!
//! The whole point of the tool is comparability: two checkpoints must be scored
//! on **identical work** — the same held-out images, the same timesteps, the
//! same noise fields — so a difference in their numbers is a difference in the
//! weights and nothing else. Every draw here is a pure function of
//! `(seed, sample_index, bucket, t)`, exactly as [`super::probe_diffusion`]
//! already derives its own, so a rerun reproduces a run bit for bit.
//!
//! Why x₀ as well as ε. The network is trained on MSE(ε̂, ε), but that number
//! *inverts the importance of the schedule*: at high `t` a copy of the input
//! (ε̂ = x_t) is already almost exact and ε̂ = 0 costs ~1, so a small ε-MSE up
//! there proves nothing. So the tool also scores the image the sampler actually
//! builds: the **clipped** reconstruction `x̂₀ = clip((x_t − √(1−ᾱ)·ε̂)/√ᾱ, −1,
//! 1)` — byte for byte [`super::LinearNoiseSchedule::x0_estimate`], the exact
//! quantity every reverse step is built from. The clip is not cosmetic: without
//! it the `1/√ᾱ` chain gain turns a small high-`t` ε error into a meaningless
//! image-space blow-up, and the "ε̂ = mean" baseline (which reconstructs exactly
//! the in-range mean image) would look artificially unbeatable. With the clip,
//! `x₀-MSE` is a bounded reconstruction error in `[-1, 1]` units and the
//! baselines are a fair bar: a model that does not beat "ε̂ = mean" in x₀ has
//! reconstructed nothing the mean image did not already give. Reported alongside
//! is the *content signal* ε carries, `√(ᾱ/(1−ᾱ)·E[x₀²])` (`trivial_baselines.py`).
//!
//! Where the clip does not bite (low `t`, a decent model), `x̂₀ − x₀ =
//! −√(1−ᾱ)/√ᾱ·(ε̂ − ε)` still holds, so the two views agree on the reweighting
//! (`x0_matches_the_factor_where_it_does_not_clip`); the clip only changes the
//! high-`t` buckets, which is exactly where it must.
//!
//! GPU-free by construction: the model enters as a `predict` closure, so the
//! schedule maths and the determinism are exercised in tests against oracle
//! predictors on a machine with no GPU, and the CLI hands in `model.predict`.

use crate::model::training::sampler::compose_diffusion_input;
use crate::model::training::schedule::LinearNoiseSchedule;

/// Odd multiplier folding the timestep into a draw's seed — the same constant
/// [`super::probe_diffusion`] uses, so the two diagnostics draw alike.
const DRAW_SEED_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// How the held-out evaluation slices the schedule and how many draws it takes.
#[derive(Debug, Clone, Copy)]
pub struct EvalConfig {
    /// Number of contiguous timestep buckets the schedule is split into.
    pub buckets: usize,
    /// Number of timesteps drawn inside each bucket.
    pub t_per_bucket: usize,
    /// Base seed. The three baselines and the model all draw from it, so the
    /// noise a checkpoint is judged on is the noise the baselines are judged on.
    pub seed: u64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            buckets: 4,
            t_per_bucket: 4,
            seed: 7,
        }
    }
}

/// Per-bucket error of a predictor, in both ε and x₀ space.
///
/// Sums are kept raw (not yet divided) so an aggregate over the whole schedule
/// is the sum of the parts, not an average of averages.
#[derive(Debug, Clone, Copy, Default)]
pub struct MseBucket {
    pub bucket: usize,
    pub t_lo: usize,
    pub t_hi: usize,
    pub draws: usize,
    pub terms: usize,
    eps_sq_sum: f64,
    x0_sq_sum: f64,
}

impl MseBucket {
    /// Mean squared error of ε̂ against ε over this bucket.
    pub fn eps_mse(&self) -> f64 {
        if self.terms == 0 {
            0.0
        } else {
            self.eps_sq_sum / self.terms as f64
        }
    }

    /// Mean squared error of the clipped x̂₀ reconstruction against the clean
    /// image — bounded, in `[-1, 1]` units.
    pub fn x0_mse(&self) -> f64 {
        if self.terms == 0 {
            0.0
        } else {
            self.x0_sq_sum / self.terms as f64
        }
    }
}

/// The three trivial predictors (`ε̂ = 0`, `ε̂ = x_t`, `ε̂ = mean`) on the same
/// draws, plus the content signal ε carries — the reference a trained model has
/// to beat.
#[derive(Debug, Clone, Copy, Default)]
pub struct BaselineBucket {
    pub bucket: usize,
    pub t_lo: usize,
    pub t_hi: usize,
    pub terms: usize,
    eps_zero_sum: f64,
    eps_copy_sum: f64,
    eps_mean_sum: f64,
    x0_zero_sum: f64,
    x0_copy_sum: f64,
    x0_mean_sum: f64,
    content_sq_sum: f64,
}

impl BaselineBucket {
    fn mean(sum: f64, terms: usize) -> f64 {
        if terms == 0 {
            0.0
        } else {
            sum / terms as f64
        }
    }
    pub fn eps_zero(&self) -> f64 {
        Self::mean(self.eps_zero_sum, self.terms)
    }
    pub fn eps_copy(&self) -> f64 {
        Self::mean(self.eps_copy_sum, self.terms)
    }
    pub fn eps_mean(&self) -> f64 {
        Self::mean(self.eps_mean_sum, self.terms)
    }
    pub fn x0_zero(&self) -> f64 {
        Self::mean(self.x0_zero_sum, self.terms)
    }
    pub fn x0_copy(&self) -> f64 {
        Self::mean(self.x0_copy_sum, self.terms)
    }
    pub fn x0_mean(&self) -> f64 {
        Self::mean(self.x0_mean_sum, self.terms)
    }
    /// RMS of the content √ᾱ·x₀/√(1−ᾱ) that lives inside ε at these timesteps —
    /// the amplitude a model's ε error has to fall under to say anything about
    /// the image.
    pub fn content_rms(&self) -> f64 {
        Self::mean(self.content_sq_sum, self.terms).sqrt()
    }
}

/// The evaluation of one checkpoint: the model's per-bucket error and, computed
/// on the very same draws, the baselines it is measured against.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub model: Vec<MseBucket>,
    pub baselines: Vec<BaselineBucket>,
}

impl EvalReport {
    /// ε-MSE over every bucket at once (element-weighted, i.e. the raw training
    /// objective on the held-out set).
    pub fn total_eps_mse(&self) -> f64 {
        let (sum, terms) = self
            .model
            .iter()
            .fold((0.0, 0usize), |(s, n), b| (s + b.eps_sq_sum, n + b.terms));
        if terms == 0 { 0.0 } else { sum / terms as f64 }
    }
}

/// The deterministic timestep of the `k`-th draw inside a bucket, spread across
/// the bucket span exactly as [`super::probe_diffusion`] spreads its own.
fn draw_timestep(t_lo: usize, span: usize, k: usize, t_per_bucket: usize) -> usize {
    t_lo + (k * span) / t_per_bucket.max(1)
}

/// The seed the `(sample, bucket, t)` draw noises with — a pure function of the
/// three, so the field is reproduced identically on every rerun and shared
/// between the model and the baselines.
fn draw_seed(seed: u64, sample_idx: usize, bucket: usize, t: usize) -> u64 {
    seed ^ ((sample_idx as u64) << 40)
        ^ ((bucket as u64) << 20)
        ^ (t as u64).wrapping_mul(DRAW_SEED_GAMMA)
}

/// Run the held-out evaluation. `predict` is handed the composed
/// `[x_t | timestep]` input and must return ε̂ over the signal channels; the CLI
/// passes `|input| model.predict(input)`, tests pass an oracle.
///
/// `mean_image` is the dataset's average clean image (signal channels only),
/// used by the `ε̂ = mean` baseline — "the model that learned only the mean".
pub fn evaluate(
    schedule: &LinearNoiseSchedule,
    samples: &[Vec<f32>],
    mean_image: &[f32],
    input_channels: usize,
    signal_channels: usize,
    cfg: &EvalConfig,
    mut predict: impl FnMut(&[f32]) -> Vec<f32>,
) -> EvalReport {
    let steps = schedule.len().max(1);
    let bucket_count = cfg.buckets.max(1);
    let t_per_bucket = cfg.t_per_bucket.max(1);
    let timestep_channels = input_channels.saturating_sub(signal_channels);

    let mut model = Vec::with_capacity(bucket_count);
    let mut baselines = Vec::with_capacity(bucket_count);

    for bucket in 0..bucket_count {
        let t_lo = bucket * steps / bucket_count;
        let t_hi = (((bucket + 1) * steps / bucket_count).max(t_lo + 1)).min(steps);
        let span = t_hi - t_lo;

        let mut m = MseBucket {
            bucket,
            t_lo,
            t_hi,
            ..Default::default()
        };
        let mut b = BaselineBucket {
            bucket,
            t_lo,
            t_hi,
            ..Default::default()
        };

        for (sample_idx, clean) in samples.iter().enumerate() {
            for k in 0..t_per_bucket {
                let t = draw_timestep(t_lo, span, k, t_per_bucket);
                let seed = draw_seed(cfg.seed, sample_idx, bucket, t);
                let (noisy, eps) = schedule.add_noise(clean, t, seed);

                // Guarded: the top of a real schedule keeps ᾱ strictly above 0
                // (it approaches pure noise, never reaches it), and t=0 keeps it
                // below 1, so neither ratio divides by zero. The clamp is a
                // belt-and-braces against a degenerate one-step schedule.
                let alpha_bar = (schedule.alpha_bar(t) as f64).clamp(1e-12, 1.0 - 1e-12);
                let signal = alpha_bar.sqrt();
                let noise = (1.0 - alpha_bar).sqrt();
                // Content √ᾱ·x₀/√(1−ᾱ) present in ε, per element (squared).
                let content_factor = alpha_bar / (1.0 - alpha_bar);
                // The clipped reconstruction the sampler builds from an ε̂ —
                // exactly `x0_hat_at` in `schedule.rs`.
                let x0_hat = |eps_pred: f64, x_t: f64| {
                    ((x_t - noise * eps_pred) / signal).clamp(-1.0, 1.0)
                };

                let features = schedule.timestep_embedding(t, timestep_channels);
                let input =
                    compose_diffusion_input(&noisy, input_channels, signal_channels, &features);
                let eps_hat = predict(&input);

                m.draws += 1;
                for (i, (&e_hat, &e)) in eps_hat.iter().zip(eps.iter()).enumerate() {
                    let e_hat = e_hat as f64;
                    let e = e as f64;
                    let x_t = noisy[i] as f64;
                    let x0 = clean[i] as f64;

                    let d_eps = e_hat - e;
                    m.eps_sq_sum += d_eps * d_eps;
                    let d_x0 = x0_hat(e_hat, x_t) - x0;
                    m.x0_sq_sum += d_x0 * d_x0;
                    m.terms += 1;

                    // The three trivial predictors, on this very draw.
                    // ε̂ = (x_t − √ᾱ·mean)/√(1−ᾱ): the "learned the mean" model;
                    // its clipped x̂₀ is exactly the (in-range) mean image.
                    let mean_px = mean_image[i % mean_image.len().max(1)] as f64;
                    let e_mean = (x_t - signal * mean_px) / noise;
                    let d_zero = 0.0 - e;
                    let d_copy = x_t - e;
                    let d_mean = e_mean - e;
                    b.eps_zero_sum += d_zero * d_zero;
                    b.eps_copy_sum += d_copy * d_copy;
                    b.eps_mean_sum += d_mean * d_mean;
                    let dz = x0_hat(0.0, x_t) - x0;
                    let dc = x0_hat(x_t, x_t) - x0;
                    let dm = x0_hat(e_mean, x_t) - x0;
                    b.x0_zero_sum += dz * dz;
                    b.x0_copy_sum += dc * dc;
                    b.x0_mean_sum += dm * dm;
                    b.content_sq_sum += content_factor * x0 * x0;
                    b.terms += 1;
                }
            }
        }

        model.push(m);
        baselines.push(b);
    }

    EvalReport { model, baselines }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule() -> LinearNoiseSchedule {
        LinearNoiseSchedule::new_linear(256, 1e-4, 2e-2)
    }

    // A couple of small "images": 2 pixels, 1 signal channel.
    fn samples() -> Vec<Vec<f32>> {
        vec![vec![0.5, -0.5], vec![0.25, 0.75], vec![-0.3, 0.1]]
    }

    /// The tool exists to be reproducible: the same work scored twice, with a
    /// deterministic predictor, must give byte-for-byte the same numbers — that
    /// is what lets two checkpoints be compared at all.
    #[test]
    fn the_same_work_scores_identically_on_a_rerun() {
        let sched = schedule();
        let s = samples();
        let mean = vec![0.0, 0.0];
        let cfg = EvalConfig::default();
        // A predictor that depends only on its input is deterministic.
        let run = || {
            evaluate(&sched, &s, &mean, 3, 1, &cfg, |input| {
                input.iter().step_by(3).map(|v| v * 0.1).collect()
            })
        };
        let a = run();
        let b = run();
        for (x, y) in a.model.iter().zip(b.model.iter()) {
            assert_eq!(x.eps_mse().to_bits(), y.eps_mse().to_bits());
            assert_eq!(x.x0_mse().to_bits(), y.x0_mse().to_bits());
        }
    }

    /// The model path and the baseline path must see the *same* draws, or the
    /// comparison is meaningless. A predictor that returns 0 must land exactly
    /// on the `ε̂ = 0` baseline, bucket by bucket.
    #[test]
    fn a_zero_predictor_lands_on_the_zero_baseline() {
        let sched = schedule();
        let s = samples();
        let mean = vec![0.0, 0.0];
        let report = evaluate(&sched, &s, &mean, 3, 1, &EvalConfig::default(), |input| {
            vec![0.0; input.len() / 3]
        });
        for (m, b) in report.model.iter().zip(report.baselines.iter()) {
            assert!(
                (m.eps_mse() - b.eps_zero()).abs() < 1e-12,
                "bucket {}: model {} vs zero baseline {}",
                m.bucket,
                m.eps_mse(),
                b.eps_zero()
            );
            assert!((m.x0_mse() - b.x0_zero()).abs() < 1e-12);
        }
    }

    /// A predictor that copies x_t (the first of every input's 3 channels) must
    /// land on the `ε̂ = x_t` baseline in BOTH ε and x₀ — this also pins that the
    /// compose layout puts the signal channel first, and that the model and
    /// baseline reconstructions run the identical clip.
    #[test]
    fn a_copy_predictor_lands_on_the_copy_baseline() {
        let sched = schedule();
        let s = samples();
        let mean = vec![0.0, 0.0];
        let report = evaluate(&sched, &s, &mean, 3, 1, &EvalConfig::default(), |input| {
            // input is [x_t, t0, t1] per pixel; ε̂ = x_t.
            input.iter().step_by(3).copied().collect()
        });
        for (m, b) in report.model.iter().zip(report.baselines.iter()) {
            assert!(
                (m.eps_mse() - b.eps_copy()).abs() < 1e-12,
                "bucket {}: ε {} vs {}",
                m.bucket,
                m.eps_mse(),
                b.eps_copy()
            );
            assert!(
                (m.x0_mse() - b.x0_copy()).abs() < 1e-12,
                "bucket {}: x₀ {} vs {}",
                m.bucket,
                m.x0_mse(),
                b.x0_copy()
            );
        }
    }

    /// Where the clip does not bite — low t, a small error — x₀-MSE is ε-MSE
    /// scaled by (1−ᾱ)/ᾱ. Bucket 0 sits at ᾱ≈1, so `n/s`≈0.02 and a small ε
    /// error leaves x̂₀ well inside [-1, 1]: the closed form must hold there.
    #[test]
    fn x0_matches_the_factor_where_it_does_not_clip() {
        let sched = schedule();
        let s = samples();
        let mean = vec![0.0, 0.0];
        let cfg = EvalConfig {
            buckets: 4,
            t_per_bucket: 1,
            seed: 7,
        };
        let report = evaluate(&sched, &s, &mean, 3, 1, &cfg, |input| {
            input.iter().step_by(3).map(|v| v * 0.3).collect()
        });
        let m = &report.model[0]; // bucket 0 ⇒ t = t_lo = 0, no clip.
        let ab = sched.alpha_bar(m.t_lo) as f64;
        let factor = (1.0 - ab) / ab;
        let expected = m.eps_mse() * factor;
        assert!(
            (m.x0_mse() - expected).abs() <= 1e-9 * expected.max(1.0),
            "bucket 0: x0 {} vs factor·ε {}",
            m.x0_mse(),
            expected
        );
    }

    /// The clipped reconstruction can never leave [-1, 1], so its error against
    /// a clean image (also in range) is at most `2² = 4` per element, whatever
    /// the predictor does. A predictor that returns absurd ε̂ must still produce
    /// a bounded x₀-MSE — this is the property that stops the high-t blow-up.
    #[test]
    fn the_reconstruction_is_clipped_into_range() {
        let sched = schedule();
        let s = samples();
        let mean = vec![0.0, 0.0];
        let report = evaluate(&sched, &s, &mean, 3, 1, &EvalConfig::default(), |input| {
            vec![1e6; input.len() / 3]
        });
        for m in &report.model {
            assert!(
                m.x0_mse() <= 4.0 + 1e-9,
                "bucket {}: x0_mse {} exceeded the clip bound",
                m.bucket,
                m.x0_mse()
            );
        }
    }

    /// The `ε̂ = mean` baseline reconstructs exactly the (in-range) mean image at
    /// every timestep, so its x₀-MSE is `mean((mean − x₀)²)`, the same number in
    /// every bucket. Here the mean image is zero, so it is `E[x₀²]`.
    #[test]
    fn the_mean_predictor_reconstructs_the_mean_image() {
        let sched = schedule();
        let s = samples();
        let mean = vec![0.0, 0.0];
        // E[x₀²] over the three 2-pixel samples.
        let ex2: f64 = s
            .iter()
            .flat_map(|v| v.iter())
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            / (s.len() * 2) as f64;
        let report = evaluate(&sched, &s, &mean, 3, 1, &EvalConfig::default(), |input| {
            vec![0.0; input.len() / 3]
        });
        for b in &report.baselines {
            assert!(
                (b.x0_mean() - ex2).abs() < 1e-9,
                "bucket {}: x0_mean {} vs E[x0²] {}",
                b.bucket,
                b.x0_mean(),
                ex2
            );
        }
    }

    /// The zero baseline is `mean(ε²)`, which for a standard normal field is ≈ 1
    /// everywhere on the schedule — a sanity anchor on the noise generator.
    #[test]
    fn the_zero_baseline_is_about_unit_variance() {
        let sched = schedule();
        // Many small samples so the field is well-sampled.
        let s: Vec<Vec<f32>> = (0..64).map(|_| vec![0.0; 16]).collect();
        let mean = vec![0.0; 16];
        let cfg = EvalConfig {
            buckets: 4,
            t_per_bucket: 4,
            seed: 1,
        };
        let report = evaluate(&sched, &s, &mean, 3, 1, &cfg, |input| {
            vec![0.0; input.len() / 3]
        });
        for b in &report.baselines {
            assert!(
                (b.eps_zero() - 1.0).abs() < 0.1,
                "bucket {}: E[ε²] = {}",
                b.bucket,
                b.eps_zero()
            );
        }
    }
}
