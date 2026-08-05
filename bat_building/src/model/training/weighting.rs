//! File purpose: loss weighting across diffusion timesteps, applied by biasing the timestep draw.
//!
//! `SCALE_UNET.md` §5 measures why the unweighted ε-MSE collapses the model onto
//! the dataset mean: the objective's implicit weight on the *image* error is the
//! signal-to-noise ratio `SNR(t) = ᾱ_t / (1 - ᾱ_t)`, which spans ~1e8 over the
//! T = 256 schedule. The gradient that should decide the global content of the
//! image (high `t`) is crushed by the gradient that polishes already-acquired
//! detail (low `t`).
//!
//! The fix here is deliberately the *least invasive* one: nothing in the loss,
//! the shaders or the backward pass changes. Instead of drawing
//! `t ~ U{0, T-1}`, training draws `t ~ p(t) ∝ w(t)`. In expectation this is
//! exactly a `w(t)`-weighted loss, up to the constant `Z = Σ w(t)`:
//!
//! ```text
//! E_{t ~ p}[ L_t ] = Σ_t (w(t)/Z) · L_t = (1/Z) · E_{t ~ U}[ T · w(t) · L_t ]
//! ```
//!
//! The constant is irrelevant under Adam (whose update is invariant to a
//! rescaling of the gradient) and is a plain learning-rate factor under SGD.
//!
//! The weight is the min-SNR-γ form (Hang et al., 2023):
//! `w(t) = min(1, γ / SNR(t)) = min(1, γ·(1-ᾱ_t)/ᾱ_t)`. It is the mission's
//! "(1-ᾱ)/ᾱ, capped": proportional to the inverse SNR — which flattens the
//! implicit image-error weight to a constant — clipped at 1 so the low-`t` end
//! keeps a finite share of the draws instead of vanishing.

use serde::{Deserialize, Serialize};

use super::LinearNoiseSchedule;

/// Cap on the inverse-SNR weight, as a multiple of the SNR.
///
/// `γ = 1` clips the weight where signal and noise carry equal power — the one
/// distinguished point of the schedule, so no tuned constant enters the default.
pub const DEFAULT_SNR_GAMMA: f32 = 1.0;

/// Share of the draws reserved for a plain uniform timestep, mixed into the
/// weighted distribution: `p(t) = (1-λ)·w(t)/Z + λ/T`.
///
/// Without it, `min(1, γ/SNR)` at `γ = 1` gives `t = 0` a probability of 1.8e-6
/// on the production schedule — about 0.04 expected draws over a 1500-step run
/// at batch 16, i.e. the first timesteps of the reverse chain would never be
/// trained at all. λ = 5% is the standard defensive mixture: it floors every
/// timestep at `λ/T` (≈ 4.7 expected draws over the same run) and bounds the
/// importance-sampling variance, while shifting the bucket masses by under one
/// point.
const UNIFORM_MIX: f64 = 0.05;

/// How the per-timestep loss is weighted during training.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum LossWeighting {
    /// `t ~ U{0, T-1}`, plain MSE on ε. The historical behaviour, kept as the
    /// default so no existing config or comparison changes meaning.
    Uniform,
    /// `t ~ p(t) ∝ min(1, γ/SNR(t))` — min-SNR-γ by importance sampling.
    Snr { gamma: f32 },
}

impl Default for LossWeighting {
    fn default() -> Self {
        LossWeighting::Uniform
    }
}

impl LossWeighting {
    /// Parses the `--loss-weighting` CLI value. The γ is passed separately
    /// (`--snr-gamma`) so the flag itself stays a plain enum, like
    /// `--optimizer`.
    pub fn parse(value: &str, gamma: f32) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "uniform" | "none" => Some(LossWeighting::Uniform),
            "snr" | "min-snr" | "minsnr" => Some(LossWeighting::Snr { gamma }),
            _ => None,
        }
    }

    pub fn label(self) -> String {
        match self {
            LossWeighting::Uniform => "uniform".to_string(),
            LossWeighting::Snr { gamma } => format!("snr(gamma={gamma})"),
        }
    }
}

/// Draws the training timestep for one sample.
///
/// Holds the inverse-CDF table for the non-uniform case. Built once per
/// schedule; the draw itself is a `SplitMix64` word plus a binary search.
#[derive(Debug, Clone, Default)]
pub(crate) struct TimestepSampler {
    /// `None` for [`LossWeighting::Uniform`] — that path is the byte-for-byte
    /// legacy draw, not a special case of the table (a uniform table would
    /// consume the random word differently and silently break the pairing with
    /// every run recorded before this change).
    cdf: Option<Vec<f64>>,
}

impl TimestepSampler {
    pub(crate) fn new(schedule: &LinearNoiseSchedule, weighting: LossWeighting) -> Self {
        let len = schedule.len();
        let gamma = match weighting {
            LossWeighting::Uniform => return Self { cdf: None },
            LossWeighting::Snr { gamma } => gamma,
        };
        if len == 0 || !(gamma > 0.0) || !gamma.is_finite() {
            return Self { cdf: None };
        }

        let weights: Vec<f64> = (0..len)
            .map(|t| snr_weight(schedule.alpha_bar(t) as f64, gamma as f64))
            .collect();
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) || !total.is_finite() {
            // Degenerate schedule — fall back rather than produce a broken CDF.
            return Self { cdf: None };
        }

        let mut cdf = Vec::with_capacity(len);
        let mut running = 0.0f64;
        let uniform_share = UNIFORM_MIX / len as f64;
        for w in &weights {
            running += (1.0 - UNIFORM_MIX) * (w / total) + uniform_share;
            cdf.push(running);
        }
        // Guard the binary search against float drift on the last bin.
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Self { cdf: Some(cdf) }
    }

    /// Weight assigned to each timestep, normalised to sum to 1. Empty for the
    /// uniform sampler. Exposed for tests and for the run banner.
    pub(crate) fn probabilities(&self) -> Vec<f64> {
        match self.cdf.as_ref() {
            None => Vec::new(),
            Some(cdf) => cdf
                .iter()
                .scan(0.0f64, |prev, &c| {
                    let p = c - *prev;
                    *prev = c;
                    Some(p)
                })
                .collect(),
        }
    }

    pub(crate) fn draw(&self, counter: usize, schedule_len: usize, seed: u64) -> usize {
        match self.cdf.as_ref() {
            None => super::diffusion::diffusion_step_for(counter, schedule_len, seed),
            Some(cdf) => {
                if schedule_len == 0 {
                    return 0;
                }
                let u = super::diffusion::uniform_unit_for(counter, seed);
                // First index whose cumulative mass exceeds u.
                let idx = cdf.partition_point(|&c| c <= u);
                idx.min(cdf.len().saturating_sub(1)).min(schedule_len - 1)
            }
        }
    }
}

/// `min(1, γ/SNR)` written so it cannot divide by zero: `SNR = ᾱ/(1-ᾱ)`, hence
/// `γ/SNR = γ(1-ᾱ)/ᾱ`, and `ᾱ = 0` (unreachable in a valid schedule, but the
/// terminal `ᾱ` is ~3e-5) is the *high*-noise end where the weight is 1 anyway.
fn snr_weight(alpha_bar: f64, gamma: f64) -> f64 {
    let alpha_bar = alpha_bar.clamp(0.0, 1.0);
    if alpha_bar <= 0.0 {
        return 1.0;
    }
    (gamma * (1.0 - alpha_bar) / alpha_bar).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::LinearNoiseSchedule;

    fn production_schedule() -> LinearNoiseSchedule {
        LinearNoiseSchedule::new_linear(256, 1e-4, 0.02)
    }

    #[test]
    fn parses_the_cli_values() {
        assert_eq!(
            LossWeighting::parse("uniform", 1.0),
            Some(LossWeighting::Uniform)
        );
        assert_eq!(
            LossWeighting::parse(" SNR ", 5.0),
            Some(LossWeighting::Snr { gamma: 5.0 })
        );
        assert_eq!(LossWeighting::parse("cosine", 1.0), None);
    }

    #[test]
    fn defaults_to_uniform() {
        assert_eq!(LossWeighting::default(), LossWeighting::Uniform);
    }

    /// Non-regression: the uniform sampler must reproduce the legacy draw
    /// exactly, for every counter and seed the training loop can present. Any
    /// deviation would make every run recorded before this change
    /// incomparable.
    #[test]
    fn uniform_sampler_is_bit_identical_to_the_legacy_draw() {
        let schedule = production_schedule();
        let sampler = TimestepSampler::new(&schedule, LossWeighting::Uniform);
        for step in [0usize, 1, 7, 500, 9_999] {
            let seed = (step as u64) << 32;
            for offset in 0..64usize {
                let counter = step * 16 + offset;
                assert_eq!(
                    sampler.draw(counter, schedule.len(), seed),
                    crate::training::diffusion::diffusion_step_for(counter, schedule.len(), seed),
                    "counter={counter} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn snr_weight_matches_the_min_snr_formula() {
        let schedule = production_schedule();
        for gamma in [0.2f64, 1.0, 5.0] {
            for t in [0usize, 32, 64, 128, 255] {
                let alpha_bar = schedule.alpha_bar(t) as f64;
                let snr = alpha_bar / (1.0 - alpha_bar);
                let expected = (gamma / snr).min(1.0);
                let got = snr_weight(alpha_bar, gamma);
                assert!(
                    (got - expected).abs() < 1e-12,
                    "gamma={gamma} t={t}: {got} vs {expected}"
                );
            }
        }
    }

    /// The whole point of the change: the low-`t` quarter must lose most of its
    /// share of the draws, and the top half must gain.
    #[test]
    fn snr_probabilities_shift_mass_towards_high_t() {
        let schedule = production_schedule();
        let sampler = TimestepSampler::new(&schedule, LossWeighting::Snr { gamma: 1.0 });
        let p = sampler.probabilities();
        assert_eq!(p.len(), schedule.len());
        let total: f64 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "probabilities sum to {total}");

        let bucket = |lo: usize, hi: usize| -> f64 { p[lo..hi].iter().sum() };
        let low = bucket(0, 64);
        let high = bucket(192, 256);
        assert!(
            low < 0.12,
            "low-t bucket still carries {low:.4} of the mass (uniform = 0.25)"
        );
        assert!(
            high > 0.28,
            "high-t bucket only carries {high:.4} of the mass (uniform = 0.25)"
        );
        // Monotone within the schedule: no timestep is ever favoured over a
        // noisier one. Tolerance is relative — once the weight saturates at 1 the
        // successive differences of the CDF wobble by an ULP.
        for t in 1..schedule.len() {
            assert!(
                p[t] >= p[t - 1] * (1.0 - 1e-9),
                "p({t}) = {} < p({}) = {}",
                p[t],
                t - 1,
                p[t - 1]
            );
        }

        // The uniform mixture must floor every timestep: with 1500 steps at
        // batch 16 that is ~4.7 expected draws for the least likely one, instead
        // of 0.04 without it.
        let floor = super::UNIFORM_MIX / schedule.len() as f64;
        assert!(
            p.iter().all(|&x| x >= floor * (1.0 - 1e-9)),
            "some timestep falls below the uniform floor {floor:.3e}"
        );
    }

    /// The empirical draw distribution must match `p(t)`. Compared over the
    /// four probe buckets, which is what the training metrics report, with a
    /// tolerance set from the binomial standard error (`sqrt(p(1-p)/n)`, at most
    /// ~1.1e-3 here) — 6σ, so the test is not flaky but would still catch any
    /// mis-indexed CDF.
    #[test]
    fn snr_empirical_draws_match_the_target_distribution() {
        let schedule = production_schedule();
        let len = schedule.len();
        let sampler = TimestepSampler::new(&schedule, LossWeighting::Snr { gamma: 1.0 });
        let target = sampler.probabilities();

        let draws = 200_000usize;
        let mut counts = vec![0usize; len];
        for counter in 0..draws {
            let seed = ((counter / 16) as u64) << 32;
            let t = sampler.draw(counter, len, seed);
            assert!(t < len, "draw {t} outside the schedule");
            counts[t] += 1;
        }

        for (lo, hi) in [(0usize, 64usize), (64, 128), (128, 192), (192, 256)] {
            let expected: f64 = target[lo..hi].iter().sum();
            let observed = counts[lo..hi].iter().sum::<usize>() as f64 / draws as f64;
            let sigma = (expected * (1.0 - expected) / draws as f64).sqrt();
            assert!(
                (observed - expected).abs() < 6.0 * sigma,
                "bucket [{lo},{hi}): observed {observed:.5} vs expected {expected:.5} \
                 (6 sigma = {:.5})",
                6.0 * sigma
            );
        }

        // Every timestep must remain reachable — an importance-sampled schedule
        // that starves a region entirely would not be trainable there.
        assert!(
            counts.iter().all(|&c| c > 0),
            "{} timesteps were never drawn in {draws} draws",
            counts.iter().filter(|&&c| c == 0).count()
        );
    }

    /// A γ that cannot define a distribution must degrade to uniform rather
    /// than produce a broken CDF.
    #[test]
    fn invalid_gamma_falls_back_to_uniform() {
        let schedule = production_schedule();
        for gamma in [0.0f32, -1.0, f32::NAN] {
            let sampler = TimestepSampler::new(&schedule, LossWeighting::Snr { gamma });
            assert!(
                sampler.probabilities().is_empty(),
                "gamma={gamma} produced a table"
            );
            assert_eq!(
                sampler.draw(3, schedule.len(), 42),
                crate::training::diffusion::diffusion_step_for(3, schedule.len(), 42)
            );
        }
    }

    #[test]
    fn serde_roundtrip_and_missing_field_default() {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Cfg {
            #[serde(default)]
            loss_weighting: LossWeighting,
        }
        let cfg: Cfg = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.loss_weighting, LossWeighting::Uniform);
        let cfg: Cfg =
            serde_json::from_str(r#"{"loss_weighting":{"kind":"snr","gamma":2.5}}"#).unwrap();
        assert_eq!(cfg.loss_weighting, LossWeighting::Snr { gamma: 2.5 });
        let text = serde_json::to_string(&cfg).unwrap();
        assert!(text.contains("snr"), "{text}");
    }
}
