//! The linear noise schedule: betas/alphas/alpha_bars and the forward, reverse
//! and gaussian-field operations built on them.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct LinearNoiseSchedule {
    betas: Vec<f32>,
    alphas: Vec<f32>,
    alpha_bars: Vec<f32>,
}

/// Which of DDPM's two admissible reverse-step variances the posterior draw uses
/// (Ho et al. 2020, §3.2; they coincide only as `T → ∞`):
///
/// - [`Beta`](PosteriorVariance::Beta) — `sigma_t^2 = beta_t`. The larger, the
///   default, and BIT-FOR-BIT the historical draw.
/// - [`Posterior`](PosteriorVariance::Posterior) — the true posterior variance
///   `beta_t · (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)`. Strictly smaller, and
///   smallest at `t = 1` (`~0.60·beta` here) where the sharpness-deciding pixels are drawn.
///
/// The formula lives in ONE place, [`LinearNoiseSchedule::posterior_sigma`], so
/// the native sampler, web crate and walkers read the same number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PosteriorVariance {
    /// `sigma_t^2 = beta_t`. The legacy draw and the default.
    #[default]
    Beta,
    /// `sigma_t^2 = beta_t · (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)`.
    Posterior,
}

impl PosteriorVariance {
    /// Parses the CLI spelling; unknown values are rejected by the caller.
    pub fn from_cli(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "beta" => Some(Self::Beta),
            "posterior" => Some(Self::Posterior),
            _ => None,
        }
    }

    /// The name the run banner and the config file use.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Beta => "beta",
            Self::Posterior => "posterior",
        }
    }
}

/// Step count the reference DDPM betas (1e-4 .. 0.02) are calibrated for.
const BETA_REFERENCE_STEPS: f32 = 1000.0;
/// Upper bound on a single beta. Keeps alpha strictly positive so alpha_bar
/// stays strictly decreasing even after rescaling for a short schedule.
const MAX_BETA: f32 = 0.999;
/// Residual signal budget at the terminal step: alpha_bar(T-1) must fall below
/// this for q(x_T) to be approximately N(0, I).
const TERMINAL_ALPHA_BAR_LIMIT: f32 = 1e-3;

impl LinearNoiseSchedule {
    /// Builds a linear beta schedule. `beta_start`/`beta_end` are calibrated for
    /// `BETA_REFERENCE_STEPS` (DDPM's T = 1000) and rescaled by
    /// `BETA_REFERENCE_STEPS / num_steps`, preserving total injected noise at any T.
    /// Without this, the paper's betas at T = 256 left 27% of the signal in x_T
    /// while sampling starts from pure noise.
    pub fn new_linear(num_steps: usize, beta_start: f32, beta_end: f32) -> Self {
        assert!(num_steps > 0, "noise schedule requires at least one step");
        assert!(
            beta_start > 0.0 && beta_end > 0.0 && beta_start <= beta_end && beta_end < 1.0,
            "noise schedule betas must be in (0, 1) and ordered"
        );

        let scale = BETA_REFERENCE_STEPS / num_steps as f32;
        let scaled_start = (beta_start * scale).min(MAX_BETA);
        let scaled_end = (beta_end * scale).min(MAX_BETA);

        let mut betas = Vec::with_capacity(num_steps);
        let mut alphas = Vec::with_capacity(num_steps);
        let mut alpha_bars = Vec::with_capacity(num_steps);
        let denom = (num_steps.saturating_sub(1)).max(1) as f32;
        let mut running_alpha_bar = 1.0f32;

        for step in 0..num_steps {
            let t = step as f32 / denom;
            let beta = (scaled_start + (scaled_end - scaled_start) * t).min(MAX_BETA);
            let alpha = 1.0 - beta;
            running_alpha_bar *= alpha;
            betas.push(beta);
            alphas.push(alpha);
            alpha_bars.push(running_alpha_bar);
        }

        let terminal = *alpha_bars.last().expect("schedule has at least one step");
        assert!(
            terminal < TERMINAL_ALPHA_BAR_LIMIT,
            "noise schedule never reaches pure noise: alpha_bar({}) = {terminal:.3e} \
             (sqrt = {:.4}, i.e. {:.1}% residual signal in x_T). \
             q(x_T) must be approximately N(0, I) since sampling starts there.",
            num_steps - 1,
            terminal.sqrt(),
            terminal.sqrt() * 100.0
        );

        Self {
            betas,
            alphas,
            alpha_bars,
        }
    }

    pub fn len(&self) -> usize {
        self.alpha_bars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.alpha_bars.is_empty()
    }

    pub fn beta(&self, step: usize) -> f32 {
        self.betas[step.min(self.betas.len().saturating_sub(1))]
    }

    pub fn alpha(&self, step: usize) -> f32 {
        self.alphas[step.min(self.alphas.len().saturating_sub(1))]
    }

    pub fn alpha_bar(&self, step: usize) -> f32 {
        self.alpha_bars[step.min(self.alpha_bars.len().saturating_sub(1))]
    }

    /// `alpha_bar` of the level *just below* `step` — the level a state sits at
    /// before a [`Self::forward_step`] at `step` carries it up, and the level a
    /// [`Self::denoise_step`] at `step` carries it down to.
    ///
    /// Below the bottom of the schedule that state is a clean `x0`, which
    /// carries all of its signal: `1`. Single definition of an off-by-one that
    /// three formulas depend on.
    pub fn alpha_bar_below(&self, step: usize) -> f32 {
        if step == 0 {
            1.0
        } else {
            self.alpha_bar(step - 1)
        }
    }

    /// The reverse-step noise scale `sigma_t` for the chosen [`PosteriorVariance`],
    /// BEFORE `denoise_magnitude` and the `step == 0` override. The SOLE definition
    /// of the two variances; `Beta` returns `sqrt(beta_t)`, the exact literal used
    /// before this method existed, so the default stays bit-identical.
    pub fn posterior_sigma(&self, step: usize, variance: PosteriorVariance) -> f32 {
        let beta = self.beta(step);
        match variance {
            PosteriorVariance::Beta => beta.sqrt(),
            PosteriorVariance::Posterior => {
                let alpha_bar = self.alpha_bar(step);
                let alpha_bar_prev = self.alpha_bar_below(step);
                // beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t).
                // 1 - alpha_bar_t is bounded away from zero for t >= 1 (the only
                // steps that draw noise), so no guard is needed here.
                (beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).sqrt()
            }
        }
    }

    pub fn normalized_step(&self, step: usize) -> f32 {
        if self.len() <= 1 {
            0.0
        } else {
            step.min(self.len() - 1) as f32 / (self.len() - 1) as f32
        }
    }

    /// Smooth multi-resolution timestep embedding: `tau = step/(T-1)`, each channel
    /// pair `i` encoding `[sin(pi·2^i·tau), cos(pi·2^i·tau)]`. The lowest pair is a
    /// monotone `cos(pi·tau)` so even one pair identifies t smoothly, unlike the old
    /// `sin(step)`/`cos(step)` (period ≈6.28 steps) that aliased adjacent timesteps.
    ///
    /// MUST stay identical to `timestep_value` in `shader/diffusion_prepare.wgsl`
    /// (guarded by `timestep_embedding_matches_shader_formula`) — training composes
    /// on the GPU, inference here on the CPU, and both must feed the same conditioning.
    pub fn timestep_embedding(&self, step: usize, channels: usize) -> Vec<f32> {
        if channels == 0 {
            return Vec::new();
        }

        let steps = self.len().max(1);
        let denom = steps.saturating_sub(1).max(1) as f32;
        let tau = step.min(steps - 1) as f32 / denom;
        let mut values = Vec::with_capacity(channels);
        let half = channels.div_ceil(2);

        for i in 0..half {
            let phase = tau * std::f32::consts::PI * 2.0f32.powi(i as i32);
            values.push(phase.sin());
            if values.len() < channels {
                values.push(phase.cos());
            }
        }

        values
    }

    pub fn add_noise(&self, clean: &[f32], step: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let alpha_bar = self.alpha_bar(step);
        let signal_scale = alpha_bar.sqrt();
        let noise_scale = (1.0 - alpha_bar).sqrt();
        let mut noisy = Vec::with_capacity(clean.len());
        let mut noise = Vec::with_capacity(clean.len());

        for (index, value) in clean.iter().enumerate() {
            let sample_noise = gaussian_at(seed, index);
            noise.push(sample_noise);
            noisy.push(signal_scale * *value + noise_scale * sample_noise);
        }

        (noisy, noise)
    }

    /// One increment of the forward Markov chain:
    /// `x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * eps`. Chaining `0..=t`
    /// from a clean `x0` is the same distribution as [`Self::add_noise`]'s jump
    /// (`climbing_the_forward_chain_matches_add_noise_in_distribution` measures it).
    /// The perpetual climb uses [`Self::forward_from`] instead (an exact Markov walk
    /// seethes — see module header); this stays as the tested definition it is
    /// measured against. `seed` must be fresh per increment — reusing one adds the
    /// same field `t` times, not a Gaussian walk.
    pub fn forward_step(&self, previous: &[f32], step: usize, seed: u64) -> Vec<f32> {
        let step = step.min(self.len().saturating_sub(1));
        let signal_scale = self.alpha(step).sqrt();
        let noise_scale = self.beta(step).sqrt();
        previous
            .iter()
            .enumerate()
            .map(|(index, value)| signal_scale * *value + noise_scale * gaussian_at(seed, index))
            .collect()
    }

    /// The forward process in closed form, from an arbitrary intermediate state.
    /// `departure` sits just BELOW `departure_step`; `step` is the level to land
    /// on. With `r = alpha_bar(step) / alpha_bar_below(departure_step)`:
    ///
    /// ```text
    ///   x_t = sqrt(r) * x_dep + sqrt(1 - r) * eps
    /// ```
    ///
    /// This is `q(x_t | x_s)` for any `s < t`, not just `s = clean`: the alpha_bar
    /// ratio IS the generalisation, which is why the departure may be a noisy
    /// latent where [`Self::add_noise`] needs a clean `x0`. At `departure_step = 0`
    /// it reduces to `add_noise` exactly.
    ///
    /// The perpetual climb walks THIS, not [`Self::forward_step`]: `seed` names ONE
    /// field for the whole climb, every level computed from the departure, so the
    /// grain is revealed (amplitude rising) rather than reshuffled — maximally
    /// correlated frames, not a Markov chain that seethes. See module header.
    pub fn forward_from(
        &self,
        departure: &[f32],
        departure_step: usize,
        step: usize,
        seed: u64,
    ) -> Vec<f32> {
        let step = step.min(self.len().saturating_sub(1));
        // Clamped so a caller that asks to land at or below its own departure
        // gets the departure back rather than a NaN out of sqrt of a negative.
        let ratio = (self.alpha_bar(step) / self.alpha_bar_below(departure_step)).clamp(0.0, 1.0);
        let signal_scale = ratio.sqrt();
        let noise_scale = (1.0 - ratio).sqrt();
        departure
            .iter()
            .enumerate()
            .map(|(index, value)| signal_scale * *value + noise_scale * gaussian_at(seed, index))
            .collect()
    }

    pub fn sample_noise(&self, len: usize, seed: u64) -> Vec<f32> {
        (0..len).map(|index| gaussian_at(seed, index)).collect()
    }

    pub fn denoise_step(
        &self,
        latent: &[f32],
        predicted_noise: &[f32],
        step: usize,
        seed: u64,
    ) -> Vec<f32> {
        self.denoise_step_with_magnitude(
            latent,
            predicted_noise,
            step,
            seed,
            1.0,
            PosteriorVariance::Beta,
        )
    }

    /// The clipped x0 estimate the posterior mean is built from, for the whole
    /// tensor: `clamp(x_t - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar)`.
    ///
    /// "What the model believes the clean image is" at `step` — the same quantity
    /// [`Self::denoise_step_with_magnitude`] computes internally (both through
    /// [`x0_hat_at`], so the visualiser cannot drift from the sampler). Derived
    /// from `latent`/`predicted_noise` alone: side-effect free and optional.
    pub fn x0_estimate(&self, latent: &[f32], predicted_noise: &[f32], step: usize) -> Vec<f32> {
        assert_eq!(
            latent.len(),
            predicted_noise.len(),
            "latent and predicted noise lengths must match"
        );
        let step = step.min(self.len().saturating_sub(1));
        let alpha_bar = self.alpha_bar(step);
        let noise_to_x0 = (1.0 - alpha_bar).sqrt();
        let x0_scale = 1.0 / alpha_bar.sqrt().max(f32::EPSILON);
        latent
            .iter()
            .zip(predicted_noise.iter())
            .map(|(value, noise)| x0_hat_at(*value, *noise, x0_scale, noise_to_x0))
            .collect()
    }

    pub fn denoise_step_with_magnitude(
        &self,
        latent: &[f32],
        predicted_noise: &[f32],
        step: usize,
        seed: u64,
        denoise_magnitude: f32,
        variance: PosteriorVariance,
    ) -> Vec<f32> {
        assert_eq!(
            latent.len(),
            predicted_noise.len(),
            "latent and predicted noise lengths must match"
        );
        let step = step.min(self.len().saturating_sub(1));
        let beta = self.beta(step);
        let alpha = self.alpha(step);
        let alpha_bar = self.alpha_bar(step);
        let alpha_bar_prev = self.alpha_bar_below(step);
        // Posterior mean from the CLIPPED x0 estimate: clamping x0 to the data
        // range bounds the chain by construction, so a degenerate ε̂ cannot be
        // amplified across the ~1/sqrt(alpha_bar_T) chain gain. A no-op for a
        // well-trained model whose x0_hat already lies in [-1, 1].
        let coeff_x0 = alpha_bar_prev.sqrt() * beta / (1.0 - alpha_bar);
        let coeff_xt = alpha.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar);
        let noise_to_x0 = (1.0 - alpha_bar).sqrt();
        let x0_scale = 1.0 / alpha_bar.sqrt().max(f32::EPSILON);
        let magnitude = denoise_magnitude.max(0.0);
        let sigma = if step == 0 {
            0.0
        } else {
            self.posterior_sigma(step, variance) * magnitude
        };

        latent
            .iter()
            .zip(predicted_noise.iter())
            .enumerate()
            .map(|(index, (value, noise))| {
                let x0_hat = x0_hat_at(*value, *noise, x0_scale, noise_to_x0);
                let mean = coeff_x0 * x0_hat + coeff_xt * *value;
                if sigma == 0.0 {
                    mean
                } else {
                    mean + sigma * gaussian_at(seed, index)
                }
            })
            .collect()
    }
}

/// The clipped x0 estimate for a single element — the SOLE definition of the
/// formula, so the reverse step's mean and the visualiser's display cannot drift.
#[inline]
fn x0_hat_at(latent: f32, predicted_noise: f32, x0_scale: f32, noise_to_x0: f32) -> f32 {
    (x0_scale * (latent - noise_to_x0 * predicted_noise)).clamp(-1.0, 1.0)
}

/// Odd increment of the SplitMix64 stream (the golden-ratio constant).
const STREAM_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// Draws element `index` of the noise field identified by `seed`. The seed is
/// AVALANCHED FIRST (`fmix64`), THEN the index is folded in by ADDITION through an
/// odd multiplier — never XOR. `seed ^ index` looks harmless but is catastrophic
/// once the CALLER also XORs its per-draw seed (`path_seed ^ diffusion_step`): the
/// whole chain then draws one field merely reindexed, which on a 32-wide image
/// came out near-constant along a row — horizontal bands whatever the model said
/// (`ANISOTROPY_HUNT.md`). Avalanching first destroys any caller-seed relation
/// (XOR or `+ step*GAMMA`) before the index is folded in; the order matters (the
/// reverse collapses to a field constant along the anti-diagonals). Guarded by
/// `injected_noise_over_reverse_chain_is_isotropic`.
fn gaussian_at(seed: u64, index: usize) -> f32 {
    let field_key = fmix64(seed);
    gaussian_from_seed(
        field_key.wrapping_add((index as u64).wrapping_add(1).wrapping_mul(STREAM_GAMMA)),
    )
}

/// murmur3's 64-bit finaliser: full avalanche, every input bit affecting every
/// output bit.
fn fmix64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51afd7ed558ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ceb9fe1a85ec53);
    value ^= value >> 33;
    value
}

fn unit_from_seed(value: u64) -> f32 {
    let normalized = (fmix64(value) >> 40) as u32;
    (normalized as f32 / ((1u32 << 24) - 1) as f32).clamp(1e-7, 1.0 - 1e-7)
}

fn gaussian_from_seed(seed: u64) -> f32 {
    let u1 = unit_from_seed(seed);
    let u2 = unit_from_seed(seed ^ 0x9e37_79b9_7f4a_7c15);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::{LinearNoiseSchedule, PosteriorVariance};

    #[test]
    fn linear_schedule_monotonically_decreases_alpha_bar() {
        let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
        for step in 1..schedule.len() {
            assert!(schedule.alpha_bar(step) < schedule.alpha_bar(step - 1));
            assert!(schedule.beta(step) >= schedule.beta(step - 1));
            assert!(schedule.alpha(step) <= schedule.alpha(step - 1));
        }
    }

    /// Finding #3 — the betas are calibrated for T = 1000 and must be rescaled
    /// for the T = 256 schedule the project actually runs, so that x_T is
    /// (approximately) pure noise.
    #[test]
    fn production_schedule_reaches_pure_noise() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let terminal = schedule.alpha_bar(schedule.len() - 1);
        assert!(
            terminal < 1e-3,
            "alpha_bar(T-1) = {terminal:.3e}, residual signal {:.2}%",
            terminal.sqrt() * 100.0
        );
    }

    /// The default variance must inject *exactly* what the step injected before
    /// the option existed — `sqrt(beta_t)`, to the bit, at every step. This is the
    /// "the default changes nothing" guarantee at the level of the sole formula.
    #[test]
    fn the_beta_variance_is_the_historical_sigma_to_the_bit() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        for step in 0..schedule.len() {
            assert_eq!(
                schedule.posterior_sigma(step, PosteriorVariance::Beta),
                schedule.beta(step).sqrt(),
                "beta sigma diverged from sqrt(beta) at step {step}"
            );
        }
    }

    /// The posterior variance is the strictly smaller of DDPM's two, and it is
    /// smallest at the very bottom of the chain. The ratio `sigma_post / sigma_beta`
    /// equals `sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t))`, which is `~0.60`
    /// at `t = 1` on this schedule — where the last, sharpness-deciding pixels are
    /// drawn — and rises toward `1` at the top.
    #[test]
    fn the_posterior_variance_is_smaller_and_shrinks_most_at_the_bottom() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let ratio = |step: usize| {
            schedule.posterior_sigma(step, PosteriorVariance::Posterior)
                / schedule.posterior_sigma(step, PosteriorVariance::Beta)
        };
        // Never larger than beta's, at any noise-drawing step.
        for step in 1..schedule.len() {
            assert!(
                ratio(step) <= 1.0 + 1e-6,
                "posterior/beta = {} > 1 at step {step}",
                ratio(step)
            );
        }
        // Smallest at the bottom, matching the mission's measured 0.600 at t = 1.
        assert!(
            (ratio(1) - 0.600).abs() < 0.01,
            "posterior/beta at t=1 = {}, expected ~0.600",
            ratio(1)
        );
        // Monotone-ish rise: the top of the chain is close to beta's.
        assert!(ratio(1) < ratio(schedule.len() - 1));
        assert!(ratio(schedule.len() - 1) > 0.99);
    }

    /// Rescaling must preserve the total injected noise across step counts:
    /// the same betas at different T land in the same ballpark at the end.
    #[test]
    fn rescaling_keeps_terminal_alpha_bar_comparable_across_step_counts() {
        for steps in [128usize, 256, 512, 1000] {
            let schedule = LinearNoiseSchedule::new_linear(steps, 1e-4, 0.02);
            let terminal = schedule.alpha_bar(schedule.len() - 1);
            assert!(
                terminal < 1e-3,
                "T={steps}: alpha_bar(T-1) = {terminal:.3e} does not reach pure noise"
            );
        }
    }

    #[test]
    fn add_noise_is_deterministic_for_same_seed() {
        let schedule = LinearNoiseSchedule::new_linear(4, 1e-4, 0.02);
        let clean = vec![0.25, 0.5, 0.75];
        let first = schedule.add_noise(&clean, 2, 1234);
        let second = schedule.add_noise(&clean, 2, 1234);
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
    }

    /// Walking the forward chain one step at a time lands where the `add_noise`
    /// jump does (same signal coefficient, same noise variance) — what licenses the
    /// gradual re-noising. Asserted by MEASUREMENT of `gaussian_at`'s actual draws,
    /// not algebra (a reused seed would satisfy the algebra and fail here). Every
    /// intermediate level is checked too — those are the frames the user watches.
    #[test]
    fn climbing_the_forward_chain_matches_add_noise_in_distribution() {
        const N: usize = 16_384;
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        // A structured x0 in range: a flat field makes the signal unmeasurable.
        let x0: Vec<f32> = (0..N).map(|i| (i as f32 * 0.017).sin() * 0.8).collect();
        let energy: f64 = x0.iter().map(|v| (*v as f64) * (*v as f64)).sum();

        // Least-squares coefficient of x0 in x — the surviving signal.
        let signal_of = |x: &[f32]| -> f64 {
            x.iter()
                .zip(x0.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum::<f64>()
                / energy
        };
        // Mean and standard deviation of what is left once that signal is removed.
        let residual_of = |x: &[f32], signal: f64| -> (f64, f64) {
            let residual: Vec<f64> = x
                .iter()
                .zip(x0.iter())
                .map(|(a, b)| *a as f64 - signal * *b as f64)
                .collect();
            let mean = residual.iter().sum::<f64>() / N as f64;
            let variance = residual
                .iter()
                .map(|r| (r - mean) * (r - mean))
                .sum::<f64>()
                / N as f64;
            (mean, variance.sqrt())
        };

        for target in [8usize, 32, 64] {
            // The walk: one increment per timestep, each with its own field.
            let mut climbed = x0.clone();
            for k in 0..=target {
                climbed = schedule.forward_step(&climbed, k, 0xc11b_0000 + k as u64);

                // Each frame the user sees must itself be a valid x_k.
                let expected = schedule.alpha_bar(k).sqrt() as f64;
                let measured = signal_of(&climbed);
                assert!(
                    (measured - expected).abs() < 0.02,
                    "intermediate frame at k={k} is not an x_k: signal {measured:.4} \
                     vs sqrt(alpha_bar) {expected:.4}"
                );
            }

            // The jump, for the same destination.
            let (direct, _) = schedule.add_noise(&x0, target, 0xd1_5ec7);

            let expected_signal = schedule.alpha_bar(target).sqrt() as f64;
            let expected_sigma = (1.0 - schedule.alpha_bar(target) as f64).sqrt();

            for (name, sample) in [("climbed", &climbed), ("direct", &direct)] {
                let signal = signal_of(sample);
                let (mean, sigma) = residual_of(sample, signal);
                assert!(
                    (signal - expected_signal).abs() < 0.02,
                    "t={target} {name}: signal {signal:.4}, want {expected_signal:.4}"
                );
                // 4 standard errors of the mean (sigma / sqrt(N)).
                assert!(
                    mean.abs() < 4.0 * expected_sigma / (N as f64).sqrt(),
                    "t={target} {name}: noise mean {mean:.4} is not centred"
                );
                assert!(
                    (sigma / expected_sigma - 1.0).abs() < 0.03,
                    "t={target} {name}: noise sd {sigma:.4}, want {expected_sigma:.4}"
                );
            }

            // …and the two agree with each other, not merely each with theory.
            let (climbed_signal, direct_signal) = (signal_of(&climbed), signal_of(&direct));
            let climbed_sigma = residual_of(&climbed, climbed_signal).1;
            let direct_sigma = residual_of(&direct, direct_signal).1;
            assert!(
                (climbed_signal - direct_signal).abs() < 0.03
                    && (climbed_sigma / direct_sigma - 1.0).abs() < 0.05,
                "t={target}: climb ({climbed_signal:.4}, {climbed_sigma:.4}) and jump \
                 ({direct_signal:.4}, {direct_sigma:.4}) disagree"
            );
        }
    }

    /// `forward_from` must be a legitimate `x_t` at EVERY level, from a clean `x0`
    /// AND from a noisy latent (breathe's case, which `add_noise` cannot serve).
    /// Measured against theory AND against an exact Markov walk from the same
    /// departure — the last level is what the next descent is handed.
    #[test]
    fn the_closed_form_climb_is_a_valid_x_t_at_every_level() {
        // Large enough that the 0.02 gate is a ~6-sigma statement: the LSQ
        // coefficient's standard error is ~0.003 here.
        const N: usize = 65_536;
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let x0: Vec<f32> = (0..N).map(|i| (i as f32 * 0.017).sin() * 0.8).collect();

        // Two departures: the settled image a wander closes on, and a genuine
        // intermediate latent of the kind breathing floors at.
        let from_clean = (0usize, x0.clone());
        let from_latent = (24usize, schedule.add_noise(&x0, 23, 0x0dd_1a7e).0);

        for (departure_step, departure) in [from_clean, from_latent] {
            let energy: f64 = departure.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let signal_of = |x: &[f32]| -> f64 {
                x.iter()
                    .zip(departure.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum::<f64>()
                    / energy
            };
            let residual_of = |x: &[f32], signal: f64| -> (f64, f64) {
                let residual: Vec<f64> = x
                    .iter()
                    .zip(departure.iter())
                    .map(|(a, b)| *a as f64 - signal * *b as f64)
                    .collect();
                let mean = residual.iter().sum::<f64>() / N as f64;
                let variance =
                    residual.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / N as f64;
                (mean, variance.sqrt())
            };

            // The reference the closed form has to match: the exact Markov walk
            // from the same departure, a fresh field per increment.
            let ceiling = departure_step + 64;
            let mut walked = departure.clone();
            let alpha_bar_dep = schedule.alpha_bar_below(departure_step) as f64;

            for level in departure_step..=ceiling {
                let carried = schedule.forward_from(&departure, departure_step, level, 0xc0e5_7e57);
                walked = schedule.forward_step(&walked, level, 0x1_0000 + level as u64);

                let ratio = schedule.alpha_bar(level) as f64 / alpha_bar_dep;
                let expected_signal = ratio.sqrt();
                let expected_sigma = (1.0 - ratio).sqrt();

                for (name, sample) in [("closed form", &carried), ("markov walk", &walked)] {
                    let signal = signal_of(sample);
                    let (mean, sigma) = residual_of(sample, signal);
                    assert!(
                        (signal - expected_signal).abs() < 0.02,
                        "dep={departure_step} level={level} {name}: signal {signal:.4}, \
                         want sqrt(alpha_bar ratio) {expected_signal:.4}"
                    );
                    assert!(
                        mean.abs() < 4.0 * expected_sigma.max(1e-3) / (N as f64).sqrt(),
                        "dep={departure_step} level={level} {name}: noise mean {mean:.4} \
                         is not centred"
                    );
                    // The first level of a climb from a clean x0 carries almost
                    // no noise (sqrt(1 - alpha_bar_0) = 0.02), where a relative
                    // tolerance is meaningless; the absolute one covers it.
                    assert!(
                        (sigma - expected_sigma).abs() < 0.03 * expected_sigma.max(0.3),
                        "dep={departure_step} level={level} {name}: noise sd {sigma:.4}, \
                         want {expected_sigma:.4}"
                    );
                }
            }

            // …and the endpoint — the x_t the next descent is handed — agrees
            // with the walk, not merely each with theory.
            let landed = schedule.forward_from(&departure, departure_step, ceiling, 0xc0e5_7e57);
            let (a, b) = (signal_of(&landed), signal_of(&walked));
            let (sa, sb) = (residual_of(&landed, a).1, residual_of(&walked, b).1);
            assert!(
                (a - b).abs() < 0.03 && (sa / sb - 1.0).abs() < 0.05,
                "dep={departure_step}: closed form ({a:.4}, {sa:.4}) and walk \
                 ({b:.4}, {sb:.4}) hand the sampler different distributions"
            );
        }
    }

    /// The point of the closed form, as a number: the noise a climb adds
    /// between two consecutive frames must be **the same pattern**, where the
    /// Markov walk draws an unrelated one every frame. Thirty unrelated grains
    /// a second is what "ça frise pendant la remontée" was.
    ///
    /// Measured on the frame-to-frame change, which is what the eye integrates,
    /// and identically on both branches so the comparison is fair.
    #[test]
    fn consecutive_climb_frames_change_by_the_same_grain() {
        const N: usize = 4096;
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let x0: Vec<f32> = (0..N).map(|i| (i as f32 * 0.017).sin() * 0.8).collect();
        let ceiling = 64usize;

        let mut carried = Vec::new();
        let mut walked = Vec::new();
        let mut walking = x0.clone();
        for level in 0..=ceiling {
            carried.push(schedule.forward_from(&x0, 0, level, 0xc0e5_7e57));
            walking = schedule.forward_step(&walking, level, 0x1_0000 + level as u64);
            walked.push(walking.clone());
        }

        // Correlation between the change at one frame and the change at the
        // next, averaged over the climb.
        let coherence = |frames: &[Vec<f32>]| -> f64 {
            let deltas: Vec<Vec<f64>> = frames
                .windows(2)
                .map(|w| {
                    w[1].iter()
                        .zip(w[0].iter())
                        .map(|(a, b)| (*a - *b) as f64)
                        .collect()
                })
                .collect();
            let dot = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
            let pairs: Vec<f64> = deltas
                .windows(2)
                .map(|w| dot(&w[0], &w[1]) / (dot(&w[0], &w[0]).sqrt() * dot(&w[1], &w[1]).sqrt()))
                .collect();
            pairs.iter().sum::<f64>() / pairs.len() as f64
        };

        let (coherent, incoherent) = (coherence(&carried), coherence(&walked));
        assert!(
            coherent > 0.95,
            "the closed-form climb does not reveal one grain: mean frame-to-frame \
             correlation {coherent:.3}"
        );
        assert!(
            incoherent.abs() < 0.15,
            "the Markov walk was supposed to be the incoherent branch, and measured \
             {incoherent:.3} — this test no longer compares what it claims to"
        );
    }

    /// The failure mode an incremental climb invites: one seed for the whole
    /// ascent adds the same field over and over (a single field scaled up, never
    /// dissolving). The variance check above would pass; only comparing two
    /// increments catches it. Guards [`LinearNoiseSchedule::forward_step`], the
    /// operator the closed form is measured against.
    #[test]
    fn a_climb_draws_a_different_field_at_every_increment() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let flat = vec![0.0f32; 1024];
        let first = schedule.forward_step(&flat, 40, 0xa11ce);
        let second = schedule.forward_step(&flat, 40, 0xb0b);
        let same = first
            .iter()
            .zip(second.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-6)
            .count();
        assert!(
            same < flat.len() / 100,
            "{same}/{} elements coincide: the two increments drew the same field",
            flat.len()
        );
    }

    #[test]
    fn denoise_step_preserves_tensor_length() {
        let schedule = LinearNoiseSchedule::new_linear(4, 1e-4, 0.02);
        let latent = vec![0.1, -0.2, 0.3];
        let predicted_noise = vec![0.05, 0.01, -0.03];
        let next = schedule.denoise_step(&latent, &predicted_noise, 2, 42);
        assert_eq!(next.len(), latent.len());
    }

    #[test]
    fn reverse_chain_stays_bounded_even_with_degenerate_noise_prediction() {
        // Worst case: the model predicts no noise. Without x0 clamping the chain
        // gain (~1/sqrt(alpha_bar_T)) amplifies the latent by two orders of
        // magnitude; with it, it stays within a few units of the data range.
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let mut latent = schedule.sample_noise(64, 0xbadc_0ffe);
        let zero_noise = vec![0.0f32; latent.len()];
        for step in (0..schedule.len()).rev() {
            latent = schedule.denoise_step(&latent, &zero_noise, step, 7 ^ step as u64);
        }
        let max_abs = latent.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            max_abs < 5.0,
            "reverse chain exploded despite x0 clamping: max |x| = {max_abs}"
        );
    }

    /// Anisotropy hunt — the defect that made every image horizontal bands. A
    /// single field was always isotropic (per-field checks passed); the field
    /// SUMMED over the reverse chain collapsed, because `seed ^ index` with a
    /// caller seed of `base ^ step` drew one field under XOR permutations of the
    /// index and the sum stopped depending on its low bits (i.e. on `x`). Walks the
    /// real recursion with ε̂=0 and asserts isotropy on 32×32. Measured: 15.6
    /// before the fix, 1.0 after (trips at 1.5).
    #[test]
    fn injected_noise_over_reverse_chain_is_isotropic() {
        const W: usize = 32;
        const H: usize = 32;
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let zero_eps = vec![0.0f32; W * H];

        // Same seed derivation as `sample_diffusion`, for one path.
        let base_seed = 0x5eed_u64;
        let mut latent = vec![0.0f32; W * H]; // start at zero: isolate the injected noise
        for step in (0..schedule.len()).rev() {
            let step_seed =
                base_seed.wrapping_add((step as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            latent = schedule.denoise_step_with_magnitude(
                &latent,
                &zero_eps,
                step,
                step_seed,
                1.0,
                PosteriorVariance::Beta,
            );
        }

        let rms =
            |d: &[f32]| (d.iter().map(|v| (v * v) as f64).sum::<f64>() / d.len() as f64).sqrt();
        let rows: Vec<f32> = (1..H)
            .flat_map(|y| (0..W).map(move |x| (y, x)))
            .map(|(y, x)| latent[y * W + x] - latent[(y - 1) * W + x])
            .collect();
        let cols: Vec<f32> = (0..H)
            .flat_map(|y| (1..W).map(move |x| (y, x)))
            .map(|(y, x)| latent[y * W + x] - latent[y * W + x - 1])
            .collect();
        let (row_rms, col_rms) = (rms(&rows), rms(&cols));
        let ratio = row_rms / col_rms.max(f64::EPSILON);

        assert!(
            ratio < 1.5 && ratio > 1.0 / 1.5,
            "injected sampler noise is anisotropic: row_diff_rms = {row_rms:.4e}, \
             col_diff_rms = {col_rms:.4e}, ratio = {ratio:.2}. The reverse chain must not \
             draw correlated fields across steps (see `gaussian_at`)."
        );
    }

    /// The narrower invariant behind the one above: two noise fields whose seeds
    /// differ must not be permutations of one another. Under `seed ^ index` the
    /// field at `seed ^ delta` was *exactly* the field at `seed` reindexed by
    /// `index ^ delta`, so this comparison was bit-for-bit equal.
    #[test]
    fn fields_of_xor_related_seeds_are_not_permutations_of_each_other() {
        let schedule = LinearNoiseSchedule::new_linear(4, 1e-4, 0.02);
        let seed = 0xabcd_1234_u64;
        for delta in [1u64, 2, 8, 64, 255] {
            let a = schedule.sample_noise(1024, seed);
            let b = schedule.sample_noise(1024, seed ^ delta);
            let permuted_matches = (0..1024).filter(|&i| a[i] == b[i ^ delta as usize]).count();
            assert!(
                permuted_matches < 1024,
                "field(seed ^ {delta}) is field(seed) permuted by `index ^ {delta}` \
                 ({permuted_matches}/1024 elements identical)"
            );
        }
    }

    /// The visualiser's "what the model believes" is only honest if it is the SAME
    /// x0 the reverse step builds its mean from. At step 0 (sigma=0,
    /// alpha_bar_prev=1) the latent is exactly `coeff_x0·x0_hat + coeff_xt·x_t`;
    /// rebuilding that from `x0_estimate` and demanding bit-equality pins both
    /// paths to one formula.
    #[test]
    fn x0_estimate_is_the_x0_the_reverse_step_uses() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let latent = schedule.sample_noise(256, 0xfeed_beef);
        // Deliberately oversized predictions: x0_hat then leaves [-1, 1] and the
        // clamp is load-bearing, so an unclamped copy would diverge here.
        let eps_hat: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();

        let step = 0usize;
        let x0 = schedule.x0_estimate(&latent, &eps_hat, step);
        assert!(
            x0.iter().all(|v| (-1.0..=1.0).contains(v)),
            "x0 estimate must be clipped to the data range"
        );

        let beta = schedule.beta(step);
        let alpha = schedule.alpha(step);
        let alpha_bar = schedule.alpha_bar(step);
        let coeff_x0 = 1.0f32.sqrt() * beta / (1.0 - alpha_bar);
        let coeff_xt = alpha.sqrt() * (1.0 - 1.0) / (1.0 - alpha_bar);

        let actual = schedule.denoise_step(&latent, &eps_hat, step, 12345);
        for (i, out) in actual.iter().enumerate() {
            let rebuilt = coeff_x0 * x0[i] + coeff_xt * latent[i];
            assert_eq!(
                *out, rebuilt,
                "step 0 output at {i} is not the posterior mean of the reported x0"
            );
        }
    }

    #[test]
    fn timestep_embedding_matches_requested_channel_count() {
        let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
        let embedding = schedule.timestep_embedding(3, 5);
        assert_eq!(embedding.len(), 5);
    }

    /// `cos(pi·tau)` (the second value of pair 0) must fall monotonically from
    /// ~+1 at t=0 to ~-1 at t=T-1 — the property that lets the network read the
    /// timestep off a single channel pair.
    #[test]
    fn timestep_embedding_low_frequency_is_monotone_over_schedule() {
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        let mut prev = f32::INFINITY;
        for step in 0..schedule.len() {
            let cos0 = schedule.timestep_embedding(step, 2)[1];
            assert!(
                cos0 <= prev + 1e-6,
                "cos(pi·tau) not monotone at step {step}"
            );
            prev = cos0;
        }
        assert!((schedule.timestep_embedding(0, 2)[1] - 1.0).abs() < 1e-4);
        assert!((schedule.timestep_embedding(255, 2)[1] + 1.0).abs() < 1e-4);
    }

    /// The consistency invariant: the CPU embedding used at inference must equal
    /// the GPU shader embedding used at training. This mirrors the WGSL
    /// `timestep_value` formula exactly; if either side changes without the
    /// other, this fails.
    #[test]
    fn timestep_embedding_matches_shader_formula() {
        fn shader_timestep_value(offset: u32, step: u32, total_steps: u32) -> f32 {
            let steps = total_steps.max(1);
            let denom = (steps.saturating_sub(1)).max(1) as f32;
            let tau = step.min(steps - 1) as f32 / denom;
            let pair_idx = offset / 2;
            let phase = tau * std::f32::consts::PI * 2.0f32.powi(pair_idx as i32);
            if offset % 2 == 0 {
                phase.sin()
            } else {
                phase.cos()
            }
        }

        for &t in &[16usize, 64, 256] {
            let schedule = LinearNoiseSchedule::new_linear(t, 1e-4, 0.02);
            for channels in [1usize, 2, 4] {
                for step in [0usize, 1, t / 3, t - 1] {
                    let cpu = schedule.timestep_embedding(step, channels);
                    for (offset, cpu_val) in cpu.iter().enumerate() {
                        let gpu = shader_timestep_value(offset as u32, step as u32, t as u32);
                        assert!(
                            (cpu_val - gpu).abs() < 1e-5,
                            "T={t} ch={channels} step={step} offset={offset}: cpu={cpu_val} gpu={gpu}"
                        );
                    }
                }
            }
        }
    }
}
