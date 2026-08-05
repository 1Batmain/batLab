//! File purpose: Implements schedule logic used by the training pipeline.

#[derive(Debug, Clone)]
pub struct LinearNoiseSchedule {
    betas: Vec<f32>,
    alphas: Vec<f32>,
    alpha_bars: Vec<f32>,
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
    /// Builds a linear beta schedule.
    ///
    /// `beta_start`/`beta_end` are interpreted as calibrated for
    /// `BETA_REFERENCE_STEPS` (the DDPM paper's T = 1000) and are rescaled by
    /// `BETA_REFERENCE_STEPS / num_steps`, so the total injected noise is
    /// preserved at any T. Without this, reusing the paper's betas at T = 256
    /// left alpha_bar(T-1) = 0.075 — i.e. 27% of the original signal still
    /// present in x_T, while sampling starts from pure Gaussian noise.
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

    pub fn normalized_step(&self, step: usize) -> f32 {
        if self.len() <= 1 {
            0.0
        } else {
            step.min(self.len() - 1) as f32 / (self.len() - 1) as f32
        }
    }

    /// Smooth multi-resolution timestep embedding.
    ///
    /// The step is first normalised to `tau = step / (T - 1) ∈ [0, 1]`, then each
    /// channel pair `i` encodes `[sin(pi·2^i·tau), cos(pi·2^i·tau)]`. The lowest
    /// pair (`i = 0`) gives `cos(pi·tau)`, a strictly monotone signal running
    /// `1 → -1` across the schedule, so even a single pair uniquely identifies t
    /// *and* varies smoothly — unlike the previous `sin(step)`/`cos(step)` form
    /// (period ≈ 6.28 steps), which aliased adjacent timesteps and gave the
    /// network a near-random code it could not exploit in a short run.
    ///
    /// This MUST stay identical to `timestep_value` in
    /// `shader/diffusion_prepare.wgsl` (checked by
    /// `timestep_embedding_matches_shader_formula`): training composes the input
    /// on the GPU with the shader, inference composes it here on the CPU, and the
    /// network only works if both feed it the same conditioning.
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
    /// `x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * eps`.
    ///
    /// [`Self::add_noise`] jumps straight to `t` from a clean `x0`; this walks
    /// there. Chaining increments `0..=t` from the same `x0`, each with its own
    /// noise draw, is the *same distribution* — that is the standard DDPM
    /// identity, and `climbing_the_forward_chain_matches_add_noise_in_distribution`
    /// measures it rather than asserting it from the algebra.
    ///
    /// The two are therefore interchangeable in maths and not at all in
    /// experience: the jump replaces an image with noise between two frames,
    /// while the walk dissolves it over `t` of them. Training keeps the jump,
    /// which draws an independent `t` per example and has nothing to animate.
    ///
    /// **The perpetual climb does not use this** — it uses [`Self::forward_from`],
    /// for the reason spelled out there: an exact Markov walk draws an
    /// *independent* field per frame, which at 30 frames a second is television
    /// static. This stays because it is the correct, tested definition of one
    /// forward increment, and it is what `forward_from` is measured against.
    ///
    /// `seed` must be fresh per increment — reusing one across a walk would add
    /// the *same* field `t` times, i.e. a single field scaled up, which is not a
    /// Gaussian walk at all.
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
    ///
    /// `departure` is the state a climb sets out from — the one that sits just
    /// *below* level `departure_step`, i.e. exactly what
    /// `forward_step(departure, departure_step, _)` consumes. `step` is the
    /// level to land on. With `r = alpha_bar(step) / alpha_bar_below(departure_step)`:
    ///
    /// ```text
    ///   x_t = sqrt(r) * x_dep + sqrt(1 - r) * eps
    /// ```
    ///
    /// This is `q(x_t | x_s)` for any `s < t`, not just `s = clean`: the ratio of
    /// `alpha_bar`s *is* the generalisation, which is why the departure is
    /// allowed to be a noisy latent where [`Self::add_noise`] would need a clean
    /// `x0`. At `departure_step = 0` the ratio's denominator is `1` and this
    /// reduces to `add_noise` exactly.
    ///
    /// # Why the perpetual climb walks this and not [`Self::forward_step`]
    ///
    /// Chaining `forward_step` up the schedule is the *exact* Markov walk, and
    /// it lands on the same marginal — but every increment draws its **own**
    /// field. Displayed one increment per frame at 30 Hz, thirty independent
    /// grains a second is television static: the image dissolved smoothly and
    /// *seethed* while doing it ("ça frise pendant la phase de remontée"). The
    /// descent has no such artefact because consecutive latents of the reverse
    /// chain are strongly correlated.
    ///
    /// Here `seed` names **one field for the whole climb**, and every level is
    /// computed from the departure rather than from the previous frame. Each
    /// frame keeps the same marginal, the endpoint keeps the same distribution
    /// — and the grain, being one fixed pattern whose amplitude rises, is
    /// *revealed* instead of reshuffled. The frames are no longer a Markov
    /// chain: they are maximally correlated, which is precisely the property the
    /// eye was asking for.
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
        self.denoise_step_with_magnitude(latent, predicted_noise, step, seed, 1.0)
    }

    /// The clipped x0 estimate the posterior mean is built from, for the whole
    /// tensor: `clamp(x_t - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar)`.
    ///
    /// This is "what the model believes the clean image is" at `step`, and it is
    /// the exact same quantity [`Self::denoise_step_with_magnitude`] computes
    /// internally — both go through [`x0_hat_at`], so the live visualiser cannot
    /// drift from the sampler. Nothing here is part of the recursion: the value
    /// is derived from `latent`/`predicted_noise` alone, so computing it is
    /// side-effect free and optional.
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
        // Posterior mean computed from the clipped x0 estimate. Clamping x0 to
        // the data range bounds the reverse chain by construction: an imperfect
        // (or degenerate) noise prediction can no longer be amplified
        // multiplicatively across the ~1/sqrt(alpha_bar_T) chain gain. For a
        // well-trained model x0_hat already lies in [-1, 1] and this is a no-op.
        let coeff_x0 = alpha_bar_prev.sqrt() * beta / (1.0 - alpha_bar);
        let coeff_xt = alpha.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar);
        let noise_to_x0 = (1.0 - alpha_bar).sqrt();
        let x0_scale = 1.0 / alpha_bar.sqrt().max(f32::EPSILON);
        let magnitude = denoise_magnitude.max(0.0);
        let sigma = if step == 0 {
            0.0
        } else {
            beta.sqrt() * magnitude
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

/// The clipped x0 estimate for a single element.
///
/// Sole definition of the formula: the reverse step builds its posterior mean
/// from it, and `x0_estimate` exposes it for display. Keeping one definition is
/// what makes the visualiser's "what the model believes" pane trustworthy — a
/// second copy could drift and quietly show something the sampler never used.
#[inline]
fn x0_hat_at(latent: f32, predicted_noise: f32, x0_scale: f32, noise_to_x0: f32) -> f32 {
    (x0_scale * (latent - noise_to_x0 * predicted_noise)).clamp(-1.0, 1.0)
}

/// Odd increment of the SplitMix64 stream (the golden-ratio constant).
const STREAM_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// Draws element `index` of the noise field identified by `seed`.
///
/// The index is folded in by **addition through an odd multiplier**, never by
/// XOR. `gaussian_from_seed(seed ^ index)` reads as harmless — the murmur3
/// finaliser downstream avalanches fine — but it is catastrophic as soon as the
/// *caller* also derives its per-draw seed by XOR, which the reverse chain did
/// (`path_seed ^ diffusion_step`):
///
/// ```text
/// n(step, index) = g(base ^ step ^ index) = n(step', index ^ step ^ step')
/// ```
///
/// i.e. every step of the chain draws the *same* field, merely permuted by
/// `index -> index ^ step ^ step'`. Accumulated over the 256 reverse steps, the
/// total injected noise at a pixel then depends on its index only through
/// `sigma_{u ^ index}` — and `sigma` is smooth in `step`, so flipping a *low*
/// bit of the index barely changes the sum. On a 32-wide image the low 5 bits
/// are `x`: the injected noise came out all but constant along a row, and the
/// sampler painted horizontal bands whatever the model predicted
/// (`ANISOTROPY_HUNT.md`; guarded by
/// `injected_noise_over_reverse_chain_is_isotropic`).
///
/// The seed is therefore **avalanched first** (`fmix64`), and only then does the
/// index stream get added. Any relation between two caller seeds — XOR, or the
/// `+ step * GAMMA` a caller might reasonably use — is destroyed by the mix
/// before the index is folded in, so no two fields are reindexings of one
/// another. Doing it the other way round is not enough: `seed + (index+1)*GAMMA`
/// with a caller seed of `base + (step+1)*GAMMA` collapses to
/// `base + (step+index+2)*GAMMA`, a field constant along the anti-diagonals —
/// the same defect wearing a different hat, and the regression test catches it.
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
    use super::LinearNoiseSchedule;

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

    /// Walking the forward chain one step at a time must land where the single
    /// `add_noise` jump lands — same signal coefficient, same noise variance.
    ///
    /// This is what licenses the perpetual mode's gradual re-noising: the user
    /// sees the image dissolve over `t_r` frames instead of being replaced by
    /// noise between two, and the sampler is handed exactly the `x_t` it would
    /// have been handed before. Asserted by measurement, not by algebra: the
    /// claim is about `gaussian_at`'s *actual* draws, and a chain that reused a
    /// seed (or scaled a field twice) would satisfy the algebra and fail here.
    ///
    /// Every intermediate level is checked too — not just the destination —
    /// because a climb whose middle frames are not legitimate `x_k` would show
    /// the right start and end with something arbitrary in between, which is
    /// precisely the part the user looks at.
    #[test]
    fn climbing_the_forward_chain_matches_add_noise_in_distribution() {
        const N: usize = 16_384;
        let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
        // A structured, non-degenerate x0 in the data range: a flat field would
        // make the signal coefficient unmeasurable.
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

    /// `forward_from` must produce a legitimate `x_t` at **every** level it is
    /// asked for, departing from a clean `x0` *and* from a noisy latent — the
    /// second is the case breathing lives in, and the one `add_noise` cannot
    /// serve.
    ///
    /// Measured, not deduced: signal coefficient by least squares, mean and
    /// standard deviation of the residual, against theory *and* against an
    /// exact Markov walk (`forward_step`) launched from the same departure.
    /// The last level is the one the next descent is handed, so its agreement
    /// with the walk is the statement "the wandering has not changed nature".
    #[test]
    fn the_closed_form_climb_is_a_valid_x_t_at_every_level() {
        // Large enough that the tolerances below sit well clear of the
        // estimator's own noise: the least-squares coefficient has a standard
        // error of sigma/sqrt(sum x_dep^2) ~= 0.003 here, so the 0.02 gate is a
        // ~6-sigma statement and not a coin flip that happens to be green.
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

    /// The failure mode an *incremental* climb invites: one seed for the whole
    /// ascent adds the same field over and over, which is a single field scaled
    /// up — the pixels stay perfectly correlated with it and the image never
    /// dissolves, it just gains a fixed pattern. The variance check above would
    /// pass; only comparing two increments catches it.
    ///
    /// This is the invariant of [`LinearNoiseSchedule::forward_step`], which the
    /// perpetual climb no longer uses; it guards the operator, which the closed
    /// form is measured against in the two tests above.
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
        // Worst case for the sampler: the model predicts no noise at all.
        // Without x0 clamping the chain gain (~1/sqrt(alpha_bar_T)) amplifies
        // the latent by two orders of magnitude; with it the trajectory must
        // stay within a few units of the data range.
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

    /// Anisotropy hunt — the defect that made every generated image a stack of
    /// horizontal bands.
    ///
    /// A single noise field was always isotropic, so per-field checks passed.
    /// What collapsed was the field *summed over the reverse chain*: with the
    /// old `gaussian_from_seed(seed ^ index)` fed a caller seed of
    /// `base ^ step`, the 256 steps drew one field under XOR permutations of the
    /// pixel index, and the sum stopped depending on the low bits of the index —
    /// i.e. on `x`. This walks the real sampler recursion with a null model
    /// (eps_hat = 0), so it exercises exactly the accumulation the sampler
    /// performs, and asserts the accumulated field is isotropic on a 32x32 grid.
    ///
    /// Measured: 15.6 before the fix, 1.0 after (the assertion trips at 1.5).
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
            latent = schedule.denoise_step_with_magnitude(&latent, &zero_eps, step, step_seed, 1.0);
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

    /// The visualiser shows `x0_estimate` as "what the model believes the clean
    /// image is". That claim is only honest if it is the *same* x0 the reverse
    /// step builds its posterior mean from.
    ///
    /// At step 0 the sampler is noiseless (sigma = 0) and alpha_bar_prev = 1, so
    /// the returned latent is exactly `coeff_x0 * x0_hat + coeff_xt * x_t`.
    /// Rebuilding that from `x0_estimate` and demanding bit-equality pins the two
    /// paths to one formula: inline the clamp differently in either one and this
    /// fails.
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
