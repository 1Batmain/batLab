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

    pub fn normalized_step(&self, step: usize) -> f32 {
        if self.len() <= 1 {
            0.0
        } else {
            step.min(self.len() - 1) as f32 / (self.len() - 1) as f32
        }
    }

    pub fn timestep_embedding(&self, step: usize, channels: usize) -> Vec<f32> {
        if channels == 0 {
            return Vec::new();
        }

        let step_value = step.min(self.len().saturating_sub(1)) as f32;
        let mut values = Vec::with_capacity(channels);
        let half = channels.div_ceil(2);

        for i in 0..half {
            let denom = (half.saturating_sub(1)).max(1) as f32;
            let exponent = i as f32 / denom;
            let frequency = 1.0 / 10000.0f32.powf(exponent);
            let phase = step_value * frequency;
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
            let sample_noise = gaussian_from_seed(seed ^ index as u64);
            noise.push(sample_noise);
            noisy.push(signal_scale * *value + noise_scale * sample_noise);
        }

        (noisy, noise)
    }

    pub fn sample_noise(&self, len: usize, seed: u64) -> Vec<f32> {
        (0..len)
            .map(|index| gaussian_from_seed(seed ^ index as u64))
            .collect()
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
        let coeff = beta / (1.0 - alpha_bar).sqrt();
        let mean_scale = 1.0 / alpha.sqrt();
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
                let mean = mean_scale * (*value - coeff * *noise);
                if sigma == 0.0 {
                    mean
                } else {
                    mean + sigma * gaussian_from_seed(seed ^ index as u64)
                }
            })
            .collect()
    }
}

fn unit_from_seed(mut value: u64) -> f32 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51afd7ed558ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ceb9fe1a85ec53);
    value ^= value >> 33;
    let normalized = (value >> 40) as u32;
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

    #[test]
    fn denoise_step_preserves_tensor_length() {
        let schedule = LinearNoiseSchedule::new_linear(4, 1e-4, 0.02);
        let latent = vec![0.1, -0.2, 0.3];
        let predicted_noise = vec![0.05, 0.01, -0.03];
        let next = schedule.denoise_step(&latent, &predicted_noise, 2, 42);
        assert_eq!(next.len(), latent.len());
    }

    #[test]
    fn timestep_embedding_matches_requested_channel_count() {
        let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
        let embedding = schedule.timestep_embedding(3, 5);
        assert_eq!(embedding.len(), 5);
    }
}
