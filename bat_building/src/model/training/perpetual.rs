//! File purpose: The itinerary of an endless denoising run — where on the noise
//! schedule the sampler is, and when it should stop descending and re-noise.
//!
//! A normal sampling run has an end: it walks the reverse chain from `T-1` to
//! `0` and hands back an image. A perpetual run has none. It descends, reaches
//! its floor, throws the image part-way back into noise, and descends again —
//! forever. What changes between the two is *only* the itinerary: every step it
//! takes is the same [`crate::reverse_step`], and every re-noising is the same
//! [`LinearNoiseSchedule::add_noise`] the forward process uses.
//!
//! That itinerary is the whole of this module, and it holds no model, no GPU
//! handle and no tensor — so the thing that is hardest to eyeball (an infinite
//! loop) is the thing that is cheapest to unit-test.
//!
//! # The two regimes
//!
//! ```text
//!   t                                    t
//!   │╲                                   │
//! t_r│ ╲    ╲    ╲    ╲                t_r│╲  ╱╲  ╱╲  ╱╲  ╱
//!   │  ╲    ╲    ╲    ╲                  │ ╲╱  ╲╱  ╲╱  ╲╱
//!   │   ╲    ╲    ╲    ╲             t_r/2│
//!  0└────╲────╲────╲────╲──▶ time       0└──────────────────▶ time
//!        Wander — resolves fully,          Breathe — never resolves,
//!        then leaps back up to t_r.        oscillates around a low t.
//! ```
//!
//! `t_r` (the *renoise depth*) is the audacity dial: low `t_r` perturbs an image
//! that is nearly settled, high `t_r` erases enough of it for a metamorphosis.
//!
//! # What gets re-noised
//!
//! Always the model's clipped x̂₀ estimate from the last step — never the raw
//! latent. [`LinearNoiseSchedule::add_noise`] implements the forward process
//! `x_t = sqrt(ᾱ_t)·x₀ + sqrt(1-ᾱ_t)·ε`, which assumes a *clean* image in the
//! data range; at a floor above zero the latent is not one, and feeding it in
//! would inflate the signal term by `1/sqrt(ᾱ_floor)` every cycle. x̂₀ is clipped
//! to `[-1, 1]` by construction, so it is always a legitimate `x₀`.
//!
//! This costs nothing at a floor of zero: at `t = 0` the sampler has `σ = 0` and
//! `ᾱ_prev = 1`, so its output *is* `x̂₀` exactly (see
//! `x0_estimate_is_the_x0_the_reverse_step_uses`). Wander therefore re-noises
//! precisely the image it just finished showing.

use serde::{Deserialize, Serialize};

/// Which way the run wanders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerpetualRegime {
    /// Descend to `t = 0` — a fully resolved image — then leap back to `t_r`.
    Wander,
    /// Descend only half-way to `t_r/2`, then back up. The image never settles.
    Breathe,
}

impl Default for PerpetualRegime {
    fn default() -> Self {
        Self::Wander
    }
}

impl PerpetualRegime {
    pub fn label(self) -> &'static str {
        match self {
            PerpetualRegime::Wander => "errance",
            PerpetualRegime::Breathe => "respiration",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            PerpetualRegime::Wander => PerpetualRegime::Breathe,
            PerpetualRegime::Breathe => PerpetualRegime::Wander,
        }
    }

    /// Parses the CLI spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "wander" | "errance" => Some(PerpetualRegime::Wander),
            "breathe" | "respiration" => Some(PerpetualRegime::Breathe),
            _ => None,
        }
    }
}

/// Shallowest renoise depth on offer. Below this a cycle is a single step and
/// the image stops moving.
pub const MIN_RENOISE_DEPTH: usize = 4;
/// How far one press of the depth key moves `t_r`.
pub const RENOISE_DEPTH_STEP: usize = 8;

/// Odd increments for the two independent seed streams a cycle needs. Distinct
/// constants so a cycle's descent noise and its re-noising field are unrelated;
/// both are avalanched by `gaussian_at` before any pixel index is folded in.
const DESCENT_STREAM: u64 = 0x9e37_79b9_7f4a_7c15;
const RENOISE_STREAM: u64 = 0xbf58_476d_1ce4_e5b9;

/// What the drift asks the caller to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftAction {
    /// Take one reverse step at `diffusion_step`, seeded from `path_seed`.
    Descend {
        diffusion_step: usize,
        path_seed: u64,
    },
    /// The floor is reached. Re-noise the last x̂₀ up to `to_step` with `seed`,
    /// which opens a new cycle.
    Renoise { to_step: usize, seed: u64 },
}

/// Where an endless run currently is on the schedule.
#[derive(Debug, Clone)]
pub struct PerpetualDrift {
    steps: usize,
    regime: PerpetualRegime,
    depth: usize,
    t: usize,
    cycle: usize,
    base_seed: u64,
    /// Set once the floor's step has been taken; the next action re-noises.
    at_floor: bool,
}

impl PerpetualDrift {
    /// Opens a run at the top of the schedule.
    ///
    /// The first descent always starts from `T-1` whatever `depth` says: the run
    /// begins on pure noise, and `depth` only governs how far back up each
    /// *subsequent* cycle throws the image.
    pub fn new(steps: usize, regime: PerpetualRegime, depth: usize, base_seed: u64) -> Self {
        let steps = steps.max(1);
        let mut drift = Self {
            steps,
            regime,
            depth: MIN_RENOISE_DEPTH,
            t: steps - 1,
            cycle: 0,
            base_seed,
            at_floor: false,
        };
        drift.set_depth(depth);
        drift
    }

    pub fn regime(&self) -> PerpetualRegime {
        self.regime
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn cycle(&self) -> usize {
        self.cycle
    }

    /// The timestep the next reverse step will be taken at.
    pub fn current_step(&self) -> usize {
        self.t
    }

    /// Deepest renoise the schedule allows.
    pub fn max_depth(&self) -> usize {
        self.steps.saturating_sub(1).max(MIN_RENOISE_DEPTH)
    }

    /// Timestep a cycle re-noises back up to.
    pub fn ceiling(&self) -> usize {
        self.depth
    }

    /// Lowest timestep a cycle descends to — the regime *is* this number.
    pub fn floor(&self) -> usize {
        match self.regime {
            PerpetualRegime::Wander => 0,
            PerpetualRegime::Breathe => self.depth / 2,
        }
    }

    /// Clamps and applies a new renoise depth.
    ///
    /// A deeper `t_r` takes effect at the next re-noising (the ceiling is only
    /// read there). A shallower one can leave `t` already at or below the new
    /// floor — [`Self::step`] compares with `<=`, so that simply ends the cycle
    /// on the next call instead of descending past it.
    pub fn set_depth(&mut self, depth: usize) {
        self.depth = depth.clamp(MIN_RENOISE_DEPTH, self.max_depth());
    }

    /// Moves the depth by `delta` notches of [`RENOISE_DEPTH_STEP`].
    pub fn nudge_depth(&mut self, delta: i32) {
        let shift = RENOISE_DEPTH_STEP.saturating_mul(delta.unsigned_abs() as usize);
        let next = if delta >= 0 {
            self.depth.saturating_add(shift)
        } else {
            self.depth.saturating_sub(shift)
        };
        self.set_depth(next);
    }

    pub fn toggle_regime(&mut self) {
        self.regime = self.regime.toggle();
    }

    /// Restarts the drift from pure noise under a new seed. The caller is
    /// expected to redraw its latent from the same seed.
    pub fn reseed(&mut self, base_seed: u64) {
        self.base_seed = base_seed;
        self.t = self.steps - 1;
        self.cycle = 0;
        self.at_floor = false;
    }

    /// The seed the run's initial latent should be drawn from.
    pub fn initial_noise_seed(&self) -> u64 {
        self.base_seed ^ 0xa5a5_5a5a_0123_4567
    }

    /// Yields the next action and advances.
    pub fn step(&mut self) -> DriftAction {
        if self.at_floor {
            self.at_floor = false;
            self.cycle += 1;
            self.t = self.ceiling();
            return DriftAction::Renoise {
                to_step: self.t,
                seed: self.stream_seed(RENOISE_STREAM),
            };
        }

        let diffusion_step = self.t;
        let action = DriftAction::Descend {
            diffusion_step,
            path_seed: self.stream_seed(DESCENT_STREAM),
        };
        if diffusion_step <= self.floor() {
            self.at_floor = true;
        } else {
            self.t = diffusion_step - 1;
        }
        action
    }

    fn stream_seed(&self, stream: u64) -> u64 {
        self.base_seed
            .wrapping_add((self.cycle as u64 + 1).wrapping_mul(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEPS: usize = 256;

    /// Collects the timesteps of the next `count` descents, closing over the
    /// re-noisings so a test can read the trajectory as a sawtooth.
    fn walk(drift: &mut PerpetualDrift, count: usize) -> (Vec<usize>, Vec<usize>) {
        let mut descents = Vec::new();
        let mut renoise_targets = Vec::new();
        for _ in 0..count {
            match drift.step() {
                DriftAction::Descend { diffusion_step, .. } => descents.push(diffusion_step),
                DriftAction::Renoise { to_step, .. } => renoise_targets.push(to_step),
            }
        }
        (descents, renoise_targets)
    }

    /// The run opens on pure noise, so the first descent must start at the top
    /// of the schedule even when `t_r` is shallow — otherwise the first image is
    /// a denoised *nothing*, and the whole piece starts from a grey field.
    #[test]
    fn the_first_descent_starts_at_the_top_of_the_schedule() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 16, 7);
        let (descents, _) = walk(&mut drift, 1);
        assert_eq!(descents, vec![STEPS - 1]);
    }

    #[test]
    fn wander_descends_to_zero_then_renoises_to_the_depth() {
        let depth = 12;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 7);

        // The opening descent is the full schedule: T-1 .. 0, then a renoise.
        let (descents, renoises) = walk(&mut drift, STEPS + 1);
        assert_eq!(descents.first().copied(), Some(STEPS - 1));
        assert_eq!(descents.last().copied(), Some(0));
        assert_eq!(descents.len(), STEPS);
        assert_eq!(renoises, vec![depth]);
        assert_eq!(drift.cycle(), 1);

        // Every later cycle is depth .. 0 followed by a renoise back to depth.
        let (descents, renoises) = walk(&mut drift, depth + 2);
        assert_eq!(descents, (0..=depth).rev().collect::<Vec<_>>());
        assert_eq!(renoises, vec![depth]);
    }

    /// The point of breathing is that the image never settles: a cycle that
    /// touched `t = 0` would resolve it, which is wander's job.
    #[test]
    fn breathe_never_reaches_a_fully_resolved_image() {
        let depth = 32;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Breathe, depth, 7);
        // Skip the opening descent from pure noise, which does pass through 0's
        // neighbourhood only as far as the floor.
        let (descents, _) = walk(&mut drift, STEPS * 2);
        let floor = depth / 2;
        assert!(
            descents.iter().all(|&t| t >= floor),
            "breathing descended below its floor {floor}: min = {:?}",
            descents.iter().min()
        );
        assert_eq!(descents.iter().max().copied(), Some(STEPS - 1));
    }

    #[test]
    fn depth_is_clamped_to_the_schedule() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 1, 7);
        assert_eq!(drift.depth(), MIN_RENOISE_DEPTH);

        drift.set_depth(usize::MAX);
        assert_eq!(drift.depth(), STEPS - 1);

        drift.nudge_depth(1);
        assert_eq!(drift.depth(), STEPS - 1, "cannot climb past the schedule");

        drift.set_depth(MIN_RENOISE_DEPTH);
        drift.nudge_depth(-1);
        assert_eq!(
            drift.depth(),
            MIN_RENOISE_DEPTH,
            "cannot sink below the floor"
        );
    }

    /// Shrinking `t_r` below the current position must end the cycle, not send
    /// the run descending past a floor it has already crossed.
    #[test]
    fn shrinking_the_depth_below_the_current_step_ends_the_cycle() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Breathe, MIN_RENOISE_DEPTH, 7);
        // The opening descent runs from the top, so walk well below half the
        // schedule before moving the floor up past the current position.
        walk(&mut drift, 200);
        let t = drift.current_step();
        assert!(t < STEPS / 2);

        drift.set_depth(drift.max_depth());
        assert!(drift.floor() > t, "test needs a floor above t={t}");

        // One more descent (at t), then the cycle must close.
        let (descents, renoises) = walk(&mut drift, 2);
        assert_eq!(descents, vec![t]);
        assert_eq!(renoises, vec![drift.ceiling()]);
    }

    /// Two cycles that drew the same noise would loop, not drift — the image
    /// would come back to where it was. Descent and re-noising must also not
    /// share a field: they are applied to the same tensor one after the other.
    #[test]
    fn every_cycle_draws_from_a_fresh_and_distinct_pair_of_streams() {
        let mut drift =
            PerpetualDrift::new(STEPS, PerpetualRegime::Wander, MIN_RENOISE_DEPTH, 0x5eed);
        let mut seeds = std::collections::HashSet::new();
        for _ in 0..(STEPS + 8 * (MIN_RENOISE_DEPTH + 2)) {
            let seed = match drift.step() {
                DriftAction::Descend { path_seed, .. } => path_seed,
                DriftAction::Renoise { seed, .. } => seed,
            };
            seeds.insert(seed);
        }
        // One descent seed and one renoise seed per cycle, all distinct.
        assert_eq!(seeds.len(), drift.cycle() * 2 + 1);
    }

    #[test]
    fn reseeding_returns_to_the_top_of_the_schedule() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 32, 7);
        walk(&mut drift, STEPS + 5);
        assert!(drift.cycle() > 0);

        drift.reseed(0x1234);
        assert_eq!(drift.cycle(), 0);
        assert_eq!(drift.current_step(), STEPS - 1);
        let (descents, _) = walk(&mut drift, 1);
        assert_eq!(descents, vec![STEPS - 1]);
    }

    #[test]
    fn toggling_the_regime_swaps_the_floor() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 40, 7);
        assert_eq!(drift.floor(), 0);
        drift.toggle_regime();
        assert_eq!(drift.regime(), PerpetualRegime::Breathe);
        assert_eq!(drift.floor(), 20);
        drift.toggle_regime();
        assert_eq!(drift.floor(), 0);
    }
}
