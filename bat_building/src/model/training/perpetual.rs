//! File purpose: The itinerary of an endless denoising run — where on the noise
//! schedule the sampler is, and when it should stop descending and re-noise.
//!
//! A normal sampling run has an end: it walks the reverse chain from `T-1` to
//! `0` and hands back an image. A perpetual run has none. It descends, reaches
//! its floor, walks the *forward* chain back up to `t_r`, and descends again —
//! forever. What changes between the two is *only* the itinerary: every step
//! down is the same [`crate::reverse_step`], and every step up is one level of
//! the forward process in closed form,
//! [`LinearNoiseSchedule::forward_from`](crate::LinearNoiseSchedule::forward_from).
//!
//! That itinerary is the whole of this module, and it holds no model, no GPU
//! handle and no tensor — so the thing that is hardest to eyeball (an infinite
//! loop) is the thing that is cheapest to unit-test.
//!
//! # The two regimes
//!
//! ```text
//!   t                                    t
//!   │╲  ╱╲  ╱╲  ╱╲  ╱                    │
//! t_r│ ╲╱  ╲╱  ╲╱  ╲╱                 t_r│╲  ╱╲  ╱╲  ╱╲  ╱
//!   │                                    │ ╲╱  ╲╱  ╲╱  ╲╱
//!   │                               t_r/2│
//!  0└──────────────────▶ time          0└──────────────────▶ time
//!     Wander — resolves fully at 0,       Breathe — never resolves,
//!     then climbs back to t_r.            oscillates around a low t.
//! ```
//!
//! `t_r` (the *renoise depth*) is the audacity dial: low `t_r` perturbs an image
//! that is nearly settled, high `t_r` erases enough of it for a metamorphosis.
//! Both regimes are triangles, not sawtooths: the climb takes exactly as many
//! frames as the descent it undoes.
//!
//! # Why the climb is walked and not jumped
//!
//! The first version leapt: one [`LinearNoiseSchedule::add_noise`] took the
//! settled image straight to `t_r` between two frames. Distributionally that is
//! the same destination, and to watch it was a slap — "an immense quantity of
//! noise, all at once", which is no way to hold a contemplative image. Walking
//! the same distance one level at a time, one visualiser frame per level,
//! dissolves the image over the same number of frames it took to resolve. Same
//! maths, opposite experience.
//!
//! # Why the climb is walked in closed form and not as a Markov chain
//!
//! The walked version was first written as the exact forward chain: one
//! [`LinearNoiseSchedule::forward_step`] per frame. That is the textbook
//! process and it is correct — and it *shimmered*. Each increment draws its own
//! independent field, so at thirty frames a second the climb showed thirty
//! unrelated grains a second: television static laid over a dissolving image.
//! The descent has no such artefact, because consecutive latents of the reverse
//! chain are strongly correlated; the climb crackled because its noise was not.
//!
//! The climb therefore runs
//! [`forward_from`](crate::LinearNoiseSchedule::forward_from): every level is
//! computed from the *departure* state with **one field per cycle**, drawn once
//! and then revealed as its amplitude rises. Every frame keeps the marginal it
//! had before and the destination keeps its distribution — only the correlation
//! *between* frames changes, from zero to one. See `CLIMB_COHERENCE.md`.
//!
//! # What the climb starts from
//!
//! The latent the descent left, wherever it left it — never a re-interpretation
//! of it as a clean image. This is what the closed form from an intermediate
//! state exists for: `x_t = sqrt(ᾱ_t/ᾱ_dep)·x_dep + sqrt(1 - ᾱ_t/ᾱ_dep)·ε` is
//! valid from *any* legitimate `x_dep`, being `q(x_t | x_dep)` of the forward
//! process, whereas `add_noise(x, t)` implements
//! `x_t = sqrt(ᾱ_t)·x₀ + sqrt(1-ᾱ_t)·ε` and is only valid when `x` really is a
//! clean `x₀` — feed it a latent that already carries noise and the signal
//! decays by an extra `sqrt(ᾱ_floor)` every cycle. The ratio of `ᾱ`s is exactly
//! the correction that makes the departure's own noise level accounted for.
//!
//! So the climb runs from `floor` to `t_r`, and the two regimes both come out
//! right for the same reason:
//!
//! - **Wander** floors at `t = 0`, where the sampler has `σ = 0` and `ᾱ_prev = 1`
//!   so its output *is* the clipped x̂₀ (see
//!   `x0_estimate_is_the_x0_the_reverse_step_uses`). The climb therefore starts
//!   from a genuine, clipped `x₀` — exactly the image the cycle just showed.
//! - **Breathe** floors above zero, and the latent there is *not* a clean image
//!   — which is the point of the regime. Climbing from it keeps the promise
//!   that the image never fully resolves; starting the climb from x̂₀ instead
//!   would flash the resolved image on the way up and make breathing a slower
//!   wander.
//!
//! The caller therefore snapshots the latent on the increment that reports
//! `opens_cycle` and feeds that same snapshot to every level of the climb —
//! [`DriftAction::Climb`] names the level it departed from so the schedule can
//! form the right `ᾱ` ratio.

use serde::{Deserialize, Serialize};

/// Which way the run wanders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerpetualRegime {
    /// Descend to `t = 0` — a fully resolved image — then climb back to `t_r`.
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

/// How fast the climb runs relative to the descent.
///
/// `1.0` — the climb takes exactly as long as the descent it undoes, which is
/// the symmetry the eye reads as breathing. It is affordable because a climb
/// increment is pure arithmetic: unlike a reverse step it never calls the
/// model, so the only thing pacing it is the tempo. Raise it to hurry the
/// dissolve, lower it to draw it out.
pub const CLIMB_TEMPO_RATIO: f32 = 1.0;

/// Odd increments for the independent seed streams a cycle needs. Distinct
/// constants so a cycle's descent noise and its climb field are unrelated;
/// both are avalanched by `gaussian_at` before any pixel index is folded in.
const DESCENT_STREAM: u64 = 0x9e37_79b9_7f4a_7c15;
const RENOISE_STREAM: u64 = 0xbf58_476d_1ce4_e5b9;

/// Which way the run is currently moving on the schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftPhase {
    /// Walking the reverse chain down: the image resolving.
    Descent,
    /// Walking the forward chain back up: the image dissolving.
    Climb,
}

impl DriftPhase {
    /// What the panel says the run is doing. A frame in which the image is
    /// coming apart is not a malfunction, but it looks like one unless the
    /// instrument says so.
    pub fn label(self) -> &'static str {
        match self {
            DriftPhase::Descent => "descente",
            DriftPhase::Climb => "remontée",
        }
    }
}

/// What the drift asks the caller to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftAction {
    /// Take one reverse step at `diffusion_step`, seeded from `path_seed`.
    Descend {
        diffusion_step: usize,
        path_seed: u64,
    },
    /// Carry the climb's departure state up to level `forward_step`.
    ///
    /// The caller applies
    /// `schedule.forward_from(departure, departure_step, forward_step, cycle_seed)`,
    /// where `departure` is the latent it snapshotted on the increment that
    /// reported `opens_cycle` — never a re-interpretation of it as a clean
    /// image, see the module header.
    Climb {
        /// The level this frame lands on. Rises by one per action, `departure_step`
        /// through the ceiling.
        forward_step: usize,
        /// The level the climb set out from — the departure state sits just
        /// below it. Constant for the whole climb, and what fixes the `ᾱ` ratio.
        departure_step: usize,
        /// The climb's **single** noise field, drawn once per cycle. Every level
        /// of one climb reveals more of this same field; the next cycle draws a
        /// different one.
        cycle_seed: u64,
        /// Set on the first increment of a climb: the cycle just closed, and
        /// the latent is the image it settled on. The moment to save a frame —
        /// and the moment to snapshot the departure.
        opens_cycle: bool,
    },
}

/// Where an endless run currently is on the schedule.
#[derive(Debug, Clone)]
pub struct PerpetualDrift {
    steps: usize,
    regime: PerpetualRegime,
    depth: usize,
    /// The level the next action works at — a timestep while descending, a
    /// forward increment while climbing.
    t: usize,
    cycle: usize,
    base_seed: u64,
    phase: DriftPhase,
    /// The level the climb in progress set out from — its departure state sits
    /// just below this. Fixed for the whole climb, so every level is formed
    /// from the same departure and the same field.
    departure_step: usize,
    /// Set when a cycle closes, cleared by the climb increment that reports it.
    opens_cycle: bool,
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
            phase: DriftPhase::Descent,
            departure_step: 0,
            opens_cycle: false,
        };
        drift.set_depth(depth);
        drift
    }

    pub fn regime(&self) -> PerpetualRegime {
        self.regime
    }

    /// Whether the run is currently resolving the image or dissolving it.
    pub fn phase(&self) -> DriftPhase {
        self.phase
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn cycle(&self) -> usize {
        self.cycle
    }

    /// The level the next action works at: the timestep of the next reverse
    /// step while descending, the level the next forward increment reaches
    /// while climbing.
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
    /// A deeper `t_r` takes effect at the next climb (the ceiling is only read
    /// there). A shallower one can leave `t` already at or below the new floor
    /// — [`Self::step`] compares with `<=`, so that simply ends the descent on
    /// the next call instead of walking past it; and a climb already above the
    /// new ceiling ends on its next increment rather than unwinding.
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
        self.phase = DriftPhase::Descent;
        self.departure_step = 0;
        self.opens_cycle = false;
    }

    /// The seed the run's initial latent should be drawn from.
    pub fn initial_noise_seed(&self) -> u64 {
        self.base_seed ^ 0xa5a5_5a5a_0123_4567
    }

    /// Yields the next action and advances.
    pub fn step(&mut self) -> DriftAction {
        match self.phase {
            DriftPhase::Descent => {
                let diffusion_step = self.t;
                let action = DriftAction::Descend {
                    diffusion_step,
                    path_seed: self.stream_seed(DESCENT_STREAM),
                };
                if diffusion_step <= self.floor() {
                    // A reverse step at `t` leaves the latent one level below,
                    // so the forward chain resumes at `t` itself: the climb
                    // re-walks precisely the ground the descent just covered.
                    self.phase = DriftPhase::Climb;
                    self.cycle += 1;
                    self.opens_cycle = true;
                    self.t = diffusion_step;
                    // Where the whole climb is formed from: the latent this
                    // very step is about to produce sits just below here.
                    self.departure_step = diffusion_step;
                } else {
                    self.t = diffusion_step - 1;
                }
                action
            }
            DriftPhase::Climb => {
                let forward_step = self.t;
                let action = DriftAction::Climb {
                    forward_step,
                    departure_step: self.departure_step,
                    cycle_seed: self.stream_seed(RENOISE_STREAM),
                    opens_cycle: std::mem::take(&mut self.opens_cycle),
                };
                if forward_step >= self.ceiling() {
                    // The latent now sits at `forward_step`; that is where the
                    // reverse chain has to be picked up, ceiling or not — the
                    // user may have moved `t_r` mid-climb.
                    self.phase = DriftPhase::Descent;
                } else {
                    self.t = forward_step + 1;
                }
                action
            }
        }
    }

    /// One seed per `(cycle, stream)`.
    ///
    /// `RENOISE_STREAM` names the climb's single noise field, and its being
    /// fresh per *cycle* is the whole anti-frozen-loop guarantee: re-adding the
    /// same field every cycle would walk the image back where it was instead of
    /// onwards. Within a cycle it is deliberately constant — that constancy is
    /// the coherence of the grain (module header).
    ///
    /// The additive structure here is harmless because `gaussian_at` avalanches
    /// the seed before folding the pixel index in; deriving the field from a
    /// seed that shares an additive relation with the index stream is the
    /// defect `ANISOTROPY_HUNT.md` documents.
    fn stream_seed(&self, stream: u64) -> u64 {
        self.base_seed
            .wrapping_add((self.cycle as u64 + 1).wrapping_mul(stream))
    }

    /// The level the climb in progress departed from, for a caller that needs
    /// it outside of a [`DriftAction`].
    pub fn departure_step(&self) -> usize {
        self.departure_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEPS: usize = 256;

    /// Splits the next `count` actions into the levels they work at, so a test
    /// can read the trajectory as the triangle wave it is: the descents on one
    /// side, the forward increments of the climbs on the other.
    fn walk(drift: &mut PerpetualDrift, count: usize) -> (Vec<usize>, Vec<usize>) {
        let mut descents = Vec::new();
        let mut climbs = Vec::new();
        for _ in 0..count {
            match drift.step() {
                DriftAction::Descend { diffusion_step, .. } => descents.push(diffusion_step),
                DriftAction::Climb { forward_step, .. } => climbs.push(forward_step),
            }
        }
        (descents, climbs)
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

    /// The climb is the correction this module received from real use: the
    /// re-noising used to be a single jump, which read as an act of violence
    /// against an image the viewer was contemplating. It is now walked, one
    /// forward increment per frame, over exactly the ground the descent covered.
    #[test]
    fn wander_descends_to_zero_then_climbs_back_one_increment_at_a_time() {
        let depth = 12;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 7);

        // The opening descent is the full schedule: T-1 .. 0, then the climb.
        let (descents, climbs) = walk(&mut drift, STEPS + depth + 1);
        assert_eq!(descents.first().copied(), Some(STEPS - 1));
        assert_eq!(descents.last().copied(), Some(0));
        assert_eq!(descents.len(), STEPS);
        assert_eq!(
            climbs,
            (0..=depth).collect::<Vec<_>>(),
            "the climb must visit every level, not leap to t_r"
        );
        assert_eq!(drift.cycle(), 1);

        // Every later cycle is depth .. 0 down, then 0 .. depth back up.
        let (descents, climbs) = walk(&mut drift, 2 * (depth + 1));
        assert_eq!(descents, (0..=depth).rev().collect::<Vec<_>>());
        assert_eq!(climbs, (0..=depth).collect::<Vec<_>>());
    }

    /// A cycle that dissolves for as long as it resolved. The symmetry is the
    /// perceptual point — an image that comes apart faster than it came
    /// together still reads as a jump, only a shorter one.
    #[test]
    fn the_climb_is_exactly_as_long_as_the_descent_it_undoes() {
        for (regime, depth) in [
            (PerpetualRegime::Wander, 12usize),
            (PerpetualRegime::Breathe, 32),
        ] {
            let mut drift = PerpetualDrift::new(STEPS, regime, depth, 7);
            walk(&mut drift, STEPS); // burn the opening descent from pure noise
            // One full cycle: climb, then descent, then the next climb starts.
            let cycle_length = 2 * (depth - drift.floor() + 1);
            let (descents, climbs) = walk(&mut drift, cycle_length);
            assert_eq!(
                climbs.len(),
                descents.len(),
                "{regime:?}: {} up vs {} down",
                climbs.len(),
                descents.len()
            );
        }
    }

    /// The point of breathing is that the image never settles: a cycle that
    /// touched `t = 0` would resolve it, which is wander's job.
    ///
    /// This holds for the climb too, and that is why the climb starts from the
    /// latent rather than from x̂₀ — a climb starting at `k = 0` would put the
    /// fully resolved image on screen on its way up (see the module header).
    #[test]
    fn breathe_never_reaches_a_fully_resolved_image() {
        let depth = 32;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Breathe, depth, 7);
        // Skip the opening descent from pure noise, which does pass through 0's
        // neighbourhood only as far as the floor.
        let (descents, climbs) = walk(&mut drift, STEPS * 2);
        let floor = depth / 2;
        assert!(
            descents.iter().all(|&t| t >= floor),
            "breathing descended below its floor {floor}: min = {:?}",
            descents.iter().min()
        );
        assert!(
            climbs.iter().all(|&t| t >= floor),
            "breathing climbed from below its floor {floor} — the image resolved \
             on the way up: min = {:?}",
            climbs.iter().min()
        );
        assert_eq!(descents.iter().max().copied(), Some(STEPS - 1));
    }

    /// The instrument panel reads this, and a viewer watching an image come
    /// apart needs to be told that is what is happening.
    #[test]
    fn the_phase_says_which_way_the_run_is_going() {
        let depth = 8;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 7);
        for _ in 0..(STEPS + 4 * (depth + 1)) {
            let phase = drift.phase();
            match drift.step() {
                DriftAction::Descend { .. } => assert_eq!(phase, DriftPhase::Descent),
                DriftAction::Climb { .. } => assert_eq!(phase, DriftPhase::Climb),
            }
        }
        assert_eq!(DriftPhase::Descent.label(), "descente");
        assert_eq!(DriftPhase::Climb.label(), "remontée");
    }

    /// Exactly one action per cycle announces that the cycle closed — the one
    /// the caller hangs "save this frame" on. Two would double-save; none would
    /// make the time-lapse stop.
    #[test]
    fn one_climb_increment_per_cycle_reports_the_cycle_closing() {
        let depth = 8;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 7);
        let mut openings = Vec::new();
        let mut actions = 0;
        while openings.len() < 4 {
            actions += 1;
            assert!(actions < 10_000, "the drift stopped closing cycles");
            if let DriftAction::Climb {
                forward_step,
                opens_cycle: true,
                ..
            } = drift.step()
            {
                openings.push(forward_step);
            }
        }
        assert_eq!(drift.cycle(), 4, "one cycle counted per announcement");
        assert!(
            openings.iter().all(|&k| k == 0),
            "wander closes its cycles at the floor t=0: {openings:?}"
        );
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

        // One more descent (at t), then the climb takes over — from t, which is
        // where that descent left the latent.
        let (descents, climbs) = walk(&mut drift, 2);
        assert_eq!(descents, vec![t]);
        assert_eq!(climbs, vec![t]);
        assert_eq!(drift.phase(), DriftPhase::Climb);
    }

    /// Raising the floor mid-climb must not leave the run climbing towards a
    /// ceiling it is already above: it turns round at the next increment.
    #[test]
    fn shrinking_the_depth_during_a_climb_ends_it_where_the_latent_is() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 200, 7);
        // Reach the floor, then climb part of the way back up.
        walk(&mut drift, STEPS + 40);
        assert_eq!(drift.phase(), DriftPhase::Climb);
        let level = drift.current_step();
        assert!(level > MIN_RENOISE_DEPTH);

        drift.set_depth(MIN_RENOISE_DEPTH);
        let (descents, climbs) = walk(&mut drift, 2);
        assert_eq!(climbs, vec![level], "one last increment, then round");
        assert_eq!(
            descents,
            vec![level],
            "the descent resumes where the latent actually is, not at t_r"
        );
    }

    /// One field per climb, a different one every cycle.
    ///
    /// **Reformulated** from `no_two_climb_increments_anywhere_draw_the_same_seed`,
    /// which demanded the exact opposite *within* a climb — a fresh field per
    /// increment — because the climb used to be an exact Markov walk, where
    /// reusing a field would have meant adding the same field `t_r` times
    /// instead of walking. The climb is now the closed form from the departure,
    /// where one field per climb is the *design*: it is what makes the grain
    /// cohere between frames. The property that has **not** moved is uniqueness
    /// per cycle — that is the anti-frozen-loop guarantee, and it is asserted
    /// here as strictly as before.
    ///
    /// Run in **both** regimes: wander departs from `0`, where a departure
    /// level that was never updated would still look right, and breathing
    /// departs from `t_r/2`, where it would not.
    #[test]
    fn one_field_per_climb_and_a_new_one_every_cycle() {
        let depth = 16;
        for regime in [PerpetualRegime::Wander, PerpetualRegime::Breathe] {
            let mut drift = PerpetualDrift::new(STEPS, regime, depth, 0x5eed);
            let floor = drift.floor();
            // Seeds grouped by climb, in order; a climb opens on `opens_cycle`.
            let mut climbs: Vec<Vec<u64>> = Vec::new();
            let mut departures: Vec<Vec<(usize, usize)>> = Vec::new();
            let mut descent_seeds = std::collections::HashSet::new();
            for _ in 0..(STEPS + 8 * 2 * (depth + 1)) {
                match drift.step() {
                    DriftAction::Descend { path_seed, .. } => {
                        descent_seeds.insert(path_seed);
                    }
                    DriftAction::Climb {
                        forward_step,
                        departure_step,
                        cycle_seed,
                        opens_cycle,
                    } => {
                        if opens_cycle {
                            climbs.push(Vec::new());
                            departures.push(Vec::new());
                        }
                        if let Some(current) = climbs.last_mut() {
                            current.push(cycle_seed);
                            departures
                                .last_mut()
                                .expect("same push")
                                .push((forward_step, departure_step));
                        }
                    }
                }
            }
            assert!(climbs.len() >= 8, "{regime:?}: not enough climbing sampled");

            for (index, seeds) in climbs.iter().enumerate() {
                assert!(
                    seeds.windows(2).all(|w| w[0] == w[1]),
                    "{regime:?} climb {index} changed field mid-ascent: the grain \
                     would crackle"
                );
            }
            // The departure is fixed for a whole climb, and it is the level the
            // climb opened at — the floor the descent stopped on. Anything else
            // and the alpha_bar ratio is formed against the wrong noise level.
            for (index, levels) in departures.iter().enumerate() {
                let opening = levels[0].0;
                assert_eq!(
                    opening, floor,
                    "{regime:?} climb {index} opened at {opening}, not at the floor \
                     {floor} the descent left the latent on"
                );
                assert!(
                    levels.iter().all(|(_, dep)| *dep == opening),
                    "{regime:?} climb {index} moved its departure mid-ascent: {levels:?}"
                );
                assert_eq!(
                    levels.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
                    (opening..=opening + levels.len() - 1).collect::<Vec<_>>(),
                    "{regime:?} climb {index} did not visit every level from its departure"
                );
            }

            // A field per cycle, never reused: the anti-frozen-loop guarantee.
            let fields: Vec<u64> = climbs.iter().map(|seeds| seeds[0]).collect();
            let unique: std::collections::HashSet<u64> = fields.iter().copied().collect();
            assert_eq!(
                unique.len(),
                fields.len(),
                "{regime:?}: two cycles dissolved the image under the same field"
            );
            // One descent seed per cycle (the timestep is folded in downstream
            // by `reverse_step_seed`), and no descent ever shares with a climb.
            assert!(unique.is_disjoint(&descent_seeds));
        }
    }

    /// The fear this addresses, stated as the user stated it: a re-noising that
    /// came back identical would leave the piece turning in circles. Seeds
    /// differing is not enough — what matters is that the *fields they produce*
    /// differ, so this runs two consecutive climbs on the same image through the
    /// real schedule and compares the noise each one actually injected.
    #[test]
    fn two_consecutive_cycles_dissolve_the_image_into_different_noise() {
        const N: usize = 4096;
        let depth = 24;
        let schedule = crate::LinearNoiseSchedule::new_linear(STEPS, 1e-4, 0.02);
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 0x5eed);

        // The plan of two consecutive climbs, in order. Collected until a
        // *third* one opens: a climb is only complete once the next has begun,
        // and comparing a full climb against a truncated one compares lengths
        // rather than fields.
        let mut climbs: Vec<Vec<(usize, usize, u64)>> = Vec::new();
        while climbs.len() < 3 {
            match drift.step() {
                DriftAction::Climb {
                    forward_step,
                    departure_step,
                    cycle_seed,
                    opens_cycle,
                } => {
                    if opens_cycle {
                        climbs.push(Vec::new());
                    }
                    if let Some(current) = climbs.last_mut() {
                        current.push((forward_step, departure_step, cycle_seed));
                    }
                }
                DriftAction::Descend { .. } => {}
            }
        }

        // Same starting image, both climbs — any difference is the noise.
        let x0: Vec<f32> = (0..N).map(|i| (i as f32 * 0.03).sin() * 0.7).collect();
        // The frames the viewer would see, walked through the real schedule.
        let climb = |plan: &[(usize, usize, u64)]| -> Vec<Vec<f32>> {
            plan.iter()
                .map(|(level, departure, seed)| {
                    schedule.forward_from(&x0, *departure, *level, *seed)
                })
                .collect()
        };
        assert_eq!(
            climbs[0].len(),
            climbs[1].len(),
            "climbs collected unevenly"
        );
        assert_eq!(climbs[0].len(), depth + 1, "a climb is t_r + 1 levels");
        let (first_frames, second_frames) = (climb(&climbs[0]), climb(&climbs[1]));
        let first = first_frames.last().expect("non-empty climb").clone();
        let second = second_frames.last().expect("non-empty climb").clone();

        // Correlation of the two injected fields: independent draws sit at 0,
        // a repeated re-noising would sit at 1.
        let signal = schedule.alpha_bar(depth).sqrt();
        let residual = |x: &[f32]| -> Vec<f64> {
            x.iter()
                .zip(x0.iter())
                .map(|(v, c)| (*v - signal * *c) as f64)
                .collect()
        };
        let (a, b) = (residual(&first), residual(&second));
        let dot = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(x, y)| x * y).sum::<f64>();
        let correlation = dot(&a, &b) / (dot(&a, &a).sqrt() * dot(&b, &b).sqrt());

        assert!(
            correlation.abs() < 0.05,
            "two cycles injected near-identical noise (r = {correlation:.3}): the run \
             loops instead of drifting"
        );

        // The other half of the same coin, on the same frames: *within* a
        // cycle the grain must hold from one frame to the next. Uncorrelated
        // across cycles and correlated along one — the run drifts without
        // crackling. (The number itself, and the before/after comparison, are
        // in `consecutive_climb_frames_change_by_the_same_grain`.)
        let deltas: Vec<Vec<f64>> = first_frames
            .windows(2)
            .map(|w| {
                w[1].iter()
                    .zip(w[0].iter())
                    .map(|(a, b)| (*a - *b) as f64)
                    .collect()
            })
            .collect();
        let along: Vec<f64> = deltas
            .windows(2)
            .map(|w| dot(&w[0], &w[1]) / (dot(&w[0], &w[0]).sqrt() * dot(&w[1], &w[1]).sqrt()))
            .collect();
        let mean_along = along.iter().sum::<f64>() / along.len() as f64;
        assert!(
            mean_along > 0.95,
            "a cycle's frames change by unrelated grain (r = {mean_along:.3}): the \
             ascent crackles"
        );
    }

    #[test]
    fn reseeding_returns_to_the_top_of_the_schedule() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 32, 7);
        walk(&mut drift, STEPS + 5);
        assert!(drift.cycle() > 0);
        assert_eq!(drift.phase(), DriftPhase::Climb);

        drift.reseed(0x1234);
        assert_eq!(drift.cycle(), 0);
        assert_eq!(drift.current_step(), STEPS - 1);
        assert_eq!(drift.phase(), DriftPhase::Descent);
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
