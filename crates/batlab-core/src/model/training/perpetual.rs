//! The itinerary of an endless denoising run: where on the schedule the sampler
//! is, and when to stop descending and re-noise. Unlike a normal run, it never
//! ends — descend to a floor, climb the forward chain back to `t_r`, repeat. Only
//! the itinerary changes: descents are [`crate::reverse_step`], climbs are one
//! forward level in closed form ([`LinearNoiseSchedule::forward_from`]). Holds no
//! model, GPU or tensor, so the infinite loop is cheap to unit-test.
//!
//! # The three regimes
//!
//! ```text
//!   t                        t                        t
//!   │╲  ╱╲  ╱╲  ╱╲  ╱        │                        │
//! t_r│ ╲╱  ╲╱  ╲╱  ╲╱     t_r│╲  ╱╲  ╱╲  ╱╲  ╱     t* │╲~~~~~~~~~~~~~~~~
//!   │                        │ ╲╱  ╲╱  ╲╱  ╲╱         │ ╲
//!   │                   t_r/2│                        │
//!  0└─────────────▶ time   0 └─────────────▶ time    0 └─────────────▶ time
//!    Wander — resolves        Breathe — never          Flux — never leaves
//!    at 0, climbs back.       resolves, oscillates.    t*, churns in place.
//! ```
//!
//! `t_r` (the renoise depth) is the audacity dial. Wander/breathe are triangles:
//! the climb takes as many frames as the descent it undoes. **Flux** removes the
//! turning points instead of smoothing them — it holds `t*` forever, one reverse
//! step down and one forward increment back per frame, so the noise level is
//! stationary and only the content moves. Descent and climb survive only as the
//! approach to `t*`, both walked at tempo so moving `t*` is a glide.
//!
//! # The climb: walked, in closed form, from the departure latent
//!
//! Walked one level per frame, not jumped ([`add_noise`] straight to `t_r`
//! watched like a slap). Computed by [`forward_from`] with ONE field per cycle,
//! not the exact forward chain (a fresh field per increment shimmered — thirty
//! unrelated grains a second). Started from the latent the descent LEFT, never a
//! re-read of it as clean: `x_t = sqrt(ᾱ_t/ᾱ_dep)·x_dep + sqrt(1-ᾱ_t/ᾱ_dep)·ε` is
//! valid from any legitimate `x_dep`, whereas `add_noise` assumes a clean `x₀` and
//! decays the signal by an extra `sqrt(ᾱ_floor)` every cycle. Rationale and
//! measurements: `CLIMB_COHERENCE.md`.
//!
//! Wander floors at `t=0` (the sampler's output IS the clipped x̂₀, see
//! `x0_estimate_is_the_x0_the_reverse_step_uses`); breathe floors above zero and
//! climbs from a still-noisy latent, so the image never fully resolves. The
//! caller snapshots the latent on the `opens_cycle` increment and feeds it to
//! every level; [`DriftAction::Climb`] names the departure level for the ᾱ ratio.

use serde::{Deserialize, Serialize};

/// What a run sets out from. Noise is the original generative-then-wander
/// opening; Image is a drift away from a real photograph. No new phase is needed
/// — a run from a real image starts where a wander cycle ENDS (a clean `x₀` at
/// `t=0`, about to climb), so [`PerpetualDrift::from_image`] opens in exactly the
/// state [`PerpetualDrift::open_climb`] leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerpetualOrigin {
    /// Pure noise at `T-1`. The run denoises down to its first image.
    Noise,
    /// A real image at `t = 0`. The run climbs away from it and drifts.
    Image,
}

impl Default for PerpetualOrigin {
    fn default() -> Self {
        Self::Image
    }
}

/// Which way the run wanders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerpetualRegime {
    /// Descend to `t = 0` — a fully resolved image — then climb back to `t_r`.
    Wander,
    /// Descend only half-way to `t_r/2`, then back up. The image never settles.
    Breathe,
    /// Never leave `t*`: one reverse step down and one forward increment back,
    /// every frame. No cycle, no phase, no turning point — see the module
    /// header.
    Flux,
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
            PerpetualRegime::Flux => "flux",
        }
    }

    /// `[m]` walks the three in a ring.
    pub fn toggle(self) -> Self {
        match self {
            PerpetualRegime::Wander => PerpetualRegime::Breathe,
            PerpetualRegime::Breathe => PerpetualRegime::Flux,
            PerpetualRegime::Flux => PerpetualRegime::Wander,
        }
    }

    /// What the depth dial is called on screen. It is the same number in the
    /// same field, but in flux it is not a *depth* — nothing is re-noised back
    /// down to it, the run simply lives there.
    pub fn depth_label(self) -> &'static str {
        match self {
            PerpetualRegime::Wander | PerpetualRegime::Breathe => "t_r",
            PerpetualRegime::Flux => "t*",
        }
    }

    /// Shallowest level this regime allows on the dial.
    ///
    /// Flux goes down to `1`: it holds a level rather than bouncing off it, so
    /// a very low `t*` is a legitimate setting (a nearly-settled image stirred
    /// by `sqrt(beta)` a frame) where for the other two it would make a cycle
    /// one step long.
    pub fn min_depth(self) -> usize {
        match self {
            PerpetualRegime::Wander | PerpetualRegime::Breathe => MIN_RENOISE_DEPTH,
            PerpetualRegime::Flux => MIN_FLUX_LEVEL,
        }
    }

    /// Parses the CLI spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "wander" | "errance" => Some(PerpetualRegime::Wander),
            "breathe" | "respiration" => Some(PerpetualRegime::Breathe),
            "flux" => Some(PerpetualRegime::Flux),
            _ => None,
        }
    }
}

/// Shallowest renoise depth on offer. Below this a cycle is a single step and
/// the image stops moving.
pub const MIN_RENOISE_DEPTH: usize = 4;
/// Shallowest level flux may hold. `0` is excluded because the reverse step has
/// `sigma = 0` there and the forward increment `beta_0 ~ 1e-4`: the churn would
/// stop and the picture freeze.
pub const MIN_FLUX_LEVEL: usize = 1;
/// How far one press of the depth key moves `t_r`.
pub const RENOISE_DEPTH_STEP: usize = 8;

/// How fast the climb runs relative to the descent. `1.0` = the symmetry the eye
/// reads as breathing. Raise to hurry the dissolve, lower to draw it out.
pub const CLIMB_TEMPO_RATIO: f32 = 1.0;

/// Distinct odd increments for the independent seed streams a cycle needs, so its
/// descent noise and climb field are unrelated (both avalanched by `gaussian_at`).
const DESCENT_STREAM: u64 = 0x9e37_79b9_7f4a_7c15;
const RENOISE_STREAM: u64 = 0xbf58_476d_1ce4_e5b9;
/// Flux's two streams, indexed by FRAME (a stationary run has no cycles, and both
/// fields must be fresh every frame). Dedicated, so no flux frame lands on a
/// wander cycle's seed. Additive, never XOR — the `ANISOTROPY_HUNT.md` defect.
const FLUX_PATH_STREAM: u64 = 0xd1b5_4a32_d192_ed03;
const FLUX_FIELD_STREAM: u64 = 0xa076_1d64_78bd_642f;

/// Which way the run is currently moving on the schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftPhase {
    /// Walking the reverse chain down: the image resolving.
    Descent,
    /// Walking the forward chain back up: the image dissolving.
    Climb,
    /// Holding one level: a step down and a step back up on every frame.
    Flux,
}

impl DriftPhase {
    /// What the panel says the run is doing — an image coming apart looks like a
    /// malfunction unless the instrument says otherwise.
    pub fn label(self) -> &'static str {
        match self {
            DriftPhase::Descent => "descente",
            DriftPhase::Climb => "remontée",
            DriftPhase::Flux => "flux",
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
    /// Carry the climb's departure state up to `forward_step`. The caller applies
    /// `schedule.forward_from(departure, departure_step, forward_step, cycle_seed)`,
    /// `departure` being the latent snapshotted on the `opens_cycle` increment —
    /// never re-read as a clean image (see module header).
    Climb {
        /// The level this frame lands on (rises by one per action).
        forward_step: usize,
        /// The level the climb set out from — constant, fixes the `ᾱ` ratio.
        departure_step: usize,
        /// The climb's SINGLE noise field, drawn once per cycle.
        cycle_seed: u64,
        /// First increment of a climb: the cycle just closed. Save a frame and
        /// snapshot the departure.
        opens_cycle: bool,
    },
    /// One stationary frame at `diffusion_step`: reverse-step down one level, then
    /// put it back with the forward increment. This is `forward_from` with equal
    /// departure/destination (collapsing to `sqrt(alpha_t)·x + sqrt(beta_t)·eps`),
    /// so the noise budget is stationary by construction. BOTH seeds are fresh
    /// every frame: `t` is constant here, so a held path seed would inject the
    /// same posterior field repeatedly — a fixed push, not a draw.
    Flux {
        /// `t*` — the level held.
        diffusion_step: usize,
        /// The reverse step's posterior draw for this frame.
        path_seed: u64,
        /// The forward increment's field for this frame.
        renoise_seed: u64,
    },
}

impl DriftAction {
    /// The phase this action IS, as opposed to the phase the drift was in before
    /// it was asked for one. The two differ by a frame wherever
    /// [`PerpetualDrift::settle_phase`] reconciles inside [`PerpetualDrift::step`]:
    /// reading `phase()` first, stepping second, labels the frame just left (in
    /// flux, the first churn frame as descent). Anything naming a frame takes it
    /// from here, so name and deed cannot come apart.
    pub fn phase(&self) -> DriftPhase {
        match self {
            DriftAction::Descend { .. } => DriftPhase::Descent,
            DriftAction::Climb { .. } => DriftPhase::Climb,
            DriftAction::Flux { .. } => DriftPhase::Flux,
        }
    }
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
    /// The level the current climb set out from — fixed for the whole climb, so
    /// every level is formed from the same departure and field.
    departure_step: usize,
    /// The current climb's single noise field, held (not re-derived) so nothing —
    /// not even a mid-ascent regime change — changes the grain half-way up.
    climb_seed: u64,
    /// Set when a cycle closes, cleared by the climb increment that reports it.
    opens_cycle: bool,
    /// Stationary frames emitted so far. Flux indexes its seeds by this; it also
    /// ticks once per flux approach, keeping its field distinct from neighbours.
    frame: u64,
    /// What the run set out from, so `[r]` re-seeds the way it opened — re-seeding
    /// a photograph drift into pure noise would change what the piece IS.
    origin: PerpetualOrigin,
}

impl PerpetualDrift {
    /// Opens a run at the top of the schedule. The first descent always starts
    /// from `T-1` whatever `depth` says (the run begins on pure noise); `depth`
    /// only governs how far each SUBSEQUENT cycle throws the image back up.
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
            climb_seed: 0,
            opens_cycle: false,
            frame: 0,
            origin: PerpetualOrigin::Noise,
        };
        drift.set_depth(depth);
        drift
    }

    /// Opens a run on a **real image**, at `t = 0`.
    ///
    /// The caller's latent must be that image (see [`PerpetualOrigin`]). The
    /// drift opens in the state a settled cycle leaves — climbing, from a
    /// departure at level 0, announcing `opens_cycle` so the caller snapshots
    /// the picture it set out from. Nothing else about the itinerary differs:
    /// the run climbs to `t_r`, descends, and cycles, exactly as it would have
    /// after denoising its way down from pure noise.
    ///
    /// The dial is honoured from the very first climb, which is the point of
    /// the mode — "jauger le niveau de bruit qu'on réinjecte" starts working on
    /// frame one rather than after a 256-step descent.
    pub fn from_image(
        steps: usize,
        regime: PerpetualRegime,
        depth: usize,
        base_seed: u64,
    ) -> Self {
        let mut drift = Self::new(steps, regime, depth, base_seed);
        drift.origin = PerpetualOrigin::Image;
        drift.open_from_image();
        drift
    }

    /// What the run set out from.
    pub fn origin(&self) -> PerpetualOrigin {
        self.origin
    }

    /// Puts the drift in the state a settled cycle leaves: about to climb away
    /// from a clean image sitting at level 0.
    fn open_from_image(&mut self) {
        self.open_climb(0, self.stream_seed(RENOISE_STREAM));
    }

    pub fn regime(&self) -> PerpetualRegime {
        self.regime
    }

    /// The phase the run INTENDS for its next action — NOT the name of a frame.
    /// [`Self::step`] reconciles it on the way in, so it can be overruled next
    /// call (in flux it reads `Descent` up to the first churn frame). Frame names
    /// come from [`DriftAction::phase`], which is what actually happened.
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

    /// Lowest timestep a cycle descends to — the regime IS this number. Flux has
    /// floor and ceiling on the same level: it lives on `t*`, not descends toward it.
    pub fn floor(&self) -> usize {
        match self.regime {
            PerpetualRegime::Wander => 0,
            PerpetualRegime::Breathe => self.depth / 2,
            PerpetualRegime::Flux => self.depth,
        }
    }

    /// Shallowest level the dial reaches in the current regime.
    pub fn min_depth(&self) -> usize {
        self.regime.min_depth()
    }

    /// Clamps and applies a new renoise depth. A deeper `t_r` takes effect at the
    /// next climb; a shallower one that leaves `t` at/below the new floor simply
    /// ends the descent next call (`step` compares with `<=`). In flux it takes
    /// effect next frame and the run WALKS to the new `t*` (one level per frame),
    /// which is why moving `t*` is a glide, not a jump.
    pub fn set_depth(&mut self, depth: usize) {
        self.depth = depth.clamp(self.min_depth(), self.max_depth());
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

    /// Walks `[m]` round the ring of regimes, re-clamping the dial (regimes do not
    /// share a floor: flux may sit at `t*=1`, below what a cycle regime accepts).
    /// Nothing else is reset — the new regime walks to its level from where it is.
    pub fn toggle_regime(&mut self) {
        self.regime = self.regime.toggle();
        self.set_depth(self.depth);
    }

    /// Restarts the drift under a new seed, from whatever it originally set out
    /// from. The caller redraws its latent to match (fresh noise, or another image).
    pub fn reseed(&mut self, base_seed: u64) {
        self.base_seed = base_seed;
        self.t = self.steps - 1;
        self.cycle = 0;
        self.phase = DriftPhase::Descent;
        self.departure_step = 0;
        self.climb_seed = 0;
        self.opens_cycle = false;
        self.frame = 0;
        if self.origin == PerpetualOrigin::Image {
            self.open_from_image();
        }
    }

    /// The seed the initial latent is drawn from — the same [`crate::base_noise_seed`]
    /// fold as the native sampler and web port, so a noise drift opens on the field
    /// an inference run would.
    pub fn initial_noise_seed(&self) -> u64 {
        crate::base_noise_seed(self.base_seed)
    }

    /// Yields the next action and advances.
    pub fn step(&mut self) -> DriftAction {
        self.settle_phase();
        match self.phase {
            DriftPhase::Descent => {
                let diffusion_step = self.t;
                let action = DriftAction::Descend {
                    diffusion_step,
                    path_seed: self.stream_seed(DESCENT_STREAM),
                };
                if diffusion_step <= self.floor() {
                    // A reverse step at `t` leaves the latent one level below, so
                    // the climb resumes at `t` itself and re-walks the ground the
                    // descent just covered.
                    self.cycle += 1;
                    self.open_climb(diffusion_step, self.stream_seed(RENOISE_STREAM));
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
                    cycle_seed: self.climb_seed,
                    opens_cycle: std::mem::take(&mut self.opens_cycle),
                };
                if forward_step >= self.ceiling() {
                    // The reverse chain picks up at `forward_step`, ceiling or not
                    // (the user may have moved `t_r` mid-climb). In flux,
                    // `settle_phase` turns this back into a stationary frame.
                    self.phase = DriftPhase::Descent;
                } else {
                    self.t = forward_step + 1;
                }
                action
            }
            DriftPhase::Flux => {
                // `settle_phase` guarantees the latent is on `t*` here.
                let diffusion_step = self.t;
                let frame = self.frame;
                self.frame += 1;
                DriftAction::Flux {
                    diffusion_step,
                    path_seed: self.frame_seed(FLUX_PATH_STREAM, frame),
                    renoise_seed: self.frame_seed(FLUX_FIELD_STREAM, frame),
                }
            }
        }
    }

    /// Reconciles the phase with the regime and the latent's actual level before
    /// any action. Everything that can move under a running drift is handled here:
    /// `[m]` swapping the regime, `↑`/`↓` moving `t*`. Both are answered by WALKING
    /// (descent/climb become the approach to the new level), never by teleporting.
    fn settle_phase(&mut self) {
        if self.regime != PerpetualRegime::Flux {
            if self.phase == DriftPhase::Flux {
                // Left flux: the latent sits on `t`, so a descent picks up there.
                self.phase = DriftPhase::Descent;
            }
            return;
        }

        // A descent in flux is only ever an approach from above; it ends the
        // moment the latent reaches `t*` rather than stepping past it.
        if self.phase == DriftPhase::Descent && self.t <= self.ceiling() {
            self.phase = DriftPhase::Flux;
        }
        if self.phase == DriftPhase::Flux {
            match self.t.cmp(&self.ceiling()) {
                // `t*` lowered: walk down, one reverse step a frame.
                std::cmp::Ordering::Greater => self.phase = DriftPhase::Descent,
                // `t*` raised: climb in closed form, one field for the whole
                // approach so the grain coheres (`CLIMB_COHERENCE.md`).
                std::cmp::Ordering::Less => {
                    self.frame += 1;
                    let seed = self.frame_seed(FLUX_FIELD_STREAM, self.frame);
                    self.open_climb(self.t + 1, seed);
                }
                std::cmp::Ordering::Equal => {}
            }
        }
    }

    /// Opens a climb from the latent sitting just below `first_level`, under a
    /// single noise field held for the whole ascent.
    fn open_climb(&mut self, first_level: usize, seed: u64) {
        self.phase = DriftPhase::Climb;
        self.t = first_level;
        self.departure_step = first_level;
        self.climb_seed = seed;
        self.opens_cycle = true;
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

    /// One seed per `(frame, stream)` — what flux uses in place of
    /// [`Self::stream_seed`], for want of a cycle to index by.
    fn frame_seed(&self, stream: u64, frame: u64) -> u64 {
        self.base_seed
            .wrapping_add(frame.wrapping_add(1).wrapping_mul(stream))
    }

    /// Stationary frames emitted since the last re-seed.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// The number the panel counts with, and its label. Wander/breathe count
    /// cycles; flux has none and counts stationary frames, since a frozen cycle
    /// counter in the regime that never stops would read as a hung run.
    pub fn counter(&self) -> (&'static str, usize) {
        match self.regime {
            PerpetualRegime::Flux => ("frames", self.frame as usize),
            PerpetualRegime::Wander | PerpetualRegime::Breathe => ("cycle", self.cycle),
        }
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

    /// Splits the next `count` actions into descent levels and climb levels, so a
    /// test can read the trajectory as the triangle wave it is.
    fn walk(drift: &mut PerpetualDrift, count: usize) -> (Vec<usize>, Vec<usize>) {
        let mut descents = Vec::new();
        let mut climbs = Vec::new();
        for _ in 0..count {
            match drift.step() {
                DriftAction::Descend { diffusion_step, .. } => descents.push(diffusion_step),
                DriftAction::Climb { forward_step, .. } => climbs.push(forward_step),
                // Flux has no triangle; its tests walk it by hand.
                DriftAction::Flux { .. } => panic!("a cycling regime yielded a stationary frame"),
            }
        }
        (descents, climbs)
    }

    /// The run opens on pure noise, so the first descent starts at the top even
    /// when `t_r` is shallow — else the first image is a denoised grey field.
    #[test]
    fn the_first_descent_starts_at_the_top_of_the_schedule() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 16, 7);
        let (descents, _) = walk(&mut drift, 1);
        assert_eq!(descents, vec![STEPS - 1]);
    }

    /// The re-noising is walked, one forward increment per frame, over exactly the
    /// ground the descent covered — it used to be a single jump (see module header).
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

    /// A cycle dissolves for as long as it resolved — the symmetry is the
    /// perceptual point; a faster dissolve still reads as a (shorter) jump.
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

    /// Breathing never settles the image (touching `t=0` would resolve it —
    /// wander's job). This holds for the climb too, which is why it starts from
    /// the latent, not x̂₀: from `k=0` it would flash the resolved image on the way up.
    #[test]
    fn breathe_never_reaches_a_fully_resolved_image() {
        let depth = 32;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Breathe, depth, 7);
        // Skip the opening descent from pure noise.
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

    // -- Setting out from a real image --------------------------------------

    /// The mode in one assertion: a run seeded on a photograph does NOT spend its
    /// first 256 frames denoising a grey field — it climbs away on frame one.
    #[test]
    fn a_run_that_sets_out_from_an_image_climbs_away_from_it_immediately() {
        let depth = 24;
        let mut drift = PerpetualDrift::from_image(STEPS, PerpetualRegime::Wander, depth, 7);
        assert_eq!(drift.origin(), PerpetualOrigin::Image);
        assert_eq!(drift.phase(), DriftPhase::Climb);
        assert_eq!(drift.current_step(), 0, "the image is a clean x0, at t = 0");

        // The first action announces the cycle, so the caller snapshots the
        // picture as the climb's departure — the same handshake a settled cycle uses.
        let first = drift.step();
        assert!(
            matches!(
                first,
                DriftAction::Climb {
                    forward_step: 0,
                    departure_step: 0,
                    opens_cycle: true,
                    ..
                }
            ),
            "{first:?}"
        );

        // …then an ordinary run: the rest of this climb (depth levels) and the
        // descent it turns into (depth + 1), short of the next cycle opening.
        let (descents, climbs) = walk(&mut drift, depth + depth + 1);
        assert_eq!(
            climbs,
            (1..=depth).collect::<Vec<_>>(),
            "the climb must visit every level from the image up to t_r"
        );
        assert_eq!(descents, (0..=depth).rev().collect::<Vec<_>>());
    }

    /// A noise-seeded run is untouched — the opening descent from `T-1` is what
    /// makes a *generative* perpetual run, and both modes have to keep working
    /// side by side.
    #[test]
    fn a_noise_seeded_run_still_opens_on_the_full_descent() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 24, 7);
        assert_eq!(drift.origin(), PerpetualOrigin::Noise);
        assert_eq!(drift.phase(), DriftPhase::Descent);
        let (descents, _) = walk(&mut drift, 1);
        assert_eq!(descents, vec![STEPS - 1]);
    }

    /// `[r]` re-seeds the way the run OPENED — a photograph drift coming back on
    /// pure noise would change what the piece is, and hand the wrong latent kind
    /// to a drift already climbing.
    #[test]
    fn re_seeding_keeps_the_origin_the_run_opened_on() {
        for (origin, opener) in [
            (
                PerpetualOrigin::Image,
                PerpetualDrift::from_image as fn(usize, PerpetualRegime, usize, u64) -> PerpetualDrift,
            ),
            (PerpetualOrigin::Noise, PerpetualDrift::new),
        ] {
            let mut drift = opener(STEPS, PerpetualRegime::Wander, 16, 7);
            walk(&mut drift, 40);
            drift.reseed(0x1234);

            assert_eq!(drift.origin(), origin);
            assert_eq!(drift.cycle(), 0);
            match origin {
                PerpetualOrigin::Image => {
                    assert_eq!(drift.phase(), DriftPhase::Climb);
                    assert_eq!(drift.current_step(), 0);
                }
                PerpetualOrigin::Noise => {
                    assert_eq!(drift.phase(), DriftPhase::Descent);
                    assert_eq!(drift.current_step(), STEPS - 1);
                }
            }
        }
    }

    /// Two image runs under different seeds must dissolve under DIFFERENT fields —
    /// sharing one would walk every re-seed down the same path.
    #[test]
    fn two_image_seeded_runs_climb_under_different_fields() {
        let field = |seed: u64| {
            let mut drift = PerpetualDrift::from_image(STEPS, PerpetualRegime::Wander, 16, seed);
            match drift.step() {
                DriftAction::Climb { cycle_seed, .. } => cycle_seed,
                other => panic!("{other:?}"),
            }
        };
        assert_ne!(field(1), field(2));

        // And a re-seed changes it too — the same drift, re-opened.
        let mut drift = PerpetualDrift::from_image(STEPS, PerpetualRegime::Wander, 16, 1);
        let first = match drift.step() {
            DriftAction::Climb { cycle_seed, .. } => cycle_seed,
            other => panic!("{other:?}"),
        };
        drift.reseed(2);
        let second = match drift.step() {
            DriftAction::Climb { cycle_seed, .. } => cycle_seed,
            other => panic!("{other:?}"),
        };
        assert_ne!(first, second);
    }

    /// The instrument panel reads this — a viewer watching an image come apart
    /// needs to be told that is what is happening.
    #[test]
    fn the_phase_says_which_way_the_run_is_going() {
        let depth = 8;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 7);
        for _ in 0..(STEPS + 4 * (depth + 1)) {
            let phase = drift.phase();
            match drift.step() {
                DriftAction::Descend { .. } => assert_eq!(phase, DriftPhase::Descent),
                DriftAction::Climb { .. } => assert_eq!(phase, DriftPhase::Climb),
                DriftAction::Flux { .. } => panic!("wander yielded a stationary frame"),
            }
        }
        assert_eq!(DriftPhase::Descent.label(), "descente");
        assert_eq!(DriftPhase::Climb.label(), "remontée");
    }

    /// Exactly one action per cycle announces the close — where the caller hangs
    /// "save this frame". Two would double-save; none would stop the time-lapse.
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

    /// Shrinking `t_r` below the current position ends the cycle, not descends
    /// past a floor already crossed.
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

    /// Raising the floor mid-climb turns the run round at the next increment, not
    /// leaves it climbing toward a ceiling it is already above.
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

    /// One field per climb (the closed-form design that makes the grain cohere),
    /// a different one every cycle (the anti-frozen-loop guarantee). Reformulated
    /// from a test that demanded a fresh field per increment, back when the climb
    /// was an exact Markov walk. Run in BOTH regimes: wander departs from `0`
    /// (where a never-updated departure level looks right), breathe from `t_r/2`.
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
                    DriftAction::Flux { .. } => {
                        panic!("{regime:?} yielded a stationary frame")
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
            // The departure is fixed for a whole climb, at the floor the descent
            // stopped on — anything else forms the alpha_bar ratio against the
            // wrong noise level.
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

    /// A re-noising that came back identical would leave the piece turning in
    /// circles. Seeds differing is not enough — what matters is the FIELDS they
    /// produce, so this runs two climbs through the real schedule and compares the
    /// noise each injected.
    #[test]
    fn two_consecutive_cycles_dissolve_the_image_into_different_noise() {
        const N: usize = 4096;
        let depth = 24;
        let schedule = crate::LinearNoiseSchedule::new_linear(STEPS, 1e-4, 0.02);
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 0x5eed);

        // Two consecutive climbs, collected until a THIRD opens: a climb is only
        // complete once the next begins, else we compare lengths not fields.
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
                DriftAction::Flux { .. } => panic!("wander yielded a stationary frame"),
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

        // The other half: WITHIN a cycle the grain must hold frame to frame.
        // Uncorrelated across cycles, correlated along one — drifts without
        // crackling. (Numbers: `consecutive_climb_frames_change_by_the_same_grain`.)
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

    /// `[m]` walks a ring of three; each regime's floor is its identity (wander 0,
    /// breathe half-way, flux does not descend — its floor IS the level it holds).
    #[test]
    fn toggling_the_regime_walks_the_ring_and_swaps_the_floor() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 40, 7);
        assert_eq!(drift.floor(), 0);
        drift.toggle_regime();
        assert_eq!(drift.regime(), PerpetualRegime::Breathe);
        assert_eq!(drift.floor(), 20);
        drift.toggle_regime();
        assert_eq!(drift.regime(), PerpetualRegime::Flux);
        assert_eq!(drift.floor(), 40, "flux floors and ceilings on t* itself");
        assert_eq!(drift.ceiling(), 40);
        drift.toggle_regime();
        assert_eq!(drift.regime(), PerpetualRegime::Wander, "the ring closes");
        assert_eq!(drift.floor(), 0);
    }

    /// The dial's bottom differs by regime, so `[m]` must re-clamp on the spot —
    /// else a flux run at `t*=1` would hand wander a `t_r` its machinery forbids.
    #[test]
    fn leaving_flux_lifts_a_level_the_cycling_regimes_would_not_accept() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, 1, 7);
        assert_eq!(drift.depth(), MIN_FLUX_LEVEL);
        drift.toggle_regime();
        assert_eq!(drift.regime(), PerpetualRegime::Wander);
        assert_eq!(drift.depth(), MIN_RENOISE_DEPTH);
    }

    // -- Flux ---------------------------------------------------------------
    // What has to be pinned: it NEVER turns round, neither on its own nor when
    // the dial moves under it.

    /// The next `count` actions as `(phase, level)` pairs.
    fn walk_tagged(drift: &mut PerpetualDrift, count: usize) -> Vec<(DriftPhase, usize)> {
        (0..count)
            .map(|_| {
                let action = drift.step();
                let level = match action {
                    DriftAction::Descend { diffusion_step, .. } => diffusion_step,
                    DriftAction::Climb { forward_step, .. } => forward_step,
                    DriftAction::Flux { diffusion_step, .. } => diffusion_step,
                };
                // Through the same `phase()` accessor the dump and TUI use, so a
                // mutation of it cannot leave these tests green.
                (action.phase(), level)
            })
            .collect()
    }

    /// The regime in one assertion: after the approach, every frame is stationary
    /// at `t*` — no descent, no climb ever again (a turning point is what the
    /// viewer was feeling).
    #[test]
    fn flux_never_turns_round_once_it_has_reached_its_level() {
        let level = 40;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, level, 7);

        let opening = walk_tagged(&mut drift, STEPS - 1 - level);
        assert_eq!(
            opening,
            (level + 1..STEPS)
                .rev()
                .map(|t| (DriftPhase::Descent, t))
                .collect::<Vec<_>>(),
            "the approach from pure noise must visit every level — a leap to t* \
             is the jump we are removing, wearing a different hat"
        );

        // Ten cycles' worth of frames, in a regime that has no cycles.
        let held = walk_tagged(&mut drift, 10 * 2 * (level + 1));
        assert!(
            held.iter().all(|frame| *frame == (DriftPhase::Flux, level)),
            "flux left its level: {:?}",
            held.iter()
                .filter(|frame| **frame != (DriftPhase::Flux, level))
                .take(4)
                .collect::<Vec<_>>()
        );
        assert_eq!(drift.counter(), ("frames", held.len()));
        assert_eq!(drift.cycle(), 0, "a stationary run closes no cycle");
    }

    /// Moving `t*` must WALK — one level a frame — not teleport the latent, which
    /// would put back the jump this regime removes, and silently (the level on
    /// screen would be right either way).
    #[test]
    fn moving_the_level_mid_flux_walks_to_it_one_frame_at_a_time() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, 32, 7);
        walk_tagged(&mut drift, STEPS); // approach, then hold

        // Up two notches: climb increments, every level, then hold again.
        drift.nudge_depth(2);
        let target = 32 + 2 * RENOISE_DEPTH_STEP;
        assert_eq!(drift.depth(), target);
        let up = walk_tagged(&mut drift, 2 * RENOISE_DEPTH_STEP + 3);
        assert_eq!(
            up.iter()
                .filter(|(phase, _)| *phase == DriftPhase::Climb)
                .map(|(_, t)| *t)
                .collect::<Vec<_>>(),
            (33..=target).collect::<Vec<_>>(),
            "the rise must visit every level between the old t* and the new"
        );
        assert!(
            up.iter()
                .skip_while(|(p, _)| *p == DriftPhase::Climb)
                .all(|frame| *frame == (DriftPhase::Flux, target)),
            "the run must settle on the new level: {up:?}"
        );

        // Down three notches: reverse steps, every level, then hold again.
        drift.nudge_depth(-3);
        let landed = target - 3 * RENOISE_DEPTH_STEP;
        let down = walk_tagged(&mut drift, 3 * RENOISE_DEPTH_STEP + 3);
        assert_eq!(
            down.iter()
                .filter(|(phase, _)| *phase == DriftPhase::Descent)
                .map(|(_, t)| *t)
                .collect::<Vec<_>>(),
            (landed + 1..=target).rev().collect::<Vec<_>>(),
            "the fall must visit every level between the old t* and the new"
        );
        assert!(
            down.iter()
                .skip_while(|(p, _)| *p == DriftPhase::Descent)
                .all(|frame| *frame == (DriftPhase::Flux, landed)),
            "the run must settle on the new level: {down:?}"
        );
    }

    /// `[m]` into flux from either side of `t*`, and back out. The latent is never
    /// re-drawn or re-interpreted: wherever it is, the drift walks from there.
    #[test]
    fn entering_and_leaving_flux_walks_from_wherever_the_latent_is() {
        // From above: wander mid-descent, well over t*.
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 64, 7);
        walk_tagged(&mut drift, 100);
        let from_above = drift.current_step();
        assert!(from_above > 64);
        drift.toggle_regime(); // breathe
        drift.toggle_regime(); // flux
        let approach = walk_tagged(&mut drift, from_above - 64 + 2);
        assert_eq!(
            approach,
            (65..=from_above)
                .rev()
                .map(|t| (DriftPhase::Descent, t))
                .chain([(DriftPhase::Flux, 64); 2])
                .collect::<Vec<_>>(),
            "entering flux from above descends to t*, then holds"
        );

        // From below: wander at its floor, under t*.
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 16, 7);
        while drift.current_step() > 0 {
            drift.step();
        }
        drift.step(); // the descent at t = 0; the latent is now a settled image
        assert_eq!(drift.phase(), DriftPhase::Climb);
        drift.toggle_regime();
        drift.toggle_regime();
        drift.set_depth(24);
        let approach = walk_tagged(&mut drift, 26);
        assert_eq!(
            approach,
            (0..=24)
                .map(|t| (DriftPhase::Climb, t))
                .chain([(DriftPhase::Flux, 24)])
                .collect::<Vec<_>>(),
            "entering flux from below climbs to t*, then holds"
        );

        // And out again: the triangle picks up from the level flux held.
        drift.toggle_regime();
        assert_eq!(drift.regime(), PerpetualRegime::Wander);
        let (descents, _) = walk(&mut drift, 3);
        assert_eq!(
            descents,
            vec![24, 23, 22],
            "leaving flux resumes descending from t*, not from the top"
        );
    }

    /// Both of a flux frame's fields must be fresh — the path seed easily wrong:
    /// `t` is constant here, so a held path seed makes `reverse_step_seed` return
    /// the same value and push the identical posterior forever (a fixed direction).
    #[test]
    fn every_flux_frame_draws_two_fresh_and_unrelated_fields() {
        use std::collections::HashSet;
        let level = 48;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, level, 0x5eed);
        let mut paths = Vec::new();
        let mut fields = Vec::new();
        let mut descents = HashSet::new();
        // The approach from pure noise, then exactly 600 stationary frames.
        for _ in 0..(STEPS - 1 - level + 600) {
            match drift.step() {
                DriftAction::Flux {
                    path_seed,
                    renoise_seed,
                    ..
                } => {
                    paths.push(crate::reverse_step_seed(path_seed, level));
                    fields.push(renoise_seed);
                }
                DriftAction::Descend { path_seed, .. } => {
                    descents.insert(crate::reverse_step_seed(path_seed, level));
                }
                DriftAction::Climb { .. } => panic!("flux climbed without being asked"),
            }
        }
        assert_eq!(paths.len(), 600);

        let unique_paths: HashSet<u64> = paths.iter().copied().collect();
        let unique_fields: HashSet<u64> = fields.iter().copied().collect();
        assert_eq!(unique_paths.len(), paths.len(), "a posterior draw repeated");
        assert_eq!(unique_fields.len(), fields.len(), "a noise field repeated");
        assert!(
            unique_paths.is_disjoint(&unique_fields),
            "a frame's two fields must be independent of one another"
        );
        assert!(
            unique_paths.is_disjoint(&descents),
            "a stationary frame reused the approach's posterior draw"
        );

        // Seeds differing is not the property — the *fields they produce*
        // differing is. Two consecutive frames, through the real draw.
        let schedule = crate::LinearNoiseSchedule::new_linear(STEPS, 1e-4, 0.02);
        let correlation = |a: u64, b: u64| {
            let (u, v) = (
                schedule.sample_noise(4096, a),
                schedule.sample_noise(4096, b),
            );
            let dot = |x: &[f32], y: &[f32]| {
                x.iter()
                    .zip(y)
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum::<f64>()
            };
            dot(&u, &v) / (dot(&u, &u).sqrt() * dot(&v, &v).sqrt())
        };
        for pair in fields.windows(2).take(16) {
            let r = correlation(pair[0], pair[1]);
            assert!(
                r.abs() < 0.05,
                "consecutive frames injected the same grain (r = {r:.3}): the run \
                 would vibrate in place instead of drifting"
            );
        }
        // And WITHIN a frame: posterior draw and re-noising field are independent,
        // not one counted twice. Comparing seeds cannot see this (the sampler
        // transforms its own before drawing); only the fields tell.
        for (path, field) in paths.iter().zip(fields.iter()).take(16) {
            let r = correlation(*path, *field);
            assert!(
                r.abs() < 0.05,
                "a frame's two fields are the same one (r = {r:.3}): the churn \
                 would push twice in a single direction instead of drawing twice"
            );
        }
    }

    /// The noise level is stationary, MEASURED not asserted: a reverse step and a
    /// forward increment cancel in law, so the latent stays a legitimate `x_{t*}`.
    /// Getting it wrong is a slow bleed no snapshot test would catch. Run against
    /// an oracle (the exact posterior-optimal predictor for a point mass at `x0`),
    /// whose stationary law `q(x_t | x0)` is known in closed form.
    #[test]
    fn the_flux_churn_holds_the_law_of_its_level() {
        const N: usize = 8192;
        let level = 64;
        let schedule = crate::LinearNoiseSchedule::new_linear(STEPS, 1e-4, 0.02);
        let signal = schedule.alpha_bar(level).sqrt();
        let spread = (1.0 - schedule.alpha_bar(level)).sqrt();

        let x0: Vec<f32> = (0..N).map(|i| (i as f32 * 0.017).sin() * 0.8).collect();
        // Deliberately NOT started on the marginal: a churn that only holds the
        // level it was handed proves nothing.
        let (mut latent, _) = schedule.add_noise(&x0, level / 2, 0xc0ffee);

        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, level, 0xf1_0000);
        let mut spreads = Vec::new();
        let mut signals = Vec::new();
        let mut changes: Vec<f64> = Vec::new();
        let mut history: Vec<Vec<f32>> = Vec::new();
        let mut frames = 0;
        let mut actions = 0;
        while frames < 400 {
            // A drift that stopped yielding stationary frames would spin forever;
            // a hanging test reports nothing.
            actions += 1;
            assert!(
                actions < STEPS + 4_000,
                "the drift stopped holding its level: {frames} stationary frames \
                 in {actions} actions"
            );
            // Only the stationary part is measured; skip the approach.
            let DriftAction::Flux {
                diffusion_step,
                path_seed,
                renoise_seed,
            } = drift.step()
            else {
                continue;
            };
            let eps_hat: Vec<f32> = latent
                .iter()
                .zip(x0.iter())
                .map(|(x, clean)| (x - signal * clean) / spread)
                .collect();
            let previous = latent.clone();
            let down = schedule.denoise_step(
                &latent,
                &eps_hat,
                diffusion_step,
                crate::reverse_step_seed(path_seed, diffusion_step),
            );
            latent = schedule.forward_step(&down, diffusion_step, renoise_seed);
            frames += 1;

            // Least squares against x0, and the residual's spread: the two
            // numbers that say "this is an x_t at t*".
            let (mut dot, mut norm) = (0.0f64, 0.0f64);
            for (x, clean) in latent.iter().zip(x0.iter()) {
                dot += (*x as f64) * (*clean as f64);
                norm += (*clean as f64) * (*clean as f64);
            }
            let coefficient = dot / norm;
            let residual: Vec<f32> = latent
                .iter()
                .zip(x0.iter())
                .map(|(x, clean)| x - coefficient as f32 * clean)
                .collect();
            let variance = residual
                .iter()
                .map(|r| (*r as f64) * (*r as f64))
                .sum::<f64>()
                / residual.len() as f64;
            // Skip the walk onto the level; the claim is about the settled regime.
            if frames > 100 {
                signals.push(coefficient);
                spreads.push(variance.sqrt());
                changes.push(
                    previous
                        .iter()
                        .zip(latent.iter())
                        .map(|(a, b)| (a - b).abs() as f64)
                        .sum::<f64>()
                        / latent.len() as f64,
                );
                history.push(latent.clone());
            }
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let observed_signal = mean(&signals);
        let observed_spread = mean(&spreads);
        assert!(
            (observed_signal - signal as f64).abs() < 0.05,
            "the signal drifted away from sqrt(alpha_bar) = {signal:.4}: {observed_signal:.4}"
        );
        assert!(
            (observed_spread - spread as f64).abs() < 0.03,
            "the noise level drifted away from sqrt(1 - alpha_bar) = {spread:.4}: \
             {observed_spread:.4}"
        );

        // No slow bleed either way: the second half must look like the first.
        let half = spreads.len() / 2;
        let (early, late) = (mean(&spreads[..half]), mean(&spreads[half..]));
        assert!(
            (early - late).abs() < 0.02,
            "the level is drifting over time: {early:.4} → {late:.4}"
        );

        // And no jumps: in a stationary churn the worst frame is the same size
        // as the typical one. This is the user's requirement, in a unit test.
        let mut sorted = changes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let median = sorted[sorted.len() / 2];
        let worst = *sorted.last().expect("non-empty");
        assert!(
            worst < 1.5 * median,
            "a flux frame moved {worst:.4} against a median of {median:.4} — that is \
             a jump, and there must never be one"
        );

        // It drifts, not vibrates: consecutive frames alike, frames 300 apart not.
        // The residual only — the oracle pins the signal to x0, so what must renew
        // itself is the noise.
        let correlation = |a: &[f32], b: &[f32]| {
            let (mean_a, mean_b) = (
                a.iter().map(|v| *v as f64).sum::<f64>() / a.len() as f64,
                b.iter().map(|v| *v as f64).sum::<f64>() / b.len() as f64,
            );
            let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
            for (x, y) in a.iter().zip(b) {
                let (u, v) = (*x as f64 - mean_a, *y as f64 - mean_b);
                num += u * v;
                da += u * u;
                db += v * v;
            }
            num / (da.sqrt() * db.sqrt())
        };
        let residual_of = |frame: &Vec<f32>| -> Vec<f32> {
            frame
                .iter()
                .zip(x0.iter())
                .map(|(x, clean)| x - signal * clean)
                .collect()
        };
        let near = correlation(&residual_of(&history[0]), &residual_of(&history[1]));
        let far = correlation(&residual_of(&history[0]), &residual_of(&history[299]));
        assert!(
            near > 0.9,
            "consecutive frames are unrelated (r = {near:.3}): the churn flickers"
        );
        assert!(
            far.abs() < 0.2,
            "the frame 300 later is the same one (r = {far:.3}): the run vibrates in \
             place instead of wandering"
        );
    }

    /// A frame is filed under the name of what it did. Found from outside: a
    /// black-box run measured a frame tagged `descent` moving with CHURN amplitude
    /// (~40% more) at the first frame of every plateau, because the caller read
    /// `phase()` before `step()` and `settle_phase` reconciles inside it. Every
    /// test here already read the phase off the action, so nothing self-consistent
    /// caught it — why [`DriftAction::phase`] exists and everything naming a frame uses it.
    #[test]
    fn a_frame_is_named_after_the_action_it_performed_not_the_phase_it_left() {
        let level = 40;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, level, 7);

        // Both notions per frame: the phase standing before the step, and the
        // phase of what the step turned out to be.
        let mut walked = Vec::new();
        for _ in 0..STEPS - level + 3 {
            let standing = drift.phase();
            let action = drift.step();
            walked.push((standing, action.phase()));
        }

        let approach = STEPS - 1 - level;
        assert!(
            walked[..approach]
                .iter()
                .all(|(_, done)| *done == DriftPhase::Descent),
            "the approach is a descent, frame by frame"
        );
        assert_eq!(
            walked[approach].1,
            DriftPhase::Flux,
            "the first frame of the churn is a churn frame, and has to be filed as one —              this is the frame that was going out labelled `descent`"
        );
        assert!(
            walked[approach..]
                .iter()
                .all(|(_, done)| *done == DriftPhase::Flux),
            "and every frame after it, too"
        );

        // The trap, pinned: at that frame the phase standing beforehand is the one
        // just LEFT. What must never come back is a frame filed under a phase it
        // did not perform.
        assert_ne!(
            walked[approach].0, walked[approach].1,
            "if these ever agree, `settle_phase` moved and the comment above is stale"
        );

        // The same seam upward: raising `t*` mid-flux walks a climb, and the frame
        // resuming the churn is a churn frame (the TUI used to flash one `descente`).
        drift.set_depth(level + 4);
        let resumed = walk_tagged(&mut drift, 6);
        assert_eq!(
            resumed
                .iter()
                .map(|(phase, _)| *phase)
                .collect::<Vec<_>>(),
            vec![
                DriftPhase::Climb,
                DriftPhase::Climb,
                DriftPhase::Climb,
                DriftPhase::Climb,
                DriftPhase::Flux,
                DriftPhase::Flux,
            ],
            "four levels climbed, then the churn resumes — and not one frame of it              is filed as the descent the reconciliation happens to leave behind"
        );
    }
}
