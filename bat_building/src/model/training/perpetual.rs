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
//! `t_r` (the *renoise depth*) is the audacity dial: low `t_r` perturbs an image
//! that is nearly settled, high `t_r` erases enough of it for a metamorphosis.
//! Wander and breathe are triangles, not sawtooths: the climb takes exactly as
//! many frames as the descent it undoes.
//!
//! # Why a third regime that has no cycle at all
//!
//! Wander and breathe are *phased*: an image resolves, then dissolves, then
//! resolves. Even with a perfectly smooth climb (`CLIMB_COHERENCE.md`) the
//! alternation itself is a pulse — "des frises entre les étapes" — and a viewer
//! feels the machine turn round twice per cycle.
//!
//! **Flux** removes the turning points instead of smoothing them. It holds one
//! noise level `t*` forever and, on every frame, takes *one* reverse step down
//! (`t* → t*-1`) and puts *one* forward increment back (`t*-1 → t*`). The noise
//! level is stationary; only the content moves, by about `sqrt(beta_t*)` a
//! frame. There is no phase to announce, no cycle to close and no moment at
//! which the run changes its mind — which is the whole point.
//!
//! Descent and climb still exist in this regime, but only as the *approach* to
//! `t*`: from above the run descends to it a step at a time, from below it
//! climbs to it in closed form, and both walk at the current tempo so that
//! moving `t*` mid-run is a glide rather than a jump.
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
/// The two streams flux draws on, indexed by its **frame** counter rather than
/// by a cycle — a stationary run has no cycles, and both of its fields have to
/// be fresh on every frame. They are dedicated (rather than reusing the two
/// above with a different index) so that no flux frame can ever land on the
/// seed of a wander cycle, whatever the counters happen to be.
///
/// Additive, never XOR: `gaussian_at` avalanches the seed before folding the
/// pixel index in, and an additive relation between a caller's seed and the
/// index stream is precisely the defect `ANISOTROPY_HUNT.md` documents.
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
    /// What the panel says the run is doing. A frame in which the image is
    /// coming apart is not a malfunction, but it looks like one unless the
    /// instrument says so.
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
    /// One stationary frame at `diffusion_step`: reverse-step down to
    /// `diffusion_step - 1`, then put exactly that one level back on with the
    /// forward increment.
    ///
    /// The caller applies
    /// `schedule.forward_step(reverse_step(x, diffusion_step, path_seed), diffusion_step, renoise_seed)`,
    /// which is `forward_from` with departure and destination equal — the
    /// closed form collapses to `sqrt(alpha_t)·x + sqrt(beta_t)·eps` — so the
    /// latent comes back to the level it started on and the noise budget is
    /// stationary by construction rather than by accounting.
    ///
    /// **Both seeds are fresh on every frame.** The path seed must be, because
    /// the timestep is constant here: `reverse_step_seed` folds `t` in, so a
    /// path seed held across frames would inject the *same* posterior field
    /// over and over — a fixed direction pushed into the image instead of a
    /// draw.
    Flux {
        /// `t*` — the level held. Constant between two changes of the dial.
        diffusion_step: usize,
        /// The reverse step's posterior draw for this frame.
        path_seed: u64,
        /// The forward increment's field for this frame.
        renoise_seed: u64,
    },
}

impl DriftAction {
    /// The phase this action **is** — as opposed to the phase the drift was in
    /// before it was asked for one.
    ///
    /// The two differ by exactly one frame wherever [`PerpetualDrift::settle_phase`]
    /// has work to do, because it reconciles *inside* [`PerpetualDrift::step`]:
    /// a caller that reads `phase()` first and steps second labels the frame
    /// with the phase that was just left. In flux that mislabels the first churn
    /// frame of every approach as a descent — caught from outside by a black-box
    /// test that saw a frame tagged `descent` moving with churn amplitude, 40 %
    /// above the reverse step at the same level.
    ///
    /// Anything that names a frame — a dump tag, the TUI's phase read-out —
    /// takes it from here, so the name and the deed cannot come apart again.
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
    /// The level the climb in progress set out from — its departure state sits
    /// just below this. Fixed for the whole climb, so every level is formed
    /// from the same departure and the same field.
    departure_step: usize,
    /// The climb in progress' single noise field, fixed when the climb opens.
    /// Held rather than re-derived per increment so that nothing — not even a
    /// regime change mid-ascent — can change the grain half-way up.
    climb_seed: u64,
    /// Set when a cycle closes, cleared by the climb increment that reports it.
    opens_cycle: bool,
    /// Stationary frames emitted so far. Flux has no cycles, so this is what
    /// its seeds are indexed by; it also ticks once per flux approach, which
    /// keeps an approach's field distinct from the frames either side of it.
    frame: u64,
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
            climb_seed: 0,
            opens_cycle: false,
            frame: 0,
        };
        drift.set_depth(depth);
        drift
    }

    pub fn regime(&self) -> PerpetualRegime {
        self.regime
    }

    /// Whether the run is currently resolving the image or dissolving it — the
    /// phase it *intends* for its next action.
    ///
    /// **Not the name of a frame.** [`Self::step`] reconciles the phase against
    /// the regime and the level on its way in, so this can be overruled by the
    /// very next call: in flux it reads `Descent` right up to the first frame of
    /// the churn, and `Descent` again for one frame at the top of an upward
    /// approach. Whatever files a frame — a dump tag, a status panel — takes its
    /// name from [`DriftAction::phase`], which is what actually happened.
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
    ///
    /// Flux has floor and ceiling on the same level: it does not descend
    /// *towards* `t*`, it lives on it.
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

    /// Clamps and applies a new renoise depth.
    ///
    /// A deeper `t_r` takes effect at the next climb (the ceiling is only read
    /// there). A shallower one can leave `t` already at or below the new floor
    /// — [`Self::step`] compares with `<=`, so that simply ends the descent on
    /// the next call instead of walking past it; and a climb already above the
    /// new ceiling ends on its next increment rather than unwinding.
    ///
    /// In flux it takes effect on the very next frame, and the run *walks* to
    /// the new `t*` — down by reverse steps, up by climb increments, one level
    /// per frame at the current tempo. That is the whole reason moving `t*` is
    /// not a jump.
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

    /// Walks `[m]` round the ring of regimes.
    ///
    /// The dial is re-clamped, because the regimes do not share a floor: flux
    /// may sit at `t* = 1`, which is below what a cycle-based regime accepts.
    /// Nothing else is reset — the latent stays where it is, and the new regime
    /// walks to its own level from there.
    pub fn toggle_regime(&mut self) {
        self.regime = self.regime.toggle();
        self.set_depth(self.depth);
    }

    /// Restarts the drift from pure noise under a new seed. The caller is
    /// expected to redraw its latent from the same seed.
    pub fn reseed(&mut self, base_seed: u64) {
        self.base_seed = base_seed;
        self.t = self.steps - 1;
        self.cycle = 0;
        self.phase = DriftPhase::Descent;
        self.departure_step = 0;
        self.climb_seed = 0;
        self.opens_cycle = false;
        self.frame = 0;
    }

    /// The seed the run's initial latent should be drawn from.
    pub fn initial_noise_seed(&self) -> u64 {
        self.base_seed ^ 0xa5a5_5a5a_0123_4567
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
                    // A reverse step at `t` leaves the latent one level below,
                    // so the forward chain resumes at `t` itself: the climb
                    // re-walks precisely the ground the descent just covered.
                    self.cycle += 1;
                    // Where the whole climb is formed from: the latent this
                    // very step is about to produce sits just below here.
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
                    // The latent now sits at `forward_step`; that is where the
                    // reverse chain has to be picked up, ceiling or not — the
                    // user may have moved `t_r` mid-climb. In flux,
                    // `settle_phase` turns this straight back into a stationary
                    // frame on the next call, since the level is now `t*`.
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

    /// Reconciles the phase with the regime and with the level the latent is
    /// actually on, before any action is emitted.
    ///
    /// Everything that can move under a running drift is handled here and only
    /// here: `[m]` swapping the regime, and `↑`/`↓` moving `t*` while flux is
    /// stationary on the old one. Both are answered by *walking* — the descent
    /// and the climb become the approach to the new level, one level per frame
    /// at the current tempo — never by teleporting the latent, which is the one
    /// thing this whole regime exists to avoid.
    fn settle_phase(&mut self) {
        if self.regime != PerpetualRegime::Flux {
            if self.phase == DriftPhase::Flux {
                // Left flux: the latent sits on `t`, so the triangle picks up
                // there with a descent, exactly as it would mid-cycle.
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
                // `t*` was lowered: walk down to it, one reverse step a frame.
                std::cmp::Ordering::Greater => self.phase = DriftPhase::Descent,
                // `t*` was raised: climb the levels in closed form, one field
                // for the whole approach so the added grain coheres
                // (`CLIMB_COHERENCE.md`).
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

    /// The number the panel counts with, and what to call it.
    ///
    /// Wander and breathe count cycles. Flux has none — and leaving a cycle
    /// counter frozen on screen, in the one regime whose whole claim is that it
    /// never stops, would read as a hung run. It counts its stationary frames
    /// instead.
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
                // `walk` reads a trajectory as the triangle wave it is; flux
                // has no triangle, and its tests walk it by hand.
                DriftAction::Flux { .. } => panic!("a cycling regime yielded a stationary frame"),
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
                DriftAction::Flux { .. } => panic!("wander yielded a stationary frame"),
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

    /// `[m]` walks a ring of three, and each regime's floor is its identity:
    /// wander bottoms out on the resolved image, breathing half-way, and flux
    /// does not descend at all — its floor *is* the level it holds.
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

    /// The dial does not have the same bottom in every regime, and `[m]` must
    /// bring the number into the new regime's range on the spot — otherwise a
    /// flux run at `t* = 1` would hand wander a `t_r` its own machinery
    /// forbids.
    #[test]
    fn leaving_flux_lifts_a_level_the_cycling_regimes_would_not_accept() {
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, 1, 7);
        assert_eq!(drift.depth(), MIN_FLUX_LEVEL);
        drift.toggle_regime();
        assert_eq!(drift.regime(), PerpetualRegime::Wander);
        assert_eq!(drift.depth(), MIN_RENOISE_DEPTH);
    }

    // -- Flux ---------------------------------------------------------------
    //
    // The regime the user asked for after "ça frise entre les étapes": no
    // steps at all. What has to be pinned is that it *never turns round* —
    // neither on its own, nor when the dial moves under it.

    /// Collects the next `count` actions as `(phase, level)` pairs, keeping
    /// flux frames distinguishable from the approach that led to them.
    fn walk_tagged(drift: &mut PerpetualDrift, count: usize) -> Vec<(DriftPhase, usize)> {
        (0..count)
            .map(|_| {
                let action = drift.step();
                let level = match action {
                    DriftAction::Descend { diffusion_step, .. } => diffusion_step,
                    DriftAction::Climb { forward_step, .. } => forward_step,
                    DriftAction::Flux { diffusion_step, .. } => diffusion_step,
                };
                // Through the same accessor the dump and the TUI file frames
                // under, so a mutation of it cannot leave these tests green.
                (action.phase(), level)
            })
            .collect()
    }

    /// The regime in one assertion: after the approach, every single frame is a
    /// stationary one at `t*`. Not a short cycle, not a shallow triangle — no
    /// descent and no climb ever again, because a turning point is exactly what
    /// the viewer was feeling.
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

    /// Moving `t*` is the one thing that can make flux move, and it must move
    /// by *walking*: one level a frame, at the tempo the run is already playing
    /// at. A `set_depth` that teleported the latent would put back the jump
    /// this regime exists to remove — and silently, since the level on screen
    /// would be right either way.
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

    /// `[m]` into flux, from either side of `t*`, and back out. The latent is
    /// never re-drawn and never re-interpreted: wherever it is, the drift walks
    /// from there.
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

    /// Both of a flux frame's fields have to be fresh, and the path seed is the
    /// one that is easy to get wrong: the timestep is *constant* here, so a
    /// path seed held across frames would make `reverse_step_seed` return the
    /// same value every frame and the sampler would push the identical
    /// posterior field into the image forever — a fixed direction, not a draw.
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
        // And *within* a frame: the posterior draw and the re-noising field are
        // two independent sources, not one counted twice. Comparing the seeds
        // cannot see this — the sampler transforms its own before drawing, so a
        // drift that handed out the same number twice would still show two
        // different seeds here. Only the fields tell.
        for (path, field) in paths.iter().zip(fields.iter()).take(16) {
            let r = correlation(*path, *field);
            assert!(
                r.abs() < 0.05,
                "a frame's two fields are the same one (r = {r:.3}): the churn \
                 would push twice in a single direction instead of drawing twice"
            );
        }
    }

    /// **The noise level is stationary, measured rather than asserted.**
    ///
    /// The claim of the regime is that a reverse step and a forward increment
    /// cancel *in law*: the latent stays a legitimate `x_{t*}` forever. Getting
    /// this wrong is not a crash, it is a slow bleed — an image that quietly
    /// washes out or saturates over minutes, which no snapshot test would see.
    ///
    /// Run against an oracle instead of a model: the exact posterior-optimal
    /// predictor for a data distribution that is a point mass at `x0`, i.e.
    /// `eps_hat = (x_t - sqrt(alpha_bar)·x0) / sqrt(1 - alpha_bar)`. For that
    /// distribution the true stationary law is known in closed form —
    /// `q(x_t | x0)` — so the churn is checked against theory rather than
    /// against itself.
    #[test]
    fn the_flux_churn_holds_the_law_of_its_level() {
        const N: usize = 8192;
        let level = 64;
        let schedule = crate::LinearNoiseSchedule::new_linear(STEPS, 1e-4, 0.02);
        let signal = schedule.alpha_bar(level).sqrt();
        let spread = (1.0 - schedule.alpha_bar(level)).sqrt();

        let x0: Vec<f32> = (0..N).map(|i| (i as f32 * 0.017).sin() * 0.8).collect();
        // Deliberately *not* started on the marginal: a churn that only holds
        // the level it was handed proves nothing.
        let (mut latent, _) = schedule.add_noise(&x0, level / 2, 0xc0ffee);

        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, level, 0xf1_0000);
        let mut spreads = Vec::new();
        let mut signals = Vec::new();
        let mut changes: Vec<f64> = Vec::new();
        let mut history: Vec<Vec<f32>> = Vec::new();
        let mut frames = 0;
        let mut actions = 0;
        while frames < 400 {
            // A drift that stopped yielding stationary frames — an approach
            // that overshoots `t*` turns flux into a two-frame cycle — would
            // spin here forever. A test that hangs reports nothing.
            actions += 1;
            assert!(
                actions < STEPS + 4_000,
                "the drift stopped holding its level: {frames} stationary frames \
                 in {actions} actions"
            );
            // The approach onto t* is walked by the caller with the same maths;
            // only the stationary part is measured, so it is skipped here.
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
            // The first frames are the walk onto the level; the claim is about
            // the regime it settles into.
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

        // It drifts rather than vibrating: consecutive frames are alike, and
        // frames three hundred apart are not. The residual only — the signal
        // term is pinned to x0 by the oracle and would flatter any lag; what
        // has to renew itself is the noise.
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

    /// A frame is filed under the name of what it did.
    ///
    /// Found from outside the code: a black-box run measured a frame tagged
    /// `descent` that was moving with **churn** amplitude — some 40 % more than
    /// a reverse step at the same level — at the first frame of every plateau,
    /// on all eight settings of the dial it tried. The sequencing was right; the
    /// name was one frame stale, because the caller read `phase()` before
    /// `step()` and `settle_phase` reconciles inside it.
    ///
    /// The whole test battery missed it: every one of these tests already reads
    /// the phase off the action, which is the *correct* notion, while the dump
    /// recorded the other one. Nothing that agrees with itself can catch that,
    /// which is why [`DriftAction::phase`] now exists and everything that names
    /// a frame goes through it.
    #[test]
    fn a_frame_is_named_after_the_action_it_performed_not_the_phase_it_left() {
        let level = 40;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, level, 7);

        // Read both notions on every frame: the phase standing before the step,
        // and the phase of what the step turned out to be.
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

        // The trap itself, pinned so the next caller cannot walk into it: at
        // that frame the phase standing beforehand is the one just *left*.
        // Whoever changes when `settle_phase` runs will see this line and can
        // delete it knowingly — what must never come back is a frame filed
        // under a phase it did not perform.
        assert_ne!(
            walked[approach].0, walked[approach].1,
            "if these ever agree, `settle_phase` moved and the comment above is stale"
        );

        // The same seam on the way up: raising `t*` mid-flux walks a climb, and
        // the frame that resumes the churn is a churn frame. This is where the
        // TUI used to flash one `descente` at the top of the approach.
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
