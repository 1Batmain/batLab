//! Carries the latent along a [`PerpetualDrift`]'s itinerary — the tensors, where
//! `perpetual.rs` holds only the plan. The split is what makes the walk testable:
//! it needs the model only through one question ("what noise in this latent, at
//! this level?"), which a closed-form oracle answers without a GPU.
//!
//! The rule this module enforces, in one line: **every frame, in every regime,
//! calls the model exactly once and publishes a fresh x̂₀.** It exists because the
//! climb is closed-form ([`LinearNoiseSchedule::forward_from`]) and used to call
//! no model, so the x̂₀ pane froze on the descent's last estimate for a whole
//! climb. The fix does not change the trajectory (one departure, one field, for
//! grain coherence — `CLIMB_COHERENCE.md`); it only ALSO asks the model each
//! frame. Cost: one model call per climb frame where there was none — `IMG2IMG_DRIFT.md`.

use super::perpetual::DriftAction;
use super::schedule::{LinearNoiseSchedule, PosteriorVariance};

/// What one frame of a perpetual run produced — both panes, and what it cost.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftFrame {
    /// x_t after the frame: the left pane, the noisy latent.
    pub latent: Vec<f32>,
    /// x̂₀ for that latent: the right pane. NEVER stale (see module header).
    pub x0_hat: Vec<f32>,
    /// Model calls this frame cost — always one, carried out so a caller reports it.
    pub model_calls: usize,
}

/// The model as the walk needs it: ε̂ from a latent at a timestep. One method —
/// the entire coupling between drift and network. Production wraps
/// [`crate::training::predict_epsilon`]; tests wrap a closed-form oracle.
pub trait NoisePredictor {
    fn predict_noise(&mut self, latent: &[f32], diffusion_step: usize) -> Vec<f32>;
}

impl<F> NoisePredictor for F
where
    F: FnMut(&[f32], usize) -> Vec<f32>,
{
    fn predict_noise(&mut self, latent: &[f32], diffusion_step: usize) -> Vec<f32> {
        self(latent, diffusion_step)
    }
}

/// [`NoisePredictor`], but the model call is a future — a browser answers only
/// when the event loop turns ([`crate::Model::predict_async`]).
/// [`DriftWalk::advance_async`] walks the exact same itinerary as
/// [`DriftWalk::advance`], one `.await` apart.
#[allow(async_fn_in_trait)]
pub trait AsyncNoisePredictor {
    async fn predict_noise(&mut self, latent: &[f32], diffusion_step: usize) -> Vec<f32>;
}

/// The latent of an endless run, and the climb departure it needs to remember.
///
/// Holds no model and no GPU handle: it is handed a [`NoisePredictor`] per
/// frame. That is what makes the regime's behaviour — not just its itinerary —
/// something a unit test can pin.
#[derive(Debug, Clone)]
pub struct DriftWalk {
    latent: Vec<f32>,
    /// The state the climb in progress set out from. Snapshotted on the
    /// increment that reports `opens_cycle` and read by every level of that
    /// climb: one departure, one field, revealed progressively. Re-noising
    /// incrementally instead would draw an independent field per frame and make
    /// the ascent crackle (`CLIMB_COHERENCE.md`).
    departure: Option<Vec<f32>>,
    denoise_magnitude: f32,
}

impl DriftWalk {
    /// Opens a walk on `latent` — pure noise at the top of the schedule, or a
    /// real image at `t = 0` (see [`crate::PerpetualOrigin`]).
    pub fn new(latent: Vec<f32>, denoise_magnitude: f32) -> Self {
        Self {
            latent,
            departure: None,
            denoise_magnitude,
        }
    }

    pub fn latent(&self) -> &[f32] {
        &self.latent
    }

    /// Restarts on a new latent (`[r]`). The departure is dropped with it — a
    /// climb reading a pre-reseed snapshot would carry the old image up the schedule.
    pub fn restart_from(&mut self, latent: Vec<f32>) {
        self.latent = latent;
        self.departure = None;
    }

    /// Performs one itinerary action and returns the frame. Every arm ends the
    /// same way — predict, estimate x̂₀, publish — the module's whole claim; a
    /// branch returning a held estimate would be the pause coming back.
    pub fn advance<P: NoisePredictor + ?Sized>(
        &mut self,
        action: DriftAction,
        schedule: &LinearNoiseSchedule,
        predictor: &mut P,
    ) -> DriftFrame {
        match action {
            DriftAction::Descend {
                diffusion_step,
                path_seed,
            } => {
                let epsilon = predictor.predict_noise(&self.latent, diffusion_step);
                let stepped = super::sampler::reverse_step_from_epsilon(
                    schedule,
                    &self.latent,
                    epsilon,
                    diffusion_step,
                    path_seed,
                    self.denoise_magnitude,
                    // Beta variance: the drift keeps the pre-posterior draw.
                    PosteriorVariance::Beta,
                    true,
                );
                self.latent = stepped.latent;
                self.frame(stepped.x0_hat.expect("x0_hat was asked for"))
            }
            DriftAction::Climb {
                forward_step,
                departure_step,
                cycle_seed,
                opens_cycle,
            } => {
                // The cycle just closed: the latent is the settled image, and the
                // departure every level of this climb is formed from.
                if opens_cycle {
                    self.departure = Some(self.latent.clone());
                }
                self.latent = schedule.forward_from(
                    self.departure
                        .as_ref()
                        .expect("a climb always opens with `opens_cycle`"),
                    departure_step,
                    forward_step,
                    cycle_seed,
                );
                // THE FIX: the latent above is arithmetic, but the estimate beside
                // it is not allowed to be — ask the model what it now sees in the
                // re-noised latent, so the image dreams on while it dissolves.
                let epsilon = predictor.predict_noise(&self.latent, forward_step);
                let x0_hat = schedule.x0_estimate(&self.latent, &epsilon, forward_step);
                self.frame(x0_hat)
            }
            DriftAction::Flux {
                diffusion_step,
                path_seed,
                renoise_seed,
            } => {
                let epsilon = predictor.predict_noise(&self.latent, diffusion_step);
                let stepped = super::sampler::reverse_step_from_epsilon(
                    schedule,
                    &self.latent,
                    epsilon,
                    diffusion_step,
                    path_seed,
                    self.denoise_magnitude,
                    // Beta variance: the drift keeps the pre-posterior draw.
                    PosteriorVariance::Beta,
                    true,
                );
                // One level down, exactly that one back on: the noise budget is
                // stationary by construction.
                self.latent = schedule.forward_step(&stepped.latent, diffusion_step, renoise_seed);
                self.frame(stepped.x0_hat.expect("x0_hat was asked for"))
            }
        }
    }

    /// The async twin of [`Self::advance`], for a browser. Identical line for line
    /// except `predict_noise` becomes an `.await` — it IS the same walk, so a web
    /// run produces the same frames (`advance_async_matches_advance_bit_for_bit`).
    /// Kept side by side, not one calling the other, so the native path needs no executor.
    pub async fn advance_async<P: AsyncNoisePredictor + ?Sized>(
        &mut self,
        action: DriftAction,
        schedule: &LinearNoiseSchedule,
        predictor: &mut P,
    ) -> DriftFrame {
        match action {
            DriftAction::Descend {
                diffusion_step,
                path_seed,
            } => {
                let epsilon = predictor.predict_noise(&self.latent, diffusion_step).await;
                let stepped = super::sampler::reverse_step_from_epsilon(
                    schedule,
                    &self.latent,
                    epsilon,
                    diffusion_step,
                    path_seed,
                    self.denoise_magnitude,
                    // Beta variance: the drift keeps the pre-posterior draw.
                    PosteriorVariance::Beta,
                    true,
                );
                self.latent = stepped.latent;
                self.frame(stepped.x0_hat.expect("x0_hat was asked for"))
            }
            DriftAction::Climb {
                forward_step,
                departure_step,
                cycle_seed,
                opens_cycle,
            } => {
                if opens_cycle {
                    self.departure = Some(self.latent.clone());
                }
                self.latent = schedule.forward_from(
                    self.departure
                        .as_ref()
                        .expect("a climb always opens with `opens_cycle`"),
                    departure_step,
                    forward_step,
                    cycle_seed,
                );
                let epsilon = predictor.predict_noise(&self.latent, forward_step).await;
                let x0_hat = schedule.x0_estimate(&self.latent, &epsilon, forward_step);
                self.frame(x0_hat)
            }
            DriftAction::Flux {
                diffusion_step,
                path_seed,
                renoise_seed,
            } => {
                let epsilon = predictor.predict_noise(&self.latent, diffusion_step).await;
                let stepped = super::sampler::reverse_step_from_epsilon(
                    schedule,
                    &self.latent,
                    epsilon,
                    diffusion_step,
                    path_seed,
                    self.denoise_magnitude,
                    // Beta variance: the drift keeps the pre-posterior draw.
                    PosteriorVariance::Beta,
                    true,
                );
                self.latent = schedule.forward_step(&stepped.latent, diffusion_step, renoise_seed);
                self.frame(stepped.x0_hat.expect("x0_hat was asked for"))
            }
        }
    }

    fn frame(&self, x0_hat: Vec<f32>) -> DriftFrame {
        DriftFrame {
            latent: self.latent.clone(),
            x0_hat,
            model_calls: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{PerpetualDrift, PerpetualRegime};

    const STEPS: usize = 256;
    const N: usize = 1024;

    fn schedule() -> LinearNoiseSchedule {
        LinearNoiseSchedule::new_linear(STEPS, 1e-4, 0.02)
    }

    /// A clean image to stand in for the dataset.
    fn clean() -> Vec<f32> {
        (0..N).map(|i| (i as f32 * 0.017).sin() * 0.8).collect()
    }

    /// How far the stand-in model falls short of the exact predictor. See
    /// [`oracle`] — a *perfect* one is useless here.
    const UNDER_CONFIDENCE: f32 = 0.9;

    /// A deliberately IMPERFECT ε̂ predictor. The exact posterior-optimal one for a
    /// point mass at `x0` is degenerate here: through `x0_estimate` it returns the
    /// constant `x0` whatever the latent, so a walk that never called the model
    /// would look identical and the frozen-pane test could not fail. Scaling by
    /// `UNDER_CONFIDENCE` gives an estimate that tracks the latent it was read from.
    fn oracle(
        schedule: &LinearNoiseSchedule,
        x0: Vec<f32>,
    ) -> impl FnMut(&[f32], usize) -> Vec<f32> {
        let alpha_bars: Vec<(f32, f32)> = (0..schedule.len())
            .map(|t| {
                let bar = schedule.alpha_bar(t);
                (bar.sqrt(), (1.0 - bar).sqrt())
            })
            .collect();
        move |latent: &[f32], t: usize| {
            let (signal, spread) = alpha_bars[t.min(alpha_bars.len() - 1)];
            latent
                .iter()
                .zip(x0.iter())
                .map(|(x, c)| UNDER_CONFIDENCE * (x - signal * c) / spread.max(1e-6))
                .collect()
        }
    }

    fn mean_abs_delta(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() as f64)
            .sum::<f64>()
            / a.len() as f64
    }

    /// The regression this module exists for: the climb was closed-form and called
    /// no model, so x̂₀ froze for the whole ascent (`t_r+1` identical frames, then a
    /// jump). Asserted on the FRAMES the viewer sees, not a call counter — a walk
    /// that called the model then published a held estimate would fool a counter.
    #[test]
    fn x0_never_repeats_itself_between_two_frames_of_a_climb() {
        let schedule = schedule();
        let x0 = clean();
        let mut predict = oracle(&schedule, x0.clone());

        let depth = 24;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 0x5eed);
        let mut walk = DriftWalk::new(schedule.sample_noise(N, drift.initial_noise_seed()), 1.0);

        // Collected until a FOURTH climb opens: a climb is complete only once the
        // next begins, else a truncated one is compared on length not content.
        let mut climbs: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut actions = 0;
        while climbs.len() < 4 {
            actions += 1;
            assert!(actions < 10_000, "the drift stopped closing cycles");
            let action = drift.step();
            let opening = matches!(action, DriftAction::Climb { opens_cycle: true, .. });
            let climbing = matches!(action, DriftAction::Climb { .. });
            let frame = walk.advance(action, &schedule, &mut predict);
            assert_eq!(frame.model_calls, 1, "every frame costs exactly one call");
            if opening {
                climbs.push(Vec::new());
            }
            if climbing && let Some(current) = climbs.last_mut() {
                current.push(frame.x0_hat);
            }
        }

        for (index, climb) in climbs[..3].iter().enumerate() {
            assert_eq!(
                climb.len(),
                depth + 1,
                "climb {index} was not walked in full"
            );
            for (frame, pair) in climb.windows(2).enumerate() {
                let delta = mean_abs_delta(&pair[0], &pair[1]);
                assert!(
                    delta > 1e-6,
                    "climb {index}: x̂₀ is identical between frames {frame} and \
                     {} (mean |Δ| = {delta:.2e}) — that is the pause",
                    frame + 1
                );
            }
        }
    }

    /// The same claim over a whole cycle (descent AND climb), and the one place it
    /// does not hold, stated exactly. What is left is ONE repeated frame per cycle,
    /// not a stale estimate: the climb's last increment lands on `t_r` and the next
    /// descent re-reads that same latent at that same level (a reverse step derives
    /// x̂₀ from `x_t` before stepping), so two frames sit on the same instant — 33 ms
    /// against a `t_r=16` freeze's 570 ms. Flux has no turn, asserted separately.
    #[test]
    fn the_only_estimate_a_cycle_repeats_is_the_one_at_its_turn() {
        let schedule = schedule();
        let mut predict = oracle(&schedule, clean());
        let depth = 16;

        for regime in [PerpetualRegime::Wander, PerpetualRegime::Breathe] {
            let mut drift = PerpetualDrift::new(STEPS, regime, depth, 0xbeef);
            let mut walk = DriftWalk::new(schedule.sample_noise(N, drift.initial_noise_seed()), 1.0);
            let cycles = 6;
            let mut previous: Option<Vec<f32>> = None;
            // The phase of the frame before this one — a repeat is only allowed
            // on the descent that resumes straight after a climb.
            let mut previously_climbing = false;
            let mut repeats = Vec::new();
            for index in 0..(STEPS + cycles * 2 * (depth + 1)) {
                let action = drift.step();
                let descending = matches!(action, DriftAction::Descend { .. });
                let frame = walk.advance(action, &schedule, &mut predict);
                if index >= STEPS
                    && let Some(previous) = previous.as_ref()
                    && mean_abs_delta(previous, &frame.x0_hat) <= 1e-9
                {
                    assert!(
                        previously_climbing && descending,
                        "{regime:?}: frame {index} repeated the previous estimate \
                         somewhere other than the turn of the cycle — that is a \
                         stale pane"
                    );
                    repeats.push(index);
                }
                previously_climbing = matches!(action, DriftAction::Climb { .. });
                previous = Some(frame.x0_hat);
            }
            // Counted against the cycles the drift says it closed, not against
            // the frame budget: breathing floors at `t_r/2`, so its cycle is
            // shorter and it turns round more often in the same number of
            // frames.
            let closed = drift.cycle();
            assert!(closed >= 4, "{regime:?}: only {closed} cycles walked");
            assert!(
                repeats.len() <= closed,
                "{regime:?}: {} repeats over {closed} cycles — more than one per \
                 turn means something else is holding an estimate",
                repeats.len()
            );
        }
    }

    /// Flux never turns round, so it has no excuse at all: over hundreds of
    /// stationary frames, not one repeats the estimate before it.
    #[test]
    fn the_flux_churn_never_shows_the_same_estimate_twice_running() {
        let schedule = schedule();
        let mut predict = oracle(&schedule, clean());
        let level = 48;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Flux, level, 0xbeef);
        let mut walk = DriftWalk::new(schedule.sample_noise(N, drift.initial_noise_seed()), 1.0);

        let mut previous: Option<Vec<f32>> = None;
        let mut checked = 0;
        for index in 0..(STEPS + 400) {
            let frame = walk.advance(drift.step(), &schedule, &mut predict);
            if index >= STEPS
                && let Some(previous) = previous.as_ref()
            {
                assert!(
                    mean_abs_delta(previous, &frame.x0_hat) > 1e-9,
                    "flux frame {index} is the previous frame over again"
                );
                checked += 1;
            }
            previous = Some(frame.x0_hat);
        }
        assert!(checked >= 390, "only {checked} frames compared");
    }

    /// The fix must not have moved the trajectory: asking the model is a READ, and
    /// a read that perturbed the state would trade the pause for a drift that no
    /// longer holds its noise level. Checked to the bit: every climb frame's latent
    /// must equal `forward_from(departure, …)`.
    #[test]
    fn asking_the_model_during_the_climb_does_not_move_the_latent() {
        let schedule = schedule();
        let mut predict = oracle(&schedule, clean());
        let depth = 20;
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, depth, 7);
        let mut walk = DriftWalk::new(schedule.sample_noise(N, drift.initial_noise_seed()), 1.0);

        let mut departure: Option<Vec<f32>> = None;
        let mut checked = 0;
        for _ in 0..(STEPS + 4 * 2 * (depth + 1)) {
            let action = drift.step();
            if let DriftAction::Climb { opens_cycle: true, .. } = action {
                departure = Some(walk.latent().to_vec());
            }
            let frame = walk.advance(action, &schedule, &mut predict);
            if let DriftAction::Climb {
                forward_step,
                departure_step,
                cycle_seed,
                ..
            } = action
                && let Some(departure) = departure.as_ref()
            {
                let expected =
                    schedule.forward_from(departure, departure_step, forward_step, cycle_seed);
                assert_eq!(
                    frame.latent, expected,
                    "the climb's latent is no longer the closed form from its departure"
                );
                checked += 1;
            }
        }
        assert!(checked > 4 * depth, "only {checked} climb frames checked");
    }

    /// A re-seed drops the climb departure with the latent. Keeping it would
    /// carry the *previous* image up the schedule on the next cycle — the old
    /// picture reappearing inside the new one.
    #[test]
    fn restarting_forgets_the_departure_of_the_climb_it_interrupted() {
        let schedule = schedule();
        let mut predict = oracle(&schedule, clean());
        let mut drift = PerpetualDrift::new(STEPS, PerpetualRegime::Wander, 12, 7);
        let mut walk = DriftWalk::new(schedule.sample_noise(N, drift.initial_noise_seed()), 1.0);

        // Reach a climb, so a departure is definitely held.
        for _ in 0..(STEPS + 3) {
            walk.advance(drift.step(), &schedule, &mut predict);
        }
        assert!(walk.departure.is_some(), "the test needs a climb in flight");

        let fresh = clean();
        walk.restart_from(fresh.clone());
        assert_eq!(walk.latent(), fresh.as_slice());
        assert!(walk.departure.is_none());
    }

    /// Wraps a synchronous oracle as an [`AsyncNoisePredictor`] whose future is
    /// already resolved — the cheapest way to drive [`DriftWalk::advance_async`]
    /// in a test without a GPU or an event loop.
    struct SyncAsync<F>(F);
    impl<F> AsyncNoisePredictor for SyncAsync<F>
    where
        F: FnMut(&[f32], usize) -> Vec<f32>,
    {
        async fn predict_noise(&mut self, latent: &[f32], diffusion_step: usize) -> Vec<f32> {
            (self.0)(latent, diffusion_step)
        }
    }

    /// The web path's safety net: `advance_async` must produce the IDENTICAL frame
    /// `advance` does for the identical action, else a browser drift silently
    /// diverges. Run over every regime (descents, climbs, the turn, flux), bit-for-bit
    /// on latent and x̂₀ every frame.
    #[test]
    fn advance_async_matches_advance_bit_for_bit() {
        let schedule = schedule();
        for regime in [
            PerpetualRegime::Wander,
            PerpetualRegime::Breathe,
            PerpetualRegime::Flux,
        ] {
            let depth = 20;
            let mut drift_sync = PerpetualDrift::new(STEPS, regime, depth, 0x5eed);
            let mut drift_async = PerpetualDrift::new(STEPS, regime, depth, 0x5eed);
            let mut walk_sync =
                DriftWalk::new(schedule.sample_noise(N, drift_sync.initial_noise_seed()), 1.0);
            let mut walk_async =
                DriftWalk::new(schedule.sample_noise(N, drift_async.initial_noise_seed()), 1.0);
            let mut predict_sync = oracle(&schedule, clean());
            let mut predict_async = SyncAsync(oracle(&schedule, clean()));

            for index in 0..(STEPS + 6 * 2 * (depth + 1)) {
                let action_sync = drift_sync.step();
                let action_async = drift_async.step();
                assert_eq!(
                    action_sync, action_async,
                    "{regime:?}: itineraries diverged at frame {index}"
                );
                let frame_sync = walk_sync.advance(action_sync, &schedule, &mut predict_sync);
                let frame_async = pollster::block_on(walk_async.advance_async(
                    action_async,
                    &schedule,
                    &mut predict_async,
                ));
                assert_eq!(
                    frame_sync, frame_async,
                    "{regime:?}: async frame {index} differs from the sync frame"
                );
            }
        }
    }
}
