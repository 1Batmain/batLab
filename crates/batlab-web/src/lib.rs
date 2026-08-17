//! File purpose: The WebAssembly front end — the engine running in a visitor's
//! browser on their own GPU (WebGPU), for the portfolio.
//!
//! This is the crate the whole engine/interface boundary in `batlab_core` was
//! drawn for. It holds **no diffusion maths of its own**: every reverse step is
//! [`batlab_core::reverse_step_from_epsilon`], every drift frame is
//! [`batlab_core::DriftWalk::advance_async`], the input is composed by
//! [`batlab_core::compose_diffusion_input`], and the schedule is the engine's.
//! A web run that diverged from a native one would be a silent failure, so the
//! rule here is: reuse, never re-derive. The one thing this crate adds is the
//! *driving* — turning the engine's async step into `requestAnimationFrame`
//! ticks and painting x̂₀ onto a canvas.
//!
//! # Why async, and why the latent still round-trips
//!
//! [`batlab_core::Model::predict`] blocks on its ε̂ readback. In a browser the
//! thread it would block is the one that has to turn the event loop for the GPU
//! to answer — a deadlock. So the engine grew [`batlab_core::Model::predict_async`]
//! and [`batlab_core::DriftWalk::advance_async`], which *await* the readback,
//! and this crate drives them from JavaScript, one frame at a time.
//!
//! Keeping the latent GPU-resident across all 256 steps (project task #19) would
//! remove the per-step ε̂ readback entirely, but it means re-expressing the
//! schedule maths in WGSL — precisely the divergence the paragraph above forbids
//! without a bit-agreement test to hold it. Measured against the target here (a
//! contemplative piece throttled to display rate), the round-trip is not the
//! wall — see the crate README. The async path is the correct, verifiable first
//! port; GPU-residency is a measured optimisation left on the table, not a
//! prerequisite.
//!
//! On a native target this crate is **empty** (the `cfg` below), so the
//! workspace builds and tests without ever pulling a web dependency.

#![cfg(target_arch = "wasm32")]

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use batlab_core::{
    AsyncNoisePredictor, CheckpointWeights, DriftWalk, GpuContext, GpuLimitsProfile,
    LinearNoiseSchedule, Model, ModelConfig, PerpetualDrift, PerpetualRegime, PosteriorVariance,
    Stats, compose_diffusion_input, compute_inferred_input, decode_u8, reverse_step_async,
};
use wasm_bindgen::prelude::*;

/// The production schedule, read from the engine so the browser samples the
/// exact chain the model was trained and sampled with.
const STEPS: usize = batlab_core::DIFFUSION_SCHEDULE_STEPS;
const BETA_START: f32 = batlab_core::DIFFUSION_BETA_START;
const BETA_END: f32 = batlab_core::DIFFUSION_BETA_END;

/// The denoise magnitude for both modes — the config's inference default.
const DENOISE_MAGNITUDE: f32 = 1.0;

/// The reverse-step variance the browser sampler injects. Kept in step with the
/// model's `config_file` inference block; the single formula behind both spellings
/// lives in [`batlab_core::PosteriorVariance`], so the page cannot drift from the
/// native sampler. `Beta` is DDPM's larger variance and the historical default.
const POSTERIOR_VARIANCE: PosteriorVariance = PosteriorVariance::Beta;

/// Same fold as [`batlab_core`]'s reverse-chain seed: distinct runs get distinct
/// noise fields, avalanched downstream by the engine's `gaussian_at`.
const SEED_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
/// The same constant `sample_diffusion` folds into the base latent's seed, so a
/// web inference run's opening noise matches path 0 of the native sampler.
const BASE_NOISE_FOLD: u64 = 0xa5a5_5a5a_0123_4567;

/// Which piece the page is showing — the two modes of the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Generate an image from pure noise, showing the denoising unfold.
    Inference,
    /// The endless img2img drift away from a real dataset image.
    Errance,
}

impl Mode {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "inference" | "inférence" | "inferer" => Some(Mode::Inference),
            "errance" | "wander" | "drift" => Some(Mode::Errance),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Seed images — a small subset of the drift's seed dataset, embedded as bytes.
// ---------------------------------------------------------------------------

/// The `seeds.bin` asset: a 16-byte header (`count, w, h, c` as u32 little
/// endian) then `count·w·h·c` bytes of u8 pixel data, channels interleaved —
/// the same u8 payload a `.batraw` file carries, decoded to `[-1, 1]` by the
/// engine's own [`decode_u8`] so a browser seed image is the very tensor the
/// native drift would set out from.
struct SeedImages {
    width: u32,
    height: u32,
    channels: u32,
    count: usize,
    data: Vec<u8>,
}

impl SeedImages {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 {
            return Err("seeds.bin is too short for its header".to_string());
        }
        let u32_at = |i: usize| {
            u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
        };
        let count = u32_at(0) as usize;
        let width = u32_at(4);
        let height = u32_at(8);
        let channels = u32_at(12);
        let per_image = (width * height * channels) as usize;
        let expected = 16 + count * per_image;
        if per_image == 0 || count == 0 {
            return Err("seeds.bin declares no images".to_string());
        }
        if bytes.len() < expected {
            return Err(format!(
                "seeds.bin declares {count} images of {per_image} bytes but is only {} bytes",
                bytes.len()
            ));
        }
        Ok(Self {
            width,
            height,
            channels,
            count,
            data: bytes[16..expected].to_vec(),
        })
    }

    /// Decodes image `index` (wrapped into range) to a `[-1, 1]` tensor, using
    /// the engine's exact u8 decode.
    fn image(&self, index: usize) -> Vec<f32> {
        let per_image = (self.width * self.height * self.channels) as usize;
        let start = (index % self.count) * per_image;
        self.data[start..start + per_image]
            .iter()
            .map(|&byte| decode_u8(byte))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The engine — everything behind the wasm handle.
// ---------------------------------------------------------------------------

/// Composes `[x_t | timestep]` and asks the model for ε̂, awaiting the readback.
///
/// This is [`batlab_core::predict_epsilon`] made async: same composition, same
/// timestep embedding, same model. Borrowing the model and the schedule as
/// disjoint fields is what lets the drift's `advance_async` call it while it
/// also holds the walk.
struct ModelPredictor<'a> {
    model: &'a mut Model,
    schedule: &'a LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
}

impl AsyncNoisePredictor for ModelPredictor<'_> {
    async fn predict_noise(&mut self, latent: &[f32], diffusion_step: usize) -> Vec<f32> {
        let timestep_channels = self.input_channels.saturating_sub(self.signal_channels);
        let features = self.schedule.timestep_embedding(diffusion_step, timestep_channels);
        let input =
            compose_diffusion_input(latent, self.input_channels, self.signal_channels, &features);
        self.model.predict_async(&input).await
    }
}

struct Engine {
    model: Model,
    /// A handle on the device, kept so a frame can run its GPU work inside a
    /// validation error scope (see [`GpuContext::guarded`]) without borrowing the
    /// model. This is the browser's only defence against a silently-invalid
    /// pipeline: without it, a frame whose shaders the browser rejected would
    /// paint zeroed ε̂ — a model that drifts into saturated noise — and say
    /// nothing. See `docs/reports/WEB_PORT.md`, "le bug de l'inférence".
    gpu: Arc<GpuContext>,
    schedule: LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
    width: u32,
    height: u32,
    output_len: usize,
    seeds: SeedImages,
    seed_counter: u64,

    /// The last frame's tensor statistics, for the on-page diagnostics line and
    /// the console. An ε̂ whose std collapses to ~0 is the whole tell of the
    /// inference bug: with ε̂ ≈ 0, x̂₀ = x_t/√ᾱ is the raw latent, saturated at
    /// every t. A healthy ε̂ has std ≈ 1.
    last_diag: String,
    /// A GPU validation error caught this session, surfaced on the page so it can
    /// never again be swallowed.
    last_error: Option<String>,
    /// Frames diagnosed so far, for throttling the console log.
    diag_frames: u64,

    mode: Mode,

    // Inference run state.
    inf_latent: Vec<f32>,
    inf_step: usize,
    inf_path_seed: u64,
    /// FNV-1a of the opening latent, and of the finished image. Two numbers on
    /// screen that answer, from outside, a question reasoning could not: does a
    /// new seed actually draw a different noise field, and does a different
    /// noise field actually end on a different picture.
    inf_noise_print: u32,
    /// The resolved image, held once the chain reaches t=0. Inference is a
    /// *finite* piece: it ends on its picture and stays there, and only
    /// « Nouveau bruit » opens the next descent. While it is `Some`, a frame
    /// costs no GPU work at all — it hands this buffer back.
    inf_resolved: Option<Vec<f32>>,

    // Errance (perpetual drift) state.
    drift: PerpetualDrift,
    walk: DriftWalk,
    depth: usize,
    regime: PerpetualRegime,
    image_index: usize,
}

impl Engine {
    fn next_seed(&mut self) -> u64 {
        self.seed_counter = self.seed_counter.wrapping_add(1);
        self.seed_counter.wrapping_mul(SEED_GAMMA)
    }

    /// Restart the inference run from fresh noise under a new seed — matching
    /// path 0 of [`batlab_core::sample_diffusion`]: the base latent is drawn
    /// from `seed ^ BASE_NOISE_FOLD`, and the path seed is the seed itself.
    fn start_new_inference(&mut self) {
        let seed = self.next_seed();
        self.inf_latent = self
            .schedule
            .sample_noise(self.output_len, seed ^ BASE_NOISE_FOLD);
        self.inf_path_seed = seed;
        self.inf_step = 0;
        self.inf_resolved = None;
        self.inf_noise_print = fingerprint(&self.inf_latent);
    }

    /// (Re)open the drift on seed image `image_index`, under a fresh seed.
    fn start_new_errance(&mut self) {
        let seed = self.next_seed();
        let image = self.seeds.image(self.image_index);
        self.drift = PerpetualDrift::from_image(STEPS, self.regime, self.depth, seed);
        self.walk = DriftWalk::new(image, DENOISE_MAGNITUDE);
    }

    /// One inference frame: predict ε̂, take one reverse step, display x̂₀. When
    /// the chain reaches the image, the run **stops on it** — the picture is the
    /// end of the piece, and only « Nouveau bruit » opens the next descent.
    ///
    /// The reverse step is [`reverse_step_async`] — the engine's own recursion,
    /// not a copy of it. A loop that composed `[x_t | timestep]` or seeded the
    /// posterior its own way could diverge from the native sampler unseen; going
    /// through the one function it cannot (guarded natively by
    /// `the_async_reverse_step_matches_the_sync_one`).
    async fn inference_frame(&mut self) -> Vec<f32> {
        // Chain finished: hold the picture. No dispatch, no readback — the page
        // idles on the result instead of burning the visitor's GPU on a loop.
        if let Some(resolved) = self.inf_resolved.as_ref() {
            return resolved.clone();
        }
        let diffusion_step = STEPS - 1 - self.inf_step;
        let stepped = reverse_step_async(
            &mut self.model,
            &self.schedule,
            self.input_channels,
            self.signal_channels,
            &self.inf_latent,
            diffusion_step,
            self.inf_path_seed,
            DENOISE_MAGNITUDE,
            POSTERIOR_VARIANCE,
            true,
        )
        .await;
        let x0_hat = stepped.x0_hat.expect("x0_hat was asked for");
        self.record_diag("inference", diffusion_step, Some(&stepped.predicted_noise), &x0_hat);
        self.inf_latent = stepped.latent;
        self.inf_step += 1;
        if self.inf_step >= STEPS {
            // The image is resolved. Keep it, and stop stepping until the
            // visitor asks for another one.
            self.inf_resolved = Some(x0_hat.clone());
        }
        x0_hat
    }

    /// Fold this frame's tensor statistics into the diagnostics line and log them
    /// once a second or so. ε̂ is the one that matters: its std tells a healthy
    /// forward pass (≈ 1) from a dead one (≈ 0), which is the whole diagnosis of
    /// the inference bug. `epsilon` is `None` for the drift frame, whose walker
    /// does not hand ε̂ back; the x̂₀ stats still speak there.
    fn record_diag(&mut self, mode: &str, t: usize, epsilon: Option<&[f32]>, x0_hat: &[f32]) {
        let x0 = Stats::of(x0_hat);
        let eps_part = match epsilon {
            Some(eps) => {
                let s = Stats::of(eps);
                format!(
                    "ε̂ std={:.3} mean={:.3} [{:.2},{:.2}]{}",
                    s.std,
                    s.mean,
                    s.min,
                    s.max,
                    if s.non_finite > 0 {
                        format!(" ⚠{}×NaN/Inf", s.non_finite)
                    } else {
                        String::new()
                    },
                )
            }
            None => "ε̂ n/a".to_string(),
        };
        self.last_diag = format!(
            "{mode} t={t} · {eps_part} · x̂₀ std={:.3} mean={:.3} [{:.2},{:.2}]",
            x0.std, x0.mean, x0.min, x0.max,
        );
        // First frames log every step (that is where a dead ε̂ is caught on
        // reload); after that, throttle so the console stays readable.
        if self.diag_frames < 6 || self.diag_frames % 64 == 0 {
            log::info!("[batlab] {}", self.last_diag);
        }
        self.diag_frames += 1;
    }

    /// One drift frame: exactly [`DriftWalk::advance_async`] on the itinerary the
    /// [`PerpetualDrift`] hands out. The two panes of the TUI collapse to one
    /// here — the page shows x̂₀ alone, the perpetual default.
    async fn errance_frame(&mut self) -> Vec<f32> {
        let action = self.drift.step();
        let mut predictor = ModelPredictor {
            model: &mut self.model,
            schedule: &self.schedule,
            input_channels: self.input_channels,
            signal_channels: self.signal_channels,
        };
        let frame = self
            .walk
            .advance_async(action, &self.schedule, &mut predictor)
            .await;
        let depth = self.depth;
        self.record_diag("errance", depth, None, &frame.x0_hat);
        frame.x0_hat
    }

    /// Apply any controls the UI queued since the last frame. Never touches the
    /// GPU — cheap, and safe to run at the top of a frame.
    fn apply_pending(&mut self, pending: &Pending) {
        if let Some(mode) = pending.mode.take() {
            if mode != self.mode {
                self.mode = mode;
                match mode {
                    Mode::Inference => self.start_new_inference(),
                    Mode::Errance => self.start_new_errance(),
                }
            }
        }
        if let Some(regime) = pending.regime.take() {
            if regime != self.regime {
                self.regime = regime;
                // Walk the ring rather than teleport: the latent stays put and
                // the new regime approaches its own level from where it is.
                while self.drift.regime() != regime {
                    self.drift.toggle_regime();
                }
                self.depth = self.drift.depth();
            }
        }
        let delta = pending.depth_delta.replace(0);
        if delta != 0 {
            self.drift.nudge_depth(delta);
            self.depth = self.drift.depth();
        }
        if pending.reseed.replace(false) {
            match self.mode {
                Mode::Inference => self.start_new_inference(),
                Mode::Errance => {
                    // `[r]`: the next dataset image, opened afresh.
                    self.image_index = self.image_index.wrapping_add(1);
                    self.start_new_errance();
                }
            }
        }
    }

    /// A short status line for the overlay.
    fn status(&self) -> String {
        match self.mode {
            // The seed is on screen in both states, and that is deliberate: it
            // is the only way to tell « the page reopens on the same noise »
            // from « the model answers the same thing whatever the noise » —
            // two very different bugs that look identical from the outside.
            Mode::Inference if self.inf_resolved.is_some() => {
                let image_print = self
                    .inf_resolved
                    .as_deref()
                    .map(fingerprint)
                    .unwrap_or_default();
                format!(
                    "Inférence terminée · graine {:08x} · bruit {:08x} · image {image_print:08x} · « Nouveau bruit »",
                    self.inf_path_seed as u32,
                    self.inf_noise_print,
                )
            }
            Mode::Inference => {
                let diffusion_step = STEPS.saturating_sub(1 + self.inf_step);
                format!(
                    "Inférence · graine {:08x} · bruit {:08x} · t={diffusion_step} → 0",
                    self.inf_path_seed as u32,
                    self.inf_noise_print,
                )
            }
            Mode::Errance => {
                let (counter_name, counter) = self.drift.counter();
                format!(
                    "Errance · {} · {}={} · {counter_name} {counter}",
                    self.regime.label(),
                    self.regime.depth_label(),
                    self.depth,
                )
            }
        }
    }

    /// The frame's GPU work, dispatched by mode. Split out so [`Engine::step`]
    /// can run *exactly this* inside a validation error scope.
    async fn frame_x0(&mut self) -> Vec<f32> {
        match self.mode {
            Mode::Inference => self.inference_frame().await,
            Mode::Errance => self.errance_frame().await,
        }
    }

    async fn step(&mut self, pending: &Pending) -> (Vec<u8>, String, String) {
        self.apply_pending(pending);
        // Run the frame under a validation error scope. A shader the browser
        // rejects, or a limit tripped at dispatch, is captured here and put on
        // the page instead of silently zeroing ε̂ — the failure mode the whole
        // instrumentation exists to make impossible to miss.
        let gpu = Arc::clone(&self.gpu);
        let x0_hat = match gpu.guarded(self.frame_x0()).await {
            Ok(x0) => x0,
            Err(err) => {
                log::error!("[batlab] WebGPU validation error in a frame: {err}");
                self.last_error = Some(err);
                vec![0.0; self.output_len]
            }
        };
        let rgba = to_rgba(&x0_hat, self.width, self.height, self.signal_channels);
        (rgba, self.status(), self.diagnostics())
    }

    /// The diagnostics line for the page: a caught GPU error takes precedence
    /// over the per-frame statistics — an error means the numbers are moot.
    fn diagnostics(&self) -> String {
        match &self.last_error {
            Some(err) => format!("⚠ WebGPU: {err}"),
            None => self.last_diag.clone(),
        }
    }
}

/// Maps a `[-1, 1]` tensor (channels interleaved) to a tightly packed RGBA byte
/// buffer for `ImageData`. A single-channel model is shown as greyscale.
fn to_rgba(x0: &[f32], width: u32, height: u32, channels: usize) -> Vec<u8> {
    let pixels = (width * height) as usize;
    let mut out = vec![0u8; pixels * 4];
    let to_u8 = |v: f32| (((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round()) as u8;
    for pixel in 0..pixels {
        let base = pixel * channels;
        let (r, g, b) = if channels >= 3 {
            (
                to_u8(x0[base]),
                to_u8(x0[base + 1]),
                to_u8(x0[base + 2]),
            )
        } else {
            let v = to_u8(x0.get(base).copied().unwrap_or(0.0));
            (v, v, v)
        };
        out[pixel * 4] = r;
        out[pixel * 4 + 1] = g;
        out[pixel * 4 + 2] = b;
        out[pixel * 4 + 3] = 255;
    }
    out
}

// ---------------------------------------------------------------------------
// Controls — queued from the UI, drained by the engine at each frame.
// ---------------------------------------------------------------------------

/// UI intents, held in `Cell`s so control methods never borrow the engine (and
/// so can be called while a `step()` future is in flight without a re-borrow).
#[derive(Default)]
struct Pending {
    reseed: Cell<bool>,
    mode: Cell<Option<Mode>>,
    regime: Cell<Option<PerpetualRegime>>,
    depth_delta: Cell<i32>,
}

// ---------------------------------------------------------------------------
// The wasm handle.
// ---------------------------------------------------------------------------

/// The object JavaScript holds: build it with [`BatDiffusion::create`], then
/// call [`BatDiffusion::step`] once per animation frame and paint the returned
/// RGBA bytes onto a canvas.
#[wasm_bindgen]
pub struct BatDiffusion {
    engine: Rc<RefCell<Engine>>,
    pending: Rc<Pending>,
    status: Rc<RefCell<String>>,
    diag: Rc<RefCell<String>>,
}

#[wasm_bindgen]
impl BatDiffusion {
    /// Build the engine: open a WebGPU device under the **web limits profile**
    /// (the browser baseline — 256 MiB/buffer, 128 MiB/binding), build the model
    /// from its config, load its weights, and parse the seed images.
    ///
    /// `config_bytes` is the model's `config_file`; `weights` a checkpoint;
    /// `seeds` the `seeds.bin` asset. All three are handed in from `fetch`, so
    /// the wasm module carries no model of its own and the same binary drives any
    /// model of this shape.
    ///
    /// Rejects (a JS exception the caller can `catch`) if WebGPU is unavailable
    /// or the assets do not parse — the page shows a message rather than a blank
    /// canvas.
    pub async fn create(
        config_bytes: Vec<u8>,
        weights: Vec<u8>,
        seeds: Vec<u8>,
    ) -> Result<BatDiffusion, JsValue> {
        let engine = build_engine(config_bytes, weights, seeds)
            .await
            .map_err(|err| JsValue::from_str(&err))?;
        let status = engine.status();
        let diag = engine.diagnostics();
        Ok(BatDiffusion {
            engine: Rc::new(RefCell::new(engine)),
            pending: Rc::new(Pending::default()),
            status: Rc::new(RefCell::new(status)),
            diag: Rc::new(RefCell::new(diag)),
        })
    }

    /// Advance one frame and resolve to the RGBA bytes of x̂₀ (`width·height·4`),
    /// ready for `new ImageData(bytes, width, height)`.
    ///
    /// Returns a `Promise` explicitly (rather than being an `async fn`) so the
    /// borrow of the engine lives inside a `'static` future; the UI awaits it
    /// once per `requestAnimationFrame` tick.
    pub fn step(&self) -> js_sys::Promise {
        let engine = self.engine.clone();
        let pending = self.pending.clone();
        let status = self.status.clone();
        let diag = self.diag.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let (rgba, line, diag_line) = {
                let mut engine = engine.borrow_mut();
                engine.step(&pending).await
            };
            *status.borrow_mut() = line;
            *diag.borrow_mut() = diag_line;
            Ok(js_sys::Uint8Array::from(rgba.as_slice()).into())
        })
    }

    /// The pane width in pixels (the model's own image width).
    pub fn width(&self) -> u32 {
        self.engine.borrow().width
    }

    /// The pane height in pixels.
    pub fn height(&self) -> u32 {
        self.engine.borrow().height
    }

    /// The last frame's status line, for an overlay.
    pub fn status(&self) -> String {
        self.status.borrow().clone()
    }

    /// The last frame's diagnostics line — ε̂ / x̂₀ statistics, or a caught
    /// WebGPU error. This is what turns "the image is wrong" into "ε̂ std is
    /// 0.000, the forward pass is dead", the difference between guessing and
    /// knowing. Shown under the canvas.
    pub fn diagnostics(&self) -> String {
        self.diag.borrow().clone()
    }

    /// Switch mode: `"inference"` or `"errance"`.
    pub fn set_mode(&self, mode: &str) {
        if let Some(mode) = Mode::parse(mode) {
            self.pending.mode.set(Some(mode));
        }
    }

    /// Set the drift regime: `"wander"`, `"breathe"`, or `"flux"`.
    pub fn set_regime(&self, regime: &str) {
        if let Some(regime) = PerpetualRegime::parse(regime) {
            self.pending.regime.set(Some(regime));
        }
    }

    /// Nudge the renoise depth `t_r` by `delta` notches (the drift clamps it).
    pub fn nudge_depth(&self, delta: i32) {
        self.pending
            .depth_delta
            .set(self.pending.depth_delta.get().saturating_add(delta));
    }

    /// `[r]`: a fresh seed — new noise for inference, the next image for errance.
    pub fn reseed(&self) {
        self.pending.reseed.set(true);
    }
}

/// The fallible half of [`BatDiffusion::create`], returning a plain `String`
/// error so the maths above stays free of `JsValue`.
async fn build_engine(
    config_bytes: Vec<u8>,
    weights: Vec<u8>,
    seeds: Vec<u8>,
) -> Result<Engine, String> {
    let config = ModelConfig::from_json_bytes(&config_bytes)
        .map_err(|err| format!("could not parse the model config: {err}"))?;

    let output = compute_inferred_input(&config.layers, config.input_size);
    let (width, height, out_channels) = (output.0, output.1, output.2);

    let gpu = Arc::new(GpuContext::new_headless_with(GpuLimitsProfile::Web).await);
    log::info!(
        "[batlab] device opened under the {} limits profile",
        gpu.limits_profile().label()
    );

    let mut model: Model = Model::new(Arc::clone(&gpu)).await;
    // Build the model INSIDE a validation error scope. Pipeline creation is
    // where the browser's WGSL compiler runs; a shader it rejects is captured
    // here and returned, so an unbuildable model becomes a message on the page
    // rather than a canvas that animates a dead forward pass in silence. This is
    // the direct guard for the class of bug that produced saturated noise — see
    // `docs/reports/WEB_PORT.md`.
    let build = gpu
        .guarded(async {
            for draft in &config.layers {
                model
                    .add_draft(draft)
                    .map_err(|err| format!("could not build a layer: {err}"))?;
            }
            model
                .build_model()
                .map_err(|err| format!("could not build the model: {err}"))?;
            Ok::<(), String>(())
        })
        .await;
    match build {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(err),
        Err(scope) => {
            return Err(format!(
                "the browser rejected a GPU pipeline while building the model — a shader it \
                 would not compile, or a limit it would not grant: {scope}"
            ));
        }
    }

    // Weight integrity, logged so a truncated or altered download is caught by
    // eye rather than by a wrong image. The checkpoint parser is already strict
    // (it refuses a length mismatch or trailing bytes), so a clean load means the
    // scalar count matches the model; these numbers let a visitor confirm the
    // bytes that reached wasm are the bytes on disk.
    let expected_params: u64 = config.layers.iter().map(|l| l.parameter_count()).sum();
    log::info!(
        "[batlab] weights: {} bytes received, fnv1a={:016x}; model expects {} trainable scalars",
        weights.len(),
        fnv1a_64(&weights),
        expected_params,
    );
    // The averaged weights when the checkpoint carries them, exactly as
    // production sampling does; falls back to the raw iterate otherwise.
    let loaded = model
        .load_checkpoint_bytes_with(&weights, CheckpointWeights::Ema)
        .map_err(|err| format!("could not load the weights: {err}"))?;
    log::info!(
        "[batlab] checkpoint loaded: carries_ema={}, using {} weights",
        loaded.carries_ema,
        if loaded.used_ema { "averaged" } else { "raw" },
    );

    let schedule = LinearNoiseSchedule::new_linear(STEPS, BETA_START, BETA_END);
    let seeds = SeedImages::parse(&seeds)?;

    let input_channels = config.input_size.2 as usize;
    let signal_channels = out_channels as usize;
    let output_len = (width * height * out_channels) as usize;

    // The portfolio opens on the piece: the endless drift from a real image.
    let regime = PerpetualRegime::Wander;
    let depth = 24;
    // Seeded from the browser, not from a constant. A fixed counter made the
    // page *deterministic across loads*: every visitor met the same elephant in
    // the same order, and a reload could never surprise its author. The engine
    // stays reproducible — a run is still a pure function of its seed — it is
    // only the opening seed that now comes from outside.
    let seed_counter = {
        let hi = (js_sys::Math::random() * (u32::MAX as f64)) as u64;
        let lo = (js_sys::Math::random() * (u32::MAX as f64)) as u64;
        (hi << 32) ^ lo ^ 0x0bad_cafe_face_5eed
    };

    let mut engine = Engine {
        model,
        gpu: Arc::clone(&gpu),
        last_diag: String::new(),
        last_error: None,
        diag_frames: 0,
        schedule,
        input_channels,
        signal_channels,
        width,
        height,
        output_len,
        seeds,
        seed_counter,
        mode: Mode::Errance,
        inf_latent: Vec::new(),
        inf_step: 0,
        inf_path_seed: 0,
        inf_resolved: None,
        inf_noise_print: 0,
        drift: PerpetualDrift::from_image(STEPS, regime, depth, 1),
        walk: DriftWalk::new(vec![0.0; output_len], DENOISE_MAGNITUDE),
        depth,
        regime,
        image_index: 0,
    };
    // Prime both runs so a mode switch never shows a blank frame.
    engine.start_new_errance();
    engine.start_new_inference();
    engine.mode = Mode::Errance;
    Ok(engine)
}

/// A 64-bit FNV-1a hash of the received weight bytes, logged so a visitor (or a
/// bug report) can confirm the download that reached wasm is the file on disk.
/// Cheap, non-cryptographic — its only job is to change when the bytes change.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Installs the panic hook and a console logger.
///
/// The logger is not decoration: wgpu reports a rejected shader, a tripped
/// limit, an uncaptured device error through the `log` facade, and with no sink
/// installed every one of those was written to nowhere on wasm — the page could
/// animate a silently-invalid pipeline with a spotless console. That was the
/// inference bug. With the sink, the browser console carries what the engine was
/// already trying to say.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

/// FNV-1a over the bits of a tensor — a cheap, stable fingerprint.
///
/// Not a hash for security: a number a human can read off a page and compare
/// between two runs. It exists because « the image looks the same » is not a
/// measurement, and two identical-looking 32x32 pictures cannot be told from
/// two genuinely identical ones by eye.
fn fingerprint(values: &[f32]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
    }
    hash
}
