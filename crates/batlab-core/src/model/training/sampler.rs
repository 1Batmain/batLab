//! The diffusion inference path: input composition, the reverse recursion and
//! the sampler. This is the chain CLAUDE.md requires to stay decoupled — it runs
//! in the visitor's browser on their own GPU (WebGPU) — so it drives the model
//! through its PUBLIC API (`predict`/`predict_async`) and the shared schedule,
//! never GPU internals. The statistics it captures along the way live next door
//! in `metrics.rs`.

use crate::model::Model;
use crate::model::training::{LinearNoiseSchedule, PosteriorVariance};

use super::metrics::{DenoiseStepStat, Stats};

/// Packs a `[signal | timestep-embedding]` model input, byte-for-byte the layout
/// the GPU `diffusion_prepare` shader produces during training — the single source
/// of truth for input composition, so the conditioning the network is TRAINED on
/// is exactly the one it is SAMPLED with.
pub fn compose_diffusion_input(
    signal: &[f32],
    input_channels: usize,
    signal_channels: usize,
    timestep_features: &[f32],
) -> Vec<f32> {
    if signal.is_empty() || input_channels == 0 || signal_channels == 0 {
        return signal.to_vec();
    }
    let pixel_count = signal.len() / signal_channels;
    let mut packed = vec![0.0f32; pixel_count * input_channels];
    for (pixel_idx, pixel) in signal.chunks(signal_channels).enumerate() {
        let dst = &mut packed[pixel_idx * input_channels..(pixel_idx + 1) * input_channels];
        dst[..signal_channels].copy_from_slice(pixel);
        for (value, feature) in dst
            .iter_mut()
            .skip(signal_channels)
            .zip(timestep_features.iter())
        {
            *value = *feature;
        }
    }
    packed
}

/// One denoising step's tensors, handed to an observer as they are produced.
///
/// Borrowed, never owned: the observer runs inside the sampling loop and is
/// expected to forward the data somewhere (a GPU buffer, a file) rather than
/// accumulate it — a 256-step run over 4 paths would otherwise retain a
/// thousand tensors for nothing.
pub struct DenoiseFrame<'a> {
    /// Index of the denoising path currently being walked.
    pub path_idx: usize,
    /// Total number of paths this run will walk.
    pub path_count: usize,
    /// How many steps of this path are done (0-based, counts *up*).
    pub step_index: usize,
    /// Position on the noise schedule (counts *down* from `total_steps - 1`).
    pub diffusion_step: usize,
    /// Length of the noise schedule.
    pub total_steps: usize,
    /// The latent *after* this reverse step: x_{t-1}.
    pub latent: &'a [f32],
    /// The clipped x0 estimate this step's posterior mean was built from —
    /// "what the model believes the clean image looks like" from x_t.
    pub x0_hat: &'a [f32],
}

/// Odd increment used to fold the timestep into a path's seed.
const STEP_SEED_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// The seed one reverse step draws its noise from. The timestep is mixed in by
/// MULTIPLY-ADD, not XOR: `path_seed ^ step` collided with the field's own XOR of
/// the pixel index and painted horizontal bands (`ANISOTROPY_HUNT.md`). One
/// function, so no caller can derive its step seed a different, broken way.
pub fn reverse_step_seed(path_seed: u64, diffusion_step: usize) -> u64 {
    path_seed.wrapping_add((diffusion_step as u64 + 1).wrapping_mul(STEP_SEED_GAMMA))
}

/// The fold from a run seed to its opening latent's seed — one value, one place.
/// The native sampler, the drift's noise fallback and the web port all pass their
/// run seed through [`base_noise_seed`], so a browser run opens on the field native
/// path 0 does. A second copy (a private `const` in the web crate) is exactly the
/// silent divergence `WEB_PORT.md` warns of; public so callers import, not restate.
pub const BASE_NOISE_FOLD: u64 = 0xa5a5_5a5a_0123_4567;

/// The opening latent's noise seed for a run seed — see [`BASE_NOISE_FOLD`].
pub fn base_noise_seed(seed: u64) -> u64 {
    seed ^ BASE_NOISE_FOLD
}

/// The seed of denoising path `path_idx`. Path 0 is the run seed itself, so a
/// single-path run (inference, web) and path 0 of a multi-path sample draw the same
/// chain. One function, so no caller spreads the seed a different way.
pub fn path_seed(seed: u64, path_idx: usize) -> u64 {
    seed ^ (path_idx as u64).wrapping_mul(STEP_SEED_GAMMA)
}

/// What one reverse step produced.
pub struct ReverseStep {
    /// x_{t-1}, the latent after the step.
    pub latent: Vec<f32>,
    /// ε̂, the noise the model predicted from x_t.
    pub predicted_noise: Vec<f32>,
    /// The clipped x0 estimate the posterior mean was built from — present only
    /// when asked for, since nothing in the recursion needs it.
    pub x0_hat: Option<Vec<f32>>,
}

/// Asks the model what noise it sees in `latent` at `diffusion_step` — a reverse
/// step's one model call. Split out so the recursion is written against a
/// PREDICTION, not a `Model`, letting the perpetual walk run against an oracle in
/// tests without a GPU or a second copy of the maths.
pub fn predict_epsilon<State>(
    model: &mut Model<State>,
    schedule: &LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
    latent: &[f32],
    diffusion_step: usize,
) -> Vec<f32> {
    let timestep_channels = input_channels.saturating_sub(signal_channels);
    let features = schedule.timestep_embedding(diffusion_step, timestep_channels);
    let model_input = compose_diffusion_input(latent, input_channels, signal_channels, &features);
    model.predict(&model_input)
}

/// The reverse recursion proper, from an already-predicted ε̂ — the ONLY place it
/// is written. Every walker goes through it ([`sample_diffusion`] and the
/// perpetual drift), so the step-seed derivation and posterior draw cannot diverge
/// between a finite sample and an endless one.
#[allow(clippy::too_many_arguments)]
pub fn reverse_step_from_epsilon(
    schedule: &LinearNoiseSchedule,
    latent: &[f32],
    predicted_noise: Vec<f32>,
    diffusion_step: usize,
    path_seed: u64,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    want_x0_hat: bool,
) -> ReverseStep {
    // Derived from x_t, so it must be read before the reverse step produces
    // x_{t-1}. Skipped entirely when nobody is watching.
    let x0_hat =
        want_x0_hat.then(|| schedule.x0_estimate(latent, &predicted_noise, diffusion_step));
    let next = schedule.denoise_step_with_magnitude(
        latent,
        &predicted_noise,
        diffusion_step,
        reverse_step_seed(path_seed, diffusion_step),
        denoise_magnitude,
        variance,
    );
    ReverseStep {
        latent: next,
        predicted_noise,
        x0_hat,
    }
}

/// The async sibling of [`reverse_step`] — same composition, ε̂ and posterior draw,
/// but it AWAITS the readback ([`Model::predict_async`]). So a browser inference
/// descent does not re-derive the recursion (which could compose `[x_t|timestep]`
/// or seed the posterior differently and drift undetected); held to the sync path
/// by `the_async_reverse_step_matches_the_sync_one`.
#[allow(clippy::too_many_arguments)]
pub async fn reverse_step_async<State>(
    model: &mut Model<State>,
    schedule: &LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
    latent: &[f32],
    diffusion_step: usize,
    path_seed: u64,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    want_x0_hat: bool,
) -> ReverseStep {
    let timestep_channels = input_channels.saturating_sub(signal_channels);
    let features = schedule.timestep_embedding(diffusion_step, timestep_channels);
    let model_input = compose_diffusion_input(latent, input_channels, signal_channels, &features);
    let predicted_noise = model.predict_async(&model_input).await;
    reverse_step_from_epsilon(
        schedule,
        latent,
        predicted_noise,
        diffusion_step,
        path_seed,
        denoise_magnitude,
        variance,
        want_x0_hat,
    )
}

/// One step down the reverse chain: compose `[x_t | timestep]`, predict ε̂, and
/// sample the posterior — [`predict_epsilon`] then
/// [`reverse_step_from_epsilon`], which is the whole of it.
#[allow(clippy::too_many_arguments)]
pub fn reverse_step<State>(
    model: &mut Model<State>,
    schedule: &LinearNoiseSchedule,
    input_channels: usize,
    signal_channels: usize,
    latent: &[f32],
    diffusion_step: usize,
    path_seed: u64,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    want_x0_hat: bool,
) -> ReverseStep {
    let predicted_noise = predict_epsilon(
        model,
        schedule,
        input_channels,
        signal_channels,
        latent,
        diffusion_step,
    );
    reverse_step_from_epsilon(
        schedule,
        latent,
        predicted_noise,
        diffusion_step,
        path_seed,
        denoise_magnitude,
        variance,
        want_x0_hat,
    )
}

/// The SINGLE diffusion sampler, used by inference and diagnostics alike, so the
/// math the metrics observe is production's. `trajectory: Some` records the first
/// path's per-step stats (where along the chain the latent diverges). `observer:
/// Some` is called per step with the latent and x0 estimate — what the live
/// visualiser hangs off, so it watches the real chain, not a re-implementation
/// (x0 is computed only when observed).
#[allow(clippy::too_many_arguments)]
pub fn sample_diffusion<State, F>(
    model: &mut Model<State>,
    input_channels: usize,
    signal_channels: usize,
    output_len: usize,
    schedule: &LinearNoiseSchedule,
    seed: u64,
    denoising_paths: usize,
    denoise_magnitude: f32,
    variance: PosteriorVariance,
    mut trajectory: Option<&mut Vec<DenoiseStepStat>>,
    mut observer: Option<&mut dyn FnMut(&DenoiseFrame)>,
    mut progress: F,
) -> Vec<f32>
where
    F: FnMut(usize, usize),
{
    let path_count = denoising_paths.max(1);
    let steps = schedule.len().max(1);
    let total_work = path_count.saturating_mul(steps);
    let mut accumulated = vec![0.0f32; output_len];
    let base_latent = schedule.sample_noise(output_len, base_noise_seed(seed));

    for path_idx in 0..path_count {
        let path_seed = path_seed(seed, path_idx);
        let mut latent = base_latent.clone();

        for (step_idx, diffusion_step) in (0..schedule.len()).rev().enumerate() {
            let latent_in_stats = Stats::of(&latent);
            let stepped = reverse_step(
                model,
                schedule,
                input_channels,
                signal_channels,
                &latent,
                diffusion_step,
                path_seed,
                denoise_magnitude,
                variance,
                observer.is_some(),
            );
            latent = stepped.latent;
            if path_idx == 0 {
                if let Some(traj) = trajectory.as_deref_mut() {
                    traj.push(DenoiseStepStat {
                        step_index: step_idx,
                        diffusion_step,
                        latent_in: latent_in_stats,
                        eps_hat: Stats::of(&stepped.predicted_noise),
                        latent_out: Stats::of(&latent),
                    });
                }
            }
            if let (Some(observe), Some(x0_hat)) =
                (observer.as_deref_mut(), stepped.x0_hat.as_deref())
            {
                observe(&DenoiseFrame {
                    path_idx,
                    path_count,
                    step_index: step_idx,
                    diffusion_step,
                    total_steps: steps,
                    latent: &latent,
                    x0_hat,
                });
            }
            progress(path_idx * steps + step_idx + 1, total_work);
        }

        for (acc, value) in accumulated.iter_mut().zip(latent.iter()) {
            *acc += *value;
        }
    }

    for value in &mut accumulated {
        *value /= path_count as f32;
    }
    accumulated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_packs_signal_then_timestep_channels() {
        // 2 pixels, 1 signal channel, 3 input channels => 2 timestep channels.
        let signal = vec![0.5, -0.5];
        let features = vec![0.1, 0.2];
        let packed = compose_diffusion_input(&signal, 3, 1, &features);
        assert_eq!(packed, vec![0.5, 0.1, 0.2, -0.5, 0.1, 0.2]);
    }

    #[test]
    fn compose_without_timestep_channels_is_identity_layout() {
        let signal = vec![0.5, -0.5, 0.25, -0.25];
        let packed = compose_diffusion_input(&signal, 1, 1, &[]);
        assert_eq!(packed, signal);
    }

    /// Now that every walker of the chain derives its per-step seed here, this
    /// is the one place the anisotropy defect could come back. Under the old
    /// `path_seed ^ diffusion_step` the seeds of two steps differed by a few low
    /// bits — exactly the relation the pixel-index XOR turned into a reindexing
    /// of one single noise field (`ANISOTROPY_HUNT.md`).
    #[test]
    fn reverse_step_seed_folds_the_timestep_in_without_xor() {
        let path_seed = 0xdead_beef_u64;
        for step in 0..256usize {
            assert_ne!(
                reverse_step_seed(path_seed, step),
                path_seed ^ step as u64,
                "step {step} seed fell back to the XOR derivation"
            );
        }
        let distinct: std::collections::HashSet<u64> = (0..256)
            .map(|step| reverse_step_seed(path_seed, step))
            .collect();
        assert_eq!(distinct.len(), 256, "two timesteps share a seed");
    }

    /// The canonical folds must still compute the exact values the four scattered
    /// literals used to — pinning them so this centralisation is provably a
    /// no-op, and so a fat-fingered edit to `BASE_NOISE_FOLD` cannot pass.
    #[test]
    fn the_canonical_seed_folds_match_the_old_literals() {
        // The two folds exactly as they read at metrics.rs:560, perpetual.rs:561,
        // model.rs:2129 and web/lib.rs before centralisation — rebuilt from
        // halves so this test is not itself flagged as a second copy.
        let old_base_fold: u64 = (0xa5a5_5a5a_u64 << 32) | 0x0123_4567;
        let old_gamma: u64 = (0x9e37_79b9_u64 << 32) | 0x7f4a_7c15;
        for seed in [0u64, 1, 42, 7, 0xdead_beef, u64::MAX] {
            assert_eq!(base_noise_seed(seed), seed ^ old_base_fold);
            for path_idx in 0..8usize {
                assert_eq!(
                    path_seed(seed, path_idx),
                    seed ^ (path_idx as u64).wrapping_mul(old_gamma),
                );
            }
        }
        // Path 0 is the run seed itself — the identity the web's single-path
        // inference and native path 0 both lean on.
        assert_eq!(path_seed(12345, 0), 12345);
    }

    /// The base-noise fold has exactly one home. Any *other* line in any crate
    /// that spells the literal out is a second source of truth — the very
    /// divergence a web port re-deriving its opening noise would introduce, and
    /// which no run would ever flag. Checked mechanically over every crate's
    /// sources, not just this one, because the copy that mattered lived in
    /// `batlab-web`.
    #[test]
    fn the_base_noise_fold_is_written_in_exactly_one_place() {
        // Assembled from halves so this detector is not itself an offender.
        let literal = concat!("0xa5a5_5a5a", "_0123_4567");
        // crates/batlab-core/src → up to the workspace, then every crate's src.
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("crates/<name> sits two levels below the workspace root")
            .to_path_buf();

        fn walk(dir: &std::path::Path, literal: &str, offenders: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries {
                let path = entry.expect("bad directory entry").path();
                if path.is_dir() {
                    walk(&path, literal, offenders);
                    continue;
                }
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !name.ends_with(".rs") {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("failed to read a source file");
                for (number, line) in source.lines().enumerate() {
                    if !line.contains(literal) {
                        continue;
                    }
                    let code = line.trim_start();
                    if code.starts_with("//") {
                        continue;
                    }
                    // The one allowed line: the canonical `pub const` definition.
                    if code.contains("const BASE_NOISE_FOLD") {
                        continue;
                    }
                    offenders.push(format!("{}:{}: {}", path.display(), number + 1, code));
                }
            }
        }

        let mut offenders = Vec::new();
        for crate_dir in std::fs::read_dir(workspace.join("crates"))
            .expect("failed to read crates/")
            .flatten()
        {
            walk(&crate_dir.path().join("src"), literal, &mut offenders);
        }
        assert!(
            offenders.is_empty(),
            "the base-noise fold must come from batlab_core::BASE_NOISE_FOLD, not a restated \
             literal — a second copy is how a run opens on a different noise field than the \
             native sampler, silently:\n{}",
            offenders.join("\n")
        );
    }
}
