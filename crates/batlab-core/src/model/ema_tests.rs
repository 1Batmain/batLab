//! File purpose: correctness tests for the weight EMA — f64 oracle of the
//! recurrence and its warmup, the untouched-without-it guarantee, and the
//! checkpoint round-trip of both weight sets.
//!
//! The weights themselves are NOT under test here. Every test reads the weights
//! the optimiser actually produced and feeds those same values to the CPU
//! oracle, so a discrepancy can only come from the averaging.

use crate::gpu_context::GpuContext;
use crate::model::debug::read_back_f32;
use crate::model::layer_types::LayerType;
use crate::model::{
    CheckpointWeights, ConvolutionType, Dim3, EmaConfig, LossMethod, Model, OptimizerKind,
    PaddingMode, Training,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CPU oracle — the recurrence exactly as documented in `ema.rs`, in f64.
// ---------------------------------------------------------------------------

/// `ema ← d(t)·ema + (1 − d(t))·w`, with `d(t) = min(decay, (1+t)/(10+t))`.
///
/// Deliberately written from the *rule*, not from the shader: it derives the
/// ramp from the step counter itself rather than being handed the effective
/// decay, so an off-by-one in `publish_optimizer_specs` shows up here.
#[derive(Debug, Clone)]
struct EmaOracle {
    decay: f64,
    shadow: Vec<f64>,
    t: u64,
}

impl EmaOracle {
    /// Seeded with the weights, like the pass is.
    fn new(decay: f64, weights: &[f32]) -> Self {
        Self {
            decay,
            shadow: weights.iter().map(|w| *w as f64).collect(),
            t: 0,
        }
    }

    fn step(&mut self, weights: &[f32]) {
        self.t += 1;
        let ramp = (1.0 + self.t as f64) / (10.0 + self.t as f64);
        let d = ramp.min(self.decay);
        for (shadow, weight) in self.shadow.iter_mut().zip(weights) {
            *shadow = d * *shadow + (1.0 - d) * (*weight as f64);
        }
    }
}

// ---------------------------------------------------------------------------
// Harness — the same one-convolution model `adam_tests` uses.
// ---------------------------------------------------------------------------

const H: u32 = 4;
const W: u32 = 4;

async fn conv_model(
    gpu: Arc<GpuContext>,
    lr: f32,
    optimizer: OptimizerKind,
    ema: Option<EmaConfig>,
) -> Model<Training> {
    let mut model =
        Model::new_training_with_optimizer(gpu, lr, 1, LossMethod::MeanSquared, optimizer).await;
    model.set_ema(ema);
    model
        .add_layer(crate::model::LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new((H, W, 1)),
            1,
            Dim3::new((3, 3, 1)),
            1,
            PaddingMode::Same,
        )))
        .unwrap();
    model.build().unwrap();
    model
}

fn weights_of(model: &Model<Training>) -> Vec<f32> {
    let layer = &model.layers[0];
    let bindings = layer.ty.get_optimizer_bindings().expect("trainable layer");
    let buf = &layer.buffers.forward[bindings.weights_forward_index];
    read_back_f32(model.gpu.as_ref(), buf, buf.size()).expect("weights readback")
}

fn bias_of(model: &Model<Training>) -> Vec<f32> {
    let layer = &model.layers[0];
    let bindings = layer.ty.get_optimizer_bindings().expect("trainable layer");
    let buf = &layer.buffers.forward[bindings.bias_forward_index];
    read_back_f32(model.gpu.as_ref(), buf, buf.size()).expect("bias readback")
}

/// `(ema_weights, ema_bias)` — the shadow of the (single) trainable layer.
fn shadow_of(model: &Model<Training>) -> (Vec<f32>, Vec<f32>) {
    let state = model.layers[0].ema_state_buffers();
    assert_eq!(state.len(), 2, "the EMA keeps a shadow of weights and bias");
    let w = read_back_f32(model.gpu.as_ref(), &state[0], state[0].size()).expect("ema w readback");
    let b = read_back_f32(model.gpu.as_ref(), &state[1], state[1].size()).expect("ema b readback");
    (w, b)
}

fn input_and_target(step: usize) -> (Vec<f32>, Vec<f32>) {
    let n = (H * W) as usize;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.37 + step as f32 * 0.11).sin()) * 0.8)
        .collect();
    let target: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.21 - step as f32 * 0.17).cos()) * 0.5)
        .collect();
    (input, target)
}

// ---------------------------------------------------------------------------
// The recurrence
// ---------------------------------------------------------------------------

/// The shadow after N steps is the f64 recurrence applied to the weights the
/// optimiser actually produced, warmup included.
///
/// Twelve steps is chosen so the run straddles the ramp: with a nominal decay
/// of 0.9 the ramp is still what binds at t=12 (`13/22 = 0.591`), so an
/// implementation that ignored the warmup and used 0.9 throughout is off by
/// more than a factor of ten in the residual — far outside the f32 tolerance.
#[test]
fn the_shadow_follows_its_f64_reference_step_by_step() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let ema = EmaConfig::new(0.9).unwrap();
        let mut model = conv_model(gpu, 1e-2, OptimizerKind::Adam, Some(ema)).await;

        let mut oracle_w = EmaOracle::new(ema.decay as f64, &weights_of(&model));
        let mut oracle_b = EmaOracle::new(ema.decay as f64, &bias_of(&model));

        for step in 0..12 {
            let (input, target) = input_and_target(step);
            model.train_step(&input, &target);
            // The weights AFTER the update are what the EMA pass read: the two
            // run in the same encoder, optimiser first.
            oracle_w.step(&weights_of(&model));
            oracle_b.step(&bias_of(&model));

            let (shadow_w, shadow_b) = shadow_of(&model);
            for (i, (got, want)) in shadow_w.iter().zip(&oracle_w.shadow).enumerate() {
                assert!(
                    (*got as f64 - *want).abs() < 1e-6,
                    "step {step}, weight {i}: {got} vs {want}"
                );
            }
            for (i, (got, want)) in shadow_b.iter().zip(&oracle_b.shadow).enumerate() {
                assert!(
                    (*got as f64 - *want).abs() < 1e-6,
                    "step {step}, bias {i}: {got} vs {want}"
                );
            }
        }
    });
}

/// The shadow starts on the weights, never on zero.
///
/// A zero-seeded shadow is the classic EMA bug: it looks right (it converges,
/// eventually) while every checkpoint written in the first few thousand steps
/// generates noise. It is caught here at step 0, before any training at all.
#[test]
fn the_shadow_starts_on_the_weights_not_on_zero() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let model = conv_model(
            gpu,
            1e-2,
            OptimizerKind::Adam,
            Some(EmaConfig::new(0.999).unwrap()),
        )
        .await;
        let weights = weights_of(&model);
        let (shadow, _) = shadow_of(&model);
        assert!(
            weights.iter().any(|w| *w != 0.0),
            "the harness needs non-zero initial weights to prove anything"
        );
        assert_eq!(shadow, weights);
    });
}

/// The average lags the weights but chases them: after enough steps at a low
/// decay it is closer to the current weights than to the initial ones, and at a
/// high decay it is the other way round. This is the property the whole feature
/// exists for, stated without reference to the implementation.
#[test]
fn a_higher_decay_holds_the_shadow_further_back() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let distance = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        };

        let mut lagging = f64::NAN;
        let mut chasing = f64::NAN;
        for decay in [0.99f32, 0.2] {
            let mut model = conv_model(
                gpu.clone(),
                5e-2,
                OptimizerKind::Adam,
                Some(EmaConfig::new(decay).unwrap()),
            )
            .await;
            let initial = weights_of(&model);
            for step in 0..40 {
                let (input, target) = input_and_target(step);
                model.train_step(&input, &target);
            }
            let final_weights = weights_of(&model);
            let (shadow, _) = shadow_of(&model);
            assert!(
                distance(&initial, &final_weights) > 1e-4,
                "the harness needs the weights to actually move"
            );
            let to_start = distance(&shadow, &initial);
            let to_now = distance(&shadow, &final_weights);
            if decay > 0.5 {
                lagging = to_start / to_now;
            } else {
                chasing = to_start / to_now;
            }
        }
        assert!(
            lagging < chasing,
            "a decay of 0.99 must sit further back along the trajectory than \
             one of 0.2 (ratios to-start/to-now: {lagging} vs {chasing})"
        );
    });
}

// ---------------------------------------------------------------------------
// The guarantee for runs that ask for nothing
// ---------------------------------------------------------------------------

/// A run without `--ema` is the run that existed before this file: same
/// weights, bit for bit, and a V2 checkpoint whose bytes are identical.
///
/// The EMA pass reads the weights and writes only its own buffers, so it
/// *cannot* perturb them — but "cannot" is what everyone says about the pass
/// they just added, and this repository has a scar from a graph edit that
/// silently changed the numbers.
#[test]
fn a_run_without_an_ema_is_bit_identical_and_writes_a_v2_checkpoint() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut plain = conv_model(gpu.clone(), 1e-2, OptimizerKind::Adam, None).await;
        let mut averaged = conv_model(
            gpu.clone(),
            1e-2,
            OptimizerKind::Adam,
            Some(EmaConfig::new(0.9).unwrap()),
        )
        .await;
        assert_eq!(
            weights_of(&plain),
            weights_of(&averaged),
            "the two models must start from the same draw"
        );

        for step in 0..8 {
            let (input, target) = input_and_target(step);
            plain.train_step(&input, &target);
            averaged.train_step(&input, &target);
        }
        assert_eq!(
            weights_of(&plain),
            weights_of(&averaged),
            "keeping an average must not move the weights by a single bit"
        );
        assert_eq!(plain.bias_bits(), averaged.bias_bits());

        let plain_bytes = plain.checkpoint_bytes().unwrap();
        let averaged_bytes = averaged.checkpoint_bytes().unwrap();
        assert_eq!(&plain_bytes[..7], b"BBCKPT2");
        assert_eq!(&averaged_bytes[..7], b"BBCKPT3");
        // The V3 file is the V2 file plus a trailer, so the V2 one is a prefix
        // of it — that is the whole compatibility claim, checked rather than
        // asserted in prose.
        assert_eq!(
            plain_bytes[7..],
            averaged_bytes[7..plain_bytes.len()],
            "V3 must be V2 plus a trailer, not a different layout"
        );
        assert!(averaged_bytes.len() > plain_bytes.len());
    });
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

/// Both weight sets survive a save/load, and they stay distinguishable: the
/// weights come back as weights and the average as the average.
#[test]
fn a_checkpoint_round_trips_both_weight_sets() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let ema = EmaConfig::new(0.8).unwrap();
        let mut trained = conv_model(gpu.clone(), 5e-2, OptimizerKind::Adam, Some(ema)).await;
        for step in 0..10 {
            let (input, target) = input_and_target(step);
            trained.train_step(&input, &target);
        }
        let weights = weights_of(&trained);
        let (shadow, shadow_bias) = shadow_of(&trained);
        assert_ne!(
            weights, shadow,
            "the harness needs the two sets to have diverged"
        );
        let bytes = trained.checkpoint_bytes().unwrap();

        let mut restored = conv_model(gpu.clone(), 5e-2, OptimizerKind::Adam, Some(ema)).await;
        let report = restored
            .load_checkpoint_bytes_with(&bytes, CheckpointWeights::Raw)
            .unwrap();
        assert!(report.carries_ema);
        assert!(!report.used_ema);
        assert_eq!(report.ema_decay, Some(0.8));
        assert_eq!(weights_of(&restored), weights);
        assert_eq!(shadow_of(&restored), (shadow.clone(), shadow_bias.clone()));
        assert_eq!(
            restored.optimizer_step(),
            trained.optimizer_step(),
            "the step counter is part of the state, EMA or not"
        );

        // …and the same file loaded for sampling puts the average where the
        // forward pass will read it.
        let mut sampling = conv_model(gpu.clone(), 5e-2, OptimizerKind::Adam, None).await;
        let report = sampling
            .load_checkpoint_bytes_with(&bytes, CheckpointWeights::Ema)
            .unwrap();
        assert!(report.used_ema);
        assert_eq!(weights_of(&sampling), shadow);
        assert_eq!(bias_of(&sampling), shadow_bias);
    });
}

/// Asking for the average of a file that has none is not an error, and it says
/// so. Anything else would make `--raw-weights` a prerequisite rather than a
/// comparison.
#[test]
fn asking_for_an_average_a_checkpoint_lacks_falls_back_on_the_weights() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut plain = conv_model(gpu.clone(), 1e-2, OptimizerKind::Adam, None).await;
        for step in 0..4 {
            let (input, target) = input_and_target(step);
            plain.train_step(&input, &target);
        }
        let weights = weights_of(&plain);
        let bytes = plain.checkpoint_bytes().unwrap();

        let mut sampling = conv_model(gpu.clone(), 1e-2, OptimizerKind::Adam, None).await;
        let report = sampling
            .load_checkpoint_bytes_with(&bytes, CheckpointWeights::Ema)
            .unwrap();
        assert!(!report.carries_ema);
        assert!(!report.used_ema);
        assert_eq!(report.ema_decay, None);
        assert_eq!(weights_of(&sampling), weights);
    });
}

/// Resuming an averaging run from a checkpoint that has no average re-seeds the
/// shadow on the weights it just loaded.
///
/// Leaving it on the random draw the rebuild produced is the failure this
/// guards: the run would look healthy for hours and its EMA checkpoints would
/// be a blend of trained weights and noise.
#[test]
fn a_v2_checkpoint_reseeds_the_shadow_instead_of_leaving_it_random() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut plain = conv_model(gpu.clone(), 5e-2, OptimizerKind::Adam, None).await;
        for step in 0..10 {
            let (input, target) = input_and_target(step);
            plain.train_step(&input, &target);
        }
        let weights = weights_of(&plain);
        let bias = bias_of(&plain);
        let bytes = plain.checkpoint_bytes().unwrap();
        assert_eq!(&bytes[..7], b"BBCKPT2");

        let mut resumed = conv_model(
            gpu.clone(),
            5e-2,
            OptimizerKind::Adam,
            Some(EmaConfig::new(0.999).unwrap()),
        )
        .await;
        let fresh_draw = weights_of(&resumed);
        assert_ne!(fresh_draw, weights, "the harness needs a real difference");
        resumed.load_checkpoint_bytes(&bytes).unwrap();
        assert_eq!(shadow_of(&resumed), (weights, bias));
    });
}

/// A V1 file — no trailer at all — still loads, and an averaging run seeds its
/// shadow from it just the same.
#[test]
fn a_v1_checkpoint_still_loads_into_an_averaging_run() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut plain = conv_model(gpu.clone(), 5e-2, OptimizerKind::Sgd, None).await;
        for step in 0..4 {
            let (input, target) = input_and_target(step);
            plain.train_step(&input, &target);
        }
        let weights = weights_of(&plain);
        // SGD writes the `none` optimiser tag, so a V2 file minus its last four
        // bytes, relabelled, IS the V1 file this model would have written.
        let v2 = plain.checkpoint_bytes().unwrap();
        let mut v1 = b"BBCKPT1".to_vec();
        v1.extend_from_slice(&v2[7..v2.len() - 4]);

        let mut resumed = conv_model(
            gpu.clone(),
            5e-2,
            OptimizerKind::Sgd,
            Some(EmaConfig::new(0.99).unwrap()),
        )
        .await;
        resumed.load_checkpoint_bytes(&v1).unwrap();
        assert_eq!(weights_of(&resumed), weights);
        assert_eq!(shadow_of(&resumed).0, weights);
    });
}

/// A checkpoint of another geometry is refused rather than silently truncated —
/// what `--resume` leans on to reject a mismatched file.
#[test]
fn a_checkpoint_of_another_geometry_is_refused() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let small = conv_model(
            gpu.clone(),
            1e-2,
            OptimizerKind::Adam,
            Some(EmaConfig::new(0.9).unwrap()),
        )
        .await;
        let bytes = small.checkpoint_bytes().unwrap();

        let mut wide = Model::new_training_with_optimizer(
            gpu.clone(),
            1e-2,
            1,
            LossMethod::MeanSquared,
            OptimizerKind::Adam,
        )
        .await;
        wide.set_ema(Some(EmaConfig::new(0.9).unwrap()));
        wide.add_layer(crate::model::LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new((H, W, 1)),
            // Four output channels where the saved model had one.
            4,
            Dim3::new((3, 3, 1)),
            1,
            PaddingMode::Same,
        )))
        .unwrap();
        wide.build().unwrap();
        let err = wide.load_checkpoint_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("mismatch"),
            "expected a length mismatch, got: {err}"
        );
    });
}

/// A resize rebuilds the graph; the shadow has to survive it, like the moments
/// and the step counter do.
#[test]
fn resizing_the_batch_preserves_the_shadow() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut model = conv_model(
            gpu,
            5e-2,
            OptimizerKind::Adam,
            Some(EmaConfig::new(0.8).unwrap()),
        )
        .await;
        for step in 0..6 {
            let (input, target) = input_and_target(step);
            model.train_step(&input, &target);
        }
        let before = shadow_of(&model);
        let weights_before = weights_of(&model);
        model.resize_batch(4).unwrap();
        assert_eq!(shadow_of(&model), before);
        assert_eq!(weights_of(&model), weights_before);
    });
}

impl Model<Training> {
    /// Bias as raw bits, so "bit-identical" means bit-identical rather than
    /// "equal under whatever f32 comparison the test happened to write".
    #[cfg(test)]
    fn bias_bits(&self) -> Vec<u32> {
        bias_of(self).iter().map(|b| b.to_bits()).collect()
    }
}

/// A quantised checkpoint is a faithful 8-bit store: loading it back yields
/// **exactly** the values the quantiser produced from the weights in the buffer
/// — bit for bit, not close. Whether those values still make a good image is the
/// separate, measured question `--eval` answers; this pins that the file format
/// itself loses nothing beyond the quantisation it declares.
#[test]
fn a_quantised_checkpoint_reloads_the_dequantised_weights_exactly() {
    use crate::model::quant;
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut model = conv_model(
            gpu.clone(),
            1e-2,
            OptimizerKind::Adam,
            Some(EmaConfig::new(0.9).unwrap()),
        )
        .await;
        for step in 0..6 {
            let (input, target) = input_and_target(step);
            model.train_step(&input, &target);
        }

        let bytes = model.checkpoint_bytes_quantized().unwrap();
        assert_eq!(&bytes[..7], b"BBCKPTQ", "the magic marks the quantised format");

        // What the file promises to reconstruct: the forward weights, run
        // through quantise → dequantise on the host.
        let expected_w = {
            let w = weights_of(&model);
            let q = quant::quantize(&w);
            quant::dequantize(q.min, q.scale, &q.bytes)
        };
        let expected_b = {
            let b = bias_of(&model);
            let q = quant::quantize(&b);
            quant::dequantize(q.min, q.scale, &q.bytes)
        };

        // A quantised file has no optimiser or EMA trailer, so it loads into a
        // plain SGD model with no shadow.
        let gpu2 = Arc::new(GpuContext::new_headless().await);
        let mut fresh = conv_model(gpu2, 1e-2, OptimizerKind::Sgd, None).await;
        fresh.load_checkpoint_bytes(&bytes).unwrap();

        assert_eq!(
            weights_of(&fresh).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected_w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "reloaded weights are not the dequantised values, bit for bit"
        );
        assert_eq!(
            bias_of(&fresh).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "reloaded bias is not the dequantised value, bit for bit"
        );
        // The size win (one byte per weight against four) is measured on a real
        // model at export time, where the per-tensor header is negligible; here
        // the tensors are too small for a byte-count assertion to mean anything.
    });
}

