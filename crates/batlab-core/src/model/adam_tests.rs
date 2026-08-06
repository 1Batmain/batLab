//! File purpose: Correctness tests for the Adam optimiser pass — f64 oracle, moment
//! state evolution, batch-mean placement, and checkpoint persistence of m/v/t.
//!
//! The gradient computation is NOT under test here. Every test reads the
//! gradient buffer the backward pass actually produced and feeds that same
//! value to the CPU oracle, so a discrepancy can only come from the update
//! rule itself.

use crate::gpu_context::GpuContext;
use crate::model::debug::read_back_f32;
use crate::model::layer_types::LayerType;
use crate::model::{
    ConvolutionType, Dim3, LossMethod, Model, OptimizerKind, PaddingMode, Training,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CPU oracle — Adam exactly as published (Kingma & Ba 2015, algorithm 1), in f64.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct AdamOracle {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl AdamOracle {
    fn new(lr: f64, len: usize) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
        }
    }

    /// One update. `grad` is the ACCUMULATED gradient; `grad_scale` is the
    /// batch mean applied to it (1/batch_size).
    fn step(&mut self, weights: &mut [f64], grad: &[f64], grad_scale: f64) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..weights.len() {
            let g = grad[i] * grad_scale;
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            weights[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const H: u32 = 4;
const W: u32 = 4;

/// A one-convolution model: small enough to read every buffer back each step,
/// and its gradients are non-trivial and vary from step to step.
async fn conv_model(gpu: Arc<GpuContext>, lr: f32, optimizer: OptimizerKind) -> Model<Training> {
    let mut model =
        Model::new_training_with_optimizer(gpu, lr, 1, LossMethod::MeanSquared, optimizer).await;
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

/// Weights of the (single) trainable layer.
fn weights_of(model: &Model<Training>) -> Vec<f32> {
    let layer = &model.layers[0];
    let bindings = layer.ty.get_optimizer_bindings().expect("trainable layer");
    let buf = &layer.buffers.forward[bindings.weights_forward_index];
    read_back_f32(model.gpu.as_ref(), buf, buf.size()).expect("weights readback")
}

/// Accumulated weight gradients of the (single) trainable layer. Valid between
/// a training step and the gradient reset at the start of the next one.
fn grads_of(model: &Model<Training>) -> Vec<f32> {
    let layer = &model.layers[0];
    let bindings = layer.ty.get_optimizer_bindings().expect("trainable layer");
    let buf = &layer.buffers.backward.as_ref().expect("backward built")
        [bindings.grad_weights_backward_index];
    read_back_f32(model.gpu.as_ref(), buf, buf.size()).expect("grad readback")
}

/// `(m_weights, v_weights)` — the persistent Adam moments of the trainable layer.
fn moments_of(model: &Model<Training>) -> (Vec<f32>, Vec<f32>) {
    let state = model.layers[0].opt_state_buffers();
    assert_eq!(state.len(), 4, "adam keeps m/v for weights and bias");
    let m = read_back_f32(model.gpu.as_ref(), &state[0], state[0].size()).expect("m readback");
    let v = read_back_f32(model.gpu.as_ref(), &state[1], state[1].size()).expect("v readback");
    (m, v)
}

fn input_and_target(step: usize) -> (Vec<f32>, Vec<f32>) {
    let n = (H * W) as usize;
    // Varies with the step so successive gradients differ — a constant gradient
    // would make the bias correction the only thing distinguishing the steps.
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.37 + step as f32 * 0.11).sin()) * 0.8)
        .collect();
    let target: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.21 + step as f32 * 0.29).cos()) * 0.5)
        .collect();
    (input, target)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The central correctness claim: over several steps, the GPU weights follow
/// the f64 Adam reference driven by the gradients the GPU itself produced —
/// bias correction included. `t` advances 1, 2, 3… so `1 - β₁ᵗ` moves from 0.1
/// to 0.271, which the reference tracks; an implementation that dropped the
/// correction would diverge on the very first step by a factor of 10.
#[test]
fn adam_matches_f64_reference_including_bias_correction() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let lr = 0.01f32;
        let mut model = conv_model(gpu, lr, OptimizerKind::Adam).await;

        let mut expected: Vec<f64> = weights_of(&model).iter().map(|&w| w as f64).collect();
        let mut oracle = AdamOracle::new(lr as f64, expected.len());

        for step in 0..6usize {
            let (input, target) = input_and_target(step);
            model.train_step(&input, &target);

            let grads: Vec<f64> = grads_of(&model).iter().map(|&g| g as f64).collect();
            // Single sample, no accumulation: the batch mean is 1.
            oracle.step(&mut expected, &grads, 1.0);

            let actual = weights_of(&model);
            for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
                let tol = 1e-6 + want.abs() * 1e-4;
                assert!(
                    (got as f64 - want).abs() < tol,
                    "step {step}, weight {i}: gpu {got} vs f64 reference {want} \
                     (grad {}, t={})",
                    grads[i],
                    oracle.t
                );
            }

            // The moments the shader persisted must be the reference's too:
            // the weights could be right for one step while m/v drifted.
            let (m, v) = moments_of(&model);
            for i in 0..m.len() {
                assert!(
                    (m[i] as f64 - oracle.m[i]).abs() < 1e-7 + oracle.m[i].abs() * 1e-4,
                    "step {step}, m[{i}]: gpu {} vs reference {}",
                    m[i],
                    oracle.m[i]
                );
                assert!(
                    (v[i] as f64 - oracle.v[i]).abs() < 1e-9 + oracle.v[i].abs() * 1e-4,
                    "step {step}, v[{i}]: gpu {} vs reference {}",
                    v[i],
                    oracle.v[i]
                );
            }
        }
    });
}

/// m and v start at zero and stay bounded by the exponential-average
/// definition: |m| ≤ max|g| and 0 ≤ v ≤ max g². Also pins the sign convention —
/// m must track the gradient's sign, not its negation.
#[test]
fn adam_moments_start_at_zero_and_track_the_gradient() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut model = conv_model(gpu, 0.01, OptimizerKind::Adam).await;

        let (m0, v0) = moments_of(&model);
        assert!(
            m0.iter().all(|&x| x == 0.0) && v0.iter().all(|&x| x == 0.0),
            "adam state must start at m = v = 0"
        );

        let mut max_abs_g = 0.0f32;
        for step in 0..4usize {
            let (input, target) = input_and_target(step);
            model.train_step(&input, &target);
            let g = grads_of(&model);
            max_abs_g = max_abs_g.max(g.iter().fold(0.0f32, |a, &b| a.max(b.abs())));

            let (m, v) = moments_of(&model);
            for i in 0..m.len() {
                assert!(
                    m[i].abs() <= max_abs_g + 1e-6,
                    "step {step}: |m[{i}]| = {} exceeds max |g| = {max_abs_g}",
                    m[i].abs()
                );
                assert!(
                    v[i] >= 0.0 && v[i] <= max_abs_g * max_abs_g + 1e-6,
                    "step {step}: v[{i}] = {} outside [0, max g²]",
                    v[i]
                );
            }
            // First step, first moment: m₁ = 0.1·g₁, same sign as g₁.
            if step == 0 {
                for i in 0..m.len() {
                    assert!(
                        (m[i] as f64 - 0.1 * g[i] as f64).abs() < 1e-7,
                        "m₁[{i}] = {} should be 0.1·g₁ = {}",
                        m[i],
                        0.1 * g[i]
                    );
                }
            }
        }
    });
}

/// Adam's update is invariant to a rescaling of the gradient: at t = 1 it is
/// exactly ±lr per weight whatever |g| is.
///
/// This is why the batch mean is applied to the GRADIENT and not folded into
/// the learning rate the way the SGD pass does it — `lr / batch_size` would
/// change the step size, `g / batch_size` (correctly) does not.
#[test]
fn adam_first_step_is_lr_sized_regardless_of_gradient_scale() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let lr = 0.01f32;

        for scale in [1.0f32, 100.0] {
            let mut model = conv_model(gpu.clone(), lr, OptimizerKind::Adam).await;
            let before = weights_of(&model);
            let (input, target) = input_and_target(0);
            let scaled_target: Vec<f32> = target.iter().map(|t| t * scale).collect();
            let scaled_input: Vec<f32> = input.iter().map(|t| t * scale).collect();
            model.train_step(&scaled_input, &scaled_target);
            let after = weights_of(&model);
            let grads = grads_of(&model);

            for i in 0..before.len() {
                if grads[i].abs() < 1e-6 {
                    continue; // g ≈ 0: the eps term dominates, no claim to make
                }
                let delta = (after[i] - before[i]).abs();
                assert!(
                    (delta - lr).abs() < lr * 1e-2,
                    "scale {scale}, weight {i}: first adam step was {delta}, expected ≈ {lr} \
                     (g = {})",
                    grads[i]
                );
            }
        }
    });
}

/// Regression guard for the SGD path, which the optimiser refactor rewrote:
/// the update must still be exactly `w -= lr·g`, with no moment state and no
/// bias correction.
#[test]
fn sgd_update_is_unchanged_and_stateless() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let lr = 0.02f32;
        let mut model = conv_model(gpu, lr, OptimizerKind::Sgd).await;
        assert!(
            model.layers[0].opt_state_buffers().is_empty(),
            "sgd must not allocate optimiser state"
        );

        let mut expected: Vec<f64> = weights_of(&model).iter().map(|&w| w as f64).collect();
        for step in 0..3usize {
            let (input, target) = input_and_target(step);
            model.train_step(&input, &target);
            let grads = grads_of(&model);
            for (w, g) in expected.iter_mut().zip(grads.iter()) {
                *w -= lr as f64 * *g as f64;
            }
            let actual = weights_of(&model);
            for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got as f64 - want).abs() < 1e-6 + want.abs() * 1e-5,
                    "step {step}, weight {i}: sgd gave {got}, expected {want}"
                );
            }
        }
    });
}

/// Resuming an Adam run must resume its m, v and t — not just its weights.
///
/// The check is behavioural: a model restored from the checkpoint and stepped
/// once must land on exactly the same weights as the original stepped once
/// more. A resume that dropped m/v (or reset t) restarts the bias correction at
/// t = 1, where m̂ = g and the step is a full ±lr on every weight — visibly
/// different, which is what the second half of the test pins down.
#[test]
fn adam_state_survives_a_checkpoint_roundtrip() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let lr = 0.01f32;
        let path = std::env::temp_dir().join(format!(
            "bat-adam-ckpt-{}-{}.ckpt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut original = conv_model(gpu.clone(), lr, OptimizerKind::Adam).await;
        for step in 0..5usize {
            let (input, target) = input_and_target(step);
            original.train_step(&input, &target);
        }
        original.save_checkpoint(&path).unwrap();
        let (saved_m, saved_v) = moments_of(&original);

        // The original takes one more step; that is the ground truth.
        let (input, target) = input_and_target(5);
        original.train_step(&input, &target);
        let continued = weights_of(&original);

        let mut restored = conv_model(gpu.clone(), lr, OptimizerKind::Adam).await;
        restored.load_checkpoint(&path).unwrap();
        let (restored_m, restored_v) = moments_of(&restored);
        for i in 0..saved_m.len() {
            assert_eq!(saved_m[i], restored_m[i], "m[{i}] not restored");
            assert_eq!(saved_v[i], restored_v[i], "v[{i}] not restored");
        }

        restored.train_step(&input, &target);
        let resumed = weights_of(&restored);
        for (i, (&a, &b)) in continued.iter().zip(resumed.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "weight {i}: continued run {a} vs resumed run {b}"
            );
        }

        // Same weights, but cold optimiser state: the step must differ, which
        // is the cost the persistence exists to avoid.
        let mut cold = conv_model(gpu.clone(), lr, OptimizerKind::Adam).await;
        cold.load_checkpoint(&path).unwrap();
        // Wipe what the checkpoint restored, keeping only the weights.
        for buffer in cold.layers[0].opt_state_buffers() {
            let zeros = vec![0.0f32; (buffer.size() as usize) / 4];
            cold.gpu
                .queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&zeros));
        }
        cold.reset_optimizer_step();
        cold.train_step(&input, &target);
        let cold_weights = weights_of(&cold);
        let warm_delta: f32 = resumed
            .iter()
            .zip(continued.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let cold_delta: f32 = cold_weights
            .iter()
            .zip(continued.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            cold_delta > 1e-3 && cold_delta > warm_delta * 100.0,
            "a cold-started optimiser should visibly diverge from the continued run \
             (cold {cold_delta}, warm {warm_delta})"
        );

        let _ = std::fs::remove_file(path);
    });
}

/// An SGD checkpoint carries no optimiser state; loading it into an Adam run
/// must succeed and leave Adam cold (m = v = 0, t = 0) rather than fail.
#[test]
fn sgd_checkpoint_loads_into_an_adam_model() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let path = std::env::temp_dir().join(format!(
            "bat-sgd-into-adam-{}-{}.ckpt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut sgd = conv_model(gpu.clone(), 0.01, OptimizerKind::Sgd).await;
        let (input, target) = input_and_target(0);
        sgd.train_step(&input, &target);
        sgd.save_checkpoint(&path).unwrap();
        let sgd_weights = weights_of(&sgd);

        let mut adam = conv_model(gpu.clone(), 0.01, OptimizerKind::Adam).await;
        adam.load_checkpoint(&path).unwrap();
        assert_eq!(weights_of(&adam), sgd_weights, "weights must transfer");
        let (m, v) = moments_of(&adam);
        assert!(
            m.iter().all(|&x| x == 0.0) && v.iter().all(|&x| x == 0.0),
            "adam must start cold from a stateless checkpoint"
        );

        let _ = std::fs::remove_file(path);
    });
}

/// Checkpoints written before the optimiser trailer existed (magic `BBCKPT1`)
/// must still load. Built by rewriting a fresh checkpoint's magic and dropping
/// its 4-byte `none` trailer — byte-for-byte the old format.
#[test]
fn legacy_v1_checkpoints_still_load() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let path = std::env::temp_dir().join(format!(
            "bat-v1-ckpt-{}-{}.ckpt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut source = conv_model(gpu.clone(), 0.01, OptimizerKind::Sgd).await;
        let (input, target) = input_and_target(0);
        source.train_step(&input, &target);
        source.save_checkpoint(&path).unwrap();
        let expected = weights_of(&source);

        let mut bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..7], b"BBCKPT2");
        // Drop the `none` optimiser-state tag and go back to the V1 magic.
        assert_eq!(&bytes[bytes.len() - 4..], &0u32.to_le_bytes());
        bytes.truncate(bytes.len() - 4);
        bytes[..7].copy_from_slice(b"BBCKPT1");
        std::fs::write(&path, &bytes).unwrap();

        let mut restored = conv_model(gpu.clone(), 0.01, OptimizerKind::Sgd).await;
        restored.load_checkpoint(&path).unwrap();
        assert_eq!(weights_of(&restored), expected);

        let _ = std::fs::remove_file(path);
    });
}
