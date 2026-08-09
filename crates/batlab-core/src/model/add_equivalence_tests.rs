//! GPU validation of the residual `Add` layer, to the depot's kernel standard:
//! an independent oracle for the forward, and central finite differences for the
//! backward — the second one built so that it fails if the gradient does NOT
//! reach both inputs.
//!
//! Add is deliberately the trivial kernel (`a + b` forward, identity backward),
//! so the risk is not the arithmetic but the *plumbing*: does the forward read
//! the skip at all, and does the backward route the gradient back to the skip's
//! producer as well as down the main path. Both tests are constructed to catch
//! exactly those failures.

use crate::gpu_context::GpuContext;
use crate::model::layer_types::{ActivationMethod, ActivationType, ConvolutionType, LayerTypes};
use crate::model::types::{Dim3, PaddingMode};
use crate::model::Model;
use std::sync::Arc;

/// Forward oracle, GPU against GPU — no maths re-implemented on the host.
///
/// Two models on the same input `x`:
///   - `main`:  x → SiLU                          ⇒ SiLU(x)
///   - `res` :  x → Linear[save skip] → SiLU → Add(skip) ⇒ SiLU(x) + x
///
/// so `res` must equal `main + x` element for element. If the forward ignored
/// the skip it would return `SiLU(x)` and miss the `+ x`; if it doubled the main
/// path it would return `2·SiLU(x)`. Either way the check below fires.
#[test]
fn the_forward_is_the_elementwise_sum_of_both_inputs() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dims = Dim3::new((2, 2, 3)); // 12 values, mixed signs below

        // A spread of positive and negative inputs so SiLU(x) ≠ x and the sum is
        // a real check rather than a coincidence.
        let x: Vec<f32> = (0..12).map(|i| 0.35 * i as f32 - 2.0).collect();

        let mut main = Model::new(gpu.clone()).await;
        main.add_layer(LayerTypes::Activation(ActivationType::new(
            ActivationMethod::Silu,
            dims,
        )))
        .unwrap();
        main.build_model().unwrap();
        let silu = main.infer_batch(x.clone()).await;

        let mut res = Model::new(gpu.clone()).await;
        res.add_layer(LayerTypes::Activation(ActivationType::new(
            ActivationMethod::Linear,
            dims,
        )))
        .unwrap();
        res.mark_output("skip").unwrap();
        res.add_layer(LayerTypes::Activation(ActivationType::new(
            ActivationMethod::Silu,
            Dim3::default(),
        )))
        .unwrap();
        res.add_residual("skip").unwrap();
        res.build_model().unwrap();
        let out = res.infer_batch(x.clone()).await;

        assert_eq!(out.len(), x.len());
        for i in 0..x.len() {
            let oracle = silu[i] as f64 + x[i] as f64;
            assert!(
                (out[i] as f64 - oracle).abs() < 1e-5,
                "element {i}: Add gave {} but SiLU(x)+x is {oracle}",
                out[i]
            );
        }
    });
}

/// Backward via finite differences, arranged so a broken skip route is caught.
///
/// The convolution is BOTH the skip's producer and the start of the main path:
///
///   x → Conv[save skip] → SiLU → Add(skip) → loss
///
/// so the loss depends on the conv weights along two routes — directly through
/// the skip (`Add`'s `grad_skip`) and through SiLU (`Add`'s `grad_input`). The
/// analytic gradient the GPU backward produces is only correct if `Add` routes
/// the incoming gradient to *both*. Central differences on the loss measure the
/// true total; if `grad_skip` were dropped (or wrong) the two would diverge on
/// the skip's share, which is a large fraction here.
#[test]
fn the_backward_feeds_the_gradient_to_the_skip_producer() {
    use crate::model::debug::read_back_f32;
    use crate::model::layer_types::LayerType;
    use crate::model::{LossMethod, Training};

    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let (h, w) = (4usize, 4usize);
        let kernel_len = 3 * 3; // 1 kernel, 3x3, 1 input channel
        let out_len = 2 * 2; // Valid 3x3 on 4x4

        let mut model =
            Model::<Training>::new_training(gpu.clone(), 0.0, 1, LossMethod::MeanSquared).await;
        // Conv is the skip producer: its output is both saved and fed onward.
        model
            .add_layer(LayerTypes::Convolution(ConvolutionType::new(
                Dim3::new((h as u32, w as u32, 1)),
                1,
                Dim3::new((3, 3, 1)),
                1,
                PaddingMode::Valid,
            )))
            .unwrap();
        model.mark_output("skip").unwrap();
        model
            .add_layer(LayerTypes::Activation(ActivationType::new(
                ActivationMethod::Silu,
                Dim3::default(),
            )))
            .unwrap();
        model.add_residual("skip").unwrap();
        model.build().unwrap();

        let weights: Vec<f32> = (0..kernel_len).map(|i| 0.3 - 0.07 * i as f32).collect();
        let input: Vec<f32> = (0..(h * w)).map(|i| 0.1 * (i as f32) - 0.8).collect();
        let target: Vec<f32> = (0..out_len).map(|i| 0.05 * (i as f32)).collect();

        let write_weights = |model: &Model<Training>, ws: &[f32]| {
            let layer = model.layers.first().unwrap();
            gpu.queue
                .write_buffer(layer.buffers.forward[1].as_ref(), 0, bytemuck::cast_slice(ws));
            gpu.queue
                .write_buffer(layer.buffers.forward[2].as_ref(), 0, &0.0f32.to_le_bytes());
        };

        // Analytic gradient from the GPU backward pass (lr = 0).
        write_weights(&model, &weights);
        model.train_step(&input, &target);
        let layer = model.layers.first().unwrap();
        let bindings = layer.ty.get_optimizer_bindings().unwrap();
        let grad_buf =
            &layer.buffers.backward.as_ref().unwrap()[bindings.grad_weights_backward_index];
        let analytic = read_back_f32(gpu.as_ref(), grad_buf, (kernel_len * 4) as u64)
            .expect("failed to read grad_weights");

        // Numeric gradient via central differences on the loss.
        let eps = 1e-3f32;
        let mut loss_at = |ws: &[f32]| -> f32 {
            write_weights(&model, ws);
            model.train_step(&input, &target);
            model.read_last_loss()
        };
        let mut numeric = vec![0.0f32; kernel_len];
        for i in 0..kernel_len {
            let mut plus = weights.clone();
            plus[i] += eps;
            let mut minus = weights.clone();
            minus[i] -= eps;
            numeric[i] = (loss_at(&plus) - loss_at(&minus)) / (2.0 * eps);
        }

        let mut worst = 0.0f32;
        for i in 0..kernel_len {
            let denom = analytic[i].abs().max(numeric[i].abs()).max(1e-4);
            worst = worst.max((analytic[i] - numeric[i]).abs() / denom);
        }
        assert!(
            worst < 5e-2,
            "\nConv-through-residual weight gradients disagree with finite differences.\n\
             A dropped or wrong Add skip route shows up here.\n\
             analytic (GPU backward): {analytic:?}\n\
             numeric  (central diff): {numeric:?}\n\
             worst relative error: {worst:.4}\n"
        );
    });
}
