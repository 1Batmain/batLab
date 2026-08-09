//! GPU validation of the TimeBias layer, to the depot's kernel standard:
//! an f64 forward oracle, central finite differences for the weight and bias
//! gradients, and a check that each sample is conditioned on ITS OWN embedding.

use crate::gpu_context::GpuContext;
use crate::model::debug::read_back_f32;
use crate::model::layer_types::{ActivationMethod, ActivationType, LayerType, LayerTypes};
use crate::model::types::Dim3;
use crate::model::{LossMethod, Model, Training};
use std::sync::Arc;

// A tiny stack: identity(save "tin") → TimeBias("tin"). C = input_z, and the
// embedding is channels [offset, offset+N) of the saved input.
const H: u32 = 3;
const W: u32 = 3;
const IN_Z: u32 = 4; // 1 signal + 3 embedding
const OFFSET: u32 = 1;
const N: u32 = 3; // embedding channels
const C: u32 = IN_Z; // TimeBias sits on the identity output, so C = IN_Z

fn infer_input(len: usize, seed: f32) -> Vec<f32> {
    (0..len).map(|i| 0.2 * i as f32 - 1.0 + seed).collect()
}

/// The oracle: TimeBias reads the embedding from PIXEL 0 of the sample and adds
/// `bias[c] + Σ_n emb[n]·W[n·C+c]` to every position of channel c.
fn oracle(input: &[f32], weights: &[f32], bias: &[f32], batch: usize) -> Vec<f32> {
    let c = C as usize;
    let per_sample = (H * W * IN_Z) as usize;
    let mut out = vec![0.0f64; input.len()];
    for b in 0..batch {
        let base = b * per_sample;
        let emb: Vec<f64> = (0..N as usize)
            .map(|nn| input[base + OFFSET as usize + nn] as f64)
            .collect();
        for local in 0..per_sample {
            let channel = local % c;
            let mut acc = bias[channel] as f64;
            for (nn, e) in emb.iter().enumerate() {
                acc += e * weights[nn * c + channel] as f64;
            }
            out[base + local] = input[base + local] as f64 + acc;
        }
    }
    out.iter().map(|&v| v as f32).collect()
}

async fn build_infer(gpu: Arc<GpuContext>) -> Model<crate::model::Infer> {
    let mut model = Model::new(gpu).await;
    model
        .add_layer(LayerTypes::Activation(ActivationType::new(
            ActivationMethod::Linear,
            Dim3::new((H, W, IN_Z)),
        )))
        .unwrap();
    model.mark_output("tin").unwrap();
    model.add_time_bias("tin", OFFSET, N).unwrap();
    model.build_model().unwrap();
    model
}

fn write_params(gpu: &GpuContext, model: &Model<impl Sized>, weights: &[f32], bias: &[f32]) {
    // TimeBias is layer index 1; weights_forward_index=2, bias_forward_index=3.
    let layer = &model.layers[1];
    gpu.queue
        .write_buffer(layer.buffers.forward[2].as_ref(), 0, bytemuck::cast_slice(weights));
    gpu.queue
        .write_buffer(layer.buffers.forward[3].as_ref(), 0, bytemuck::cast_slice(bias));
}

#[test]
fn the_forward_matches_the_f64_projection() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut model = build_infer(gpu.clone()).await;

        let weights: Vec<f32> = (0..(N * C)).map(|i| 0.1 * i as f32 - 0.3).collect();
        let bias: Vec<f32> = (0..C).map(|i| 0.05 * i as f32).collect();
        write_params(&gpu, &model, &weights, &bias);

        let input = infer_input((H * W * IN_Z) as usize, 0.0);
        let out = model.infer_batch(input.clone()).await;
        let expected = oracle(&input, &weights, &bias, 1);
        for i in 0..out.len() {
            assert!(
                (out[i] as f64 - expected[i] as f64).abs() < 1e-5,
                "position {i}: {} vs oracle {}",
                out[i],
                expected[i]
            );
        }
    });
}

#[test]
fn each_sample_is_conditioned_on_its_own_embedding() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        // Batch 2: two samples with DIFFERENT embeddings must get different
        // biases. Built at batch 2 so the per-sample stride is exercised.
        let mut model = Model::<Training>::new_training(gpu.clone(), 0.0, 2, LossMethod::MeanSquared).await;
        model
            .add_layer(LayerTypes::Activation(ActivationType::new(
                ActivationMethod::Linear,
                Dim3::new((H, W, IN_Z)),
            )))
            .unwrap();
        model.mark_output("tin").unwrap();
        model.add_time_bias("tin", OFFSET, N).unwrap();
        model.build().unwrap();

        let weights: Vec<f32> = (0..(N * C)).map(|i| 0.1 * i as f32 - 0.3).collect();
        let bias: Vec<f32> = (0..C).map(|i| 0.05 * i as f32).collect();
        write_params(&gpu, &model, &weights, &bias);

        let per = (H * W * IN_Z) as usize;
        let mut input = infer_input(per, 0.0);
        input.extend(infer_input(per, 0.7)); // sample 2, shifted so its emb differs
        let target = vec![0.0f32; per * 2];
        model.train_step(&input, &target);

        // Read the TimeBias output (layer 1 forward output buffer, last binding).
        let layer = &model.layers[1];
        let out_buf = layer.buffers.forward.last().unwrap();
        let out = read_back_f32(gpu.as_ref(), out_buf, (per * 2 * 4) as u64).unwrap();
        let expected = oracle(&input, &weights, &bias, 2);
        for i in 0..out.len() {
            assert!(
                (out[i] as f64 - expected[i] as f64).abs() < 1e-5,
                "position {i}: {} vs oracle {} (per-sample embedding)",
                out[i],
                expected[i]
            );
        }
    });
}

#[test]
fn the_weight_and_bias_gradients_match_finite_differences() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut model =
            Model::<Training>::new_training(gpu.clone(), 0.0, 1, LossMethod::MeanSquared).await;
        model
            .add_layer(LayerTypes::Activation(ActivationType::new(
                ActivationMethod::Linear,
                Dim3::new((H, W, IN_Z)),
            )))
            .unwrap();
        model.mark_output("tin").unwrap();
        model.add_time_bias("tin", OFFSET, N).unwrap();
        model.build().unwrap();

        let weights: Vec<f32> = (0..(N * C)).map(|i| 0.13 * i as f32 - 0.4).collect();
        let bias: Vec<f32> = (0..C).map(|i| 0.07 * i as f32 - 0.1).collect();
        let input = infer_input((H * W * IN_Z) as usize, 0.0);
        let target: Vec<f32> = (0..(H * W * C) as usize).map(|i| 0.03 * i as f32).collect();

        // Analytic gradients (lr = 0 so nothing moves between read and reuse).
        write_params(&gpu, &model, &weights, &bias);
        model.train_step(&input, &target);
        let layer = &model.layers[1];
        let bindings = layer.ty.get_optimizer_bindings().unwrap();
        let bwd = layer.buffers.backward.as_ref().unwrap();
        let grad_w = read_back_f32(gpu.as_ref(), &bwd[bindings.grad_weights_backward_index], (N * C * 4) as u64).unwrap();
        let grad_b = read_back_f32(gpu.as_ref(), &bwd[bindings.grad_bias_backward_index], (C * 4) as u64).unwrap();

        let eps = 1e-3f32;
        let mut loss_at = |w: &[f32], b: &[f32]| -> f32 {
            write_params(&gpu, &model, w, b);
            model.train_step(&input, &target);
            model.read_last_loss()
        };

        // Weights.
        let mut worst = 0.0f32;
        for i in 0..(N * C) as usize {
            let mut p = weights.clone();
            p[i] += eps;
            let mut m = weights.clone();
            m[i] -= eps;
            let numeric = (loss_at(&p, &bias) - loss_at(&m, &bias)) / (2.0 * eps);
            let denom = grad_w[i].abs().max(numeric.abs()).max(1e-4);
            worst = worst.max((grad_w[i] - numeric).abs() / denom);
        }
        assert!(
            worst < 5e-2,
            "TimeBias weight gradients disagree with finite differences, worst {worst:.4}\n\
             analytic {grad_w:?}"
        );

        // Bias.
        let mut worst_b = 0.0f32;
        for i in 0..C as usize {
            let mut p = bias.clone();
            p[i] += eps;
            let mut m = bias.clone();
            m[i] -= eps;
            let numeric = (loss_at(&weights, &p) - loss_at(&weights, &m)) / (2.0 * eps);
            let denom = grad_b[i].abs().max(numeric.abs()).max(1e-4);
            worst_b = worst_b.max((grad_b[i] - numeric).abs() / denom);
        }
        assert!(
            worst_b < 5e-2,
            "TimeBias bias gradients disagree with finite differences, worst {worst_b:.4}\n\
             analytic {grad_b:?}"
        );
    });
}
