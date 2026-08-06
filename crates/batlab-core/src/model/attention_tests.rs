//! Correctness of the spatial self-attention layer.
//!
//! The reference is an f64 CPU implementation written from the FORMULA —
//! `y = x + W_o · softmax(qᵀk/√d) · v` — and not from the shader. It shares no
//! index arithmetic with the WGSL: it walks its own nested loops, over its own
//! `Vec`s, in double precision. When the two agree the agreement means
//! something; a reference transcribed from the kernel would only prove the
//! kernel equals itself.
//!
//! What each family of tests is for:
//!
//! - the f64 comparison pins the FORWARD (including the softmax normalisation
//!   and the 1/√d scale, both easy to get subtly wrong);
//! - the finite differences pin the BACKWARD against the forward, for all four
//!   projections AND the input — a backward can be self-consistent and still be
//!   the gradient of a different function;
//! - the batch tests pin the one failure this layer is uniquely exposed to:
//!   attention is the only layer in the graph where one position reads another,
//!   so a missing per-sample offset would let image 3 attend to image 4. That
//!   bug does not crash and does not even look wrong — the loss still falls.

use crate::gpu_context::GpuContext;
use crate::model::debug::read_back_f32;
use crate::model::layer_types::{AttentionType, LayerType, LayerTypes, LossMethod};
use crate::model::{Dim3, Infer, Model, Training};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// The independent f64 reference
// ---------------------------------------------------------------------------

/// Packed parameter layout, mirroring the single weight tensor the optimiser
/// sees: `[W_q | W_k | W_v | W_o]` then `[b_q | b_k | b_v | b_o]`.
struct Params {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl Params {
    /// Deterministic, decorrelated, and DIFFERENT for each projection: a shader
    /// that mixed up two projections' offsets would still produce finite,
    /// plausible numbers if they held the same values.
    fn pseudo_random(channels: usize, seed: u32) -> Self {
        let mut state = seed | 1;
        let mut draw = |scale: f64| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (((state >> 8) as f64 / (1u32 << 24) as f64) - 0.5) * scale
        };
        let cc = channels * channels;
        // W_o deliberately non-zero here: the identity case is its own test.
        let weights = (0..4 * cc)
            .map(|i| draw(0.4 + 0.2 * (i / cc) as f64) as f32)
            .collect();
        let bias = (0..4 * channels)
            .map(|i| draw(0.15 + 0.1 * (i / channels) as f64) as f32)
            .collect();
        Self { weights, bias }
    }

    fn zero_output_projection(mut self, channels: usize) -> Self {
        let cc = channels * channels;
        self.weights[3 * cc..].fill(0.0);
        self.bias[3 * channels..].fill(0.0);
        self
    }
}

/// One sample of self-attention, in f64, written from the formula.
///
/// `x` is `seq * channels`, laid out position-major (the repository's HWC
/// convention: index = position * channels + channel).
fn reference_attention(x: &[f32], seq: usize, channels: usize, params: &Params) -> Vec<f64> {
    let cc = channels * channels;
    let w = |which: usize, row: usize, col: usize| -> f64 {
        params.weights[which * cc + row * channels + col] as f64
    };
    let b = |which: usize, index: usize| -> f64 { params.bias[which * channels + index] as f64 };

    // q, k, v = W·x + b, one projection at a time.
    let project = |which: usize| -> Vec<Vec<f64>> {
        (0..seq)
            .map(|t| {
                (0..channels)
                    .map(|i| {
                        let mut acc = b(which, i);
                        for c in 0..channels {
                            acc += w(which, i, c) * x[t * channels + c] as f64;
                        }
                        acc
                    })
                    .collect()
            })
            .collect()
    };
    let q = project(0);
    let k = project(1);
    let v = project(2);

    let scale = 1.0 / (channels as f64).sqrt();

    let mut out = vec![0.0f64; seq * channels];
    for n in 0..seq {
        // scores, then a softmax computed the textbook way.
        let scores: Vec<f64> = (0..seq)
            .map(|m| {
                let dot: f64 = (0..channels).map(|i| q[n][i] * k[m][i]).sum();
                dot * scale
            })
            .collect();
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|e| e / total).collect();

        // context = probs · v, then the output projection and the residual.
        for c in 0..channels {
            let mut ctx_proj = b(3, c);
            for i in 0..channels {
                let ctx_i: f64 = (0..seq).map(|m| probs[m] * v[m][i]).sum();
                ctx_proj += w(3, c, i) * ctx_i;
            }
            out[n * channels + c] = x[n * channels + c] as f64 + ctx_proj;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn pseudo_random_input(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5) * scale
        })
        .collect()
}

async fn infer_model(gpu: Arc<GpuContext>, dim: (u32, u32, u32)) -> Model<Infer> {
    let mut model: Model<Infer> = Model::new(gpu).await;
    model
        .add_layer(LayerTypes::Attention(AttentionType::new(Dim3::new(dim))))
        .unwrap();
    model.build_model().unwrap();
    model
}

async fn training_model(gpu: Arc<GpuContext>, dim: (u32, u32, u32), batch: u32) -> Model<Training> {
    // lr = 0: the optimiser runs but moves nothing, so a gradient can be read
    // back after the step that produced it.
    let mut model = Model::<Training>::new_training(gpu, 0.0, batch, LossMethod::MeanSquared).await;
    model
        .add_layer(LayerTypes::Attention(AttentionType::new(Dim3::new(dim))))
        .unwrap();
    model.build().unwrap();
    model
}

fn write_params(model_gpu: &GpuContext, layer_buffers: (&wgpu::Buffer, &wgpu::Buffer), p: &Params) {
    model_gpu
        .queue
        .write_buffer(layer_buffers.0, 0, bytemuck::cast_slice(&p.weights));
    model_gpu
        .queue
        .write_buffer(layer_buffers.1, 0, bytemuck::cast_slice(&p.bias));
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

/// The forward, against the f64 reference, on three shapes.
///
/// `(9, 8, 3)` is there on purpose: 72 positions is MORE than the 64-thread
/// workgroup of the softmax pass, so each thread walks several rows and the
/// strided loop is exercised. A kernel that assumed one thread per position
/// would pass the other two shapes and fail this one.
#[test]
fn attention_forward_matches_the_f64_reference() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);

        for (dim, seed) in [((3u32, 3u32, 4u32), 12345u32), ((2, 2, 8), 777), ((9, 8, 3), 4242)] {
            let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);
            let mut model = infer_model(gpu.clone(), dim).await;
            let params = Params::pseudo_random(channels, seed);
            let layer = model.layers.first().unwrap();
            write_params(
                gpu.as_ref(),
                (&layer.buffers.forward[1], &layer.buffers.forward[2]),
                &params,
            );

            let input = pseudo_random_input(seq * channels, seed ^ 0xABCD, 1.6);
            let got = model.predict(&input);
            let want = reference_attention(&input, seq, channels, &params);

            let mut worst = 0.0f64;
            for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let error = (*g as f64 - w).abs() / w.abs().max(1e-3);
                if error > worst {
                    worst = error;
                }
                assert!(
                    error < 2e-4,
                    "shape {dim:?}, element {index}: GPU {g} vs f64 reference {w} \
                     (relative error {error:.2e})"
                );
            }
            assert!(worst < 2e-4, "shape {dim:?}: worst relative error {worst:.2e}");
        }
    });
}

/// With `W_o = 0` and `b_o = 0` the layer is the exact identity — BIT FOR BIT,
/// not merely close. This is what makes the zero-initialised output projection
/// safe to drop into a trained network: at step 0 it contributes nothing at all.
#[test]
fn a_zero_output_projection_makes_the_layer_the_identity() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = (4u32, 4u32, 6u32);
        let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);

        let mut model = infer_model(gpu.clone(), dim).await;
        let params = Params::pseudo_random(channels, 999).zero_output_projection(channels);
        let layer = model.layers.first().unwrap();
        write_params(
            gpu.as_ref(),
            (&layer.buffers.forward[1], &layer.buffers.forward[2]),
            &params,
        );

        let input = pseudo_random_input(seq * channels, 31337, 2.0);
        let got = model.predict(&input);

        assert_eq!(
            got, input,
            "a zero output projection must reproduce the input bit for bit"
        );
    });
}

/// The layer as it is actually BUILT starts as the identity: nothing writes the
/// weights here, the initialisation does. Guards the wiring of
/// `BufferInit::RandomWeightsZeroTail`, which the test above bypasses.
#[test]
fn a_freshly_built_attention_layer_starts_as_the_identity() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = (4u32, 4u32, 6u32);
        let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);

        let mut model = infer_model(gpu.clone(), dim).await;
        let layer = model.layers.first().unwrap();

        // Q, K and V must NOT be zero — a layer initialised to all zeros would
        // also pass the identity check, and would be dead on arrival.
        let weights = read_back_f32(
            gpu.as_ref(),
            &layer.buffers.forward[1],
            layer.buffers.forward[1].size(),
        )
        .unwrap();
        let cc = channels * channels;
        assert!(
            weights[..3 * cc].iter().any(|w| *w != 0.0),
            "Q/K/V must be drawn, not zeroed"
        );
        assert!(
            weights[3 * cc..].iter().all(|w| *w == 0.0),
            "W_o must start at exactly zero"
        );

        let input = pseudo_random_input(seq * channels, 24680, 1.3);
        assert_eq!(
            model.predict(&input),
            input,
            "a freshly built attention layer must pass its input through untouched"
        );
    });
}

/// Softmax stability. Scores large enough to overflow `exp` in f32 must still
/// produce a finite, normalised distribution — that is what subtracting the row
/// max buys. Driven through the public forward with a deliberately huge input.
#[test]
fn attention_survives_scores_that_would_overflow_exp() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = (4u32, 4u32, 4u32);
        let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);

        let mut model = infer_model(gpu.clone(), dim).await;
        let params = Params::pseudo_random(channels, 5150);
        let layer = model.layers.first().unwrap();
        write_params(
            gpu.as_ref(),
            (&layer.buffers.forward[1], &layer.buffers.forward[2]),
            &params,
        );

        // |x| ~ 300 with |W| ~ 0.5 puts qᵀk in the tens of thousands; exp() of
        // that is +inf in f32, and inf/inf is NaN.
        let input = pseudo_random_input(seq * channels, 8675309, 600.0);
        let got = model.predict(&input);

        assert!(
            got.iter().all(|value| value.is_finite()),
            "attention produced non-finite values on large scores: {:?}",
            got.iter().take(8).collect::<Vec<_>>()
        );
        // And it is still the reference's answer, which computes in f64 and has
        // no overflow to survive in the first place.
        let want = reference_attention(&input, seq, channels, &params);
        for (g, w) in got.iter().zip(want.iter()) {
            let error = (*g as f64 - w).abs() / w.abs().max(1e-1);
            assert!(error < 1e-3, "GPU {g} vs reference {w}");
        }
    });
}

// ---------------------------------------------------------------------------
// Backward — finite differences
// ---------------------------------------------------------------------------

/// Central differences on the loss, for every scalar of the packed weight (and
/// bias) tensor, against the gradient the backward pass produced.
///
/// The step `eps` is a compromise the f32 graph forces: too small and the loss
/// difference disappears into the quantisation floor of a float, too large and
/// the second-order term of the expansion shows up. 2e-3 sits between the two
/// for activations of this scale; the same floor is why the tolerance is 3% of
/// the gradient's own magnitude rather than something like 1e-6.
fn assert_gradients_match_finite_differences(
    label: &str,
    dim: (u32, u32, u32),
    take_bias: bool,
) {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);
        let mut model = training_model(gpu.clone(), dim, 1).await;

        let params = Params::pseudo_random(channels, 606);
        let input = pseudo_random_input(seq * channels, 909, 1.2);
        let target = pseudo_random_input(seq * channels, 313, 0.9);

        let write = |model: &Model<Training>, weights: &[f32], bias: &[f32]| {
            let layer = model.layers.first().unwrap();
            model.gpu.queue.write_buffer(
                layer.buffers.forward[1].as_ref(),
                0,
                bytemuck::cast_slice(weights),
            );
            model.gpu.queue.write_buffer(
                layer.buffers.forward[2].as_ref(),
                0,
                bytemuck::cast_slice(bias),
            );
        };

        // Analytic gradients, from the GPU backward pass.
        write(&model, &params.weights, &params.bias);
        model.train_step(&input, &target);

        let layer = model.layers.first().unwrap();
        let bindings = layer.ty.get_optimizer_bindings().unwrap();
        let backward = layer.buffers.backward.as_ref().unwrap();
        let index = if take_bias {
            bindings.grad_bias_backward_index
        } else {
            bindings.grad_weights_backward_index
        };
        let grad_buffer = &backward[index];
        let analytic = read_back_f32(gpu.as_ref(), grad_buffer, grad_buffer.size()).unwrap();

        // Numeric gradients, from the loss alone.
        let eps = 2e-3f32;
        let loss_at = |model: &mut Model<Training>, weights: &[f32], bias: &[f32]| -> f32 {
            write(model, weights, bias);
            model.train_step(&input, &target);
            model.read_last_loss()
        };

        let count = analytic.len();
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        let mut numeric = vec![0.0f32; count];
        for i in 0..count {
            let (mut plus_w, mut plus_b) = (params.weights.clone(), params.bias.clone());
            let (mut minus_w, mut minus_b) = (params.weights.clone(), params.bias.clone());
            if take_bias {
                plus_b[i] += eps;
                minus_b[i] -= eps;
            } else {
                plus_w[i] += eps;
                minus_w[i] -= eps;
            }
            let lp = loss_at(&mut model, &plus_w, &plus_b);
            let lm = loss_at(&mut model, &minus_w, &minus_b);
            numeric[i] = (lp - lm) / (2.0 * eps);

            let denom = analytic[i].abs().max(numeric[i].abs()).max(1e-3);
            let error = (analytic[i] - numeric[i]).abs() / denom;
            if error > worst {
                worst = error;
                worst_at = i;
            }
        }

        let cc = channels * channels;
        let block = |i: usize| ["W_q", "W_k", "W_v", "W_o"][i / if take_bias { channels } else { cc }];
        assert!(
            worst < 3e-2,
            "\n{label}: backward disagrees with finite differences.\n\
             worst at index {worst_at} (projection {}): analytic {} vs numeric {}\n\
             worst relative error {worst:.4}\n",
            block(worst_at),
            analytic[worst_at],
            numeric[worst_at]
        );
    });
}

/// All four projection matrices at once — they live in one tensor, so one
/// sweep covers Q, K, V and W_o, and the assertion names which block failed.
#[test]
fn attention_weight_gradients_match_finite_differences() {
    assert_gradients_match_finite_differences("weights", (3, 3, 4), false);
}

#[test]
fn attention_bias_gradients_match_finite_differences() {
    assert_gradients_match_finite_differences("biases", (3, 3, 4), true);
}

/// The gradient that flows BACKWARD OUT of the layer, including the residual
/// branch. A layer can have perfect weight gradients and still starve
/// everything upstream of it.
#[test]
fn attention_input_gradients_match_finite_differences() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = (3u32, 3u32, 4u32);
        let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);
        let mut model = training_model(gpu.clone(), dim, 1).await;

        let params = Params::pseudo_random(channels, 2024);
        let input = pseudo_random_input(seq * channels, 55, 1.1);
        let target = pseudo_random_input(seq * channels, 66, 0.8);

        {
            let layer = model.layers.first().unwrap();
            write_params(
                gpu.as_ref(),
                (&layer.buffers.forward[1], &layer.buffers.forward[2]),
                &params,
            );
        }
        model.train_step(&input, &target);

        let layer = model.layers.first().unwrap();
        let grad_index = layer.ty.get_back_grad_input_index().unwrap();
        let grad_buffer = &layer.buffers.backward.as_ref().unwrap()[grad_index];
        let analytic = read_back_f32(gpu.as_ref(), grad_buffer, grad_buffer.size()).unwrap();

        let eps = 2e-3f32;
        let loss_at = |model: &mut Model<Training>, x: &[f32]| -> f32 {
            {
                let layer = model.layers.first().unwrap();
                write_params(
                    model.gpu.as_ref(),
                    (&layer.buffers.forward[1], &layer.buffers.forward[2]),
                    &params,
                );
            }
            model.train_step(x, &target);
            model.read_last_loss()
        };

        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for i in 0..input.len() {
            let mut plus = input.clone();
            plus[i] += eps;
            let mut minus = input.clone();
            minus[i] -= eps;
            let numeric = (loss_at(&mut model, &plus) - loss_at(&mut model, &minus)) / (2.0 * eps);
            let denom = analytic[i].abs().max(numeric.abs()).max(1e-3);
            let error = (analytic[i] - numeric).abs() / denom;
            if error > worst {
                worst = error;
                worst_at = i;
            }
        }

        assert!(
            worst < 3e-2,
            "\ngrad_input disagrees with finite differences at element {worst_at} \
             (analytic {}), worst relative error {worst:.4}\n",
            analytic[worst_at]
        );
    });
}

// ---------------------------------------------------------------------------
// The batch axis — the failure this layer is uniquely exposed to
// ---------------------------------------------------------------------------

/// A batch of B samples must produce exactly what B separate single-sample
/// passes produce.
///
/// The samples are given DIFFERENT SCALES on purpose (0.4, 1.5, 2.6, …): a
/// missing slice offset that mixes two samples of similar magnitude can hide
/// under a relative tolerance, mixing a 0.4-scaled sample with a 3.7-scaled one
/// cannot.
#[test]
fn a_batched_pass_equals_the_same_samples_run_separately() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = (3u32, 3u32, 4u32);
        let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);
        let sample_len = seq * channels;
        let batch = 5usize;

        let params = Params::pseudo_random(channels, 4711);
        let samples: Vec<Vec<f32>> = (0..batch)
            .map(|s| pseudo_random_input(sample_len, 100 + s as u32, 0.4 + 0.8 * s as f32))
            .collect();

        // Reference: one sample at a time, through a batch-1 graph.
        let mut single = infer_model(gpu.clone(), dim).await;
        {
            let layer = single.layers.first().unwrap();
            write_params(
                gpu.as_ref(),
                (&layer.buffers.forward[1], &layer.buffers.forward[2]),
                &params,
            );
        }
        let separately: Vec<Vec<f32>> = samples.iter().map(|s| single.predict(s)).collect();

        // The batched graph, all samples at once.
        let mut batched = Model::<Training>::new_training(
            gpu.clone(),
            0.0,
            batch as u32,
            LossMethod::MeanSquared,
        )
        .await;
        batched
            .add_layer(LayerTypes::Attention(AttentionType::new(Dim3::new(dim))))
            .unwrap();
        batched.build().unwrap();
        {
            let layer = batched.layers.first().unwrap();
            write_params(
                gpu.as_ref(),
                (&layer.buffers.forward[1], &layer.buffers.forward[2]),
                &params,
            );
        }

        let interleaved: Vec<f32> = samples.concat();
        batched.gpu.queue.write_buffer(
            batched.layers.first().unwrap().buffers.forward[0].as_ref(),
            0,
            bytemuck::cast_slice(&interleaved),
        );
        let mut encoder = batched.gpu.device.create_command_encoder(&Default::default());
        for layer in &batched.layers {
            layer.encode_pass(&mut encoder);
        }
        batched.gpu.queue.submit([encoder.finish()]);

        let output_buffer = batched.layers.last().unwrap().buffers.forward.last().unwrap();
        let all = read_back_f32(gpu.as_ref(), output_buffer, output_buffer.size()).unwrap();

        for (s, expected) in separately.iter().enumerate() {
            let slice = &all[s * sample_len..(s + 1) * sample_len];
            for (i, (got, want)) in slice.iter().zip(expected.iter()).enumerate() {
                let error = (got - want).abs() / want.abs().max(1e-3);
                assert!(
                    error < 1e-4,
                    "sample {s}, element {i}: batched {got} vs alone {want} \
                     (relative error {error:.2e}) — the batch axis is leaking"
                );
            }
        }
    });
}

/// The anti-leak test, in the sharpest form available: change ONE sample's data
/// and every OTHER sample's output must be bit-for-bit unchanged.
///
/// This is the GroupNorm anti-leak test's shape, applied where it matters more.
/// GroupNorm could only leak through a shared statistic; attention has an
/// explicit all-to-all over positions, so a base offset computed once for the
/// dispatch instead of once per sample would silently let position `n` of one
/// image attend to a position of another. The relative-tolerance test above
/// could conceivably absorb that; equality cannot.
#[test]
fn one_samples_data_cannot_reach_another_samples_output() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = (3u32, 3u32, 4u32);
        let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);
        let sample_len = seq * channels;
        let batch = 4usize;
        let params = Params::pseudo_random(channels, 1234);

        let mut model =
            Model::<Training>::new_training(gpu.clone(), 0.0, batch as u32, LossMethod::MeanSquared)
                .await;
        model
            .add_layer(LayerTypes::Attention(AttentionType::new(Dim3::new(dim))))
            .unwrap();
        model.build().unwrap();
        {
            let layer = model.layers.first().unwrap();
            write_params(
                gpu.as_ref(),
                (&layer.buffers.forward[1], &layer.buffers.forward[2]),
                &params,
            );
        }

        let run = |data: &[f32]| -> Vec<f32> {
            model.gpu.queue.write_buffer(
                model.layers.first().unwrap().buffers.forward[0].as_ref(),
                0,
                bytemuck::cast_slice(data),
            );
            let mut encoder = model.gpu.device.create_command_encoder(&Default::default());
            for layer in &model.layers {
                layer.encode_pass(&mut encoder);
            }
            model.gpu.queue.submit([encoder.finish()]);
            let buffer = model.layers.last().unwrap().buffers.forward.last().unwrap();
            read_back_f32(gpu.as_ref(), buffer, buffer.size()).unwrap()
        };

        let pristine: Vec<f32> = (0..batch)
            .flat_map(|s| pseudo_random_input(sample_len, 700 + s as u32, 1.0 + s as f32))
            .collect();
        let before = run(&pristine);

        // EVERY sample takes its turn as the perturbed one. Perturbing a single
        // fixed sample only proves that *it* does not leak outward; a kernel
        // that offsets q correctly but reads k and v from a hard-coded sample 0
        // passes that weaker check completely — the samples that read sample 0
        // are unaffected by a change to sample 2. Sweeping the whole batch
        // tests both directions for every pair, which is the actual property:
        // no sample's data may reach any other sample's output.
        for victim in 0..batch {
            let mut data = pristine.clone();
            let perturbed = pseudo_random_input(sample_len, 31415 + victim as u32, 9.0);
            data[victim * sample_len..(victim + 1) * sample_len].copy_from_slice(&perturbed);
            let after = run(&data);

            for s in 0..batch {
                let range = s * sample_len..(s + 1) * sample_len;
                if s == victim {
                    assert_ne!(
                        before[range.clone()],
                        after[range],
                        "sample {victim} was replaced; its own output must change"
                    );
                } else {
                    assert_eq!(
                        before[range.clone()],
                        after[range],
                        "sample {s} changed when only sample {victim}'s data was replaced — \
                         attention is mixing samples"
                    );
                }
            }
        }
    });
}

/// The parameter gradients of a batch must be the SUM of the per-sample
/// gradients — a reduction over the batch, not a slice of it. A kernel that
/// summed over one sample's positions and stopped would halve, or quarter, the
/// gradient without any other symptom.
#[test]
fn batched_parameter_gradients_sum_the_per_sample_gradients() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = (3u32, 3u32, 4u32);
        let (seq, channels) = ((dim.0 * dim.1) as usize, dim.2 as usize);
        let sample_len = seq * channels;
        let batch = 3usize;
        let params = Params::pseudo_random(channels, 8080);

        let samples: Vec<Vec<f32>> = (0..batch)
            .map(|s| pseudo_random_input(sample_len, 200 + s as u32, 0.5 + 0.6 * s as f32))
            .collect();
        let targets: Vec<Vec<f32>> = (0..batch)
            .map(|s| pseudo_random_input(sample_len, 300 + s as u32, 0.7))
            .collect();

        let gradients_of = |model: &Model<Training>| -> (Vec<f32>, Vec<f32>) {
            let layer = model.layers.first().unwrap();
            let bindings = layer.ty.get_optimizer_bindings().unwrap();
            let backward = layer.buffers.backward.as_ref().unwrap();
            let gw = &backward[bindings.grad_weights_backward_index];
            let gb = &backward[bindings.grad_bias_backward_index];
            (
                read_back_f32(gpu.as_ref(), gw, gw.size()).unwrap(),
                read_back_f32(gpu.as_ref(), gb, gb.size()).unwrap(),
            )
        };

        // Per-sample, through a batch-1 graph, summed on the CPU.
        let mut single = training_model(gpu.clone(), dim, 1).await;
        let mut sum_w = vec![0.0f64; 4 * channels * channels];
        let mut sum_b = vec![0.0f64; 4 * channels];
        for s in 0..batch {
            {
                let layer = single.layers.first().unwrap();
                write_params(
                    gpu.as_ref(),
                    (&layer.buffers.forward[1], &layer.buffers.forward[2]),
                    &params,
                );
            }
            single.train_step(&samples[s], &targets[s]);
            let (w, b) = gradients_of(&single);
            for (acc, value) in sum_w.iter_mut().zip(w.iter()) {
                *acc += *value as f64;
            }
            for (acc, value) in sum_b.iter_mut().zip(b.iter()) {
                *acc += *value as f64;
            }
        }

        // The same three samples, batched.
        let mut model =
            Model::<Training>::new_training(gpu.clone(), 0.0, batch as u32, LossMethod::MeanSquared)
                .await;
        model
            .add_layer(LayerTypes::Attention(AttentionType::new(Dim3::new(dim))))
            .unwrap();
        model.build().unwrap();
        {
            let layer = model.layers.first().unwrap();
            write_params(
                gpu.as_ref(),
                (&layer.buffers.forward[1], &layer.buffers.forward[2]),
                &params,
            );
        }
        model.gpu.queue.write_buffer(
            model.layers.first().unwrap().buffers.forward[0].as_ref(),
            0,
            bytemuck::cast_slice(&samples.concat()),
        );
        model.gpu.queue.write_buffer(
            model.loss_layer.as_ref().unwrap().buffers.forward[1].as_ref(),
            0,
            bytemuck::cast_slice(&targets.concat()),
        );
        model.train_step_report_batched(batch, false, |_| {});
        let (batched_w, batched_b) = gradients_of(&model);

        for (label, got, want) in [
            ("weights", &batched_w, &sum_w),
            ("biases", &batched_b, &sum_b),
        ] {
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let error = (*g as f64 - w).abs() / w.abs().max(1e-3);
                assert!(
                    error < 1e-3,
                    "{label}[{i}]: batched gradient {g} vs the sum of the per-sample \
                     gradients {w} (relative error {error:.2e})"
                );
            }
        }
    });
}

