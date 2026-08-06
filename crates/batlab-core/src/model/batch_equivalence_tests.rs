//! Equivalence of the batched graph with the sequential one it replaced.
//!
//! The batch axis now lives in the buffer sizes and the dispatch grids
//! (`docs/reports/BATCH_DISPATCH_DESIGN.md`). Two failure modes dominate that
//! kind of refactor, and both are silent:
//!
//! 1. **a missing slice offset** — a kernel reads or writes sample 0's data
//!    while computing sample 3's. Nothing crashes: the shapes are right, the
//!    values are plausible, the loss still goes down.
//! 2. **a truncated reduction** — a gradient sums over `positions` instead of
//!    `batch * positions`. Again nothing crashes; the model just trains on a
//!    fraction of its batch.
//!
//! The tests below are built to catch exactly those. The reference is not a
//! frozen fixture but `DiffusionTask::train_step_batch_sequential` and the
//! batch-1 graph — i.e. the pre-batching code, still executable — plus an f64
//! oracle that knows neither implementation.

use crate::gpu_context::GpuContext;
use crate::model::debug::read_back_f32;
use crate::model::layer_types::{
    ActivationMethod, ActivationType, ConvolutionType, GroupNormType, LayerType, LayerTypes,
    LossMethod, UpsampleConvType,
};
use crate::model::optimizer::OptimizerKind;
use crate::model::{Dim3, Model, PaddingMode, Training};
use std::sync::Arc;

const BATCH: usize = 5;
const DIM_IN: (u32, u32, u32) = (6, 6, 3);
const KERNELS: u32 = 4;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// conv(3x3, Same) -> GroupNorm(2) -> SiLU -> conv(3x3, Same, 1 kernel).
///
/// Small, but it exercises all three families the batch axis touches: a
/// per-activation kernel, a per-group reduction whose statistics must stay
/// per-sample, and two parameter reductions that must now sum over the batch.
async fn stacked_model(gpu: Arc<GpuContext>, batch: u32) -> Model<Training> {
    let mut model =
        Model::new_training_with_optimizer(gpu, 0.01, batch, LossMethod::MeanSquared, OptimizerKind::Sgd)
            .await;
    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new(DIM_IN),
            KERNELS,
            Dim3::new((3, 3, DIM_IN.2)),
            1,
            PaddingMode::Same,
        )))
        .unwrap();
    model
        .add_layer(LayerTypes::GroupNorm(GroupNormType::new(
            Dim3::new((DIM_IN.0, DIM_IN.1, KERNELS)),
            2,
        )))
        .unwrap();
    model
        .add_layer(LayerTypes::Activation(ActivationType::new(
            ActivationMethod::Silu,
            Dim3::default(),
        )))
        .unwrap();
    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new((DIM_IN.0, DIM_IN.1, KERNELS)),
            1,
            Dim3::new((3, 3, KERNELS)),
            1,
            PaddingMode::Same,
        )))
        .unwrap();
    model.build().unwrap();
    model
}

/// A single convolution — the shape the f64 oracle can arbitrate end to end.
async fn single_conv_model(gpu: Arc<GpuContext>, batch: u32) -> Model<Training> {
    let mut model =
        Model::new_training_with_optimizer(gpu, 0.01, batch, LossMethod::MeanSquared, OptimizerKind::Sgd)
            .await;
    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new(DIM_IN),
            KERNELS,
            Dim3::new((3, 3, DIM_IN.2)),
            1,
            PaddingMode::Same,
        )))
        .unwrap();
    model.build().unwrap();
    model
}

/// conv(s2) -> up-conv(x2) -> concat(skip) -> conv.
///
/// The U-net shape, and the one where the batch axis is easiest to get wrong:
/// Concat is the only layer whose three tensors have DIFFERENT per-sample
/// lengths (out = input + skip channels), so a single shared `sample * len`
/// offset is wrong for it — and UpsampleConv reduces onto its weights over a
/// tap map that is not the convolution's. Both are in every real model of the
/// repository and in neither of the fixtures above.
async fn skip_model(gpu: Arc<GpuContext>, batch: u32) -> Model<Training> {
    let mut model = Model::new_training_with_optimizer(
        gpu,
        0.01,
        batch,
        LossMethod::MeanSquared,
        OptimizerKind::Sgd,
    )
    .await;
    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new(DIM_IN),
            KERNELS,
            Dim3::new((3, 3, DIM_IN.2)),
            1,
            PaddingMode::Same,
        )))
        .unwrap();
    model.mark_output("skip").unwrap();
    // Stride 2: 6x6 -> 3x3, so the up-conv has real work to undo.
    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new((DIM_IN.0, DIM_IN.1, KERNELS)),
            KERNELS,
            Dim3::new((3, 3, KERNELS)),
            2,
            PaddingMode::Same,
        )))
        .unwrap();
    model
        .add_layer(LayerTypes::UpsampleConv(UpsampleConvType::new(
            Dim3::default(),
            2,
            KERNELS,
            Dim3::new((3, 3, KERNELS)),
            PaddingMode::Same,
        )))
        .unwrap();
    model.add_concat("skip").unwrap();
    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new((DIM_IN.0, DIM_IN.1, 2 * KERNELS)),
            1,
            Dim3::new((3, 3, 2 * KERNELS)),
            1,
            PaddingMode::Same,
        )))
        .unwrap();
    model.build().unwrap();
    model
}

/// Deterministic pseudo-random data, decorrelated between samples.
///
/// Sample `sample` is deliberately given its own *scale* as well as its own
/// values: a slice-offset bug that mixes two samples of similar magnitude can
/// stay under a relative tolerance, whereas mixing a 0.2-scaled sample with a
/// 3.0-scaled one cannot.
fn sample_data(sample: usize, len: usize) -> Vec<f32> {
    let scale = 0.2 + 0.7 * sample as f32;
    let mut state = 0x9E37_79B9u32 ^ (sample as u32 + 1).wrapping_mul(0x85EB_CA6B);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

fn input_len(model: &Model<Training>) -> usize {
    model.input_dim().unwrap().length() as usize
}

fn output_len(model: &Model<Training>) -> usize {
    model.output_dim().unwrap().length() as usize
}

fn write_input(model: &Model<Training>, data: &[f32]) {
    model.gpu.queue.write_buffer(
        model.layers.first().unwrap().buffers.forward[0].as_ref(),
        0,
        bytemuck::cast_slice(data),
    );
}

fn write_target(model: &Model<Training>, data: &[f32]) {
    model.gpu.queue.write_buffer(
        model.loss_layer.as_ref().unwrap().buffers.forward[1].as_ref(),
        0,
        bytemuck::cast_slice(data),
    );
}

fn run_forward(model: &Model<Training>, batch: u32) {
    let mut encoder = model
        .gpu
        .device
        .create_command_encoder(&Default::default());
    for layer in &model.layers {
        layer.encode_pass_with_batch(&mut encoder, batch);
    }
    model.gpu.queue.submit([encoder.finish()]);
}

fn read_output(model: &Model<Training>, count: usize) -> Vec<f32> {
    let last = model.layers.last().unwrap();
    let bytes = (count * std::mem::size_of::<f32>()) as u64;
    read_back_f32(model.gpu.as_ref(), last.buffers.forward.last().unwrap(), bytes).unwrap()
}

/// `(grad_weights, grad_bias)` of every trainable layer, in layer order.
fn read_parameter_gradients(model: &Model<Training>) -> Vec<(Vec<f32>, Vec<f32>)> {
    model
        .layers
        .iter()
        .filter_map(|layer| {
            let bindings = layer.ty.get_optimizer_bindings()?;
            let backward = layer.buffers.backward.as_ref()?;
            let gw = &backward[bindings.grad_weights_backward_index];
            let gb = &backward[bindings.grad_bias_backward_index];
            Some((
                read_back_f32(model.gpu.as_ref(), gw, gw.size()).unwrap(),
                read_back_f32(model.gpu.as_ref(), gb, gb.size()).unwrap(),
            ))
        })
        .collect()
}

/// Run one batched step over `inputs`/`targets` (already interleaved) and
/// return the parameter gradients it accumulated.
fn batched_gradients(
    model: &mut Model<Training>,
    inputs: &[f32],
    targets: &[f32],
    batch: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    write_input(model, inputs);
    write_target(model, targets);
    model.train_step_report_batched(batch, false, |_| {});
    read_parameter_gradients(model)
}

/// The pre-batching accumulation: one submit per sample, `+=` between them.
fn sequential_gradients(
    model: &mut Model<Training>,
    inputs: &[f32],
    targets: &[f32],
    batch: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let in_len = input_len(model);
    let out_len = output_len(model);
    model.begin_batch_accumulation();
    for sample in 0..batch {
        write_input(model, &inputs[sample * in_len..(sample + 1) * in_len]);
        write_target(model, &targets[sample * out_len..(sample + 1) * out_len]);
        model.train_step_with_prepass_no_opt(|_| {});
    }
    // Read BEFORE the optimiser update: it does not touch the gradient
    // buffers, but reading first makes that independent of whether it does.
    read_parameter_gradients(model)
}

/// Largest disagreement between two gradient vectors, **relative to the scale
/// of the vector** (`max |a-b| / max(||a||inf, ||b||inf)`).
///
/// Not the per-element relative error, and the difference is not cosmetic. A
/// gradient buffer spans orders of magnitude, and its small entries are small
/// because they are nearly-cancelled sums of large terms. On the U-net fixture
/// below, entry 100 of the first convolution's `grad_weights` is 7.12e-4 in a
/// buffer whose largest entry is 4.61: the two paths differ there by 2.98e-7,
/// which is the f32 rounding floor for terms of that size — and which the
/// per-element metric reports as a 4.2e-4 "relative error". That number does
/// not measure agreement on the gradient; it measures how close to zero its
/// smallest entry happens to fall.
///
/// The vector-relative error is the standard answer for a quantity like this,
/// and it stays strict where it matters: a truncated batch reduction or a
/// missed slice offset moves the LARGE entries, which are exactly the ones
/// `||.||inf` is made of. That claim is checked by mutation (§4.6 of the
/// report), not asserted.
fn worst_relative_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let scale = a
        .iter()
        .chain(b)
        .map(|v| v.abs() as f64)
        .fold(0.0f64, f64::max)
        .max(1e-12);
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
        / scale
}

/// Same metric, against the f64 oracle.
fn worst_relative_to_f64(a: &[f32], oracle: &[f64]) -> f64 {
    assert_eq!(a.len(), oracle.len(), "length mismatch");
    let scale = a
        .iter()
        .map(|v| v.abs() as f64)
        .chain(oracle.iter().map(|v| v.abs()))
        .fold(0.0f64, f64::max)
        .max(1e-12);
    a.iter()
        .zip(oracle)
        .map(|(x, y)| ((*x as f64) - y).abs())
        .fold(0.0f64, f64::max)
        / scale
}

// ---------------------------------------------------------------------------
// 1. Forward — bit for bit, sample by sample
// ---------------------------------------------------------------------------

/// The forward pass contains no reduction over the batch axis: sample `i` of a
/// batch of `B` reads only its own slice, in the same order, from the same
/// weights. Its output must therefore be **bit-identical** to running that
/// sample alone.
///
/// Asserting on `to_bits()` rather than a tolerance is deliberate, and it is
/// what gives this test its power: a wrong slice offset that happens to land on
/// a neighbouring sample produces numbers of the right magnitude, which a 1e-5
/// threshold would only catch by luck.
#[test]
fn forward_is_bit_identical_sample_by_sample() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let batched = stacked_model(gpu.clone(), BATCH as u32).await;
        let single = stacked_model(gpu.clone(), 1).await;

        let in_len = input_len(&batched);
        let out_len = output_len(&batched);
        let inputs: Vec<f32> = (0..BATCH).flat_map(|s| sample_data(s, in_len)).collect();

        write_input(&batched, &inputs);
        run_forward(&batched, BATCH as u32);
        let batched_out = read_output(&batched, BATCH * out_len);

        for sample in 0..BATCH {
            write_input(&single, &inputs[sample * in_len..(sample + 1) * in_len]);
            run_forward(&single, 1);
            let alone = read_output(&single, out_len);
            let slice = &batched_out[sample * out_len..(sample + 1) * out_len];

            let differing = alone
                .iter()
                .zip(slice)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "\nsample {sample} of a batch of {BATCH} is not bit-identical to the \
                 same sample run alone ({differing}/{out_len} elements differ)\n\
                 alone[..4]: {:?}\nbatched[..4]: {:?}\n",
                &alone[..4.min(alone.len())],
                &slice[..4.min(slice.len())],
            );
        }

        // Guard against a vacuous pass: the samples must genuinely differ, or
        // "every slice matches" would be trivially true.
        let first = &batched_out[..out_len];
        let last = &batched_out[(BATCH - 1) * out_len..];
        assert!(
            worst_relative_diff(first, last) > 1e-2,
            "the test samples are too alike for slice mixing to be detectable"
        );
    });
}

// ---------------------------------------------------------------------------
// 2. Gradients — batched reduction vs sequential accumulation
// ---------------------------------------------------------------------------

/// `grad_weights` and `grad_bias` used to be built by `B` tree reductions
/// followed by a sequential `+=` chain; they are now built by ONE tree
/// reduction over `batch * positions`. The sum is mathematically the same, the
/// floating-point order is not — so this is a tolerance comparison, and §3.1 of
/// the design note says so in advance rather than after the fact.
#[test]
fn parameter_gradients_match_the_sequential_accumulation() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut batched = stacked_model(gpu.clone(), BATCH as u32).await;
        let mut single = stacked_model(gpu.clone(), 1).await;

        let in_len = input_len(&batched);
        let out_len = output_len(&batched);
        let inputs: Vec<f32> = (0..BATCH).flat_map(|s| sample_data(s, in_len)).collect();
        let targets: Vec<f32> = (0..BATCH)
            .flat_map(|s| sample_data(s + 100, out_len))
            .collect();

        let new = batched_gradients(&mut batched, &inputs, &targets, BATCH);
        let old = sequential_gradients(&mut single, &inputs, &targets, BATCH);

        assert_eq!(new.len(), old.len(), "trainable layer count changed");
        for (layer, ((new_w, new_b), (old_w, old_b))) in new.iter().zip(&old).enumerate() {
            for (name, a, b) in [
                ("grad_weights", new_w, old_w),
                ("grad_bias", new_b, old_b),
            ] {
                let diff = worst_relative_diff(a, b);
                println!("layer {layer} {name}: worst error vs scale {diff:e}");
                assert!(
                    diff < 1e-4,
                    "\nlayer {layer} {name} disagrees with the sequential accumulation \
                     (worst error vs scale {diff:e})\nbatched[..4]: {:?}\nsequential[..4]: {:?}\n",
                    &a[..4.min(a.len())],
                    &b[..4.min(b.len())],
                );
                // Not vacuous: a gradient of all zeros would match anything.
                assert!(
                    a.iter().any(|v| v.abs() > 1e-6),
                    "layer {layer} {name} is all zeros — the test proves nothing"
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 3. The batch reduction, arbitrated by an f64 oracle
// ---------------------------------------------------------------------------

/// f64 reference for a `Same`-padded convolution and its parameter gradients,
/// summed over the batch.
///
/// Written as a **scatter** over the forward tap map
/// `(oy,ox,ky,kx) -> (oy*s+ky-pad_y, ox*s+kx-pad_x)` with out-of-bounds taps
/// contributing zero — the definition of finding #1 of `AUDIT_TRAINING.md` —
/// so it validates the shaders' indexing instead of restating it.
struct ConvOracle {
    grad_weights: Vec<f64>,
    grad_bias: Vec<f64>,
}

fn conv_oracle(
    inputs: &[f32],
    targets: &[f32],
    weights: &[f32],
    bias: &[f32],
    batch: usize,
) -> ConvOracle {
    let (ih, iw, ic) = (DIM_IN.0 as usize, DIM_IN.1 as usize, DIM_IN.2 as usize);
    let (oh, ow, k) = (ih, iw, KERNELS as usize); // Same padding, stride 1
    let (kh, kw) = (3usize, 3usize);
    let (pad_y, pad_x) = (kh / 2, kw / 2);
    let in_len = ih * iw * ic;
    let out_len = oh * ow * k;

    let mut grad_weights = vec![0.0f64; k * kh * kw * ic];
    let mut grad_bias = vec![0.0f64; k];

    for b in 0..batch {
        // Forward.
        let mut out = vec![0.0f64; out_len];
        for oy in 0..oh {
            for ox in 0..ow {
                for kk in 0..k {
                    let mut sum = bias[kk] as f64;
                    for ky in 0..kh {
                        let iy = oy as isize + ky as isize - pad_y as isize;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = ox as isize + kx as isize - pad_x as isize;
                            if ix < 0 || ix >= iw as isize {
                                continue;
                            }
                            for kz in 0..ic {
                                let in_i = b * in_len
                                    + iy as usize * iw * ic
                                    + ix as usize * ic
                                    + kz;
                                let w_i = kk * kh * kw * ic + ky * kw * ic + kx * ic + kz;
                                sum += inputs[in_i] as f64 * weights[w_i] as f64;
                            }
                        }
                    }
                    out[oy * ow * k + ox * k + kk] = sum;
                }
            }
        }

        // MSE gradient: 2 * (pred - target) / N, N the PER-SAMPLE length.
        let grad_out: Vec<f64> = (0..out_len)
            .map(|i| 2.0 * (out[i] - targets[b * out_len + i] as f64) / out_len as f64)
            .collect();

        // Scatter into the parameter gradients — summing over the batch.
        for oy in 0..oh {
            for ox in 0..ow {
                for kk in 0..k {
                    let g = grad_out[oy * ow * k + ox * k + kk];
                    grad_bias[kk] += g;
                    for ky in 0..kh {
                        let iy = oy as isize + ky as isize - pad_y as isize;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = ox as isize + kx as isize - pad_x as isize;
                            if ix < 0 || ix >= iw as isize {
                                continue;
                            }
                            for kz in 0..ic {
                                let in_i = b * in_len
                                    + iy as usize * iw * ic
                                    + ix as usize * ic
                                    + kz;
                                let w_i = kk * kh * kw * ic + ky * kw * ic + kx * ic + kz;
                                grad_weights[w_i] += g * inputs[in_i] as f64;
                            }
                        }
                    }
                }
            }
        }
    }

    ConvOracle {
        grad_weights,
        grad_bias,
    }
}

/// Who is right when the two f32 results differ?
///
/// The hard requirement is the first assertion: within 1e-4 of the f64 truth.
///
/// The second one needs more care than the earlier reduction missions did, and
/// the difference is worth stating rather than hiding. `PERF_GROUP_NORM.md`
/// §3.2 and `PERF_CONVOLUTION.md` §4.2 could demand `new <= 1.5 * old` outright,
/// because there the change was purely *how* a fixed set of terms was
/// associated — a tree instead of a chain, over the same `positions` terms.
/// Here the term count itself changes: a lane that used to walk `positions`
/// values and hand its partial to a `+=` between samples now walks
/// `batch * positions` of them in one f32 accumulator. Longer chain, more
/// rounding — measured on `grad_bias` at batch 5: 2.1e-6 against the oracle
/// where the sequential path gets 5.3e-7. That is not a bug and not a
/// regression to fix; it is the arithmetic of the operation being asked for.
///
/// So the clause becomes: either the batched result beats the old 1.5x bar, or
/// it sits within a small multiple of the **rounding floor of the sum it now
/// performs** — `sqrt(n) * f32::EPSILON` for `n` accumulated terms, the
/// standard random-walk estimate. The floor is *derived from the shape*, not
/// fitted to the measurement, and a reduction that is genuinely broken (a
/// truncated batch, a missed slice) is off by O(1) relative, some four orders
/// of magnitude above it. The test still bites.
#[test]
fn batch_reduction_is_arbitrated_by_an_f64_oracle() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut batched = single_conv_model(gpu.clone(), BATCH as u32).await;
        let mut single = single_conv_model(gpu.clone(), 1).await;

        let in_len = input_len(&batched);
        let out_len = output_len(&batched);
        let inputs: Vec<f32> = (0..BATCH).flat_map(|s| sample_data(s, in_len)).collect();
        let targets: Vec<f32> = (0..BATCH)
            .flat_map(|s| sample_data(s + 100, out_len))
            .collect();

        let layer = batched.layers.first().unwrap();
        let weights = read_back_f32(
            gpu.as_ref(),
            &layer.buffers.forward[1],
            layer.buffers.forward[1].size(),
        )
        .unwrap();
        let bias = read_back_f32(
            gpu.as_ref(),
            &layer.buffers.forward[2],
            layer.buffers.forward[2].size(),
        )
        .unwrap();
        let oracle = conv_oracle(&inputs, &targets, &weights, &bias, BATCH);

        let new = batched_gradients(&mut batched, &inputs, &targets, BATCH);
        let old = sequential_gradients(&mut single, &inputs, &targets, BATCH);

        for (name, a, b, reference) in [
            (
                "grad_weights",
                &new[0].0,
                &old[0].0,
                &oracle.grad_weights,
            ),
            ("grad_bias", &new[0].1, &old[0].1, &oracle.grad_bias),
        ] {
            let new_err = worst_relative_to_f64(a, reference);
            let old_err = worst_relative_to_f64(b, reference);
            println!("{name}: batched^f64 {new_err:e} | sequential^f64 {old_err:e}");

            assert!(
                new_err < 1e-4,
                "\nbatched {name} is off the f64 reference by {new_err:e} \
                 (sequential: {old_err:e})\n"
            );
            // Terms a single accumulator now chains: the whole batch's
            // positions, split across the lanes the uniform asks for.
            let positions = (DIM_IN.0 * DIM_IN.1) as f64 * BATCH as f64;
            let floor = positions.sqrt() * f32::EPSILON as f64;
            assert!(
                new_err <= (old_err * 1.5).max(4.0 * floor),
                "\nthe batched {name} reduction is not merely re-associated, it is \
                 genuinely off.\nbatched vs f64:    {new_err:e}\n\
                 sequential vs f64: {old_err:e}\n\
                 rounding floor for a {positions}-term f32 sum: {floor:e}\n"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 4. Isolation between slices
// ---------------------------------------------------------------------------

/// A batch in which every sample but one is zero must produce exactly the
/// gradients of a batch of one holding that sample.
///
/// This is the test that catches a missed offset. If a kernel reads sample 0
/// while writing sample 3, the zero slices leak into the non-zero one and the
/// gradient changes — whereas an equality between two "reasonable" batches
/// could still hold under a symmetric mistake.
#[test]
fn a_batch_with_one_live_sample_equals_a_batch_of_one() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut batched = stacked_model(gpu.clone(), BATCH as u32).await;
        let mut single = stacked_model(gpu.clone(), 1).await;

        let in_len = input_len(&batched);
        let out_len = output_len(&batched);
        // The live sample is deliberately NOT slot 0: an implementation that
        // ignores the offset entirely would pass if it were.
        let live = BATCH - 2;
        let live_input = sample_data(7, in_len);
        let live_target = sample_data(107, out_len);

        let mut inputs = vec![0.0f32; BATCH * in_len];
        let mut targets = vec![0.0f32; BATCH * out_len];
        inputs[live * in_len..(live + 1) * in_len].copy_from_slice(&live_input);
        targets[live * out_len..(live + 1) * out_len].copy_from_slice(&live_target);

        let new = batched_gradients(&mut batched, &inputs, &targets, BATCH);
        let alone = sequential_gradients(&mut single, &live_input, &live_target, 1);

        for (layer, ((new_w, new_b), (one_w, one_b))) in new.iter().zip(&alone).enumerate() {
            for (name, a, b) in [("grad_weights", new_w, one_w), ("grad_bias", new_b, one_b)] {
                // A zero input is not a zero gradient: the zeroed samples still
                // travel through bias, GroupNorm's beta and the MSE target, so
                // this comparison is not "x vs x".
                let diff = worst_relative_diff(a, b);
                assert!(
                    diff < 1e-4,
                    "\nlayer {layer} {name}: a live sample in slot {live} of a batch of \
                     {BATCH} does not reproduce the batch of one it should \
                     (worst error vs scale {diff:e}) — a slice offset is leaking\n"
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 5. GroupNorm statistics stay inside their sample
// ---------------------------------------------------------------------------

/// GroupNorm normalises over (channels-of-a-group x space) **of one sample**.
/// Pooling the statistics over the batch would make a sample's output depend on
/// the images it happens to be batched with, and would break inference (batch
/// 1) against training.
///
/// The two samples here are scaled two orders of magnitude apart, so any
/// pooling of mean or variance moves the normalised output far outside the
/// tolerance instead of nudging it.
#[test]
fn group_norm_statistics_do_not_leak_across_the_batch() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let batch = 2usize;
        let batched = stacked_model(gpu.clone(), batch as u32).await;
        let single = stacked_model(gpu.clone(), 1).await;

        let in_len = input_len(&batched);
        let out_len = output_len(&batched);
        let quiet: Vec<f32> = sample_data(3, in_len).iter().map(|v| v * 0.01).collect();
        let loud: Vec<f32> = sample_data(3, in_len).iter().map(|v| v * 100.0).collect();

        let mut inputs = Vec::with_capacity(batch * in_len);
        inputs.extend_from_slice(&quiet);
        inputs.extend_from_slice(&loud);

        write_input(&batched, &inputs);
        run_forward(&batched, batch as u32);
        let together = read_output(&batched, batch * out_len);

        for (slot, alone_input) in [quiet, loud].iter().enumerate() {
            write_input(&single, alone_input);
            run_forward(&single, 1);
            let alone = read_output(&single, out_len);
            let slice = &together[slot * out_len..(slot + 1) * out_len];
            let diff = worst_relative_diff(&alone, slice);
            assert!(
                diff < 1e-5,
                "\nsample {slot} changes when batched with a sample scaled 10 000x \
                 differently (worst error vs scale {diff:e}) — the group statistics are \
                 being pooled across the batch\n"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 6. The reported loss is still the last sample's
// ---------------------------------------------------------------------------

/// `train_step_report_batch` has always returned the loss of the batch's LAST
/// sample. Keeping that definition is what makes an old-path/new-path loss
/// trajectory comparable; and since the forward carries no batch reduction, the
/// number must match the batch-of-one value to the bit.
#[test]
fn reported_loss_is_the_last_sample_of_the_batch() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut batched = stacked_model(gpu.clone(), BATCH as u32).await;
        let mut single = stacked_model(gpu.clone(), 1).await;

        let in_len = input_len(&batched);
        let out_len = output_len(&batched);
        let inputs: Vec<f32> = (0..BATCH).flat_map(|s| sample_data(s, in_len)).collect();
        let targets: Vec<f32> = (0..BATCH)
            .flat_map(|s| sample_data(s + 100, out_len))
            .collect();

        write_input(&batched, &inputs);
        write_target(&batched, &targets);
        let reported = batched
            .train_step_report_batched(BATCH, true, |_| {})
            .expect("batched step reports a loss");

        let last = BATCH - 1;
        write_input(&single, &inputs[last * in_len..]);
        write_target(&single, &targets[last * out_len..]);
        single.begin_batch_accumulation();
        let expected = single
            .train_step_report_with_prepass_no_opt(|_| {})
            .expect("sequential step reports a loss");

        assert_eq!(
            reported.to_bits(),
            expected.to_bits(),
            "\nreported batch loss {reported} is not the last sample's loss {expected}\n"
        );

        // And it is NOT the first sample's, which is what a forgotten offset
        // would return.
        write_input(&single, &inputs[..in_len]);
        write_target(&single, &targets[..out_len]);
        let first = single
            .train_step_report_with_prepass_no_opt(|_| {})
            .unwrap();
        assert!(
            (first - expected).abs() > 1e-4,
            "the first and last samples have indistinguishable losses; \
             the offset assertion above proves nothing"
        );
    });
}

// ---------------------------------------------------------------------------
// 7. Concat and UpsampleConv — the U-net shape
// ---------------------------------------------------------------------------

/// The same two properties as tests 1 and 2, on the layers the fixtures above
/// do not reach.
///
/// Concat earns its own test: it is the ONLY layer whose input, skip and output
/// have different per-sample lengths, so it needs three distinct sample offsets
/// where every other kernel needs one. A single shared offset there is exactly
/// the mistake that produces plausible-looking, wrong data.
#[test]
fn concat_and_upsample_conv_carry_the_batch_correctly() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let mut batched = skip_model(gpu.clone(), BATCH as u32).await;
        let mut single = skip_model(gpu.clone(), 1).await;

        let in_len = input_len(&batched);
        let out_len = output_len(&batched);
        let inputs: Vec<f32> = (0..BATCH).flat_map(|s| sample_data(s, in_len)).collect();
        let targets: Vec<f32> = (0..BATCH)
            .flat_map(|s| sample_data(s + 100, out_len))
            .collect();

        // Forward: bit for bit, sample by sample.
        write_input(&batched, &inputs);
        run_forward(&batched, BATCH as u32);
        let batched_out = read_output(&batched, BATCH * out_len);
        for sample in 0..BATCH {
            write_input(&single, &inputs[sample * in_len..(sample + 1) * in_len]);
            run_forward(&single, 1);
            let alone = read_output(&single, out_len);
            let slice = &batched_out[sample * out_len..(sample + 1) * out_len];
            let differing = alone
                .iter()
                .zip(slice)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "\nsample {sample} of the skip/upsample model is not bit-identical to \
                 the same sample run alone ({differing}/{out_len} elements differ)\n"
            );
        }

        // Gradients, including the up-conv's and both convs' around the concat.
        let new = batched_gradients(&mut batched, &inputs, &targets, BATCH);
        let old = sequential_gradients(&mut single, &inputs, &targets, BATCH);
        assert_eq!(new.len(), old.len());
        for (layer, ((new_w, new_b), (old_w, old_b))) in new.iter().zip(&old).enumerate() {
            for (name, a, b) in [("grad_weights", new_w, old_w), ("grad_bias", new_b, old_b)] {
                let diff = worst_relative_diff(a, b);
                println!("skip model, layer {layer} {name}: worst error vs scale {diff:e}");
                assert!(
                    diff < 1e-4,
                    "\nskip model layer {layer} {name} disagrees with the sequential \
                     accumulation (worst error vs scale {diff:e})\n"
                );
                assert!(
                    a.iter().any(|v| v.abs() > 1e-6),
                    "skip model layer {layer} {name} is all zeros — the test proves nothing"
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 8. The 65 535-workgroup ceiling
// ---------------------------------------------------------------------------

/// WebGPU caps a dispatch at 65 535 workgroups **per dimension**. Nothing in
/// this repository came close before: the largest single-tensor dispatch is one
/// workgroup per 64 elements, i.e. 1024 for the biggest layer of
/// `Greyscale_Diffusion_L`. Multiply by a batch of 64 and it is 65 536 — one
/// over.
///
/// The failure mode is why this test exists rather than a comment. wgpu reports
/// the violation on the queue and the run CONTINUES: the observed symptom was
/// `loss 0.000000` scrolling past at batch 64, a step that computed nothing at
/// all. A silent wrong answer, at exactly the batch sizes the batching exists
/// to make possible.
///
/// The shape here is chosen to cross the ceiling cheaply: 64 elements per
/// sample is one workgroup per sample, so `batch` IS the workgroup count.
#[test]
fn a_dispatch_past_the_65535_workgroup_ceiling_is_still_correct() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        // 65 600 > 65 535, by enough that the second row of the grid is not
        // just a rounding artefact.
        let batch = 65_600usize;
        let dim = Dim3::new((8, 8, 1));
        let len = dim.length() as usize;
        assert_eq!(len, 64, "one workgroup per sample is what makes this cheap");

        let mut model = Model::new_training_with_optimizer(
            gpu.clone(),
            0.01,
            batch as u32,
            LossMethod::MeanSquared,
            OptimizerKind::Sgd,
        )
        .await;
        model
            .add_layer(LayerTypes::Activation(ActivationType::new(
                ActivationMethod::Linear,
                dim,
            )))
            .unwrap();
        model.build().unwrap();

        // A value that identifies its own sample, so a thread landing in the
        // wrong row of the grid is visible rather than plausible.
        let inputs: Vec<f32> = (0..batch)
            .flat_map(|sample| (0..len).map(move |i| (sample % 4096) as f32 + i as f32 / 128.0))
            .collect();
        write_input(&model, &inputs);
        run_forward(&model, batch as u32);
        let output = read_output(&model, batch * len);

        let differing = output
            .iter()
            .zip(&inputs)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "\n{differing} of {} elements wrong across a {batch}-workgroup dispatch \
             (the ceiling is 65 535 per dimension)\n",
            output.len()
        );

        // And the tail specifically: the last row of the 2-D grid is the part a
        // 1-D dispatch would have dropped entirely.
        let tail = &output[(batch - 1) * len..];
        assert!(
            tail.iter().any(|v| *v != 0.0),
            "the last sample is all zeros — the grid's second row never ran"
        );
    });
}

/// The split itself: below the ceiling it must stay 1-D (so nothing changes for
/// inference and small batches), above it, it must cover the count exactly.
#[test]
fn dispatch_grid_covers_the_count_without_exceeding_the_ceiling() {
    use crate::model::layer::dispatch_grid;
    for count in [0u32, 1, 64, 65_534, 65_535, 65_536, 131_070, 131_071, 4_194_304] {
        let (x, y) = dispatch_grid(count);
        assert!(x <= 65_535 && y <= 65_535, "{count} -> ({x}, {y}) exceeds the ceiling");
        assert!(
            x as u64 * y as u64 >= count as u64,
            "{count} -> ({x}, {y}) does not cover the count"
        );
        if count <= 65_535 {
            assert_eq!((x, y), (count, 1), "{count} should stay a 1-D dispatch");
        }
    }
}
