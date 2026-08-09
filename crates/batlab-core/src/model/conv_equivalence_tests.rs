//! File purpose: Proves the tiled `convolution` kernels are numerically
//! equivalent to the naive per-output-element implementation they replaced,
//! and measures the two against each other on the same buffers.
//!
//! Same standard of evidence as `group_norm_equivalence_tests`:
//!
//! 1. **Old vs new on the very same buffers.** `shader/legacy/*_naive.wgsl`
//!    are verbatim copies of the pre-optimisation shaders (commit `d73ed32`),
//!    kept as test fixtures. Each test builds a real model with the new
//!    shaders, runs it, then re-dispatches the legacy kernels over the *same*
//!    bind group and compares.
//! 2. **An f64 oracle** that knows neither implementation, to arbitrate
//!    wherever the two f32 results differ.
//! 3. **Finite differences** on the loss, which know nothing about the
//!    convolution at all.

use crate::gpu_context::GpuContext;
use crate::model::debug::read_back_f32;
use crate::model::layer::Layer;
use crate::model::layer_types::LayerType;
use crate::model::{
    ConvolutionType, Dim3, Infer, LayerTypes, LossMethod, Model, PaddingMode, Training,
};
use std::sync::Arc;

const LEGACY_FORWARD: &str = include_str!("shader/legacy/convolution_naive.wgsl");
const LEGACY_BACKWARD: &str = include_str!("shader/legacy/back_convolution_naive.wgsl");

// ---------------------------------------------------------------------------
// Shapes under test
// ---------------------------------------------------------------------------

/// `(dim_input, nb_kernel, dim_kernel, stride, mode)`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    input: (u32, u32, u32),
    nb_kernel: u32,
    kernel: (u32, u32, u32),
    stride: u32,
    mode: PaddingMode,
    label: &'static str,
}

/// The four convolutions of `Greyscale_Diffusion`, plus cases that stress the
/// tiling in ways the model does not: `Valid` padding, a stride that does not
/// divide the input, a 1x1 kernel, a 5x5 kernel, and extents smaller than a
/// tile.
const SHAPES: &[Shape] = &[
    Shape {
        input: (32, 32, 3),
        nb_kernel: 16,
        kernel: (3, 3, 3),
        stride: 1,
        mode: PaddingMode::Same,
        label: "conv1 32x32x3 ->16 s1",
    },
    Shape {
        input: (32, 32, 16),
        nb_kernel: 32,
        kernel: (3, 3, 16),
        stride: 2,
        mode: PaddingMode::Same,
        label: "conv2 32x32x16->32 s2",
    },
    Shape {
        input: (16, 16, 32),
        nb_kernel: 32,
        kernel: (3, 3, 32),
        stride: 1,
        mode: PaddingMode::Same,
        label: "conv3 16x16x32->32 s1",
    },
    Shape {
        input: (32, 32, 32),
        nb_kernel: 1,
        kernel: (3, 3, 32),
        stride: 1,
        mode: PaddingMode::Same,
        label: "conv4 32x32x32-> 1 s1",
    },
    Shape {
        input: (9, 7, 5),
        nb_kernel: 4,
        kernel: (3, 3, 5),
        stride: 1,
        mode: PaddingMode::Valid,
        label: "valid 9x7x5 -> 4 s1",
    },
    Shape {
        input: (11, 11, 6),
        nb_kernel: 5,
        kernel: (3, 3, 6),
        stride: 2,
        mode: PaddingMode::Same,
        label: "odd 11x11x6 -> 5 s2",
    },
    Shape {
        input: (8, 8, 12),
        nb_kernel: 7,
        kernel: (1, 1, 12),
        stride: 1,
        mode: PaddingMode::Same,
        label: "1x1 8x8x12 -> 7 s1",
    },
    Shape {
        input: (10, 10, 4),
        nb_kernel: 3,
        kernel: (5, 5, 4),
        stride: 1,
        mode: PaddingMode::Same,
        label: "5x5 10x10x4 -> 3 s1",
    },
    Shape {
        input: (3, 5, 2),
        nb_kernel: 2,
        kernel: (3, 3, 2),
        stride: 1,
        mode: PaddingMode::Same,
        label: "tiny 3x5x2 -> 2 s1",
    },
];

impl Shape {
    fn dim_input(&self) -> Dim3 {
        Dim3::new(self.input)
    }
    fn dim_kernel(&self) -> Dim3 {
        Dim3::new(self.kernel)
    }
    fn conv_type(&self) -> ConvolutionType {
        ConvolutionType::new(
            self.dim_input(),
            self.nb_kernel,
            self.dim_kernel(),
            self.stride,
            self.mode,
        )
    }
    fn dim_output(&self) -> Dim3 {
        let mut ty = self.conv_type();
        ty.set_dim_output().expect("invalid shape in test table")
    }
    fn weight_len(&self) -> usize {
        (self.dim_kernel().length() * self.nb_kernel) as usize
    }
}

// ---------------------------------------------------------------------------
// Deterministic, asymmetric test data
// ---------------------------------------------------------------------------

/// No symmetry that could mask an indexing mistake, and a per-channel offset so
/// channels are not interchangeable.
fn synthetic_input(dim: Dim3) -> Vec<f32> {
    let z = dim.z as usize;
    (0..dim.length() as usize)
        .map(|i| {
            let channel = i % z;
            let spatial = i / z;
            0.7 * ((i as f32) * 0.37).sin() + 0.13 * channel as f32 - 0.004 * spatial as f32
        })
        .collect()
}

/// Weights that differ in every tap, every channel and every kernel, so a
/// transposed or truncated weight index cannot produce the right answer.
fn synthetic_weights(shape: &Shape) -> Vec<f32> {
    (0..shape.weight_len())
        .map(|i| 0.21 * ((i as f32) * 0.73 + 0.4).cos() - 0.03 + 0.002 * (i % 7) as f32)
        .collect()
}

fn synthetic_bias(nb_kernel: u32) -> Vec<f32> {
    (0..nb_kernel as usize)
        .map(|k| 0.05 - 0.017 * k as f32)
        .collect()
}

fn synthetic_target(dim: Dim3) -> Vec<f32> {
    (0..dim.length() as usize)
        .map(|i| 0.3 * ((i as f32) * 0.11).cos())
        .collect()
}

// ---------------------------------------------------------------------------
// GPU plumbing
// ---------------------------------------------------------------------------

fn write_weights_and_bias(gpu: &GpuContext, layer: &Layer, weights: &[f32], bias: &[f32]) {
    gpu.queue.write_buffer(
        layer.buffers.forward[1].as_ref(),
        0,
        bytemuck::cast_slice(weights),
    );
    gpu.queue.write_buffer(
        layer.buffers.forward[2].as_ref(),
        0,
        bytemuck::cast_slice(bias),
    );
}

fn read_layer_buffer(gpu: &GpuContext, buf: &wgpu::Buffer) -> Vec<f32> {
    read_back_f32(gpu, buf, buf.size()).expect("buffer readback failed")
}

fn legacy_pipeline(
    gpu: &GpuContext,
    source: &str,
    entry_point: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy_conv"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    });
    let pl = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("legacy_conv_pl"),
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
    gpu.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("legacy_conv_pipeline"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: Default::default(),
        })
}

fn dispatch_legacy(
    gpu: &GpuContext,
    source: &str,
    entry_point: &str,
    bgl: &wgpu::BindGroupLayout,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let pipeline = legacy_pipeline(gpu, source, entry_point, bgl);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
}

/// Workgroup counts the *legacy* kernels expect: one thread per element,
/// `@workgroup_size(64)`. Kept explicit so the legacy dispatch never silently
/// inherits the new kernels' tiled counts.
fn legacy_forward_workgroups(shape: &Shape) -> u32 {
    shape.dim_output().length().div_ceil(64)
}

fn legacy_backward_workgroups(shape: &Shape) -> [u32; 3] {
    [
        shape.dim_input().length().div_ceil(64),
        (shape.dim_kernel().length() * shape.nb_kernel).div_ceil(64),
        shape.nb_kernel.div_ceil(64),
    ]
}

/// One-convolution inference model with the synthetic weights/bias installed.
async fn infer_conv(gpu: Arc<GpuContext>, shape: &Shape) -> Model<Infer> {
    let mut model: Model<Infer> = Model::new(gpu.clone()).await;
    model
        .add_layer(LayerTypes::Convolution(shape.conv_type()))
        .unwrap();
    model.build_model().unwrap();
    write_weights_and_bias(
        gpu.as_ref(),
        model.layers.first().unwrap(),
        &synthetic_weights(shape),
        &synthetic_bias(shape.nb_kernel),
    );
    model
}

/// One-convolution training model, one step run at lr = 0 so the gradients can
/// be read back without the optimizer having perturbed anything.
async fn trained_conv(
    gpu: Arc<GpuContext>,
    shape: &Shape,
) -> (Model<Training>, Vec<f32>, Vec<f32>) {
    let mut model =
        Model::<Training>::new_training(gpu.clone(), 0.0, 1, LossMethod::MeanSquared).await;
    model
        .add_layer(LayerTypes::Convolution(shape.conv_type()))
        .unwrap();
    model.build().unwrap();
    write_weights_and_bias(
        gpu.as_ref(),
        model.layers.first().unwrap(),
        &synthetic_weights(shape),
        &synthetic_bias(shape.nb_kernel),
    );

    let input = synthetic_input(shape.dim_input());
    let target = synthetic_target(shape.dim_output());
    model.train_step(&input, &target);
    (model, input, target)
}

// ---------------------------------------------------------------------------
// f64 oracle
// ---------------------------------------------------------------------------

/// Convolution forward + MSE loss + backward, in f64.
///
/// Deliberately written as a **scatter** over the forward tap map
/// `(oy,ox,ky,kx) -> (oy*s+ky-pad_y, ox*s+kx-pad_x)`, i.e. the definition in
/// `AUDIT_TRAINING.md` finding #1. `conv_back_input` instead *inverts* that map
/// (`oy = (iy + pad_y - ky) / s`, with a divisibility test), so the oracle
/// validates the inversion rather than restating it.
struct Reference {
    output: Vec<f64>,
    grad_input: Vec<f64>,
    grad_weights: Vec<f64>,
    grad_bias: Vec<f64>,
}

fn reference_conv(
    shape: &Shape,
    input: &[f32],
    target: &[f32],
    weights: &[f32],
    bias: &[f32],
) -> Reference {
    let (ih, iw, ic) = shape.input;
    let (kh, kw, _) = shape.kernel;
    let dim_out = shape.dim_output();
    let (oh, ow, k_count) = (dim_out.x, dim_out.y, dim_out.z);
    let s = shape.stride;
    let (pad_y, pad_x) = match shape.mode {
        PaddingMode::Same => ((kh / 2) as i32, (kw / 2) as i32),
        PaddingMode::Valid => (0, 0),
    };

    let in_idx = |iy: u32, ix: u32, iz: u32| (iy * iw * ic + ix * ic + iz) as usize;
    let w_idx =
        |k: u32, ky: u32, kx: u32, kz: u32| (k * kh * kw * ic + ky * kw * ic + kx * ic + kz) as usize;
    let out_idx = |oy: u32, ox: u32, k: u32| (oy * ow * k_count + ox * k_count + k) as usize;

    // --- forward
    let mut output = vec![0.0f64; (oh * ow * k_count) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            for k in 0..k_count {
                let mut sum = bias[k as usize] as f64;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sy = (oy * s) as i32 + ky as i32 - pad_y;
                        let sx = (ox * s) as i32 + kx as i32 - pad_x;
                        if sy < 0 || sy >= ih as i32 || sx < 0 || sx >= iw as i32 {
                            continue; // zero padding
                        }
                        for kz in 0..ic {
                            sum += input[in_idx(sy as u32, sx as u32, kz)] as f64
                                * weights[w_idx(k, ky, kx, kz)] as f64;
                        }
                    }
                }
                output[out_idx(oy, ox, k)] = sum;
            }
        }
    }

    // --- MSE loss gradient, matching shader/loss.wgsl
    let n = output.len();
    let grad_output: Vec<f64> = (0..n)
        .map(|i| 2.0 * (output[i] - target[i] as f64) / n as f64)
        .collect();

    // --- backward, scattered along the same tap map
    let mut grad_input = vec![0.0f64; (ih * iw * ic) as usize];
    let mut grad_weights = vec![0.0f64; shape.weight_len()];
    let mut grad_bias = vec![0.0f64; k_count as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            for k in 0..k_count {
                let go = grad_output[out_idx(oy, ox, k)];
                grad_bias[k as usize] += go;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sy = (oy * s) as i32 + ky as i32 - pad_y;
                        let sx = (ox * s) as i32 + kx as i32 - pad_x;
                        if sy < 0 || sy >= ih as i32 || sx < 0 || sx >= iw as i32 {
                            continue;
                        }
                        for kz in 0..ic {
                            grad_input[in_idx(sy as u32, sx as u32, kz)] +=
                                go * weights[w_idx(k, ky, kx, kz)] as f64;
                            grad_weights[w_idx(k, ky, kx, kz)] +=
                                go * input[in_idx(sy as u32, sx as u32, kz)] as f64;
                        }
                    }
                }
            }
        }
    }

    Reference {
        output,
        grad_input,
        grad_weights,
        grad_bias,
    }
}

/// Worst relative difference between two f32 vectors, with a floor on the
/// denominator so near-zero entries do not blow the ratio up.
fn worst_relative_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    let scale = a
        .iter()
        .chain(b.iter())
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    a.iter()
        .zip(b)
        .fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs() / scale))
}

/// Worst difference between an f32 GPU result and the f64 oracle, relative to
/// the oracle's own scale.
fn worst_relative_to_reference(got: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(got.len(), reference.len());
    let scale = reference
        .iter()
        .fold(0.0f64, |a, v| a.max(v.abs()))
        .max(1e-12);
    got.iter()
        .zip(reference)
        .fold(0.0f64, |acc, (g, r)| acc.max((*g as f64 - r).abs() / scale))
}

// ---------------------------------------------------------------------------
// The lane rule is duplicated in Rust and WGSL — pin them together
// ---------------------------------------------------------------------------

/// The lane counts the shader reads out of the uniform and the slot counts the
/// Rust side sizes the dispatch with are two views of one decision. If they
/// drift, the dispatch silently under-covers the weight buffer and some
/// gradients are never written — a bug no shape-independent assertion would
/// notice. This checks them against each other on every shape, and pins the
/// uniform word to the offset the legacy fixtures expect to be padding.
///
/// The word carries **two** counts since the bias reduction stopped inheriting
/// the weights': `grad_weights` in the low half, `grad_bias` in the high half.
/// Each half is checked against the dispatch it sizes.
#[test]
fn conv_reduction_lanes_agrees_with_dispatch() {
    for shape in SHAPES {
        let mut ty = shape.conv_type();
        ty.set_dim_output().unwrap();

        // Word 3 of the uniform is the packed lane pair (bytes 12..16).
        let bytes = ty.get_spec_uniform_bytes();
        let packed = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let weight_lanes = packed & 0xffff;
        let bias_lanes = packed >> 16;

        // The packing is the shader's decoding, spelled out here so a change to
        // either half is caught by this test rather than by a silent
        // mis-dispatch.
        assert_eq!(
            packed,
            ConvolutionType::pack_lanes(weight_lanes, bias_lanes),
            "\n{}: the packed lane word does not round-trip\n",
            shape.label
        );

        let dim_out = shape.dim_output();
        let positions = dim_out.x * dim_out.y;
        let weight_len = shape.weight_len() as u32;
        let counts = ty.get_back_workgroup_counts(1);

        for (name, lanes, sums, dispatched) in [
            ("grad_weights", weight_lanes, weight_len, counts[1]),
            ("grad_bias", bias_lanes, shape.nb_kernel, counts[2]),
        ] {
            assert!(
                lanes.is_power_of_two() && (1..=64).contains(&lanes),
                "\n{} {name}: lanes = {lanes}, must be a power of two in 1..=64 \
                 (the tree reduction halves it down to 1)\n",
                shape.label
            );
            assert!(
                lanes == 1 || positions / lanes >= 16,
                "\n{} {name}: {lanes} lanes over {positions} positions leaves \
                 {} per lane — the reduction would cost more than the sum\n",
                shape.label,
                positions / lanes
            );

            // Every sum must be covered by the dispatch, and covered once.
            let slots = 64 / lanes;
            assert_eq!(
                dispatched,
                sums.div_ceil(slots),
                "\n{} {name}: dispatch is {dispatched} workgroups x {slots} \
                 slots for {sums} sums\n",
                shape.label
            );
            assert!(
                dispatched * slots >= sums,
                "\n{} {name}: dispatch leaves {} sums unwritten\n",
                shape.label,
                sums - dispatched * slots
            );
        }

        // The point of the split: the bias reduction has orders of magnitude
        // fewer sums than the weights one, so it must never be left with fewer
        // lanes. Inheriting the weights' count is what put a 48-bias reduction
        // on a single workgroup.
        assert!(
            bias_lanes >= weight_lanes,
            "\n{}: grad_bias got {bias_lanes} lanes for {} sums while \
             grad_weights got {weight_lanes} for {weight_len} — the pass with \
             fewer sums must not be the one with fewer threads\n",
            shape.label,
            shape.nb_kernel
        );
    }
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

#[test]
fn forward_matches_naive_implementation() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);

        for shape in SHAPES {
            let mut model = infer_conv(gpu.clone(), shape).await;
            let input = synthetic_input(shape.dim_input());
            let optimised = model.predict(&input);

            let layer = model.layers.first().unwrap();
            let pipeline = &layer.pipeline.forward[0].0;
            dispatch_legacy(
                gpu.as_ref(),
                LEGACY_FORWARD,
                "main",
                &pipeline.get_bind_group_layout(0),
                layer.bind_group.forward.as_ref().unwrap(),
                legacy_forward_workgroups(shape),
            );
            let naive = read_layer_buffer(gpu.as_ref(), layer.buffers.forward[4].as_ref());

            // The forward optimisation only hoists integer address arithmetic
            // out of the loops: every tap is visited in the same order, from
            // the same bias, so the agreement must be EXACT, not merely within
            // a tolerance. Asserting bitwise equality is what makes this test
            // able to catch a reordering — a 1e-5 threshold would not.
            let differing = optimised
                .iter()
                .zip(&naive)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing,
                0,
                "\n{} forward is not bit-identical to the naive implementation \
                 ({differing}/{} elements differ).\n\
                 The optimisation only reassociates *integer* index arithmetic, \
                 so any float difference means the tap order changed.\n\
                 worst relative difference: {:e}\n\
                 optimised[..8]: {:?}\n\
                 naive[..8]:     {:?}\n",
                shape.label,
                optimised.len(),
                worst_relative_diff(&optimised, &naive),
                &optimised[..optimised.len().min(8)],
                &naive[..naive.len().min(8)],
            );

            // And against the f64 oracle, which knows neither implementation.
            let reference = reference_conv(
                shape,
                &input,
                &synthetic_target(shape.dim_output()),
                &synthetic_weights(shape),
                &synthetic_bias(shape.nb_kernel),
            );
            let err = worst_relative_to_reference(&optimised, &reference.output);
            assert!(
                err < 1e-5,
                "\n{} forward is off the f64 reference by {err:e}\n",
                shape.label
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

#[test]
fn backward_matches_naive_implementation() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);

        for shape in SHAPES {
            let (model, input, target) = trained_conv(gpu.clone(), shape).await;
            let layer = model.layers.first().unwrap();
            let bwd = layer.buffers.backward.as_ref().unwrap();

            let new_grad_input = read_layer_buffer(gpu.as_ref(), bwd[4].as_ref());
            let new_grad_weights = read_layer_buffer(gpu.as_ref(), bwd[5].as_ref());
            let new_grad_bias = read_layer_buffer(gpu.as_ref(), bwd[6].as_ref());

            // grad_weights / grad_bias accumulate with `+=` across a batch, so
            // they must start from zero for the legacy re-run to be comparable.
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.clear_buffer(bwd[5].as_ref(), 0, None);
            encoder.clear_buffer(bwd[6].as_ref(), 0, None);
            gpu.queue.submit([encoder.finish()]);

            let bgl = layer.pipeline.backward[0].0.get_bind_group_layout(0);
            let bind_group = layer.bind_group.backward.as_ref().unwrap();
            let legacy_wg = legacy_backward_workgroups(shape);
            for (entry_point, workgroups) in [
                ("conv_back_input", legacy_wg[0]),
                ("conv_back_weights", legacy_wg[1]),
                ("conv_back_bias", legacy_wg[2]),
            ] {
                dispatch_legacy(
                    gpu.as_ref(),
                    LEGACY_BACKWARD,
                    entry_point,
                    &bgl,
                    bind_group,
                    workgroups,
                );
            }

            let old_grad_input = read_layer_buffer(gpu.as_ref(), bwd[4].as_ref());
            let old_grad_weights = read_layer_buffer(gpu.as_ref(), bwd[5].as_ref());
            let old_grad_bias = read_layer_buffer(gpu.as_ref(), bwd[6].as_ref());

            let reference = reference_conv(
                shape,
                &input,
                &target,
                &synthetic_weights(shape),
                &synthetic_bias(shape.nb_kernel),
            );

            for (name, new, old, oracle) in [
                (
                    "grad_input",
                    &new_grad_input,
                    &old_grad_input,
                    &reference.grad_input,
                ),
                (
                    "grad_weights",
                    &new_grad_weights,
                    &old_grad_weights,
                    &reference.grad_weights,
                ),
                (
                    "grad_bias",
                    &new_grad_bias,
                    &old_grad_bias,
                    &reference.grad_bias,
                ),
            ] {
                let cross = worst_relative_diff(new, old);
                let new_err = worst_relative_to_reference(new, oracle);
                let old_err = worst_relative_to_reference(old, oracle);

                // Visible with --nocapture: the actual magnitudes behind the
                // thresholds, and which of the two f32 results the f64 oracle
                // says is closer to the truth.
                println!(
                    "{:<24} {:<13} new^old {:>9.2e} | new^f64 {:>9.2e} | old^f64 {:>9.2e}{}",
                    shape.label,
                    name,
                    cross,
                    new_err,
                    old_err,
                    if new_err < old_err { "  (new closer)" } else { "" },
                );

                assert!(
                    cross < 1e-4,
                    "\n{} backward {name} disagrees with the naive implementation.\n\
                     worst relative difference: {cross:e}\n\
                     new[..8]: {:?}\n\
                     old[..8]: {:?}\n",
                    shape.label,
                    &new[..new.len().min(8)],
                    &old[..old.len().min(8)],
                );
                assert!(
                    new_err < 1e-4,
                    "\n{} backward {name} is off the f64 reference by {new_err:e} \
                     (naive: {old_err:e}).\n",
                    shape.label
                );
                // A tree reduction over f32 is not less accurate than the
                // sequential sum it replaces; the slack absorbs the cases where
                // both already sit at the rounding floor.
                assert!(
                    new_err <= old_err.max(1e-7) * 1.5,
                    "\n{} backward {name} is LESS accurate than the implementation \
                     it replaces.\n\
                     new vs f64 reference: {new_err:e}\n\
                     old vs f64 reference: {old_err:e}\n",
                    shape.label
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Isolated-layer profile of the CURRENT implementation
// ---------------------------------------------------------------------------

/// Encode `iters` back-to-back dispatches, submit, block until drained.
fn time_dispatches(
    gpu: &GpuContext,
    passes: &[(&wgpu::ComputePipeline, u32)],
    bind_group: &wgpu::BindGroup,
    iters: u32,
) -> std::time::Duration {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    for _ in 0..iters {
        for (pipeline, workgroups) in passes {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(*workgroups, 1, 1);
        }
    }
    let start = std::time::Instant::now();
    gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU poll failed");
    start.elapsed()
}

/// Where does the convolution time actually go? Times each pass of each real
/// layer separately, on the current implementation. Ignored by default:
///
///   cargo test --release -p batlab_core --lib profile_convolution \
///       -- --ignored --nocapture
///
/// Every pipeline is warmed before any timing starts (on Metal the first
/// dispatch of a pipeline pays its compilation, which would otherwise be
/// charged to whichever layer happens to be measured first), and the whole
/// sweep is repeated so drift from the GPU being shared is visible rather than
/// silently folded into a single number.
#[test]
#[ignore]
fn profile_convolution() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let iters = 200u32;
        let rounds = 3;

        // Build every layer up front, and warm every pipeline.
        let mut built = Vec::new();
        for shape in SHAPES.iter().take(4) {
            let (model, _, _) = trained_conv(gpu.clone(), shape).await;
            built.push((shape, model));
        }
        for (_, model) in &built {
            let layer = model.layers.first().unwrap();
            let fwd_bg = layer.bind_group.forward.as_ref().unwrap();
            let back_bg = layer.bind_group.backward.as_ref().unwrap();
            for _ in 0..3 {
                time_dispatches(
                    gpu.as_ref(),
                    &[(
                        &layer.pipeline.forward[0].0,
                        layer.ty.get_forward_workgroup_count(layer.batch),
                    )],
                    fwd_bg,
                    32,
                );
                for (pipeline, wg) in layer.pipeline.backward.iter() {
                    time_dispatches(gpu.as_ref(), &[(pipeline, *wg)], back_bg, 32);
                }
            }
        }

        // Per-compute-pass overhead floor: the smallest possible dispatch, so
        // whatever this costs is encoder/pass bookkeeping, not arithmetic.
        let tiny = &SHAPES[8];
        let tiny_model = infer_conv(gpu.clone(), tiny).await;
        let tiny_layer = tiny_model.layers.first().unwrap();
        let tiny_bg = tiny_layer.bind_group.forward.as_ref().unwrap();
        let tiny_pipe = &tiny_layer.pipeline.forward[0].0;
        time_dispatches(gpu.as_ref(), &[(tiny_pipe, 1)], tiny_bg, 64);
        let floor = time_dispatches(gpu.as_ref(), &[(tiny_pipe, 1)], tiny_bg, iters);
        println!(
            "\nper-compute-pass floor (1 workgroup, 30 outputs): {:.4} ms",
            floor.as_secs_f64() * 1e3 / iters as f64
        );

        for round in 1..=rounds {
            println!(
                "\n--- round {round} ---\n{:<24} {:>10} {:>11} {:>10} {:>10} {:>10}",
                "layer", "forward", "back_input", "back_wts", "back_bias", "total"
            );
            let mut grand = 0.0f64;
            for (shape, model) in &built {
                let layer = model.layers.first().unwrap();
                let fwd_bg = layer.bind_group.forward.as_ref().unwrap();
                let fwd = time_dispatches(
                    gpu.as_ref(),
                    &[(
                        &layer.pipeline.forward[0].0,
                        layer.ty.get_forward_workgroup_count(layer.batch),
                    )],
                    fwd_bg,
                    iters,
                );
                let back_bg = layer.bind_group.backward.as_ref().unwrap();
                let backs: Vec<_> = layer
                    .pipeline
                    .backward
                    .iter()
                    .map(|(pipeline, wg)| {
                        time_dispatches(gpu.as_ref(), &[(pipeline, *wg)], back_bg, iters)
                    })
                    .collect();

                let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3 / iters as f64;
                let total = ms(fwd) + backs.iter().map(|d| ms(*d)).sum::<f64>();
                grand += total;
                println!(
                    "{:<24} {:>10.4} {:>11.4} {:>10.4} {:>10.4} {:>10.4}",
                    shape.label,
                    ms(fwd),
                    ms(backs[0]),
                    ms(backs[1]),
                    ms(backs[2]),
                    total
                );
            }
            println!("{:<24} {:>56.4}", "all four, per sample", grand);
        }
        println!();
    });
}

// ---------------------------------------------------------------------------
// Paired old-vs-new benchmark
// ---------------------------------------------------------------------------

/// Times the legacy kernels against the current ones, **interleaved in the same
/// process and on the same buffers**:
///
///   cargo test --release -p batlab_core --lib bench_convolution_isolated \
///       -- --ignored --nocapture
///
/// The GPU is shared with other work (a second agent trains a larger model on
/// it), so absolute timings taken at different moments are not comparable.
/// Every pass is therefore measured old/new/old/new/... within one run, and
/// reported by the **minimum** round of each side.
///
/// The minimum, not the median: contention can only ever *add* time, so the
/// fastest round is the closest estimate of what the kernel costs on its own,
/// and taking it on both sides keeps the comparison fair. A median is not
/// robust enough here — a neighbour's burst routinely outlasts a whole round,
/// and it distorts the two sides unequally, because the optimised kernels are
/// small enough (~0.03 ms) for one burst to triple them while the naive ones
/// (~0.35 ms) barely move. The full min-max spread is printed for both sides
/// so that distortion stays visible instead of being averaged away.
#[test]
#[ignore]
fn bench_convolution_isolated() {
    fn stats(v: &[f64]) -> (f64, f64) {
        (
            v.iter().cloned().fold(f64::INFINITY, f64::min),
            v.iter().cloned().fold(0.0f64, f64::max),
        )
    }

    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let iters = 200u32;
        let rounds = 9;

        println!(
            "\n{:<24} {:>12} {:>10} {:>10} {:>8}   {:<17} {:<17}",
            "layer", "pass", "naive", "new", "speedup", "naive spread", "new spread"
        );

        let mut total_old = 0.0f64;
        let mut total_new = 0.0f64;

        for shape in SHAPES.iter().take(4) {
            let (model, _, _) = trained_conv(gpu.clone(), shape).await;
            let layer = model.layers.first().unwrap();

            let fwd_bg = layer.bind_group.forward.as_ref().unwrap();
            let fwd_new = &layer.pipeline.forward[0].0;
            let fwd_bgl = fwd_new.get_bind_group_layout(0);
            let fwd_old = legacy_pipeline(gpu.as_ref(), LEGACY_FORWARD, "main", &fwd_bgl);

            let back_bg = layer.bind_group.backward.as_ref().unwrap();
            let back_bgl = layer.pipeline.backward[0].0.get_bind_group_layout(0);
            let back_old: Vec<wgpu::ComputePipeline> =
                ["conv_back_input", "conv_back_weights", "conv_back_bias"]
                    .iter()
                    .map(|ep| legacy_pipeline(gpu.as_ref(), LEGACY_BACKWARD, ep, &back_bgl))
                    .collect();
            let legacy_wg = legacy_backward_workgroups(shape);

            // (name, bind_group, old pipeline + wg, new pipeline + wg)
            let cases: Vec<(&str, &wgpu::BindGroup, (&wgpu::ComputePipeline, u32), (&wgpu::ComputePipeline, u32))> = vec![
                (
                    "forward",
                    fwd_bg,
                    (&fwd_old, legacy_forward_workgroups(shape)),
                    (fwd_new, layer.ty.get_forward_workgroup_count(layer.batch)),
                ),
                (
                    "back_input",
                    back_bg,
                    (&back_old[0], legacy_wg[0]),
                    (&layer.pipeline.backward[0].0, layer.pipeline.backward[0].1),
                ),
                (
                    "back_weights",
                    back_bg,
                    (&back_old[1], legacy_wg[1]),
                    (&layer.pipeline.backward[1].0, layer.pipeline.backward[1].1),
                ),
                (
                    "back_bias",
                    back_bg,
                    (&back_old[2], legacy_wg[2]),
                    (&layer.pipeline.backward[2].0, layer.pipeline.backward[2].1),
                ),
            ];

            for (name, bg, old, new) in cases {
                // Warm both pipelines before either is timed.
                for _ in 0..3 {
                    time_dispatches(gpu.as_ref(), &[old], bg, 32);
                    time_dispatches(gpu.as_ref(), &[new], bg, 32);
                }
                let mut olds = Vec::new();
                let mut news = Vec::new();
                for _ in 0..rounds {
                    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3 / iters as f64;
                    olds.push(ms(time_dispatches(gpu.as_ref(), &[old], bg, iters)));
                    news.push(ms(time_dispatches(gpu.as_ref(), &[new], bg, iters)));
                }
                let (old_ms, old_hi) = stats(&olds);
                let (new_ms, new_hi) = stats(&news);
                total_old += old_ms;
                total_new += new_ms;
                println!(
                    "{:<24} {:>12} {:>10.4} {:>10.4} {:>7.2}x   {:<17} {:<17}",
                    shape.label,
                    name,
                    old_ms,
                    new_ms,
                    old_ms / new_ms,
                    format!("{old_ms:.4}-{old_hi:.4}"),
                    format!("{new_ms:.4}-{new_hi:.4}"),
                );
            }
        }

        println!(
            "\nAll four convolutions, forward + backward, per sample: \
             {total_old:.4} ms -> {total_new:.4} ms ({:.2}x)\n",
            total_old / total_new
        );
    });
}

/// Sweeps the `grad_weights` reduction split over every legal lane count, per
/// layer, so `ConvolutionType::reduction_lanes` is calibrated on measurement
/// rather than on intuition:
///
///   cargo test --release -p batlab_core --lib bench_conv_reduction_lanes \
///       -- --ignored --nocapture
///
/// The lane count lives in the uniform, so a sweep only has to rewrite that one
/// word and re-dispatch with the matching workgroup count — the layer, its
/// buffers and its pipeline are untouched between points, which is what makes
/// the comparison apples-to-apples. `*` marks the value the shipped rule picks.
#[test]
#[ignore]
fn bench_conv_reduction_lanes() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let iters = 200u32;
        let rounds = 7;

        for shape in SHAPES.iter().take(4) {
            let (model, _, _) = trained_conv(gpu.clone(), shape).await;
            let layer = model.layers.first().unwrap();
            let back_bg = layer.bind_group.backward.as_ref().unwrap();
            let pipeline = &layer.pipeline.backward[1].0;
            let specs = layer.buffers.forward[3].as_ref();

            let dim_out = shape.dim_output();
            let positions = dim_out.x * dim_out.y;
            let weight_len = shape.weight_len() as u32;
            let chosen = ConvolutionType::reduction_lanes(weight_len, positions);

            let mut base = shape.conv_type();
            base.set_dim_output().unwrap();
            let uniform = base.get_spec_uniform_bytes();

            println!(
                "\n{}  ({weight_len} weights, {positions} positions)",
                shape.label
            );
            for lanes in [1u32, 2, 4, 8, 16, 32, 64] {
                if lanes > 1 && positions / lanes < 16 {
                    continue;
                }
                let mut bytes = uniform.clone();
                bytes[12..16].copy_from_slice(&lanes.to_le_bytes());
                gpu.queue.write_buffer(specs, 0, &bytes);

                let slots = 64 / lanes;
                let wg = weight_len.div_ceil(slots);
                let case = [(pipeline, wg)];

                for _ in 0..2 {
                    time_dispatches(gpu.as_ref(), &case, back_bg, 32);
                }
                let mut times = Vec::new();
                for _ in 0..rounds {
                    times.push(
                        time_dispatches(gpu.as_ref(), &case, back_bg, iters).as_secs_f64() * 1e3
                            / iters as f64,
                    );
                }
                let lo = times.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = times.iter().cloned().fold(0.0f64, f64::max);
                println!(
                    "  lanes {lanes:>3} ({wg:>5} wg) : {lo:.4} ms   (spread {lo:.4}-{hi:.4}){}",
                    if lanes == chosen { "  *" } else { "" }
                );
            }
            // Leave the uniform as the model built it.
            gpu.queue.write_buffer(specs, 0, &uniform);
        }
        println!();
    });
}

// ---------------------------------------------------------------------------
// Fixture health
// ---------------------------------------------------------------------------

/// The legacy fixtures must stay loadable and dispatchable: if a binding or the
/// uniform layout ever drifts, this fails before any equivalence test does.
#[test]
fn legacy_fixtures_still_match_the_current_bind_group_layout() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let shape = &SHAPES[0];
        let model = infer_conv(gpu.clone(), shape).await;
        let layer = model.layers.first().unwrap();
        let bgl = layer.pipeline.forward[0].0.get_bind_group_layout(0);
        // Creating the pipeline is the check: a mismatched binding fails here.
        let _ = legacy_pipeline(gpu.as_ref(), LEGACY_FORWARD, "main", &bgl);

        let (model, _, _) = trained_conv(gpu.clone(), shape).await;
        let layer = model.layers.first().unwrap();
        let back_bgl = layer.pipeline.backward[0].0.get_bind_group_layout(0);
        for ep in ["conv_back_input", "conv_back_weights", "conv_back_bias"] {
            let _ = legacy_pipeline(gpu.as_ref(), LEGACY_BACKWARD, ep, &back_bgl);
        }
    });
}
