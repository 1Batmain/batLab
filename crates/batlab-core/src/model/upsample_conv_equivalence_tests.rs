//! File purpose: Proves that inverting the tap map in `upsample_conv_back_input`
//! computes the same gradient as the output-map scan it replaced.
//!
//! The kernel used to walk the *entire* output map for every input element and
//! `continue` on the `OH·OW/scale² − 1` taps out of `OH·OW/scale²` that could
//! not land in that element's upsampling window. It now enumerates the window
//! directly. Nothing about the mathematics changed; only the order and the
//! number of iterations did — so this file has to show that the value is the
//! same, and that the new summation order is no less accurate.
//!
//! Same standard of evidence as `conv_equivalence_tests`:
//!
//! 1. **Old vs new on the very same buffers.**
//!    `shader/legacy/back_upsample_conv_naive.wgsl` is a verbatim copy of the
//!    pre-optimisation shader. Each test builds a real model with the new
//!    shader, runs a step, then re-dispatches the legacy kernel over the *same*
//!    bind group and compares.
//! 2. **An f64 oracle** that knows neither implementation. It is written as a
//!    **scatter** along the forward tap map — `(oy,ox,ky,kx) → up_y = oy+ky−pad`,
//!    then `iy = up_y/scale` — which is the definition in `upsample_conv.wgsl`.
//!    The new kernel *inverts* that map (`oy = up_y + pad − ky`), so the oracle
//!    validates the inversion rather than restating it. This is exactly the
//!    argument of `PERF_CONVOLUTION.md` §4.2, and it is the reason a hand-written
//!    reference is worth more here than a second copy of the kernel.
//! 3. **Finite differences** on the loss, which know nothing about upsampling at
//!    all.

use crate::gpu_context::GpuContext;
use crate::model::debug::read_back_f32;
use crate::model::layer::Layer;
use crate::model::layer_types::LayerType;
use crate::model::{
    Dim3, LayerTypes, LossMethod, Model, PaddingMode, Training, UpsampleConvType,
};
use std::sync::Arc;

const LEGACY_BACKWARD: &str = include_str!("shader/legacy/back_upsample_conv_naive.wgsl");

// ---------------------------------------------------------------------------
// Shapes under test
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Shape {
    input: (u32, u32, u32),
    scale: u32,
    nb_kernel: u32,
    kernel: (u32, u32, u32),
    mode: PaddingMode,
    label: &'static str,
}

/// The geometry of `Color_Diffusion_XL`'s two UpsampleConv layers (at reduced
/// channel counts, so the f64 oracle stays a test and not a benchmark), plus the
/// cases the model does not exercise and the rewrite could break: `Valid`
/// padding, a scale that is not 2, a scale of **1** (the window collapses to a
/// single `up_y`, which is where an off-by-one in the loop bounds would hide), a
/// 1×1 kernel (`pad = 0`), a 5×5 kernel (`pad = 2`, so `oy` goes out of range at
/// both ends), and a non-square input smaller than a workgroup.
const SHAPES: &[Shape] = &[
    Shape {
        input: (16, 16, 8),
        scale: 2,
        nb_kernel: 8,
        kernel: (3, 3, 8),
        mode: PaddingMode::Same,
        label: "XL L22-shaped 16x16x8 x2 -> 8",
    },
    Shape {
        input: (8, 8, 12),
        scale: 2,
        nb_kernel: 6,
        kernel: (3, 3, 12),
        mode: PaddingMode::Same,
        label: "XL L17-shaped  8x8x12 x2 -> 6",
    },
    Shape {
        input: (5, 7, 4),
        scale: 2,
        nb_kernel: 3,
        kernel: (3, 3, 4),
        mode: PaddingMode::Valid,
        label: "valid         5x7x4  x2 -> 3",
    },
    Shape {
        input: (4, 4, 5),
        scale: 3,
        nb_kernel: 2,
        kernel: (3, 3, 5),
        mode: PaddingMode::Same,
        label: "scale 3       4x4x5  x3 -> 2",
    },
    Shape {
        input: (7, 7, 4),
        scale: 1,
        nb_kernel: 3,
        kernel: (3, 3, 4),
        mode: PaddingMode::Same,
        label: "scale 1       7x7x4  x1 -> 3",
    },
    Shape {
        input: (6, 6, 3),
        scale: 2,
        nb_kernel: 4,
        kernel: (1, 1, 3),
        mode: PaddingMode::Same,
        label: "1x1 kernel    6x6x3  x2 -> 4",
    },
    Shape {
        input: (5, 5, 4),
        scale: 2,
        nb_kernel: 3,
        kernel: (5, 5, 4),
        mode: PaddingMode::Same,
        label: "5x5 kernel    5x5x4  x2 -> 3",
    },
    Shape {
        input: (2, 3, 2),
        scale: 2,
        nb_kernel: 2,
        kernel: (3, 3, 2),
        mode: PaddingMode::Same,
        label: "tiny          2x3x2  x2 -> 2",
    },
];

impl Shape {
    fn dim_input(&self) -> Dim3 {
        Dim3::new(self.input)
    }
    fn dim_kernel(&self) -> Dim3 {
        Dim3::new(self.kernel)
    }
    fn layer_type(&self) -> UpsampleConvType {
        UpsampleConvType::new(
            self.dim_input(),
            self.scale,
            self.nb_kernel,
            self.dim_kernel(),
            self.mode,
        )
    }
    fn dim_output(&self) -> Dim3 {
        let mut ty = self.layer_type();
        ty.set_dim_output().expect("invalid shape in test table")
    }
    fn weight_len(&self) -> usize {
        (self.dim_kernel().length() * self.nb_kernel) as usize
    }
    fn pads(&self) -> (i32, i32) {
        match self.mode {
            PaddingMode::Same => ((self.kernel.0 / 2) as i32, (self.kernel.1 / 2) as i32),
            PaddingMode::Valid => (0, 0),
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic, asymmetric test data — no symmetry that could mask an index bug
// ---------------------------------------------------------------------------

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

fn dispatch_legacy(
    gpu: &GpuContext,
    entry_point: &str,
    bgl: &wgpu::BindGroupLayout,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy_upsample_conv"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(LEGACY_BACKWARD)),
    });
    let pl = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("legacy_upsample_conv_pl"),
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("legacy_upsample_conv_pipeline"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: Default::default(),
        });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
}

/// One-layer training model, one step at lr = 0 so the gradients can be read
/// back without the optimiser having moved anything.
async fn trained_layer(
    gpu: Arc<GpuContext>,
    shape: &Shape,
) -> (Model<Training>, Vec<f32>, Vec<f32>) {
    let mut model =
        Model::<Training>::new_training(gpu.clone(), 0.0, 1, LossMethod::MeanSquared).await;
    model
        .add_layer(LayerTypes::UpsampleConv(shape.layer_type()))
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

struct Reference {
    output: Vec<f64>,
    grad_input: Vec<f64>,
    grad_weights: Vec<f64>,
    grad_bias: Vec<f64>,
}

/// Upsample-convolution forward + MSE loss + backward, in f64, written along the
/// FORWARD tap map (a scatter). See the module docs for why that matters.
fn reference_upsample_conv(
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
    let scale = shape.scale;
    let (up_h, up_w) = (ih * scale, iw * scale);
    let (pad_y, pad_x) = shape.pads();

    let in_idx = |iy: u32, ix: u32, iz: u32| (iy * iw * ic + ix * ic + iz) as usize;
    let w_idx = |k: u32, ky: u32, kx: u32, kz: u32| {
        (k * kh * kw * ic + ky * kw * ic + kx * ic + kz) as usize
    };
    let out_idx = |oy: u32, ox: u32, k: u32| (oy * ow * k_count + ox * k_count + k) as usize;

    // --- forward
    let mut output = vec![0.0f64; (oh * ow * k_count) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            for k in 0..k_count {
                let mut sum = bias[k as usize] as f64;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let uy = oy as i32 + ky as i32 - pad_y;
                        let ux = ox as i32 + kx as i32 - pad_x;
                        if uy < 0 || uy >= up_h as i32 || ux < 0 || ux >= up_w as i32 {
                            continue; // zero padding
                        }
                        // Nearest-neighbour upsample: the pixel of the small map
                        // that the upsampled position reads from.
                        let iy = uy as u32 / scale;
                        let ix = ux as u32 / scale;
                        for kz in 0..ic {
                            sum += input[in_idx(iy, ix, kz)] as f64
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
                        let uy = oy as i32 + ky as i32 - pad_y;
                        let ux = ox as i32 + kx as i32 - pad_x;
                        if uy < 0 || uy >= up_h as i32 || ux < 0 || ux >= up_w as i32 {
                            continue;
                        }
                        let iy = uy as u32 / scale;
                        let ix = ux as u32 / scale;
                        for kz in 0..ic {
                            grad_input[in_idx(iy, ix, kz)] +=
                                go * weights[w_idx(k, ky, kx, kz)] as f64;
                            grad_weights[w_idx(k, ky, kx, kz)] +=
                                go * input[in_idx(iy, ix, kz)] as f64;
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
// The tests
// ---------------------------------------------------------------------------

/// The lane count the bias kernel reads out of the uniform and the slot count
/// the Rust side sizes its dispatch with are two views of one decision. If they
/// drift, the dispatch silently under-covers `grad_bias` and some gradients are
/// never written — a bug no shape-independent assertion would notice.
///
/// It also pins **where** the count sits: word 3, the one the legacy fixture
/// declares as `_pad` and never reads. Moving it would shift every field the
/// fixture binds against, and the fixture would then compare garbage without
/// failing to compile.
#[test]
fn upsample_bias_lanes_agree_with_dispatch() {
    for shape in SHAPES {
        let mut ty = shape.layer_type();
        ty.set_dim_output().unwrap();

        let bytes = ty.get_spec_uniform_bytes();
        assert_eq!(
            bytes.len(),
            64,
            "\n{}: the uniform changed size — the legacy fixture binds a \
             64-byte struct at fixed offsets\n",
            shape.label
        );
        let lanes = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

        assert!(
            lanes.is_power_of_two() && (1..=64).contains(&lanes),
            "\n{}: bias lanes = {lanes}, must be a power of two in 1..=64 \
             (the tree reduction halves it down to 1)\n",
            shape.label
        );

        let dim_out = shape.dim_output();
        let positions = dim_out.x * dim_out.y;
        assert!(
            lanes == 1 || positions / lanes >= 16,
            "\n{}: {lanes} lanes over {positions} positions leaves {} per lane \
             — the reduction would cost more than the sum\n",
            shape.label,
            positions / lanes
        );

        let slots = 64 / lanes;
        let counts = ty.get_back_workgroup_counts(1);
        assert_eq!(
            counts[2],
            shape.nb_kernel.div_ceil(slots),
            "\n{}: grad_bias dispatch is {} workgroups x {slots} slots for {} \
             biases\n",
            shape.label,
            counts[2],
            shape.nb_kernel
        );
        assert!(
            counts[2] * slots >= shape.nb_kernel,
            "\n{}: grad_bias dispatch leaves biases unwritten\n",
            shape.label
        );
    }
}

/// The forward was not touched, and this pins that: any drift in the shared
/// uniform, the shared bind group or the padding rule would show here first,
/// on an implementation that has no "old" to be compared against.
#[test]
fn the_forward_still_matches_the_f64_reference() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        for shape in SHAPES {
            let (model, input, target) = trained_layer(gpu.clone(), shape).await;
            let layer = model.layers.first().unwrap();
            let got = read_layer_buffer(gpu.as_ref(), layer.buffers.forward[4].as_ref());
            let reference = reference_upsample_conv(
                shape,
                &input,
                &target,
                &synthetic_weights(shape),
                &synthetic_bias(shape.nb_kernel),
            );
            let err = worst_relative_to_reference(&got, &reference.output);
            assert!(
                err < 1e-5,
                "\n{} forward is off the f64 reference by {err:e}\n",
                shape.label
            );
        }
    });
}

/// The claim the whole change rests on: the inverted map computes the number the
/// scan computed.
///
/// `grad_weights` and `grad_bias` are compared too even though their kernels are
/// untouched — they share the bind group and the uniform with `grad_input`, so a
/// change that broke the layout would show up here rather than in a crash.
#[test]
fn the_inverted_tap_map_computes_the_gradient_the_output_scan_computed() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);

        for shape in SHAPES {
            let (model, input, target) = trained_layer(gpu.clone(), shape).await;
            let layer = model.layers.first().unwrap();
            let bwd = layer.buffers.backward.as_ref().unwrap();

            let new_grad_input = read_layer_buffer(gpu.as_ref(), bwd[4].as_ref());
            let new_grad_weights = read_layer_buffer(gpu.as_ref(), bwd[5].as_ref());
            let new_grad_bias = read_layer_buffer(gpu.as_ref(), bwd[6].as_ref());

            // `+=` accumulators: zeroed so the legacy re-run is comparable.
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.clear_buffer(bwd[5].as_ref(), 0, None);
            encoder.clear_buffer(bwd[6].as_ref(), 0, None);
            gpu.queue.submit([encoder.finish()]);

            let bgl = layer.pipeline.backward[0].0.get_bind_group_layout(0);
            let bind_group = layer.bind_group.backward.as_ref().unwrap();
            // The legacy counts: one thread per element, `@workgroup_size(64)`.
            // Written out rather than read from the layer so the legacy dispatch
            // can never silently inherit a count the new kernel chose.
            for (entry_point, workgroups) in [
                (
                    "upsample_conv_back_input",
                    shape.dim_input().length().div_ceil(64),
                ),
                (
                    "upsample_conv_back_weights",
                    (shape.dim_kernel().length() * shape.nb_kernel).div_ceil(64),
                ),
                ("upsample_conv_back_bias", shape.nb_kernel.div_ceil(64)),
            ] {
                dispatch_legacy(gpu.as_ref(), entry_point, &bgl, bind_group, workgroups);
            }

            let old_grad_input = read_layer_buffer(gpu.as_ref(), bwd[4].as_ref());
            let old_grad_weights = read_layer_buffer(gpu.as_ref(), bwd[5].as_ref());
            let old_grad_bias = read_layer_buffer(gpu.as_ref(), bwd[6].as_ref());

            let reference = reference_upsample_conv(
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

                println!(
                    "{:<30} {:<13} new^old {:>9.2e} | new^f64 {:>9.2e} | old^f64 {:>9.2e}{}",
                    shape.label,
                    name,
                    cross,
                    new_err,
                    old_err,
                    if new_err < old_err { "  (new closer)" } else { "" },
                );

                assert!(
                    cross < 1e-4,
                    "\n{} backward {name} disagrees with the output-scan implementation.\n\
                     worst relative difference: {cross:e}\n\
                     new[..8]: {:?}\n\
                     old[..8]: {:?}\n",
                    shape.label,
                    &new[..new.len().min(8)],
                    &old[..old.len().min(8)],
                );

                // `grad_weights` claims MORE than agreement within a tolerance.
                // Its rewrite hoisted loop-invariant work and turned two
                // divisions per position into two counters; it visits the very
                // same taps in the very same `(b, oy, ox)` order, skipping the
                // very same positions. So the sum is the same sum, summed the
                // same way, and the only honest assertion is bit-identity.
                //
                // A tolerance here would pass on a rewrite that quietly
                // reordered the accumulation — which is precisely the mistake
                // this claim exists to rule out.
                if name == "grad_weights" {
                    let differing = new
                        .iter()
                        .zip(old.iter())
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        differing,
                        0,
                        "\n{} grad_weights is not bit-identical to the loop it \
                         replaces ({differing}/{} elements differ).\n\
                         That rewrite only hoists integer work out of the inner \
                         loop — every tap, and their order, is unchanged — so \
                         any float difference means the accumulation moved.\n\
                         worst relative difference: {cross:e}\n",
                        shape.label,
                        new.len(),
                    );
                }
                assert!(
                    new_err < 1e-4,
                    "\n{} backward {name} is off the f64 reference by {new_err:e} \
                     (scan: {old_err:e}).\n",
                    shape.label
                );
                // Reordering a sum of f32 must not cost accuracy. The slack
                // absorbs the shapes where both already sit at the rounding
                // floor — the same clause, and the same 1.5, as
                // `conv_equivalence_tests`.
                assert!(
                    new_err <= old_err.max(1e-7) * 1.5,
                    "\n{} backward {name} is LESS accurate than the implementation \
                     it replaces.\n\
                     new vs f64 reference: {new_err:e}\n\
                     scan vs f64 reference: {old_err:e}\n",
                    shape.label
                );
            }
        }
    });
}

/// An oracle that knows nothing about either implementation: perturb one input
/// element, watch the loss.
///
/// The two above share a premise — that the forward tap map is what both files
/// say it is. This one does not: it only assumes that `grad_input` is the
/// derivative of the loss the model reports. If the rewrite and the legacy
/// shader were wrong in the *same* way, this is the test that would notice.
#[test]
fn upsample_conv_grad_input_matches_finite_differences() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        // Two small shapes: a finite-difference check costs two forward passes
        // per probed element, and its value is in the independence of the
        // oracle, not in the size of the tensor. BOTH padding modes, though —
        // with `Valid` the pad offset is zero, so a mutation that drops it is
        // invisible on a `Valid` shape. (Measured: that mutation was caught by
        // the equivalence test alone until `Same` was added here.)
        for shape in [&SHAPES[2], &SHAPES[7]] {
            check_finite_differences(gpu.clone(), shape).await;
        }
    });
}

async fn check_finite_differences(gpu: Arc<GpuContext>, shape: &Shape) {
    let weights = synthetic_weights(shape);
    let bias = synthetic_bias(shape.nb_kernel);
    let input = synthetic_input(shape.dim_input());
    let target = synthetic_target(shape.dim_output());

    let (model, _, _) = trained_layer(gpu.clone(), shape).await;
    let bwd = model
        .layers
        .first()
        .unwrap()
        .buffers
        .backward
        .as_ref()
        .unwrap();
    let grad_input = read_layer_buffer(gpu.as_ref(), bwd[4].as_ref());

    // The loss is the ORACLE's forward, not the GPU's: the point of this check
    // is an oracle that shares nothing with the kernel under test.
    let loss_of = |x: &[f32]| -> f64 {
        let out = reference_upsample_conv(shape, x, &target, &weights, &bias).output;
        out.iter()
            .zip(&target)
            .map(|(o, t)| (o - *t as f64).powi(2))
            .sum::<f64>()
            / out.len() as f64
    };

    const EPS: f32 = 1e-3;
    let mut worst = 0.0f64;
    // A stride coprime with the channel count, so the probe visits every
    // channel and both parities of the upsampling window — but every element on
    // a tensor small enough that a stride would leave most of it unprobed.
    let stride = if input.len() < 64 { 1 } else { 7 };
    for i in (0..input.len()).step_by(stride) {
        let mut plus = input.clone();
        plus[i] += EPS;
        let mut minus = input.clone();
        minus[i] -= EPS;
        let numeric = (loss_of(&plus) - loss_of(&minus)) / (2.0 * EPS as f64);
        let analytic = grad_input[i] as f64;
        let scale = numeric.abs().max(analytic.abs()).max(1e-4);
        worst = worst.max((numeric - analytic).abs() / scale);
    }
    assert!(
        worst < 5e-2,
        "\n{}: grad_input disagrees with finite differences by {worst:e}\n",
        shape.label
    );
}
