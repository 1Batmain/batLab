//! File purpose: Proves the workgroup-reduction `group_norm` is numerically
//! equivalent to the naive per-element implementation it replaced.
//!
//! Two independent lines of evidence:
//!
//! 1. **Old vs new on the very same buffers.** `shader/legacy/*_naive.wgsl`
//!    are verbatim copies of the pre-optimisation shaders, kept as test
//!    fixtures. Each test builds a real model (new shaders), runs it, then
//!    re-dispatches the legacy kernels over the *same* bind group and compares
//!    the results. Only the floating-point summation order differs, so the
//!    residual is at the rounding-noise level.
//! 2. **Finite differences.** The gradients the new backward produces are
//!    checked against central differences of the loss — an oracle that knows
//!    nothing about either implementation.

use crate::gpu_context::GpuContext;
use crate::model::debug::read_back_f32;
use crate::model::layer::Layer;
use crate::model::{Dim3, GroupNormType, Infer, LayerTypes, LossMethod, Model, Training};
use std::sync::Arc;

const LEGACY_FORWARD: &str = include_str!("shader/legacy/group_norm_naive.wgsl");
const LEGACY_BACKWARD: &str = include_str!("shader/legacy/back_group_norm_naive.wgsl");

/// Worst relative difference between two f32 vectors, using a floor on the
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

/// A deterministic, non-degenerate input: no symmetry that could mask an
/// indexing mistake, and a per-channel offset so the groups have genuinely
/// different means and variances.
fn synthetic_input(dim: Dim3) -> Vec<f32> {
    let z = dim.z as usize;
    (0..dim.length() as usize)
        .map(|i| {
            let channel = i % z;
            let spatial = i / z;
            0.7 * ((i as f32) * 0.37).sin() + 0.15 * channel as f32 - 0.01 * spatial as f32
        })
        .collect()
}

/// Dispatch a legacy entry point over an existing bind group. The legacy
/// shaders declare a subset of the current bindings, which a bind group layout
/// is allowed to be a superset of.
fn dispatch_legacy(
    gpu: &GpuContext,
    source: &str,
    entry_point: &str,
    bgl: &wgpu::BindGroupLayout,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("legacy_group_norm"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    });
    let pl = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("legacy_pl"),
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("legacy_pipeline"),
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

fn read_layer_buffer(gpu: &GpuContext, buf: &wgpu::Buffer) -> Vec<f32> {
    read_back_f32(gpu, buf, buf.size()).expect("buffer readback failed")
}

/// Overwrite `gamma` and `beta` with values that are neither all-ones nor
/// all-zeros, so a bug that drops either of them cannot hide.
fn write_channel_params(gpu: &GpuContext, layer: &Layer, channels: usize) {
    let gamma: Vec<f32> = (0..channels).map(|c| 0.6 + 0.25 * c as f32).collect();
    let beta: Vec<f32> = (0..channels).map(|c| -0.4 + 0.11 * c as f32).collect();
    gpu.queue.write_buffer(
        layer.buffers.forward[1].as_ref(),
        0,
        bytemuck::cast_slice(&gamma),
    );
    gpu.queue.write_buffer(
        layer.buffers.forward[2].as_ref(),
        0,
        bytemuck::cast_slice(&beta),
    );
}

/// Shapes worth covering: several groups, a group that spans several channels,
/// a spatial extent larger and smaller than the 256-thread workgroup, and a
/// single-group degenerate case.
const SHAPES: &[((u32, u32, u32), u32)] = &[
    ((32, 32, 32), 8), // the widest GroupNorm in Greyscale_Diffusion
    ((16, 16, 32), 8), // the middle one
    ((32, 32, 16), 4), // the first one
    ((4, 4, 8), 2),    // group_len = 64 < workgroup size
    ((3, 5, 4), 1),    // single group, non-power-of-two spatial extent
    ((2, 2, 6), 6),    // one channel per group
];

// ---------------------------------------------------------------------------
// f64 oracle
// ---------------------------------------------------------------------------

/// Group norm forward + backward through an MSE loss, in f64. Independent of
/// both GPU implementations, and precise enough to arbitrate between them
/// where the f32 grad_input formula cancels catastrophically.
struct Reference {
    grad_input: Vec<f64>,
    grad_gamma: Vec<f64>,
    grad_beta: Vec<f64>,
}

fn reference_backward(
    input: &[f32],
    target: &[f32],
    gamma: &[f32],
    beta: &[f32],
    dim: Dim3,
    num_groups: u32,
    epsilon: f64,
) -> Reference {
    let z = dim.z as usize;
    let cpg = z / num_groups as usize;
    let spatial = (dim.x * dim.y) as usize;
    let n = input.len();
    let group_len = spatial * cpg;

    let indices_of = |g: usize| -> Vec<usize> {
        (0..spatial)
            .flat_map(|s| (0..cpg).map(move |c| s * z + g * cpg + c))
            .collect()
    };

    let mut x_hat = vec![0.0f64; n];
    let mut output = vec![0.0f64; n];
    let mut inv_std = vec![0.0f64; num_groups as usize];
    for g in 0..num_groups as usize {
        let idx = indices_of(g);
        let mean = idx.iter().map(|&i| input[i] as f64).sum::<f64>() / group_len as f64;
        let var = idx
            .iter()
            .map(|&i| (input[i] as f64 - mean).powi(2))
            .sum::<f64>()
            / group_len as f64;
        inv_std[g] = 1.0 / (var + epsilon).sqrt();
        for &i in &idx {
            x_hat[i] = (input[i] as f64 - mean) * inv_std[g];
            output[i] = x_hat[i] * gamma[i % z] as f64 + beta[i % z] as f64;
        }
    }

    // MSE loss gradient, matching shader/loss.wgsl.
    let grad_output: Vec<f64> = (0..n)
        .map(|i| 2.0 * (output[i] - target[i] as f64) / n as f64)
        .collect();

    let mut grad_input = vec![0.0f64; n];
    for g in 0..num_groups as usize {
        let idx = indices_of(g);
        let dxhat: Vec<f64> = idx
            .iter()
            .map(|&i| grad_output[i] * gamma[i % z] as f64)
            .collect();
        let sum_dxhat: f64 = dxhat.iter().sum();
        let sum_dxhat_xhat: f64 = idx.iter().zip(&dxhat).map(|(&i, d)| d * x_hat[i]).sum();
        let m = group_len as f64;
        for (&i, d) in idx.iter().zip(&dxhat) {
            grad_input[i] = inv_std[g] / m * (m * d - sum_dxhat - x_hat[i] * sum_dxhat_xhat);
        }
    }

    let mut grad_gamma = vec![0.0f64; z];
    let mut grad_beta = vec![0.0f64; z];
    for c in 0..z {
        for s in 0..spatial {
            let i = s * z + c;
            grad_gamma[c] += grad_output[i] * x_hat[i];
            grad_beta[c] += grad_output[i];
        }
    }

    Reference {
        grad_input,
        grad_gamma,
        grad_beta,
    }
}

/// Worst difference between an f32 GPU result and the f64 oracle, relative to
/// the oracle's own scale.
fn worst_relative_to_reference(got: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(got.len(), reference.len());
    let scale = reference.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-12);
    got.iter()
        .zip(reference)
        .fold(0.0f64, |acc, (g, r)| acc.max((*g as f64 - r).abs() / scale))
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

#[test]
fn forward_matches_naive_implementation() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);

        for &(dims, num_groups) in SHAPES {
            let dim = Dim3::new(dims);
            let mut model: Model<Infer> = Model::new(gpu.clone()).await;
            model
                .add_layer(LayerTypes::GroupNorm(GroupNormType::new(dim, num_groups)))
                .unwrap();
            model.build_model().unwrap();

            let layer = model.layers.first().unwrap();
            write_channel_params(gpu.as_ref(), layer, dim.z as usize);

            let input = synthetic_input(dim);
            let optimised = model.predict(&input);

            // Same buffers, same input, legacy kernel: one thread per element.
            let layer = model.layers.first().unwrap();
            let pipeline = layer.pipeline.forward.as_ref().unwrap();
            dispatch_legacy(
                gpu.as_ref(),
                LEGACY_FORWARD,
                "group_norm",
                &pipeline.get_bind_group_layout(0),
                layer.bind_group.forward.as_ref().unwrap(),
                dim.length().div_ceil(64),
            );
            let naive = read_layer_buffer(gpu.as_ref(), layer.buffers.forward[4].as_ref());

            let worst = worst_relative_diff(&optimised, &naive);
            assert!(
                worst < 1e-5,
                "\nGroupNorm forward {dims:?} / {num_groups} groups disagrees with the \
                 naive implementation.\n\
                 worst relative difference: {worst:e}\n\
                 optimised[..8]: {:?}\n\
                 naive[..8]:     {:?}\n",
                &optimised[..optimised.len().min(8)],
                &naive[..naive.len().min(8)],
            );
        }
    });
}

/// The defining property of the operator, independent of either
/// implementation: with gamma = 1 and beta = 0 every group of the output has
/// zero mean and unit variance.
#[test]
fn forward_normalises_each_group() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = Dim3::new((16, 16, 32));
        let num_groups = 8u32;
        let cpg = (dim.z / num_groups) as usize;
        let z = dim.z as usize;

        let mut model: Model<Infer> = Model::new(gpu.clone()).await;
        model
            .add_layer(LayerTypes::GroupNorm(GroupNormType::new(dim, num_groups)))
            .unwrap();
        model.build_model().unwrap();
        // gamma is initialised to ones and beta to zeros by default.

        let out = model.predict(&synthetic_input(dim));

        for g in 0..num_groups as usize {
            let values: Vec<f64> = out
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    let channel = i % z;
                    channel / cpg == g
                })
                .map(|(_, v)| *v as f64)
                .collect();
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            assert!(
                mean.abs() < 1e-4,
                "group {g}: mean = {mean:e}, expected ~0 (n = {n})"
            );
            assert!(
                (var - 1.0).abs() < 1e-3,
                "group {g}: variance = {var}, expected ~1 (n = {n})"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

/// Build a one-GroupNorm training model, run one step at lr = 0 (so nothing is
/// perturbed between producing and reading the gradients) and return it.
async fn trained_group_norm(
    gpu: Arc<GpuContext>,
    dim: Dim3,
    num_groups: u32,
) -> (Model<Training>, Vec<f32>, Vec<f32>) {
    let mut model = Model::<Training>::new_training(gpu.clone(), 0.0, 1, LossMethod::MeanSquared)
        .await;
    model
        .add_layer(LayerTypes::GroupNorm(GroupNormType::new(dim, num_groups)))
        .unwrap();
    model.build().unwrap();

    write_channel_params(gpu.as_ref(), model.layers.first().unwrap(), dim.z as usize);

    let input = synthetic_input(dim);
    let target: Vec<f32> = (0..dim.length() as usize)
        .map(|i| 0.3 * ((i as f32) * 0.11).cos())
        .collect();
    model.train_step(&input, &target);
    (model, input, target)
}

#[test]
fn backward_matches_naive_implementation() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);

        for &(dims, num_groups) in SHAPES {
            let dim = Dim3::new(dims);
            let (model, input, target) = trained_group_norm(gpu.clone(), dim, num_groups).await;

            let layer = model.layers.first().unwrap();
            let bwd = layer.buffers.backward.as_ref().unwrap();
            let optimised_grad_input = read_layer_buffer(gpu.as_ref(), bwd[4].as_ref());
            let optimised_grad_gamma = read_layer_buffer(gpu.as_ref(), bwd[5].as_ref());
            let optimised_grad_beta = read_layer_buffer(gpu.as_ref(), bwd[6].as_ref());

            // grad_gamma / grad_beta accumulate with `+=` across a batch, so
            // they must start from zero for the legacy re-run to be comparable.
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.clear_buffer(bwd[5].as_ref(), 0, None);
            encoder.clear_buffer(bwd[6].as_ref(), 0, None);
            gpu.queue.submit([encoder.finish()]);

            let bgl = layer.pipeline.backward[0].0.get_bind_group_layout(0);
            let bind_group = layer.bind_group.backward.as_ref().unwrap();
            for (entry_point, workgroups) in [
                ("group_norm_back_input", dim.length().div_ceil(64)),
                ("group_norm_back_gamma", dim.z.div_ceil(64)),
                ("group_norm_back_beta", dim.z.div_ceil(64)),
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

            let naive_grad_input = read_layer_buffer(gpu.as_ref(), bwd[4].as_ref());
            let naive_grad_gamma = read_layer_buffer(gpu.as_ref(), bwd[5].as_ref());
            let naive_grad_beta = read_layer_buffer(gpu.as_ref(), bwd[6].as_ref());

            // f64 oracle, to arbitrate wherever the two f32 results differ.
            let channels = dim.z as usize;
            let gamma: Vec<f32> = (0..channels).map(|c| 0.6 + 0.25 * c as f32).collect();
            let beta: Vec<f32> = (0..channels).map(|c| -0.4 + 0.11 * c as f32).collect();
            let reference = reference_backward(
                &input,
                &target,
                &gamma,
                &beta,
                dim,
                num_groups,
                GroupNormType::new(dim, num_groups).epsilon as f64,
            );

            for (name, optimised, naive, oracle) in [
                (
                    "grad_input",
                    &optimised_grad_input,
                    &naive_grad_input,
                    &reference.grad_input,
                ),
                (
                    "grad_gamma",
                    &optimised_grad_gamma,
                    &naive_grad_gamma,
                    &reference.grad_gamma,
                ),
                (
                    "grad_beta",
                    &optimised_grad_beta,
                    &naive_grad_beta,
                    &reference.grad_beta,
                ),
            ] {
                let cross = worst_relative_diff(optimised, naive);
                let new_err = worst_relative_to_reference(optimised, oracle);
                let old_err = worst_relative_to_reference(naive, oracle);

                // grad_input evaluates `M*dxhat - sum_dxhat - xhat*sum_dxhat_xhat`,
                // a difference of terms that are individually ~M times larger than
                // the result. Any change in the summation order of `sum_dxhat` is
                // amplified by that cancellation, so f32 old-vs-new cannot be held
                // to the rounding of the output alone. The oracle checks below are
                // what actually bounds the error.
                assert!(
                    cross < 1e-4,
                    "\nGroupNorm backward {name} on {dims:?} / {num_groups} groups disagrees \
                     with the naive implementation.\n\
                     worst relative difference: {cross:e}\n\
                     optimised[..8]: {:?}\n\
                     naive[..8]:     {:?}\n",
                    &optimised[..optimised.len().min(8)],
                    &naive[..naive.len().min(8)],
                );
                assert!(
                    new_err < 1e-4,
                    "\nGroupNorm backward {name} on {dims:?} / {num_groups} groups is off the \
                     f64 reference by {new_err:e} (naive: {old_err:e}).\n"
                );
                // A tree reduction over f32 is not less accurate than the naive
                // sequential sum it replaces; the slack absorbs the cases where
                // both are already at the rounding floor.
                assert!(
                    new_err <= old_err.max(1e-7) * 1.5,
                    "\nGroupNorm backward {name} on {dims:?} / {num_groups} groups is LESS \
                     accurate than the implementation it replaces.\n\
                     new vs f64 reference:   {new_err:e}\n\
                     naive vs f64 reference: {old_err:e}\n"
                );
            }
        }
    });
}

/// grad_gamma / grad_beta from the GPU backward, checked against central
/// differences of the loss. Catches an error the old-vs-new comparison cannot:
/// one both implementations would share.
#[test]
fn gamma_beta_gradients_match_finite_differences() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = Dim3::new((4, 4, 8));
        let num_groups = 2u32;
        let channels = dim.z as usize;

        let (mut model, input, target) = trained_group_norm(gpu.clone(), dim, num_groups).await;
        let layer = model.layers.first().unwrap();
        let bwd = layer.buffers.backward.as_ref().unwrap();
        let analytic_gamma = read_layer_buffer(gpu.as_ref(), bwd[5].as_ref());
        let analytic_beta = read_layer_buffer(gpu.as_ref(), bwd[6].as_ref());

        let base_gamma: Vec<f32> = (0..channels).map(|c| 0.6 + 0.25 * c as f32).collect();
        let base_beta: Vec<f32> = (0..channels).map(|c| -0.4 + 0.11 * c as f32).collect();

        let eps = 1e-3f32;
        let loss_at = |model: &mut Model<Training>, gamma: &[f32], beta: &[f32]| -> f32 {
            let layer = model.layers.first().unwrap();
            gpu.queue.write_buffer(
                layer.buffers.forward[1].as_ref(),
                0,
                bytemuck::cast_slice(gamma),
            );
            gpu.queue.write_buffer(
                layer.buffers.forward[2].as_ref(),
                0,
                bytemuck::cast_slice(beta),
            );
            model.train_step(&input, &target);
            model.read_last_loss()
        };

        let mut numeric_gamma = vec![0.0f32; channels];
        let mut numeric_beta = vec![0.0f32; channels];
        for c in 0..channels {
            let mut plus = base_gamma.clone();
            plus[c] += eps;
            let mut minus = base_gamma.clone();
            minus[c] -= eps;
            let lp = loss_at(&mut model, &plus, &base_beta);
            let lm = loss_at(&mut model, &minus, &base_beta);
            numeric_gamma[c] = (lp - lm) / (2.0 * eps);

            let mut plus = base_beta.clone();
            plus[c] += eps;
            let mut minus = base_beta.clone();
            minus[c] -= eps;
            let lp = loss_at(&mut model, &base_gamma, &plus);
            let lm = loss_at(&mut model, &base_gamma, &minus);
            numeric_beta[c] = (lp - lm) / (2.0 * eps);
        }

        for (name, analytic, numeric) in [
            ("grad_gamma", &analytic_gamma, &numeric_gamma),
            ("grad_beta", &analytic_beta, &numeric_beta),
        ] {
            let mut worst = 0.0f32;
            for c in 0..channels {
                let denom = analytic[c].abs().max(numeric[c].abs()).max(1e-4);
                worst = worst.max((analytic[c] - numeric[c]).abs() / denom);
            }
            assert!(
                worst < 5e-2,
                "\nGroupNorm {name} disagrees with finite differences.\n\
                 analytic (GPU backward): {analytic:?}\n\
                 numeric  (central diff): {numeric:?}\n\
                 worst relative error: {worst:.4}\n"
            );
        }
    });
}

/// grad_input from the GPU backward, checked against central differences of
/// the loss with respect to the layer's input. This is the formula the two
/// hoisted reductions feed, so it is the one most exposed to the refactor.
#[test]
fn input_gradients_match_finite_differences() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let dim = Dim3::new((4, 4, 8));
        let num_groups = 2u32;
        let len = dim.length() as usize;

        let (mut model, input, target) = trained_group_norm(gpu.clone(), dim, num_groups).await;
        let layer = model.layers.first().unwrap();
        let bwd = layer.buffers.backward.as_ref().unwrap();
        let analytic = read_layer_buffer(gpu.as_ref(), bwd[4].as_ref());

        let eps = 1e-2f32;
        let mut loss_at = |model: &mut Model<Training>, x: &[f32]| -> f32 {
            model.train_step(x, &target);
            model.read_last_loss()
        };
        let base_loss = loss_at(&mut model, &input);

        // The loss comes back as a single f32, so `lp - lm` is quantised to
        // ulp(loss) and the finite difference cannot resolve anything below
        // ulp(loss) / (2 * eps). Probes whose gradient sits under that floor
        // carry no information — measuring them would only be measuring the
        // readback's rounding. They are reported, not asserted on.
        let quantisation_floor = (base_loss.abs() * f32::EPSILON) / (2.0 * eps);
        let resolvable = 20.0 * quantisation_floor;

        // Sampling a slice of the tensor keeps the test quick while still
        // covering every group and several spatial positions.
        let probes: Vec<usize> = (0..len).step_by(5).collect();
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        let mut asserted = 0usize;
        for &i in &probes {
            if analytic[i].abs() < resolvable {
                continue;
            }
            asserted += 1;
            let mut plus = input.clone();
            plus[i] += eps;
            let mut minus = input.clone();
            minus[i] -= eps;
            let lp = loss_at(&mut model, &plus);
            let lm = loss_at(&mut model, &minus);
            let numeric = (lp - lm) / (2.0 * eps);
            let rel = (analytic[i] - numeric).abs() / analytic[i].abs().max(numeric.abs());
            if rel > worst {
                worst = rel;
                worst_at = i;
            }
        }

        // Guards the filter itself: if it ever swallowed most of the tensor the
        // test would be vacuous.
        assert!(
            asserted * 2 >= probes.len(),
            "\nOnly {asserted}/{} probes were above the finite-difference \
             quantisation floor ({resolvable:e}) — the test has gone vacuous.\n",
            probes.len()
        );
        assert!(
            worst < 5e-2,
            "\nGroupNorm grad_input disagrees with finite differences.\n\
             worst relative error {worst:.4} at element {worst_at} \
             (analytic {}), over {asserted} probes\n\
             finite-difference quantisation floor: {quantisation_floor:e}\n",
            analytic[worst_at]
        );
    });
}
