// File purpose: WGSL compute shader implementing back activation operations for model forward/backward or optimizer passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

// Bindings match back_buffers_specs for ActivationType:
//   [0] fwd_input  — the activation input that was used in the forward pass
//   [1] specs      — shared LayerSpec uniform from forward
//   [2] grad_output — incoming gradient from the next layer / loss
//   [3] grad_input  — outgoing gradient to the previous layer

@group(0) @binding(0) var<storage, read>       fwd_input:   array<f32>;
@group(0) @binding(1) var<uniform>             layer_spec:  LayerSpec;
@group(0) @binding(2) var<storage, read>       grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input:  array<f32>;

struct LayerSpec {
    dim_input:  vec3<u32>,
    dim_output: vec3<u32>,
}

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

// d/dx max(x, 0)  =  1 if x > 0 else 0
@compute @workgroup_size(64)
fn relu_back(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if i >= arrayLength(&fwd_input) { return; }
    grad_input[i] = grad_output[i] * select(0.0, 1.0, fwd_input[i] > 0.0);
}

// d/dx x  =  1
@compute @workgroup_size(64)
fn linear_back(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if i >= arrayLength(&fwd_input) { return; }
    grad_input[i] = grad_output[i];
}

// d/dx (x * sigmoid(x))
@compute @workgroup_size(64)
fn silu_back(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if i >= arrayLength(&fwd_input) { return; }
    let sig = sigmoid(fwd_input[i]);
    let grad = sig * (1.0 + fwd_input[i] * (1.0 - sig));
    grad_input[i] = grad_output[i] * grad;
}
