// File purpose: WGSL compute shader implementing activation operations for model forward/backward or optimizer passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

@group(0) @binding(0) var<storage, read>       input:       array<f32>;
@group(0) @binding(1) var<uniform>             layer_spec:  LayerSpec;
@group(0) @binding(2) var<storage, read_write> output:      array<f32>;

struct LayerSpec {
    dim_input:  vec3<u32>,
    dim_output: vec3<u32>,
}

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

@compute @workgroup_size(64)
fn relu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if i >= arrayLength(&input) { return; }
    output[i] = max(input[i], 0.0);
}

@compute @workgroup_size(64)
fn linear(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if i >= arrayLength(&input) { return; }
    output[i] = input[i]; // identity
}

@compute @workgroup_size(64)
fn silu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if i >= arrayLength(&input) { return; }
    let sig = sigmoid(input[i]);
    output[i] = input[i] * sig;
}
