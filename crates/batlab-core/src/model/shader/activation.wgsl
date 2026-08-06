// File purpose: WGSL compute shader implementing activation operations for model forward/backward or optimizer passes.

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
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) { return; }
    output[i] = max(input[i], 0.0);
}

@compute @workgroup_size(64)
fn linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) { return; }
    output[i] = input[i]; // identity
}

@compute @workgroup_size(64)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) { return; }
    let sig = sigmoid(input[i]);
    output[i] = input[i] * sig;
}
