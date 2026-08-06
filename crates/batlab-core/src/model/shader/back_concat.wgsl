// File purpose: WGSL compute shader implementing back concat operations for model forward/backward or optimizer passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

// Bindings match ConcatType::get_back_buffers_specs():
//   [0] grad_output — incoming gradient for concatenated output
//   [1] specs       — concat dimensions
//   [2] grad_input  — gradient for current sequential input
//   [3] grad_skip   — gradient for saved skip input

@group(0) @binding(0) var<storage, read>       grad_output: array<f32>;
@group(0) @binding(1) var<uniform>             layer_spec:  ConcatSpec;
@group(0) @binding(2) var<storage, read_write> grad_input:  array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_skip:   array<f32>;

struct ConcatSpec {
    dim_input:  vec3<u32>,
    dim_skip:   vec3<u32>,
    dim_output: vec3<u32>,
}

// Mirror of the forward split, with the same caveat: source and destination
// have different per-sample lengths, so each gets its own offset.
@compute @workgroup_size(64)
fn concat_back_input(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let total = layer_spec.dim_input.x * layer_spec.dim_input.y * layer_spec.dim_input.z;
    if idx >= arrayLength(&grad_input) { return; }

    let sample = idx / total;
    let local = idx % total;
    let input_c = layer_spec.dim_input.z;
    let out_c = layer_spec.dim_output.z;
    let pixels = layer_spec.dim_output.x * layer_spec.dim_output.y;
    let channel = local % input_c;
    let pixel_idx = local / input_c;
    let out_idx = sample * pixels * out_c + pixel_idx * out_c + channel;
    grad_input[idx] = grad_output[out_idx];
}

@compute @workgroup_size(64)
fn concat_back_skip(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let total = layer_spec.dim_skip.x * layer_spec.dim_skip.y * layer_spec.dim_skip.z;
    if idx >= arrayLength(&grad_skip) { return; }

    let sample = idx / total;
    let local = idx % total;
    let skip_c = layer_spec.dim_skip.z;
    let out_c = layer_spec.dim_output.z;
    let out_offset = layer_spec.dim_input.z;
    let pixels = layer_spec.dim_output.x * layer_spec.dim_output.y;
    let channel = local % skip_c;
    let pixel_idx = local / skip_c;
    let out_idx = sample * pixels * out_c + pixel_idx * out_c + out_offset + channel;
    grad_skip[idx] = grad_output[out_idx];
}
