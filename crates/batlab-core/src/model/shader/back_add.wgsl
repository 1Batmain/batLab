// File purpose: WGSL compute shader for the residual Add layer's backward pass.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

// Bindings match AddType::get_back_buffers_specs():
//   [0] grad_output — incoming gradient for the sum
//   [1] specs       — the one shared shape
//   [2] grad_input  — gradient for the current sequential input
//   [3] grad_skip   — gradient for the saved skip input

@group(0) @binding(0) var<storage, read>       grad_output: array<f32>;
@group(0) @binding(1) var<uniform>             layer_spec:  AddSpec;
@group(0) @binding(2) var<storage, read_write> grad_input:  array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_skip:   array<f32>;

struct AddSpec {
    dim: vec3<u32>,
}

// d(a + b)/da = d(a + b)/db = 1: the incoming gradient flows to BOTH inputs
// unchanged. Every tensor here is the same length (Add guarantees it), so one
// thread copies grad_output into both grad_input and grad_skip. Writing both in
// a SINGLE pass — rather than two passes over the same range — keeps the skip
// gradient's producer and the merge that reads it one hazard apart, not two.
@compute @workgroup_size(64)
fn add_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&grad_input) { return; }
    let g = grad_output[idx];
    grad_input[idx] = g;
    grad_skip[idx] = g;
}
