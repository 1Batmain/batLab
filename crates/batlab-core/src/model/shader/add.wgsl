// File purpose: WGSL compute shader for the residual Add layer — `out = input + skip`.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

// Bindings match AddType::get_buffers_specs():
//   [0] input      — current sequential tensor (HWC)
//   [1] skip_input — saved skip tensor (HWC), SAME shape as input
//   [2] specs      — the one shared shape
//   [3] output     — the sum (HWC), same shape again

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       skip_input: array<f32>;
@group(0) @binding(2) var<uniform>             layer_spec: AddSpec;
@group(0) @binding(3) var<storage, read_write> output:     array<f32>;

struct AddSpec {
    dim: vec3<u32>,
}

// Unlike concat, all three tensors share one length, so a single index reaches
// the same element in every one of them — the operation is a plain elementwise
// sum with no channel bookkeeping. `layer_spec.dim` is read only to keep the
// binding live and to bound the guard to the declared shape.
@compute @workgroup_size(64)
fn add(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&output) { return; }
    output[idx] = input[idx] + skip_input[idx];
}
