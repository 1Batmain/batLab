// Residual Add layer — `out = input + skip`. 2-D dispatch grid past WebGPU's
// 65 535-per-dim limit, so the thread index comes from `num_workgroups`
// (`dispatch_grid` in layer.rs). Bindings match AddType::get_buffers_specs().
@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       skip_input: array<f32>;
@group(0) @binding(2) var<uniform>             layer_spec: AddSpec;
@group(0) @binding(3) var<storage, read_write> output:     array<f32>;

struct AddSpec {
    dim: vec3<u32>,
}

// All three tensors share one length, so one index reaches the same element in
// each — plain elementwise sum. `layer_spec.dim` is read only to keep the
// binding live.
@compute @workgroup_size(64)
fn add(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&output) { return; }
    output[idx] = input[idx] + skip_input[idx];
}
