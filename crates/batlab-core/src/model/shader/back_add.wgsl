// Residual Add layer, backward pass. 2-D dispatch grid past WebGPU's
// 65 535-per-dim limit, so the thread index comes from `num_workgroups`
// (`dispatch_grid` in layer.rs). Bindings match AddType::get_back_buffers_specs().
@group(0) @binding(0) var<storage, read>       grad_output: array<f32>;
@group(0) @binding(1) var<uniform>             layer_spec:  AddSpec;
@group(0) @binding(2) var<storage, read_write> grad_input:  array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_skip:   array<f32>;

struct AddSpec {
    dim: vec3<u32>,
}

// d(a+b)/da = d(a+b)/db = 1: grad_output flows unchanged to both inputs. Done in
// a SINGLE pass, not two over the same range, so the skip gradient's producer
// and the merge that reads it stay one hazard apart, not two.
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
