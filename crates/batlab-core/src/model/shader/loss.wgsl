// Loss forward/backward. 2-D dispatch grid past WebGPU's 65 535-per-dim limit,
// so the thread index comes from `num_workgroups` (`dispatch_grid` in layer.rs).

// Bindings match LossType::get_buffers_specs(). target: CPU writes each step;
// loss_terms: read back on CPU; grad_output: feeds the backward pass.
@group(0) @binding(0) var<storage, read>       model_result:        array<f32>;
@group(0) @binding(1) var<storage, read>       target_result:       array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_terms:          array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_output:         array<f32>;
@group(0) @binding(4) var<uniform>             layer_spec:          LossSpec;

struct LossSpec {
    dim_input:  vec3<u32>,
    dim_output: vec3<u32>,
}

// MSE:  grad[i] = 2 * (pred[i] - target[i]) / N. `N` is the PER-SAMPLE element
// count from the uniform, NOT `arrayLength` (= `batch * N`): dividing by the
// latter silently shrinks the effective LR by a factor of `batch`. Batch
// averaging happens later, in the optimiser pass (`grad_scale = 1 / batch`).
@compute @workgroup_size(64)
fn mean_squared(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if i >= arrayLength(&model_result) { return; }
    let n = layer_spec.dim_input.x * layer_spec.dim_input.y * layer_spec.dim_input.z;
    let diff = model_result[i] - target_result[i];
    loss_terms[i] = diff * diff;
    grad_output[i] = 2.0 * diff / f32(n);
}
