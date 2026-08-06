// File purpose: WGSL compute shader implementing loss operations for model forward/backward or optimizer passes.

// Bindings match LossType::get_buffers_specs():
//   [0] model_result — the model's forward output  (read)
//   [1] target       — ground-truth labels          (read, CPU writes each step)
//   [2] loss_terms   — per-element squared error    (read_write, read back on CPU)
//   [3] grad_output  — dL/d(output) gradient        (read_write, feeds backward pass)
//   [4] specs        — LossUniform (dims), for the per-sample element count

@group(0) @binding(0) var<storage, read>       model_result:        array<f32>;
@group(0) @binding(1) var<storage, read>       target_result:       array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_terms:          array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_output:         array<f32>;
@group(0) @binding(4) var<uniform>             layer_spec:          LossSpec;

struct LossSpec {
    dim_input:  vec3<u32>,
    dim_output: vec3<u32>,
}

// MSE forward gradient:  grad[i] = 2 * (pred[i] - target_result[i]) / N
//
// `N` is the PER-SAMPLE element count, from the uniform — not
// `arrayLength(&model_result)`, which is `batch * N` once the buffers carry a
// batch axis. Reading it off the array would divide every gradient by an extra
// factor of `batch`: no crash, no NaN, just a silently smaller effective
// learning rate that scales with a knob nobody thinks of as one.
//
// The batch is not an axis of the loss at all: each sample's loss is its own
// mean, and the averaging over the batch happens once, later, in the optimiser
// pass (`grad_scale = 1 / batch`).
@compute @workgroup_size(64)
fn mean_squared(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&model_result) { return; }
    let n = layer_spec.dim_input.x * layer_spec.dim_input.y * layer_spec.dim_input.z;
    let diff = model_result[i] - target_result[i];
    loss_terms[i] = diff * diff;
    grad_output[i] = 2.0 * diff / f32(n);
}
