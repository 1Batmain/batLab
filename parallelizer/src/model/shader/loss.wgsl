// Bindings match LossType::get_buffers_specs():
//   [0] model_result — the model's forward output  (read)
//   [1] target       — ground-truth labels          (read, CPU writes each step)
//   [2] grad_output  — dL/d(output) gradient        (read_write, feeds backward pass)

@group(0) @binding(0) var<storage, read>       model_result: array<f32>;
@group(0) @binding(1) var<storage, read>       target_result:       array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_output:  array<f32>;

// MSE forward gradient:  grad[i] = 2 * (pred[i] - target_result[i]) / N
@compute @workgroup_size(64)
fn mean_squared(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&model_result);
    if i >= n { return; }
    let diff = model_result[i] - target_result[i];
    grad_output[i] = 2.0 * diff / f32(n);
}
