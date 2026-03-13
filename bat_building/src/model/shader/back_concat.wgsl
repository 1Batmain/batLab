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

@compute @workgroup_size(64)
fn concat_back_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = layer_spec.dim_input.x * layer_spec.dim_input.y * layer_spec.dim_input.z;
    if idx >= total { return; }

    let input_c = layer_spec.dim_input.z;
    let out_c = layer_spec.dim_output.z;
    let channel = idx % input_c;
    let pixel_idx = idx / input_c;
    let out_idx = pixel_idx * out_c + channel;
    grad_input[idx] = grad_output[out_idx];
}

@compute @workgroup_size(64)
fn concat_back_skip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = layer_spec.dim_skip.x * layer_spec.dim_skip.y * layer_spec.dim_skip.z;
    if idx >= total { return; }

    let skip_c = layer_spec.dim_skip.z;
    let out_c = layer_spec.dim_output.z;
    let out_offset = layer_spec.dim_input.z;
    let channel = idx % skip_c;
    let pixel_idx = idx / skip_c;
    let out_idx = pixel_idx * out_c + out_offset + channel;
    grad_skip[idx] = grad_output[out_idx];
}
