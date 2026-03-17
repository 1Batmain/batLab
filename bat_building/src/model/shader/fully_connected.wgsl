// File purpose: WGSL compute shader implementing fully connected operations for model forward/backward or optimizer passes.

// Bindings match FullyConnectedType::get_buffers_specs():
//   [0] input   — flattened input vector
//   [1] weights — output-major matrix: neuron_idx * input_len + in_idx
//   [2] bias    — one bias per neuron
//   [3] specs   — FullyConnectedUniform
//   [4] pre_activation — raw neuron sums before activation
//   [5] output  — activated neuron outputs

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       weights:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias:       array<f32>;
@group(0) @binding(3) var<uniform>             layer_spec: FullyConnectedSpec;
@group(0) @binding(4) var<storage, read_write> pre_activation: array<f32>;
@group(0) @binding(5) var<storage, read_write> output:     array<f32>;

struct FullyConnectedSpec {
    input_len:  u32,
    nb_neurons: u32,
    activation_method: u32, // 0 = ReLU, 1 = Linear, 2 = SiLU
    _pad0:      u32,
    dim_input:  vec3<u32>,
    dim_output: vec3<u32>,
}

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn apply_activation(value: f32, method: u32) -> f32 {
    if method == 0u {
        return max(value, 0.0);
    }
    if method == 2u {
        return value * sigmoid(value);
    }
    return value;
}

@compute @workgroup_size(64)
fn fully_connected(@builtin(global_invocation_id) gid: vec3<u32>) {
    let neuron_idx = gid.x;
    if neuron_idx >= layer_spec.nb_neurons { return; }

    var sum: f32 = bias[neuron_idx];
    for (var in_idx: u32 = 0u; in_idx < layer_spec.input_len; in_idx++) {
        let weight_idx = neuron_idx * layer_spec.input_len + in_idx;
        sum += input[in_idx] * weights[weight_idx];
    }
    pre_activation[neuron_idx] = sum;
    output[neuron_idx] = apply_activation(sum, layer_spec.activation_method);
}
