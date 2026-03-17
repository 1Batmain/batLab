// File purpose: WGSL compute shader implementing back fully connected operations for model forward/backward or optimizer passes.

// Bindings match FullyConnectedType::get_back_buffers_specs():
//   [0] fwd_input    — flattened input vector used during forward
//   [1] weights      — forward weights
//   [2] pre_activation — raw neuron sums before activation
//   [3] specs        — FullyConnectedSpec uniform
//   [4] grad_output  — incoming gradient for each output neuron
//   [5] grad_input   — outgoing gradient for each input element
//   [6] grad_weights — gradient for each weight
//   [7] grad_bias    — gradient for each bias

@group(0) @binding(0) var<storage, read>       fwd_input:    array<f32>;
@group(0) @binding(1) var<storage, read>       weights:      array<f32>;
@group(0) @binding(2) var<storage, read>       pre_activation: array<f32>;
@group(0) @binding(3) var<uniform>             layer_spec:   FullyConnectedSpec;
@group(0) @binding(4) var<storage, read>       grad_output:  array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_input:   array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_weights: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_bias:    array<f32>;

struct FullyConnectedSpec {
    input_len:  u32,
    nb_neurons: u32,
    activation_method: u32,
    _pad0:      u32,
    dim_input:  vec3<u32>,
    dim_output: vec3<u32>,
}

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn activation_grad(pre_activated: f32, method: u32) -> f32 {
    if method == 0u {
        return select(0.0, 1.0, pre_activated > 0.0);
    }
    if method == 2u {
        let sig = sigmoid(pre_activated);
        return sig * (1.0 + pre_activated * (1.0 - sig));
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn fully_connected_back_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_idx = gid.x;
    if in_idx >= layer_spec.input_len { return; }

    var grad: f32 = 0.0;
    for (var neuron_idx: u32 = 0u; neuron_idx < layer_spec.nb_neurons; neuron_idx++) {
        let local_grad =
            grad_output[neuron_idx] * activation_grad(pre_activation[neuron_idx], layer_spec.activation_method);
        let weight_idx = neuron_idx * layer_spec.input_len + in_idx;
        grad += local_grad * weights[weight_idx];
    }
    grad_input[in_idx] = grad;
}

@compute @workgroup_size(64)
fn fully_connected_back_weights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = layer_spec.input_len * layer_spec.nb_neurons;
    if idx >= total { return; }

    let in_idx = idx % layer_spec.input_len;
    let neuron_idx = idx / layer_spec.input_len;
    let local_grad =
        grad_output[neuron_idx] * activation_grad(pre_activation[neuron_idx], layer_spec.activation_method);
    grad_weights[idx] += local_grad * fwd_input[in_idx];
}

@compute @workgroup_size(64)
fn fully_connected_back_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let neuron_idx = gid.x;
    if neuron_idx >= layer_spec.nb_neurons { return; }
    grad_bias[neuron_idx] +=
        grad_output[neuron_idx] * activation_grad(pre_activation[neuron_idx], layer_spec.activation_method);
}
