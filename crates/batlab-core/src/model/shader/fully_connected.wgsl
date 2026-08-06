// File purpose: WGSL compute shader implementing fully connected operations for model forward/backward or optimizer passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

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
fn fully_connected(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&output) { return; }

    let sample = idx / layer_spec.nb_neurons;
    let neuron_idx = idx % layer_spec.nb_neurons;
    let in_base = sample * layer_spec.input_len;

    var sum: f32 = bias[neuron_idx];
    for (var in_idx: u32 = 0u; in_idx < layer_spec.input_len; in_idx++) {
        let weight_idx = neuron_idx * layer_spec.input_len + in_idx;
        sum += input[in_base + in_idx] * weights[weight_idx];
    }
    pre_activation[idx] = sum;
    output[idx] = apply_activation(sum, layer_spec.activation_method);
}
