// SGD weight update.
// Bindings match create_opt_pass() in layer.rs:
//   [0] weights      — trainable weights (read_write, updated in-place)
//   [1] bias         — trainable biases  (read_write, updated in-place)
//   [2] grad_weights — weight gradients  (read)
//   [3] grad_bias    — bias gradients    (read)
//   [4] specs        — { lr: f32 } uniform

@group(0) @binding(0) var<storage, read_write> weights:      array<f32>;
@group(0) @binding(1) var<storage, read_write> bias:         array<f32>;
@group(0) @binding(2) var<storage, read>       grad_weights: array<f32>;
@group(0) @binding(3) var<storage, read>       grad_bias:    array<f32>;
@group(0) @binding(4) var<uniform>             specs:        SgdSpecs;

struct SgdSpecs {
    lr:   f32,
    _p1:  f32,
    _p2:  f32,
    _p3:  f32,
}

@compute @workgroup_size(64)
fn sgd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&weights) {
        weights[i] -= specs.lr * grad_weights[i];
    }
    if i < arrayLength(&bias) {
        bias[i] -= specs.lr * grad_bias[i];
    }
}
