// File purpose: WGSL compute shader implementing the Adam weight update for the optimizer pass.

// Adam (Kingma & Ba 2015) weight update.
// Bindings match create_opt_pass() in layer.rs:
//   [0] weights      — trainable weights (read_write, updated in-place)
//   [1] bias         — trainable biases  (read_write, updated in-place)
//   [2] grad_weights — accumulated weight gradients (read)
//   [3] grad_bias    — accumulated bias gradients   (read)
//   [4] specs        — AdamSpecs uniform
//   [5] m_weights    — first  moment, weights (read_write, persistent)
//   [6] v_weights    — second moment, weights (read_write, persistent)
//   [7] m_bias       — first  moment, bias    (read_write, persistent)
//   [8] v_bias       — second moment, bias    (read_write, persistent)

@group(0) @binding(0) var<storage, read_write> weights:      array<f32>;
@group(0) @binding(1) var<storage, read_write> bias:         array<f32>;
@group(0) @binding(2) var<storage, read>       grad_weights: array<f32>;
@group(0) @binding(3) var<storage, read>       grad_bias:    array<f32>;
@group(0) @binding(4) var<uniform>             specs:        AdamSpecs;
@group(0) @binding(5) var<storage, read_write> m_weights:    array<f32>;
@group(0) @binding(6) var<storage, read_write> v_weights:    array<f32>;
@group(0) @binding(7) var<storage, read_write> m_bias:       array<f32>;
@group(0) @binding(8) var<storage, read_write> v_bias:       array<f32>;

struct AdamSpecs {
    lr:    f32,
    beta1: f32,
    beta2: f32,
    eps:   f32,
    // The accumulated gradient is a SUM over the batch; Adam's update is
    // scale-invariant in g, so the batch mean has to be applied here and not
    // folded into lr the way the SGD pass does it.
    grad_scale: f32,
    // 1 - beta1^t and 1 - beta2^t, computed on the CPU in f64 (t is the global
    // step counter of the run).
    bias_correction1: f32,
    bias_correction2: f32,
    _pad: f32,
}

@compute @workgroup_size(64)
fn adam(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i < arrayLength(&weights) {
        let g = grad_weights[i] * specs.grad_scale;
        let m = specs.beta1 * m_weights[i] + (1.0 - specs.beta1) * g;
        let v = specs.beta2 * v_weights[i] + (1.0 - specs.beta2) * g * g;
        m_weights[i] = m;
        v_weights[i] = v;
        let m_hat = m / specs.bias_correction1;
        let v_hat = v / specs.bias_correction2;
        weights[i] = weights[i] - specs.lr * m_hat / (sqrt(v_hat) + specs.eps);
    }

    if i < arrayLength(&bias) {
        let g = grad_bias[i] * specs.grad_scale;
        let m = specs.beta1 * m_bias[i] + (1.0 - specs.beta1) * g;
        let v = specs.beta2 * v_bias[i] + (1.0 - specs.beta2) * g * g;
        m_bias[i] = m;
        v_bias[i] = v;
        let m_hat = m / specs.bias_correction1;
        let v_hat = v / specs.bias_correction2;
        bias[i] = bias[i] - specs.lr * m_hat / (sqrt(v_hat) + specs.eps);
    }
}
