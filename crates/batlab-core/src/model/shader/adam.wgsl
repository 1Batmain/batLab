// Adam (Kingma & Ba 2015) weight update. 2-D dispatch grid past WebGPU's
// 65 535-per-dim limit, so the thread index comes from `num_workgroups`
// (`dispatch_grid` in layer.rs). Bindings match create_opt_pass(); m_*/v_* are
// the persistent first/second moments.
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
    // Adam is scale-invariant in g, so the batch mean of the summed gradient
    // must be applied here, not folded into lr the way SGD does it.
    grad_scale: f32,
    // 1 - beta1^t and 1 - beta2^t, computed on the CPU in f64.
    bias_correction1: f32,
    bias_correction2: f32,
    _pad: f32,
}

@compute @workgroup_size(64)
fn adam(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;

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
