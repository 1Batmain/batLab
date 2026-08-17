// EMA of the trainable weights, one pass after the optimiser. 2-D dispatch grid
// past WebGPU's 65 535-per-dim limit, so the thread index comes from
// `num_workgroups` (`dispatch_grid` in layer.rs). Bindings match create_ema_pass().
//
// `specs.decay` is the EFFECTIVE decay for this step: the warmup ramp
// min(decay, (1+t)/(10+t)) is applied on the CPU, where the step counter lives.
// The shader is the recurrence and nothing else — do not add the ramp here.
@group(0) @binding(0) var<storage, read>       weights:     array<f32>;
@group(0) @binding(1) var<storage, read>       bias:        array<f32>;
@group(0) @binding(2) var<storage, read_write> ema_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> ema_bias:    array<f32>;
@group(0) @binding(4) var<uniform>             specs:       EmaSpecs;

struct EmaSpecs {
    decay: f32,
    _p1:   f32,
    _p2:   f32,
    _p3:   f32,
}

@compute @workgroup_size(64)
fn ema(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.y * nwg.x * 64u + gid.x;

    if i < arrayLength(&weights) {
        ema_weights[i] = specs.decay * ema_weights[i] + (1.0 - specs.decay) * weights[i];
    }

    if i < arrayLength(&bias) {
        ema_bias[i] = specs.decay * ema_bias[i] + (1.0 - specs.decay) * bias[i];
    }
}
