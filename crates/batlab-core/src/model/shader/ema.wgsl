// File purpose: WGSL compute shader keeping an exponential moving average of the
// trainable weights, one pass after the optimiser updated them.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.
//
// Bindings match create_ema_pass() in layer.rs:
//   [0] weights     — trainable weights, as the optimiser just left them (read)
//   [1] bias        — trainable biases                                   (read)
//   [2] ema_weights — the shadow copy (read_write, persistent)
//   [3] ema_bias    — the shadow copy (read_write, persistent)
//   [4] specs       — EmaSpecs uniform
//
// The decay in the uniform is the EFFECTIVE decay of this step: the warmup ramp
// min(decay, (1+t)/(10+t)) is applied on the CPU, where the step counter lives,
// exactly as Adam's bias corrections are. The shader is the recurrence and
// nothing else.

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
