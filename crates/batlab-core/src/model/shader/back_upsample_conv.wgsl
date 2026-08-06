// File purpose: WGSL compute shader implementing back upsample conv operations for model forward/backward or optimizer passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

// Bindings match UpsampleConvType::get_back_buffers_specs():
//   [0] fwd_input    — original forward input (HWC)
//   [1] weights      — forward weights (KHKWKC)
//   [2] specs        — UpsampleConvSpec
//   [3] grad_output  — incoming gradient from next layer/loss (HWK)
//   [4] grad_input   — outgoing gradient to previous layer (HWC)
//   [5] grad_weights — gradient accumulator for weights (KHKWKC)
//   [6] grad_bias    — gradient accumulator for bias (K)

@group(0) @binding(0) var<storage, read>       fwd_input:    array<f32>;
@group(0) @binding(1) var<storage, read>       weights:      array<f32>;
@group(0) @binding(2) var<uniform>             layer_spec:   UpsampleConvSpec;
@group(0) @binding(3) var<storage, read>       grad_output:  array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input:   array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_weights: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_bias:    array<f32>;

struct UpsampleConvSpec {
    nb_kernel:    u32,
    scale_factor: u32,
    padding_mode: u32,
    _pad:         u32,
    dim_kernel:   vec3<u32>,
    dim_input:    vec3<u32>,
    dim_output:   vec3<u32>,
}

fn upsampled_height() -> u32 {
    return layer_spec.dim_input.x * layer_spec.scale_factor;
}

fn upsampled_width() -> u32 {
    return layer_spec.dim_input.y * layer_spec.scale_factor;
}

fn pad_y() -> i32 {
    if layer_spec.padding_mode == 1u {
        return i32(layer_spec.dim_kernel.x / 2u);
    }
    return 0;
}

fn pad_x() -> i32 {
    if layer_spec.padding_mode == 1u {
        return i32(layer_spec.dim_kernel.y / 2u);
    }
    return 0;
}

@compute @workgroup_size(64)
fn upsample_conv_back_input(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let IH = layer_spec.dim_input.x;
    let IW = layer_spec.dim_input.y;
    let IC = layer_spec.dim_input.z;
    let in_len = IH * IW * IC;
    if idx >= arrayLength(&grad_input) { return; }

    let sample = idx / in_len;
    let local  = idx % in_len;

    let iz = local % IC;
    let ix = (local / IC) % IW;
    let iy = local / (IC * IW);

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let K = layer_spec.dim_output.z;
    let go_sample = sample * OH * OW * K;
    let KH = layer_spec.dim_kernel.x;
    let KW = layer_spec.dim_kernel.y;
    let scale = layer_spec.scale_factor;
    let up_y_min = iy * scale;
    let up_y_max = up_y_min + scale;
    let up_x_min = ix * scale;
    let up_x_max = up_x_min + scale;
    let up_h = i32(upsampled_height());
    let up_w = i32(upsampled_width());

    var g: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        for (var oy: u32 = 0u; oy < OH; oy++) {
            for (var ox: u32 = 0u; ox < OW; ox++) {
                let go_i = go_sample + oy * OW * K + ox * K + k;
                for (var ky: u32 = 0u; ky < KH; ky++) {
                    for (var kx: u32 = 0u; kx < KW; kx++) {
                        let up_y = i32(oy) + i32(ky) - pad_y();
                        let up_x = i32(ox) + i32(kx) - pad_x();
                        if up_y < 0 || up_y >= up_h || up_x < 0 || up_x >= up_w {
                            continue;
                        }
                        let up_y_u = u32(up_y);
                        let up_x_u = u32(up_x);
                        if up_y_u < up_y_min || up_y_u >= up_y_max || up_x_u < up_x_min || up_x_u >= up_x_max {
                            continue;
                        }
                        let w_i = k * KH * KW * IC + ky * KW * IC + kx * IC + iz;
                        g += grad_output[go_i] * weights[w_i];
                    }
                }
            }
        }
    }
    grad_input[idx] = g;
}

@compute @workgroup_size(64)
// One thread per weight, whatever the batch: `grad_weights` is a parameter, so
// the batch is the outer loop of its reduction, not an axis of its grid.
fn upsample_conv_back_weights(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let K = layer_spec.dim_output.z;
    let KH = layer_spec.dim_kernel.x;
    let KW = layer_spec.dim_kernel.y;
    let IC = layer_spec.dim_input.z;
    if idx >= K * KH * KW * IC { return; }

    let kz = idx % IC;
    let kx = (idx / IC) % KW;
    let ky = (idx / (IC * KW)) % KH;
    let k = idx / (IC * KW * KH);

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let IW = layer_spec.dim_input.y;
    let scale = layer_spec.scale_factor;
    let up_h = i32(upsampled_height());
    let up_w = i32(upsampled_width());
    let out_len = OH * OW * K;
    let in_len = layer_spec.dim_input.x * IW * IC;
    let batch = arrayLength(&grad_output) / out_len;

    var g: f32 = 0.0;
    for (var b: u32 = 0u; b < batch; b++) {
        for (var oy: u32 = 0u; oy < OH; oy++) {
            for (var ox: u32 = 0u; ox < OW; ox++) {
                let up_y = i32(oy) + i32(ky) - pad_y();
                let up_x = i32(ox) + i32(kx) - pad_x();
                if up_y < 0 || up_y >= up_h || up_x < 0 || up_x >= up_w {
                    continue;
                }
                let iy = u32(up_y) / scale;
                let ix = u32(up_x) / scale;
                let in_i = b * in_len + iy * IW * IC + ix * IC + kz;
                let go_i = b * out_len + oy * OW * K + ox * K + k;
                g += grad_output[go_i] * fwd_input[in_i];
            }
        }
    }
    grad_weights[idx] += g;
}

@compute @workgroup_size(64)
fn upsample_conv_back_bias(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let k = gid.y * nwg.x * 64u + gid.x;
    let K = layer_spec.dim_output.z;
    if k >= K { return; }

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let out_len = OH * OW * K;
    let batch = arrayLength(&grad_output) / out_len;

    var g: f32 = 0.0;
    for (var b: u32 = 0u; b < batch; b++) {
        for (var oy: u32 = 0u; oy < OH; oy++) {
            for (var ox: u32 = 0u; ox < OW; ox++) {
                g += grad_output[b * out_len + oy * OW * K + ox * K + k];
            }
        }
    }
    grad_bias[k] += g;
}
