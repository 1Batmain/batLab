// File purpose: WGSL compute shader implementing convolution operations for model forward/backward or optimizer passes.

// Bindings match ConvolutionType::get_buffers_specs():
//   [0] input   — HWC layout: index = iy*W*C + ix*C + iz
//   [1] weights — KHKWKC layout: index = k*KH*KW*KC + ky*KW*KC + kx*KC + kz
//   [2] bias    — K layout: index = k
//   [3] specs   — ConvolutionUniform
//   [4] output  — HWK layout: index = oy*OW*K + ox*K + k

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       weights:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias:       array<f32>;
@group(0) @binding(3) var<uniform>             layer_spec: ConvSpec;
@group(0) @binding(4) var<storage, read_write> output:     array<f32>;

struct ConvSpec {
    nb_kernel:    u32,
    stride:       u32,
    padding_mode: u32, // 0 = Valid, 1 = Same
    _pad:         u32,
    dim_kernel:   vec3<u32>,
    dim_input:    vec3<u32>,
    dim_output:   vec3<u32>,
}

// In Same mode the receptive field is centered on the output position, so the
// kernel window starts KH/2 (resp. KW/2) before it. Positions falling outside
// the input contribute zero (explicit zero-padding — relying on WGSL access
// robustness would clamp to the last element instead).
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let K  = layer_spec.dim_output.z;
    if idx >= OH * OW * K { return; }

    let k  = idx % K;
    let ox = (idx / K) % OW;
    let oy = idx / (K * OW);

    let IH = layer_spec.dim_input.x;
    let IW = layer_spec.dim_input.y;
    let IC = layer_spec.dim_input.z;
    let KH = layer_spec.dim_kernel.x;
    let KW = layer_spec.dim_kernel.y;
    let s  = layer_spec.stride;

    var sum: f32 = bias[k];
    for (var ky: u32 = 0u; ky < KH; ky++) {
        for (var kx: u32 = 0u; kx < KW; kx++) {
            let sy = i32(oy * s) + i32(ky) - pad_y();
            let sx = i32(ox * s) + i32(kx) - pad_x();
            if sy < 0 || sy >= i32(IH) || sx < 0 || sx >= i32(IW) {
                continue; // zero padding
            }
            let iy = u32(sy);
            let ix = u32(sx);
            for (var kz: u32 = 0u; kz < IC; kz++) {
                let in_i  = iy * IW * IC + ix * IC + kz;
                let w_i   = k * KH * KW * IC + ky * KW * IC + kx * IC + kz;
                sum += input[in_i] * weights[w_i];
            }
        }
    }
    output[idx] = sum;
}

