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
    padding_mode: u32, // 0 = Valid, 1 = Same (unused in kernel — dims already computed)
    _pad:         u32,
    dim_kernel:   vec3<u32>,
    dim_input:    vec3<u32>,
    dim_output:   vec3<u32>,
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

    let IW = layer_spec.dim_input.y;
    let IC = layer_spec.dim_input.z;
    let KH = layer_spec.dim_kernel.x;
    let KW = layer_spec.dim_kernel.y;
    let s  = layer_spec.stride;

    var sum: f32 = bias[k];
    for (var ky: u32 = 0u; ky < KH; ky++) {
        for (var kx: u32 = 0u; kx < KW; kx++) {
            for (var kz: u32 = 0u; kz < IC; kz++) {
                let iy    = oy * s + ky;
                let ix    = ox * s + kx;
                let in_i  = iy * IW * IC + ix * IC + kz;
                let w_i   = k * KH * KW * IC + ky * KW * IC + kx * IC + kz;
                sum += input[in_i] * weights[w_i];
            }
        }
    }
    output[idx] = sum;
}

