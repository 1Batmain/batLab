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

// One thread per output element, as before.
//
// Register-blocking this kernel (one thread producing a run of 2 or 4 adjacent
// output pixels, so each weight load is reused across them) was tried and is
// SLOWER on every layer of the model — see PERF_CONVOLUTION.md. The batch is
// looped sample by sample on the CPU, so a convolution here works on a single
// small tensor: conv4's forward has only 1024 output elements to begin with,
// and dividing the thread count by 4 starves the GPU faster than the saved
// bandwidth pays back. These dispatches are parallelism-bound, not
// bandwidth-bound.
//
// What is left is the free part: hoist the padding offsets and the per-tap base
// addresses out of the inner loop, and reject a kernel row that falls entirely
// in the padding once per row instead of once per column. Same taps, same
// order, same skips — bit-identical output, strictly less integer work.
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
    let py = pad_y();
    let px = pad_x();
    let kbase = k * KH * KW * IC;

    var sum: f32 = bias[k];
    for (var ky: u32 = 0u; ky < KH; ky++) {
        let sy = i32(oy * s) + i32(ky) - py;
        if sy < 0 || sy >= i32(IH) {
            continue; // whole kernel row sits in the zero padding
        }
        let row = u32(sy) * IW * IC;
        for (var kx: u32 = 0u; kx < KW; kx++) {
            let sx = i32(ox * s) + i32(kx) - px;
            if sx < 0 || sx >= i32(IW) {
                continue; // zero padding
            }
            let in_base = row + u32(sx) * IC;
            let w_base  = kbase + ky * KW * IC + kx * IC;
            for (var kz: u32 = 0u; kz < IC; kz++) {
                sum += input[in_base + kz] * weights[w_base + kz];
            }
        }
    }
    output[idx] = sum;
}

