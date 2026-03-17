// File purpose: WGSL compute shader implementing back convolution operations for model forward/backward or optimizer passes.

// Bindings match ConvolutionType::get_back_buffers_specs():
//   [0] fwd_input    — input used in the forward pass  (HWC: iy*W*C + ix*C + iz)
//   [1] weights      — forward weights                 (KHKWKC: k*KH*KW*KC + ky*KW*KC + kx*KC + kz)
//   [2] specs        — ConvSpec uniform (shared from forward)
//   [3] grad_output  — incoming gradient from next layer/loss (HWK: oy*OW*K + ox*K + k)
//   [4] grad_input   — outgoing gradient to previous layer    (HWC)
//   [5] grad_weights — weight gradient accumulator            (KHKWKC)
//   [6] grad_bias    — bias gradient accumulator              (K)
//
// Three separate compute passes prevent write races:
//   conv_back_input   dispatched over input  elements
//   conv_back_weights dispatched over weight elements
//   conv_back_bias    dispatched over kernel count

@group(0) @binding(0) var<storage, read>       fwd_input:    array<f32>;
@group(0) @binding(1) var<storage, read>       weights:      array<f32>;
@group(0) @binding(2) var<uniform>             layer_spec:   ConvSpec;
@group(0) @binding(3) var<storage, read>       grad_output:  array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input:   array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_weights: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_bias:    array<f32>;

struct ConvSpec {
    nb_kernel:    u32,
    stride:       u32,
    padding_mode: u32,
    _pad:         u32,
    dim_kernel:   vec3<u32>,
    dim_input:    vec3<u32>,
    dim_output:   vec3<u32>,
}

// ---------------------------------------------------------------------------
// Pass 1 — grad_input
//   grad_input[iy][ix][iz] = Σ_{k,ky,kx} grad_output[oy][ox][k] * weights[k][ky][kx][iz]
//   where oy=(iy-ky)/s, ox=(ix-kx)/s  (only valid integer positions)
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn conv_back_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let IH = layer_spec.dim_input.x;
    let IW = layer_spec.dim_input.y;
    let IC = layer_spec.dim_input.z;
    if idx >= IH * IW * IC { return; }

    let iz = idx % IC;
    let ix = (idx / IC) % IW;
    let iy = idx / (IC * IW);

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let K  = layer_spec.dim_output.z;
    let KH = layer_spec.dim_kernel.x;
    let KW = layer_spec.dim_kernel.y;
    let s  = layer_spec.stride;

    var g: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        for (var ky: u32 = 0u; ky < KH; ky++) {
            for (var kx: u32 = 0u; kx < KW; kx++) {
                if iy < ky || ix < kx { continue; }
                let dy = iy - ky;
                let dx = ix - kx;
                if dy % s != 0u || dx % s != 0u { continue; }
                let oy = dy / s;
                let ox = dx / s;
                if oy >= OH || ox >= OW { continue; }
                let go_i = oy * OW * K + ox * K + k;
                let w_i  = k * KH * KW * IC + ky * KW * IC + kx * IC + iz;
                g += grad_output[go_i] * weights[w_i];
            }
        }
    }
    grad_input[idx] = g;
}

// ---------------------------------------------------------------------------
// Pass 2 — grad_weights
//   grad_weights[k][ky][kx][kz] += Σ_{oy,ox} grad_output[oy][ox][k] * fwd_input[oy*s+ky][ox*s+kx][kz]
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn conv_back_weights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let K   = layer_spec.dim_output.z;
    let KH  = layer_spec.dim_kernel.x;
    let KW  = layer_spec.dim_kernel.y;
    let IC  = layer_spec.dim_input.z;
    if idx >= K * KH * KW * IC { return; }

    let kz = idx % IC;
    let kx = (idx / IC) % KW;
    let ky = (idx / (IC * KW)) % KH;
    let k  = idx / (IC * KW * KH);

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let IW = layer_spec.dim_input.y;
    let s  = layer_spec.stride;

    var g: f32 = 0.0;
    for (var oy: u32 = 0u; oy < OH; oy++) {
        for (var ox: u32 = 0u; ox < OW; ox++) {
            let iy     = oy * s + ky;
            let ix     = ox * s + kx;
            let in_i   = iy * IW * IC + ix * IC + kz;
            let go_i   = oy * OW * K + ox * K + k;
            g += grad_output[go_i] * fwd_input[in_i];
        }
    }
    grad_weights[idx] += g;
}

// ---------------------------------------------------------------------------
// Pass 3 — grad_bias
//   grad_bias[k] += Σ_{oy,ox} grad_output[oy][ox][k]
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn conv_back_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    let K  = layer_spec.dim_output.z;
    if k >= K { return; }

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;

    var g: f32 = 0.0;
    for (var oy: u32 = 0u; oy < OH; oy++) {
        for (var ox: u32 = 0u; ox < OW; ox++) {
            g += grad_output[oy * OW * K + ox * K + k];
        }
    }
    grad_bias[k] += g;
}
