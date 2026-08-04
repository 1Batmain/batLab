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

// Must mirror convolution.wgsl exactly: the forward maps
// (oy,ox,ky,kx) -> iy = oy*s + ky - pad_y, ix = ox*s + kx - pad_x,
// with out-of-bounds taps contributing zero.
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

// ---------------------------------------------------------------------------
// Pass 1 — grad_input
//   grad_input[iy][ix][iz] = Σ_{k,ky,kx} grad_output[oy][ox][k] * weights[k][ky][kx][iz]
//   inverting the forward map: oy = (iy + pad_y - ky)/s, ox = (ix + pad_x - kx)/s
//   (only non-negative, divisible, in-range positions contribute)
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
                let sy = i32(iy) + pad_y() - i32(ky);
                let sx = i32(ix) + pad_x() - i32(kx);
                if sy < 0 || sx < 0 { continue; }
                let dy = u32(sy);
                let dx = u32(sx);
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
// Cooperative reduction layout, shared by passes 2 and 3.
//
// Both are sums over the OH*OW output positions, one sum per output element
// (a weight, resp. a kernel's bias). The naive version gave each sum a single
// thread, which left conv1's bias pass running 16 threads over 1024 positions.
//
// Instead a workgroup of 64 threads is split into `slots` independent sums of
// `lanes` threads each (lanes * slots == 64). Each lane walks the position
// axis in strides of `lanes`, then the lanes of a slot are tree-reduced.
//
// `lanes` is derived from the shape alone — no new uniform field, so the
// legacy fixtures keep binding the very same uniform buffer. The Rust side
// mirrors this exact function to size the dispatch
// (`ConvolutionType::reduction_lanes`); a `conv_reduction_lanes_matches_shader`
// test pins the two together.
const WG_SIZE: u32 = 64u;
var<workgroup> partial: array<f32, WG_SIZE>;

// Largest power of two <= 64 that still leaves at least 16 positions per lane,
// so the tree reduction never costs more than the sum it parallelises.
fn reduction_lanes(positions: u32) -> u32 {
    var lanes: u32 = 1u;
    loop {
        if lanes >= WG_SIZE { break; }
        if positions / (lanes * 2u) < 16u { break; }
        lanes = lanes * 2u;
    }
    return lanes;
}

// Tree-reduce `partial` within each slot. `tid == lane * slots + slot`, so the
// partner of a lane sits `stride * slots` further along. The barriers sit in
// uniform control flow: `stride` and `slots` are the same for every invocation.
fn reduce_slot(tid: u32, lane: u32, lanes: u32, slots: u32) {
    workgroupBarrier();
    var stride: u32 = lanes / 2u;
    loop {
        if stride == 0u { break; }
        if lane < stride {
            partial[tid] += partial[tid + stride * slots];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

// ---------------------------------------------------------------------------
// Pass 2 — grad_weights
//   grad_weights[k][ky][kx][kz] += Σ_{oy,ox} grad_output[oy][ox][k] * fwd_input[oy*s+ky][ox*s+kx][kz]
//
// One slot per weight element. Consecutive slots are consecutive `kz`, so the
// threads of a slot-group read consecutive `fwd_input` addresses and share the
// same `grad_output` value.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn conv_back_weights(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let K   = layer_spec.dim_output.z;
    let KH  = layer_spec.dim_kernel.x;
    let KW  = layer_spec.dim_kernel.y;
    let IC  = layer_spec.dim_input.z;
    let total = K * KH * KW * IC;

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let IH = layer_spec.dim_input.x;
    let IW = layer_spec.dim_input.y;
    let s  = layer_spec.stride;
    let positions = OH * OW;

    let lanes = reduction_lanes(positions);
    let slots = WG_SIZE / lanes;
    let tid   = lid.x;
    let slot  = tid % slots;
    let lane  = tid / slots;
    let idx   = wid.x * slots + slot;

    var g: f32 = 0.0;
    if idx < total {
        let kz = idx % IC;
        let kx = (idx / IC) % KW;
        let ky = (idx / (IC * KW)) % KH;
        let k  = idx / (IC * KW * KH);

        // Same tap geometry and same zero-padding rule as the forward pass.
        for (var p: u32 = lane; p < positions; p += lanes) {
            let oy = p / OW;
            let ox = p % OW;
            let sy = i32(oy * s) + i32(ky) - pad_y();
            let sx = i32(ox * s) + i32(kx) - pad_x();
            if sy < 0 || sy >= i32(IH) || sx < 0 || sx >= i32(IW) {
                continue; // padded position — contributes zero to the gradient
            }
            let in_i = u32(sy) * IW * IC + u32(sx) * IC + kz;
            let go_i = oy * OW * K + ox * K + k;
            g += grad_output[go_i] * fwd_input[in_i];
        }
    }

    partial[tid] = g;
    reduce_slot(tid, lane, lanes, slots);
    if lane == 0u && idx < total {
        grad_weights[idx] += partial[slot];
    }
}

// ---------------------------------------------------------------------------
// Pass 3 — grad_bias
//   grad_bias[k] += Σ_{oy,ox} grad_output[oy][ox][k]
//
// One slot per kernel. Consecutive slots are consecutive `k`, which is the
// fastest-varying axis of `grad_output`.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn conv_back_bias(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let K  = layer_spec.dim_output.z;
    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let positions = OH * OW;

    let lanes = reduction_lanes(positions);
    let slots = WG_SIZE / lanes;
    let tid   = lid.x;
    let slot  = tid % slots;
    let lane  = tid / slots;
    let k     = wid.x * slots + slot;

    var g: f32 = 0.0;
    if k < K {
        for (var p: u32 = lane; p < positions; p += lanes) {
            let oy = p / OW;
            let ox = p % OW;
            g += grad_output[oy * OW * K + ox * K + k];
        }
    }

    partial[tid] = g;
    reduce_slot(tid, lane, lanes, slots);
    if lane == 0u && k < K {
        grad_bias[k] += partial[slot];
    }
}
