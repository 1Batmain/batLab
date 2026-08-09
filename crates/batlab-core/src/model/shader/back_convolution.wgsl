// File purpose: WGSL compute shader implementing back convolution operations for model forward/backward or optimizer passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

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
//   conv_back_input   dispatched over input  elements  x batch
//   conv_back_weights dispatched over weight elements   (NOT x batch)
//   conv_back_bias    dispatched over kernel count      (NOT x batch)
//
// The batch axis splits these three in two, and the split IS the design (see
// docs/reports/BATCH_DISPATCH_DESIGN.md §3.1). `conv_back_input` writes one
// activation per thread, so it grows with the batch like any elementwise pass.
// The other two write *parameters*: there are exactly as many sums as there are
// weights whatever the batch, so their grid is unchanged and the batch enters
// INSIDE the kernel, as `batch * OH * OW` positions to reduce instead of
// `OH * OW`. One `+=` per weight and per step, instead of one per weight and
// per sample.
//
// Neither kernel is told the batch: it is `arrayLength(&grad_output) / (OH*OW*K)`,
// read off the buffer that carries it.

@group(0) @binding(0) var<storage, read>       fwd_input:    array<f32>;
@group(0) @binding(1) var<storage, read>       weights:      array<f32>;
@group(0) @binding(2) var<uniform>             layer_spec:   ConvSpec;
@group(0) @binding(3) var<storage, read>       grad_output:  array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input:   array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_weights: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_bias:    array<f32>;

struct ConvSpec {
    nb_kernel:       u32,
    stride:          u32,
    padding_mode:    u32,
    // Cooperating threads per sum, for BOTH reductions: grad_weights in the low
    // half of the word, grad_bias in the high half. Sits in the word that used
    // to be padding, so the uniform's size and layout are unchanged and the
    // legacy fixtures still bind the very same buffer.
    //
    // Two counts and not one because the two reductions have wildly different
    // sum counts: `KH*KW*IC*K` weights against `K` biases. Giving the bias the
    // weights' split left it on ONE workgroup — see `reduction_lanes_for_bias`.
    reduction_lanes: u32,
    dim_kernel:      vec3<u32>,
    dim_input:       vec3<u32>,
    dim_output:      vec3<u32>,
}

fn weight_lanes() -> u32 {
    return layer_spec.reduction_lanes & 0xffffu;
}

fn bias_lanes() -> u32 {
    return layer_spec.reduction_lanes >> 16u;
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
fn conv_back_input(
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
    let K  = layer_spec.dim_output.z;
    let go_sample = sample * OH * OW * K;
    let KH = layer_spec.dim_kernel.x;
    let KW = layer_spec.dim_kernel.y;
    let s  = layer_spec.stride;

    // The tap geometry — the inverted forward map, its divisibility test and
    // its bounds checks — depends only on (ky, kx), never on k. The original
    // loop nest had k outermost and so recomputed all of it K times over: 288
    // integer divisions and modulos per thread on the 3x3/32-kernel layers,
    // where 9 suffice. Hoisting the k loop inward leaves a tight dot product.
    //
    // `grad_output` is then walked contiguously in k (it is the fastest-varying
    // axis of the HWK layout), and neighbouring threads — neighbouring `iz` —
    // read neighbouring `weights`, so both accesses coalesce.
    let py = pad_y();
    let px = pad_x();
    let stride_w = KH * KW * IC;

    var g: f32 = 0.0;
    for (var ky: u32 = 0u; ky < KH; ky++) {
        for (var kx: u32 = 0u; kx < KW; kx++) {
            let sy = i32(iy) + py - i32(ky);
            let sx = i32(ix) + px - i32(kx);
            if sy < 0 || sx < 0 { continue; }
            let dy = u32(sy);
            let dx = u32(sx);
            if dy % s != 0u || dx % s != 0u { continue; }
            let oy = dy / s;
            let ox = dx / s;
            if oy >= OH || ox >= OW { continue; }

            let go_base = go_sample + oy * OW * K + ox * K;
            var w_i = ky * KW * IC + kx * IC + iz;
            for (var k: u32 = 0u; k < K; k++) {
                g += grad_output[go_base + k] * weights[w_i];
                w_i += stride_w;
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
// `lanes` comes from the uniform (`ConvolutionType::reduction_lanes` picks it,
// calibrated by `bench_conv_reduction_lanes`), so the dispatch on the Rust side
// and the split in here cannot drift apart. `lanes == 1` degenerates to exactly
// one thread per sum — the naive scheme — which is the right choice for the
// layers that already have thousands of independent sums.
const WG_SIZE: u32 = 64u;
var<workgroup> partial: array<f32, WG_SIZE>;

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
    @builtin(num_workgroups) nwg: vec3<u32>,
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
    // The batch, straight off the buffer that carries it.
    let batch = arrayLength(&grad_output) / (OH * OW * K);
    let work = positions * batch;

    let lanes = weight_lanes();
    let slots = WG_SIZE / lanes;
    let tid   = lid.x;
    let slot  = tid % slots;
    let lane  = tid / slots;
    let idx   = (wid.y * nwg.x + wid.x) * slots + slot;

    var g: f32 = 0.0;
    if idx < total {
        let kz = idx % IC;
        let kx = (idx / IC) % KW;
        let ky = (idx / (IC * KW)) % KH;
        let k  = idx / (IC * KW * KH);

        // Same tap geometry and same zero-padding rule as the forward pass.
        // The position axis now runs over the whole batch: `p / positions` is
        // the sample, `p % positions` the output position within it. A lane
        // therefore strides across sample boundaries, which is what turns B
        // sequential accumulations into one tree reduction.
        for (var p: u32 = lane; p < work; p += lanes) {
            let sample = p / positions;
            let pos    = p % positions;
            let oy = pos / OW;
            let ox = pos % OW;
            let sy = i32(oy * s) + i32(ky) - pad_y();
            let sx = i32(ox * s) + i32(kx) - pad_x();
            if sy < 0 || sy >= i32(IH) || sx < 0 || sx >= i32(IW) {
                continue; // padded position — contributes zero to the gradient
            }
            let in_i = sample * IH * IW * IC + u32(sy) * IW * IC + u32(sx) * IC + kz;
            let go_i = sample * OH * OW * K + oy * OW * K + ox * K + k;
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
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let K  = layer_spec.dim_output.z;
    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let positions = OH * OW;
    let batch = arrayLength(&grad_output) / (OH * OW * K);
    let work = positions * batch;

    let lanes = bias_lanes();
    let slots = WG_SIZE / lanes;
    let tid   = lid.x;
    let slot  = tid % slots;
    let lane  = tid / slots;
    let k     = (wid.y * nwg.x + wid.x) * slots + slot;

    // `sample*OH*OW*K + oy*OW*K + ox*K` is `(sample*positions + oy*OW + ox)*K`,
    // which is `p*K` — the flat position index the loop already carries. The
    // decomposition into (sample, oy, ox) and its four integer divisions only
    // existed to be reassembled into the number it started from. Same taps,
    // same order, bit-identical sum; four divisions and two multiplications per
    // element less.
    var g: f32 = 0.0;
    if k < K {
        for (var p: u32 = lane; p < work; p += lanes) {
            g += grad_output[p * K + k];
        }
    }

    partial[tid] = g;
    reduce_slot(tid, lane, lanes, slots);
    if lane == 0u && k < K {
        grad_bias[k] += partial[slot];
    }
}
