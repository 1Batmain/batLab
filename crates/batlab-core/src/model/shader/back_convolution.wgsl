// Convolution backward. 2-D dispatch grid past WebGPU's 65 535-per-dim limit, so
// the thread index comes from `num_workgroups` (`dispatch_grid` in layer.rs).
//
// Bindings match ConvolutionType::get_back_buffers_specs(): [0] fwd_input (HWC:
// iy*W*C + ix*C + iz), [1] weights (KHKWKC), [2] specs, [3] grad_output (HWK), and
// the three accumulators [4] grad_input, [5] grad_weights, [6] grad_bias.
//
// Three passes avoid write races: `conv_back_input` over input elements × batch;
// `conv_back_weights`/`conv_back_bias` over parameters, NOT × batch — the same
// sum count whatever the batch, so the batch enters INSIDE the kernel as
// `batch * OH * OW` positions (one `+=` per weight per STEP). The batch is read
// as `arrayLength(&grad_output) / (OH*OW*K)`. Design: BATCH_DISPATCH_DESIGN.md §3.1.

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
    // Cooperating threads per sum for BOTH reductions: grad_weights in the low half
    // of the word, grad_bias in the high half (in the old padding word, so layout is
    // pinned). Two counts because the sum counts differ wildly (`KH*KW*IC*K` vs `K`):
    // the bias needs its own or it lands on ONE workgroup. See `reduction_lanes_for_bias`.
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

// `sx` at `ox = 0` for a given kernel column: the intercept of the forward's
// `sx = ox*s + kx - pad_x` line.
fn pad_x_signed(kx: u32) -> i32 {
    return i32(kx) - pad_x();
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

// Cooperative reduction shared by passes 2 and 3: sums over OH*OW positions, one
// sum per output element. A 64-thread workgroup is split into `slots` sums of
// `lanes` threads each (lanes*slots == 64); each lane strides the position axis by
// `lanes`, then a slot's lanes are tree-reduced. `lanes` comes from the uniform
// (`ConvolutionType::reduction_lanes`), so the Rust dispatch and this split cannot
// drift. `lanes == 1` degenerates to one thread per sum.
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
// One slot per weight; consecutive slots are consecutive `kz` (coalesced reads).
// Walked as ROWS, not a flat position axis: a flat walk paid four runtime integer
// divisions per product (none foldable) for one FMA, which read as a 4.1× memory
// handicap (`GPU_PROFILE.md` §3.2) but was largely that. Per row `(sample, oy)`,
// `ox` runs a closed-form interval (`sx` monotonic in `ox`), and both addresses are
// incremented, never recomputed — leaving the forward's own two-loads-plus-FMA loop.
// This reassociates the sum (float add is not associative), so the equivalence tests
// use a tolerance + f64 oracle, not bit-identity.
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
    let rows = OH * batch;

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

        let in_len = IH * IW * IC;
        let row_in = IW * IC;
        let step_in = s * IC;

        // `sx` at `ox = 0`. Monotonic in `ox` with slope `s > 0`, so
        // `0 <= sx < IW` is the interval [ox_lo, ox_hi).
        let sx0 = pad_x_signed(kx);
        var ox_lo: u32 = 0u;
        if sx0 < 0 {
            ox_lo = u32((-sx0 + i32(s) - 1) / i32(s));
        }
        var ox_hi: u32 = 0u;
        let last = i32(IW) - 1 - sx0;
        if last >= 0 {
            ox_hi = min(OW, u32(last / i32(s)) + 1u);
        }

        let sy_at_zero = i32(ky) - pad_y();
        for (var r: u32 = lane; r < rows; r += lanes) {
            let sample = r / OH;
            let oy     = r % OH;
            let sy = i32(oy * s) + sy_at_zero;
            if sy < 0 || sy >= i32(IH) {
                continue; // whole kernel row sits in the zero padding
            }
            var gi = (sample * positions + oy * OW + ox_lo) * K + k;
            var ii = sample * in_len + u32(sy) * row_in
                   + u32(i32(ox_lo * s) + sx0) * IC + kz;
            for (var ox: u32 = ox_lo; ox < ox_hi; ox++) {
                g += grad_output[gi] * fwd_input[ii];
                gi += K;
                ii += step_in;
            }
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
