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
//
// # Why this loop is a nest and not a flat walk
//
// This pass contracts exactly as many products as the forward does — the same
// triple `(b, oy, ox)` against the same `(ky, kx, kz)` — and took **4,1× longer**
// (`GPU_PROFILE.md` §3.2), which that report read as a memory-hierarchy
// handicap. It is not, or not only: the flat walk paid **four integer
// divisions per product**. `p / positions`, `p % positions`, `pos / OW`,
// `pos % OW` — all by values only known at runtime, so none of them folds — for
// one multiply-add. The forward's inner loop, for comparison, is two loads and
// an FMA.
//
// So the position axis is walked as what it is: rows, then columns.
//
//   - A **row** is one `(sample, oy)`. The lane split moves there — one
//     div/mod per row instead of four per position, and a lane still strides
//     across sample boundaries, which is what turns `batch` sequential
//     accumulations into one tree reduction.
//   - Within a row, `ox` runs over a **closed-form interval**. `sx = ox*s + kx
//     - pad_x` is monotonic in `ox`, so the positions whose tap falls in the
//     input are contiguous: computing the two ends once replaces a bounds test
//     taken `OW` times per row. The kernel row test on `sy` stays, once per
//     row, exactly as `convolution.wgsl` does it.
//   - Both addresses are then **incremented**, never recomputed: `grad_output`
//     advances by `K` per `ox` (it is HWK) and `fwd_input` by `s * IC`.
//
// The inner loop is left with two loads, an FMA and two adds — the forward's
// loop. Same taps, same padding rule, same products.
//
// The **order** does change: a lane used to visit positions `lane, lane+lanes,
// …` across the flat axis and now visits whole rows. Float addition is not
// associative, so this is a reassociation, and the equivalence tests treat it
// as one (tolerance against the legacy kernel, an f64 oracle to arbitrate, and
// a "never less accurate than what it replaces" clause) rather than asserting
// bit-identity, which would be false.
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
