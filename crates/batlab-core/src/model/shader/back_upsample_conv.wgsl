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
    // Cooperating threads per grad_bias sum. Sits in the word that used to be
    // padding, so the uniform's size and layout are unchanged and the legacy
    // fixture still binds the very same buffer.
    bias_lanes:   u32,
    dim_kernel:   vec3<u32>,
    dim_input:    vec3<u32>,
    dim_output:   vec3<u32>,
}

// ---------------------------------------------------------------------------
// Cooperative reduction layout — the same one back_convolution.wgsl uses, and
// for the same reason. A workgroup of 64 threads is split into `slots`
// independent sums of `lanes` threads each (lanes * slots == 64); each lane
// walks the position axis in strides of `lanes`, then the lanes of a slot are
// tree-reduced. `lanes` comes from the uniform, so the dispatch on the Rust
// side and the split in here cannot drift apart.
// ---------------------------------------------------------------------------
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

    // The map is INVERTED here, not scanned.
    //
    // The forward writes `output[oy,ox,k] += input[up_y/scale, up_x/scale, kz]
    // * w[k,ky,kx,kz]` with `up_y = oy + ky - pad_y`. This kernel used to walk
    // the ENTIRE output map — `k × OH × OW × KH × KW` iterations — and `continue`
    // on the ones that did not land in this thread's `scale × scale` window.
    // On `Color_Diffusion_XL`'s second UpsampleConv (16×16×96 → 32×32×48) that
    // is 48·32·32·9 = 442 368 iterations per thread to accumulate 48·2·2·9 =
    // 1728 products: **256 useless iterations for every useful one**, and the
    // ratio is `OH·OW / scale²`, so it grows with the resolution of the layer.
    // Measured cost of the two such passes: 4 140 ms of a 4 400 ms training
    // step (`GPU_PROFILE.md`).
    //
    // The window is known in closed form. `up_y` ranges over exactly
    // `[iy·scale, (iy+1)·scale)` — inside `[0, up_h)` by construction, which is
    // why the two bounds tests on `up_y`/`up_x` are gone rather than merely
    // moved — and `oy = up_y + pad_y - ky` is the inverse of the forward map,
    // the same inversion `conv_back_input` performs (and the same one the f64
    // oracle validates instead of repeating).
    //
    // `k` innermost, as in `conv_back_input` and for the same two reasons: the
    // tap geometry depends only on `(up_y, ky, up_x, kx)` and would otherwise be
    // recomputed `K` times, and `grad_output` is then walked contiguously along
    // `k`, the fastest axis of the HWK layout.
    let py = pad_y();
    let px = pad_x();
    let stride_w = KH * KW * IC;

    var g: f32 = 0.0;
    for (var up_y: u32 = up_y_min; up_y < up_y_max; up_y++) {
        for (var ky: u32 = 0u; ky < KH; ky++) {
            let sy = i32(up_y) + py - i32(ky);
            if sy < 0 || sy >= i32(OH) { continue; }
            let oy = u32(sy);
            for (var up_x: u32 = up_x_min; up_x < up_x_max; up_x++) {
                for (var kx: u32 = 0u; kx < KW; kx++) {
                    let sx = i32(up_x) + px - i32(kx);
                    if sx < 0 || sx >= i32(OW) { continue; }
                    let ox = u32(sx);

                    let go_base = go_sample + oy * OW * K + ox * K;
                    var w_i = ky * KW * IC + kx * IC + iz;
                    for (var k: u32 = 0u; k < K; k++) {
                        g += grad_output[go_base + k] * weights[w_i];
                        w_i += stride_w;
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

// grad_bias[k] += Σ_{b,oy,ox} grad_output[b][oy][ox][k]
//
// One slot per kernel. Consecutive slots are consecutive `k`, which is the
// fastest-varying axis of `grad_output`.
//
// This pass used to give each bias a single thread. With 48 kernels that is
// `48.div_ceil(64)` = ONE workgroup — 48 threads walking 32 768 positions each,
// 9,8 ms to read 6,3 Mio. Nothing about the arithmetic was wrong: 48 threads
// simply cannot cover the memory latency of any GPU. The sum is unchanged; only
// how many threads carry it is.
//
// The position axis is walked flat — `p / positions` is the sample and
// `p % positions` the position within it — which is what turns `batch`
// sequential accumulations into one tree reduction.
@compute @workgroup_size(64)
fn upsample_conv_back_bias(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let K = layer_spec.dim_output.z;
    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let positions = OH * OW;
    let batch = arrayLength(&grad_output) / (positions * K);
    let work = positions * batch;

    let lanes = layer_spec.bias_lanes;
    let slots = WG_SIZE / lanes;
    let tid   = lid.x;
    let slot  = tid % slots;
    let lane  = tid / slots;
    let k     = (wid.y * nwg.x + wid.x) * slots + slot;

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
