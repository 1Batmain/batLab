// Upsample-conv backward. 2-D dispatch grid past WebGPU's 65 535-per-dim limit
// (`dispatch_grid` in layer.rs). Bindings match UpsampleConvType::get_back_buffers_specs():
// [0] fwd_input (HWC), [1] weights (KHKWKC), [2] specs, [3] grad_output (HWK), and
// the accumulators [4] grad_input, [5] grad_weights, [6] grad_bias.

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
    // Cooperating threads per grad_bias sum, in the old padding word (layout pinned).
    bias_lanes:   u32,
    dim_kernel:   vec3<u32>,
    dim_input:    vec3<u32>,
    dim_output:   vec3<u32>,
}

// Cooperative reduction, same as back_convolution.wgsl: a 64-thread workgroup
// splits into `slots` sums of `lanes` threads (lanes*slots == 64), each lane
// striding by `lanes`, then tree-reduced. `lanes` from the uniform, so the Rust
// dispatch and this split cannot drift.
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

    // The map is INVERTED here, not scanned. This kernel used to walk the ENTIRE
    // output map and `continue` on positions outside this thread's `scale × scale`
    // window — a ratio of `OH·OW / scale²` useless iterations that GREW with the
    // layer's resolution (256:1 on an XL UpsampleConv, 4 140 of a 4 400 ms step,
    // `GPU_PROFILE.md`). Instead `up_y ∈ [iy·scale, (iy+1)·scale)` is known in
    // closed form (so the bounds tests are gone), and `oy = up_y + pad_y - ky`
    // inverts the forward map — the same inversion `conv_back_input` does, `k`
    // innermost (tap geometry reused, `grad_output` walked contiguously).
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

// One thread per weight whatever the batch: `grad_weights` is a parameter, so the
// batch is the reduction's outer loop, not a grid axis. The inner loop no longer
// recomputes per position the `pad_*()` calls, the bounds tests, and two integer
// divisions by `scale`: `up_y`/`up_x` are monotonic so the in-map taps are a
// contiguous interval (ends computed once), `iy`/`ix` advance by counters, and both
// addresses are incremented. Unlike `conv_back_weights`, this visits the SAME taps
// in the SAME order, so the sum is BIT-IDENTICAL — the test asserts that, not a tolerance.
@compute @workgroup_size(64)
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
    let row_in = IW * IC;

    // The two intervals, in closed form. `up_y = oy + ky - pad_y` lies in
    // [0, up_h) exactly for oy in [oy_lo, oy_hi); same for the columns.
    let dy = i32(ky) - pad_y();
    let dx = i32(kx) - pad_x();
    let oy_lo = max(0, -dy);
    let oy_hi = min(i32(OH), up_h - dy);
    let ox_lo = max(0, -dx);
    let ox_hi = min(i32(OW), up_w - dx);

    var g: f32 = 0.0;
    if oy_hi > oy_lo && ox_hi > ox_lo {
        let oy0 = u32(oy_lo);
        let oy1 = u32(oy_hi);
        let ox0 = u32(ox_lo);
        let ox1 = u32(ox_hi);

        // Where the two counters start: the upsampled coordinate of the first
        // position of each interval, split into "which input element" and "how
        // far into its scale-wide window".
        let uy_lo = u32(oy_lo + dy);
        let ux_lo = u32(ox_lo + dx);
        let iy0 = uy_lo / scale;
        let cy0 = uy_lo % scale;
        let ix0 = ux_lo / scale;
        let cx0 = ux_lo % scale;

        for (var b: u32 = 0u; b < batch; b++) {
            let in_sample = b * in_len;
            let go_sample = b * out_len;
            var iy = iy0;
            var cy = cy0;
            for (var oy: u32 = oy0; oy < oy1; oy++) {
                let in_row = in_sample + iy * row_in + kz;
                var gi = go_sample + (oy * OW + ox0) * K + k;
                var ii = in_row + ix0 * IC;
                var cx = cx0;
                for (var ox: u32 = ox0; ox < ox1; ox++) {
                    g += grad_output[gi] * fwd_input[ii];
                    gi += K;
                    cx += 1u;
                    if cx == scale {
                        cx = 0u;
                        ii += IC;
                    }
                }
                cy += 1u;
                if cy == scale {
                    cy = 0u;
                    iy += 1u;
                }
            }
        }
    }
    grad_weights[idx] += g;
}

// grad_bias[k] += Σ_{b,oy,ox} grad_output[b][oy][ox][k]. One slot per kernel
// (consecutive `k`, the fastest axis). Giving each bias one thread left 48 kernels
// on ONE workgroup (48 threads, 9.8 ms), latency-bound; the sum is unchanged, only
// the thread count. Position axis walked flat, which folds `batch` into one reduction.
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
