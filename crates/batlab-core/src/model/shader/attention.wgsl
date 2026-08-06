// File purpose: WGSL compute shader implementing spatial self-attention forward passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.
//
// THE LAYER IS RESIDUAL: `output = input + W_o · softmax(QᵀK / √d) · V`. The
// residual lives inside the layer on purpose — the graph stays a chain, and
// with `W_o` initialised to zero the layer starts life as the exact identity.
//
// Four sequential passes, because attention needs GLOBAL barriers a single
// kernel cannot provide: row `n`'s scores read `k[m]` and `v[m]` for every `m`,
// which other workgroups produce. wgpu inserts the barrier between compute
// passes of one encoder, so the split *is* the synchronisation:
//
//   1. attn_qkv     — q, k, v = W·x + b            (one thread per q/k/v scalar)
//   2. attn_scores  — softmax(qᵀk · scale) rows    (ONE WORKGROUP PER (sample, row))
//   3. attn_context — ctx = probs · v              (one thread per ctx scalar)
//   4. attn_out     — out = x + W_o·ctx + b_o      (one thread per output scalar)
//
// Bindings match AttentionType::get_buffers_specs():
//   [0] input   — HWC layout, N=H*W positions of C channels: index = n*C + c
//   [1] weights — the FOUR projections concatenated, C*C each, in order
//                 Wq, Wk, Wv at [i*C + c] (row i = output feature), then
//                 Wo at [c*C + i] (row c = output channel). One buffer so the
//                 optimiser sees one weight tensor, exactly like a convolution.
//   [2] bias    — 4*C: bq, bk, bv, bo back to back
//   [3] specs   — AttentionSpec
//   [4] qkv     — scratch, 3*N*C per sample: q block, k block, v block
//   [5] probs   — scratch, N*N per sample (softmax rows)
//   [6] ctx     — scratch, N*C per sample (probs·v, before W_o)
//   [7] output  — HWC, same shape as input (LAST: the model chains .last())

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       weights:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias:       array<f32>;
@group(0) @binding(3) var<uniform>             layer_spec: AttentionSpec;
@group(0) @binding(4) var<storage, read_write> qkv:        array<f32>;
@group(0) @binding(5) var<storage, read_write> probs:      array<f32>;
@group(0) @binding(6) var<storage, read_write> ctx:        array<f32>;
@group(0) @binding(7) var<storage, read_write> output:     array<f32>;

struct AttentionSpec {
    seq_len:   u32,  // N = H*W, the number of attending positions
    channels:  u32,  // C, also the head dimension d (single head)
    scale:     f32,  // 1 / sqrt(d)
    use_bias:  u32,  // 1 = add the projection biases, 0 = pure linear
    dim_input:  vec3<u32>,
    dim_output: vec3<u32>,
}

const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> partial: array<f32, WORKGROUP_SIZE>;

/// Elements of one sample's activation tensor (N*C).
fn sample_len() -> u32 {
    return layer_spec.seq_len * layer_spec.channels;
}

/// Base offset of sample `b` inside the q/k/v scratch (3 blocks of N*C).
fn qkv_sample_base(b: u32) -> u32 {
    return b * 3u * sample_len();
}

fn bias_at(index: u32) -> f32 {
    if layer_spec.use_bias == 0u {
        return 0.0;
    }
    return bias[index];
}

/// Tree reduction over the workgroup. Every invocation must reach this call
/// (the barriers are in uniform control flow) and every invocation gets the
/// total back.
fn workgroup_sum(tid: u32, value: f32) -> f32 {
    // Guards against a previous reduction's reads of partial[0] racing with
    // this one's writes.
    workgroupBarrier();
    partial[tid] = value;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            partial[tid] += partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return partial[0];
}

/// Same shape of reduction, for the softmax's max.
fn workgroup_max(tid: u32, value: f32) -> f32 {
    workgroupBarrier();
    partial[tid] = value;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            partial[tid] = max(partial[tid], partial[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return partial[0];
}

// ---------------------------------------------------------------------------
// 1. q, k, v projections
// ---------------------------------------------------------------------------
//
// One thread per scalar of the q/k/v scratch — of the whole batch. The sample
// is recovered by dividing by the per-sample length, never passed in.

@compute @workgroup_size(64)
fn attn_qkv(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&qkv) { return; }

    let c_count = layer_spec.channels;
    let nc = sample_len();
    let block = 3u * nc;

    let b = idx / block;
    let rest = idx % block;
    let which = rest / nc;       // 0 = q, 1 = k, 2 = v
    let inner = rest % nc;
    let n = inner / c_count;     // position
    let i = inner % c_count;     // output feature

    let w_row = which * c_count * c_count + i * c_count;
    let x_row = b * nc + n * c_count;

    var acc = bias_at(which * c_count + i);
    for (var c: u32 = 0u; c < c_count; c += 1u) {
        acc += weights[w_row + c] * input[x_row + c];
    }
    qkv[idx] = acc;
}

// ---------------------------------------------------------------------------
// 2. scores + softmax
// ---------------------------------------------------------------------------
//
// ONE WORKGROUP PER (SAMPLE, ROW): the softmax denominator is a reduction over
// the whole row, so the row is the unit of work, not the element.
//
// THE ROW NEVER LEAVES ITS SAMPLE. `b` is fixed for the workgroup and every
// read of q/k/v is offset into that sample's slice, so position `n` of image 3
// cannot attend to position `m` of image 4. Batching two images together must
// not change either one's output — that is the whole contract of the batch
// axis, and it is the specific failure mode this layer is exposed to.
//
// Trailing workgroups from the 2-D grid split are clamped onto the last real
// row rather than returned early: they then recompute that row identically
// (the pass is idempotent), which keeps every barrier in uniform control flow.

@compute @workgroup_size(64)
fn attn_scores(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let seq = layer_spec.seq_len;
    let c_count = layer_spec.channels;
    let rows = arrayLength(&probs) / seq;   // batch * N
    let unit = min(wid.y * nwg.x + wid.x, rows - 1u);

    let b = unit / seq;
    let n = unit % seq;

    let q_row = qkv_sample_base(b) + n * c_count;
    let k_base = qkv_sample_base(b) + sample_len();
    let p_row = b * seq * seq + n * seq;

    // Pass 1 — raw scores, and the row max for the stable softmax.
    var local_max: f32 = -3.4028235e38;
    for (var m: u32 = tid; m < seq; m += WORKGROUP_SIZE) {
        var s: f32 = 0.0;
        for (var i: u32 = 0u; i < c_count; i += 1u) {
            s += qkv[q_row + i] * qkv[k_base + m * c_count + i];
        }
        s *= layer_spec.scale;
        probs[p_row + m] = s;
        local_max = max(local_max, s);
    }
    let row_max = workgroup_max(tid, local_max);

    // Pass 2 — exponentiate around the max. Subtracting it is not cosmetic:
    // without it exp() overflows to inf on large scores and the row becomes
    // NaN. Every thread re-reads only the slots it wrote itself, so no
    // cross-thread ordering is involved.
    var local_sum: f32 = 0.0;
    for (var m: u32 = tid; m < seq; m += WORKGROUP_SIZE) {
        let e = exp(probs[p_row + m] - row_max);
        probs[p_row + m] = e;
        local_sum += e;
    }
    let total = workgroup_sum(tid, local_sum);

    // Pass 3 — normalise.
    for (var m: u32 = tid; m < seq; m += WORKGROUP_SIZE) {
        probs[p_row + m] = probs[p_row + m] / total;
    }
}

// ---------------------------------------------------------------------------
// 3. context = probs · v
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn attn_context(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&ctx) { return; }

    let seq = layer_spec.seq_len;
    let c_count = layer_spec.channels;
    let nc = sample_len();

    let b = idx / nc;
    let inner = idx % nc;
    let n = inner / c_count;
    let i = inner % c_count;

    let v_base = qkv_sample_base(b) + 2u * nc;
    let p_row = b * seq * seq + n * seq;

    var acc: f32 = 0.0;
    for (var m: u32 = 0u; m < seq; m += 1u) {
        acc += probs[p_row + m] * qkv[v_base + m * c_count + i];
    }
    ctx[idx] = acc;
}

// ---------------------------------------------------------------------------
// 4. output projection + residual
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn attn_out(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&output) { return; }

    let c_count = layer_spec.channels;
    let nc = sample_len();

    let b = idx / nc;
    let inner = idx % nc;
    let n = inner / c_count;
    let c = inner % c_count;

    let o_base = 3u * c_count * c_count;
    let ctx_row = b * nc + n * c_count;

    var acc = bias_at(3u * c_count + c);
    for (var i: u32 = 0u; i < c_count; i += 1u) {
        acc += weights[o_base + c * c_count + i] * ctx[ctx_row + i];
    }
    // The residual. `input` and `output` share an index because the layer is
    // shape-preserving: with W_o = 0 (the initialisation) this is a copy.
    output[idx] = input[idx] + acc;
}
