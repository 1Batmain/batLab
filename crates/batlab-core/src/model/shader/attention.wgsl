// File purpose: WGSL compute shader implementing spatial self-attention forward passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.
//
// THE LAYER IS RESIDUAL: `output = input + W_o · softmax(qᵀk/√d) · V`. The
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
// ---------------------------------------------------------------------------
// WHY ONE SCRATCH BUFFER AND NOT FIVE
// ---------------------------------------------------------------------------
//
// WebGPU guarantees only EIGHT storage buffers per shader stage
// (`max_storage_buffers_per_shader_stage`), and this engine targets the
// visitor's own GPU through a browser — raising the limit at `request_device`
// would trade portability for convenience. The backward pass needs the layer
// input, the weights, the incoming gradient, grad_input, grad_weights,
// grad_bias, everything the forward saved, and its own scratch. Bound
// separately that is twelve, and the failure is SILENT AND TOTAL: the bind
// group layout is rejected, so the whole command buffer — forward included —
// is dropped, the loss reads 0.000000, and nothing else says a word.
//
// So q, k, v, ctx and probs share ONE buffer, and their gradients share
// another with THE SAME LAYOUT. Block `X`'s gradient sits at block `X`'s
// offset in the twin buffer, which is what keeps the arithmetic legible:
//
//   [ q | k | v | ctx | probs ]   per sample, stride 4·N·C + N²
//     0  NC  2NC  3NC   4NC
//
// Bindings match AttentionType::get_buffers_specs():
//   [0] input   — HWC layout, N=H*W positions of C channels: index = n*C + c
//   [1] weights — the FOUR projections concatenated, C*C each, in order
//                 Wq, Wk, Wv at [i*C + c] (row i = output feature), then
//                 Wo at [c*C + i] (row c = output channel). One buffer so the
//                 optimiser sees one weight tensor, exactly like a convolution.
//   [2] bias    — 4*C: bq, bk, bv, bo back to back
//   [3] specs   — AttentionSpec
//   [4] scratch — the five blocks above, saved for the backward pass
//   [5] output  — HWC, same shape as input (LAST: the model chains .last())

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       weights:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias:       array<f32>;
@group(0) @binding(3) var<uniform>             layer_spec: AttentionSpec;
@group(0) @binding(4) var<storage, read_write> scratch:    array<f32>;
@group(0) @binding(5) var<storage, read_write> output:     array<f32>;

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

/// Elements of one sample's scratch: q, k, v, ctx (N*C each) then probs (N*N).
fn scratch_stride() -> u32 {
    return 4u * sample_len() + layer_spec.seq_len * layer_spec.seq_len;
}

/// Start of block `block` (0=q, 1=k, 2=v, 3=ctx, 4=probs) for sample `b`.
fn block_base(b: u32, block: u32) -> u32 {
    return b * scratch_stride() + block * sample_len();
}

/// How many samples the activation buffers carry. Read off the buffer, never
/// passed in — the batch axis has exactly one source of truth.
fn batch_count() -> u32 {
    return arrayLength(&input) / sample_len();
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
// One thread per scalar of the q/k/v blocks — of the whole batch. The sample is
// recovered by dividing by the per-sample length, never passed in.

@compute @workgroup_size(64)
fn attn_qkv(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= 3u * arrayLength(&input) { return; }

    let c_count = layer_spec.channels;
    let nc = sample_len();

    let b = idx / (3u * nc);
    let rest = idx % (3u * nc);
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
    scratch[block_base(b, which) + inner] = acc;
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
// Trailing workgroups from the 2-D grid split RETURN. They must not be folded
// onto the last real row instead: this pass is a read-modify-write on that row
// (write the score, read it back to exponentiate, read that back to normalise),
// so two workgroups sharing a row would interleave — one reading a value the
// other has already transformed. The result is not "the same work done twice",
// it is corruption, and only above 65 535 row-workgroups, which is where nobody
// is looking.
//
// The early return is legal precisely because the condition is WORKGROUP-
// UNIFORM: `unit` comes from `wid` and `nwg`, identical for every invocation of
// the workgroup, so all of them take the same branch and the barriers below
// stay in uniform control flow. A per-thread condition could not do this.

@compute @workgroup_size(64)
fn attn_scores(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let seq = layer_spec.seq_len;
    let c_count = layer_spec.channels;
    let rows = batch_count() * seq;
    let unit = wid.y * nwg.x + wid.x;
    if unit >= rows { return; }

    let b = unit / seq;
    let n = unit % seq;

    let q_row = block_base(b, 0u) + n * c_count;
    let k_base = block_base(b, 1u);
    let p_row = block_base(b, 4u) + n * seq;

    // Pass 1 — raw scores, and the row max for the stable softmax.
    var local_max: f32 = -3.4028235e38;
    for (var m: u32 = tid; m < seq; m += WORKGROUP_SIZE) {
        var s: f32 = 0.0;
        for (var i: u32 = 0u; i < c_count; i += 1u) {
            s += scratch[q_row + i] * scratch[k_base + m * c_count + i];
        }
        s *= layer_spec.scale;
        scratch[p_row + m] = s;
        local_max = max(local_max, s);
    }
    let row_max = workgroup_max(tid, local_max);

    // Pass 2 — exponentiate around the max. Subtracting it is not cosmetic:
    // without it exp() overflows to inf on large scores and the row becomes
    // NaN. Every thread re-reads only the slots it wrote itself, so no
    // cross-thread ordering is involved.
    var local_sum: f32 = 0.0;
    for (var m: u32 = tid; m < seq; m += WORKGROUP_SIZE) {
        let e = exp(scratch[p_row + m] - row_max);
        scratch[p_row + m] = e;
        local_sum += e;
    }
    let total = workgroup_sum(tid, local_sum);

    // Pass 3 — normalise.
    for (var m: u32 = tid; m < seq; m += WORKGROUP_SIZE) {
        scratch[p_row + m] = scratch[p_row + m] / total;
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
    if idx >= arrayLength(&input) { return; }

    let seq = layer_spec.seq_len;
    let c_count = layer_spec.channels;
    let nc = sample_len();

    let b = idx / nc;
    let inner = idx % nc;
    let n = inner / c_count;
    let i = inner % c_count;

    let v_base = block_base(b, 2u);
    let p_row = block_base(b, 4u) + n * seq;

    var acc: f32 = 0.0;
    for (var m: u32 = 0u; m < seq; m += 1u) {
        acc += scratch[p_row + m] * scratch[v_base + m * c_count + i];
    }
    scratch[block_base(b, 3u) + inner] = acc;
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
    let ctx_row = block_base(b, 3u) + n * c_count;

    var acc = bias_at(3u * c_count + c);
    for (var i: u32 = 0u; i < c_count; i += 1u) {
        acc += weights[o_base + c * c_count + i] * scratch[ctx_row + i];
    }
    // The residual. `input` and `output` share an index because the layer is
    // shape-preserving: with W_o = 0 (the initialisation) this is a copy.
    output[idx] = input[idx] + acc;
}
