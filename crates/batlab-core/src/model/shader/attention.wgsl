// Spatial self-attention, forward. 2-D dispatch grid past WebGPU's 65 535-per-dim
// limit (`dispatch_grid` in layer.rs).
//
// RESIDUAL: `output = input + W_o · softmax(qᵀk/√d) · V`. The residual is inside
// the layer so the graph stays a chain, and with `W_o` init to zero it starts as
// the exact identity. FOUR passes, because attention needs GLOBAL barriers (row
// `n` reads every `k[m]`/`v[m]`) that only wgpu's inter-pass barrier provides:
//   1. attn_qkv     — q, k, v = W·x + b            (one thread per q/k/v scalar)
//   2. attn_scores  — softmax(qᵀk · scale) rows    (ONE WORKGROUP PER (sample, row))
//   3. attn_context — ctx = probs · v              (one thread per ctx scalar)
//   4. attn_out     — out = x + W_o·ctx + b_o      (one thread per output scalar)
//
// ONE scratch buffer, not five: WebGPU guarantees only EIGHT storage buffers per
// stage, and the backward already needs most of them — bind q/k/v/ctx/probs
// separately and it overflows, a SILENT TOTAL failure (bind group rejected, whole
// command buffer dropped, loss 0.000000). So they share one buffer, gradients a
// twin with the same layout (block `X`'s gradient at block `X`'s offset):
//
//   [ q | k | v | ctx | probs ]   per sample, stride 4·N·C + N²
//     0  NC  2NC  3NC   4NC
//
// Bindings (AttentionType::get_buffers_specs()): [0] input (HWC, n*C+c), [1]
// weights (Wq,Wk,Wv at [i*C+c] then Wo at [c*C+i], one tensor for the optimiser),
// [2] bias (4*C), [3] specs, [4] scratch (the five blocks, saved for backward),
// [5] output (HWC, LAST — the model chains .last()).

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
// ONE WORKGROUP PER (SAMPLE, ROW): the softmax denominator reduces over the row.
// The row NEVER leaves its sample (`b` fixed, all reads offset into that sample),
// so batching two images cannot change either's output — the batch contract, and
// this layer's specific failure mode. Trailing workgroups RETURN, not folded onto
// the last row: this is a read-modify-write, so two workgroups sharing a row would
// corrupt it (only above 65 535 row-workgroups, where nobody looks). The return is
// legal because the condition is WORKGROUP-UNIFORM, keeping the barriers below in
// uniform control flow.

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

    // Pass 1 — raw scores and the row max for the stable softmax. The sentinel is
    // NOT `f32::MIN`'s `-3.4028235e38`: read as an exact decimal it sits ABOVE
    // f32::MAX, so a strict WGSL front-end (the browser's, unlike naga→Metal)
    // rejects it and the pipeline fails silently. Any literal under f32::MAX works.
    var local_max: f32 = -3.4028234e38;
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
