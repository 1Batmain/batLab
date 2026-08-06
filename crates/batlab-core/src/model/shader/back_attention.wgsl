// File purpose: WGSL compute shader implementing the backward pass of spatial self-attention.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.
//
// Forward recap (attention.wgsl):
//   q,k,v = W·x + b ;  s = qᵀk·scale ;  p = softmax(s)
//   ctx   = p·v     ;  out = x + W_o·ctx + b_o
//
// Backward, in the order the passes are dispatched. Each pass reads only what
// an earlier pass has already written — that ordering IS the dependency graph,
// since wgpu barriers between compute passes of one encoder:
//
//   1. attn_back_ctx     grad_ctx[n,i] = Σ_c go[n,c]·W_o[c,i]
//   2. attn_back_v       grad_v[m,i]   = Σ_n p[n,m]·grad_ctx[n,i]
//   3. attn_back_scores  grad_p[n,m]   = Σ_i grad_ctx[n,i]·v[m,i]
//                        grad_s[n,m]   = p[n,m]·(grad_p[n,m] − Σ_j p[n,j]·grad_p[n,j])·scale
//   4. attn_back_qk      grad_q[n,i]   = Σ_m grad_s[n,m]·k[m,i]
//                        grad_k[n,i]   = Σ_m grad_s[m,n]·q[m,i]
//   5. attn_back_input   grad_x[n,c]   = go[n,c] + Σ_i Σ_{Q,K,V} grad_·[n,i]·W_·[i,c]
//   6. attn_back_weights one thread per weight, summing over batch AND positions
//   7. attn_back_bias    one thread per bias,   summing over batch AND positions
//
// Pass 5's leading `go[n,c]` is the residual: the identity branch carries the
// incoming gradient through untouched, which is what makes a zero-initialised
// W_o a harmless no-op rather than a dead end for the gradient.
//
// Bindings match AttentionType::get_back_buffers_specs():
//   [0] fwd_input     [1] weights      [2] specs      [3] grad_output
//   [4] grad_input    [5] grad_weights [6] grad_bias
//   [7] qkv           [8] probs        [9] ctx            (saved forward scratch)
//   [10] grad_qkv     [11] grad_ctx    [12] grad_scores   (backward scratch)

@group(0) @binding(0)  var<storage, read>       fwd_input:    array<f32>;
@group(0) @binding(1)  var<storage, read>       weights:      array<f32>;
@group(0) @binding(2)  var<uniform>             layer_spec:   AttentionSpec;
@group(0) @binding(3)  var<storage, read>       grad_output:  array<f32>;
@group(0) @binding(4)  var<storage, read_write> grad_input:   array<f32>;
@group(0) @binding(5)  var<storage, read_write> grad_weights: array<f32>;
@group(0) @binding(6)  var<storage, read_write> grad_bias:    array<f32>;
@group(0) @binding(7)  var<storage, read>       qkv:          array<f32>;
@group(0) @binding(8)  var<storage, read>       probs:        array<f32>;
@group(0) @binding(9)  var<storage, read>       ctx:          array<f32>;
@group(0) @binding(10) var<storage, read_write> grad_qkv:     array<f32>;
@group(0) @binding(11) var<storage, read_write> grad_ctx:     array<f32>;
@group(0) @binding(12) var<storage, read_write> grad_scores:  array<f32>;

struct AttentionSpec {
    seq_len:   u32,
    channels:  u32,
    scale:     f32,
    use_bias:  u32,
    dim_input:  vec3<u32>,
    dim_output: vec3<u32>,
}

const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> partial: array<f32, WORKGROUP_SIZE>;

fn sample_len() -> u32 {
    return layer_spec.seq_len * layer_spec.channels;
}

fn qkv_sample_base(b: u32) -> u32 {
    return b * 3u * sample_len();
}

/// How many samples the activation buffers carry. Read off the buffer, never
/// passed in — the batch axis has exactly one source of truth.
fn batch_count() -> u32 {
    return arrayLength(&fwd_input) / sample_len();
}

fn workgroup_sum(tid: u32, value: f32) -> f32 {
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

// ---------------------------------------------------------------------------
// 1. grad_ctx — through the output projection
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn attn_back_ctx(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&grad_ctx) { return; }

    let c_count = layer_spec.channels;
    let nc = sample_len();
    let b = idx / nc;
    let inner = idx % nc;
    let n = inner / c_count;
    let i = inner % c_count;

    let o_base = 3u * c_count * c_count;
    let go_row = b * nc + n * c_count;

    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < c_count; c += 1u) {
        acc += grad_output[go_row + c] * weights[o_base + c * c_count + i];
    }
    grad_ctx[idx] = acc;
}

// ---------------------------------------------------------------------------
// 2. grad_v — v is read by every query, so its gradient sums over the column
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn attn_back_v(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let nc = sample_len();
    if idx >= arrayLength(&grad_ctx) { return; }

    let seq = layer_spec.seq_len;
    let c_count = layer_spec.channels;
    let b = idx / nc;
    let inner = idx % nc;
    let m = inner / c_count;
    let i = inner % c_count;

    let p_base = b * seq * seq;
    var acc: f32 = 0.0;
    for (var n: u32 = 0u; n < seq; n += 1u) {
        acc += probs[p_base + n * seq + m] * grad_ctx[b * nc + n * c_count + i];
    }
    grad_qkv[qkv_sample_base(b) + 2u * nc + m * c_count + i] = acc;
}

// ---------------------------------------------------------------------------
// 3. grad_probs then the softmax Jacobian
// ---------------------------------------------------------------------------
//
// ONE WORKGROUP PER (SAMPLE, ROW), because the softmax backward is a reduction
// over the row: dL/ds_i = p_i·(dL/dp_i − Σ_j p_j·dL/dp_j). The subtracted term
// is the same for the whole row — it is what makes the gradient of a
// probability distribution sum to zero.
//
// grad_p is staged in `grad_scores` and then overwritten in place: each thread
// re-reads only the slots it wrote itself, so no cross-thread ordering is
// involved beyond the reduction's own barriers.

@compute @workgroup_size(64)
fn attn_back_scores(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let seq = layer_spec.seq_len;
    let c_count = layer_spec.channels;
    let nc = sample_len();
    let rows = arrayLength(&probs) / seq;
    let unit = min(wid.y * nwg.x + wid.x, rows - 1u);

    let b = unit / seq;
    let n = unit % seq;

    let p_row = b * seq * seq + n * seq;
    let gctx_row = b * nc + n * c_count;
    let v_base = qkv_sample_base(b) + 2u * nc;

    var local_dot: f32 = 0.0;
    for (var m: u32 = tid; m < seq; m += WORKGROUP_SIZE) {
        var gp: f32 = 0.0;
        for (var i: u32 = 0u; i < c_count; i += 1u) {
            gp += grad_ctx[gctx_row + i] * qkv[v_base + m * c_count + i];
        }
        grad_scores[p_row + m] = gp;
        local_dot += probs[p_row + m] * gp;
    }
    let dot = workgroup_sum(tid, local_dot);

    for (var m: u32 = tid; m < seq; m += WORKGROUP_SIZE) {
        // The scale of the forward (`s = qᵀk · scale`) rides along here so the
        // q/k pass can use grad_scores as-is.
        grad_scores[p_row + m] =
            probs[p_row + m] * (grad_scores[p_row + m] - dot) * layer_spec.scale;
    }
}

// ---------------------------------------------------------------------------
// 4. grad_q and grad_k
// ---------------------------------------------------------------------------
//
// Same index space (both are N×C per sample), so one thread produces both.
// Note the transposed read for k: q[n] is the ROW of the score matrix and k[n]
// its COLUMN, so grad_k sums down `grad_s[·, n]` while grad_q sums along
// `grad_s[n, ·]`. Swapping the two is a silent, plausible-looking bug — it
// produces finite gradients that are simply wrong.

@compute @workgroup_size(64)
fn attn_back_qk(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let nc = sample_len();
    if idx >= arrayLength(&grad_ctx) { return; }

    let seq = layer_spec.seq_len;
    let c_count = layer_spec.channels;
    let b = idx / nc;
    let inner = idx % nc;
    let n = inner / c_count;
    let i = inner % c_count;

    let base = qkv_sample_base(b);
    let k_base = base + nc;
    let p_base = b * seq * seq;

    var gq: f32 = 0.0;
    var gk: f32 = 0.0;
    for (var m: u32 = 0u; m < seq; m += 1u) {
        gq += grad_scores[p_base + n * seq + m] * qkv[k_base + m * c_count + i];
        gk += grad_scores[p_base + m * seq + n] * qkv[base + m * c_count + i];
    }
    grad_qkv[base + n * c_count + i] = gq;
    grad_qkv[k_base + n * c_count + i] = gk;
}

// ---------------------------------------------------------------------------
// 5. grad_input — the three projections plus the residual
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn attn_back_input(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&grad_input) { return; }

    let c_count = layer_spec.channels;
    let nc = sample_len();
    let b = idx / nc;
    let inner = idx % nc;
    let n = inner / c_count;
    let c = inner % c_count;

    let base = qkv_sample_base(b) + n * c_count;
    let cc = c_count * c_count;

    // The residual branch: out = x + …, so the incoming gradient reaches the
    // input directly as well as through the projections.
    var acc: f32 = grad_output[idx];
    for (var i: u32 = 0u; i < c_count; i += 1u) {
        let row = i * c_count + c;
        acc += grad_qkv[base + i] * weights[row];
        acc += grad_qkv[base + nc + i] * weights[cc + row];
        acc += grad_qkv[base + 2u * nc + i] * weights[2u * cc + row];
    }
    grad_input[idx] = acc;
}

// ---------------------------------------------------------------------------
// 6. grad_weights — one thread per weight, the batch folded into its loop
// ---------------------------------------------------------------------------
//
// A weight is a PARAMETER: there are exactly as many sums as there are weights
// whatever the batch, so this pass does not scale with it — the batch enters
// inside the kernel as more positions to reduce. One `=` per weight per step,
// not one per sample.

@compute @workgroup_size(64)
fn attn_back_weights(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let c_count = layer_spec.channels;
    let cc = c_count * c_count;
    if idx >= 4u * cc { return; }

    let seq = layer_spec.seq_len;
    let nc = sample_len();
    let samples = batch_count();

    let which = idx / cc;
    let rest = idx % cc;
    let row = rest / c_count;
    let col = rest % c_count;

    var acc: f32 = 0.0;
    if which == 3u {
        // W_o[c, i]: the output projection sees the incoming gradient and the
        // context it multiplied.
        for (var b: u32 = 0u; b < samples; b += 1u) {
            for (var n: u32 = 0u; n < seq; n += 1u) {
                let pos = b * nc + n * c_count;
                acc += grad_output[pos + row] * ctx[pos + col];
            }
        }
    } else {
        // W_q / W_k / W_v [i, c]: their own gradient against the layer input.
        for (var b: u32 = 0u; b < samples; b += 1u) {
            let g_base = qkv_sample_base(b) + which * nc;
            for (var n: u32 = 0u; n < seq; n += 1u) {
                acc += grad_qkv[g_base + n * c_count + row]
                     * fwd_input[b * nc + n * c_count + col];
            }
        }
    }
    grad_weights[idx] = acc;
}

// ---------------------------------------------------------------------------
// 7. grad_bias — same reduction, one thread per bias scalar
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn attn_back_bias(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let c_count = layer_spec.channels;
    if idx >= 4u * c_count { return; }

    let seq = layer_spec.seq_len;
    let nc = sample_len();
    let samples = batch_count();

    let which = idx / c_count;
    let j = idx % c_count;

    var acc: f32 = 0.0;
    if which == 3u {
        for (var b: u32 = 0u; b < samples; b += 1u) {
            for (var n: u32 = 0u; n < seq; n += 1u) {
                acc += grad_output[b * nc + n * c_count + j];
            }
        }
    } else {
        for (var b: u32 = 0u; b < samples; b += 1u) {
            let g_base = qkv_sample_base(b) + which * nc;
            for (var n: u32 = 0u; n < seq; n += 1u) {
                acc += grad_qkv[g_base + n * c_count + j];
            }
        }
    }
    grad_bias[idx] = acc;
}
