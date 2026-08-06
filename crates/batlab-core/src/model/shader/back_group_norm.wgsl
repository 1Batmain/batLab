// File purpose: WGSL compute shader implementing back group norm operations for model forward/backward or optimizer passes.
//
// The backward pass is split into five dispatches (see
// `GroupNormType::get_back_entrypoints` / `get_back_workgroup_counts`), run in
// order by `Layer::encode_back_pass`:
//
//   1. group_norm_stats      one wg per (sample, group) -> mean, inv_std
//   2. group_norm_grad_stats one wg per (sample, group) -> sum_dxhat, sum_dxhat_xhat
//   3. group_norm_back_input one thread per element     -> grad_input (O(1) per element)
//   4. group_norm_back_gamma one wg per channel         -> grad_gamma
//   5. group_norm_back_beta  one wg per channel         -> grad_beta
//
// Passes 1 and 2 hoist the per-group reductions that the naive version redid
// inside every invocation. Their results live in the `stats` buffer, laid out
// as 4 f32 per (sample, group): [mean, inv_std, sum_dxhat, sum_dxhat_xhat].
//
// The batch axis splits the five passes exactly as it splits the convolution's
// three. Passes 1–3 are per activation and grow with the batch — and passes 1
// and 2 give each sample its OWN statistics, which is what keeps this a group
// norm rather than something batch-dependent. Passes 4 and 5 reduce onto
// gamma/beta, which are *parameters*: one workgroup per channel whatever the
// batch, sweeping `batch * spatial_len` positions instead of `spatial_len`.

@group(0) @binding(0) var<storage, read>       fwd_input:   array<f32>;
@group(0) @binding(1) var<storage, read>       gamma:       array<f32>;
@group(0) @binding(2) var<uniform>             layer_spec:  GroupNormSpec;
@group(0) @binding(3) var<storage, read>       grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input:  array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_gamma:  array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_beta:   array<f32>;
@group(0) @binding(7) var<storage, read_write> stats:       array<f32>;

struct GroupNormSpec {
    num_groups:         u32,
    channels_per_group: u32,
    spatial_len:        u32,
    epsilon:            f32,
    dim_input:          vec3<u32>,
    dim_output:         vec3<u32>,
}

const WORKGROUP_SIZE: u32 = 256u;

const STAT_MEAN:           u32 = 0u;
const STAT_INV_STD:        u32 = 1u;
const STAT_SUM_DXHAT:      u32 = 2u;
const STAT_SUM_DXHAT_XHAT: u32 = 3u;
const STATS_PER_GROUP:     u32 = 4u;

var<workgroup> partial: array<f32, WORKGROUP_SIZE>;

fn flat_index(spatial_idx: u32, channel: u32) -> u32 {
    return spatial_idx * layer_spec.dim_input.z + channel;
}

/// Elements of one sample's tensor.
fn sample_len() -> u32 {
    return layer_spec.spatial_len * layer_spec.dim_input.z;
}

/// Flat buffer index of the `ordinal`-th element of `group`, in the same
/// (spatial-major, channel-minor) order the naive version walked.
fn group_element_index(group: u32, ordinal: u32) -> u32 {
    let cpg = layer_spec.channels_per_group;
    let spatial_idx = ordinal / cpg;
    let channel = group * cpg + ordinal % cpg;
    return spatial_idx * layer_spec.dim_input.z + channel;
}

/// `slot` is `sample * num_groups + group` — the statistics are indexed by the
/// pair, never by the group alone.
fn stat(slot: u32, which: u32) -> f32 {
    return stats[slot * STATS_PER_GROUP + which];
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

// ---------------------------------------------------------------------------
// 1. Per-group mean / inv_std.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WORKGROUP_SIZE)
fn group_norm_stats(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let slot = wid.x;
    let group = wid.x % layer_spec.num_groups;
    let base = (wid.x / layer_spec.num_groups) * sample_len();
    let group_len = layer_spec.spatial_len * layer_spec.channels_per_group;
    let group_len_f = f32(group_len);

    var local_sum: f32 = 0.0;
    for (var i: u32 = tid; i < group_len; i += WORKGROUP_SIZE) {
        local_sum += fwd_input[base + group_element_index(group, i)];
    }
    let mean = workgroup_sum(tid, local_sum) / group_len_f;

    var local_var: f32 = 0.0;
    for (var i: u32 = tid; i < group_len; i += WORKGROUP_SIZE) {
        let centered = fwd_input[base + group_element_index(group, i)] - mean;
        local_var += centered * centered;
    }
    let variance = workgroup_sum(tid, local_var) / group_len_f;

    if tid == 0u {
        stats[slot * STATS_PER_GROUP + STAT_MEAN] = mean;
        stats[slot * STATS_PER_GROUP + STAT_INV_STD] =
            1.0 / sqrt(variance + layer_spec.epsilon);
    }
}

// ---------------------------------------------------------------------------
// 2. Per-group sum(dxhat) and sum(dxhat * xhat), needed by grad_input.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WORKGROUP_SIZE)
fn group_norm_grad_stats(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let slot = wid.x;
    let group = wid.x % layer_spec.num_groups;
    let base = (wid.x / layer_spec.num_groups) * sample_len();
    let group_len = layer_spec.spatial_len * layer_spec.channels_per_group;
    let mean = stat(slot, STAT_MEAN);
    let inv_std = stat(slot, STAT_INV_STD);
    let cpg = layer_spec.channels_per_group;

    var local_dxhat: f32 = 0.0;
    var local_dxhat_xhat: f32 = 0.0;
    for (var i: u32 = tid; i < group_len; i += WORKGROUP_SIZE) {
        let index = base + group_element_index(group, i);
        let channel = group * cpg + i % cpg;
        let x_hat = (fwd_input[index] - mean) * inv_std;
        let dxhat = grad_output[index] * gamma[channel];
        local_dxhat += dxhat;
        local_dxhat_xhat += dxhat * x_hat;
    }

    let sum_dxhat = workgroup_sum(tid, local_dxhat);
    let sum_dxhat_xhat = workgroup_sum(tid, local_dxhat_xhat);

    if tid == 0u {
        stats[slot * STATS_PER_GROUP + STAT_SUM_DXHAT] = sum_dxhat;
        stats[slot * STATS_PER_GROUP + STAT_SUM_DXHAT_XHAT] = sum_dxhat_xhat;
    }
}

// ---------------------------------------------------------------------------
// 3. grad_input — now O(1) per element.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn group_norm_back_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&fwd_input) { return; }

    let channel = index % layer_spec.dim_input.z;
    let group = channel / layer_spec.channels_per_group;
    // The statistics slot of THIS element's sample, not of its group alone.
    let slot = (index / sample_len()) * layer_spec.num_groups + group;
    let mean = stat(slot, STAT_MEAN);
    let inv_std = stat(slot, STAT_INV_STD);
    let sum_dxhat = stat(slot, STAT_SUM_DXHAT);
    let sum_dxhat_xhat = stat(slot, STAT_SUM_DXHAT_XHAT);

    let group_len_f = f32(layer_spec.spatial_len * layer_spec.channels_per_group);
    let x_hat = (fwd_input[index] - mean) * inv_std;
    let dxhat = grad_output[index] * gamma[channel];

    grad_input[index] = inv_std / group_len_f
        * (group_len_f * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat);
}

// ---------------------------------------------------------------------------
// 4. grad_gamma — one workgroup per channel, reducing over the spatial axis
//    OF THE WHOLE BATCH. gamma is a parameter: the batch is not an axis it
//    keeps, it is an axis it sums away. `mean`/`inv_std` still come from the
//    sample the position belongs to, which is why they are re-read inside the
//    loop instead of being hoisted.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WORKGROUP_SIZE)
fn group_norm_back_gamma(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let channel = wid.x;
    let group = channel / layer_spec.channels_per_group;
    let len = sample_len();
    let batch = arrayLength(&fwd_input) / len;
    let work = layer_spec.spatial_len * batch;

    var local_sum: f32 = 0.0;
    for (var p: u32 = tid; p < work; p += WORKGROUP_SIZE) {
        let sample = p / layer_spec.spatial_len;
        let s = p % layer_spec.spatial_len;
        let slot = sample * layer_spec.num_groups + group;
        let index = sample * len + flat_index(s, channel);
        let x_hat = (fwd_input[index] - stat(slot, STAT_MEAN)) * stat(slot, STAT_INV_STD);
        local_sum += grad_output[index] * x_hat;
    }
    let total = workgroup_sum(tid, local_sum);

    if tid == 0u {
        grad_gamma[channel] += total;
    }
}

// ---------------------------------------------------------------------------
// 5. grad_beta — one workgroup per channel, reducing over the spatial axis.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WORKGROUP_SIZE)
fn group_norm_back_beta(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let channel = wid.x;
    let len = sample_len();
    let batch = arrayLength(&fwd_input) / len;
    let work = layer_spec.spatial_len * batch;

    var local_sum: f32 = 0.0;
    for (var p: u32 = tid; p < work; p += WORKGROUP_SIZE) {
        let sample = p / layer_spec.spatial_len;
        let s = p % layer_spec.spatial_len;
        local_sum += grad_output[sample * len + flat_index(s, channel)];
    }
    let total = workgroup_sum(tid, local_sum);

    if tid == 0u {
        grad_beta[channel] += total;
    }
}
