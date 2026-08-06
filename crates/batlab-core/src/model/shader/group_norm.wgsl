// File purpose: WGSL compute shader implementing group norm operations for model forward/backward or optimizer passes.
//
// Dispatch convention: ONE WORKGROUP PER GROUP (see
// `GroupNormType::get_forward_workgroup_count`), not one thread per element.
// The workgroup cooperates on two shared-memory reductions (sum, then sum of
// squared deviations) and then writes its group's normalised elements. The
// group statistics are therefore computed once per group instead of once per
// element, which is what the naive version did.

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       gamma:  array<f32>;
@group(0) @binding(2) var<storage, read>       beta:   array<f32>;
@group(0) @binding(3) var<uniform>             layer_spec: GroupNormSpec;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

struct GroupNormSpec {
    num_groups:         u32,
    channels_per_group: u32,
    spatial_len:        u32,
    epsilon:            f32,
    dim_input:          vec3<u32>,
    dim_output:         vec3<u32>,
}

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> partial: array<f32, WORKGROUP_SIZE>;

/// Flat buffer index of the `ordinal`-th element of `group`, in the same
/// (spatial-major, channel-minor) order the naive version walked.
fn group_element_index(group: u32, ordinal: u32) -> u32 {
    let cpg = layer_spec.channels_per_group;
    let spatial_idx = ordinal / cpg;
    let channel = group * cpg + ordinal % cpg;
    return spatial_idx * layer_spec.dim_input.z + channel;
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

@compute @workgroup_size(WORKGROUP_SIZE)
fn group_norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let group = wid.x;
    let group_len = layer_spec.spatial_len * layer_spec.channels_per_group;
    let group_len_f = f32(group_len);

    // Pass 1 — sum.
    var local_sum: f32 = 0.0;
    for (var i: u32 = tid; i < group_len; i += WORKGROUP_SIZE) {
        local_sum += input[group_element_index(group, i)];
    }
    let mean = workgroup_sum(tid, local_sum) / group_len_f;

    // Pass 2 — sum of squared deviations from the mean (same centered form as
    // the naive version; deliberately not the E[x^2] - E[x]^2 shortcut, which
    // would change the numerics).
    var local_var: f32 = 0.0;
    for (var i: u32 = tid; i < group_len; i += WORKGROUP_SIZE) {
        let centered = input[group_element_index(group, i)] - mean;
        local_var += centered * centered;
    }
    let variance = workgroup_sum(tid, local_var) / group_len_f;
    let inv_std = 1.0 / sqrt(variance + layer_spec.epsilon);

    // Pass 3 — normalise, scale, shift.
    for (var i: u32 = tid; i < group_len; i += WORKGROUP_SIZE) {
        let index = group_element_index(group, i);
        let channel = group * layer_spec.channels_per_group + i % layer_spec.channels_per_group;
        let normalized = (input[index] - mean) * inv_std;
        output[index] = normalized * gamma[channel] + beta[channel];
    }
}
