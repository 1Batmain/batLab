@group(0) @binding(0) var<storage, read>       fwd_input:   array<f32>;
@group(0) @binding(1) var<storage, read>       gamma:       array<f32>;
@group(0) @binding(2) var<uniform>             layer_spec:  GroupNormSpec;
@group(0) @binding(3) var<storage, read>       grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input:  array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_gamma:  array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_beta:   array<f32>;

struct GroupNormSpec {
    num_groups:         u32,
    channels_per_group: u32,
    spatial_len:        u32,
    epsilon:            f32,
    dim_input:          vec3<u32>,
    dim_output:         vec3<u32>,
}

fn flat_index(spatial_idx: u32, channel: u32) -> u32 {
    return spatial_idx * layer_spec.dim_input.z + channel;
}

fn compute_group_mean(group: u32) -> f32 {
    let channel_start = group * layer_spec.channels_per_group;
    let channel_end = channel_start + layer_spec.channels_per_group;
    var sum: f32 = 0.0;
    for (var spatial_idx: u32 = 0u; spatial_idx < layer_spec.spatial_len; spatial_idx++) {
        for (var channel: u32 = channel_start; channel < channel_end; channel++) {
            sum += fwd_input[flat_index(spatial_idx, channel)];
        }
    }
    let group_len = layer_spec.spatial_len * layer_spec.channels_per_group;
    return sum / f32(group_len);
}

fn compute_group_inv_std(group: u32, mean: f32) -> f32 {
    let channel_start = group * layer_spec.channels_per_group;
    let channel_end = channel_start + layer_spec.channels_per_group;
    var variance_sum: f32 = 0.0;
    for (var spatial_idx: u32 = 0u; spatial_idx < layer_spec.spatial_len; spatial_idx++) {
        for (var channel: u32 = channel_start; channel < channel_end; channel++) {
            let centered = fwd_input[flat_index(spatial_idx, channel)] - mean;
            variance_sum += centered * centered;
        }
    }
    let group_len = layer_spec.spatial_len * layer_spec.channels_per_group;
    let variance = variance_sum / f32(group_len);
    return 1.0 / sqrt(variance + layer_spec.epsilon);
}

@compute @workgroup_size(64)
fn group_norm_back_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&fwd_input) { return; }

    let channel = index % layer_spec.dim_input.z;
    let group = channel / layer_spec.channels_per_group;
    let channel_start = group * layer_spec.channels_per_group;
    let channel_end = channel_start + layer_spec.channels_per_group;
    let mean = compute_group_mean(group);
    let inv_std = compute_group_inv_std(group, mean);
    let group_len = layer_spec.spatial_len * layer_spec.channels_per_group;
    let x_hat = (fwd_input[index] - mean) * inv_std;
    let dxhat = grad_output[index] * gamma[channel];

    var sum_dxhat: f32 = 0.0;
    var sum_dxhat_xhat: f32 = 0.0;
    for (var spatial_idx: u32 = 0u; spatial_idx < layer_spec.spatial_len; spatial_idx++) {
        for (var group_channel: u32 = channel_start; group_channel < channel_end; group_channel++) {
            let group_index = flat_index(spatial_idx, group_channel);
            let group_x_hat = (fwd_input[group_index] - mean) * inv_std;
            let group_dxhat = grad_output[group_index] * gamma[group_channel];
            sum_dxhat += group_dxhat;
            sum_dxhat_xhat += group_dxhat * group_x_hat;
        }
    }

    let group_len_f = f32(group_len);
    grad_input[index] = inv_std / group_len_f
        * (group_len_f * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat);
}

@compute @workgroup_size(64)
fn group_norm_back_gamma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channel = gid.x;
    if channel >= layer_spec.dim_input.z { return; }

    let group = channel / layer_spec.channels_per_group;
    let mean = compute_group_mean(group);
    let inv_std = compute_group_inv_std(group, mean);
    var accum: f32 = 0.0;
    for (var spatial_idx: u32 = 0u; spatial_idx < layer_spec.spatial_len; spatial_idx++) {
        let index = flat_index(spatial_idx, channel);
        let x_hat = (fwd_input[index] - mean) * inv_std;
        accum += grad_output[index] * x_hat;
    }
    grad_gamma[channel] = accum;
}

@compute @workgroup_size(64)
fn group_norm_back_beta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channel = gid.x;
    if channel >= layer_spec.dim_input.z { return; }

    var accum: f32 = 0.0;
    for (var spatial_idx: u32 = 0u; spatial_idx < layer_spec.spatial_len; spatial_idx++) {
        accum += grad_output[flat_index(spatial_idx, channel)];
    }
    grad_beta[channel] = accum;
}
