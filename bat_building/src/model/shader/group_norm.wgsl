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

fn flat_index(spatial_idx: u32, channel: u32) -> u32 {
    return spatial_idx * layer_spec.dim_input.z + channel;
}

fn compute_group_mean(group: u32) -> f32 {
    let channel_start = group * layer_spec.channels_per_group;
    let channel_end = channel_start + layer_spec.channels_per_group;
    var sum: f32 = 0.0;
    for (var spatial_idx: u32 = 0u; spatial_idx < layer_spec.spatial_len; spatial_idx++) {
        for (var channel: u32 = channel_start; channel < channel_end; channel++) {
            sum += input[flat_index(spatial_idx, channel)];
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
            let centered = input[flat_index(spatial_idx, channel)] - mean;
            variance_sum += centered * centered;
        }
    }
    let group_len = layer_spec.spatial_len * layer_spec.channels_per_group;
    let variance = variance_sum / f32(group_len);
    return 1.0 / sqrt(variance + layer_spec.epsilon);
}

@compute @workgroup_size(64)
fn group_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&input) { return; }

    let channel = index % layer_spec.dim_input.z;
    let group = channel / layer_spec.channels_per_group;
    let mean = compute_group_mean(group);
    let inv_std = compute_group_inv_std(group, mean);
    let normalized = (input[index] - mean) * inv_std;
    output[index] = normalized * gamma[channel] + beta[channel];
}
