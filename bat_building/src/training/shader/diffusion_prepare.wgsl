@group(0) @binding(0) var<storage, read>       clean_target: array<f32>;
@group(0) @binding(1) var<uniform>             specs: DiffusionPrepareSpec;
@group(0) @binding(2) var<storage, read_write> model_input:  array<f32>;
@group(0) @binding(3) var<storage, read_write> target_noise: array<f32>;

struct DiffusionPrepareSpec {
    alpha_bar:         f32,
    step:              u32,
    seed:              u32,
    input_channels:    u32,
    signal_channels:   u32,
    timestep_channels: u32,
    pixel_count:       u32,
}

fn hash_u32(value_in: u32) -> u32 {
    var value = value_in;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

fn unit_from_seed(seed: u32) -> f32 {
    let bits = hash_u32(seed) >> 8u;
    let max_bits = f32((1u << 24u) - 1u);
    let normalized = f32(bits) / max_bits;
    return clamp(normalized, 1e-7, 1.0 - 1e-7);
}

fn gaussian_from_seed(seed: u32) -> f32 {
    let u1 = unit_from_seed(seed);
    let u2 = unit_from_seed(seed ^ 0x9e3779b9u);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

fn timestep_value(offset: u32) -> f32 {
    if specs.timestep_channels == 0u {
        return 0.0;
    }
    let half = (specs.timestep_channels + 1u) / 2u;
    let pair_idx = offset / 2u;
    let denom = max(half, 2u) - 1u;
    let exponent = f32(pair_idx) / f32(denom);
    let frequency = 1.0 / pow(10000.0, exponent);
    let phase = f32(specs.step) * frequency;
    if (offset & 1u) == 0u {
        return sin(phase);
    }
    return cos(phase);
}

@compute @workgroup_size(64)
fn diffusion_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let total = specs.pixel_count * specs.input_channels;
    if index >= total {
        return;
    }

    let pixel = index / specs.input_channels;
    let channel = index % specs.input_channels;

    if channel < specs.signal_channels {
        let clean_idx = pixel * specs.signal_channels + channel;
        let noise = gaussian_from_seed(specs.seed ^ clean_idx);
        let signal_scale = sqrt(specs.alpha_bar);
        let noise_scale = sqrt(max(1.0 - specs.alpha_bar, 0.0));
        model_input[index] = signal_scale * clean_target[clean_idx] + noise_scale * noise;
        target_noise[clean_idx] = noise;
    } else {
        let extra_channel = channel - specs.signal_channels;
        model_input[index] = timestep_value(extra_channel);
    }
}
