// File purpose: WGSL compute shader implementing diffusion prepare preprocessing for diffusion training inputs.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

// `specs` is an ARRAY, one entry per sample of the batch, and that is the whole
// of how batching enters this kernel.
//
// Before batching it was a single uniform, rewritten between two submissions:
// the CPU loop set `alpha_bar`, `step` and `seed` for sample i, submitted, and
// went round again. The batched step submits once, so the per-sample values
// have to travel together — hence the array, indexed by `index / total`.
//
// What matters is what did NOT change: each entry is computed on the CPU by
// exactly the expression that produced it before (same `batch_offset`, same
// `sample_index` out of the same SampleShuffle, same `fold_seed`), so sample i
// of a batch draws bit-for-bit the noise and timestep it drew before. That is
// what makes an old-path/new-path training run comparable at all.
//
// On the CLAUDE.md rule against `seed ^ index`: the forbidden pattern is the
// COMPOSITION of two XORs — one folding the diffusion step into the path seed,
// one folding the pixel index into the noise field — whose sum collapsed onto
// the anti-diagonals (ANISOTROPY_HUNT.md). There is a single XOR here, the one
// that was already here, and batching adds none: the sample index SELECTS an
// entry, it never enters a seed.

@group(0) @binding(0) var<storage, read>       clean_target: array<f32>;
@group(0) @binding(1) var<storage, read>       specs: array<DiffusionPrepareSpec>;
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
    total_steps:       u32,
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

// Smooth multi-resolution timestep embedding. MUST stay byte-for-byte identical
// to `LinearNoiseSchedule::timestep_embedding` (schedule.rs): the network is
// trained with this GPU-side embedding and sampled with the CPU one, so any
// divergence conditions it on a signal it was never trained on.
fn timestep_value(spec: DiffusionPrepareSpec, offset: u32) -> f32 {
    if spec.timestep_channels == 0u {
        return 0.0;
    }
    let steps = max(spec.total_steps, 1u);
    let denom = f32(max(steps - 1u, 1u));
    let tau = f32(min(spec.step, steps - 1u)) / denom;
    let pair_idx = offset / 2u;
    let phase = tau * 3.14159265358979 * pow(2.0, f32(pair_idx));
    if (offset & 1u) == 0u {
        return sin(phase);
    }
    return cos(phase);
}

@compute @workgroup_size(64)
fn diffusion_prepare(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let index = gid.y * nwg.x * 64u + gid.x;
    if index >= arrayLength(&model_input) {
        return;
    }

    // Every entry carries the same shape, so entry 0 is enough to slice the
    // batch; only alpha_bar / step / seed differ from sample to sample.
    let total = specs[0].pixel_count * specs[0].input_channels;
    let sample = index / total;
    let local = index % total;
    let spec = specs[sample];

    let pixel = local / spec.input_channels;
    let channel = local % spec.input_channels;

    if channel < spec.signal_channels {
        let clean_idx = pixel * spec.signal_channels + channel;
        let noise = gaussian_from_seed(spec.seed ^ clean_idx);
        let signal_scale = sqrt(spec.alpha_bar);
        let noise_scale = sqrt(max(1.0 - spec.alpha_bar, 0.0));
        let clean_i = sample * spec.pixel_count * spec.signal_channels + clean_idx;
        model_input[index] = signal_scale * clean_target[clean_i] + noise_scale * noise;
        target_noise[clean_i] = noise;
    } else {
        let extra_channel = channel - spec.signal_channels;
        model_input[index] = timestep_value(spec, extra_channel);
    }
}
