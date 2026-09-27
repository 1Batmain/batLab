// Widens an 8-bit BATRAW3 chunk into the f32 `[-1, 1]` batch slot the diffusion
// prepass reads. The f32 path skips this shader (plain `copy_buffer_to_buffer`,
// see `GpuDataset::copy_samples_to`). Rationale and traffic numbers: AGENTS.md.
//
// 2-D dispatch grid when the batch pushes the workgroup count past WebGPU's
// 65 535-per-dimension limit (`dispatch_grid` in layer.rs), so the linear thread
// index comes from `num_workgroups`, not `gid.x` — same as `diffusion_prepare.wgsl`.

// Chunk addressed as words (WGSL has no `array<u8>`); byte `b` is byte `b & 3` of
// word `b >> 2`, little-endian, as `write_buffer` laid it down.
@group(0) @binding(0) var<storage, read>       chunk: array<u32>;
// Per batch sample: `.x` = index inside the resident chunk, `.y` = destination
// slot. Two numbers because the batch is grouped by chunk before copy, so the
// orders differ — caller's slot order is preserved through `.y`.
@group(0) @binding(1) var<storage, read>       plan: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> destination: array<f32>;
@group(0) @binding(3) var<uniform>             spec: DecodeSpec;
// 256-entry lookup table, NOT the `v/127.5 - 1` formula: Metal compiles the
// division as an approximate reciprocal and diverged from the CPU by 1 ULP on
// 111/256 values. A table has no arithmetic to reassociate. Guarded bit-for-bit
// by `the_gpu_decode_agrees_with_the_cpu_one` (dataset.rs).
@group(0) @binding(4) var<storage, read>       palette: array<f32>;

struct DecodeSpec {
    sample_len: u32,
    pair_count: u32,
    pad0: u32,
    pad1: u32,
}

@compute @workgroup_size(64)
fn dataset_decode(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let index = gid.y * nwg.x * 64u + gid.x;
    if index >= spec.sample_len * spec.pair_count {
        return;
    }

    let pair = plan[index / spec.sample_len];
    let element = index % spec.sample_len;

    let source_byte = pair.x * spec.sample_len + element;
    let word = chunk[source_byte >> 2u];
    let byte = (word >> ((source_byte & 3u) * 8u)) & 0xffu;

    destination[pair.y * spec.sample_len + element] = palette[byte];
}
