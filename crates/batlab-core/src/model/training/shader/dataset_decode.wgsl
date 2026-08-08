// File purpose: WGSL compute shader turning an 8-bit dataset chunk into the
// f32 `[-1, 1]` batch slot the diffusion prepass reads.
//
// This shader is the whole point of BATRAW3. The dataset is natively 8-bit —
// every image the project has ever trained on came out of a JPEG, a PNG or a
// CIFAR byte plane — and storing it as f32 quadrupled BOTH the file on disk and
// the bytes crossing the host↔GPU boundary on every chunk miss. The widening is
// exact (`v / 127.5 - 1`, 256 values, no rounding), so doing it here rather
// than on the CPU costs nothing and saves 4× of the most expensive traffic the
// dataset does.
//
// The f32 path does not come through here: it stays a `copy_buffer_to_buffer`,
// bit for bit what it was. See `GpuDataset::copy_samples_to`.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x` — same convention as `diffusion_prepare.wgsl`.

// The resident chunk, addressed as words because WGSL has no `array<u8>`.
// Byte `b` of the chunk is byte `b & 3` of word `b >> 2`, little-endian, which
// is how `write_buffer` laid the host's bytes down.
@group(0) @binding(0) var<storage, read>       chunk: array<u32>;
// One entry per sample of the batch: `.x` is the sample's index INSIDE the
// resident chunk, `.y` the destination slot. Two separate numbers because the
// batch is grouped by chunk before it is copied, so neither order matches the
// other — the caller's slot order is preserved through `.y`.
@group(0) @binding(1) var<storage, read>       plan: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> destination: array<f32>;
@group(0) @binding(3) var<uniform>             spec: DecodeSpec;
// The 256 decoded values, computed on the CPU by `decode_u8` and uploaded once.
//
// This shader does NOT recompute the convention, and that is deliberate. It
// used to say `f32(byte) / 127.5 - 1.0` — the same expression as the CPU, to
// the character — and disagreed with it on 111 of the 256 values, by one ULP
// each: Metal compiles a float division as a multiply by an approximate
// reciprocal. One ULP is nothing to a network, but it means an 8-bit dataset
// and the f32 file it was converted from are not the same dataset, and the
// project's claim that BATRAW3 is a LOSSLESS re-encoding would have been false
// in a way no training curve would ever have shown. A table has no arithmetic
// for a compiler to reassociate, so the two sides cannot drift.
@group(0) @binding(4) var<storage, read>       palette: array<f32>;

struct DecodeSpec {
    // Values (not bytes — they are the same number here) in one sample.
    sample_len: u32,
    // Entries of `plan` this dispatch is asked for, which is at most the batch.
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
