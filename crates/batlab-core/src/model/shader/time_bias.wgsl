// File purpose: WGSL forward for TimeBias — add a learned, per-channel bias
// derived from the timestep embedding to every spatial position.
//
//   output[b,h,w,c] = input[b,h,w,c] + bias[c] + Σ_n emb[b,n] · W[n,c]
//
// The embedding emb[b] is read from `time`, a saved copy of the model input, at
// pixel 0 of sample b, channels [embed_offset, embed_offset + embed_channels).
// It is spatially constant, so pixel 0 carries it whatever the resolution.
//
// The 2-D dispatch grid recovers the linear index from num_workgroups (see
// dispatch_grid in layer.rs); `nwg.x * 64` is one row of threads.

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       time:       array<f32>;
@group(0) @binding(2) var<storage, read>       weights:    array<f32>;
@group(0) @binding(3) var<storage, read>       bias:       array<f32>;
@group(0) @binding(4) var<uniform>             layer_spec: TimeBiasSpec;
@group(0) @binding(5) var<storage, read_write> output:     array<f32>;

// params = (embed_channels, embed_offset, time_sample_len); packed in a vec3 so
// the uniform is two vec3s, the layout encase and WGSL agree on.
struct TimeBiasSpec {
    dim: vec3<u32>,          // H, W, C of this layer
    params: vec3<u32>,
}

@compute @workgroup_size(64)
fn time_bias(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&output) { return; }

    let embed_channels = layer_spec.params.x;
    let embed_offset = layer_spec.params.y;
    let time_sample_len = layer_spec.params.z;

    let c_count = layer_spec.dim.z;
    let total = layer_spec.dim.x * layer_spec.dim.y * c_count;
    let sample = idx / total;
    let channel = (idx % total) % c_count; // channels are innermost

    let emb_base = sample * time_sample_len + embed_offset;
    var acc = bias[channel];
    for (var n = 0u; n < embed_channels; n = n + 1u) {
        acc = acc + time[emb_base + n] * weights[n * c_count + channel];
    }
    output[idx] = input[idx] + acc;
}
