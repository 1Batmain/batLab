// File purpose: WGSL backward for TimeBias.
//
//   grad_input[b,h,w,c] = grad_output[b,h,w,c]            (the +input is identity)
//   grad_bias[c]        = Σ_{b,h,w} grad_output[b,h,w,c]
//   grad_weights[n,c]   = Σ_b emb[b,n] · Σ_{h,w} grad_output[b,h,w,c]
//
// No gradient flows to `time`: the embedding is an untrained model input.
//
// This mirrors GroupNorm's backward exactly, because that is the shape that
// survives naga→MSL on this machine. Two SEPARATE passes: an elementwise
// grad_input, and a parameter pass with ONE WORKGROUP PER OUTPUT CHANNEL whose
// single writer accumulates a register sum onto the framework-cleared buffer
// (`+=`). Every other arrangement tried came back zero build-dependently:
//   - folding the parameter reduction into the grad_input pass;
//   - overwriting (`=`) instead of accumulating — `clear_buffer` then an
//     overwriting compute pass is a WAW wgpu did not reliably barrier on Metal;
//   - a same-invocation read-modify-write (`grad_weights[i] = grad_weights[i] +
//     …` after zeroing it in the same thread), undefined under WGSL's relaxed
//     storage model.
// The `+=` here reads a value written by ANOTHER pass (the clear), which is
// well defined and is what forces the clear-before-this ordering. Pinned by
// `the_weight_and_bias_gradients_match_finite_differences`.

@group(0) @binding(0) var<storage, read>       time:        array<f32>;
@group(0) @binding(1) var<uniform>             layer_spec:  TimeBiasSpec;
@group(0) @binding(2) var<storage, read>       grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input:  array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_weights:array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_bias:   array<f32>;

// params = (embed_channels, embed_offset, time_sample_len).
struct TimeBiasSpec {
    dim: vec3<u32>,
    params: vec3<u32>,
}

// grad_input = grad_output (the +input term is the identity), elementwise.
@compute @workgroup_size(64)
fn time_bias_back_input(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if idx >= arrayLength(&grad_input) { return; }
    grad_input[idx] = grad_output[idx];
}

// One workgroup per output channel; a single writer accumulates onto the cleared
// grad_weights / grad_bias. Dispatched with `num_workgroups = C`, so the channel
// is the workgroup id (2-D grid folded back the way GroupNorm's does).
@compute @workgroup_size(1)
fn time_bias_grad_params(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let c = wid.y * nwg.x + wid.x;
    let c_count = layer_spec.dim.z;
    if c >= c_count { return; }

    let embed_channels = layer_spec.params.x;
    let embed_offset = layer_spec.params.y;
    let time_sample_len = layer_spec.params.z;

    let sp = layer_spec.dim.x * layer_spec.dim.y;
    let total_pos = arrayLength(&grad_output) / c_count; // batch * sp
    let batch = total_pos / max(sp, 1u);

    // grad_bias[c] = Σ_b Σ_{h,w} grad_output — register sum, folded on once.
    var gb = 0.0;
    for (var b = 0u; b < batch; b = b + 1u) {
        let base = b * sp;
        for (var p = 0u; p < sp; p = p + 1u) {
            gb = gb + grad_output[(base + p) * c_count + c];
        }
    }
    grad_bias[c] = grad_bias[c] + gb;

    // grad_weights[n,c] = Σ_b emb[b,n]·(Σ_{h,w} grad_output). n outermost so each
    // entry is one register accumulation folded on once; the per-sample spatial
    // sum is recomputed per n (embed_channels is small).
    for (var n = 0u; n < embed_channels; n = n + 1u) {
        var accw = 0.0;
        for (var b = 0u; b < batch; b = b + 1u) {
            var sb = 0.0;
            let base = b * sp;
            for (var p = 0u; p < sp; p = p + 1u) {
                sb = sb + grad_output[(base + p) * c_count + c];
            }
            accw = accw + time[b * time_sample_len + embed_offset + n] * sb;
        }
        let widx = n * c_count + c;
        grad_weights[widx] = grad_weights[widx] + accw;
    }
}
