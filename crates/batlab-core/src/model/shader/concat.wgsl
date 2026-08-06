// File purpose: WGSL compute shader implementing concat operations for model forward/backward or optimizer passes.
//
// The dispatch grid is 2-D when the batch pushes the workgroup count past
// WebGPU's 65 535-per-dimension limit (see `dispatch_grid` in layer.rs), so the
// linear thread index is recovered from `num_workgroups` rather than read
// straight out of `gid.x`. `nwg.x * 64` is the width of one row of threads.

// Bindings match ConcatType::get_buffers_specs():
//   [0] input      — current sequential tensor (HWC)
//   [1] skip_input — saved skip tensor (HWC)
//   [2] specs      — concat dimensions
//   [3] output     — concatenated tensor (HWC, channel-wise)

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       skip_input: array<f32>;
@group(0) @binding(2) var<uniform>             layer_spec: ConcatSpec;
@group(0) @binding(3) var<storage, read_write> output:     array<f32>;

struct ConcatSpec {
    dim_input:  vec3<u32>,
    dim_skip:   vec3<u32>,
    dim_output: vec3<u32>,
}

// The three tensors have DIFFERENT per-sample lengths (out = input + skip
// channels), so each needs its own sample offset — this is the layer where a
// single shared `sample * len` would be wrong.
@compute @workgroup_size(64)
fn concat(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let total = layer_spec.dim_output.x * layer_spec.dim_output.y * layer_spec.dim_output.z;
    if idx >= arrayLength(&output) { return; }

    let sample = idx / total;
    let local = idx % total;

    let out_c = layer_spec.dim_output.z;
    let input_c = layer_spec.dim_input.z;
    let skip_c = layer_spec.dim_skip.z;
    let pixels = layer_spec.dim_output.x * layer_spec.dim_output.y;
    let channel = local % out_c;
    let pixel_idx = local / out_c;

    if channel < input_c {
        output[idx] = input[sample * pixels * input_c + pixel_idx * input_c + channel];
        return;
    }

    let skip_channel = channel - input_c;
    if skip_channel < skip_c {
        output[idx] = skip_input[sample * pixels * skip_c + pixel_idx * skip_c + skip_channel];
    }
}
