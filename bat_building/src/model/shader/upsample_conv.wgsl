// File purpose: WGSL compute shader implementing upsample conv operations for model forward/backward or optimizer passes.

// Bindings match UpsampleConvType::get_buffers_specs():
//   [0] input   — original input tensor (HWC)
//   [1] weights — output-major convolution weights (KHKWKC)
//   [2] bias    — one bias per output channel
//   [3] specs   — UpsampleConvSpec
//   [4] output  — convolved output after virtual nearest-neighbor upsampling

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       weights:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias:       array<f32>;
@group(0) @binding(3) var<uniform>             layer_spec: UpsampleConvSpec;
@group(0) @binding(4) var<storage, read_write> output:     array<f32>;

struct UpsampleConvSpec {
    nb_kernel:    u32,
    scale_factor: u32,
    padding_mode: u32, // 0 = Valid, 1 = Same
    _pad:         u32,
    dim_kernel:   vec3<u32>,
    dim_input:    vec3<u32>,
    dim_output:   vec3<u32>,
}

fn upsampled_height() -> u32 {
    return layer_spec.dim_input.x * layer_spec.scale_factor;
}

fn upsampled_width() -> u32 {
    return layer_spec.dim_input.y * layer_spec.scale_factor;
}

fn pad_y() -> i32 {
    if layer_spec.padding_mode == 1u {
        return i32(layer_spec.dim_kernel.x / 2u);
    }
    return 0;
}

fn pad_x() -> i32 {
    if layer_spec.padding_mode == 1u {
        return i32(layer_spec.dim_kernel.y / 2u);
    }
    return 0;
}

@compute @workgroup_size(64)
fn upsample_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    let OH = layer_spec.dim_output.x;
    let OW = layer_spec.dim_output.y;
    let K  = layer_spec.dim_output.z;
    if idx >= OH * OW * K { return; }

    let k  = idx % K;
    let ox = (idx / K) % OW;
    let oy = idx / (K * OW);

    let IW = layer_spec.dim_input.y;
    let IC = layer_spec.dim_input.z;
    let KH = layer_spec.dim_kernel.x;
    let KW = layer_spec.dim_kernel.y;
    let up_h = i32(upsampled_height());
    let up_w = i32(upsampled_width());
    let scale = layer_spec.scale_factor;

    var sum: f32 = bias[k];
    for (var ky: u32 = 0u; ky < KH; ky++) {
        for (var kx: u32 = 0u; kx < KW; kx++) {
            let up_y = i32(oy) + i32(ky) - pad_y();
            let up_x = i32(ox) + i32(kx) - pad_x();
            if up_y < 0 || up_y >= up_h || up_x < 0 || up_x >= up_w {
                continue;
            }
            let iy = u32(up_y) / scale;
            let ix = u32(up_x) / scale;
            for (var kz: u32 = 0u; kz < IC; kz++) {
                let in_i = iy * IW * IC + ix * IC + kz;
                let w_i = k * KH * KW * IC + ky * KW * IC + kx * IC + kz;
                sum += input[in_i] * weights[w_i];
            }
        }
    }
    output[idx] = sum;
}
