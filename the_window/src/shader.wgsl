// Model-output visualiser shaders
//
// The data buffer holds the flat f32 model-output tensor in row-major,
// interleaved-channel order: buf[(y * width + x) * channels + c].
// Values are expected in approximately [-1, 1]; they are rescaled to [0, 1]
// for display.

struct Uniforms {
    width:    u32,
    height:   u32,
    channels: u32,
    _pad:     u32,
}

@group(0) @binding(0) var<storage, read> buf: array<f32>;
@group(0) @binding(1) var<uniform>       uni: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

// Full-screen quad emitted as a 4-vertex triangle strip.
// vi = 0 → (−1, +1)  top-left
// vi = 1 → (+1, +1)  top-right
// vi = 2 → (−1, −1)  bottom-left
// vi = 3 → (+1, −1)  bottom-right
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32(vi & 1u);
    let y = f32((vi >> 1u) & 1u);
    return VsOut(
        vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0),
        vec2<f32>(x, y),
    );
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if uni.channels == 0u || uni.width == 0u || uni.height == 0u {
        return vec4<f32>(0.1, 0.1, 0.1, 1.0);
    }

    // Map UV → pixel coordinate (clamped to valid range).
    let px = min(u32(in.uv.x * f32(uni.width)),  uni.width  - 1u);
    let py = min(u32(in.uv.y * f32(uni.height)), uni.height - 1u);

    let base = (py * uni.width + px) * uni.channels;

    // Rescale [-1, 1] → [0, 1] and clamp for display.
    let r = clamp(buf[base] * 0.5 + 0.5, 0.0, 1.0);

    if uni.channels >= 3u {
        let g = clamp(buf[base + 1u] * 0.5 + 0.5, 0.0, 1.0);
        let b = clamp(buf[base + 2u] * 0.5 + 0.5, 0.0, 1.0);
        return vec4<f32>(r, g, b, 1.0);
    }

    return vec4<f32>(r, r, r, 1.0);
}
