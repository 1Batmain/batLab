struct Dims {
    out_w: u32;
    out_h: u32;
    out_ch: u32;
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> output_buf: array<f32>;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}

fn sample_output(x: u32, y: u32, channel: u32) -> f32 {
    let w = dims.out_w;
    let h = dims.out_h;
    let ch = dims.out_ch;
    if (x >= w || y >= h || channel >= ch) {
        return 0.0;
    }
    let idx: u32 = ((y * w + x) * ch) + channel;
    return output_buf[idx];
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy * 0.5 + vec2<f32>(0.5, 0.5));
    let fx = u32(uv.x * f32(dims.out_w));
    let fy = u32(uv.y * f32(dims.out_h));
    let r = sample_output(fx, fy, 0u);
    let g = sample_output(fx, fy, 1u);
    let b = sample_output(fx, fy, 2u);
    return vec4<f32>(r, g, b, 1.0);
}
