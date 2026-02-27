// Uniforms are packed as two vec4<u32> to respect std140-like alignment.
struct Dims {
    a: vec4<u32>, // in_w, in_h, in_ch, padding
    b: vec4<u32>, // out_w, out_h, out_ch, padding
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> output_buf: array<f32>;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0));
    return vec4(pos[idx], 0.0, 1.0);
}

fn sample_channel(x: u32, y: u32, w: u32, h: u32, ch: u32, c: u32) -> f32 {
    let idx = ((y * w + x) * ch) + c;
    return output_buf[idx];
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // `pos.xy` is in framebuffer pixel coordinates. Normalize by the
    // framebuffer (screen) size to obtain UV in [0,1]. Use the output
    // resolution stored in the uniform (dims.b.x / dims.b.y).
    let screen_w = f32(dims.b.x);
    let screen_h = f32(dims.b.y);
    var uv = pos.xy / vec2<f32>(screen_w, screen_h);
    // Guard against uv==1.0 which would produce an index == width.
    // We clamp after scaling so we can safely convert to integer and
    // then clamp to (width-1,height-1).
    let out_w = dims.b.x;
    let out_h = dims.b.y;
    let xf = u32(clamp(uv.x * f32(out_w), 0.0, f32(out_w - 1u)));
    let yf = u32(clamp(uv.y * f32(out_h), 0.0, f32(out_h - 1u)));
    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;
    if (out_w > 0u && out_h > 0u) {
        let out_ch = dims.b.z;
        if (out_ch > 0u) {
            r = sample_channel(xf, yf, out_w, out_h, out_ch, 0u);
        }
        if (out_ch > 1u) {
            g = sample_channel(xf, yf, out_w, out_h, out_ch, 1u);
        } else {
            g = r;
        }
        if (out_ch > 2u) {
            b = sample_channel(xf, yf, out_w, out_h, out_ch, 2u);
        } else {
            b = (r + g) * 0.5;
        }
    }
    return vec4(r, g, b, 1.0);
}
