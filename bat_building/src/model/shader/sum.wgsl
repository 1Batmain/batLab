// File purpose: WGSL compute shader implementing sum operations for model forward/backward or optimizer passes.

@group(0) @binding(0) var<storage, read>       lhs: array<f32>;
@group(0) @binding(1) var<storage, read>       rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&out) { return; }
    out[idx] = lhs[idx] + rhs[idx];
}
