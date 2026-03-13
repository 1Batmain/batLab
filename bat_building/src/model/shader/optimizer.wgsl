@group(0) @binding(0) var<storage, read> result: array<f32>;
@group(0) @binding(1) var<storage, read> expected: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_output: array<f32>;
@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let index = global_invocation_id.x;
    let total = arrayLength(&result);

    if (index >= total) {
        return;
    }

    let diff = result[index] - expected[index];
    grad_output[index] = 2.0 * diff / f32(total);
}
