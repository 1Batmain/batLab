@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let index = global_invocation_id.x;
    let total = arrayLength(&input);

    if (index >= total) {
        return;
    }
    
    output[index] = input[index] + 1.;
}