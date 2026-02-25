@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> grad_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_weights: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_bias: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_output: array<f32>;

@compute
@workgroup_size(64)
fn inference(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let index = global_invocation_id.x;
    let total = arrayLength(&input);

    if (index >= total) {
        return;
    }
    
    output[index] = input[index];
}

@compute
@workgroup_size(64)
fn backpropagate (
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {

}