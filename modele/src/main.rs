use parallelizer::{
    LayerSpec,
    Wrapper
};

#[tokio::main]
async fn main() {
    let layer: LayerSpec = LayerSpec::new(
        "Test".to_string(),
        "../shader/test.wgsl".to_string()
    )
    .workgroup_size(64)
    .input(vec![0., 1., 2., 3., 4., 5.])
    .output(Vec::new());

    let mut wrapper = Wrapper::new().await;
    wrapper.add_layer(layer);
    let result = wrapper.run().await;
    println!("Resulting : {:?}", result);

}
