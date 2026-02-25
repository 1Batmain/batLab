use parallelizer::{
    ActivationLayerSpec, ActivationMethod, ConvolutionLayerSpec, Dim3, LayerSpec, Model, PaddingMode
};

#[tokio::main]
async fn main() {
    println!("Starting visualisation smoke test...");
    let mut model = Model::new(None).await;
    model.add_layer(LayerSpec::Convolution(ConvolutionLayerSpec {
        nb_kernel: 1,
        dim_kernel: Dim3 {x:1, y:1, z:1},
        stripe: 1,
        mode: PaddingMode::Valid,
        dim_input: Some(Dim3 {x:512, y:512, z:3}),
    }));
    model.add_layer(LayerSpec::Activation(ActivationLayerSpec { method: ActivationMethod::Linear, dim_input: None }));
    model.build_model();
    let input = vec![700.0; 512 * 512 * 3];
    let res = model.infer(input).await;
    println!("Result({}) : {:?}", res.len(), res);
    println!("Visual frame generated in ./visualisation_frames (PPM format)");
    model.save("name");
}