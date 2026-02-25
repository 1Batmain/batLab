use parallelizer::{
    LayerSpec,
    ConvolutionLayerSpec,
    Model,
    Dim3,
    PaddingMode,
};

#[tokio::main]
async fn main() {
    let mut model = Model::new(None).await;
    model.add_layer(LayerSpec::Convolution(ConvolutionLayerSpec {
        nb_kernel: 10,
        dim_kernel: Dim3 {x:1, y:1, z:1},
        stripe: 1,
        mode: PaddingMode::Valid,
        dim_input: Some(Dim3 {x:200, y:20, z:1}),
    }));
    model.add_layer(LayerSpec::Convolution(ConvolutionLayerSpec {
        nb_kernel: 1,
        dim_kernel: Dim3 {x:1, y:1, z:10},
        stripe: 1,
        mode: PaddingMode::Valid,
        dim_input: None,
    }));
    model.add_layer(LayerSpec::Convolution(ConvolutionLayerSpec {
        nb_kernel: 1,
        dim_kernel: Dim3 {x:1, y:1, z:1},
        stripe: 1,
        mode: PaddingMode::Valid,
        dim_input: None,
    }));
    model.build_model();
    let res = model.infer(vec![1.; 200 * 20]).await;
    println! ("Result({}) : {:?}", res.len(), res);
    model.save("name");
}