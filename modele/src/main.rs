use std::sync::Arc;
use parallelizer::{
    ActivationLayerSpec, ActivationMethod, ConvolutionLayerSpec, Dim3, GpuContext, LayerSpec, Model, PaddingMode, Visualizer,
};

fn load_image_as_f32(path :&str, width: u32, height: u32) -> Vec<f32>
{
    use image::{ ImageReader, imageops::FilterType };

    let image = ImageReader::open(path)
    .expect("Failed to open")
    .decode()
    .expect("failed to decode")
    .resize_exact(width, height, FilterType::Lanczos3)
    .to_rgb8();

    image.pixels()
    .flat_map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
    .collect()
}


#[tokio::main]
async fn main() {
    println!("Starting visualisation smoke test...");
    
    let gpu = Arc::new(GpuContext::new_headless().await);

    let mut model = Model::new(gpu, None).await;
    model.add_layer(LayerSpec::Convolution(ConvolutionLayerSpec {
        nb_kernel: 1,
        dim_kernel: Dim3::new((1,1,1)),
        stripe: 1,
        mode: PaddingMode::Valid,
        dim_input: Some(Dim3::new((512,512, 1))),
    }));
    model.add_layer(LayerSpec::Activation(ActivationLayerSpec { method: ActivationMethod::Linear, dim_input: None }));
    model.build_model();

    let vis = Visualiser::new(&model);
    
    println!("Visualizer window spawned on background thread");
    println!("Loading input image...");
    let input = load_image_as_f32("images/bear.jpg", 512, 512);
    
    println!("Running inference...");
    let res = model.infer(input).await;
    println!("Result({}) : {:?}", res.len(), res);
    println!("Close the visualizer window to continue");
    //let _ = _vis_handle.join();

}