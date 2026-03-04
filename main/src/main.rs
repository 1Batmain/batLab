use parallelizer::{
    ActivationMethod, ActivationType, ConvolutionType, Dim3, GpuContext, Model, PaddingMode,
    model::layer::LayerTypes,
};
use std::sync::Arc;

#[allow(dead_code)]
fn load_image_as_f32(path: &str, width: u32, height: u32) -> Vec<f32> {
    use image::{ImageReader, imageops::FilterType};

    let image = ImageReader::open(path)
        .expect("Failed to open")
        .decode()
        .expect("failed to decode")
        .resize_exact(width, height, FilterType::Lanczos3)
        .to_rgb8();

    image
        .pixels()
        .flat_map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let gpu = Arc::new(GpuContext::new_headless().await);
    let mut model = Model::new(gpu.clone()).await;

    model.add_layer(LayerTypes::Convolution(ConvolutionType {
        nb_kernel: 10,
        dim_kernel: Dim3::new((1, 1, 1)),
        stride: 1,
        mode: PaddingMode::Valid,
        dim_input: Some(Dim3::new((5, 5, 1))),
    }));
    model.add_layer(LayerSpec::Activation(ActivationType {
        method: ActivationMethod::Linear,
        dim_input: None,
    }));
    model.build_model();

    println!("loading image");
    //let image = load_image_as_f32("images/bear.jpg", 512, 512);
    let image = vec![10.; 25];

    println!("Running inference");
    let result = model.infer_batch(image).await;

    println!("{:?}", result);
    //let event_loop = EventLoop::new().unwrap();
    //let mut visualizer = Visualizer::new(gpu.clone(), model.clone());
    //event_loop.run_app(&mut visualizer).unwrap();
}
