use parallelizer::{
    ActivationMethod, ActivationType, ConvolutionType, Dim3, GpuContext, LayerTypes, Model,
    PaddingMode, visualizer::Visualizer,
};
use std::sync::Arc;
use winit::event_loop::EventLoop;

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

    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new((512, 512, 1)),
            10,
            Dim3::new((3, 3, 1)),
            1,
            PaddingMode::Valid,
        )))
        .expect("failed to add first convolution layer");
    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new((512, 512, 1)),
            10,
            Dim3::new((3, 3, 10)),
            1,
            PaddingMode::Same,
        )))
        .expect("failed to add second convolution layer");
    model
        .add_layer(LayerTypes::Activation(ActivationType::new(
            ActivationMethod::Linear,
            Dim3::default(),
        )))
        .expect("failed to add activation layer");
    model.build_model();
    println!("loading image");
    //let image = load_image_as_f32("images/bear.jpg", 512, 512);
    let image = vec![10.; 512 * 512];

    println!("Running inference");
    let result = model.infer_batch(image).await;

    // println!("{:?}", result);
    // dbg!(&model);
    let event_loop = EventLoop::new().unwrap();
    let mut visualizer = Visualizer::new(gpu.clone(), Arc::new(model));
    event_loop.run_app(&mut visualizer).unwrap();
}
