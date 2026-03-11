use parallelizer::{
    ActivationMethod, ActivationType, ConvolutionType, Dim3, GpuContext, LayerTypes, LossMethod,
    Model, PaddingMode,
};

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
    let gpu = std::sync::Arc::new(GpuContext::new_headless().await);
    let mut model = Model::new_training(gpu.clone(), 0.01, 1, LossMethod::MeanSquared).await;

    model
        .add_layer(LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new((10, 10, 1)),
            3,
            Dim3::new((2, 2, 1)),
            1,
            PaddingMode::Valid,
        )))
        .expect("failed to add first convolution layer");
    model
        .add_layer(LayerTypes::Activation(ActivationType::new(
            ActivationMethod::Linear,
            Dim3::default(),
        )))
        .expect("failed to add activation layer");
    model.build();
    // Input: 512x512x1; after valid 3x3 conv → 510x510x10
    let input = vec![0.5_f32; 10 * 10];
    let target = vec![1.0_f32; 10 * 10];
    println!("Training step...");
    for step in 0..5 {
        model.train_step(&input, &target);
        println!("Step {step} complete");
    }

    println!("\n{model:#?}");
}
