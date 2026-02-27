use std::sync::Arc;
use parallelizer::{
    ActivationLayerSpec, ActivationMethod, ConvolutionLayerSpec, Dim3, LayerSpec, Model, PaddingMode,
};
use parallelizer_visualisation::Visualiser;

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
    let mut model = Model::new(None).await;
    model.add_layer(LayerSpec::Convolution(ConvolutionLayerSpec {
        nb_kernel: 1,
        dim_kernel: Dim3::new((1,1,1)),
        stripe: 1,
        mode: PaddingMode::Valid,
        dim_input: Some(Dim3::new((512,512, 1))),
    }));
    model.add_layer(LayerSpec::Activation(ActivationLayerSpec { method: ActivationMethod::Linear, dim_input: None }));
    model.build_model();

    // Load input data
    println!("Loading input image...");
    let input = load_image_as_f32("images/bear.jpg", 512, 512);

    // Prepare visualiser and launch before inference
    let layers = model.layers();
    if !layers.is_empty() {
        let first_layer = &layers[0];
        let last_layer = layers.last().unwrap();

        let gpu = model.gpu_context();
        let input_dim = first_layer.dim_input();
        let output_dim = last_layer.dim_output();

        // Clone buffer Arc handles for the visualizer
        let input_buf = first_layer.gpu_input_arc();
        let output_buf = last_layer.gpu_output_arc();

        let vis = Visualiser::new(&model);
        let _vis_handle = vis.spawn_with_buffers(
            gpu,
            input_buf,
            input_dim,
            output_buf,
            output_dim,
        );
        
        println!("Visualizer window spawned on background thread");
        println!("Loading input image...");
        let input = load_image_as_f32("images/bear.jpg", 512, 512);
        
        println!("Running inference...");
        let res = model.infer(input).await;
        println!("Result({}) : {:?}", res.len(), res);
        println!("Close the visualizer window to continue");
        
        // Wait for the window thread to finish before exiting.
        let _ = _vis_handle.join();
    }

    println!("Visual frame generated in ./visualisation_frames (PPM format)");

    // optionally open a live window showing the input and output side-by-side.
    // the call blocks until the window is closed.

    model.save("name");
}