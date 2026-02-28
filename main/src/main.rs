use std::sync::{Arc, Mutex};
use std::sync::RwLock;
use parallelizer::visualizer;
use parallelizer::{
    ActivationLayerSpec, ActivationMethod, ConvolutionLayerSpec, Dim3, GpuContext, LayerSpec, Model, PaddingMode, visualizer::{Visualizer, desktop::DesktopWindow, platform::PlatformWindow}, 
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;

use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

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
    let gpu = Arc::new(GpuContext::new_headless().await);
    let mut model = Model::new(gpu.clone(), None).await;
    
    model.add_layer(LayerSpec::Convolution(ConvolutionLayerSpec {
        nb_kernel: 1,
        dim_kernel: Dim3::new((1,1,1)),
        stripe: 1,
        mode: PaddingMode::Valid,
        dim_input: Some(Dim3::new((512,512, 1))),
    }));
    model.add_layer(LayerSpec::Activation(ActivationLayerSpec { method: ActivationMethod::Linear, dim_input: None }));
    model.build_model();

    let image = load_image_as_f32("images/bear.jpg", 512, 512);
    model.infer(image).await;
    let model = Arc::new(model);
    
    let event_loop = EventLoop::new().unwrap();
    let mut visualizer = Visualizer::new(gpu.clone(), model.clone());
    event_loop.run_app(&mut visualizer).unwrap();
    

}