// Visualizer.rs
use std::sync::Arc;
use wgpu::Buffer;
use crate::{GpuContext};
use crate::model::Model;
use crate::visualizer::platform::PlatformWindow;

pub struct Visualizer<P: PlatformWindow> {
    gpu: Arc<GpuContext>,
    model: Arc<Model>,
    window: Arc<P>,
    output_buf: Arc<Buffer>,
}

impl<P: PlatformWindow> Visualizer<P> {
    pub fn new(gpu: Arc<GpuContext>, model: Arc<Model>, window: Arc<P>) -> Self {
        // Prepare output buffer for rendering the model
        let last_layer = model.layers().last().expect("Model must have at least one layer");
        let output_buf = last_layer.output_arc();

        Self {
            gpu,
            model,
            window,
            output_buf,
        }
    }

    pub fn create_surface(&self) -> wgpu::Surface<'_> {
        self.window.create_surface(&self.gpu)
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn output_buffer(&self) -> Arc<Buffer> {
        self.output_buf.clone()
    }
}