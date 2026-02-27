// visualiser.rs
use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, Surface};
use crate::{Model, GpuContext};
use crate::platform::PlatformWindow;

pub struct Visualiser<P: PlatformWindow> {
    gpu: Arc<GpuContext>,
    model: Arc<Model>,
    window: P,
    surface: Surface,
    output_buf: Arc<Buffer>,
}

impl<P: PlatformWindow> Visualiser<P> {
    pub fn new(gpu: Arc<GpuContext>, model: Arc<Model>, window: P) -> Self {
        let surface = window.create_surface(&gpu.device);

        // Prepare output buffer for rendering the model
        let output_buf = model.layer.last().buffer.gpu.output;

        Self {
            gpu,
            model,
            window,
            surface,
            output_buf: Arc::new(output_buf),
        }
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn output_buffer(&self) -> Arc<Buffer> {
        self.output_buf.clone()
    }
}