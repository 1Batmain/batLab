// wasm.rs
use crate::GpuContext;
use crate::visualizer::platform::PlatformWindow;
use std::sync::Arc;
use wgpu::Surface;
use winit::window::WindowAttributes;

pub struct WasmWindow {
    pub window: winit::window::Window,
}

impl WasmWindow {
    // Constructor is now handled by the PlatformWindow trait methods
}

impl PlatformWindow for WasmWindow {
    fn new(event_loop: &winit::event_loop::ActiveEventLoop, width: u32, height: u32) -> Self {
        let window = WindowAttributes::default()
            .with_title("Model Visualizer")
            .with_inner_size(winit::dpi::LogicalSize::new(width as f64, height as f64))
            .with_visible(true);
        let window = event_loop
            .create_window(window)
            .expect("Failed to create window");

        Self { window }
    }

    fn create_surface(&self, gpu: &Arc<GpuContext>) -> Surface<'_> {
        gpu.instance().create_surface(&self.window).unwrap()
    }

    fn request_redraw(&self) {
        // WASM redraws usually happen in a callback; you can leave empty or hook in JS
    }

    fn window_id(&self) -> winit::window::WindowId {
        self.window.id()
    }
}
