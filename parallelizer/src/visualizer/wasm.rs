// wasm.rs
use crate::visualizer::platform::PlatformWindow;
use crate::GpuContext;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowAttributes;
use wgpu::Surface;
use std::sync::Arc;

pub struct WasmWindow {
    pub window: winit::window::Window,
}

impl WasmWindow {
    pub fn create_in_resumed(event_loop: &ActiveEventLoop, _width: u32, _height: u32) -> Self {
        let window = event_loop
            .create_window(WindowAttributes::default().with_title("Visualizer"))
            .expect("Failed to create window");

        Self { window }
    }
}

impl PlatformWindow for WasmWindow {
    fn create_surface(&self, gpu: &Arc<GpuContext>) -> Surface<'_> {
        gpu.instance().create_surface(&self.window).unwrap()
    }

    fn request_redraw(&self) {
        // WASM redraws usually happen in a callback; you can leave empty or hook in JS
    }
}