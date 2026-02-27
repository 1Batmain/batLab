// wasm.rs
use super::PlatformWindow;
use winit::application::{ActiveEventLoop, WindowAttributes, Window};
use wgpu::Surface;
use wgpu::Device;
use std::sync::Arc;

pub struct WasmWindow {
    pub window: Window,
}

impl WasmWindow {
    pub fn create_in_resumed(event_loop: &ActiveEventLoop, width: u32, height: u32) -> Self {
        let window = event_loop
            .create_window(WindowAttributes::default().with_title("Visualiser"))
            .expect("Failed to create window");

        Self { window }
    }
}

impl PlatformWindow for WasmWindow {
    fn create_surface(&self, device: &Arc<Device>) -> Surface {
        unsafe { device.instance().create_surface(&self.window) }.unwrap()
    }

    fn request_redraw(&self) {
        // WASM redraws usually happen in a callback; you can leave empty or hook in JS
    }
}