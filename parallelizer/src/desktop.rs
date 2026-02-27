// desktop.rs
use super::PlatformWindow;
use winit::{
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use wgpu::{Device, Surface};
use std::sync::Arc;

pub struct DesktopWindow {
    pub window: Window,
}

impl DesktopWindow {
    pub fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> Self {
        let window = WindowBuilder::new()
            .with_title("Visualiser")
            .with_inner_size(winit::dpi::LogicalSize::new(width as f64, height as f64))
            .build(event_loop)
            .expect("Failed to create window");

        Self { window }
    }
}

impl PlatformWindow for DesktopWindow {
    fn create_surface(&self, device: &Arc<Device>) -> Surface {
        unsafe { device.instance().create_surface(&self.window) }.unwrap()
    }

    fn request_redraw(&self) {
        self.window.request_redraw();
    }
}