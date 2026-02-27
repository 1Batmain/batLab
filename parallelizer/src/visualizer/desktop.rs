// desktop.rs
use crate::visualizer::platform::PlatformWindow;
use crate::GpuContext;
use winit::{
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};
use wgpu::Surface;
use std::sync::Arc;

pub struct DesktopWindow {
    pub window: Window,
}

impl DesktopWindow {
    pub fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> Self {
        let window = WindowAttributes::default()
            .with_title("Visualizer")
            .with_inner_size(winit::dpi::LogicalSize::new(width as f64, height as f64));
        let window = event_loop.create_window(window).expect("Failed to create window");

        Self { window }
    }
}

impl PlatformWindow for DesktopWindow {
    fn create_surface(&self, gpu: &Arc<GpuContext>) -> Surface<'_> {
        gpu.instance().create_surface(&self.window).unwrap()
    }

    fn request_redraw(&self) {
        self.window.request_redraw();
    }
}