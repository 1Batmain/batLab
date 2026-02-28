// platform.rs
use wgpu::Surface;
use std::sync::Arc;

pub trait PlatformWindow {
    /// Create a window using the modern ActiveEventLoop API
    fn new(event_loop: &winit::event_loop::ActiveEventLoop, width: u32, height: u32) -> Self;
    
    fn create_surface(&self, gpu: &Arc<crate::GpuContext>) -> Surface<'_>;
    fn request_redraw(&self);
    fn window_id(&self) -> winit::window::WindowId;
}