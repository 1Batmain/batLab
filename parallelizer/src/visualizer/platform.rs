// platform.rs
use wgpu::Surface;
use std::sync::Arc;

pub trait PlatformWindow {
    fn create_surface(&self, gpu: &Arc<crate::GpuContext>) -> Surface<'_>;
    fn request_redraw(&self);
}