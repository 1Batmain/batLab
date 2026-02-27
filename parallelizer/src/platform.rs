// platform.rs
use wgpu::Device;
use wgpu::Queue;
use wgpu::Surface;
use std::sync::Arc;

pub trait PlatformWindow {
    fn create_surface(&self, device: &Arc<Device>) -> Surface;
    fn request_redraw(&self);
}