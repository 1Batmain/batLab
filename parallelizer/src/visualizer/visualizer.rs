// Visualizer.rs
use std::sync::{Arc, Mutex};
use wgpu::Buffer;
use crate::{GpuContext};
use crate::model::Model;
use crate::visualizer::platform::PlatformWindow;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

pub struct Visualizer {
    gpu: Arc<GpuContext>,
    model: Arc<Model>,  // Thread-safe access to model
    window: Option<Window>,
}

impl Visualizer
{
    pub fn new(gpu: Arc<GpuContext>, model: Arc<Model>) -> Self
    {
        Self {
            gpu, 
            model,
            window: None,
        }
    }
}
impl ApplicationHandler for Visualizer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Ma fenêtre");
        
        self.window = Some(
            event_loop.create_window(window_attributes).unwrap()
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }

}