// Visualizer.rs
use crate::GpuContext;
use crate::model::Model;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

pub struct Visualizer {
    #[allow(dead_code)]
    gpu: Arc<GpuContext>,
    #[allow(dead_code)]
    model: Arc<Model>,
    window: Option<Window>,
}

impl Visualizer {
    pub fn new(gpu: Arc<GpuContext>, model: Arc<Model>) -> Self {
        Self {
            gpu,
            model,
            window: None,
        }
    }

    // pub fn create_surface(&mut self) {
    //     if let Some(window) = &self.window {
    //         self.surface = Some(
    //             self.gpu
    //                 .instance()
    //                 .create_surface(window)
    //                 .unwrap()
    //         );
    //     }
    // }
}

impl ApplicationHandler for Visualizer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("Ma fenêtre");

        self.window = Some(event_loop.create_window(window_attributes).unwrap());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}
