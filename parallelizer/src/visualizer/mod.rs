// Visualizer module declarations
pub mod platform;
pub mod desktop;
pub mod wasm;
pub mod visualizer;

// Re-export key types for easier access
pub use visualizer::Visualizer;

/// Creates an event loop and window for desktop visualization
pub fn create_desktop_event_loop() -> winit::event_loop::EventLoop<()> {
    winit::event_loop::EventLoop::new().unwrap()
}