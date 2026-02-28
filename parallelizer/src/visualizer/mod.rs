// Visualizer module declarations
pub mod platform;
pub mod desktop;
pub mod wasm;
pub mod visualizer;

// Re-export key types for easier access
pub use visualizer::Visualizer;

use std::sync::Arc;
use crate::visualizer::{desktop::DesktopWindow, platform::PlatformWindow};