//! File purpose: Native front-end crate — the ratatui terminal UI, the winit
//! visualiser window, and the on-disk layout of `Models/` and `datasets/`.
//!
//! # Why this is a crate of its own
//! `batlab_core` is meant to run in a browser on the visitor's GPU (WebGPU via
//! wgpu). A terminal (ratatui, crossterm), a native window (winit) and a
//! filesystem are exactly the three things a browser does not have, so they are
//! quarantined here rather than in the engine. The dependency arrow only ever
//! points this way: `batlab_ui` → `batlab_core`, never back.

pub mod clock;
pub mod storage;
pub mod tui;
pub mod visualiser;

pub use tui::{MonitorOutcome, TrainingEvent};
pub use visualiser::{VisualiserHandle, run_on_main_thread, spawn_window};
