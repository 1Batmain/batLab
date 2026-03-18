//! File purpose: Owns visualiser lifecycle from TUI and provides a simple public control API.

use crate::GpuContext;
use crate::visualiser::{VisualiserHandle, spawn_window_with_visibility, warmup_manager};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone)]
struct VisualiserSource {
    gpu: Arc<GpuContext>,
    output_buf: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    channels: u32,
    title: String,
}

#[derive(Default)]
struct VisualiserController {
    source: Option<VisualiserSource>,
    handle: Option<VisualiserHandle>,
    desired_visible: bool,
}

static CONTROLLER: OnceLock<Mutex<VisualiserController>> = OnceLock::new();

fn controller() -> &'static Mutex<VisualiserController> {
    CONTROLLER.get_or_init(|| Mutex::new(VisualiserController::default()))
}

fn spawn_from_source(source: &VisualiserSource, visible: bool) -> VisualiserHandle {
    spawn_window_with_visibility(
        Arc::clone(&source.gpu),
        Arc::clone(&source.output_buf),
        source.width,
        source.height,
        source.channels,
        source.title.clone(),
        visible,
    )
}

fn same_source(a: &VisualiserSource, b: &VisualiserSource) -> bool {
    Arc::ptr_eq(&a.gpu, &b.gpu)
        && Arc::ptr_eq(&a.output_buf, &b.output_buf)
        && a.width == b.width
        && a.height == b.height
        && a.channels == b.channels
        && a.title == b.title
}

fn is_effectively_visible(ctrl: &VisualiserController) -> bool {
    ctrl.desired_visible && ctrl.handle.as_ref().is_some_and(|h| !h.is_closed())
}

/// Start the background visualiser manager/event loop early.
pub fn warmup_visualiser() {
    warmup_manager();
}

/// Register/update the model source used by the visualiser.
///
/// This prepares a window immediately using the current desired visibility so
/// later toggles are fast.
pub fn register_visualiser_source(
    gpu: Arc<GpuContext>,
    output_buf: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    channels: u32,
    title: String,
) {
    let mut ctrl = controller()
        .lock()
        .expect("visualiser controller mutex poisoned");

    let next_source = VisualiserSource {
        gpu,
        output_buf,
        width,
        height,
        channels,
        title,
    };

    let source_changed = ctrl
        .source
        .as_ref()
        .is_none_or(|existing| !same_source(existing, &next_source));
    if source_changed {
        ctrl.source = Some(next_source);
        let source = ctrl
            .source
            .as_ref()
            .expect("visualiser source must exist after registration");
        ctrl.handle = Some(spawn_from_source(source, ctrl.desired_visible));
        return;
    }

    if ctrl.handle.as_ref().is_some_and(|h| h.is_closed()) {
        let source = ctrl
            .source
            .as_ref()
            .expect("visualiser source must exist after registration");
        ctrl.handle = Some(spawn_from_source(source, ctrl.desired_visible));
    } else if let Some(handle) = ctrl.handle.as_ref() {
        handle.set_visible(ctrl.desired_visible);
    }
}

/// Clear registered visualiser source and close any active window.
pub fn clear_visualiser_source() {
    let mut ctrl = controller()
        .lock()
        .expect("visualiser controller mutex poisoned");
    ctrl.handle = None;
    ctrl.source = None;
    ctrl.desired_visible = false;
}

/// Set visualiser visibility.
pub fn set_visualiser_visible(visible: bool) -> Result<(), String> {
    let mut ctrl = controller()
        .lock()
        .map_err(|_| "visualiser controller mutex poisoned".to_string())?;

    ctrl.desired_visible = visible;

    if !visible {
        if let Some(handle) = ctrl.handle.as_ref() {
            handle.set_visible(false);
        }
        return Ok(());
    }

    if ctrl.source.is_none() {
        return Ok(());
    }

    let needs_respawn = ctrl.handle.as_ref().is_none_or(|h| h.is_closed());
    if needs_respawn {
        let source = ctrl
            .source
            .as_ref()
            .expect("visualiser source must exist when respawning");
        ctrl.handle = Some(spawn_from_source(source, true));
    } else if let Some(handle) = ctrl.handle.as_ref() {
        handle.set_visible(true);
    }
    Ok(())
}

/// Toggle visualiser visibility.
pub fn toggle_visualiser() -> Result<(), String> {
    let next = {
        let ctrl = controller()
            .lock()
            .map_err(|_| "visualiser controller mutex poisoned".to_string())?;
        !is_effectively_visible(&ctrl)
    };
    set_visualiser_visible(next)?;
    Ok(())
}
