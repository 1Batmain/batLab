//! File purpose: Manages the visualiser window lifecycle and GPU-backed frame presentation of model outputs.

//! Live GPU visualiser for model output buffers.
//!
//! Opens a native window that renders the model's actual GPU output buffer
//! directly – no CPU readback, no snapshot, no extra copy.  The visualiser
//! shares the same wgpu `Device` / `Queue` as the training process so that
//! the fragment shader reads the live buffer contents on every frame.
//!
//! # Usage
//! ```ignore
//! use std::sync::Arc;
//! use batlab_core::GpuContext;
//! use batlab_core::visualiser::spawn_window;
//!
//! // (during training, after model.build())
//! let gpu: Arc<GpuContext> = model.gpu_context();
//! let buf: Arc<wgpu::Buffer> = model.last_output_buffer().unwrap();
//!
//! let handle = spawn_window(
//!     gpu, buf,
//!     32, 32, 3,       // width, height, channels of the output tensor
//!     "Model Output".to_string(),
//! );
//! // Press [v] again to close → drops the handle → window exits.
//! drop(handle);
//! ```

mod live_frame;

pub use live_frame::{LiveFrame, compose_live_frame, live_frame_width};

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use crate::GpuContext;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Target frame interval for the visualiser (~30 fps).
/// FIFO present mode provides additional vsync pacing on top of this.
const FRAME_INTERVAL_MS: u64 = 33;
/// Poll interval when no visible visualiser is active.
const IDLE_INTERVAL_MS: u64 = 100;
/// Longest edge, in logical pixels, the window aims for when it opens.
///
/// The size is derived from the frame, never fixed: the window used to open at
/// a hard-coded 512×512 whatever it was about to show, which stretched every
/// non-square source. The live perpetual/inference frame is `2W + 3` wide by
/// `H` high (67×32 on this model), so a square window squashed it by 2.1× —
/// the pixels were rectangles and the images read as distorted.
const INITIAL_WINDOW_TARGET_EDGE: u32 = 560;
static VISUALISER_CMD_TX: OnceLock<Sender<ManagerCommand>> = OnceLock::new();

/// Colour of the letterbox bands, as a plain neutral.
///
/// Not black: the frame's own darkest pixel is black too, so black bands would
/// merge with the image and hide where it actually ends. Not mid-grey either,
/// which competes with the picture. This is the "outside the image" chrome.
///
/// **Values are linear, the surface is sRGB.** The clear colour is not passed
/// through the transfer function, so an innocent-looking `0.08` came out at
/// `sRGB 80/255` — a mid-grey that framed the image like a mount. Measured on a
/// screenshot, not assumed: these three land near `sRGB 28/255`.
const LETTERBOX_COLOUR: wgpu::Color = wgpu::Color {
    r: 0.0115,
    g: 0.0115,
    b: 0.0135,
    a: 1.0,
};

/// Size the window opens at: the largest **integer** multiple of the frame that
/// fits inside [`INITIAL_WINDOW_TARGET_EDGE`] on both axes.
///
/// An integer multiple is what makes every source pixel the same square block
/// of screen pixels — the point of the exercise. A 67×32 frame therefore opens
/// at ×8 = 536×256, not at a square 512×512.
fn initial_window_size(frame_width: u32, frame_height: u32) -> (u32, u32) {
    let (width, height) = (frame_width.max(1), frame_height.max(1));
    let scale = (INITIAL_WINDOW_TARGET_EDGE / width)
        .min(INITIAL_WINDOW_TARGET_EDGE / height)
        .max(1);
    (width * scale, height * scale)
}

/// The `(x, y, width, height)` viewport that fits `frame` inside `surface`
/// without distorting it — the same scale on both axes, centred, the remainder
/// left to the clear colour as bands.
///
/// Resizing the window used to stretch the quad to whatever shape the surface
/// had, because the shader draws a full-screen quad and nothing else set a
/// viewport. Letterboxing here rather than in the shader keeps the shader a
/// pure row-major indexer: the fragment stage never learns about aspect ratio,
/// it just gets asked for fewer pixels.
fn letterbox_viewport(surface: (u32, u32), frame: (u32, u32)) -> (f32, f32, f32, f32) {
    let (surface_w, surface_h) = (surface.0.max(1) as f32, surface.1.max(1) as f32);
    let (frame_w, frame_h) = (frame.0.max(1) as f32, frame.1.max(1) as f32);

    let scale = (surface_w / frame_w).min(surface_h / frame_h);
    let width = (frame_w * scale).min(surface_w).max(1.0);
    let height = (frame_h * scale).min(surface_h).max(1.0);

    // Floor the offsets so the viewport never spills past the attachment on the
    // far edge (wgpu rejects that outright).
    (
        ((surface_w - width) * 0.5).floor().max(0.0),
        ((surface_h - height) * 0.5).floor().max(0.0),
        width,
        height,
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A handle to a running visualiser window.
///
/// Dropping the handle sends a close signal to the window thread, which
/// causes it to exit on the next event-loop iteration.
///
/// The handle also tracks whether the window was closed by the user (e.g.
/// via the window's close button).  Poll [`VisualiserHandle::is_closed`] to
/// check this and clear the handle on the caller's side.
pub struct VisualiserHandle {
    /// Caller → window: request the window to close.
    _close_flag: Arc<AtomicBool>,
    /// Caller → window: request the window to show/hide.
    visible_flag: Arc<AtomicBool>,
    /// Window → caller: window has exited (either via close button or flag).
    closed_flag: Arc<AtomicBool>,
}

impl VisualiserHandle {
    /// Returns `true` if the visualiser window has already been closed,
    /// either because the user clicked the window's close button or because
    /// the handle was previously dropped and the thread has since exited.
    pub fn is_closed(&self) -> bool {
        self.closed_flag.load(Ordering::Relaxed)
    }

    /// Set whether the visualiser window should be visible.
    pub fn set_visible(&self, visible: bool) {
        self.visible_flag.store(visible, Ordering::Relaxed);
    }
}

impl Drop for VisualiserHandle {
    fn drop(&mut self) {
        self._close_flag.store(true, Ordering::Relaxed);
    }
}

/// Spawn the model-output visualiser in a new OS thread.
///
/// The window renders `output_buf` – the model's actual GPU output buffer –
/// directly on every frame via a read-only storage binding.  No CPU copy is
/// performed.
///
/// The window closes when:
/// - the user clicks the window's close button, or
/// - the returned [`VisualiserHandle`] is dropped (e.g. user presses `[v]`
///   again to toggle off).
///
/// `width`, `height`, `channels` describe the tensor layout inside
/// `output_buf` (row-major, interleaved channels).  Values are expected in
/// roughly `[-1, 1]`; the shader normalises them to `[0, 1]` for display.
pub fn spawn_window(
    gpu: Arc<GpuContext>,
    output_buf: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    channels: u32,
    title: String,
) -> VisualiserHandle {
    spawn_window_with_visibility(gpu, output_buf, width, height, channels, title, true)
}

/// Spawn a visualiser window with an explicit initial visibility.
pub fn spawn_window_with_visibility(
    gpu: Arc<GpuContext>,
    output_buf: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    channels: u32,
    title: String,
    initial_visible: bool,
) -> VisualiserHandle {
    let close_flag = Arc::new(AtomicBool::new(false));
    let visible_flag = Arc::new(AtomicBool::new(initial_visible));
    let closed_flag = Arc::new(AtomicBool::new(false));

    let request = OpenRequest {
        gpu,
        output_buf,
        width,
        height,
        channels,
        title,
        close_flag: Arc::clone(&close_flag),
        visible_flag: Arc::clone(&visible_flag),
        closed_flag: Arc::clone(&closed_flag),
        initial_visible,
    };

    match visualiser_manager_tx() {
        Some(tx) => {
            if let Err(err) = tx.send(ManagerCommand::Open(request)) {
                eprintln!("[visualiser] failed to enqueue open request: {err}");
                close_flag.store(true, Ordering::Relaxed);
                closed_flag.store(true, Ordering::Relaxed);
            }
        }
        None => {
            eprintln!(
                "[visualiser] no manager running; the process must be started via \
                 visualiser::run_on_main_thread"
            );
            close_flag.store(true, Ordering::Relaxed);
            closed_flag.store(true, Ordering::Relaxed);
        }
    }

    VisualiserHandle {
        _close_flag: close_flag,
        visible_flag,
        closed_flag,
    }
}

/// No-op kept for API compatibility.
///
/// The manager event loop is now started by [`run_on_main_thread`] before any
/// application code runs, so there is nothing left to warm up.
pub fn warmup_manager() {
    if visualiser_manager_tx().is_none() {
        eprintln!(
            "[visualiser] manager not running; visualiser windows will be unavailable \
             (start the process via visualiser::run_on_main_thread)"
        );
    }
}

fn visualiser_manager_tx() -> Option<&'static Sender<ManagerCommand>> {
    VISUALISER_CMD_TX.get()
}

/// Run `worker` on a background thread while the visualiser's winit event loop
/// owns the calling thread.
///
/// This **must** be called from the process main thread: on macOS AppKit
/// requires `NSApplication` (and therefore winit's event loop) to live on the
/// main thread, and winit panics outright otherwise. Windows and Linux have the
/// same expectation, merely less strictly enforced.
///
/// `worker` receives the whole application (TUI, training, inference); the event
/// loop exits once it returns, so this function returns when the application is
/// done.
pub fn run_on_main_thread<F>(worker: F)
where
    F: FnOnce() + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<ManagerCommand>();
    if VISUALISER_CMD_TX.set(tx).is_err() {
        eprintln!("[visualiser] manager already running; refusing to start a second event loop");
        worker();
        return;
    }
    let shutdown_tx = visualiser_manager_tx()
        .expect("manager sender was just installed")
        .clone();

    // Sending `Shutdown` from a drop guard means a panicking worker still
    // releases the event loop instead of hanging the process.
    let worker_thread = std::thread::spawn(move || {
        let _guard = ShutdownOnDrop(shutdown_tx);
        worker();
    });

    let Some(event_loop) = build_event_loop() else {
        // No GUI backend: drop the receiver so later `spawn_window` calls fail
        // fast, and simply let the application run to completion.
        drop(rx);
        let _ = worker_thread.join();
        return;
    };

    let mut app = VisualiserManagerApp {
        rx,
        active: None,
        shutdown: false,
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("[visualiser] event loop error: {e}");
    }

    let _ = worker_thread.join();
}

struct ShutdownOnDrop(Sender<ManagerCommand>);

impl Drop for ShutdownOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(ManagerCommand::Shutdown);
    }
}

enum ManagerCommand {
    Open(OpenRequest),
    Shutdown,
}

struct OpenRequest {
    gpu: Arc<GpuContext>,
    output_buf: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    channels: u32,
    title: String,
    close_flag: Arc<AtomicBool>,
    visible_flag: Arc<AtomicBool>,
    closed_flag: Arc<AtomicBool>,
    initial_visible: bool,
}

struct ActiveVisualiser {
    close_flag: Arc<AtomicBool>,
    visible_flag: Arc<AtomicBool>,
    closed_flag: Arc<AtomicBool>,
    window: Arc<Window>,
    is_visible: bool,
    is_occluded: bool,
    /// When the last redraw was asked for, used to hold the frame rate to
    /// `FRAME_INTERVAL_MS` (see `about_to_wait`).
    last_frame_request: Instant,
    state: RenderState,
}

impl ActiveVisualiser {
    fn from_open_request(event_loop: &ActiveEventLoop, req: OpenRequest) -> Option<Self> {
        let (window_w, window_h) = initial_window_size(req.width, req.height);
        let attrs = Window::default_attributes()
            .with_title(req.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(window_w, window_h))
            .with_visible(req.initial_visible);

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("[visualiser] failed to create window: {e}");
                req.closed_flag.store(true, Ordering::Relaxed);
                return None;
            }
        };

        // Create a surface using the SAME wgpu instance that the training
        // device was created with, so they are compatible.
        let surface = match req.gpu.instance().create_surface(Arc::clone(&window)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[visualiser] failed to create surface: {e}");
                req.closed_flag.store(true, Ordering::Relaxed);
                return None;
            }
        };

        let size = window.inner_size();
        if let Some(mut config) =
            surface.get_default_config(req.gpu.adapter(), size.width.max(1), size.height.max(1))
        {
            let caps = surface.get_capabilities(req.gpu.adapter());
            if !caps.formats.is_empty() {
                let format = caps
                    .formats
                    .iter()
                    .find(|f| f.is_srgb())
                    .copied()
                    .unwrap_or(caps.formats[0]);

                config.format = format;
                config.present_mode = caps
                    .present_modes
                    .iter()
                    .find(|&&m| m == wgpu::PresentMode::Fifo)
                    .copied()
                    .unwrap_or(config.present_mode);
                config.alpha_mode = caps
                    .alpha_modes
                    .first()
                    .copied()
                    .unwrap_or(config.alpha_mode);
                config.desired_maximum_frame_latency = 2;

                let configured = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    surface.configure(req.gpu.device(), &config);
                }));
                if configured.is_ok() {
                    let state = RenderState::new(
                        Arc::clone(&req.gpu),
                        surface,
                        config,
                        &req.output_buf,
                        req.width,
                        req.height,
                        req.channels,
                    );
                    return Some(Self {
                        close_flag: req.close_flag,
                        visible_flag: req.visible_flag,
                        closed_flag: req.closed_flag,
                        window,
                        is_visible: req.initial_visible,
                        is_occluded: false,
                        last_frame_request: Instant::now(),
                        state,
                    });
                }
            }
        }

        let adapter_info = req.gpu.adapter().get_info();
        eprintln!(
            "[visualiser] failed to start shared-GPU visualiser (adapter='{}' backend={:?}); closing visualiser",
            adapter_info.name, adapter_info.backend
        );
        req.closed_flag.store(true, Ordering::Relaxed);
        None
    }
}

// ---------------------------------------------------------------------------
// WGSL shader
// ---------------------------------------------------------------------------

const SHADER_SRC: &str = include_str!("shader.wgsl");

// ---------------------------------------------------------------------------
// Render state – uses the SHARED wgpu device from GpuContext
// ---------------------------------------------------------------------------

struct RenderState {
    /// Shared with the training process.
    gpu: Arc<GpuContext>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    /// Geometry of the source frame, kept so every frame can be letterboxed
    /// into whatever shape the window currently has.
    frame_size: (u32, u32),
    // Keep uniforms alive for the lifetime of the bind group/pipeline.
    _uniform_buf: wgpu::Buffer,
}

impl RenderState {
    fn configure_surface(&mut self, reason: &str) -> bool {
        // On some Wayland setups, surface configuration failures are raised via
        // wgpu's uncaptured-error panic path instead of a recoverable Result.
        // Keep this contained to the visualiser thread.
        let configured = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.surface.configure(self.gpu.device(), &self.config);
        }));
        if configured.is_err() {
            eprintln!(
                "[visualiser] failed to configure surface ({reason}): invalid or incompatible surface"
            );
            return false;
        }
        true
    }

    fn new(
        gpu: Arc<GpuContext>,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        output_buf: &Arc<wgpu::Buffer>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> Self {
        let device = gpu.device();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visualiser_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("visualiser_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visualiser_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("visualiser_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Build a uniform buffer with the tensor dimensions.
        let uniforms = [width, height, channels, 0u32];
        use wgpu::util::DeviceExt as _;
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visualiser_uniforms"),
            contents: bytemuck::cast_slice::<u32, u8>(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visualiser_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // Bind the model's actual output buffer directly – read-only.
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        RenderState {
            gpu,
            surface,
            config,
            pipeline,
            bind_group,
            frame_size: (width, height),
            _uniform_buf: uniform_buf,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) -> bool {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            return self.configure_surface("resize");
        }
        true
    }

    fn render(&mut self) -> bool {
        let output = match self.surface.get_current_texture() {
            Ok(o) => o,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // Swapchain invalidated (resize/minimise/etc.) – reconfigure and
                // retry on the next frame.
                return self.configure_surface("surface lost/outdated");
            }
            Err(wgpu::SurfaceError::Timeout) => {
                // The GPU is busy (e.g. training is running on the same
                // device).  This is a transient stall – skip the frame and
                // retry on the next tick rather than flooding stderr.
                return true;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                eprintln!("[visualiser] GPU out of memory");
                return false;
            }
            Err(e) => {
                eprintln!("[visualiser] surface error: {e}");
                return false;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("visualiser_encoder"),
                });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("visualiser_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // The clear covers the whole attachment; the viewport
                        // below only lets the quad rasterise over the part that
                        // keeps the frame's aspect ratio, so what is left of
                        // this colour *is* the letterbox.
                        load: wgpu::LoadOp::Clear(LETTERBOX_COLOUR),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            let (x, y, w, h) =
                letterbox_viewport((self.config.width, self.config.height), self.frame_size);
            pass.set_viewport(x, y, w, h, 0.0, 1.0);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // 4 vertices → TriangleStrip → a quad filling the viewport, which is
            // the frame's own shape rather than the window's.
            pass.draw(0..4, 0..1);
        }
        // Submit to the SHARED queue – training compute commands and render
        // commands are ordered by the GPU, so the render always sees the
        // latest buffer state written by the last training pass.
        self.gpu.queue().submit(std::iter::once(encoder.finish()));
        output.present();
        true
    }
}

// ---------------------------------------------------------------------------
// Winit application
// ---------------------------------------------------------------------------

struct VisualiserManagerApp {
    rx: Receiver<ManagerCommand>,
    active: Option<ActiveVisualiser>,
    /// Set once the application worker has finished; the loop then exits.
    shutdown: bool,
}

impl VisualiserManagerApp {
    fn close_active(&mut self) {
        if let Some(active) = self.active.take() {
            active.closed_flag.store(true, Ordering::Relaxed);
        }
    }

    fn drain_commands(&mut self, event_loop: &ActiveEventLoop) {
        let mut pending_open: Option<OpenRequest> = None;
        while let Ok(cmd) = self.rx.try_recv() {
            match cmd {
                ManagerCommand::Open(req) => {
                    if let Some(previous) = pending_open.replace(req) {
                        previous.closed_flag.store(true, Ordering::Relaxed);
                    }
                }
                ManagerCommand::Shutdown => {
                    self.shutdown = true;
                    if let Some(req) = pending_open.take() {
                        req.closed_flag.store(true, Ordering::Relaxed);
                    }
                }
            }
        }
        if self.shutdown {
            self.close_active();
            event_loop.exit();
            return;
        }
        if let Some(open) = pending_open {
            self.close_active();
            self.active = ActiveVisualiser::from_open_request(event_loop, open);
        }
    }
}

impl ApplicationHandler for VisualiserManagerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.drain_commands(event_loop);
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let is_active_window = self.active.as_ref().is_some_and(|a| a.window.id() == id);
        if !is_active_window {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                self.close_active();
            }
            WindowEvent::Occluded(occluded) => {
                if let Some(active) = self.active.as_mut() {
                    active.is_occluded = occluded;
                }
            }
            WindowEvent::Resized(size) => {
                let should_close = self
                    .active
                    .as_mut()
                    .is_some_and(|active| !active.state.resize(size));
                if should_close {
                    self.close_active();
                }
            }
            WindowEvent::RedrawRequested => {
                let should_close = self
                    .active
                    .as_mut()
                    .is_some_and(|active| !active.is_occluded && !active.state.render());
                if should_close {
                    self.close_active();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let should_close = self
            .active
            .as_ref()
            .is_some_and(|active| active.close_flag.load(Ordering::Relaxed));
        if should_close {
            self.close_active();
        }
        self.drain_commands(event_loop);
        if self.shutdown {
            return;
        }

        if let Some(active) = self.active.as_mut() {
            let desired_visible = active.visible_flag.load(Ordering::Relaxed);
            if desired_visible != active.is_visible {
                active.window.set_visible(desired_visible);
                active.is_visible = desired_visible;
            }
        }

        let interval_ms = if self
            .active
            .as_ref()
            .is_some_and(|a| a.is_visible && !a.is_occluded)
        {
            FRAME_INTERVAL_MS
        } else {
            IDLE_INTERVAL_MS
        };
        let next = Instant::now() + Duration::from_millis(interval_ms);
        event_loop.set_control_flow(ControlFlow::WaitUntil(next));
        if let Some(active) = self.active.as_mut() {
            // `WaitUntil` only sets a deadline to wake up *by*; it does not stop
            // the loop waking earlier, and on macOS the display link drives
            // `about_to_wait` at the screen's refresh rate. Requesting a redraw
            // on every pass therefore rendered at 120 fps here, not the 30 fps
            // intended — and since the visualiser shares one GPU with the model,
            // that cost inference two thirds of its sampling throughput
            // (measured: 332 steps/s hidden vs 111 visible). Gating on elapsed
            // time is what actually enforces the budget.
            let due = active.last_frame_request.elapsed() >= Duration::from_millis(interval_ms);
            if active.is_visible && !active.is_occluded && due {
                active.last_frame_request = Instant::now();
                active.window.request_redraw();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal entry point
// ---------------------------------------------------------------------------

/// Build the winit event loop for the calling (main) thread.
///
/// Returns `None` when no usable windowing backend is available; the caller then
/// runs the application without a visualiser instead of failing outright.
fn build_event_loop() -> Option<EventLoop<()>> {
    #[cfg(target_os = "linux")]
    let event_loop = {
        let mut try_x11_first = false;

        // On some NVIDIA+Wayland setups, forcing X11/XWayland is more
        // reliable for swapchain configuration than native Wayland.
        if std::env::var("XDG_SESSION_TYPE").ok().as_deref() == Some("wayland")
            && std::env::var_os("DISPLAY").is_some()
        {
            try_x11_first = true;
        }

        // `with_any_thread` is deliberately absent: the loop now runs on the
        // main thread, which every winit backend expects.
        let try_build_x11 = || {
            let mut builder = EventLoop::builder();
            winit::platform::x11::EventLoopBuilderExtX11::with_x11(&mut builder);
            builder.build()
        };

        let try_build_wayland = || {
            let mut builder = EventLoop::builder();
            winit::platform::wayland::EventLoopBuilderExtWayland::with_wayland(&mut builder);
            builder.build()
        };

        let event_loop = if try_x11_first {
            match try_build_x11() {
                Ok(el) => Ok(el),
                Err(x11_err) => {
                    eprintln!(
                        "[visualiser] failed to init with X11 backend ({x11_err}); retrying Wayland backend"
                    );
                    try_build_wayland()
                }
            }
        } else {
            try_build_wayland()
        };

        match event_loop {
            Ok(el) => el,
            Err(wayland_err) => {
                if !try_x11_first {
                    match try_build_x11() {
                        Ok(el) => {
                            eprintln!(
                                "[visualiser] failed to init Wayland backend ({wayland_err}); using X11 backend"
                            );
                            el
                        }
                        Err(x11_err) => {
                            eprintln!(
                                "[visualiser] failed to create event loop: Wayland={wayland_err}; X11={x11_err}"
                            );
                            return None;
                        }
                    }
                } else {
                    eprintln!("[visualiser] failed to create event loop: {wayland_err}");
                    return None;
                }
            }
        }
    };

    // The TUI owns the terminal and stays the primary interface, so the
    // visualiser registers as an accessory: it shows its window without adding
    // a dock icon or stealing focus from the terminal at startup.
    #[cfg(target_os = "macos")]
    let event_loop = {
        use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
        let mut builder = EventLoop::builder();
        builder.with_activation_policy(ActivationPolicy::Accessory);
        match builder.build() {
            Ok(el) => el,
            Err(e) => {
                eprintln!("[visualiser] failed to create event loop: {e}");
                return None;
            }
        }
    };

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("[visualiser] failed to create event loop: {e}");
            return None;
        }
    };

    Some(event_loop)
}

#[cfg(test)]
mod tests {
    use super::{INITIAL_WINDOW_TARGET_EDGE, initial_window_size, letterbox_viewport};

    /// Aspect ratio of a viewport, to compare against the source's.
    fn ratio(w: f32, h: f32) -> f32 {
        w / h
    }

    /// The live frame of this project: two 32×32 panes and a 3-column rule.
    const LIVE_FRAME: (u32, u32) = (67, 32);

    /// The reported defect: the window opened square and stretched the frame.
    /// ×8 is the largest integer scale that fits the target edge, and 536×256
    /// has exactly the frame's proportions.
    #[test]
    fn the_window_opens_at_an_integer_multiple_of_the_frame() {
        let (w, h) = initial_window_size(LIVE_FRAME.0, LIVE_FRAME.1);
        assert_eq!((w, h), (536, 256));
        assert_eq!(w % LIVE_FRAME.0, 0, "width is not a whole number of panes");
        assert_eq!(h % LIVE_FRAME.1, 0);
        assert_eq!(
            w / LIVE_FRAME.0,
            h / LIVE_FRAME.1,
            "the two axes were scaled differently — that is the distortion itself"
        );
    }

    #[test]
    fn the_opening_size_stays_within_the_target_edge_for_any_frame() {
        for frame in [(67u32, 32u32), (32, 32), (1, 1), (0, 0), (700, 3), (4, 900)] {
            let (w, h) = initial_window_size(frame.0, frame.1);
            assert!(w >= 1 && h >= 1, "degenerate window for {frame:?}");
            let fits = w <= INITIAL_WINDOW_TARGET_EDGE.max(frame.0.max(1))
                && h <= INITIAL_WINDOW_TARGET_EDGE.max(frame.1.max(1));
            assert!(fits, "{frame:?} opened at {w}×{h}");
        }
    }

    /// A frame already in the window's proportions must use the whole surface:
    /// letterboxing that bands a correctly-shaped window would be a regression
    /// of its own.
    #[test]
    fn a_matching_surface_is_filled_edge_to_edge() {
        let (x, y, w, h) = letterbox_viewport((536, 256), LIVE_FRAME);
        assert_eq!((x, y, w, h), (0.0, 0.0, 536.0, 256.0));
    }

    /// The resize case the user hits: whatever shape the window is dragged to,
    /// the drawn area keeps the frame's ratio and is centred, and the leftover
    /// is symmetric bands.
    #[test]
    fn resizing_letterboxes_instead_of_stretching() {
        let frame_ratio = ratio(LIVE_FRAME.0 as f32, LIVE_FRAME.1 as f32);

        // Too tall: bands above and below.
        let (x, y, w, h) = letterbox_viewport((536, 800), LIVE_FRAME);
        assert!((ratio(w, h) - frame_ratio).abs() < 1e-4, "{w}×{h}");
        assert_eq!(x, 0.0, "no horizontal band when width is the binding edge");
        assert!(y > 0.0 && (y + h) <= 800.0, "band at y={y}, height={h}");
        assert!(
            (y - (800.0 - h - y)).abs() <= 1.0,
            "bands are not the same size: {y} vs {}",
            800.0 - h - y
        );

        // Too wide: bands left and right.
        let (x, y, w, h) = letterbox_viewport((1600, 256), LIVE_FRAME);
        assert!((ratio(w, h) - frame_ratio).abs() < 1e-4, "{w}×{h}");
        assert_eq!(y, 0.0);
        assert!(x > 0.0 && (x + w) <= 1600.0);
    }

    /// wgpu rejects a viewport that leaves the attachment, so the arithmetic
    /// has to hold for shapes nobody would choose on purpose too.
    #[test]
    fn the_viewport_never_leaves_the_surface() {
        let surfaces = [(1u32, 1u32), (0, 0), (3, 1000), (1000, 3), (537, 257)];
        for surface in surfaces {
            for frame in [LIVE_FRAME, (32, 32), (1, 1)] {
                let (x, y, w, h) = letterbox_viewport(surface, frame);
                let (sw, sh) = (surface.0.max(1) as f32, surface.1.max(1) as f32);
                assert!(
                    w > 0.0 && h > 0.0,
                    "empty viewport for {surface:?}/{frame:?}"
                );
                assert!(
                    x >= 0.0 && y >= 0.0 && x + w <= sw && y + h <= sh,
                    "viewport ({x},{y},{w},{h}) escapes {surface:?}"
                );
            }
        }
    }
}
