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
//! use bat_building::GpuContext;
//! use bat_building::visualiser::spawn_window;
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
static VISUALISER_CMD_TX: OnceLock<Sender<ManagerCommand>> = OnceLock::new();

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

    let tx = visualiser_manager_tx();
    if let Err(err) = tx.send(ManagerCommand::Open(request)) {
        eprintln!("[visualiser] failed to enqueue open request: {err}");
        close_flag.store(true, Ordering::Relaxed);
        closed_flag.store(true, Ordering::Relaxed);
    }

    VisualiserHandle {
        _close_flag: close_flag,
        visible_flag,
        closed_flag,
    }
}

/// Ensure the background visualiser manager thread/event loop is running.
pub fn warmup_manager() {
    let _ = visualiser_manager_tx();
}

fn visualiser_manager_tx() -> &'static Sender<ManagerCommand> {
    VISUALISER_CMD_TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ManagerCommand>();
        std::thread::spawn(move || {
            open_window_manager(rx);
        });
        tx
    })
}

enum ManagerCommand {
    Open(OpenRequest),
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
    state: RenderState,
}

impl ActiveVisualiser {
    fn from_open_request(event_loop: &ActiveEventLoop, req: OpenRequest) -> Option<Self> {
        let attrs = Window::default_attributes()
            .with_title(req.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(512u32, 512u32))
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // 4 vertices → TriangleStrip → full-screen quad.
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
            }
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
        if let Some(active) = &self.active {
            if active.is_visible && !active.is_occluded {
                active.window.request_redraw();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal entry point
// ---------------------------------------------------------------------------

fn open_window_manager(rx: Receiver<ManagerCommand>) {
    // On Linux (Wayland and X11) winit requires `with_any_thread(true)` when
    // the event loop is created outside the main OS thread.  Both platform
    // extensions write to the same underlying `any_thread` flag, so importing
    // either one is sufficient.
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

        let try_build_x11 = || {
            let mut builder = EventLoop::builder();
            winit::platform::x11::EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
            winit::platform::x11::EventLoopBuilderExtX11::with_x11(&mut builder);
            builder.build()
        };

        let try_build_wayland = || {
            let mut builder = EventLoop::builder();
            winit::platform::wayland::EventLoopBuilderExtWayland::with_any_thread(
                &mut builder,
                true,
            );
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
                            return;
                        }
                    }
                } else {
                    eprintln!("[visualiser] failed to create event loop: {wayland_err}");
                    return;
                }
            }
        }
    };
    #[cfg(not(target_os = "linux"))]
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("[visualiser] failed to create event loop: {e}");
            return;
        }
    };

    let mut app = VisualiserManagerApp { rx, active: None };

    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("[visualiser] event loop error: {e}");
    }
}
