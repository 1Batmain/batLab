//! `the_window` – live GPU visualiser for model output buffers.
//!
//! Opens a native window that renders the model's actual GPU output buffer
//! directly – no CPU readback, no snapshot, no extra copy.  The visualiser
//! shares the same wgpu `Device` / `Queue` as the training process so that
//! the fragment shader reads the live buffer contents on every frame.
//!
//! # Usage
//! ```no_run
//! use std::sync::Arc;
//! use bat_building::GpuContext;
//! use the_window::spawn_window;
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bat_building::GpuContext;

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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A handle to a running visualiser window.
///
/// Dropping the handle sends a close signal to the window thread, which
/// causes it to exit on the next event-loop iteration.
pub struct VisualiserHandle {
    _close_flag: Arc<AtomicBool>,
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
    let close_flag = Arc::new(AtomicBool::new(false));
    let close_flag_clone = Arc::clone(&close_flag);

    std::thread::spawn(move || {
        open_window(
            gpu,
            output_buf,
            width,
            height,
            channels,
            title,
            close_flag_clone,
        );
    });

    VisualiserHandle {
        _close_flag: close_flag,
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
}

impl RenderState {
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
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(self.gpu.device(), &self.config);
        }
    }

    fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(o) => o,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // Swapchain invalidated (resize/minimise/etc.) – reconfigure and
                // retry on the next frame.
                self.surface.configure(self.gpu.device(), &self.config);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                eprintln!("[the_window] GPU out of memory");
                return;
            }
            Err(e) => {
                eprintln!("[the_window] surface error: {e}");
                return;
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
    }
}

// ---------------------------------------------------------------------------
// Winit application
// ---------------------------------------------------------------------------

struct Viewer {
    gpu: Arc<GpuContext>,
    output_buf: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    channels: u32,
    title: String,
    close_flag: Arc<AtomicBool>,
    window: Option<Arc<Window>>,
    state: Option<RenderState>,
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(512u32, 512u32));

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("[the_window] failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        // Create a surface using the SAME wgpu instance that the training
        // device was created with, so they are compatible.
        let surface = match self.gpu.instance().create_surface(Arc::clone(&window)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[the_window] failed to create surface: {e}");
                event_loop.exit();
                return;
            }
        };

        let caps = surface.get_capabilities(self.gpu.adapter());
        if caps.formats.is_empty() {
            eprintln!("[the_window] training GPU adapter does not support surface presentation");
            event_loop.exit();
            return;
        }

        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            // Prefer FIFO (vsync) to avoid burning CPU/GPU during training.
            present_mode: caps
                .present_modes
                .iter()
                .find(|&&m| m == wgpu::PresentMode::Fifo)
                .copied()
                .unwrap_or(caps.present_modes[0]),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(self.gpu.device(), &config);

        let state = RenderState::new(
            Arc::clone(&self.gpu),
            surface,
            config,
            &self.output_buf,
            self.width,
            self.height,
            self.channels,
        );

        self.window = Some(window);
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    state.render();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Close when the VisualiserHandle has been dropped.
        if self.close_flag.load(Ordering::Relaxed) {
            event_loop.exit();
            return;
        }
        // Throttle to ~30 fps so the visualiser does not burn CPU/GPU during
        // training.  FIFO present mode provides additional vsync pacing.
        let next = Instant::now() + Duration::from_millis(FRAME_INTERVAL_MS);
        event_loop.set_control_flow(ControlFlow::WaitUntil(next));
        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }
}

// ---------------------------------------------------------------------------
// Internal entry point
// ---------------------------------------------------------------------------

fn open_window(
    gpu: Arc<GpuContext>,
    output_buf: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    channels: u32,
    title: String,
    close_flag: Arc<AtomicBool>,
) {
    // On Linux (Wayland and X11) winit requires `with_any_thread(true)` when
    // the event loop is created outside the main OS thread.  Both platform
    // extensions write to the same underlying `any_thread` flag, so importing
    // either one is sufficient.
    #[cfg(target_os = "linux")]
    let event_loop = {
        use winit::platform::wayland::EventLoopBuilderExtWayland as _;
        match EventLoop::builder().with_any_thread(true).build() {
            Ok(el) => el,
            Err(e) => {
                eprintln!("[the_window] failed to create event loop: {e}");
                return;
            }
        }
    };
    #[cfg(not(target_os = "linux"))]
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("[the_window] failed to create event loop: {e}");
            return;
        }
    };

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut viewer = Viewer {
        gpu,
        output_buf,
        width,
        height,
        channels,
        title,
        close_flag,
        window: None,
        state: None,
    };

    if let Err(e) = event_loop.run_app(&mut viewer) {
        eprintln!("[the_window] event loop error: {e}");
    }
}
