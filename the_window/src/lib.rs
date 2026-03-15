//! `the_window` – live GPU visualiser for model output buffers.
//!
//! Opens a native window that renders the latest snapshot of the model's
//! pre-loss output tensor in real-time.  The visualiser runs in its own OS
//! thread and communicates with the training loop through a simple
//! `Sender<WindowFrame>` / `Receiver<WindowFrame>` channel pair.
//!
//! # Usage
//! ```no_run
//! use std::sync::mpsc;
//! use the_window::{WindowFrame, spawn_window};
//!
//! let (tx, rx) = mpsc::channel::<WindowFrame>();
//! let handle = spawn_window(rx, "Model Output".to_string());
//!
//! // from the training thread, after each step:
//! let _ = tx.send(WindowFrame {
//!     data: vec![0.5_f32; 32 * 32 * 3],
//!     width: 32,
//!     height: 32,
//!     channels: 3,
//! });
//!
//! // When training is done or the sender is dropped, the window closes.
//! drop(tx);
//! handle.join().ok();
//! ```

use std::sync::Arc;
use std::sync::mpsc::Receiver;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// A snapshot of the model's output tensor to be rendered in the window.
///
/// `data` is a flat `f32` slice in row-major, interleaved-channel order:
/// `data[(y * width + x) * channels + c]`.
/// Values are expected in roughly `[-1, 1]`; the shader normalises them to
/// `[0, 1]` for display.
#[derive(Debug, Clone)]
pub struct WindowFrame {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

/// Spawn a model-output visualiser window in a new OS thread.
///
/// The thread blocks on the winit event loop until:
/// - the user closes the window, or
/// - the `rx` channel is disconnected (all `Sender`s have been dropped).
///
/// Returns a `JoinHandle` that can be used to wait for the thread.
pub fn spawn_window(rx: Receiver<WindowFrame>, title: String) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || open_window(rx, title))
}

/// Open a model-output visualiser window, blocking the current thread.
///
/// Returns once the window is closed or the channel is disconnected.
pub fn open_window(rx: Receiver<WindowFrame>, title: String) {
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("[the_window] failed to create event loop: {e}");
            return;
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut viewer = ModelViewer {
        rx,
        title,
        window: None,
        gpu: None,
        pending: None,
    };

    if let Err(e) = event_loop.run_app(&mut viewer) {
        eprintln!("[the_window] event loop error: {e}");
    }
}

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

const SHADER_SRC: &str = include_str!("shader.wgsl");

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    data_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Currently configured frame dimensions `(width, height, channels)`.
    frame_dims: (u32, u32, u32),
}

impl GpuState {
    /// Create GPU state with an initial 1×1 black frame.
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model_viewer_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        // Bind-group layout: binding 0 = storage buffer (data), binding 1 = uniforms.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model_viewer_bgl"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("model_viewer_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("model_viewer_pipeline"),
            layout: Some(&pipeline_layout),
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

        // Start with a minimal 1×1 placeholder frame so all buffers are valid.
        let init_frame = WindowFrame {
            data: vec![0.0f32],
            width: 1,
            height: 1,
            channels: 1,
        };
        let (data_buf, uniform_buf, bind_group) = Self::build_buffers(&device, &bgl, &init_frame);

        GpuState {
            device,
            queue,
            surface,
            config,
            pipeline,
            bind_group_layout: bgl,
            data_buf,
            uniform_buf,
            bind_group,
            frame_dims: (1, 1, 1),
        }
    }

    /// Create GPU buffers and a matching bind group for `frame`.
    fn build_buffers(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        frame: &WindowFrame,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
        use wgpu::util::DeviceExt as _;

        let raw = bytemuck::cast_slice::<f32, u8>(&frame.data);
        // Ensure at least 4 bytes so the binding is never empty.
        let data_bytes: &[u8] = if raw.is_empty() { &[0u8; 4] } else { raw };

        let data_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model_data"),
            contents: data_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = [frame.width, frame.height, frame.channels, 0u32];
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model_uniforms"),
            contents: bytemuck::cast_slice::<u32, u8>(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        (data_buf, uniform_buf, bind_group)
    }

    /// Upload a new frame.  Recreates the storage buffer and bind group when
    /// the tensor dimensions change (common only at start-up).
    fn update_frame(&mut self, frame: &WindowFrame) {
        let new_dims = (frame.width, frame.height, frame.channels);
        let expected_bytes = (frame.data.len() * std::mem::size_of::<f32>()) as u64;

        if new_dims != self.frame_dims || expected_bytes != self.data_buf.size() {
            let (db, ub, bg) = Self::build_buffers(&self.device, &self.bind_group_layout, frame);
            self.data_buf = db;
            self.uniform_buf = ub;
            self.bind_group = bg;
            self.frame_dims = new_dims;
        } else {
            let raw = bytemuck::cast_slice::<f32, u8>(&frame.data);
            self.queue.write_buffer(&self.data_buf, 0, raw);
        }
    }

    fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(o) => o,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
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
            // 4 vertices → 2 triangles (TriangleStrip) → full-screen quad
            pass.draw(0..4, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

// ---------------------------------------------------------------------------
// Winit application
// ---------------------------------------------------------------------------

struct ModelViewer {
    rx: Receiver<WindowFrame>,
    title: String,
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    /// Latest unrendered frame received from the channel.
    pending: Option<WindowFrame>,
}

impl ApplicationHandler for ModelViewer {
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

        let instance = wgpu::Instance::default();

        let surface = match instance.create_surface(Arc::clone(&window)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[the_window] failed to create wgpu surface: {e}");
                event_loop.exit();
                return;
            }
        };

        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })) {
                Ok(a) => a,
                Err(_) => {
                    eprintln!("[the_window] no suitable GPU adapter found for window surface");
                    event_loop.exit();
                    return;
                }
            };

        let (device, queue) = match pollster::block_on(adapter.request_device(&Default::default()))
        {
            Ok(dq) => dq,
            Err(e) => {
                eprintln!("[the_window] failed to create wgpu device: {e}");
                event_loop.exit();
                return;
            }
        };

        let caps = surface.get_capabilities(&adapter);
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
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let gpu = GpuState::new(device, queue, surface, config);
        self.window = Some(window);
        self.gpu = Some(gpu);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                // Drain the channel, keeping only the most recent frame.
                while let Ok(frame) = self.rx.try_recv() {
                    self.pending = Some(frame);
                }

                if let (Some(gpu), Some(frame)) = (&mut self.gpu, self.pending.take()) {
                    gpu.update_frame(&frame);
                }

                if let Some(gpu) = &mut self.gpu {
                    gpu.render();
                }

                // Keep rendering at ~60 fps via continuous redraws.
                if let Some(win) = &self.window {
                    win.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Close the window when the sender side has been dropped.
        match self.rx.try_recv() {
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                event_loop.exit();
                return;
            }
            Ok(frame) => {
                self.pending = Some(frame);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }

        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }
}
