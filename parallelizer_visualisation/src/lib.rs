use std::sync::Arc;
use std::{fs, path::PathBuf, thread};

use wgpu::{Buffer, Color, Device, Queue};

use parallelizer::{Model, ModelVisualState, Dim3};

#[derive(Debug)]
pub struct Visualiser {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter: Arc<wgpu::Adapter>,
    clear_color: Color,
    last_input_len: usize,
    last_output_len: usize,
    last_state: Option<ModelVisualState>,
    frame_id: u64,
    output_dir: PathBuf,
}

impl Visualiser {
    /// Helper creating a winit `EventLoop`.  On Linux the default builder
    /// panics if the loop is constructed off the main thread, so we enable
    /// `any_thread` via the appropriate extension trait.
    fn make_event_loop() -> winit::event_loop::EventLoop<()> {
        #[cfg(target_os = "linux")]
        {
            use winit::event_loop::{EventLoopBuilder, EventLoopBuilderExtX11, EventLoopBuilderExtWayland};
            let mut builder = EventLoopBuilder::new();
            // extension trait method available whichever backend is active
            builder.any_thread();
            builder.build().expect("failed to create event loop")
        }
        #[cfg(not(target_os = "linux"))]
        {
            winit::event_loop::EventLoop::new().expect("failed to create event loop")
        }
    }

    /// Construct a visualiser tied to the provided model. Only a shared
    /// reference to the underlying device/queue is kept; the model itself is
    /// passed to `capture` when a snapshot is required.
    pub fn new(model: &Model) -> Self {
        let gpu = model.gpu_context();
        Self {
            device: Arc::new(gpu.device().clone()),
            queue: Arc::new(gpu.queue().clone()),
            adapter: Arc::new(gpu.adapter().clone()),
            clear_color: Color::BLACK,
            last_input_len: 0,
            last_output_len: 0,
            last_state: None,
            frame_id: 0,
            output_dir: PathBuf::from("visualisation_frames"),
        }
    }

    fn channel_value(data: &[f32], dim: Dim3, x: u32, y: u32, c: u32) -> f32 {
        if x >= dim.x || y >= dim.y || c >= dim.z {
            return 0.0;
        }
        let idx = ((y * dim.x + x) * dim.z + c) as usize;
        data.get(idx).copied().unwrap_or(0.0)
    }

    fn normalize(value: f32, min_value: f32, max_value: f32) -> u8 {
        if (max_value - min_value).abs() < f32::EPSILON {
            return 0;
        }
        let scaled = ((value - min_value) / (max_value - min_value)).clamp(0.0, 1.0);
        (scaled * 255.0) as u8
    }

    fn tensor_min_max(data: &[f32]) -> (f32, f32) {
        let mut min_value = f32::INFINITY;
        let mut max_value = f32::NEG_INFINITY;
        for value in data.iter().copied() {
            if value < min_value {
                min_value = value;
            }
            if value > max_value {
                max_value = value;
            }
        }
        if min_value.is_infinite() || max_value.is_infinite() {
            (0.0, 1.0)
        } else {
            (min_value, max_value)
        }
    }

    fn pixel_rgb(
        data: &[f32],
        dim: Dim3,
        x: u32,
        y: u32,
        min_value: f32,
        max_value: f32,
    ) -> [u8; 3] {
        if dim.z == 0 {
            return [0, 0, 0];
        }

        let red = Self::channel_value(data, dim, x, y, 0);
        let green = if dim.z > 1 {
            Self::channel_value(data, dim, x, y, 1)
        } else {
            red
        };
        let blue = if dim.z > 2 {
            Self::channel_value(data, dim, x, y, 2)
        } else if dim.z > 1 {
            (red + green) * 0.5
        } else {
            red
        };

        [
            Self::normalize(red, min_value, max_value),
            Self::normalize(green, min_value, max_value),
            Self::normalize(blue, min_value, max_value),
        ]
    }

    fn write_side_by_side_ppm(
        &mut self,
        input: &[f32],
        input_dim: Dim3,
        output: &[f32],
        output_dim: Dim3,
    ) {
        let in_width = input_dim.x.max(1);
        let in_height = input_dim.y.max(1);
        let out_width = output_dim.x.max(1);
        let out_height = output_dim.y.max(1);
        let width = in_width + out_width;
        let height = in_height.max(out_height);

        let (in_min, in_max) = Self::tensor_min_max(input);
        let (out_min, out_max) = Self::tensor_min_max(output);

        let mut rgb = vec![0_u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..in_width {
                if y < in_height {
                    let pixel = Self::pixel_rgb(input, input_dim, x, y, in_min, in_max);
                    let idx = ((y * width + x) * 3) as usize;
                    rgb[idx] = pixel[0];
                    rgb[idx + 1] = pixel[1];
                    rgb[idx + 2] = pixel[2];
                }
            }

            for x in 0..out_width {
                if y < out_height {
                    let pixel = Self::pixel_rgb(output, output_dim, x, y, out_min, out_max);
                    let idx = ((y * width + (x + in_width)) * 3) as usize;
                    rgb[idx] = pixel[0];
                    rgb[idx + 1] = pixel[1];
                    rgb[idx + 2] = pixel[2];
                }
            }
        }

        let _ = fs::create_dir_all(&self.output_dir);
        let file_path = self
            .output_dir
            .join(format!("inference_{:06}.ppm", self.frame_id));

        let mut bytes = Vec::with_capacity(32 + rgb.len());
        bytes.extend_from_slice(format!("P6\n{} {}\n255\n", width, height).as_bytes());
        bytes.extend_from_slice(&rgb);

        if fs::write(&file_path, bytes).is_ok() {
            println!("visualiser frame saved to {}", file_path.display());
        }
    }

    fn read_back_f32_buffer(&self, source: &Buffer, size_bytes: u64) -> Vec<f32> {
        if size_bytes == 0 {
            return Vec::new();
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(source, 0, &staging_buffer, 0, size_bytes);
        self.queue.submit([encoder.finish()]);

        let buffer_slice = staging_buffer.slice(..);
        let (gpu, cpu) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = gpu.send(result);
        });

        if let Err(error) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            panic!("failed to poll device while reading buffer: {}", error);
        }

        let map_result = pollster::block_on(async { cpu.await });
        match map_result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => panic!("failed to map readback buffer: {}", error),
            Err(_) => panic!("failed to receive map callback for readback buffer"),
        }

        let values = {
            let bytes = buffer_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };
        staging_buffer.unmap();
        values
    }

    /// Capture a snapshot of the first/last layer and write a frame.
    pub fn capture(&mut self, model: &Model) {
        let layers = model.layers();
        if layers.is_empty() {
            return;
        }
        let first = &layers[0];
        let last = layers.last().unwrap();

        let input = self.read_back_f32_buffer(first.gpu_input(), first.input_size_bytes());
        let output = self.read_back_f32_buffer(last.gpu_output(), last.output_size_bytes());
        let input_dim = first.dim_input();
        let output_dim = last.dim_output();
        let visual_state = model.visual_state();

        self.write_side_by_side_ppm(&input, input_dim, &output, output_dim);
        self.last_input_len = input.len();
        self.last_output_len = output.len();
        self.last_state = Some(visual_state);
        self.frame_id = self.frame_id.saturating_add(1);
    }

    /// Helper to build a raw RGB buffer that represents the input tensor and
    /// output tensor side‑by‑side.  Used by the real‑time window display.
    fn side_by_side_rgb(
        &self,
        input: &[f32],
        input_dim: Dim3,
        output: &[f32],
        output_dim: Dim3,
    ) -> (Vec<u8>, u32, u32) {
        let in_width = input_dim.x.max(1);
        let in_height = input_dim.y.max(1);
        let out_width = output_dim.x.max(1);
        let out_height = output_dim.y.max(1);
        let width = in_width + out_width;
        let height = in_height.max(out_height);

        let (in_min, in_max) = Self::tensor_min_max(input);
        let (out_min, out_max) = Self::tensor_min_max(output);

        let mut rgb = vec![0_u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..in_width {
                if y < in_height {
                    let pixel = Self::pixel_rgb(input, input_dim, x, y, in_min, in_max);
                    let idx = ((y * width + x) * 3) as usize;
                    rgb[idx] = pixel[0];
                    rgb[idx + 1] = pixel[1];
                    rgb[idx + 2] = pixel[2];
                }
            }
            for x in 0..out_width {
                if y < out_height {
                    let pixel = Self::pixel_rgb(output, output_dim, x, y, out_min, out_max);
                    let idx = ((y * width + (x + in_width)) * 3) as usize;
                    rgb[idx] = pixel[0];
                    rgb[idx + 1] = pixel[1];
                    rgb[idx + 2] = pixel[2];
                }
            }
        }
        (rgb, width, height)
    }

    // Shader is loaded from an external WGSL file so it can be edited easily.

    /// Run a live visualisation window that continuously updates using the
    /// GPU model buffers directly.  This uses the model's device/adapter so that
    /// no additional wgpu instance is created, and binds the input/output
    /// buffers into a simple shader pipeline.
    pub async fn run(mut self, model: &Model) {
        use wgpu::util::DeviceExt;
        use winit::dpi::PhysicalSize;
        use winit::event::{Event, WindowEvent};
        use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
        use winit::window::Window;

        let layers = model.layers();
        if layers.is_empty() {
            return;
        }
        let first = &layers[0];
        let last = layers.last().unwrap();

        let input_dim = first.dim_input();
        let output_dim = last.dim_output();
        let in_width = input_dim.x.max(1);
        let in_height = input_dim.y.max(1);
        let out_width = output_dim.x.max(1);
        let out_height = output_dim.y.max(1);
        let total_width = in_width + out_width;
        let total_height = in_height.max(out_height);

        // Pack dims into two vec4<u32> entries to satisfy WGSL uniform
        // alignment. Layout: [in_w,in_h,in_ch,0, out_w,out_h,out_ch,0]
        let dims_data: [u32; 8] = [
            in_width,
            in_height,
            input_dim.z,
            0,
            out_width,
            out_height,
            output_dim.z,
            0,
        ];
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims_uniform"),
            contents: bytemuck::cast_slice(&dims_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let event_loop = EventLoop::new().expect("failed to create event loop");
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(PhysicalSize::new(total_width, total_height))
                        .with_title("parallelizer visualiser"),
                )
                .expect("failed to create window"),
        );
        // keep a clone for the event loop so that the original `window` can still
        // be borrowed by the surface
        let window_for_loop = window.clone();

        // `gpu_context()` returns an `Arc<GpuContext>`; dereference to call method
        let surface = (&*model.gpu_context())
            .create_surface(&*window)
            .expect("failed to create surface");
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: total_width,
            height: total_height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&self.device, &config);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visualiser_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/show_output.wgsl").into()),
        });

        let bind_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("visualiser_bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visualiser_bind_group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                // Bind output at binding 1 (shader expects output_buf at 1)
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: last.gpu_output().as_entire_binding(),
                },
                // Keep input bound at 2 in case other shaders need it.
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: first.gpu_input().as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visualiser_pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            immediate_size: 0,
        });

        let render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("visualiser_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // run the event loop, wiring up our render pass.  explicitly
        // annotate the closure types so that `Event::RedrawRequested` and
        // friends are resolved correctly.
        event_loop
            .run(move |event: Event<()>, event_loop: &ActiveEventLoop| {
                event_loop.set_control_flow(ControlFlow::Poll);
                match event {
                    Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                        event_loop.exit();
                    }
                    Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                        let output = match surface.get_current_texture() {
                            Ok(o) => o,
                            Err(_) => return,
                        };
                        let view = output
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("visualiser_encoder"),
                        });
                        {
                            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("visualiser_render_pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(self.clear_color),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                multiview_mask: None,
                                occlusion_query_set: None,
                                timestamp_writes: None,
                            });
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_bind_group(0, &bind_group, &[]);
                            rpass.draw(0..6, 0..1);
                        }
                        self.queue.submit(Some(encoder.finish()));
                        output.present();
                        // request next frame immediately
                        window_for_loop.request_redraw();
                    }
                    _ => {}
                }
            })
            .ok();
    }

    /// Run visualiser using explicit buffers instead of a `Model` reference.
    /// This allows the visualiser to be spawned on a background thread even
    /// when `Model` is not `Send`/`Sync`.
    pub async fn run_with_buffers(
        mut self,
        gpu: Arc<parallelizer::GpuContext>,
        input_buf: Arc<Buffer>,
        input_dim: Dim3,
        output_buf: Arc<Buffer>,
        output_dim: Dim3,
    ) {
        use wgpu::util::DeviceExt;
        use winit::dpi::PhysicalSize;
        use winit::event::{Event, WindowEvent};
        use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
        use winit::window::Window;

        let in_width = input_dim.x.max(1);
        let in_height = input_dim.y.max(1);
        let out_width = output_dim.x.max(1);
        let out_height = output_dim.y.max(1);
        let total_width = in_width + out_width;
        let total_height = in_height.max(out_height);

        let dims_data: [u32; 8] = [
            in_width,
            in_height,
            input_dim.z,
            0,
            out_width,
            out_height,
            output_dim.z,
            0,
        ];
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims_uniform"),
            contents: bytemuck::cast_slice(&dims_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let event_loop = EventLoop::new().expect("failed to create event loop");
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(PhysicalSize::new(total_width, total_height))
                        .with_title("parallelizer visualiser"),
                )
                .expect("failed to create window"),
        );
        let window_for_loop = window.clone();

        let surface = gpu
            .create_surface(&*window)
            .expect("failed to create surface");
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: total_width,
            height: total_height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&self.device, &config);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visualiser_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/show_output.wgsl").into()),
        });

        let bind_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("visualiser_bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visualiser_bind_group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visualiser_pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            immediate_size: 0,
        });

        let render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("visualiser_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // run the event loop, wiring up our render pass.  explicitly
        // annotate the closure types so that `Event::RedrawRequested` and
        // friends are resolved correctly.
        event_loop
            .run(move |event: Event<()>, event_loop: &ActiveEventLoop| {
                event_loop.set_control_flow(ControlFlow::Poll);
                match event {
                    Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                        event_loop.exit();
                    }
                    Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                        let output = match surface.get_current_texture() {
                            Ok(o) => o,
                            Err(_) => return,
                        };
                        let view = output
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("visualiser_encoder"),
                        });
                        {
                            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("visualiser_render_pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(self.clear_color),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                multiview_mask: None,
                                occlusion_query_set: None,
                                timestamp_writes: None,
                            });
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_bind_group(0, &bind_group, &[]);
                            rpass.draw(0..6, 0..1);
                        }
                        self.queue.submit(Some(encoder.finish()));
                        output.present();
                        // request next frame immediately
                        window_for_loop.request_redraw();
                    }
                    _ => {}
                }
            })
            .ok();
    }

    /// Spawn the visualiser on a background thread using explicit buffers,
    /// This avoids requiring `Model` to be `Send`/`Sync` by capturing only the
    /// GPU context and buffer handles (which are `Arc` and shareable).
    pub fn spawn_with_buffers(
        self,
        gpu: Arc<parallelizer::GpuContext>,
        input_buf: Arc<Buffer>,
        input_dim: Dim3,
        output_buf: Arc<Buffer>,
        output_dim: Dim3,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let _ = pollster::block_on(self.run_with_buffers(gpu, input_buf, input_dim, output_buf, output_dim));
        })
    }

    pub fn last_input_len(&self) -> usize {
        self.last_input_len
    }

    pub fn last_output_len(&self) -> usize {
        self.last_output_len
    }

    pub fn last_state(&self) -> Option<ModelVisualState> {
        self.last_state
    }

    pub fn set_clear_color(&mut self, clear_color: Color) {
        self.clear_color = clear_color;
    }

    pub fn clear_color(&self) -> Color {
        self.clear_color
    }

    pub fn device(&self) -> &Device {
        self.device.as_ref()
    }

    pub fn queue(&self) -> &Queue {
        self.queue.as_ref()
    }
}
