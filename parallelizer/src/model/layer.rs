use crate::gpu_context::GpuContext;
use crate::model::error::ModelError;
use crate::model::layer_types::{LayerType, LayerTypes};
use crate::model::types::Dim3;
use std::sync::Arc;
use wgpu::{
    BindGroup, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, Device,
    ShaderModule,
};

// ---------------------------------------------------------------------------
// Buffer / pipeline / bind-group containers
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub(crate) struct Buffers {
    pub(crate) forward: Vec<Arc<Buffer>>,
    /// Populated by create_back_buffers; None until backward is built.
    pub(crate) backward: Option<Vec<Arc<Buffer>>>,
}

/// Forward: single pipeline.
/// Backward: one pipeline per sub-pass (e.g. Conv has 3: grad_input, grad_weights, grad_bias).
#[derive(Debug, Default, Clone)]
pub(crate) struct Pipelines {
    pub(crate) forward: Option<ComputePipeline>,
    pub(crate) backward: Vec<(ComputePipeline, u32)>, // (pipeline, num_workgroups)
}

#[derive(Debug, Default, Clone)]
pub(crate) struct BindGroups {
    pub(crate) forward: Option<BindGroup>,
    /// All backward sub-passes share a single bind group (same layout).
    pub(crate) backward: Option<BindGroup>,
}

#[derive(Debug, Clone)]
pub(crate) struct Shaders {
    pub(crate) forward: ShaderModule,
    pub(crate) backward: Option<ShaderModule>,
}

/// Per-layer SGD optimiser pass (only present on trainable layers).
#[derive(Debug, Clone)]
pub(crate) struct OptPass {
    pub(crate) pipeline: ComputePipeline,
    pub(crate) bind_group: BindGroup,
    /// Keeps the lr uniform buffer alive for the lifetime of this pass.
    #[allow(dead_code)]
    pub(crate) buffers: Vec<Arc<Buffer>>,
    pub(crate) num_workgroups: u32,
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct Layer {
    pub(crate) ty: LayerTypes,
    pub(crate) buffers: Buffers,
    pub(crate) shader: Shaders,
    pub(crate) pipeline: Pipelines,
    pub(crate) num_workgroups: u32,
    pub(crate) bind_group: BindGroups,
    /// Present after create_opt_pass is called on trainable layers.
    pub(crate) opt_pass: Option<OptPass>,
}

impl Layer {
    pub(crate) fn new(
        device: &Device,
        spec: LayerTypes,
        last_output: Option<Dim3>,
    ) -> Result<Self, ModelError> {
        let mut ty = spec;
        if let Some(input) = last_output {
            ty.set_dim_input(input);
        }
        ty.set_dim_output()?;
        let num_workgroups = ty.get_dim_output().length().div_ceil(64);
        let shader = Shaders {
            forward: Self::create_shader(device, &ty),
            backward: None,
        };
        Ok(Self {
            ty,
            shader,
            buffers: Buffers::default(),
            pipeline: Pipelines::default(),
            num_workgroups,
            bind_group: BindGroups::default(),
            opt_pass: None,
        })
    }

    pub(crate) fn clear(&mut self) {
        self.buffers.forward.clear();
        self.buffers.backward = None;
        self.pipeline.forward = None;
        self.pipeline.backward.clear();
        self.bind_group.forward = None;
        self.bind_group.backward = None;
        self.opt_pass = None;
    }

    // -----------------------------------------------------------------------
    // Shader helpers
    // -----------------------------------------------------------------------

    fn create_shader(device: &Device, spec: &LayerTypes) -> ShaderModule {
        let (code, name): (&str, &str) = match spec {
            LayerTypes::Convolution(_) => (include_str!("shader/convolution.wgsl"), "convolution"),
            LayerTypes::Activation(_) => (include_str!("shader/activation.wgsl"), "activation"),
            LayerTypes::Loss(_) => (include_str!("shader/loss.wgsl"), "loss"),
        };
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(code)),
        })
    }

    fn create_back_shader(device: &Device, spec: &LayerTypes) -> ShaderModule {
        let (code, name): (&str, &str) = match spec {
            LayerTypes::Convolution(_) => {
                (include_str!("shader/back_convolution.wgsl"), "back_convolution")
            }
            LayerTypes::Activation(_) => {
                (include_str!("shader/back_activation.wgsl"), "back_activation")
            }
            other => panic!(
                "backward shader not supported for layer type {:?}",
                other
            ),
        };
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(code)),
        })
    }

    /// Compile and store the backward shader module. Must be called before set_back_pipeline.
    pub(crate) fn init_back_shader(&mut self, device: &Device) {
        self.shader.backward = Some(Self::create_back_shader(device, &self.ty));
    }

    // -----------------------------------------------------------------------
    // Forward pass build
    // -----------------------------------------------------------------------

    pub(crate) fn create_buffers(
        &mut self,
        gpu: &GpuContext,
        last_output: Option<Arc<Buffer>>,
    ) -> Arc<Buffer> {
        let specs = self.ty.get_buffers_specs();
        for (i, (name, spec)) in specs.iter().enumerate() {
            // Binding 0 is always the "input" — share the previous layer's output buffer.
            if i == 0 {
                if let Some(ref prev) = last_output {
                    self.buffers.forward.push(Arc::clone(prev));
                    continue;
                }
            }
            let buf = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                label: Some(name.as_str()),
                size: spec.size as u64,
                usage: spec.usage,
                mapped_at_creation: false,
            }));
            match name.as_str() {
                "specs" => {
                    let bytes = self.ty.get_spec_uniform_bytes();
                    gpu.queue.write_buffer(&buf, 0, &bytes);
                }
                "weights" => {
                    let count = spec.size as usize / 4;
                    let weights = Self::init_random_weights(count);
                    gpu.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&weights));
                }
                // bias — zero-init is the wgpu default, nothing to do
                _ => {}
            }
            self.buffers.forward.push(buf);
        }
        self.buffers.forward.last().unwrap().clone()
    }

    pub(crate) fn set_pipeline(&mut self, device: &Device) {
        let specs = self.ty.get_buffers_specs();
        let entries: Vec<_> = specs
            .iter()
            .enumerate()
            .map(|(binding, (_, s))| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: s.visibility,
                ty: s.ty,
                count: None,
            })
            .collect();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fwd_bgl"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fwd_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        self.pipeline.forward = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("fwd_pipeline"),
                layout: Some(&pl),
                module: &self.shader.forward,
                entry_point: Some(self.ty.get_entrypoint()),
                compilation_options: Default::default(),
                cache: Default::default(),
            },
        ));
    }

    pub(crate) fn set_bind_group(&mut self, device: &Device) {
        let entries: Vec<_> = self
            .buffers
            .forward
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.bind_group.forward = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fwd_bg"),
            layout: &self
                .pipeline
                .forward
                .as_ref()
                .expect("set_pipeline must be called before set_bind_group")
                .get_bind_group_layout(0),
            entries: &entries,
        }));
    }

    pub(crate) fn encode_pass(&self, encoder: &mut CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(
            self.pipeline
                .forward
                .as_ref()
                .expect("forward pipeline not initialised"),
        );
        pass.set_bind_group(
            0,
            self.bind_group
                .forward
                .as_ref()
                .expect("forward bind group not initialised"),
            &[],
        );
        pass.dispatch_workgroups(self.num_workgroups, 1, 1);
    }

    // -----------------------------------------------------------------------
    // Backward pass build
    // -----------------------------------------------------------------------

    /// Build the backward buffer list.
    /// Returns the grad_input buffer so the caller can chain it as
    /// grad_output into the next (earlier) layer.
    pub(crate) fn create_back_buffers(
        &mut self,
        gpu: &GpuContext,
        grad_output: Option<Arc<Buffer>>,
    ) -> Arc<Buffer> {
        let (buffers_to_add, grad_input) = match &self.ty {
            LayerTypes::Convolution(conv) => {
                let gi_size = conv.dim_input.bytes_size().max(4) as u64;
                let gw_size = (conv.dim_kernel.bytes_size() * conv.nb_kernel).max(4) as u64;
                let gb_size = (conv.nb_kernel * 4).max(4) as u64;

                // Shared from forward:
                //   forward[0] = input (fwd_input)
                //   forward[1] = weights
                //   forward[3] = specs uniform
                let fwd_input = Arc::clone(&self.buffers.forward[0]);
                let fwd_weights = Arc::clone(&self.buffers.forward[1]);
                let fwd_specs = Arc::clone(&self.buffers.forward[3]);

                let go = grad_output
                    .expect("conv backward requires an incoming grad_output buffer");

                let make = |label, size| {
                    Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                        label: Some(label),
                        size,
                        usage: BufferUsages::COPY_SRC
                            | BufferUsages::COPY_DST
                            | BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }))
                };

                let gi = make("grad_input", gi_size);
                let gw = make("grad_weights", gw_size);
                let gb = make("grad_bias", gb_size);

                let bufs = vec![
                    fwd_input,
                    fwd_weights,
                    fwd_specs,
                    go,
                    Arc::clone(&gi),
                    gw,
                    gb,
                ];
                (bufs, gi)
            }

            LayerTypes::Activation(act) => {
                let gi_size = act.dim_input.bytes_size().max(4) as u64;

                // Shared from forward:
                //   forward[0] = input (fwd_input)
                //   forward[1] = specs uniform
                let fwd_input = Arc::clone(&self.buffers.forward[0]);
                let fwd_specs = Arc::clone(&self.buffers.forward[1]);

                let go = grad_output
                    .expect("activation backward requires an incoming grad_output buffer");

                let gi = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                    label: Some("grad_input"),
                    size: gi_size,
                    usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }));

                let bufs = vec![fwd_input, fwd_specs, go, Arc::clone(&gi)];
                (bufs, gi)
            }

            other => panic!(
                "create_back_buffers not supported for layer type {:?}",
                other
            ),
        };

        self.buffers.backward = Some(buffers_to_add);
        grad_input
    }

    pub(crate) fn set_back_pipeline(&mut self, device: &Device) {
        let specs = self.ty.get_back_buffers_specs();
        if specs.is_empty() {
            return;
        }

        let entries: Vec<_> = specs
            .iter()
            .enumerate()
            .map(|(binding, (_, s))| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: s.visibility,
                ty: s.ty,
                count: None,
            })
            .collect();

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("back_bgl"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("back_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let shader = self
            .shader
            .backward
            .as_ref()
            .expect("init_back_shader must be called before set_back_pipeline");

        let entry_points = self.ty.get_back_entrypoints();
        let workgroup_counts = self.ty.get_back_workgroup_counts();

        self.pipeline.backward = entry_points
            .iter()
            .zip(workgroup_counts)
            .map(|(ep, wg)| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("back_pipeline_{ep}")),
                    layout: Some(&pl),
                    module: shader,
                    entry_point: Some(ep),
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });
                (pipeline, wg)
            })
            .collect();
    }

    pub(crate) fn set_back_bind_group(&mut self, device: &Device) {
        let bwd = match self.buffers.backward.as_ref() {
            Some(b) if !b.is_empty() => b,
            _ => return,
        };
        if self.pipeline.backward.is_empty() {
            return;
        }
        let entries: Vec<_> = bwd
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.bind_group.backward =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("back_bg"),
                layout: &self.pipeline.backward[0].0.get_bind_group_layout(0),
                entries: &entries,
            }));
    }

    /// Encode all backward sub-passes (e.g. Conv encodes 3 sequential passes).
    /// wgpu inserts implicit pipeline barriers between compute passes in the same encoder.
    pub(crate) fn encode_back_pass(&self, encoder: &mut CommandEncoder) {
        let bg = match self.bind_group.backward.as_ref() {
            Some(bg) => bg,
            None => return,
        };
        for (pipeline, num_wg) in &self.pipeline.backward {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(*num_wg, 1, 1);
        }
    }

    // -----------------------------------------------------------------------
    // SGD optimiser pass (per-layer, trainable layers only)
    // -----------------------------------------------------------------------

    /// Build the SGD compute pass for this layer.
    /// Conv backward buffers must already exist before calling this.
    pub(crate) fn create_opt_pass(&mut self, gpu: &GpuContext, lr: f32) {
        let opt_data = match &self.ty {
            LayerTypes::Convolution(conv) => {
                // weight_count drives workgroup dispatch
                let weight_count = (conv.dim_kernel.bytes_size() * conv.nb_kernel) / 4;
                // Forward buffers: [0]=input, [1]=weights, [2]=bias, [3]=specs, [4]=output
                let weights = Arc::clone(&self.buffers.forward[1]);
                let bias = Arc::clone(&self.buffers.forward[2]);
                // Backward buffers: [0-3]=shared+grad_out, [4]=grad_input, [5]=grad_weights, [6]=grad_bias
                let bwd = self
                    .buffers
                    .backward
                    .as_ref()
                    .expect("backward buffers must be built before create_opt_pass");
                let grad_weights = Arc::clone(&bwd[5]);
                let grad_bias = Arc::clone(&bwd[6]);
                Some((weight_count, weights, bias, grad_weights, grad_bias))
            }
            _ => None,
        };

        let Some((weight_count, weights, bias, grad_weights, grad_bias)) = opt_data else {
            return;
        };

        // lr uniform: 4 bytes f32 padded to 16 bytes for alignment safety
        let lr_buf = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
            label: Some("lr_uniform"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let mut lr_bytes = [0u8; 16];
        lr_bytes[0..4].copy_from_slice(&lr.to_le_bytes());
        gpu.queue.write_buffer(&lr_buf, 0, &lr_bytes);

        let sgd_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sgd"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shader/sgd.wgsl"
            ))),
        });

        let layout_entries = [
            // [0] weights  read_write storage
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // [1] bias  read_write storage
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // [2] grad_weights  read storage
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // [3] grad_bias  read storage
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // [4] lr  uniform
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sgd_bgl"),
            entries: &layout_entries,
        });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sgd_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("sgd_pipeline"),
                    layout: Some(&pl),
                    module: &sgd_shader,
                    entry_point: Some("sgd"),
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sgd_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lr_buf.as_entire_binding(),
                },
            ],
        });

        self.opt_pass = Some(OptPass {
            pipeline,
            bind_group,
            buffers: vec![lr_buf],
            num_workgroups: weight_count.div_ceil(64),
        });
    }

    pub(crate) fn encode_opt_pass(&self, encoder: &mut CommandEncoder) {
        let Some(opt) = &self.opt_pass else { return };
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&opt.pipeline);
        pass.set_bind_group(0, &opt.bind_group, &[]);
        pass.dispatch_workgroups(opt.num_workgroups, 1, 1);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// XorShift32 PRNG for deterministic weight initialisation (±0.1 range).
    fn init_random_weights(count: usize) -> Vec<f32> {
        let mut state: u32 = 2463534242;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 0.2 - 0.1
            })
            .collect()
    }
}
