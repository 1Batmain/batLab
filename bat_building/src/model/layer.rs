use crate::gpu_context::GpuContext;
use crate::model::error::ModelError;
use crate::model::layer_types::{
    BackwardBufferSource, BufferInit, ForwardBufferSource, LayerType, LayerTypes,
};
use crate::model::types::Dim3;
use std::collections::HashMap;
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

#[derive(Debug, Clone)]
pub(crate) struct MergePass {
    pub(crate) pipeline: ComputePipeline,
    pub(crate) bind_group: BindGroup,
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
    pub(crate) merge_pass: Option<MergePass>,
    pub(crate) saved_output_key: Option<String>,
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
            merge_pass: None,
            saved_output_key: None,
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
        self.merge_pass = None;
    }

    // -----------------------------------------------------------------------
    // Shader helpers
    // -----------------------------------------------------------------------

    fn create_shader(device: &Device, spec: &LayerTypes) -> ShaderModule {
        let shader = spec.get_forward_shader();
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader.label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader.source)),
        })
    }

    fn create_back_shader(device: &Device, spec: &LayerTypes) -> ShaderModule {
        let shader = spec
            .get_backward_shader()
            .expect("backward shader not supported for layer type");
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader.label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader.source)),
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
        saved_outputs: &HashMap<String, Arc<Buffer>>,
    ) -> Result<Arc<Buffer>, ModelError> {
        let bindings = self.ty.get_forward_buffer_bindings();
        for binding in bindings.iter() {
            match &binding.source {
                ForwardBufferSource::PreviousOutput => {
                    if let Some(ref prev) = last_output {
                        self.buffers.forward.push(Arc::clone(prev));
                        continue;
                    }
                }
                ForwardBufferSource::SavedOutput(key) => {
                    let saved = saved_outputs
                        .get(key)
                        .ok_or_else(|| ModelError::MissingSavedOutput { key: key.clone() })?;
                    self.buffers.forward.push(Arc::clone(saved));
                    continue;
                }
                ForwardBufferSource::Allocate => {}
            }
            let buf = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                label: Some(binding.name.as_str()),
                size: binding.spec.size as u64,
                usage: binding.spec.usage,
                mapped_at_creation: false,
            }));
            match binding.init {
                BufferInit::SpecsUniform => {
                    let bytes = self.ty.get_spec_uniform_bytes();
                    gpu.queue.write_buffer(&buf, 0, &bytes);
                }
                BufferInit::RandomWeights => {
                    let count = binding.spec.size as usize / 4;
                    let weights = Self::init_random_weights(count);
                    gpu.queue
                        .write_buffer(&buf, 0, bytemuck::cast_slice(&weights));
                }
                BufferInit::Ones => {
                    let count = binding.spec.size as usize / 4;
                    let ones = vec![1.0f32; count];
                    gpu.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&ones));
                }
                BufferInit::None => {}
            }
            self.buffers.forward.push(buf);
        }
        Ok(self.buffers.forward.last().unwrap().clone())
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
        self.bind_group.forward = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fwd_bg"),
                layout: &self
                    .pipeline
                    .forward
                    .as_ref()
                    .expect("set_pipeline must be called before set_bind_group")
                    .get_bind_group_layout(0),
                entries: &entries,
            }),
        );
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
        let bindings = self.ty.get_back_buffer_bindings();
        let incoming_grad = grad_output
            .as_ref()
            .expect("backward pass requires an incoming grad_output buffer");
        let buffers: Vec<_> = bindings
            .iter()
            .map(|binding| match binding.source {
                BackwardBufferSource::Forward(index) => Arc::clone(&self.buffers.forward[index]),
                BackwardBufferSource::IncomingGradient => Arc::clone(incoming_grad),
                BackwardBufferSource::Allocate => {
                    Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                        label: Some(binding.name.as_str()),
                        size: binding.spec.size as u64,
                        usage: binding.spec.usage,
                        mapped_at_creation: false,
                    }))
                }
            })
            .collect();

        let grad_input = Arc::clone(
            &buffers[self
                .ty
                .get_back_grad_input_index()
                .expect("backward pass must expose grad_input buffer index")],
        );
        self.buffers.backward = Some(buffers);
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
        self.bind_group.backward = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
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

    pub(crate) fn mark_saved_output(&mut self, key: impl Into<String>) {
        self.saved_output_key = Some(key.into());
    }

    pub(crate) fn saved_output_key(&self) -> Option<&str> {
        self.saved_output_key.as_deref()
    }

    pub(crate) fn saved_gradient_buffers(&self) -> Vec<(String, Arc<Buffer>)> {
        let Some(backward) = self.buffers.backward.as_ref() else {
            return vec![];
        };
        self.ty
            .get_saved_gradient_routes()
            .into_iter()
            .map(|route| (route.key, Arc::clone(&backward[route.buffer_index])))
            .collect()
    }

    // -----------------------------------------------------------------------
    // SGD optimiser pass (per-layer, trainable layers only)
    // -----------------------------------------------------------------------

    /// Build the SGD compute pass for this layer.
    pub(crate) fn create_opt_pass(&mut self, gpu: &GpuContext, lr: f32) {
        let Some(layout) = self.ty.get_optimizer_bindings() else {
            return;
        };
        let bwd = self
            .buffers
            .backward
            .as_ref()
            .expect("backward buffers must be built before create_opt_pass");
        let weights = Arc::clone(&self.buffers.forward[layout.weights_forward_index]);
        let bias = Arc::clone(&self.buffers.forward[layout.bias_forward_index]);
        let grad_weights = Arc::clone(&bwd[layout.grad_weights_backward_index]);
        let grad_bias = Arc::clone(&bwd[layout.grad_bias_backward_index]);

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

        let sgd_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
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

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let pipeline = gpu
            .device
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
            num_workgroups: layout.weight_count.div_ceil(64),
        });
    }

    pub(crate) fn encode_opt_pass(&self, encoder: &mut CommandEncoder) {
        let Some(opt) = &self.opt_pass else { return };
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&opt.pipeline);
        pass.set_bind_group(0, &opt.bind_group, &[]);
        pass.dispatch_workgroups(opt.num_workgroups, 1, 1);
    }

    pub(crate) fn set_opt_learning_rate(&self, gpu: &GpuContext, lr: f32) {
        let Some(opt) = &self.opt_pass else {
            return;
        };
        let Some(lr_buf) = opt.buffers.first() else {
            return;
        };
        let mut lr_bytes = [0u8; 16];
        lr_bytes[0..4].copy_from_slice(&lr.to_le_bytes());
        gpu.queue.write_buffer(lr_buf, 0, &lr_bytes);
    }

    pub(crate) fn encode_zero_opt_gradients(&self, encoder: &mut CommandEncoder) {
        let Some(layout) = self.ty.get_optimizer_bindings() else {
            return;
        };
        let Some(backward) = self.buffers.backward.as_ref() else {
            return;
        };
        encoder.clear_buffer(
            backward[layout.grad_weights_backward_index].as_ref(),
            0,
            None,
        );
        encoder.clear_buffer(backward[layout.grad_bias_backward_index].as_ref(), 0, None);
    }

    pub(crate) fn create_merge_pass(
        &mut self,
        gpu: &GpuContext,
        primary: Arc<Buffer>,
        secondary: Arc<Buffer>,
    ) -> Arc<Buffer> {
        let merged = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
            label: Some("grad_merge"),
            size: self.ty.get_dim_output().bytes_size() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("grad_merge"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "shader/sum.wgsl"
                ))),
            });
        let entries = [
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("grad_merge_bgl"),
                entries: &entries,
            });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("grad_merge_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("grad_merge_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("sum"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grad_merge_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: primary.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: secondary.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: merged.as_entire_binding(),
                },
            ],
        });
        self.merge_pass = Some(MergePass {
            pipeline,
            bind_group,
            num_workgroups: self.ty.get_dim_output().length().div_ceil(64),
        });
        merged
    }

    pub(crate) fn encode_merge_pass(&self, encoder: &mut CommandEncoder) {
        let Some(merge) = &self.merge_pass else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&merge.pipeline);
        pass.set_bind_group(0, &merge.bind_group, &[]);
        pass.dispatch_workgroups(merge.num_workgroups, 1, 1);
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
