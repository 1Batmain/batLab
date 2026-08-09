//! File purpose: Implements layer functionality for model execution, state, or diagnostics.

use crate::gpu_context::GpuContext;
use crate::model::error::ModelError;
use crate::model::layer_types::{
    Batch, BackwardBufferSource, BufferInit, ForwardBufferSource, LayerType, LayerTypes,
};
use crate::model::ema::{EmaConfig, EmaSpecs};
use crate::model::optimizer::{AdamHyperparameters, AdamSpecs, OptimizerKind};
use crate::model::types::Dim3;
use crate::model::weight_init::{self, WeightInit};
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

/// One pipeline per sub-pass, forward and backward alike (e.g. Conv's backward
/// has 3: grad_input, grad_weights, grad_bias; attention's forward has 4).
///
/// The forward was a single `Option<ComputePipeline>` until attention needed a
/// global barrier mid-forward. Every other layer still declares one entry point
/// and lands in a one-element list, so the shape of the change is: the common
/// case is a list of length 1, not a special case.
#[derive(Debug, Default, Clone)]
pub(crate) struct Pipelines {
    pub(crate) forward: Vec<(ComputePipeline, u32)>, // (pipeline, num_workgroups)
    pub(crate) backward: Vec<(ComputePipeline, u32)>,
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

/// Per-layer optimiser pass (only present on trainable layers).
#[derive(Debug, Clone)]
pub(crate) struct OptPass {
    pub(crate) kind: OptimizerKind,
    pub(crate) pipeline: ComputePipeline,
    pub(crate) bind_group: BindGroup,
    /// Hyperparameter uniform, rewritten before every optimiser dispatch.
    pub(crate) specs: Arc<Buffer>,
    /// Adam only: `[m_weights, v_weights, m_bias, v_bias]`. Empty for SGD.
    pub(crate) state: Vec<Arc<Buffer>>,
    pub(crate) num_workgroups: u32,
}

impl OptPass {
    /// Every GPU buffer owned by the pass (uniform + optimiser state), for
    /// memory accounting.
    pub(crate) fn owned_buffers(&self) -> impl Iterator<Item = &Arc<Buffer>> {
        std::iter::once(&self.specs).chain(self.state.iter())
    }
}

/// Per-layer exponential-moving-average pass (only on trainable layers, and
/// only when the run asked for an EMA).
///
/// Separate from [`OptPass`] rather than folded into the update shaders: the
/// recurrence is the same whichever optimiser produced the weights, and a pass
/// of its own is one that can be read, tested and switched off on its own.
#[derive(Debug, Clone)]
pub(crate) struct EmaPass {
    pub(crate) pipeline: ComputePipeline,
    pub(crate) bind_group: BindGroup,
    /// The effective decay of the step about to be dispatched.
    pub(crate) specs: Arc<Buffer>,
    /// `[ema_weights, ema_bias]`, in checkpoint order.
    pub(crate) state: Vec<Arc<Buffer>>,
    pub(crate) num_workgroups: u32,
}

impl EmaPass {
    pub(crate) fn owned_buffers(&self) -> impl Iterator<Item = &Arc<Buffer>> {
        std::iter::once(&self.specs).chain(self.state.iter())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MergePass {
    pub(crate) pipeline: ComputePipeline,
    pub(crate) bind_group: BindGroup,
    pub(crate) num_workgroups: u32,
}

/// Split a 1-D workgroup count into a 2-D grid that respects
/// `max_compute_workgroups_per_dimension` (65 535 in the WebGPU spec, and on
/// this adapter).
///
/// Nothing in the graph needed this before the batch axis: the largest dispatch
/// was one workgroup per 64 elements of a single tensor, which for the biggest
/// layer of `Greyscale_Diffusion_L` is 1024. Multiply that by a batch of 64 and
/// it is 65 536 — one over the limit.
///
/// The failure mode is what makes this worth a named function. wgpu reports the
/// violation on the queue, the run does NOT stop, and the step simply computes
/// nothing: the observed symptom was `loss 0.000000` scrolling by. A silent
/// wrong answer at exactly the batch sizes the batching exists to enable.
///
/// The kernels recover their linear index as `gid.y * nwg.x * 64 + gid.x` (or
/// `wid.y * nwg.x + wid.x` for the workgroup-per-unit passes), reading `nwg`
/// from `@builtin(num_workgroups)` — so, like the batch itself, the split needs
/// no uniform and cannot drift from what was dispatched. Trailing workgroups
/// past the real count are absorbed by the bounds check every kernel already
/// has.
pub(crate) fn dispatch_grid(workgroups: u32) -> (u32, u32) {
    const MAX_PER_DIMENSION: u32 = 65_535;
    if workgroups <= MAX_PER_DIMENSION {
        return (workgroups, 1);
    }
    (MAX_PER_DIMENSION, workgroups.div_ceil(MAX_PER_DIMENSION))
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
    /// Present after create_ema_pass, i.e. on trainable layers of a run that
    /// asked for a weight average.
    pub(crate) ema_pass: Option<EmaPass>,
    pub(crate) merge_pass: Option<MergePass>,
    pub(crate) saved_output_key: Option<String>,
    /// How many samples this layer's activation buffers hold.
    ///
    /// Set by the model before `create_buffers`, and from then on the single
    /// authority on every size and dispatch count of this layer. A layer built
    /// for `batch` and encoded for a different one is not an error — see
    /// `encode_pass_with_batch` — but a layer *allocated* for the wrong one is.
    pub(crate) batch: Batch,
    /// Position in the model's layer list. Used for one thing only: naming this
    /// layer's passes in a GPU profile, where `L13 Convolution` has to be
    /// distinguishable from the seven other convolutions of the same stack.
    pub(crate) index: usize,
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
        // Batch 1 until the model says otherwise: `add_layer` runs long before
        // `build()`, which is where the batch is known.
        let num_workgroups = ty.get_forward_workgroup_count(1);
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
            ema_pass: None,
            merge_pass: None,
            saved_output_key: None,
            batch: 1,
            index: 0,
        })
    }

    /// How this layer's passes are named in a profile: `L03 Convolution`.
    ///
    /// Only ever called from behind a profiling check — it allocates, and a
    /// training step encodes 151 passes.
    fn profile_prefix(&self) -> String {
        format!("L{:02} {}", self.index, self.ty.variant_name())
    }

    pub(crate) fn clear(&mut self) {
        self.buffers.forward.clear();
        self.buffers.backward = None;
        self.pipeline.forward.clear();
        self.pipeline.backward.clear();
        self.bind_group.forward = None;
        self.bind_group.backward = None;
        self.opt_pass = None;
        self.ema_pass = None;
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

    /// `layer_index` only seeds the weight PRNG: He draws a distinct stream per
    /// layer, so two same-shaped layers do not start life identical.
    pub(crate) fn create_buffers(
        &mut self,
        gpu: &GpuContext,
        last_output: Option<Arc<Buffer>>,
        saved_outputs: &HashMap<String, Arc<Buffer>>,
        init: WeightInit,
        layer_index: usize,
    ) -> Result<Arc<Buffer>, ModelError> {
        self.num_workgroups = self.ty.get_forward_workgroup_count(self.batch);
        let bindings = self.ty.get_forward_buffer_bindings(self.batch);
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
                    gpu.write_buffer(&buf, 0, &bytes);
                }
                BufferInit::RandomWeights | BufferInit::RandomWeightsZeroTail(_) => {
                    let count = binding.spec.size as usize / 4;
                    let mut weights = match (init, self.ty.get_weight_fan_in()) {
                        (WeightInit::He, Some(fan_in)) => weight_init::he_weights(
                            count,
                            fan_in,
                            // Odd multiplier: distinct, well-spread seeds that
                            // never land on xorshift's absorbing 0.
                            0x9E37_79B9u32.wrapping_mul(layer_index as u32 + 1) | 1,
                        ),
                        // No fan-in to speak of (e.g. GroupNorm's scale
                        // parameters) — the historical draw stands.
                        _ => weight_init::uniform_weights(count),
                    };
                    // A residual layer whose output projection starts at zero
                    // starts as the identity: the tail is zeroed AFTER the draw
                    // so the leading projections keep the exact stream they
                    // would have had on their own.
                    if let BufferInit::RandomWeightsZeroTail(tail) = binding.init {
                        let from = weights.len().saturating_sub(tail as usize);
                        weights[from..].fill(0.0);
                    }
                    gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&weights));
                }
                BufferInit::Ones => {
                    let count = binding.spec.size as usize / 4;
                    let ones = vec![1.0f32; count];
                    gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&ones));
                }
                BufferInit::None => {}
            }
            self.buffers.forward.push(buf);
        }
        Ok(self.buffers.forward.last().unwrap().clone())
    }

    pub(crate) fn set_pipeline(&mut self, device: &Device) {
        let specs = self.ty.get_buffers_specs(self.batch);
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
        // A layer that declares no forward entry points is the ordinary
        // single-kernel case: one pass at its own workgroup count.
        let declared = self.ty.get_forward_entrypoints();
        let (entry_points, counts): (Vec<&str>, Vec<u32>) = if declared.is_empty() {
            (
                vec![self.ty.get_entrypoint()],
                vec![self.ty.get_forward_workgroup_count(self.batch)],
            )
        } else {
            (declared, self.ty.get_forward_workgroup_counts(self.batch))
        };

        self.pipeline.forward = entry_points
            .iter()
            .zip(counts)
            .map(|(ep, wg)| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("fwd_pipeline_{ep}")),
                    layout: Some(&pl),
                    module: &self.shader.forward,
                    entry_point: Some(ep),
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });
                (pipeline, wg)
            })
            .collect();
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
                    .first()
                    .expect("set_pipeline must be called before set_bind_group")
                    .0
                    .get_bind_group_layout(0),
                entries: &entries,
            }),
        );
    }

    pub(crate) fn encode_pass(&self, gpu: &GpuContext, encoder: &mut CommandEncoder) {
        self.encode_pass_with_batch(gpu, encoder, self.batch);
    }

    /// Encode the forward pass for the first `batch` samples only.
    ///
    /// Truncating the dispatch is sound because every batched kernel recovers
    /// its sample as `global_index / per_sample_length`: dispatching a prefix
    /// of the grid computes a prefix of the samples and touches nothing else.
    /// `predict()` uses it to run a single image through a graph whose buffers
    /// are sized for a full training batch — the probe would otherwise pay 16x
    /// the work for one result.
    pub(crate) fn encode_pass_with_batch(
        &self,
        gpu: &GpuContext,
        encoder: &mut CommandEncoder,
        batch: Batch,
    ) {
        assert!(
            !self.pipeline.forward.is_empty(),
            "forward pipeline not initialised"
        );
        let bg = self
            .bind_group
            .forward
            .as_ref()
            .expect("forward bind group not initialised");

        // A truncated batch re-derives every sub-pass's count, not just the
        // first: the sub-passes of one layer do not share a workgroup count
        // (attention's softmax runs one workgroup per row, its projections one
        // thread per scalar).
        let truncated = if batch == self.batch {
            None
        } else {
            let smaller = batch.min(self.batch);
            let declared = self.ty.get_forward_entrypoints();
            Some(if declared.is_empty() {
                vec![self.ty.get_forward_workgroup_count(smaller)]
            } else {
                self.ty.get_forward_workgroup_counts(smaller)
            })
        };

        let entrypoints = self.ty.get_forward_entrypoints();
        for (index, (pipeline, built_wg)) in self.pipeline.forward.iter().enumerate() {
            let workgroups = match &truncated {
                Some(counts) => counts[index],
                None => *built_wg,
            };
            // wgpu inserts a pipeline barrier between compute passes of one
            // encoder — that barrier is what makes a multi-pass forward (q/k/v,
            // then scores, then context) see the previous pass's writes.
            gpu.compute_pass(
                encoder,
                workgroups,
                || {
                    let ep = entrypoints.get(index).copied().unwrap_or("forward");
                    format!("{} · {ep}", self.profile_prefix())
                },
                |pass| {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    let (x, y) = dispatch_grid(workgroups);
                    pass.dispatch_workgroups(x, y, 1);
                },
            );
        }
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
        let bindings = self.ty.get_back_buffer_bindings(self.batch);
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
        let specs = self.ty.get_back_buffers_specs(self.batch);
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
        let workgroup_counts = self.ty.get_back_workgroup_counts(self.batch);

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
    pub(crate) fn encode_back_pass(&self, gpu: &GpuContext, encoder: &mut CommandEncoder) {
        let bg = match self.bind_group.backward.as_ref() {
            Some(bg) => bg,
            None => return,
        };
        let entrypoints = self.ty.get_back_entrypoints();
        for (index, (pipeline, num_wg)) in self.pipeline.backward.iter().enumerate() {
            gpu.compute_pass(
                encoder,
                *num_wg,
                || {
                    let ep = entrypoints.get(index).copied().unwrap_or("backward");
                    format!("{} · {ep}", self.profile_prefix())
                },
                |pass| {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    let (x, y) = dispatch_grid(*num_wg);
                    pass.dispatch_workgroups(x, y, 1);
                },
            );
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
    // Optimiser pass (per-layer, trainable layers only)
    // -----------------------------------------------------------------------

    /// Build the optimiser compute pass for this layer.
    ///
    /// SGD binds 5 buffers; Adam binds 4 more (the first and second moment for
    /// weights and bias), which is why the pass is built per optimiser kind
    /// rather than shared.
    pub(crate) fn create_opt_pass(&mut self, gpu: &GpuContext, lr: f32, kind: OptimizerKind) {
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

        // Storage bindings shared by both kinds: weights, bias (rw) then the
        // two gradient buffers (ro), then the hyperparameter uniform.
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut layout_entries = vec![
            storage_entry(0, false), // weights
            storage_entry(1, false), // bias
            storage_entry(2, true),  // grad_weights
            storage_entry(3, true),  // grad_bias
            // [4] specs uniform
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

        // The uniform is 16 bytes for SGD (a padded lr) and 32 for Adam; the
        // larger allocation is harmless and keeps one code path.
        let specs = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
            label: Some("opt_specs_uniform"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Adam moment buffers, zero-initialised. wgpu zeroes new buffers, and
        // m₀ = v₀ = 0 is exactly the algorithm's initial state.
        let state: Vec<Arc<Buffer>> = match kind {
            OptimizerKind::Sgd => Vec::new(),
            OptimizerKind::Adam => {
                let make = |label: &str, size: u64| {
                    Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                        label: Some(label),
                        size,
                        usage: BufferUsages::STORAGE
                            | BufferUsages::COPY_SRC
                            | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }))
                };
                let w_size = weights.size();
                let b_size = bias.size();
                vec![
                    make("adam_m_weights", w_size),
                    make("adam_v_weights", w_size),
                    make("adam_m_bias", b_size),
                    make("adam_v_bias", b_size),
                ]
            }
        };
        for binding in 5..(5 + state.len() as u32) {
            layout_entries.push(storage_entry(binding, false));
        }

        let (label, source, entry_point) = match kind {
            OptimizerKind::Sgd => ("sgd", include_str!("shader/sgd.wgsl"), "sgd"),
            OptimizerKind::Adam => ("adam", include_str!("shader/adam.wgsl"), "adam"),
        };
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
            });

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("opt_bgl"),
                entries: &layout_entries,
            });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("opt_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("opt_pipeline"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let mut entries = vec![
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
                resource: specs.as_entire_binding(),
            },
        ];
        for (i, buffer) in state.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 5 + i as u32,
                resource: buffer.as_entire_binding(),
            });
        }
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("opt_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        self.opt_pass = Some(OptPass {
            kind,
            pipeline,
            bind_group,
            specs,
            state,
            num_workgroups: layout.weight_count.div_ceil(64),
        });

        // Sensible contents before the first dispatch: a step of t=1 with no
        // batch averaging, matching what the SGD pass used to be built with.
        self.write_opt_specs(gpu, lr, 1.0, 1);
    }

    pub(crate) fn encode_opt_pass(&self, gpu: &GpuContext, encoder: &mut CommandEncoder) {
        let Some(opt) = &self.opt_pass else { return };
        gpu.compute_pass(
            encoder,
            opt.num_workgroups,
            || format!("{} · {}", self.profile_prefix(), opt.kind.label()),
            |pass| {
                pass.set_pipeline(&opt.pipeline);
                pass.set_bind_group(0, &opt.bind_group, &[]);
                let (x, y) = dispatch_grid(opt.num_workgroups);
                pass.dispatch_workgroups(x, y, 1);
            },
        );
    }

    /// Refresh the optimiser uniform for the step about to be dispatched.
    ///
    /// `grad_scale` is `1 / batch_size` — the batch mean. SGD folds it into the
    /// learning rate (its update is linear in the gradient), Adam applies it to
    /// the gradient itself (its update is not).
    pub(crate) fn write_opt_specs(
        &self,
        gpu: &GpuContext,
        lr: f32,
        grad_scale: f32,
        step: u64,
    ) -> bool {
        let Some(opt) = &self.opt_pass else {
            return false;
        };
        match opt.kind {
            OptimizerKind::Sgd => {
                let mut bytes = [0u8; 16];
                bytes[0..4].copy_from_slice(&(lr * grad_scale).to_le_bytes());
                gpu.write_buffer(&opt.specs, 0, &bytes);
            }
            OptimizerKind::Adam => {
                let hp = AdamHyperparameters::default();
                let (bias_correction1, bias_correction2) = hp.bias_corrections(step);
                let bytes = AdamSpecs {
                    lr,
                    beta1: hp.beta1,
                    beta2: hp.beta2,
                    eps: hp.eps,
                    grad_scale,
                    bias_correction1,
                    bias_correction2,
                }
                .to_bytes();
                gpu.write_buffer(&opt.specs, 0, &bytes);
            }
        }
        true
    }

    /// The persistent optimiser state buffers, in checkpoint order
    /// (`[m_weights, v_weights, m_bias, v_bias]` for Adam, empty for SGD).
    pub(crate) fn opt_state_buffers(&self) -> &[Arc<Buffer>] {
        self.opt_pass
            .as_ref()
            .map(|o| o.state.as_slice())
            .unwrap_or(&[])
    }

    /// Build the EMA pass for this layer: two shadow buffers and the kernel that
    /// pulls them towards the weights after every update.
    ///
    /// The shadow is **seeded with the weights as they stand now**, not with
    /// zeros. `create_opt_pass` runs during `build()`, so "now" is either the
    /// fresh initialisation or — on a rebuild — the weights that were just
    /// restored. A zero-seeded shadow would need thousands of steps to shed the
    /// zeros, and every checkpoint written before then would carry an EMA set
    /// that generates noise. See `ema.rs` for the pairing with the warmup ramp.
    pub(crate) fn create_ema_pass(&mut self, gpu: &GpuContext, ema: EmaConfig) {
        let Some(layout) = self.ty.get_optimizer_bindings() else {
            return;
        };
        let weights = Arc::clone(&self.buffers.forward[layout.weights_forward_index]);
        let bias = Arc::clone(&self.buffers.forward[layout.bias_forward_index]);

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout_entries = vec![
            storage_entry(0, true),  // weights
            storage_entry(1, true),  // bias
            storage_entry(2, false), // ema_weights
            storage_entry(3, false), // ema_bias
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

        let specs = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
            label: Some("ema_specs_uniform"),
            size: EmaSpecs::BYTES as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let make = |label: &str, size: u64| {
            Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        };
        let state = vec![
            make("ema_weights", weights.size()),
            make("ema_bias", bias.size()),
        ];

        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ema"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "shader/ema.wgsl"
                ))),
            });
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ema_bgl"),
                entries: &layout_entries,
            });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ema_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ema_pipeline"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("ema"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ema_bg"),
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
                    resource: state[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: specs.as_entire_binding(),
                },
            ],
        });

        self.ema_pass = Some(EmaPass {
            pipeline,
            bind_group,
            specs,
            state,
            num_workgroups: layout.weight_count.div_ceil(64),
        });
        self.write_ema_specs(gpu, ema.effective_decay(1));
        self.seed_ema_from_weights(gpu);
    }

    /// Copy the live weights over the shadow, on the GPU.
    ///
    /// Called when the pass is built, and again by the model after a checkpoint
    /// that carried no EMA set was loaded: in both cases the shadow would
    /// otherwise describe weights the model no longer has.
    pub(crate) fn seed_ema_from_weights(&self, gpu: &GpuContext) {
        let Some(ema) = &self.ema_pass else { return };
        let Some(layout) = self.ty.get_optimizer_bindings() else {
            return;
        };
        let weights = &self.buffers.forward[layout.weights_forward_index];
        let bias = &self.buffers.forward[layout.bias_forward_index];
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(weights, 0, &ema.state[0], 0, weights.size());
        encoder.copy_buffer_to_buffer(bias, 0, &ema.state[1], 0, bias.size());
        gpu.submit([encoder.finish()]);
    }

    pub(crate) fn encode_ema_pass(&self, gpu: &GpuContext, encoder: &mut CommandEncoder) {
        let Some(ema) = &self.ema_pass else { return };
        gpu.compute_pass(
            encoder,
            ema.num_workgroups,
            || format!("{} · ema", self.profile_prefix()),
            |pass| {
                pass.set_pipeline(&ema.pipeline);
                pass.set_bind_group(0, &ema.bind_group, &[]);
                let (x, y) = dispatch_grid(ema.num_workgroups);
                pass.dispatch_workgroups(x, y, 1);
            },
        );
    }

    /// Refresh the EMA uniform for the step about to be dispatched. `decay` is
    /// the EFFECTIVE decay — the warmup ramp is resolved by the caller, which
    /// is the only place the step counter lives.
    pub(crate) fn write_ema_specs(&self, gpu: &GpuContext, decay: f32) -> bool {
        let Some(ema) = &self.ema_pass else {
            return false;
        };
        gpu.write_buffer(&ema.specs, 0, &EmaSpecs { decay }.to_bytes());
        true
    }

    /// The shadow buffers, in checkpoint order (`[ema_weights, ema_bias]`).
    pub(crate) fn ema_state_buffers(&self) -> &[Arc<Buffer>] {
        self.ema_pass
            .as_ref()
            .map(|e| e.state.as_slice())
            .unwrap_or(&[])
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
            // An activation-shaped buffer: one slot per sample.
            size: (self.ty.get_dim_output().bytes_size() * self.batch) as u64,
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
            num_workgroups: (self.ty.get_dim_output().length() * self.batch).div_ceil(64),
        });
        merged
    }

    pub(crate) fn encode_merge_pass(&self, gpu: &GpuContext, encoder: &mut CommandEncoder) {
        let Some(merge) = &self.merge_pass else {
            return;
        };
        gpu.compute_pass(
            encoder,
            merge.num_workgroups,
            || format!("{} · grad_merge", self.profile_prefix()),
            |pass| {
                pass.set_pipeline(&merge.pipeline);
                pass.set_bind_group(0, &merge.bind_group, &[]);
                let (x, y) = dispatch_grid(merge.num_workgroups);
                pass.dispatch_workgroups(x, y, 1);
            },
        );
    }
}
