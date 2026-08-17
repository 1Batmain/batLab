//! What a model costs on the GPU: an itemised inventory of every buffer the graph
//! allocates, judged against a real or hypothetical device's limits.
//!
//! In the engine, not the UI, because "does my model fit on that machine?" is
//! asked about the engine — and it is the question a browser must answer before
//! it starts (WebGPU refuses an allocation past its limits). So the inventory is
//! computable WITHOUT a GPU, from a [`ModelConfig`], against GIVEN limits.
//!
//! Not a formula: an inventory that lies is worse than none. It walks the graph
//! exactly as `Model::build` does — the same buffer bindings, the same sharing
//! rules (a layer's input IS the previous output; backward re-binds forward
//! buffers) — so it counts what is actually allocated, held to that by
//! `the_predicted_inventory_matches_a_real_model`.

use std::collections::HashMap;

use crate::config::{LayerDraft, ModelConfig};
use crate::gpu_context::{GpuContext, GpuLimitsProfile};
use crate::model::ema::EmaSpecs;
use crate::model::error::ModelError;
use crate::model::layer_types::{
    ActivationType, AddType, AttentionType, BackwardBufferSource, ConcatType, ConvolutionType,
    ForwardBufferSource, FullyConnectedType, GroupNormType, LayerType, LayerTypes, LossMethod,
    LossType, TimeBiasType, UpsampleConvType,
};
use crate::model::optimizer::OptimizerKind;
use crate::model::types::Dim3;

/// Bytes of the uniform `create_opt_pass` allocates per trainable layer. Not
/// `size_of::<AdamSpecs>()`: it is allocated at the larger of the two optimiser
/// shapes so one code path serves both (`layer.rs`), and this module mirrors that.
const OPT_SPECS_BYTES: u64 = 32;

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

/// Where a [`DeviceProfile`]'s numbers come from — "your GPU refuses this" and "a
/// card like this would refuse this" are different claims the page must not blur.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileSource {
    /// Read off an adapter this process opened.
    Measured,
    /// Supplied by the caller — a "what if" about another machine.
    Hypothetical,
}

/// The limits an inventory is judged against.
///
/// Everything here is a *hard* limit of the device or of the WebGPU contract,
/// except `memory_budget`, which no portable API exposes (see its comment).
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub name: String,
    pub backend: String,
    /// `IntegratedGpu` / `DiscreteGpu` / … as reported by the adapter.
    pub device_type: String,
    /// Integrated adapters share system RAM — no separate VRAM to exhaust — which
    /// is why the numbers here flatter this Mac relative to a discrete card.
    pub unified_memory: bool,
    /// What the DEVICE will allocate — the limits `request_device` asked for (which
    /// profile: [`DeviceProfile::limits_profile`]). Under `web` these are wgpu's
    /// defaults (256 MiB/buffer even on an adapter that would allow 28 GiB, the
    /// contract a browser hands the engine); under `native`, the adapter's own.
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    /// What the ADAPTER could allocate in one buffer — kept distinct from the
    /// device limit above (they differ by four orders of magnitude). This is what
    /// [`crate::training::dataset::select_chunk_bytes`] sizes chunks from; using the
    /// device figure predicted the wrong chunk size until
    /// `the_predicted_chunk_plan_is_the_one_the_dataset_allocates`. On Metal it is
    /// ~the whole unified memory, but it is a max allocation size, not a budget.
    pub adapter_max_buffer_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    /// Total memory the inventory may fill. `None` on every backend here (no
    /// portable API exposes a VRAM budget); `Some` only when the caller states one
    /// ("would this fit on an 8 GB card?").
    pub memory_budget: Option<u64>,
    pub source: ProfileSource,
    /// Which profile the device was opened under. `None` for a hypothetical
    /// machine — no device was requested, so nothing was granted.
    pub limits_profile: Option<GpuLimitsProfile>,
}

impl DeviceProfile {
    /// The adapter this process actually opened.
    pub fn from_gpu(gpu: &GpuContext) -> Self {
        let info = gpu.adapter().get_info();
        let limits = gpu.device().limits();
        Self {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            unified_memory: matches!(
                info.device_type,
                wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
            ),
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
            adapter_max_buffer_size: gpu.specs().memory_size(),
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            memory_budget: None,
            source: ProfileSource::Measured,
            limits_profile: Some(gpu.limits_profile()),
        }
    }

    /// A card that is not here: the WebGPU DEFAULT limits (128 MiB/storage binding,
    /// 256 MiB/buffer) plus a memory budget — the conservative floor for "would
    /// this run in a visitor's browser".
    pub fn hypothetical(name: impl Into<String>, memory_budget: Option<u64>) -> Self {
        let limits = wgpu::Limits::default();
        Self {
            name: name.into(),
            backend: "—".to_string(),
            device_type: "—".to_string(),
            unified_memory: false,
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
            // Nothing better to say about a machine that is not here: a device
            // that would allocate no more than its own limit.
            adapter_max_buffer_size: limits.max_buffer_size,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            memory_budget,
            source: ProfileSource::Hypothetical,
            limits_profile: None,
        }
    }

    /// The same profile with a memory budget stated by the caller.
    pub fn with_budget(mut self, bytes: u64) -> Self {
        self.memory_budget = Some(bytes);
        self
    }

    /// What the driver's own allocator reports, when the backend has one.
    ///
    /// `None` on Metal (and on any backend wgpu does not sub-allocate for), so
    /// callers must treat it as a bonus, never as the capacity figure.
    pub fn allocator_report(gpu: &GpuContext) -> Option<(u64, u64)> {
        gpu.device()
            .generate_allocator_report()
            .map(|report| (report.total_allocated_bytes, report.total_reserved_bytes))
    }
}

// ---------------------------------------------------------------------------
// Inventory vocabulary
// ---------------------------------------------------------------------------

/// The posts an allocation is charged to.
///
/// The split that matters is the first one: [`Weights`](Kind::Weights),
/// [`ParameterGradients`](Kind::ParameterGradients),
/// [`OptimizerState`](Kind::OptimizerState) and [`Ema`](Kind::Ema) do **not**
/// carry a batch axis, [`Activations`](Kind::Activations) and everything after
/// it do. That is the whole shape of the memory curve: raising the batch moves
/// one half of this table and leaves the other exactly where it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// Trainable scalars: weights, biases, γ/β.
    Weights,
    /// Their gradients — a reduction over the batch, so batch-independent.
    ParameterGradients,
    /// Adam's `m` and `v` (nothing at all for SGD).
    OptimizerState,
    /// The weight average, when the run keeps one.
    Ema,
    /// Every intermediate tensor of the forward pass: one slot per sample.
    Activations,
    /// `grad_input` / `grad_output` / the skip merge: activations too.
    ActivationGradients,
    /// Attention's `q|k|v|ctx|probs` block — the `N²` term.
    AttentionScratch,
    /// The loss layer's own tensors (target, per-element terms).
    LossScratch,
    /// The diffusion prepass: the clean image batch and its per-sample specs.
    DiffusionScratch,
    /// Every `specs` uniform. Tiny, listed because "tiny" is a measurement.
    Uniforms,
    /// The resident dataset chunk — the one post that is *streamed*.
    DatasetChunk,
    /// The frame the visualiser reads.
    LiveFrame,
}

impl Kind {
    pub const ALL: [Kind; 12] = [
        Kind::Weights,
        Kind::ParameterGradients,
        Kind::OptimizerState,
        Kind::Ema,
        Kind::Activations,
        Kind::ActivationGradients,
        Kind::AttentionScratch,
        Kind::LossScratch,
        Kind::DiffusionScratch,
        Kind::Uniforms,
        Kind::DatasetChunk,
        Kind::LiveFrame,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Weights => "weights",
            Kind::ParameterGradients => "param gradients",
            Kind::OptimizerState => "optimizer state",
            Kind::Ema => "EMA weights",
            Kind::Activations => "activations",
            Kind::ActivationGradients => "activation gradients",
            Kind::AttentionScratch => "attention scratch",
            Kind::LossScratch => "loss scratch",
            Kind::DiffusionScratch => "diffusion prepass",
            Kind::Uniforms => "uniforms",
            Kind::DatasetChunk => "dataset chunk",
            Kind::LiveFrame => "live frame",
        }
    }

    /// Whether this post is sized by the batch. See the type's own comment:
    /// this predicate *is* the memory curve.
    pub fn scales_with_batch(self) -> bool {
        matches!(
            self,
            Kind::Activations
                | Kind::ActivationGradients
                | Kind::AttentionScratch
                | Kind::LossScratch
                | Kind::DiffusionScratch
        )
    }
}

/// One `wgpu::Buffer` the graph allocates.
#[derive(Debug, Clone)]
pub struct Allocation {
    /// The label the real allocation carries, so a line here can be found in a
    /// capture.
    pub label: String,
    pub kind: Kind,
    pub bytes: u64,
    /// Index in the layer stack, or `None` for the loss layer, the prepass and
    /// the dataset.
    pub layer: Option<usize>,
}

/// What one layer of the stack costs, all posts together.
#[derive(Debug, Clone)]
pub struct LayerFootprint {
    pub index: usize,
    /// The one-line description the layer builder already shows.
    pub display: String,
    /// Trainable scalars, derived from the weight buffers themselves.
    pub parameters: u64,
    pub bytes: u64,
    /// Of which carries a batch axis.
    pub batched_bytes: u64,
    /// Largest single binding of this layer — the one that meets
    /// `max_storage_buffer_binding_size` first.
    pub largest_binding: u64,
}

// ---------------------------------------------------------------------------
// Workload
// ---------------------------------------------------------------------------

/// Which graph is built, which is what decides more than half the total.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Workload {
    /// Forward buffers only, one sample, no gradient and no optimiser.
    Inference,
    Training {
        optimizer: OptimizerKind,
        /// Whether the run keeps a weight average (`--ema`).
        ema: bool,
        loss: LossMethod,
        /// Diffusion runs allocate a prepass on top of the model.
        diffusion: bool,
    },
}

impl Workload {
    /// The training a `config_file` describes.
    pub fn training(optimizer: OptimizerKind, ema: bool) -> Self {
        Workload::Training {
            optimizer,
            ema,
            loss: LossMethod::MeanSquared,
            diffusion: true,
        }
    }

    pub fn is_training(self) -> bool {
        matches!(self, Workload::Training { .. })
    }

    pub fn label(self) -> &'static str {
        match self {
            Workload::Inference => "inference",
            Workload::Training { .. } => "training",
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// A dataset, as far as GPU residency is concerned.
#[derive(Debug, Clone, Copy)]
pub struct DatasetSpec {
    pub sample_count: u64,
    pub sample_bytes: u64,
}

impl DatasetSpec {
    pub fn total_bytes(&self) -> u64 {
        self.sample_count.saturating_mul(self.sample_bytes)
    }
}

/// How a dataset is cut up and how much of it is on the GPU at once.
///
/// This is the answer to "do we load everything, or piece by piece?" for the
/// data half of the question: **piece by piece, one piece at a time**.
#[derive(Debug, Clone, Copy)]
pub struct DatasetPlan {
    pub total_bytes: u64,
    /// Bytes of one chunk — and, since exactly one chunk is resident, the
    /// dataset's entire GPU footprint.
    pub chunk_bytes: u64,
    pub chunk_sample_capacity: u64,
    pub chunk_count: u64,
    /// Distinct chunks a shuffled batch touches: `Σᵢ (1 − (1 − pᵢ)^batch)` summed
    /// per chunk, NOT the tidy `C·(1 − ((C−1)/C)^batch)` — the last chunk is short
    /// (17 232 vs 32 768 on CIFAR grey), and charging it as full over-predicted 31 %.
    pub expected_uploads_per_step: f64,
    /// The same, in bytes, weighting each chunk by its own size.
    pub expected_upload_bytes_per_step: f64,
}

impl DatasetPlan {
    pub fn expected_upload_bytes_per_step(&self) -> f64 {
        self.expected_upload_bytes_per_step
    }
}

/// The chunk size `GpuDataset` will pick on a device with these limits. Calls the
/// dataset's own `select_chunk_bytes`, so the prediction cannot drift from the
/// allocation.
pub fn plan_dataset(
    device: &DeviceProfile,
    spec: DatasetSpec,
    batch: u32,
) -> DatasetPlan {
    let total = spec.total_bytes();
    let chunk_bytes = crate::model::training::dataset::select_chunk_bytes(
        device.max_storage_buffer_binding_size,
        device.max_buffer_size,
        // The ADAPTER's figure, the one `GpuDataset` reads: the device's predicted
        // 64 MiB chunks where the dataset allocates 128 MiB.
        device.adapter_max_buffer_size,
        total,
    );
    let sample_bytes = spec.sample_bytes.max(1);
    let capacity = (chunk_bytes / sample_bytes).max(1);
    let chunk_count = spec.sample_count.div_ceil(capacity).max(1);
    let batch = batch.max(1) as i32;

    // Per chunk, because the last one is short. `pᵢ` is the chance one drawn
    // sample lands in chunk `i`; the chunk is uploaded unless all `batch` draws
    // missed it.
    let mut expected_uploads = 0.0f64;
    let mut expected_bytes = 0.0f64;
    let mut remaining = spec.sample_count;
    for _ in 0..chunk_count {
        let held = remaining.min(capacity);
        remaining -= held;
        let p = if spec.sample_count == 0 {
            0.0
        } else {
            held as f64 / spec.sample_count as f64
        };
        let touched = 1.0 - (1.0 - p).powi(batch);
        expected_uploads += touched;
        expected_bytes += touched * (held * sample_bytes) as f64;
    }

    DatasetPlan {
        total_bytes: total,
        chunk_bytes: (capacity * sample_bytes).max(4),
        chunk_sample_capacity: capacity,
        chunk_count,
        expected_uploads_per_step: expected_uploads,
        expected_upload_bytes_per_step: expected_bytes,
    }
}

// ---------------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------------

/// Everything the inventory needs, and nothing it does not: no GPU, no
/// filesystem, no checkpoint.
#[derive(Debug, Clone)]
pub struct InventoryRequest {
    pub layers: Vec<LayerDraft>,
    pub input_size: (u32, u32, u32),
    pub batch: u32,
    pub workload: Workload,
    pub dataset: Option<DatasetSpec>,
    /// Whether a visualiser window is attached (`x_t | x̂₀` doubles the width).
    pub live_frame: bool,
}

impl InventoryRequest {
    /// A training inventory for what a `config_file` says it will run.
    ///
    /// The optimiser and the EMA come from the file rather than from a default:
    /// Adam doubles the parameter footprint and an EMA adds another copy, so a
    /// page that assumed SGD would under-report by 3× on the parameter posts.
    pub fn from_config(config: &ModelConfig, batch: u32) -> Self {
        let workload = match &config.run.mode {
            crate::config::RunMode::Train(train) => {
                Workload::training(train.optimizer, train.ema_decay.is_some())
            }
            _ => Workload::Inference,
        };
        Self {
            layers: config.layers.clone(),
            input_size: config.input_size,
            batch: batch.max(1),
            workload,
            dataset: None,
            live_frame: false,
        }
    }

    pub fn with_batch(mut self, batch: u32) -> Self {
        self.batch = batch.max(1);
        self
    }

    pub fn with_workload(mut self, workload: Workload) -> Self {
        self.workload = workload;
        self
    }

    pub fn with_dataset(mut self, dataset: Option<DatasetSpec>) -> Self {
        self.dataset = dataset;
        self
    }

    pub fn with_live_frame(mut self, live_frame: bool) -> Self {
        self.live_frame = live_frame;
        self
    }
}

// ---------------------------------------------------------------------------
// The verdict
// ---------------------------------------------------------------------------

/// Why an inventory does not fit. Each variant names the limit it met, because
/// "does not fit" without the limit is not actionable.
#[derive(Debug, Clone, PartialEq)]
pub enum Obstacle {
    /// One binding is larger than `max_storage_buffer_binding_size`. This is
    /// the one that bites first on a large model at a large batch, and it bites
    /// *per buffer*: total memory can be comfortable and the build still fail.
    BindingTooLarge {
        label: String,
        bytes: u64,
        limit: u64,
    },
    BufferTooLarge {
        label: String,
        bytes: u64,
        limit: u64,
    },
    /// The stated memory budget is smaller than the total.
    OverBudget { bytes: u64, budget: u64 },
    /// Past `max_compute_workgroups_per_dimension²`, which no 2-D split saves.
    DispatchTooLarge { workgroups: u64, limit: u64 },
}

impl std::fmt::Display for Obstacle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Obstacle::BindingTooLarge {
                label,
                bytes,
                limit,
            } => write!(
                f,
                "binding `{label}` is {} — over max_storage_buffer_binding_size ({})",
                format_bytes(*bytes),
                format_bytes(*limit)
            ),
            Obstacle::BufferTooLarge {
                label,
                bytes,
                limit,
            } => write!(
                f,
                "buffer `{label}` is {} — over max_buffer_size ({})",
                format_bytes(*bytes),
                format_bytes(*limit)
            ),
            Obstacle::OverBudget { bytes, budget } => write!(
                f,
                "total {} exceeds the {} budget",
                format_bytes(*bytes),
                format_bytes(*budget)
            ),
            Obstacle::DispatchTooLarge { workgroups, limit } => write!(
                f,
                "a dispatch of {workgroups} workgroups exceeds {limit}² even split in 2-D"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The inventory
// ---------------------------------------------------------------------------

/// Every GPU allocation of a run, itemised and confronted with a device.
#[derive(Debug, Clone)]
pub struct GpuInventory {
    pub device: DeviceProfile,
    pub workload: Workload,
    pub batch: u32,
    pub allocations: Vec<Allocation>,
    pub layers: Vec<LayerFootprint>,
    pub dataset: Option<DatasetPlan>,
    /// Trainable scalars over the whole stack, counted from the weight buffers.
    pub parameters: u64,
    /// Largest single dispatch of the graph, before the 2-D split.
    pub largest_dispatch: u64,
    /// `N²` floats of attention's softmax rows, over every attention layer —
    /// the term that grows quadratically in the number of positions.
    pub attention_probs_bytes: u64,
    pub obstacles: Vec<Obstacle>,
}

impl GpuInventory {
    pub fn total_bytes(&self) -> u64 {
        self.allocations.iter().map(|a| a.bytes).sum()
    }

    /// Everything that stays on the GPU for the whole run — the total minus the
    /// streamed dataset chunk.
    pub fn resident_bytes(&self) -> u64 {
        self.allocations
            .iter()
            .filter(|a| a.kind != Kind::DatasetChunk)
            .map(|a| a.bytes)
            .sum()
    }

    pub fn bytes_of(&self, kind: Kind) -> u64 {
        self.allocations
            .iter()
            .filter(|a| a.kind == kind)
            .map(|a| a.bytes)
            .sum()
    }

    /// Per-post totals, largest first, skipping empty posts.
    pub fn by_kind(&self) -> Vec<(Kind, u64)> {
        let mut totals: Vec<(Kind, u64)> = Kind::ALL
            .iter()
            .map(|&k| (k, self.bytes_of(k)))
            .filter(|(_, b)| *b > 0)
            .collect();
        totals.sort_by(|a, b| b.1.cmp(&a.1));
        totals
    }

    /// Bytes that move if the batch changes, and bytes that do not.
    pub fn batch_split(&self) -> (u64, u64) {
        let mut scaling = 0;
        let mut fixed = 0;
        for alloc in &self.allocations {
            if alloc.kind.scales_with_batch() {
                scaling += alloc.bytes;
            } else if alloc.kind != Kind::DatasetChunk {
                fixed += alloc.bytes;
            }
        }
        (scaling, fixed)
    }

    pub fn largest_allocation(&self) -> Option<&Allocation> {
        self.allocations.iter().max_by_key(|a| a.bytes)
    }

    pub fn fits(&self) -> bool {
        self.obstacles.is_empty()
    }
}

/// Walk the graph and count what it allocates.
pub fn inventory(
    request: &InventoryRequest,
    device: &DeviceProfile,
) -> Result<GpuInventory, ModelError> {
    let planned = plan_graph(&request.layers, request.input_size)?;
    let batch = match request.workload {
        // Inference runs one sample through the graph, whatever the training
        // batch of the config says. `predict()` writes slot 0 and reads slot 0.
        Workload::Inference => 1,
        Workload::Training { .. } => request.batch.max(1),
    };

    let mut walker = Walker::default();
    walker.forward(&planned, batch);
    if let Workload::Training {
        optimizer,
        ema,
        loss,
        diffusion,
    } = request.workload
    {
        walker.loss_and_backward(&planned, batch, loss, optimizer, ema);
        if diffusion {
            walker.diffusion_prepass(&planned, batch);
        }
    }

    let dataset = request
        .dataset
        .map(|spec| plan_dataset(device, spec, batch));
    if let Some(plan) = dataset {
        walker.allocations.push(Allocation {
            label: "training_dataset_chunk".to_string(),
            kind: Kind::DatasetChunk,
            bytes: plan.chunk_bytes,
            layer: None,
        });
    }

    if request.live_frame {
        let out = planned.output_dim();
        // `LiveView::Both` — the widest of the two, so the page never
        // under-reports.
        let bytes = (out.x as u64 * 2) * out.y as u64 * out.z as u64 * 4;
        walker.allocations.push(Allocation {
            label: "live_denoise_frame".to_string(),
            kind: Kind::LiveFrame,
            bytes: bytes.max(4),
            layer: None,
        });
    }

    let layers = walker.layer_footprints(&planned, &request.layers);
    let parameters = walker
        .allocations
        .iter()
        .filter(|a| a.kind == Kind::Weights)
        .map(|a| a.bytes / 4)
        .sum();
    let largest_dispatch = planned.largest_dispatch(batch, request.workload.is_training());
    let attention_probs_bytes = planned.attention_probs_bytes(batch);

    let mut inventory = GpuInventory {
        device: device.clone(),
        workload: request.workload,
        batch,
        allocations: walker.allocations,
        layers,
        dataset,
        parameters,
        largest_dispatch,
        attention_probs_bytes,
        obstacles: Vec::new(),
    };
    inventory.obstacles = obstacles_of(&inventory, device);
    Ok(inventory)
}

fn obstacles_of(inventory: &GpuInventory, device: &DeviceProfile) -> Vec<Obstacle> {
    let mut obstacles = Vec::new();
    for alloc in &inventory.allocations {
        // Uniforms are bound as uniforms, not as storage: the storage binding
        // cap does not apply to them (and they are 32 bytes anyway).
        if alloc.kind != Kind::Uniforms && alloc.bytes > device.max_storage_buffer_binding_size {
            obstacles.push(Obstacle::BindingTooLarge {
                label: alloc.label.clone(),
                bytes: alloc.bytes,
                limit: device.max_storage_buffer_binding_size,
            });
        }
        if alloc.bytes > device.max_buffer_size {
            obstacles.push(Obstacle::BufferTooLarge {
                label: alloc.label.clone(),
                bytes: alloc.bytes,
                limit: device.max_buffer_size,
            });
        }
    }
    if let Some(budget) = device.memory_budget {
        let total = inventory.total_bytes();
        if total > budget {
            obstacles.push(Obstacle::OverBudget {
                bytes: total,
                budget,
            });
        }
    }
    let per_dim = device.max_compute_workgroups_per_dimension as u64;
    if inventory.largest_dispatch > per_dim.saturating_mul(per_dim) {
        obstacles.push(Obstacle::DispatchTooLarge {
            workgroups: inventory.largest_dispatch,
            limit: per_dim,
        });
    }
    obstacles
}

/// Largest batch that still fits, searched over the same inventory.
///
/// Every post either grows linearly with the batch or does not move, so "fits"
/// is monotone in the batch and a doubling-then-bisection search is exact
/// rather than a sampling. `None` means not even one sample fits.
pub fn max_batch_that_fits(
    request: &InventoryRequest,
    device: &DeviceProfile,
    ceiling: u32,
) -> Result<Option<u32>, ModelError> {
    let fits = |batch: u32| -> Result<bool, ModelError> {
        Ok(inventory(&request.clone().with_batch(batch), device)?.fits())
    };
    if !fits(1)? {
        return Ok(None);
    }
    let mut low = 1u32;
    let mut high = 2u32;
    while high <= ceiling && fits(high)? {
        low = high;
        high = high.saturating_mul(2);
    }
    let mut high = high.min(ceiling.saturating_add(1));
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if fits(mid)? {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(Some(low.min(ceiling)))
}

// ---------------------------------------------------------------------------
// The graph, planned without a device
// ---------------------------------------------------------------------------

/// The layer stack as `Model` would build it: dimensions chained, skip targets
/// resolved.
#[derive(Debug, Clone)]
pub struct PlannedGraph {
    pub layers: Vec<LayerTypes>,
    /// `save_key` of each layer, in step with `layers`.
    pub save_keys: Vec<Option<String>>,
    pub displays: Vec<String>,
}

impl PlannedGraph {
    pub fn output_dim(&self) -> Dim3 {
        self.layers
            .last()
            .map(|l| l.get_dim_output())
            .unwrap_or_default()
    }

    pub fn input_dim(&self) -> Dim3 {
        self.layers
            .first()
            .map(|l| l.get_dim_input())
            .unwrap_or_default()
    }

    fn largest_dispatch(&self, batch: u32, training: bool) -> u64 {
        let mut largest = 0u64;
        for layer in &self.layers {
            let declared = layer.get_forward_entrypoints();
            let forward = if declared.is_empty() {
                vec![layer.get_forward_workgroup_count(batch)]
            } else {
                layer.get_forward_workgroup_counts(batch)
            };
            for count in forward {
                largest = largest.max(count as u64);
            }
            if training {
                for count in layer.get_back_workgroup_counts(batch) {
                    largest = largest.max(count as u64);
                }
            }
        }
        largest
    }

    fn attention_probs_bytes(&self, batch: u32) -> u64 {
        self.layers
            .iter()
            .filter_map(|layer| match layer {
                LayerTypes::Attention(att) => {
                    let dim = att.get_dim_input();
                    let seq = dim.x as u64 * dim.y as u64;
                    Some(seq * seq * batch as u64 * 4)
                }
                _ => None,
            })
            .sum()
    }
}

/// Turn a `config_file`'s layer list into the layer types the model builds, with
/// dimensions chained as `Layer::new` chains them — a draft carries the
/// `dim_input` it was WRITTEN with, and the model overrides it with the previous
/// layer's real output, so reading the draft's field would agree with the config
/// and disagree with the GPU.
pub fn plan_graph(
    drafts: &[LayerDraft],
    input_size: (u32, u32, u32),
) -> Result<PlannedGraph, ModelError> {
    let mut layers: Vec<LayerTypes> = Vec::with_capacity(drafts.len());
    let mut save_keys: Vec<Option<String>> = Vec::with_capacity(drafts.len());
    let mut displays: Vec<String> = Vec::with_capacity(drafts.len());
    let mut saved: HashMap<String, usize> = HashMap::new();

    for draft in drafts {
        let last_output = layers.last().map(|l: &LayerTypes| l.get_dim_output());
        let skip_dim = match draft {
            LayerDraft::Concat { skip_key, .. } | LayerDraft::Add { skip_key, .. } => Some(
                saved
                    .get(skip_key)
                    .map(|&index| layers[index].get_dim_output())
                    .ok_or_else(|| ModelError::MissingSavedOutput {
                        key: skip_key.clone(),
                    })?,
            ),
            // TimeBias resolves its referenced tensor the same way; `skip_dim`
            // carries that tensor's geometry so layer_type_of can size the
            // per-sample stride into the embedding.
            LayerDraft::TimeBias { time_key, .. } => Some(
                saved
                    .get(time_key)
                    .map(|&index| layers[index].get_dim_output())
                    .ok_or_else(|| ModelError::MissingSavedOutput {
                        key: time_key.clone(),
                    })?,
            ),
            _ => None,
        };
        let mut ty = layer_type_of(draft, skip_dim);
        if let Some(input) = last_output.or(Some(Dim3::new(input_size))) {
            ty.set_dim_input(input);
        }
        ty.set_dim_output()?;
        if let Some(key) = draft.save_key() {
            saved.insert(key.to_string(), layers.len());
        }
        save_keys.push(draft.save_key().map(str::to_string));
        displays.push(draft.display());
        layers.push(ty);
    }
    Ok(PlannedGraph {
        layers,
        save_keys,
        displays,
    })
}

/// The engine-side reading of one `LayerDraft`.
///
/// `skip_dim` is the resolved output of the layer a `Concat` re-injects — the
/// model resolves it from its saved outputs, and so must anything that wants
/// the same graph.
pub fn layer_type_of(draft: &LayerDraft, skip_dim: Option<Dim3>) -> LayerTypes {
    match draft {
        LayerDraft::Convolution {
            dim_input,
            nb_kernel,
            dim_kernel,
            stride,
            padding,
            ..
        } => LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new(*dim_input),
            *nb_kernel,
            Dim3::new(*dim_kernel),
            *stride,
            padding.into(),
        )),
        LayerDraft::Activation {
            dim_input, method, ..
        } => LayerTypes::Activation(ActivationType::new(
            (*method).into(),
            Dim3::new(*dim_input),
        )),
        LayerDraft::GroupNorm {
            dim_input,
            num_groups,
            ..
        } => LayerTypes::GroupNorm(GroupNormType::new(Dim3::new(*dim_input), *num_groups)),
        LayerDraft::Attention { dim_input, .. } => {
            LayerTypes::Attention(AttentionType::new(Dim3::new(*dim_input)))
        }
        LayerDraft::FullyConnected {
            dim_input,
            nb_neurons,
            method,
            ..
        } => LayerTypes::FullyConnected(FullyConnectedType::new(
            Dim3::new(*dim_input),
            *nb_neurons,
            (*method).into(),
        )),
        LayerDraft::UpsampleConv {
            dim_input,
            scale_factor,
            nb_kernel,
            dim_kernel,
            padding,
            ..
        } => LayerTypes::UpsampleConv(UpsampleConvType::new(
            Dim3::new(*dim_input),
            *scale_factor,
            *nb_kernel,
            Dim3::new(*dim_kernel),
            padding.into(),
        )),
        LayerDraft::Concat {
            dim_skip, skip_key, ..
        } => LayerTypes::Concat(ConcatType::new(
            skip_key.clone(),
            Dim3::default(),
            skip_dim.unwrap_or_else(|| Dim3::new(*dim_skip)),
        )),
        LayerDraft::Add {
            dim_skip, skip_key, ..
        } => LayerTypes::Add(AddType::new(
            skip_key.clone(),
            Dim3::default(),
            skip_dim.unwrap_or_else(|| Dim3::new(*dim_skip)),
        )),
        LayerDraft::TimeBias {
            time_key,
            embed_offset,
            embed_channels,
            ..
        } => {
            let time_dim = skip_dim.unwrap_or_default();
            let time_sample_len = time_dim.x * time_dim.y * time_dim.z;
            LayerTypes::TimeBias(TimeBiasType::new(
                time_key.clone(),
                Dim3::default(),
                *embed_offset,
                *embed_channels,
                time_sample_len,
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// Mirrors `Model::build`: same order, same binding lists, same sharing.
#[derive(Default)]
struct Walker {
    allocations: Vec<Allocation>,
    /// Buffer index of every forward binding of every layer — what the backward
    /// pass re-binds instead of allocating.
    forward_ids: Vec<Vec<usize>>,
}

impl Walker {
    fn alloc(&mut self, label: &str, bytes: u64, kind: Kind, layer: Option<usize>) -> usize {
        self.allocations.push(Allocation {
            label: label.to_string(),
            kind,
            bytes,
            layer,
        });
        self.allocations.len() - 1
    }

    fn forward(&mut self, graph: &PlannedGraph, batch: u32) {
        let mut last: Option<usize> = None;
        let mut saved: HashMap<String, usize> = HashMap::new();

        for (index, layer) in graph.layers.iter().enumerate() {
            let bindings = layer.get_forward_buffer_bindings(batch);
            let mut ids = Vec::with_capacity(bindings.len());
            for binding in bindings {
                let shared = match &binding.source {
                    ForwardBufferSource::PreviousOutput => last,
                    ForwardBufferSource::SavedOutput(key) => saved.get(key).copied(),
                    ForwardBufferSource::Allocate => None,
                };
                let id = match shared {
                    Some(id) => id,
                    None => {
                        let kind = forward_kind(&binding.name, layer);
                        self.alloc(&binding.name, binding.spec.size as u64, kind, Some(index))
                    }
                };
                ids.push(id);
            }
            last = ids.last().copied();
            if let (Some(key), Some(id)) = (graph.save_keys[index].as_ref(), last) {
                saved.insert(key.clone(), id);
            }
            self.forward_ids.push(ids);
        }
    }

    fn loss_and_backward(
        &mut self,
        graph: &PlannedGraph,
        batch: u32,
        loss: LossMethod,
        optimizer: OptimizerKind,
        ema: bool,
    ) {
        // --- the loss layer, whose binding 0 is the model's own output ---
        const LOSS_GRAD_OUTPUT_INDEX: usize = 3;
        let loss_type = LossType::new(loss, graph.output_dim());
        let mut loss_ids = Vec::new();
        for binding in loss_type.get_forward_buffer_bindings(batch) {
            let id = match binding.source {
                ForwardBufferSource::PreviousOutput => {
                    self.forward_ids.last().and_then(|ids| ids.last().copied())
                }
                _ => None,
            };
            let id = match id {
                Some(id) => id,
                None => {
                    let kind = if binding.name == "specs" {
                        Kind::Uniforms
                    } else {
                        Kind::LossScratch
                    };
                    self.alloc(&binding.name, binding.spec.size as u64, kind, None)
                }
            };
            loss_ids.push(id);
        }

        // --- backward, in reverse order, exactly as `build()` walks it ---
        let mut incoming = loss_ids.get(LOSS_GRAD_OUTPUT_INDEX).copied();
        let mut pending_saved_grads: HashMap<String, usize> = HashMap::new();

        for index in (0..graph.layers.len()).rev() {
            let layer = &graph.layers[index];
            let mut grad_for_layer = incoming;
            if let Some(key) = graph.save_keys[index].as_ref() {
                if pending_saved_grads.remove(key).is_some() {
                    // `create_merge_pass` allocates one activation-shaped buffer
                    // to sum the two gradients arriving at a saved output.
                    let bytes =
                        layer.get_dim_output().bytes_size() as u64 * batch.max(1) as u64;
                    grad_for_layer = Some(self.alloc(
                        "grad_merge",
                        bytes.max(4),
                        Kind::ActivationGradients,
                        Some(index),
                    ));
                }
            }

            let bindings = layer.get_back_buffer_bindings(batch);
            if bindings.is_empty() {
                continue;
            }
            let mut ids = Vec::with_capacity(bindings.len());
            for binding in bindings {
                let shared = match binding.source {
                    BackwardBufferSource::Forward(i) => self.forward_ids[index].get(i).copied(),
                    BackwardBufferSource::IncomingGradient => grad_for_layer,
                    BackwardBufferSource::Allocate => None,
                };
                let id = match shared {
                    Some(id) => id,
                    None => {
                        let kind = backward_kind(&binding.name, layer);
                        self.alloc(&binding.name, binding.spec.size as u64, kind, Some(index))
                    }
                };
                ids.push(id);
            }
            if let Some(grad_input) = layer.get_back_grad_input_index() {
                incoming = ids.get(grad_input).copied();
            }
            for route in layer.get_saved_gradient_routes() {
                if let Some(&id) = ids.get(route.buffer_index) {
                    pending_saved_grads.insert(route.key, id);
                }
            }

            // --- optimiser and EMA, on trainable layers only ---
            if let Some(bindings) = layer.get_optimizer_bindings() {
                let weights = self.allocations[self.forward_ids[index]
                    [bindings.weights_forward_index]]
                    .bytes;
                let bias = self.allocations[self.forward_ids[index][bindings.bias_forward_index]]
                    .bytes;
                self.alloc(
                    "opt_specs_uniform",
                    OPT_SPECS_BYTES,
                    Kind::Uniforms,
                    Some(index),
                );
                if matches!(optimizer, OptimizerKind::Adam) {
                    for label in ["adam_m_weights", "adam_v_weights"] {
                        self.alloc(label, weights, Kind::OptimizerState, Some(index));
                    }
                    for label in ["adam_m_bias", "adam_v_bias"] {
                        self.alloc(label, bias, Kind::OptimizerState, Some(index));
                    }
                }
                if ema {
                    self.alloc(
                        "ema_specs_uniform",
                        EmaSpecs::BYTES as u64,
                        Kind::Uniforms,
                        Some(index),
                    );
                    self.alloc("ema_weights", weights, Kind::Ema, Some(index));
                    self.alloc("ema_bias", bias, Kind::Ema, Some(index));
                }
            }
        }
    }

    fn diffusion_prepass(&mut self, graph: &PlannedGraph, batch: u32) {
        let output = graph.output_dim();
        self.alloc(
            "diffusion_clean_target",
            crate::training::DiffusionTask::prepare_target_bytes(output, batch),
            Kind::DiffusionScratch,
            None,
        );
        self.alloc(
            "diffusion_prepare_specs",
            crate::training::DiffusionTask::prepare_specs_bytes(batch),
            Kind::DiffusionScratch,
            None,
        );
    }

    fn layer_footprints(
        &self,
        graph: &PlannedGraph,
        drafts: &[LayerDraft],
    ) -> Vec<LayerFootprint> {
        (0..graph.layers.len())
            .map(|index| {
                let mine: Vec<&Allocation> = self
                    .allocations
                    .iter()
                    .filter(|a| a.layer == Some(index))
                    .collect();
                LayerFootprint {
                    index,
                    display: graph
                        .displays
                        .get(index)
                        .cloned()
                        .or_else(|| drafts.get(index).map(|d| d.display()))
                        .unwrap_or_default(),
                    parameters: mine
                        .iter()
                        .filter(|a| a.kind == Kind::Weights)
                        .map(|a| a.bytes / 4)
                        .sum(),
                    bytes: mine.iter().map(|a| a.bytes).sum(),
                    batched_bytes: mine
                        .iter()
                        .filter(|a| a.kind.scales_with_batch())
                        .map(|a| a.bytes)
                        .sum(),
                    largest_binding: mine.iter().map(|a| a.bytes).max().unwrap_or(0),
                }
            })
            .collect()
    }
}

/// Which post a forward binding belongs to, from the layer's own name for it —
/// reading what the layer says the buffer is, not guessing from its size.
fn forward_kind(name: &str, layer: &LayerTypes) -> Kind {
    match name {
        "specs" => Kind::Uniforms,
        "weights" | "bias" | "gamma" | "beta" => Kind::Weights,
        "scratch" => match layer {
            LayerTypes::Attention(_) => Kind::AttentionScratch,
            _ => Kind::Activations,
        },
        _ => Kind::Activations,
    }
}

fn backward_kind(name: &str, layer: &LayerTypes) -> Kind {
    match name {
        "specs" => Kind::Uniforms,
        "grad_weights" | "grad_bias" | "grad_gamma" | "grad_beta" => Kind::ParameterGradients,
        "weights" | "bias" | "gamma" | "beta" => Kind::Weights,
        "fwd_scratch" | "grad_scratch" => match layer {
            LayerTypes::Attention(_) => Kind::AttentionScratch,
            _ => Kind::ActivationGradients,
        },
        name if name.starts_with("grad_") => Kind::ActivationGradients,
        _ => Kind::Activations,
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Bytes in binary units (MiB/GiB), the units GPU limits use — NOT the SI form the
/// checkpoint selector uses, which next to a limit would invite a wrong subtraction.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if b < KIB * KIB {
        format!("{:.1} KiB", b / KIB)
    } else if b < KIB * KIB * KIB {
        format!("{:.1} MiB", b / (KIB * KIB))
    } else {
        format!("{:.2} GiB", b / (KIB * KIB * KIB))
    }
}

pub mod report;

#[cfg(test)]
mod tests;
