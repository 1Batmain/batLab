//! File purpose: Engine crate root — GPU context, model/layers/shaders, training
//! and inference, and the serialisable model description.
//!
//! # Scope
//! This crate is the part of batLab meant to run anywhere wgpu runs, including a
//! browser on the visitor's own GPU (WebGPU). It therefore holds the *whole*
//! inference chain — config parsing, checkpoint decoding, `compose_diffusion_input`,
//! `reverse_step`/`sample_diffusion`, `PerpetualDrift` — and nothing that assumes
//! a terminal or a native window. Those live in `batlab_ui`.
//!
//! [`LiveFrame`] is the seam between the two: the engine composes the frame into
//! a GPU buffer here, whoever displays it (a winit window today, a canvas later)
//! only reads that buffer.

/// The production diffusion schedule, as one definition.
///
/// These were three `const`s in the binary (`crates/batlab/src/main.rs`), which
/// meant a second consumer of "the schedule the model was trained and sampled
/// with" — the web build being the first — had to copy them and could drift a
/// step or a beta out of agreement, silently changing every ᾱ. They live here
/// now; the binary re-exports these, and the browser reads the same three.
pub const DIFFUSION_SCHEDULE_STEPS: usize = 256;
/// Start of the linear beta schedule, calibrated for T = 1000 and rescaled by
/// [`LinearNoiseSchedule::new_linear`] for the actual step count.
pub const DIFFUSION_BETA_START: f32 = 1e-4;
/// End of the linear beta schedule (see [`DIFFUSION_BETA_START`]).
pub const DIFFUSION_BETA_END: f32 = 2e-2;

pub mod config;
pub mod gpu_context;
pub mod live_frame;
pub mod model;
pub mod profile;
pub mod resources;
pub mod transfers;
pub mod tuning;

// The config mirror types `PaddingMode`, `ActivationMethod` and `LossMethod`
// deliberately stay behind `config::`: the engine exports layer types under those
// same names, and flattening both sets into the root would make every `use` site
// guess which one it got.
pub use config::{
    ArchitectureRow, ArchitectureSummary, InferenceConfig, LayerDraft, ModelConfig,
    PerpetualConfig, RunConfig, RunMode, TrainingConfig, TrainingControlCommand,
    compute_inferred_input, summarize_architecture,
};
pub use gpu_context::{GpuContext, GpuLimitsProfile, pass_profiling_requested, request_pass_profiling};
pub use profile::{PassSummary, PassTiming, ProfileRun, ProfileSummary};
pub use resources::report::{MeasuredTransfers, ReportOptions, report_lines};
pub use resources::{
    Allocation, DatasetPlan, DatasetSpec, DeviceProfile, GpuInventory, InventoryRequest, Kind,
    LayerFootprint, Obstacle, ProfileSource, Workload, format_bytes, inventory,
    max_batch_that_fits,
};
pub use transfers::{TransferRate, TransferSnapshot};
pub use live_frame::{
    LiveFrame, LiveView, compose_live_frame, compose_live_frame_view, live_frame_width,
};
pub use model::Model;
pub use model::training;
pub use model::training::{
    AsyncNoisePredictor,
    BaselineBucket, BucketStat, CLIMB_TEMPO_RATIO, DEFAULT_SNR_GAMMA, DatasetPayload, DenoiseFrame,
    DenoiseStepStat, DiffusionTask, DriftAction, DriftFrame, DriftPhase, DriftWalk, EvalConfig,
    EvalReport, GpuDataset, GpuDatasetError,
    LinearNoiseSchedule, LossWeighting, MIN_FLUX_LEVEL, MIN_RENOISE_DEPTH, MetricsLogger, MseBucket,
    evaluate,
    NoisePredictor, PerpetualDrift, PerpetualOrigin, PerpetualRegime, ProbeConfig,
    RENOISE_DEPTH_STEP, Stats,
    TaskPassSpec, Trainer, TrainingTask, TrainingTaskError, Workgroups, compose_diffusion_input,
    decode_u8, log_probe, log_train_loss, log_trajectory, predict_epsilon, probe_diffusion, reverse_step,
    reverse_step_from_epsilon, reverse_step_seed, sample_diffusion,
};
pub use model::{
    ActivationMethod, ActivationType, AttentionType, AdamHyperparameters, CheckpointLoad,
    AddType, CheckpointWeights, ConcatType, ConvolutionType, Dim3, EmaConfig,
    FullyConnectedType, GroupNormType, LayerTypes, LossMethod, LossType, ModelError, OptimizerKind,
    PaddingMode, TimeBiasType, UpsampleConvType, WeightInit,
};
