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

pub mod config;
pub mod gpu_context;
pub mod live_frame;
pub mod model;

// The config mirror types `PaddingMode`, `ActivationMethod` and `LossMethod`
// deliberately stay behind `config::`: the engine exports layer types under those
// same names, and flattening both sets into the root would make every `use` site
// guess which one it got.
pub use config::{
    InferenceConfig, LayerDraft, ModelConfig, PerpetualConfig, RunConfig, RunMode, TrainingConfig,
    TrainingControlCommand, compute_inferred_input,
};
pub use gpu_context::GpuContext;
pub use live_frame::{LiveFrame, compose_live_frame, live_frame_width};
pub use model::Model;
pub use model::training;
pub use model::training::{
    BucketStat, CLIMB_TEMPO_RATIO, DEFAULT_SNR_GAMMA, DenoiseFrame, DenoiseStepStat, DiffusionTask,
    DriftAction, DriftPhase, GpuDataset, GpuDatasetError, LinearNoiseSchedule, LossWeighting,
    MIN_FLUX_LEVEL, MIN_RENOISE_DEPTH, MetricsLogger, PerpetualDrift, PerpetualRegime, ProbeConfig,
    RENOISE_DEPTH_STEP, Stats, TaskPassSpec, Trainer, TrainingTask, TrainingTaskError, Workgroups,
    compose_diffusion_input, log_probe, log_train_loss, log_trajectory, probe_diffusion,
    reverse_step, reverse_step_seed, sample_diffusion,
};
pub use model::{
    ActivationMethod, ActivationType, AdamHyperparameters, ConcatType, ConvolutionType, Dim3,
    FullyConnectedType, GroupNormType, LayerTypes, LossMethod, LossType, ModelError, OptimizerKind,
    PaddingMode, UpsampleConvType, WeightInit,
};
