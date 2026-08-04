//! File purpose: Core crate root that exposes GPU context, model, training, TUI, and visualiser modules.

pub mod gpu_context;
pub mod model;
pub mod tui;
pub mod visualiser;

pub use gpu_context::GpuContext;
pub use model::Model;
pub use model::training;
pub use model::training::{
    BucketStat, DenoiseStepStat, DiffusionTask, GpuDataset, GpuDatasetError, LinearNoiseSchedule,
    MetricsLogger, ProbeConfig, Stats, TaskPassSpec, Trainer, TrainingTask, TrainingTaskError,
    Workgroups, compose_diffusion_input, log_probe, log_train_loss, log_trajectory,
    probe_diffusion, sample_diffusion,
};
pub use model::{
    ActivationMethod, ActivationType, ConcatType, ConvolutionType, Dim3, FullyConnectedType,
    GroupNormType, LayerTypes, LossMethod, LossType, ModelError, PaddingMode, UpsampleConvType,
};
