//! File purpose: Core crate root that exposes GPU context, model, training, TUI, and visualiser modules.

pub mod gpu_context;
pub mod model;
pub mod tui;
pub mod visualiser;

pub use gpu_context::GpuContext;
pub use model::Model;
pub use model::training;
pub use model::training::{
    BucketStat, DEFAULT_SNR_GAMMA, DenoiseFrame, DenoiseStepStat, DiffusionTask, GpuDataset,
    GpuDatasetError,
    LinearNoiseSchedule, LossWeighting, MetricsLogger, ProbeConfig, Stats, TaskPassSpec, Trainer,
    TrainingTask, TrainingTaskError, Workgroups, compose_diffusion_input, log_probe,
    log_train_loss, log_trajectory, probe_diffusion, sample_diffusion,
};
pub use visualiser::LiveFrame;
pub use model::{
    ActivationMethod, ActivationType, AdamHyperparameters, ConcatType, ConvolutionType, Dim3,
    FullyConnectedType, GroupNormType, LayerTypes, LossMethod, LossType, ModelError, OptimizerKind,
    PaddingMode, UpsampleConvType, WeightInit,
};
