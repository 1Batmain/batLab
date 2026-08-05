//! File purpose: Core crate root that exposes GPU context, model, training, TUI, and visualiser modules.

pub mod gpu_context;
pub mod model;
pub mod tui;
pub mod visualiser;

pub use gpu_context::GpuContext;
pub use model::Model;
pub use model::training;
pub use model::training::{
    BucketStat, CLIMB_TEMPO_RATIO, DEFAULT_SNR_GAMMA, DenoiseFrame, DenoiseStepStat, DiffusionTask,
    DriftAction, DriftPhase, GpuDataset, GpuDatasetError, LinearNoiseSchedule, LossWeighting,
    MIN_RENOISE_DEPTH, MetricsLogger, PerpetualDrift, PerpetualRegime, ProbeConfig,
    RENOISE_DEPTH_STEP, Stats, TaskPassSpec, Trainer, TrainingTask, TrainingTaskError, Workgroups,
    compose_diffusion_input, log_probe, log_train_loss, log_trajectory, probe_diffusion,
    reverse_step, reverse_step_seed, sample_diffusion,
};
pub use model::{
    ActivationMethod, ActivationType, AdamHyperparameters, ConcatType, ConvolutionType, Dim3,
    FullyConnectedType, GroupNormType, LayerTypes, LossMethod, LossType, ModelError, OptimizerKind,
    PaddingMode, UpsampleConvType, WeightInit,
};
pub use visualiser::{LiveFrame, compose_live_frame, live_frame_width};
