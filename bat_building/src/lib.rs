//! File purpose: Core crate root that exposes GPU context, model, training, TUI, and visualiser modules.

pub mod gpu_context;
pub mod model;
pub mod tui;
pub mod visualiser;

pub use gpu_context::GpuContext;
pub use model::Model;
pub use model::training;
pub use model::training::{
    DiffusionTask, GpuDataset, GpuDatasetError, LinearNoiseSchedule, TaskPassSpec, Trainer,
    TrainingTask, TrainingTaskError, Workgroups,
};
pub use model::{
    ActivationMethod, ActivationType, ConcatType, ConvolutionType, Dim3, FullyConnectedType,
    GroupNormType, LayerTypes, LossMethod, LossType, ModelError, PaddingMode, UpsampleConvType,
};
