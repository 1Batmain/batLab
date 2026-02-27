mod gpu_context;
mod layer;
mod model;
mod persistence;
mod spec;
mod types;

pub use gpu_context::GpuContext;
pub use model::{Model, TrainingSpec};
pub use spec::{ActivationLayerSpec, ConvolutionLayerSpec, LayerSpec};
pub use types::{ActivationMethod, Dim3, Optimizer, PaddingMode};

// visualiser lives in another crate now
pub use model::ModelVisualState;