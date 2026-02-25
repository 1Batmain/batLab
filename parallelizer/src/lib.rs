mod gpu_context;
mod layer;
mod model;
mod persistence;
mod spec;
mod types;
#[cfg(feature = "visualisation")]
mod visualiser;

pub use gpu_context::GpuContext;
pub use model::{Model, TrainingSpec};
pub use spec::{ActivationLayerSpec, ConvolutionLayerSpec, LayerSpec};
pub use types::{ActivationMethod, Dim3, Optimizer, PaddingMode};
#[cfg(feature = "visualisation")]
pub use visualiser::Visualiser;