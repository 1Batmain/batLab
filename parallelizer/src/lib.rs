pub mod gpu_context;
pub mod model;
pub mod visualizer;

pub use gpu_context::GpuContext;
pub use model::{Model, TrainingSpec, ModelVisualState};
pub use model::{ActivationLayerSpec, ConvolutionLayerSpec, LayerSpec};
pub use model::{ActivationMethod, Dim3, Optimizer, PaddingMode};