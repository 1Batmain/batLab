// Model module declarations
pub mod layer;
pub mod persistence;
pub mod spec;
pub mod types;
pub mod model;

// Re-export key types for easier access
pub use model::{Model, TrainingSpec, ModelVisualState};
pub use spec::{ActivationLayerSpec, ConvolutionLayerSpec, LayerSpec};
pub use types::{ActivationMethod, Dim3, Optimizer, PaddingMode};

