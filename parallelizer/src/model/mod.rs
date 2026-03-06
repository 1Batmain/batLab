// Model module declarations
pub mod error;
pub mod layer;
pub mod layer_types;
pub mod model;
pub mod types;

// Re-export key types for easier access
pub use error::ModelError;
pub use layer_types::{ActivationMethod, ActivationType, ConvolutionType, LayerTypes};
pub use model::Model;
pub use types::{Dim3, Optimizer, PaddingMode};
