// Model module declarations
pub mod layer;
pub mod model;
pub mod types;

// Re-export key types for easier access
pub use layer::{ActivationType, ConvolutionType};
pub use model::Model;
pub use types::{ActivationMethod, Dim3, Optimizer, PaddingMode};
