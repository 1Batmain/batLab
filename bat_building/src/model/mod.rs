// Model module declarations
pub mod debug;
pub mod error;
pub mod layer;
pub mod layer_types;
pub mod model;
pub mod types;

pub use error::ModelError;
pub use layer_types::{
    ActivationMethod, ActivationType, ConcatType, ConvolutionType, FullyConnectedType,
    GroupNormType, LayerTypes, LossMethod, LossType, UpsampleConvType,
};
pub use model::{Infer, Model, Training};
pub use types::{Dim3, PaddingMode};
