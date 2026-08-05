//! File purpose: Module entry point for model; wires submodules and shared exports.

// Model module declarations
pub mod debug;
pub mod error;
pub mod layer;
pub mod layer_types;
pub mod model;
pub mod optimizer;
pub mod training;
pub mod types;
pub mod weight_init;

pub use error::ModelError;
pub use layer_types::{
    ActivationMethod, ActivationType, ConcatType, ConvolutionType, FullyConnectedType,
    GroupNormType, LayerTypes, LossMethod, LossType, UpsampleConvType,
};
pub use model::{Infer, Model, Training};
pub use optimizer::{AdamHyperparameters, OptimizerKind};
pub use types::{Dim3, PaddingMode};
pub use weight_init::WeightInit;

#[cfg(test)]
mod adam_tests;
#[cfg(test)]
mod audit_tests;
#[cfg(test)]
mod group_norm_equivalence_tests;
