//! File purpose: Module entry point for model; wires submodules and shared exports.

// Model module declarations
pub mod debug;
pub mod ema;
pub mod error;
pub mod layer;
pub mod layer_types;
pub mod model;
pub mod optimizer;
pub mod training;
pub mod types;
pub mod weight_init;

pub use ema::EmaConfig;
pub use error::ModelError;
pub use layer_types::{
    ActivationMethod, ActivationType, AddType, AttentionType, ConcatType, ConvolutionType,
    FullyConnectedType, GroupNormType, LayerTypes, LossMethod, LossType, UpsampleConvType,
};
pub use model::{CheckpointLoad, CheckpointWeights, Infer, Model, Training};
pub use optimizer::{AdamHyperparameters, OptimizerKind};
pub use types::{Dim3, PaddingMode};
pub use weight_init::WeightInit;

#[cfg(test)]
mod adam_tests;
#[cfg(test)]
mod add_equivalence_tests;
#[cfg(test)]
mod attention_tests;
#[cfg(test)]
mod audit_tests;
#[cfg(test)]
mod batch_equivalence_tests;
#[cfg(test)]
mod conv_equivalence_tests;
#[cfg(test)]
mod ema_tests;
#[cfg(test)]
mod group_norm_equivalence_tests;
#[cfg(test)]
mod upsample_conv_equivalence_tests;
