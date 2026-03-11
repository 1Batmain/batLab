use crate::model::error::ModelError;
use crate::model::types::{BufferSpec, Dim3};
use enum_dispatch::enum_dispatch;

mod activation;
mod convolution;
mod loss;

pub use activation::{ActivationMethod, ActivationType};
pub use convolution::ConvolutionType;
pub use loss::{LossMethod, LossType};

#[enum_dispatch]
pub(crate) trait LayerType: std::fmt::Debug + Send + Sync {
    fn get_entrypoint(&self) -> &str {
        "main"
    }
    /// Entry points for the backward compute passes (one per sub-pass).
    /// Empty for layers that have no backward pass (e.g. Loss).
    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec![]
    }
    /// Workgroup counts for each backward sub-pass (must match get_back_entrypoints length).
    fn get_back_workgroup_counts(&self) -> Vec<u32> {
        vec![]
    }
    /// Whether this layer has trainable weight buffers.
    fn has_weights(&self) -> bool {
        false
    }
    fn get_dim_input(&self) -> Dim3;
    fn get_dim_output(&self) -> Dim3;
    fn get_buffers_specs(&self) -> Vec<(String, BufferSpec)>;
    /// Specs for ALL bindings in the backward bind group (shared forward + new buffers).
    fn get_back_buffers_specs(&self) -> Vec<(String, BufferSpec)>;
    fn set_dim_input(&mut self, input: Dim3);
    fn set_dim_output(&mut self) -> Result<Dim3, ModelError>;
    fn get_spec_uniform_bytes_size(&self) -> u32;
    fn get_spec_uniform_bytes(&self) -> Vec<u8>;
}

#[enum_dispatch(LayerType)]
#[derive(Debug, Clone)]
pub enum LayerTypes {
    Convolution(ConvolutionType),
    Activation(ActivationType),
    Loss(LossType),
}
