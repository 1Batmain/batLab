use crate::model::error::ModelError;
use crate::model::types::{BufferSpec, Dim3};
use enum_dispatch::enum_dispatch;

mod activation;
mod convolution;

pub use activation::{ActivationMethod, ActivationType};
pub use convolution::ConvolutionType;

#[enum_dispatch]
pub(crate) trait LayerType: std::fmt::Debug + Send + Sync {
    fn get_entrypoint(&self) -> &str {
        "main"
    }
    fn get_dim_input(&self) -> Dim3;
    fn get_dim_output(&self) -> Dim3;
    fn get_buffers_specs(&self) -> Vec<(String, BufferSpec)>;
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
}
