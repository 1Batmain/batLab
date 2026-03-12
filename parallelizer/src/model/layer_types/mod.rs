use crate::model::error::ModelError;
use crate::model::types::{BufferSpec, Dim3};
use enum_dispatch::enum_dispatch;

mod activation;
mod convolution;
mod fully_connected;
mod loss;
mod pooling;

pub use activation::{ActivationMethod, ActivationType};
pub use convolution::ConvolutionType;
pub use fully_connected::FullyConnectedType;
pub use loss::{LossMethod, LossType};
pub use pooling::PoolingType;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ShaderDescriptor {
    pub label: &'static str,
    pub source: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BufferInit {
    None,
    RandomWeights,
    SpecsUniform,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BackwardBufferSource {
    Forward(usize),
    IncomingGradient,
    Allocate,
}

#[derive(Debug, Clone)]
pub(crate) struct ForwardBufferBinding {
    pub name: String,
    pub spec: BufferSpec,
    pub init: BufferInit,
}

#[derive(Debug, Clone)]
pub(crate) struct BackwardBufferBinding {
    pub name: String,
    pub spec: BufferSpec,
    pub source: BackwardBufferSource,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OptimizerBindings {
    pub weight_count: u32,
    pub weights_forward_index: usize,
    pub bias_forward_index: usize,
    pub grad_weights_backward_index: usize,
    pub grad_bias_backward_index: usize,
}

#[enum_dispatch]
pub(crate) trait LayerType: std::fmt::Debug + Send + Sync {
    fn get_forward_shader(&self) -> ShaderDescriptor;
    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        None
    }
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
    fn get_forward_buffer_bindings(&self) -> Vec<ForwardBufferBinding>;
    fn get_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
        self.get_forward_buffer_bindings()
            .into_iter()
            .map(|binding| (binding.name, binding.spec))
            .collect()
    }
    fn get_back_buffer_bindings(&self) -> Vec<BackwardBufferBinding> {
        vec![]
    }
    /// Specs for ALL bindings in the backward bind group (shared forward + new buffers).
    fn get_back_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
        self.get_back_buffer_bindings()
            .into_iter()
            .map(|binding| (binding.name, binding.spec))
            .collect()
    }
    fn get_back_grad_input_index(&self) -> Option<usize> {
        None
    }
    fn get_optimizer_bindings(&self) -> Option<OptimizerBindings> {
        None
    }
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
    FullyConnected(FullyConnectedType),
    Loss(LossType),
}
