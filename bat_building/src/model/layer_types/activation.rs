//! File purpose: Defines the activation layer type, shapes, and GPU bindings used by the model graph.

use crate::model::error::ModelError;
use crate::model::layer_types::{
    BackwardBufferBinding, BackwardBufferSource, BufferInit, ForwardBufferBinding,
    ForwardBufferSource, LayerType, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use serde::{Deserialize, Serialize};
use wgpu::BufferUsages;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationMethod {
    Relu,
    Silu,
    Linear,
}

impl ActivationMethod {
    pub(crate) fn shader_index(self) -> u32 {
        match self {
            ActivationMethod::Relu => 0,
            ActivationMethod::Linear => 1,
            ActivationMethod::Silu => 2,
        }
    }

    pub(crate) fn forward_entrypoint(self) -> &'static str {
        match self {
            ActivationMethod::Relu => "relu",
            ActivationMethod::Linear => "linear",
            ActivationMethod::Silu => "silu",
        }
    }

    pub(crate) fn backward_entrypoint(self) -> &'static str {
        match self {
            ActivationMethod::Relu => "relu_back",
            ActivationMethod::Linear => "linear_back",
            ActivationMethod::Silu => "silu_back",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActivationType {
    pub method: ActivationMethod,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct ActivationUniform {
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl ActivationType {
    pub fn new(method: ActivationMethod, dim_input: Dim3) -> Self {
        Self {
            method,
            dim_input,
            dim_output: dim_input,
        }
    }
}

impl LayerType for ActivationType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "activation",
            source: include_str!("../shader/activation.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_activation",
            source: include_str!("../shader/back_activation.wgsl"),
        })
    }

    fn get_entrypoint(&self) -> &str {
        self.method.forward_entrypoint()
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec![self.method.backward_entrypoint()]
    }

    fn get_back_workgroup_counts(&self) -> Vec<u32> {
        vec![self.dim_input.length().div_ceil(64)]
    }

    fn get_dim_input(&self) -> Dim3 {
        self.dim_input
    }

    fn get_dim_output(&self) -> Dim3 {
        self.dim_output
    }

    fn get_forward_buffer_bindings(&self) -> Vec<ForwardBufferBinding> {
        self.get_buffers_specs()
            .into_iter()
            .map(|(name, spec)| ForwardBufferBinding {
                init: if name == "specs" {
                    BufferInit::SpecsUniform
                } else {
                    BufferInit::None
                },
                source: if name == "input" {
                    ForwardBufferSource::PreviousOutput
                } else {
                    ForwardBufferSource::Allocate
                },
                name,
                spec,
            })
            .collect()
    }

    fn get_back_buffer_bindings(&self) -> Vec<BackwardBufferBinding> {
        self.get_back_buffers_specs()
            .into_iter()
            .enumerate()
            .map(|(index, (name, spec))| BackwardBufferBinding {
                name,
                spec,
                source: match index {
                    0 => BackwardBufferSource::Forward(0),
                    1 => BackwardBufferSource::Forward(1),
                    2 => BackwardBufferSource::IncomingGradient,
                    _ => BackwardBufferSource::Allocate,
                },
            })
            .collect()
    }

    fn get_back_grad_input_index(&self) -> Option<usize> {
        Some(3)
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        self.dim_output = self.dim_input;
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
        vec![
            // [0] input  — shared with previous layer's output
            (
                "input".to_string(),
                BufferSpec {
                    size: self.get_dim_input().bytes_size().max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            // [1] specs uniform
            (
                "specs".to_string(),
                BufferSpec {
                    size: self.get_spec_uniform_bytes_size().max(4),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(self.get_spec_uniform_bytes_size() as u64)
                                .unwrap(),
                        ),
                    },
                },
            ),
            // [2] output
            (
                "output".to_string(),
                BufferSpec {
                    size: self.get_dim_output().bytes_size().max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
        ]
    }

    fn get_back_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
        // Backward bind group layout:
        //   [0] fwd_input  — shared from forward[0]
        //   [1] specs      — shared from forward[1]
        //   [2] grad_output — incoming gradient from next layer / loss
        //   [3] grad_input  — outgoing gradient to previous layer (NEW)
        vec![
            (
                "fwd_input".to_string(),
                BufferSpec {
                    size: self.dim_input.bytes_size().max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            (
                "specs".to_string(),
                BufferSpec {
                    size: self.get_spec_uniform_bytes_size().max(4),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(self.get_spec_uniform_bytes_size() as u64)
                                .unwrap(),
                        ),
                    },
                },
            ),
            (
                "grad_output".to_string(),
                BufferSpec {
                    size: self.dim_output.bytes_size().max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            (
                "grad_input".to_string(),
                BufferSpec {
                    size: self.dim_input.bytes_size().max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        ActivationUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = ActivationUniform {
            dim_input: self.dim_input,
            dim_output: self.dim_output,
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&uniform)
            .expect("failed to encode activation uniform");
        buffer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::ActivationMethod;

    #[test]
    fn silu_uses_expected_shader_metadata() {
        assert_eq!(ActivationMethod::Silu.shader_index(), 2);
        assert_eq!(ActivationMethod::Silu.forward_entrypoint(), "silu");
        assert_eq!(ActivationMethod::Silu.backward_entrypoint(), "silu_back");
    }
}
