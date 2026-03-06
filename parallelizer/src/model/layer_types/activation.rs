use crate::model::error::ModelError;
use crate::model::layer_types::LayerType;
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use serde::{Deserialize, Serialize};
use wgpu::BufferUsages;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationMethod {
    Relu,
    Linear,
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
    fn get_entrypoint(&self) -> &str {
        match self.method {
            ActivationMethod::Relu => "relu",
            ActivationMethod::Linear => "linear",
        }
    }
    fn get_dim_input(&self) -> Dim3 {
        self.dim_input
    }
    fn get_dim_output(&self) -> Dim3 {
        self.dim_output
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
            (
                "input".to_string(),
                BufferSpec {
                    size: self.get_dim_input().bytes_size().max(4) as u32,
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
                "output".to_string(),
                BufferSpec {
                    size: (self.get_dim_output().bytes_size()).max(4) as u32,
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
