//! File purpose: Defines the group norm layer type, shapes, and GPU bindings used by the model graph.

use crate::model::error::ModelError;
use crate::model::layer_types::{
    BackwardBufferBinding, BackwardBufferSource, BufferInit, ForwardBufferBinding,
    ForwardBufferSource, LayerType, OptimizerBindings, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

const DEFAULT_EPSILON: f32 = 1e-5;

#[derive(Debug, Clone, Copy)]
pub struct GroupNormType {
    pub num_groups: u32,
    pub epsilon: f32,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct GroupNormUniform {
    pub num_groups: u32,
    pub channels_per_group: u32,
    pub spatial_len: u32,
    pub epsilon: f32,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl GroupNormType {
    pub fn new(dim_input: Dim3, num_groups: u32) -> Self {
        Self {
            num_groups,
            epsilon: DEFAULT_EPSILON,
            dim_input,
            dim_output: dim_input,
        }
    }

    fn channel_param_bytes(&self) -> u32 {
        (self.dim_input.z * 4).max(4)
    }

    fn channels_per_group(&self) -> u32 {
        self.dim_input.z / self.num_groups
    }
}

impl LayerType for GroupNormType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "group_norm",
            source: include_str!("../shader/group_norm.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_group_norm",
            source: include_str!("../shader/back_group_norm.wgsl"),
        })
    }

    fn has_weights(&self) -> bool {
        true
    }

    fn get_entrypoint(&self) -> &str {
        "group_norm"
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec![
            "group_norm_back_input",
            "group_norm_back_gamma",
            "group_norm_back_beta",
        ]
    }

    fn get_back_workgroup_counts(&self) -> Vec<u32> {
        vec![
            self.dim_input.length().div_ceil(64),
            self.dim_input.z.div_ceil(64),
            self.dim_input.z.div_ceil(64),
        ]
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
                init: match name.as_str() {
                    "gamma" => BufferInit::Ones,
                    "specs" => BufferInit::SpecsUniform,
                    _ => BufferInit::None,
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
                    2 => BackwardBufferSource::Forward(3),
                    3 => BackwardBufferSource::IncomingGradient,
                    _ => BackwardBufferSource::Allocate,
                },
            })
            .collect()
    }

    fn get_back_grad_input_index(&self) -> Option<usize> {
        Some(4)
    }

    fn get_optimizer_bindings(&self) -> Option<OptimizerBindings> {
        Some(OptimizerBindings {
            weight_count: self.dim_input.z,
            weights_forward_index: 1,
            bias_forward_index: 2,
            grad_weights_backward_index: 5,
            grad_bias_backward_index: 6,
        })
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        if self.num_groups == 0 || self.dim_input.z == 0 || self.dim_input.z % self.num_groups != 0
        {
            return Err(ModelError::InvalidGroupCount {
                channels: self.dim_input.z,
                groups: self.num_groups,
            });
        }
        self.dim_output = self.dim_input;
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
        vec![
            (
                "input".to_string(),
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
                "gamma".to_string(),
                BufferSpec {
                    size: self.channel_param_bytes(),
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
                "beta".to_string(),
                BufferSpec {
                    size: self.channel_param_bytes(),
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
                    size: self.dim_output.bytes_size().max(4),
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
        let read_storage = |size: u32| BufferSpec {
            size: size.max(4),
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        };
        let write_storage = |size: u32| BufferSpec {
            size: size.max(4),
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        };

        vec![
            (
                "fwd_input".to_string(),
                read_storage(self.dim_input.bytes_size()),
            ),
            (
                "gamma".to_string(),
                read_storage(self.channel_param_bytes()),
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
                read_storage(self.dim_output.bytes_size()),
            ),
            (
                "grad_input".to_string(),
                write_storage(self.dim_input.bytes_size()),
            ),
            (
                "grad_gamma".to_string(),
                write_storage(self.channel_param_bytes()),
            ),
            (
                "grad_beta".to_string(),
                write_storage(self.channel_param_bytes()),
            ),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        GroupNormUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = GroupNormUniform {
            num_groups: self.num_groups,
            channels_per_group: self.channels_per_group(),
            spatial_len: self.dim_input.x * self.dim_input.y,
            epsilon: self.epsilon,
            dim_input: self.dim_input,
            dim_output: self.dim_output,
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer.write(&uniform).unwrap();
        buffer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::GroupNormType;
    use crate::model::error::ModelError;
    use crate::model::layer_types::LayerType;
    use crate::model::types::Dim3;

    #[test]
    fn group_norm_preserves_input_shape() {
        let mut layer = GroupNormType::new(Dim3::new((4, 4, 8)), 4);
        let output = layer.set_dim_output().unwrap();
        assert_eq!(output.x, 4);
        assert_eq!(output.y, 4);
        assert_eq!(output.z, 8);
    }

    #[test]
    fn group_norm_requires_divisible_groups() {
        let mut layer = GroupNormType::new(Dim3::new((2, 2, 6)), 4);
        assert!(matches!(
            layer.set_dim_output(),
            Err(ModelError::InvalidGroupCount {
                channels: 6,
                groups: 4
            })
        ));
    }
}
