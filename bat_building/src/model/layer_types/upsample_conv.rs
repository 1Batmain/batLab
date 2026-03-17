//! File purpose: Defines the upsample conv layer type, shapes, and GPU bindings used by the model graph.

use crate::model::error::ModelError;
use crate::model::layer_types::{
    BackwardBufferBinding, BackwardBufferSource, BufferInit, ForwardBufferBinding,
    ForwardBufferSource, LayerType, OptimizerBindings, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3, PaddingMode};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

#[derive(Debug, Clone, Copy)]
pub struct UpsampleConvType {
    pub scale_factor: u32,
    pub nb_kernel: u32,
    pub dim_kernel: Dim3,
    pub mode: PaddingMode,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct UpsampleConvUniform {
    pub nb_kernel: u32,
    pub scale_factor: u32,
    pub padding_mode: u32, // 0 = Valid, 1 = Same
    pub _padding: u32,
    pub dim_kernel: Dim3,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl UpsampleConvType {
    pub fn new(
        dim_input: Dim3,
        scale_factor: u32,
        nb_kernel: u32,
        dim_kernel: Dim3,
        mode: PaddingMode,
    ) -> Self {
        Self {
            scale_factor,
            nb_kernel,
            dim_kernel,
            mode,
            dim_input,
            dim_output: Dim3::default(),
        }
    }

    fn upsampled_dims(&self) -> Dim3 {
        Dim3::new((
            self.dim_input.x * self.scale_factor,
            self.dim_input.y * self.scale_factor,
            self.dim_input.z,
        ))
    }

    fn kernel_bytes(&self) -> u32 {
        self.dim_kernel.bytes_size() * self.nb_kernel
    }
}

impl LayerType for UpsampleConvType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "upsample_conv",
            source: include_str!("../shader/upsample_conv.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_upsample_conv",
            source: include_str!("../shader/back_upsample_conv.wgsl"),
        })
    }

    fn has_weights(&self) -> bool {
        true
    }

    fn get_entrypoint(&self) -> &str {
        "upsample_conv"
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
                    "weights" => BufferInit::RandomWeights,
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
            weight_count: self.kernel_bytes() / 4,
            weights_forward_index: 1,
            bias_forward_index: 2,
            grad_weights_backward_index: 5,
            grad_bias_backward_index: 6,
        })
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec![
            "upsample_conv_back_input",
            "upsample_conv_back_weights",
            "upsample_conv_back_bias",
        ]
    }

    fn get_back_workgroup_counts(&self) -> Vec<u32> {
        vec![
            self.dim_input.length().div_ceil(64),
            (self.dim_kernel.length() * self.nb_kernel).div_ceil(64),
            self.nb_kernel.div_ceil(64),
        ]
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        if self.scale_factor == 0 {
            return Err(ModelError::InvalidScaleFactor {
                scale_factor: self.scale_factor,
            });
        }
        let upsampled = self.upsampled_dims();
        let x = match self.mode {
            PaddingMode::Valid => {
                let delta = upsampled.x.checked_sub(self.dim_kernel.x).ok_or(
                    ModelError::KernelLargerThanInput {
                        input: upsampled,
                        kernel: self.dim_kernel,
                        mode: self.mode,
                    },
                )?;
                delta + 1
            }
            PaddingMode::Same => upsampled.x,
        };
        let y = match self.mode {
            PaddingMode::Valid => {
                let delta = upsampled.y.checked_sub(self.dim_kernel.y).ok_or(
                    ModelError::KernelLargerThanInput {
                        input: upsampled,
                        kernel: self.dim_kernel,
                        mode: self.mode,
                    },
                )?;
                delta + 1
            }
            PaddingMode::Same => upsampled.y,
        };
        self.dim_output = Dim3::new((x, y, self.nb_kernel));
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
                "weights".to_string(),
                BufferSpec {
                    size: self.kernel_bytes().max(4),
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
                "bias".to_string(),
                BufferSpec {
                    size: (self.nb_kernel * 4).max(4),
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
            ("weights".to_string(), read_storage(self.kernel_bytes())),
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
                "grad_weights".to_string(),
                write_storage(self.kernel_bytes()),
            ),
            ("grad_bias".to_string(), write_storage(self.nb_kernel * 4)),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        UpsampleConvUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = UpsampleConvUniform {
            nb_kernel: self.nb_kernel,
            scale_factor: self.scale_factor,
            padding_mode: match self.mode {
                PaddingMode::Valid => 0,
                PaddingMode::Same => 1,
            },
            _padding: 0,
            dim_kernel: self.dim_kernel,
            dim_input: self.dim_input,
            dim_output: self.dim_output,
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&uniform)
            .expect("failed to encode upsample conv uniform");
        buffer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::UpsampleConvType;
    use crate::model::layer_types::LayerType;
    use crate::model::types::{Dim3, PaddingMode};

    #[test]
    fn upsample_conv_scales_then_convolves() {
        let mut layer = UpsampleConvType::new(
            Dim3::new((4, 4, 3)),
            2,
            8,
            Dim3::new((3, 3, 3)),
            PaddingMode::Same,
        );
        let output = layer.set_dim_output().unwrap();
        assert_eq!((output.x, output.y, output.z), (8, 8, 8));
    }

    #[test]
    fn upsample_conv_uses_expected_shader_entrypoints() {
        let layer = UpsampleConvType::new(
            Dim3::new((4, 4, 3)),
            2,
            8,
            Dim3::new((3, 3, 3)),
            PaddingMode::Same,
        );
        assert_eq!(layer.get_entrypoint(), "upsample_conv");
        assert_eq!(
            layer.get_back_entrypoints(),
            vec![
                "upsample_conv_back_input",
                "upsample_conv_back_weights",
                "upsample_conv_back_bias"
            ]
        );
    }
}
