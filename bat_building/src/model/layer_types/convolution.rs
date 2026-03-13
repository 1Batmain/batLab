use crate::model::error::ModelError;
use crate::model::layer_types::{
    BackwardBufferBinding, BackwardBufferSource, BufferInit, ForwardBufferBinding,
    ForwardBufferSource, LayerType, OptimizerBindings, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3, PaddingMode};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

#[derive(Debug, Default, Clone, Copy)]
pub struct ConvolutionType {
    pub nb_kernel: u32,
    pub dim_kernel: Dim3,
    pub stride: u32,
    pub mode: PaddingMode,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct ConvolutionUniform {
    pub nb_kernel: u32,
    pub stride: u32,
    pub padding_mode: u32, // 0 = Valid, 1 = Same
    pub _padding: u32,
    pub dim_kernel: Dim3,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl ConvolutionType {
    pub fn new(
        dim_input: Dim3,
        nb_kernel: u32,
        dim_kernel: Dim3,
        stride: u32,
        mode: PaddingMode,
    ) -> Self {
        Self {
            nb_kernel,
            dim_kernel,
            stride,
            mode,
            dim_input,
            dim_output: Dim3::default(),
        }
    }

    fn kernel_bytes(&self) -> u32 {
        self.dim_kernel.bytes_size() * self.nb_kernel
    }
}

impl LayerType for ConvolutionType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "convolution",
            source: include_str!("../shader/convolution.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_convolution",
            source: include_str!("../shader/back_convolution.wgsl"),
        })
    }

    fn has_weights(&self) -> bool {
        true
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
            weight_count: (self.dim_kernel.bytes_size() * self.nb_kernel) / 4,
            weights_forward_index: 1,
            bias_forward_index: 2,
            grad_weights_backward_index: 5,
            grad_bias_backward_index: 6,
        })
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        // Three separate sub-passes to avoid write races:
        //   1. grad_input  — one thread per input element
        //   2. grad_weights — one thread per weight element
        //   3. grad_bias    — one thread per kernel
        vec!["conv_back_input", "conv_back_weights", "conv_back_bias"]
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
        if self.stride == 0 {
            return Err(ModelError::InvalidStride {
                stride: self.stride,
            });
        }
        let x = match self.mode {
            PaddingMode::Valid => {
                let delta = self.dim_input.x.checked_sub(self.dim_kernel.x).ok_or(
                    ModelError::KernelLargerThanInput {
                        input: self.dim_input,
                        kernel: self.dim_kernel,
                        mode: self.mode,
                    },
                )?;
                (delta / self.stride) + 1
            }
            PaddingMode::Same => self.dim_input.x.div_ceil(self.stride),
        };
        let y = match self.mode {
            PaddingMode::Valid => {
                let delta = self.dim_input.y.checked_sub(self.dim_kernel.y).ok_or(
                    ModelError::KernelLargerThanInput {
                        input: self.dim_input,
                        kernel: self.dim_kernel,
                        mode: self.mode,
                    },
                )?;
                (delta / self.stride) + 1
            }
            PaddingMode::Same => self.dim_input.y.div_ceil(self.stride),
        };
        let z = self.nb_kernel;
        self.dim_output = Dim3::new((x, y, z));
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
        vec![
            // [0] input — shared with previous layer's output
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
            // [1] weights — initialised with random values
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
            // [2] bias — initialised to zero
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
            // [3] specs uniform
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
            // [4] output
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
        // Backward bind group layout (all three sub-passes share this layout):
        //   [0] fwd_input    — shared from forward[0]
        //   [1] weights      — shared from forward[1]
        //   [2] specs        — shared from forward[3]
        //   [3] grad_output  — incoming gradient (from next layer or loss)
        //   [4] grad_input   — outgoing gradient to previous layer  (NEW)
        //   [5] grad_weights — accumulated weight gradients         (NEW)
        //   [6] grad_bias    — accumulated bias gradients           (NEW)
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
        ConvolutionUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = ConvolutionUniform {
            nb_kernel: self.nb_kernel,
            stride: self.stride,
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
            .expect("failed to encode convolution uniform");
        buffer.into_inner()
    }
}
