use crate::model::error::ModelError;
use crate::model::layer_types::LayerType;
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
}

impl LayerType for ConvolutionType {
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
        let res = Dim3::new((x, y, z));
        self.dim_output = res;
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
                "weights".to_string(),
                BufferSpec {
                    size: (self.dim_kernel.bytes_size() * self.nb_kernel).max(4) as u32,
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
                    size: (self.nb_kernel * (std::mem::size_of_val(&self.nb_kernel)).max(4) as u32),
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
