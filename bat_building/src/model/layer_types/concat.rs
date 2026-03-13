use crate::model::error::ModelError;
use crate::model::layer_types::{
    BackwardBufferBinding, BackwardBufferSource, BufferInit, ForwardBufferBinding,
    ForwardBufferSource, LayerType, SavedGradientRoute, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

#[derive(Debug, Clone)]
pub struct ConcatType {
    pub skip_key: String,
    pub dim_input: Dim3,
    pub dim_skip: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct ConcatUniform {
    pub dim_input: Dim3,
    pub dim_skip: Dim3,
    pub dim_output: Dim3,
}

impl ConcatType {
    pub fn new(skip_key: impl Into<String>, dim_input: Dim3, dim_skip: Dim3) -> Self {
        Self {
            skip_key: skip_key.into(),
            dim_input,
            dim_skip,
            dim_output: Dim3::default(),
        }
    }
}

impl LayerType for ConcatType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "concat",
            source: include_str!("../shader/concat.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_concat",
            source: include_str!("../shader/back_concat.wgsl"),
        })
    }

    fn get_entrypoint(&self) -> &str {
        "concat"
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec!["concat_back_input", "concat_back_skip"]
    }

    fn get_back_workgroup_counts(&self) -> Vec<u32> {
        vec![
            self.dim_input.length().div_ceil(64),
            self.dim_skip.length().div_ceil(64),
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
            .map(|(name, spec)| {
                let source = match name.as_str() {
                    "input" => ForwardBufferSource::PreviousOutput,
                    "skip_input" => ForwardBufferSource::SavedOutput(self.skip_key.clone()),
                    _ => ForwardBufferSource::Allocate,
                };
                ForwardBufferBinding {
                    init: if name == "specs" {
                        BufferInit::SpecsUniform
                    } else {
                        BufferInit::None
                    },
                    name,
                    spec,
                    source,
                }
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
                    0 => BackwardBufferSource::IncomingGradient,
                    1 => BackwardBufferSource::Forward(2),
                    _ => BackwardBufferSource::Allocate,
                },
            })
            .collect()
    }

    fn get_back_grad_input_index(&self) -> Option<usize> {
        Some(2)
    }

    fn get_saved_gradient_routes(&self) -> Vec<SavedGradientRoute> {
        vec![SavedGradientRoute {
            key: self.skip_key.clone(),
            buffer_index: 3,
        }]
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        if self.dim_input.x != self.dim_skip.x || self.dim_input.y != self.dim_skip.y {
            return Err(ModelError::ConcatSpatialMismatch {
                input: self.dim_input,
                skip: self.dim_skip,
            });
        }
        self.dim_output = Dim3::new((
            self.dim_input.x,
            self.dim_input.y,
            self.dim_input.z + self.dim_skip.z,
        ));
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
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
        vec![
            (
                "input".to_string(),
                read_storage(self.dim_input.bytes_size()),
            ),
            (
                "skip_input".to_string(),
                read_storage(self.dim_skip.bytes_size()),
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
        let storage = |size: u32, read_only: bool| BufferSpec {
            size: size.max(4),
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        };
        vec![
            (
                "grad_output".to_string(),
                storage(self.dim_output.bytes_size(), true),
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
                "grad_input".to_string(),
                storage(self.dim_input.bytes_size(), false),
            ),
            (
                "grad_skip".to_string(),
                storage(self.dim_skip.bytes_size(), false),
            ),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        ConcatUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = ConcatUniform {
            dim_input: self.dim_input,
            dim_skip: self.dim_skip,
            dim_output: self.dim_output,
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&uniform)
            .expect("failed to encode concat uniform");
        buffer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::ConcatType;
    use crate::model::layer_types::LayerType;
    use crate::model::types::Dim3;

    #[test]
    fn concat_adds_channels() {
        let mut layer = ConcatType::new("skip", Dim3::new((8, 8, 16)), Dim3::new((8, 8, 8)));
        let output = layer.set_dim_output().unwrap();
        assert_eq!((output.x, output.y, output.z), (8, 8, 24));
    }
}
