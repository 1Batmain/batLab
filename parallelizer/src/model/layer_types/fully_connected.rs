use crate::model::error::ModelError;
use crate::model::layer_types::{
    ActivationMethod, BackwardBufferBinding, BackwardBufferSource, BufferInit,
    ForwardBufferBinding, LayerType, OptimizerBindings, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

#[derive(Debug, Clone, Copy)]
pub struct FullyConnectedType {
    pub nb_neurons: u32,
    pub method: ActivationMethod,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct FullyConnectedUniform {
    pub input_len: u32,
    pub nb_neurons: u32,
    pub activation_method: u32,
    pub _padding: u32,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl FullyConnectedType {
    pub fn new(dim_input: Dim3, nb_neurons: u32, method: ActivationMethod) -> Self {
        Self {
            nb_neurons,
            method,
            dim_input,
            dim_output: Dim3::default(),
        }
    }

    fn weight_count(&self) -> u32 {
        self.dim_input.length() * self.nb_neurons
    }
}

impl LayerType for FullyConnectedType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "fully_connected",
            source: include_str!("../shader/fully_connected.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_fully_connected",
            source: include_str!("../shader/back_fully_connected.wgsl"),
        })
    }

    fn get_entrypoint(&self) -> &str {
        "fully_connected"
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec![
            "fully_connected_back_input",
            "fully_connected_back_weights",
            "fully_connected_back_bias",
        ]
    }

    fn get_back_workgroup_counts(&self) -> Vec<u32> {
        vec![
            self.dim_input.length().div_ceil(64),
            self.weight_count().div_ceil(64),
            self.nb_neurons.div_ceil(64),
        ]
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
                    2 => BackwardBufferSource::Forward(4),
                    3 => BackwardBufferSource::Forward(3),
                    4 => BackwardBufferSource::IncomingGradient,
                    _ => BackwardBufferSource::Allocate,
                },
            })
            .collect()
    }

    fn get_back_grad_input_index(&self) -> Option<usize> {
        Some(5)
    }

    fn get_optimizer_bindings(&self) -> Option<OptimizerBindings> {
        Some(OptimizerBindings {
            weight_count: self.weight_count(),
            weights_forward_index: 1,
            bias_forward_index: 2,
            grad_weights_backward_index: 6,
            grad_bias_backward_index: 7,
        })
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        self.dim_output = Dim3::new((1, 1, self.nb_neurons));
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
                    size: (self.weight_count() * 4).max(4),
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
                    size: (self.nb_neurons * 4).max(4),
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
                "pre_activation".to_string(),
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
            ("weights".to_string(), read_storage(self.weight_count() * 4)),
            (
                "pre_activation".to_string(),
                read_storage(self.dim_output.bytes_size()),
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
                "grad_weights".to_string(),
                write_storage(self.weight_count() * 4),
            ),
            ("grad_bias".to_string(), write_storage(self.nb_neurons * 4)),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        FullyConnectedUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = FullyConnectedUniform {
            input_len: self.dim_input.length(),
            nb_neurons: self.nb_neurons,
            activation_method: self.method.shader_index(),
            _padding: 0,
            dim_input: self.dim_input,
            dim_output: self.dim_output,
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&uniform)
            .expect("failed to encode fully connected uniform");
        buffer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::FullyConnectedType;
    use crate::model::layer_types::{ActivationMethod, LayerType};
    use crate::model::types::Dim3;

    #[test]
    fn fully_connected_outputs_vector_shape() {
        let mut layer = FullyConnectedType::new(Dim3::new((4, 4, 3)), 10, ActivationMethod::Relu);
        assert_eq!(layer.set_dim_output().unwrap().length(), 10);
        assert_eq!(layer.get_dim_output().x, 1);
        assert_eq!(layer.get_dim_output().y, 1);
        assert_eq!(layer.get_dim_output().z, 10);
    }
}
