//! File purpose: Defines the loss layer type, shapes, and GPU bindings used by the model graph.

use crate::model::error::ModelError;
use crate::model::layer_types::{
    Batch, batched_bytes,
    BufferInit, ForwardBufferBinding, ForwardBufferSource, LayerType, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use serde::{Deserialize, Serialize};
use wgpu::BufferUsages;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossMethod {
    MeanSquared,
}

#[derive(Debug, Clone)]
pub struct LossType {
    pub method: LossMethod,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct LossUniform {
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl LossType {
    pub fn new(method: LossMethod, dim_input: Dim3) -> Self {
        Self {
            method,
            dim_input,
            dim_output: dim_input,
        }
    }
}

impl LayerType for LossType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "loss",
            source: include_str!("../shader/loss.wgsl"),
        }
    }

    fn get_entrypoint(&self) -> &str {
        match self.method {
            LossMethod::MeanSquared => "mean_squared",
        }
    }

    // Loss has no separate backward pass — its forward IS the gradient computation.
    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec![]
    }

    fn get_back_workgroup_counts(&self, _batch: Batch) -> Vec<u32> {
        vec![]
    }

    fn get_dim_input(&self) -> Dim3 {
        self.dim_input
    }

    fn get_dim_output(&self) -> Dim3 {
        self.dim_output
    }

    fn get_forward_buffer_bindings(&self, batch: Batch) -> Vec<ForwardBufferBinding> {
        self.get_buffers_specs(batch)
            .into_iter()
            .map(|(name, spec)| ForwardBufferBinding {
                init: if name == "specs" {
                    BufferInit::SpecsUniform
                } else {
                    BufferInit::None
                },
                source: if name == "model_result" {
                    ForwardBufferSource::PreviousOutput
                } else {
                    ForwardBufferSource::Allocate
                },
                name,
                spec,
            })
            .collect()
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        self.dim_output = self.dim_input;
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
        vec![
            // [0] model_result — shared from the last forward layer's output
            (
                "model_result".to_string(),
                BufferSpec {
                    size: batched_bytes(self.dim_input, batch).max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            // [1] target — CPU writes ground-truth labels here each step
            (
                "target".to_string(),
                BufferSpec {
                    size: batched_bytes(self.dim_input, batch).max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            // [2] loss_terms — per-element squared error, read back on the CPU and averaged there
            (
                "loss_terms".to_string(),
                BufferSpec {
                    size: batched_bytes(self.dim_input, batch).max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            // [3] grad_output — the gradient fed into the first backward layer
            (
                "grad_output".to_string(),
                BufferSpec {
                    size: batched_bytes(self.dim_input, batch).max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            // [4] specs uniform — the ONE thing the loss kernel cannot deduce
            // from its buffers once they carry a batch axis.
            //
            // MSE divides the gradient by the number of elements it averages
            // over: `grad[i] = 2 * (pred - target) / N`. That `N` used to be
            // `arrayLength(&model_result)`, which was the per-sample length
            // because the buffer held exactly one sample. Batched, that
            // expression returns `batch * N` and would quietly divide every
            // gradient by an extra factor of `batch` — a silent learning-rate
            // change, not a crash. `N` therefore comes from the uniform, and
            // does not depend on the batch.
            //
            // Appended LAST on purpose: `model.rs` addresses this layer's
            // buffers positionally (`forward[1]` = target, `forward[2]` =
            // loss_terms, `forward[3]` = grad_output), and so does
            // `diffusion.rs`. Inserting anywhere else would have moved them.
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
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        LossUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = LossUniform {
            dim_input: self.dim_input,
            dim_output: self.dim_output,
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&uniform)
            .expect("failed to encode loss uniform");
        buffer.into_inner()
    }

    fn get_back_buffers_specs(&self, _batch: Batch) -> Vec<(String, BufferSpec)> {
        // Loss has no backward pass — the forward pass IS the gradient computation.
        vec![]
    }
}
