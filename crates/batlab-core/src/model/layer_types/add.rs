//! File purpose: the residual `Add` layer — the SHORT-RANGE skip `out = f(x) + x`
//! that a residual block needs, as opposed to the long-range `Concat` skip of a
//! U-Net (encoder→decoder, which *concatenates* channels).
//!
//! It reuses [`super::concat`]'s hard part verbatim: the saved-tensor plumbing
//! (a `skip_input` sourced from a marked output) and the gradient fork that
//! routes one incoming gradient back to two producers. What it adds is only the
//! operation — `a + b` forward, and the *trivial* backward where the gradient
//! flows to both inputs unchanged.
//!
//! The one real constraint, and the one real trap: addition **requires
//! identical shapes**, channels included, where concatenation accepts anything.
//! A block that changes width (48→96) must put a 1×1 convolution on the shortcut
//! to realign it; this layer does not do that silently. A shape mismatch is
//! **refused at build** by [`AddType::set_dim_output`], naming both sides — the
//! same discipline `UpsampleConv` learned the hard way when it indexed weights
//! with `dim_input.z` and corrupted itself without a word.

use crate::model::error::ModelError;
use crate::model::layer_types::{
    Batch, BackwardBufferBinding, BackwardBufferSource, BufferInit, ForwardBufferBinding,
    ForwardBufferSource, LayerType, SavedGradientRoute, ShaderDescriptor, batched_bytes,
};
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

#[derive(Debug, Clone)]
pub struct AddType {
    pub skip_key: String,
    pub dim_input: Dim3,
    pub dim_skip: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct AddUniform {
    /// The shared shape of both inputs and the output. `Add` has only one, by
    /// construction — that is the whole point of the layer.
    pub dim: Dim3,
}

impl AddType {
    pub fn new(skip_key: impl Into<String>, dim_input: Dim3, dim_skip: Dim3) -> Self {
        Self {
            skip_key: skip_key.into(),
            dim_input,
            dim_skip,
            dim_output: Dim3::default(),
        }
    }
}

impl LayerType for AddType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "add",
            source: include_str!("../shader/add.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_add",
            source: include_str!("../shader/back_add.wgsl"),
        })
    }

    fn get_entrypoint(&self) -> &str {
        "add"
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        // ONE pass writes both gradients. It is not just tidier that they are the
        // same value: a two-pass backward (one writing grad_input, one writing
        // grad_skip through the *same* backward bind group, which binds both
        // read-write) left the downstream merge that reads grad_skip without a
        // barrier — the conv that produced the skip then computed its weight
        // gradient from grad_skip = 0, intermittently and build-dependently. A
        // residual puts the skip's producer only a couple of passes from the
        // Add, so the hazard actually fires (Concat's U-Net skips are always far
        // enough that it never showed). One dispatch, one write of grad_skip, one
        // clean hazard to the merge. Pinned by
        // `the_backward_feeds_the_gradient_to_the_skip_producer`.
        vec!["add_backward"]
    }

    fn get_back_workgroup_counts(&self, batch: Batch) -> Vec<u32> {
        // One elementwise pass over the shared length.
        vec![(self.dim_output.length() * batch).div_ceil(64)]
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

    fn get_back_buffer_bindings(&self, batch: Batch) -> Vec<BackwardBufferBinding> {
        self.get_back_buffers_specs(batch)
            .into_iter()
            .enumerate()
            .map(|(index, (name, spec))| BackwardBufferBinding {
                name,
                spec,
                source: match index {
                    // grad_output is the incoming gradient; the specs uniform is
                    // the very buffer the forward pass filled (its index 2), so
                    // the shape is written once and shared.
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
        // Addition cannot widen or reshape: every axis must match, channels
        // included. Named on both sides so a mis-wired block says which layer
        // and which two widths, instead of indexing past a buffer.
        if self.dim_input.x != self.dim_skip.x
            || self.dim_input.y != self.dim_skip.y
            || self.dim_input.z != self.dim_skip.z
        {
            return Err(ModelError::ResidualDimMismatch {
                input: self.dim_input,
                skip: self.dim_skip,
            });
        }
        self.dim_output = self.dim_input;
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
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
                read_storage(batched_bytes(self.dim_input, batch)),
            ),
            (
                "skip_input".to_string(),
                read_storage(batched_bytes(self.dim_skip, batch)),
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
                    size: batched_bytes(self.dim_output, batch).max(4),
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

    fn get_back_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
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
                storage(batched_bytes(self.dim_output, batch), true),
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
                storage(batched_bytes(self.dim_input, batch), false),
            ),
            (
                "grad_skip".to_string(),
                storage(batched_bytes(self.dim_skip, batch), false),
            ),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        AddUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = AddUniform {
            dim: self.dim_output,
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&uniform)
            .expect("failed to encode add uniform");
        buffer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::AddType;
    use crate::model::error::ModelError;
    use crate::model::layer_types::LayerType;
    use crate::model::types::Dim3;

    #[test]
    fn add_preserves_shape_when_both_sides_match() {
        let mut layer = AddType::new("skip", Dim3::new((8, 8, 16)), Dim3::new((8, 8, 16)));
        let output = layer.set_dim_output().unwrap();
        assert_eq!((output.x, output.y, output.z), (8, 8, 16));
    }

    #[test]
    fn a_channel_mismatch_is_refused_by_name() {
        let mut layer = AddType::new("skip", Dim3::new((8, 8, 48)), Dim3::new((8, 8, 96)));
        match layer.set_dim_output() {
            Err(ModelError::ResidualDimMismatch { input, skip }) => {
                assert_eq!(input.z, 48);
                assert_eq!(skip.z, 96);
            }
            other => panic!("expected ResidualDimMismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_spatial_mismatch_is_refused_too() {
        let mut layer = AddType::new("skip", Dim3::new((8, 8, 16)), Dim3::new((16, 16, 16)));
        assert!(matches!(
            layer.set_dim_output(),
            Err(ModelError::ResidualDimMismatch { .. })
        ));
    }
}
