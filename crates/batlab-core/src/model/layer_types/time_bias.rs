//! File purpose: the `TimeBias` layer — a learned per-channel bias derived from
//! the timestep embedding and added to every spatial position of its input.
//!
//! This is the missing half of timestep conditioning. Today `t` enters only at
//! the input, as a few embedding channels concatenated to the image, and the
//! deep layers see it only through whatever survives the convolutions. Standard
//! DDPM instead injects `t` into every block: a small learned projection of the
//! timestep embedding produces a per-channel bias that is added to the feature
//! map. That is exactly this layer.
//!
//! `output[b, h, w, c] = input[b, h, w, c] + bias[c] + Σ_n emb[b, n] · W[n, c]`
//!
//! where `emb[b]` is the timestep embedding of sample `b`, read from a SAVED
//! copy of the model input (the embedding is spatially constant, so any pixel
//! carries it — this layer reads pixel 0 of the referenced tensor). The
//! reference works at any resolution: the layer only reads the embedding vector,
//! never the spatial content, so a block at 8×8 can be conditioned from a 32×32
//! saved input just as well.
//!
//! It starts as the identity: `W` and `bias` are zero-initialised, so inserting
//! a `TimeBias` into a working network is neutral at step 0 and can only earn
//! its keep from there (the same discipline as attention's zero-init output
//! projection). No gradient flows back to the embedding — it is an untrained
//! model input — so this layer needs no saved-gradient route, unlike Concat/Add.

use crate::model::error::ModelError;
use crate::model::layer_types::{
    Batch, BackwardBufferBinding, BackwardBufferSource, BufferInit, ForwardBufferBinding,
    ForwardBufferSource, LayerType, OptimizerBindings, ShaderDescriptor, batched_bytes,
};
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

#[derive(Debug, Clone)]
pub struct TimeBiasType {
    /// Saved tensor the embedding is read from (a copy of the model input).
    pub time_key: String,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
    /// First embedding channel inside the referenced tensor (= signal channels).
    pub embed_offset: u32,
    /// How many embedding channels to project.
    pub embed_channels: u32,
    /// Per-sample length of the referenced tensor (`Ht*Wt*Zt`), resolved from
    /// its geometry at build time so the shader can find sample `b`'s embedding.
    pub time_sample_len: u32,
}

#[derive(ShaderType, Clone, Copy)]
pub struct TimeBiasUniform {
    /// H, W, C of this layer.
    pub dim: Dim3,
    /// (embed_channels, embed_offset, time_sample_len), packed as a second vec3
    /// so the uniform is two `Dim3`s — the alignment encase and WGSL agree on
    /// (a bare `vec3` followed by scalars does not round-trip cleanly).
    pub params: Dim3,
}

impl TimeBiasType {
    pub fn new(
        time_key: impl Into<String>,
        dim_input: Dim3,
        embed_offset: u32,
        embed_channels: u32,
        time_sample_len: u32,
    ) -> Self {
        Self {
            time_key: time_key.into(),
            dim_input,
            dim_output: dim_input,
            embed_offset,
            embed_channels,
            time_sample_len,
        }
    }

    fn channels(&self) -> u32 {
        self.dim_input.z
    }

    /// `W` has `embed_channels × C` entries; `bias` has `C`.
    fn weight_count(&self) -> u32 {
        (self.embed_channels * self.channels()).max(1)
    }

    fn weight_bytes(&self) -> u32 {
        (self.weight_count() * 4).max(4)
    }

    fn bias_bytes(&self) -> u32 {
        (self.channels() * 4).max(4)
    }
}

impl LayerType for TimeBiasType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "time_bias",
            source: include_str!("../shader/time_bias.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_time_bias",
            source: include_str!("../shader/back_time_bias.wgsl"),
        })
    }

    fn has_weights(&self) -> bool {
        true
    }

    fn get_entrypoint(&self) -> &str {
        "time_bias"
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        // The GroupNorm shape, which is the one that survives on Metal: a separate
        // elementwise grad_input pass, and a per-channel parameter pass — ONE
        // WORKGROUP PER CHANNEL — that accumulates onto the framework-cleared
        // buffers. See back_time_bias.wgsl for why folding them, or overwriting
        // instead of accumulating, both came back zero build-dependently.
        vec!["time_bias_back_input", "time_bias_grad_params"]
    }

    fn get_back_workgroup_counts(&self, batch: Batch) -> Vec<u32> {
        vec![
            // grad_input is elementwise over the activation.
            (self.dim_input.length() * batch).div_ceil(64),
            // grad_params: one workgroup per output channel (like GroupNorm's
            // grad_gamma), which sweeps the batch×spatial extent itself.
            self.channels(),
        ]
    }

    fn get_weight_fan_in(&self) -> Option<u32> {
        // Zero-initialised on purpose; leave the He path alone.
        None
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
                    "time" => ForwardBufferSource::SavedOutput(self.time_key.clone()),
                    _ => ForwardBufferSource::Allocate,
                };
                // W and bias start at zero (BufferInit::None → wgpu zero-clears),
                // so the layer opens as the identity.
                let init = if name == "specs" {
                    BufferInit::SpecsUniform
                } else {
                    BufferInit::None
                };
                ForwardBufferBinding {
                    init,
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
                    // Reuse the forward time tensor (its buffer 1) and the
                    // forward specs (buffer 4); take the incoming gradient.
                    0 => BackwardBufferSource::Forward(1),
                    1 => BackwardBufferSource::Forward(4),
                    2 => BackwardBufferSource::IncomingGradient,
                    _ => BackwardBufferSource::Allocate,
                },
            })
            .collect()
    }

    fn get_back_grad_input_index(&self) -> Option<usize> {
        Some(3)
    }

    fn get_optimizer_bindings(&self) -> Option<OptimizerBindings> {
        Some(OptimizerBindings {
            weight_count: self.weight_count(),
            weights_forward_index: 2,
            bias_forward_index: 3,
            grad_weights_backward_index: 4,
            grad_bias_backward_index: 5,
        })
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        if self.embed_channels == 0 {
            return Err(ModelError::TimeBiasNoEmbedding {
                time_key: self.time_key.clone(),
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
                "time".to_string(),
                read_storage(batched_bytes(
                    Dim3::new((self.time_sample_len.max(1), 1, 1)),
                    batch,
                )),
            ),
            ("weights".to_string(), read_storage(self.weight_bytes())),
            ("bias".to_string(), read_storage(self.bias_bytes())),
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
                "time".to_string(),
                read_storage(batched_bytes(
                    Dim3::new((self.time_sample_len.max(1), 1, 1)),
                    batch,
                )),
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
                read_storage(batched_bytes(self.dim_output, batch)),
            ),
            (
                "grad_input".to_string(),
                write_storage(batched_bytes(self.dim_input, batch)),
            ),
            ("grad_weights".to_string(), write_storage(self.weight_bytes())),
            ("grad_bias".to_string(), write_storage(self.bias_bytes())),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        TimeBiasUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = TimeBiasUniform {
            dim: self.dim_output,
            params: Dim3::new((
                self.embed_channels,
                self.embed_offset,
                self.time_sample_len,
            )),
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&uniform)
            .expect("failed to encode time_bias uniform");
        buffer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::TimeBiasType;
    use crate::model::error::ModelError;
    use crate::model::layer_types::LayerType;
    use crate::model::types::Dim3;

    #[test]
    fn time_bias_preserves_shape() {
        let mut layer = TimeBiasType::new("tin", Dim3::new((8, 8, 16)), 1, 3, 32 * 32 * 4);
        let out = layer.set_dim_output().unwrap();
        assert_eq!((out.x, out.y, out.z), (8, 8, 16));
    }

    #[test]
    fn a_projection_of_no_channels_is_refused() {
        let mut layer = TimeBiasType::new("tin", Dim3::new((8, 8, 16)), 1, 0, 4096);
        assert!(matches!(
            layer.set_dim_output(),
            Err(ModelError::TimeBiasNoEmbedding { .. })
        ));
    }

    #[test]
    fn the_weight_matrix_is_embedding_by_channels() {
        let layer = TimeBiasType::new("tin", Dim3::new((8, 8, 16)), 1, 3, 4096);
        assert_eq!(layer.weight_count(), 3 * 16);
    }
}
