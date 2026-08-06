//! File purpose: Defines the spatial self-attention layer type, shapes, and GPU bindings used by the model graph.

use crate::model::error::ModelError;
use crate::model::layer_types::{
    Batch, batched_bytes, BackwardBufferBinding, BackwardBufferSource, BufferInit,
    ForwardBufferBinding, ForwardBufferSource, LayerType, OptimizerBindings, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

/// `@workgroup_size(64)` in every attention shader.
const WG_SIZE: u32 = 64;

/// The four projections packed back to back in ONE weight buffer: Q, K, V, O.
const PROJECTIONS: u32 = 4;

/// Single-head spatial self-attention with the residual inside the layer.
///
/// `y = x + W_o · softmax(qᵀk / √d) · v`, the sequence being the `H*W` spatial
/// positions and the feature dimension the channel count. Shape-preserving, so
/// it drops into the chain anywhere without disturbing the dims around it.
///
/// # Why the four matrices share one buffer
///
/// `OptimizerBindings` names exactly one weight tensor and one bias tensor per
/// layer, and Adam's moments, the SGD update, the checkpoint reader and the
/// gradient zeroing are all built on that. Packing Q|K|V|O into a single
/// `4·C·C` buffer (and their biases into a single `4·C`) means every one of
/// those paths treats attention exactly like a convolution — no new optimiser
/// plumbing, no fifth code path to keep in step. The shaders address a
/// projection by its `which * C * C` offset; that arithmetic is the entire cost
/// of the choice.
///
/// # Cost, and why this belongs at a bottleneck
///
/// The `probs` scratch is `N²` per sample: 64 positions at the 8×8 bottleneck
/// is 4 096 floats, but the same layer at 32×32 would be 1 048 576 — 256× more
/// for one layer. Attention is affordable where the spatial grid is small.
#[derive(Debug, Clone, Copy)]
pub struct AttentionType {
    /// Whether the four projections carry biases.
    pub use_bias: bool,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct AttentionUniform {
    pub seq_len: u32,
    pub channels: u32,
    pub scale: f32,
    pub use_bias: u32,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl AttentionType {
    pub fn new(dim_input: Dim3) -> Self {
        Self {
            use_bias: true,
            dim_input,
            dim_output: dim_input,
        }
    }

    pub fn without_bias(dim_input: Dim3) -> Self {
        Self {
            use_bias: false,
            ..Self::new(dim_input)
        }
    }

    /// Attending positions: the whole spatial grid.
    pub(crate) fn seq_len(&self) -> u32 {
        self.dim_input.x * self.dim_input.y
    }

    fn channels(&self) -> u32 {
        self.dim_input.z
    }

    /// Elements of one sample's activation tensor.
    fn sample_len(&self) -> u32 {
        self.seq_len() * self.channels()
    }

    /// Scalars in the packed weight tensor.
    pub(crate) fn weight_count(&self) -> u32 {
        PROJECTIONS * self.channels() * self.channels()
    }

    /// Scalars in the packed bias tensor.
    fn bias_count(&self) -> u32 {
        PROJECTIONS * self.channels()
    }

    /// Floats of `W_o`, which initialise to zero — see
    /// `BufferInit::RandomWeightsZeroTail`.
    fn output_projection_len(&self) -> u32 {
        self.channels() * self.channels()
    }

    fn activation_bytes(&self, batch: Batch) -> u32 {
        batched_bytes(self.dim_input, batch)
    }

    /// Elements of one sample's scratch: q, k, v and ctx (`N*C` each) followed
    /// by the softmax rows (`N*N`).
    ///
    /// The five live in ONE buffer because WebGPU guarantees only eight storage
    /// buffers per shader stage. Bound separately, the backward pass needed
    /// twelve — and the rejection is silent and total: the bind group layout
    /// fails, so the whole command buffer (forward included) is dropped and the
    /// loss reads 0.000000 with no other symptom. Raising the limit at
    /// `request_device` would have worked on this machine and broken the point
    /// of wgpu, which is the visitor's GPU through a browser.
    fn scratch_stride(&self) -> u32 {
        4 * self.sample_len() + self.seq_len() * self.seq_len()
    }

    fn scratch_bytes(&self, batch: Batch) -> u32 {
        (self.scratch_stride() * batch * 4).max(4)
    }

    fn elementwise_workgroups(&self, batch: Batch) -> u32 {
        (self.sample_len() * batch).div_ceil(WG_SIZE)
    }

    /// One workgroup per (sample, attention row).
    fn row_workgroups(&self, batch: Batch) -> u32 {
        self.seq_len() * batch
    }

    fn read_storage(size: u32) -> BufferSpec {
        BufferSpec {
            size: size.max(4),
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        }
    }

    fn write_storage(size: u32) -> BufferSpec {
        BufferSpec {
            size: size.max(4),
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        }
    }

    fn uniform_spec(&self) -> BufferSpec {
        BufferSpec {
            size: self.get_spec_uniform_bytes_size().max(4),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(
                    std::num::NonZeroU64::new(self.get_spec_uniform_bytes_size() as u64).unwrap(),
                ),
            },
        }
    }
}

impl LayerType for AttentionType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "attention",
            source: include_str!("../shader/attention.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_attention",
            source: include_str!("../shader/back_attention.wgsl"),
        })
    }

    fn has_weights(&self) -> bool {
        true
    }

    fn get_entrypoint(&self) -> &str {
        "attn_qkv"
    }

    /// Four sequential forward passes: attention needs a GLOBAL barrier between
    /// the projections and the scores (row `n` reads every other row's k and v),
    /// and between the scores and the context. The barrier wgpu inserts between
    /// compute passes of one encoder is the only one available at this scale.
    fn get_forward_entrypoints(&self) -> Vec<&'static str> {
        vec!["attn_qkv", "attn_scores", "attn_context", "attn_out"]
    }

    fn get_forward_workgroup_counts(&self, batch: Batch) -> Vec<u32> {
        vec![
            // q, k, v: three projections' worth of scalars.
            (3 * self.sample_len() * batch).div_ceil(WG_SIZE),
            // softmax: one workgroup per row, cooperating on the reduction.
            self.row_workgroups(batch),
            self.elementwise_workgroups(batch),
            self.elementwise_workgroups(batch),
        ]
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec![
            "attn_back_ctx",
            "attn_back_v",
            "attn_back_scores",
            "attn_back_qk",
            "attn_back_input",
            "attn_back_weights",
            "attn_back_bias",
        ]
    }

    fn get_back_workgroup_counts(&self, batch: Batch) -> Vec<u32> {
        vec![
            self.elementwise_workgroups(batch),
            self.elementwise_workgroups(batch),
            self.row_workgroups(batch),
            self.elementwise_workgroups(batch),
            self.elementwise_workgroups(batch),
            // The two parameter reductions: as many sums as there are
            // parameters whatever the batch, which they fold into their loop.
            self.weight_count().div_ceil(WG_SIZE),
            self.bias_count().div_ceil(WG_SIZE),
        ]
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
                init: match name.as_str() {
                    // Q, K and V draw from the usual scheme; W_o starts at ZERO
                    // so the whole layer starts as the identity and the network
                    // can only gain from it.
                    "weights" => BufferInit::RandomWeightsZeroTail(self.output_projection_len()),
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

    fn get_back_buffer_bindings(&self, batch: Batch) -> Vec<BackwardBufferBinding> {
        self.get_back_buffers_specs(batch)
            .into_iter()
            .enumerate()
            .map(|(index, (name, spec))| BackwardBufferBinding {
                name,
                spec,
                source: match index {
                    0 => BackwardBufferSource::Forward(0), // input
                    1 => BackwardBufferSource::Forward(1), // weights
                    2 => BackwardBufferSource::Forward(3), // specs
                    3 => BackwardBufferSource::IncomingGradient,
                    // What the forward saved — q/k/v, the context and the
                    // softmax rows — re-read instead of recomputed.
                    7 => BackwardBufferSource::Forward(4),
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
            weight_count: self.weight_count(),
            weights_forward_index: 1,
            bias_forward_index: 2,
            grad_weights_backward_index: 5,
            grad_bias_backward_index: 6,
        })
    }

    fn get_weight_fan_in(&self) -> Option<u32> {
        // Every projection sums over the C channels of one position.
        Some(self.channels().max(1))
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        if self.seq_len() == 0 || self.channels() == 0 {
            return Err(ModelError::InvalidAttentionShape {
                dim_input: (self.dim_input.x, self.dim_input.y, self.dim_input.z),
            });
        }
        self.dim_output = self.dim_input;
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
        vec![
            (
                "input".to_string(),
                Self::read_storage(self.activation_bytes(batch)),
            ),
            // Q|K|V|O packed: one tensor, so one optimiser binding.
            (
                "weights".to_string(),
                Self::read_storage(self.weight_count() * 4),
            ),
            ("bias".to_string(), Self::read_storage(self.bias_count() * 4)),
            ("specs".to_string(), self.uniform_spec()),
            // Forward scratch — q | k | v | ctx | probs, kept for the backward
            // pass. It carries a batch axis: these are activations, not
            // parameters.
            (
                "scratch".to_string(),
                Self::write_storage(self.scratch_bytes(batch)),
            ),
            // LAST: `create_buffers` chains `.last()` as this layer's output.
            (
                "output".to_string(),
                Self::write_storage(batched_bytes(self.dim_output, batch)),
            ),
        ]
    }

    fn get_back_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
        vec![
            (
                "fwd_input".to_string(),
                Self::read_storage(self.activation_bytes(batch)),
            ),
            (
                "weights".to_string(),
                Self::read_storage(self.weight_count() * 4),
            ),
            ("specs".to_string(), self.uniform_spec()),
            (
                "grad_output".to_string(),
                Self::read_storage(batched_bytes(self.dim_output, batch)),
            ),
            (
                "grad_input".to_string(),
                Self::write_storage(self.activation_bytes(batch)),
            ),
            // Gradients of PARAMETERS: no batch axis. They are a reduction over
            // the batch, not a slice of it.
            (
                "grad_weights".to_string(),
                Self::write_storage(self.weight_count() * 4),
            ),
            (
                "grad_bias".to_string(),
                Self::write_storage(self.bias_count() * 4),
            ),
            // The two twins: what the forward saved, and the backward's own
            // working set at the SAME block offsets.
            (
                "fwd_scratch".to_string(),
                Self::read_storage(self.scratch_bytes(batch)),
            ),
            (
                "grad_scratch".to_string(),
                Self::write_storage(self.scratch_bytes(batch)),
            ),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        AttentionUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = AttentionUniform {
            seq_len: self.seq_len(),
            channels: self.channels(),
            scale: 1.0 / (self.channels().max(1) as f32).sqrt(),
            use_bias: u32::from(self.use_bias),
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
    use super::AttentionType;
    use crate::model::error::ModelError;
    use crate::model::layer_types::LayerType;
    use crate::model::types::{BufferSpec, Dim3};

    #[test]
    fn attention_preserves_input_shape() {
        let mut layer = AttentionType::new(Dim3::new((8, 8, 192)));
        let output = layer.set_dim_output().unwrap();
        assert_eq!((output.x, output.y, output.z), (8, 8, 192));
    }

    #[test]
    fn attention_rejects_an_empty_grid() {
        let mut layer = AttentionType::new(Dim3::new((0, 8, 16)));
        assert!(matches!(
            layer.set_dim_output(),
            Err(ModelError::InvalidAttentionShape { .. })
        ));
    }

    /// The four projections are one tensor as far as the optimiser is
    /// concerned — that is what lets Adam treat this layer like any other.
    #[test]
    fn the_four_projections_are_one_weight_tensor() {
        let layer = AttentionType::new(Dim3::new((8, 8, 32)));
        let bindings = layer.get_optimizer_bindings().unwrap();
        assert_eq!(bindings.weight_count, 4 * 32 * 32);
        let specs = layer.get_buffers_specs(1);
        assert_eq!(specs[1].0, "weights");
        assert_eq!(specs[1].1.size, 4 * 32 * 32 * 4);
        assert_eq!(specs[2].0, "bias");
        assert_eq!(specs[2].1.size, 4 * 32 * 4);
    }

    /// The scratch that carries a batch axis must scale with it, and the
    /// parameters must not.
    #[test]
    fn only_the_activations_carry_the_batch() {
        let layer = AttentionType::new(Dim3::new((4, 4, 8)));
        let one = layer.get_buffers_specs(1);
        let eight = layer.get_buffers_specs(8);
        for (index, name) in [(0, "input"), (4, "scratch"), (5, "output")] {
            assert_eq!(one[index].0, name);
            assert_eq!(
                eight[index].1.size,
                one[index].1.size * 8,
                "{name} must scale with the batch"
            );
        }
        for (index, name) in [(1, "weights"), (2, "bias")] {
            assert_eq!(one[index].0, name);
            assert_eq!(
                eight[index].1.size, one[index].1.size,
                "{name} is a parameter and must not scale with the batch"
            );
        }
    }

    /// WebGPU guarantees only EIGHT storage buffers per shader stage, and this
    /// layer is the first in the repository to come near it.
    ///
    /// This is a guard and not a note because of HOW it fails. The bind group
    /// layout is rejected at build time, which invalidates the whole command
    /// buffer — the forward passes encoded alongside it never run either. The
    /// observable symptom is a loss of exactly 0.000000 and gradients of
    /// exactly zero, with nothing in the way of an exception; it cost this
    /// layer's first backward implementation an hour. Adding a binding here
    /// without merging another must fail in CI, not in a training run.
    #[test]
    fn neither_bind_group_exceeds_the_webgpu_storage_buffer_limit() {
        const WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE: usize = 8;
        let layer = AttentionType::new(Dim3::new((8, 8, 192)));

        let storage_count = |specs: Vec<(String, BufferSpec)>| {
            specs
                .iter()
                .filter(|(_, spec)| {
                    matches!(
                        spec.ty,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { .. },
                            ..
                        }
                    )
                })
                .count()
        };

        let forward = storage_count(layer.get_buffers_specs(1));
        let backward = storage_count(layer.get_back_buffers_specs(1));
        assert!(
            forward <= WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE,
            "forward binds {forward} storage buffers, WebGPU guarantees {WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE}"
        );
        assert!(
            backward <= WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE,
            "backward binds {backward} storage buffers, WebGPU guarantees {WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE}"
        );
    }

    /// The parameter reductions do not scale with the batch; the elementwise
    /// passes do.
    #[test]
    fn backward_parameter_passes_do_not_scale_with_the_batch() {
        let layer = AttentionType::new(Dim3::new((4, 4, 8)));
        let one = layer.get_back_workgroup_counts(1);
        let eight = layer.get_back_workgroup_counts(8);
        assert_eq!(one.len(), layer.get_back_entrypoints().len());
        // grad_weights (5) and grad_bias (6) are per-parameter.
        assert_eq!(one[5], eight[5]);
        assert_eq!(one[6], eight[6]);
        // The elementwise ones scale.
        assert_eq!(eight[0], one[0] * 8);
        assert_eq!(eight[2], one[2] * 8);
    }
}
