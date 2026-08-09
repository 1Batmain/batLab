//! File purpose: Module entry point for layer types; wires submodules and shared exports.

use crate::model::error::ModelError;
use crate::model::types::{BufferSpec, Dim3};
use enum_dispatch::enum_dispatch;

mod activation;
mod attention;
mod concat;
mod convolution;
mod fully_connected;
mod group_norm;
mod loss;
mod pooling;
mod upsample_conv;

pub use activation::{ActivationMethod, ActivationType};
pub use attention::AttentionType;
pub use concat::ConcatType;
pub use convolution::ConvolutionType;
pub use fully_connected::FullyConnectedType;
pub use group_norm::GroupNormType;
pub use loss::{LossMethod, LossType};
pub use pooling::PoolingType;
pub use upsample_conv::UpsampleConvType;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ShaderDescriptor {
    pub label: &'static str,
    pub source: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BufferInit {
    None,
    RandomWeights,
    /// `RandomWeights`, except the last `n` floats are left at ZERO.
    ///
    /// Attention's four projections share one buffer, and the output one
    /// (`W_o`, the tail) must start at zero: the layer is then the exact
    /// identity at step 0, so inserting it into a working network can only be
    /// neutral or better. Drawing it randomly instead injects noise into a
    /// residual stream that has no reason to want it.
    RandomWeightsZeroTail(u32),
    Ones,
    SpecsUniform,
}

#[derive(Debug, Clone)]
pub(crate) enum ForwardBufferSource {
    PreviousOutput,
    SavedOutput(String),
    Allocate,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BackwardBufferSource {
    Forward(usize),
    IncomingGradient,
    Allocate,
}

#[derive(Debug, Clone)]
pub(crate) struct ForwardBufferBinding {
    pub name: String,
    pub spec: BufferSpec,
    pub init: BufferInit,
    pub source: ForwardBufferSource,
}

#[derive(Debug, Clone)]
pub(crate) struct BackwardBufferBinding {
    pub name: String,
    pub spec: BufferSpec,
    pub source: BackwardBufferSource,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OptimizerBindings {
    pub weight_count: u32,
    pub weights_forward_index: usize,
    pub bias_forward_index: usize,
    pub grad_weights_backward_index: usize,
    pub grad_bias_backward_index: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SavedGradientRoute {
    pub key: String,
    pub buffer_index: usize,
}

/// How many samples the graph carries at once.
///
/// The batch axis is *implicit* everywhere: it is carried by the size of the
/// per-sample buffers and by the dispatch grid, never by a uniform field (see
/// `docs/reports/BATCH_DISPATCH_DESIGN.md` §2). A kernel recovers its sample
/// index as `global_index / per_sample_length`, and the per-sample length is
/// already in its uniform. Consequences worth spelling out:
///
/// - the uniform of every layer keeps its exact size and offsets, so the
///   legacy fixtures of `conv_equivalence_tests` / `group_norm_equivalence_tests`
///   still bind the very same buffer;
/// - there is a single source of truth for the batch — the buffer size — so a
///   uniform can never disagree with what was actually allocated.
///
/// A layer type receives the batch in the four methods that size buffers and
/// count workgroups, and applies it *itself* to the bindings that carry a batch
/// axis. That inventory (activations yes, parameters and their gradients no) is
/// the substance of the change and belongs with the layer that owns it.
pub(crate) type Batch = u32;

/// Bytes for a per-sample tensor laid out `batch` times back to back.
///
/// Every call site of this function is, by construction, a buffer that carries
/// a batch axis. The buffers that do *not* — weights, biases, γ/β, their
/// gradients, the optimiser state and every `specs` uniform — size themselves
/// without it. Grepping for this name gives the inventory of §2 of the design
/// note, and the absence of it on a gradient accumulator is the statement that
/// the accumulator is a *reduction* over the batch, not a slice of it.
pub(crate) fn batched_bytes(dim: Dim3, batch: Batch) -> u32 {
    (dim.bytes_size() * batch).max(4)
}

#[enum_dispatch]
pub(crate) trait LayerType: std::fmt::Debug + Send + Sync {
    fn get_forward_shader(&self) -> ShaderDescriptor;
    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        None
    }
    fn get_entrypoint(&self) -> &str {
        "main"
    }
    /// Entry points of the forward pass, in dispatch order.
    ///
    /// Almost every layer is a single kernel and inherits this default. A layer
    /// overrides it when its forward needs a GLOBAL barrier partway through —
    /// attention does, because a score row reads every other row's k and v, and
    /// the barrier wgpu inserts between compute passes of one encoder is the
    /// only synchronisation available at that scale.
    fn get_forward_entrypoints(&self) -> Vec<&'static str> {
        vec![]
    }
    /// Workgroup counts for each forward sub-pass, same split as
    /// `get_back_workgroup_counts`: a pass that maps a thread to an activation
    /// scales with the batch, a pass that maps one to a parameter does not.
    /// Ignored unless `get_forward_entrypoints` is overridden.
    fn get_forward_workgroup_counts(&self, batch: Batch) -> Vec<u32> {
        vec![self.get_forward_workgroup_count(batch)]
    }
    /// Entry points for the backward compute passes (one per sub-pass).
    /// Empty for layers that have no backward pass (e.g. Loss).
    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        vec![]
    }
    /// Workgroup counts for each backward sub-pass (must match get_back_entrypoints length).
    ///
    /// `batch` is applied per sub-pass, and the split is the whole point: a
    /// pass that maps one thread to one *activation* scales with the batch,
    /// a pass that reduces onto a *parameter* does not — it keeps its
    /// workgroup count and folds the batch into its own reduction loop.
    fn get_back_workgroup_counts(&self, batch: Batch) -> Vec<u32> {
        let _ = batch;
        vec![]
    }
    /// Whether this layer has trainable weight buffers.
    fn has_weights(&self) -> bool {
        false
    }
    /// Workgroup count for the forward dispatch. The default assumes the
    /// elementwise convention shared by most shaders: one thread per output
    /// element, `@workgroup_size(64)`. Layers whose forward kernel maps a
    /// workgroup to something else (e.g. GroupNorm: one workgroup per group,
    /// cooperating on a reduction) override this.
    fn get_forward_workgroup_count(&self, batch: Batch) -> u32 {
        (self.get_dim_output().length() * batch).div_ceil(64)
    }
    fn get_dim_input(&self) -> Dim3;
    fn get_dim_output(&self) -> Dim3;
    fn get_forward_buffer_bindings(&self, batch: Batch) -> Vec<ForwardBufferBinding>;
    fn get_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
        self.get_forward_buffer_bindings(batch)
            .into_iter()
            .map(|binding| (binding.name, binding.spec))
            .collect()
    }
    fn get_back_buffer_bindings(&self, batch: Batch) -> Vec<BackwardBufferBinding> {
        let _ = batch;
        vec![]
    }
    /// Specs for ALL bindings in the backward bind group (shared forward + new buffers).
    fn get_back_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
        self.get_back_buffer_bindings(batch)
            .into_iter()
            .map(|binding| (binding.name, binding.spec))
            .collect()
    }
    fn get_back_grad_input_index(&self) -> Option<usize> {
        None
    }
    fn get_saved_gradient_routes(&self) -> Vec<SavedGradientRoute> {
        vec![]
    }
    fn get_optimizer_bindings(&self) -> Option<OptimizerBindings> {
        None
    }
    /// Number of input activations each output unit sums over — the `fan_in` of
    /// He initialisation (`std = sqrt(2 / fan_in)`).
    ///
    /// `None` means the layer has no fan-in-sensitive weight matrix (GroupNorm's
    /// γ/β are scale parameters, not a projection), and its initialisation is
    /// left alone whatever the chosen scheme.
    fn get_weight_fan_in(&self) -> Option<u32> {
        None
    }
    fn set_dim_input(&mut self, input: Dim3);
    fn set_dim_output(&mut self) -> Result<Dim3, ModelError>;
    fn get_spec_uniform_bytes_size(&self) -> u32;
    fn get_spec_uniform_bytes(&self) -> Vec<u8>;
}

#[enum_dispatch(LayerType)]
#[derive(Debug, Clone)]
pub enum LayerTypes {
    Convolution(ConvolutionType),
    Activation(ActivationType),
    Attention(AttentionType),
    Concat(ConcatType),
    FullyConnected(FullyConnectedType),
    GroupNorm(GroupNormType),
    UpsampleConv(UpsampleConvType),
    Loss(LossType),
}

impl LayerTypes {
    /// The variant's name, for humans reading a profile or a debug dump.
    ///
    /// Written as a match rather than derived from `Debug`: `Debug` on these
    /// variants prints the whole type, and a table column has to stay a word.
    pub fn variant_name(&self) -> &'static str {
        match self {
            LayerTypes::Convolution(_) => "Convolution",
            LayerTypes::Activation(_) => "Activation",
            LayerTypes::Attention(_) => "Attention",
            LayerTypes::Concat(_) => "Concat",
            LayerTypes::FullyConnected(_) => "FullyConnected",
            LayerTypes::GroupNorm(_) => "GroupNorm",
            LayerTypes::UpsampleConv(_) => "UpsampleConv",
            LayerTypes::Loss(_) => "Loss",
        }
    }
}
