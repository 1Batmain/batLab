//! File purpose: Serialisable description of a model and of the run to perform with it — the schema behind every `Models/<name>/config_file`.
//!
//! This is the contract between whoever *describes* a model (the TUI builder, a
//! generator script, tomorrow a web page) and the engine that *builds* it. It is
//! therefore pure data: serde in, serde out, no filesystem, no terminal. The
//! bytes-in/bytes-out entry points below are what a wasm build will use, where
//! the config arrives over the network rather than from `Models/`.

use crate::model::training::{LossWeighting, PerpetualRegime};
use crate::model::{OptimizerKind, WeightInit};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Mirror types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PaddingMode {
    Valid,
    Same,
}

impl fmt::Display for PaddingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingMode::Valid => write!(f, "Valid"),
            PaddingMode::Same => write!(f, "Same"),
        }
    }
}

impl PaddingMode {
    pub fn toggle(&self) -> Self {
        match self {
            PaddingMode::Valid => PaddingMode::Same,
            PaddingMode::Same => PaddingMode::Valid,
        }
    }
}

/// The engine's own padding enum, from the config's mirror of it.
///
/// The two enums exist because the config is serialisable and the engine's is
/// not, but the translation between them was written out by hand in the CLI and
/// nowhere else — so anything else that wanted to turn a `config_file` into a
/// graph (the resource inventory does) had to write it a second time. One
/// conversion, in the crate that owns both types.
impl From<&PaddingMode> for crate::model::types::PaddingMode {
    fn from(mode: &PaddingMode) -> Self {
        match mode {
            PaddingMode::Valid => crate::model::types::PaddingMode::Valid,
            PaddingMode::Same => crate::model::types::PaddingMode::Same,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationMethod {
    Relu,
    Silu,
    Linear,
}

impl From<ActivationMethod> for crate::model::layer_types::ActivationMethod {
    fn from(method: ActivationMethod) -> Self {
        match method {
            ActivationMethod::Relu => crate::model::layer_types::ActivationMethod::Relu,
            ActivationMethod::Silu => crate::model::layer_types::ActivationMethod::Silu,
            ActivationMethod::Linear => crate::model::layer_types::ActivationMethod::Linear,
        }
    }
}

impl fmt::Display for ActivationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActivationMethod::Relu => write!(f, "ReLU"),
            ActivationMethod::Silu => write!(f, "SiLU"),
            ActivationMethod::Linear => write!(f, "Linear"),
        }
    }
}

impl ActivationMethod {
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "ReLU" => Some(ActivationMethod::Relu),
            "SiLU" => Some(ActivationMethod::Silu),
            "Linear" => Some(ActivationMethod::Linear),
            _ => None,
        }
    }

    pub fn toggle(&self) -> Self {
        match self {
            ActivationMethod::Relu => ActivationMethod::Silu,
            ActivationMethod::Silu => ActivationMethod::Linear,
            ActivationMethod::Linear => ActivationMethod::Relu,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LossMethod {
    MeanSquared,
}

impl fmt::Display for LossMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeanSquared")
    }
}

// ---------------------------------------------------------------------------
// LayerDraft — dim_input always stored (set at add-time from inferred chain)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerDraft {
    Convolution {
        dim_input: (u32, u32, u32),
        nb_kernel: u32,
        dim_kernel: (u32, u32, u32),
        stride: u32,
        padding: PaddingMode,
        save_key: Option<String>,
    },
    Activation {
        dim_input: (u32, u32, u32),
        method: ActivationMethod,
        save_key: Option<String>,
    },
    GroupNorm {
        dim_input: (u32, u32, u32),
        num_groups: u32,
        save_key: Option<String>,
    },
    /// Spatial self-attention over the `H*W` positions, residual included.
    /// Shape-preserving, so it carries no dimension of its own beyond its input.
    Attention {
        dim_input: (u32, u32, u32),
        save_key: Option<String>,
    },
    FullyConnected {
        dim_input: (u32, u32, u32),
        nb_neurons: u32,
        method: ActivationMethod,
        save_key: Option<String>,
    },
    UpsampleConv {
        dim_input: (u32, u32, u32),
        scale_factor: u32,
        nb_kernel: u32,
        dim_kernel: (u32, u32, u32),
        padding: PaddingMode,
        save_key: Option<String>,
    },
    Concat {
        dim_input: (u32, u32, u32),
        dim_skip: (u32, u32, u32),
        skip_key: String,
        save_key: Option<String>,
    },
}

pub fn compute_out_conv(
    dim_input: (u32, u32, u32),
    dim_kernel: (u32, u32, u32),
    stride: u32,
    nb_kernel: u32,
    padding: &PaddingMode,
) -> (u32, u32, u32) {
    let (iw, ih, _) = dim_input;
    let (kw, kh, _) = dim_kernel;
    let s = stride.max(1);
    let (ow, oh) = match padding {
        PaddingMode::Valid => (iw.saturating_sub(kw) / s + 1, ih.saturating_sub(kh) / s + 1),
        PaddingMode::Same => (iw.div_ceil(s), ih.div_ceil(s)),
    };
    (ow, oh, nb_kernel)
}

pub fn compute_out_upsample_conv(
    dim_input: (u32, u32, u32),
    scale_factor: u32,
    dim_kernel: (u32, u32, u32),
    nb_kernel: u32,
    padding: &PaddingMode,
) -> (u32, u32, u32) {
    let upsampled = (
        dim_input.0 * scale_factor.max(1),
        dim_input.1 * scale_factor.max(1),
        dim_input.2,
    );
    match padding {
        PaddingMode::Valid => (
            upsampled.0.saturating_sub(dim_kernel.0) + 1,
            upsampled.1.saturating_sub(dim_kernel.1) + 1,
            nb_kernel,
        ),
        PaddingMode::Same => (upsampled.0, upsampled.1, nb_kernel),
    }
}

pub fn compute_inferred_input(
    layers: &[LayerDraft],
    model_input: (u32, u32, u32),
) -> (u32, u32, u32) {
    let mut current = model_input;
    let mut saved_outputs = HashMap::new();
    for layer in layers {
        current = layer.output_dims();
        if let Some(key) = layer.save_key() {
            saved_outputs.insert(key.to_string(), current);
        }
    }
    current
}

impl LayerDraft {
    pub fn save_key(&self) -> Option<&str> {
        match self {
            LayerDraft::Convolution { save_key, .. }
            | LayerDraft::Activation { save_key, .. }
            | LayerDraft::GroupNorm { save_key, .. }
            | LayerDraft::Attention { save_key, .. }
            | LayerDraft::FullyConnected { save_key, .. }
            | LayerDraft::UpsampleConv { save_key, .. }
            | LayerDraft::Concat { save_key, .. } => save_key.as_deref(),
        }
    }

    pub fn output_dims(&self) -> (u32, u32, u32) {
        match self {
            LayerDraft::Convolution {
                dim_input,
                nb_kernel,
                dim_kernel,
                stride,
                padding,
                ..
            } => compute_out_conv(*dim_input, *dim_kernel, *stride, *nb_kernel, padding),
            LayerDraft::Activation { dim_input, .. } => *dim_input,
            LayerDraft::GroupNorm { dim_input, .. } => *dim_input,
            LayerDraft::Attention { dim_input, .. } => *dim_input,
            LayerDraft::FullyConnected { nb_neurons, .. } => (1, 1, *nb_neurons),
            LayerDraft::UpsampleConv {
                dim_input,
                scale_factor,
                nb_kernel,
                dim_kernel,
                padding,
                ..
            } => compute_out_upsample_conv(
                *dim_input,
                *scale_factor,
                *dim_kernel,
                *nb_kernel,
                padding,
            ),
            LayerDraft::Concat {
                dim_input,
                dim_skip,
                ..
            } => (dim_input.0, dim_input.1, dim_input.2 + dim_skip.2),
        }
    }

    pub fn display(&self) -> String {
        match self {
            LayerDraft::Convolution {
                dim_input,
                nb_kernel,
                dim_kernel,
                stride,
                padding,
                save_key,
            } => {
                let (ow, oh, oz) =
                    compute_out_conv(*dim_input, *dim_kernel, *stride, *nb_kernel, padding);
                format!(
                    "Conv  {}x{}x{} -> {}x{}x{}{}",
                    dim_input.0,
                    dim_input.1,
                    dim_input.2,
                    ow,
                    oh,
                    oz,
                    display_save_key(save_key)
                )
            }
            LayerDraft::Activation {
                dim_input,
                method,
                save_key,
            } => {
                format!(
                    "{:<6} {}x{}x{}{}",
                    method.to_string(),
                    dim_input.0,
                    dim_input.1,
                    dim_input.2,
                    display_save_key(save_key)
                )
            }
            LayerDraft::GroupNorm {
                dim_input,
                num_groups,
                save_key,
            } => {
                format!(
                    "GroupNorm(g={}) {}x{}x{}{}",
                    num_groups,
                    dim_input.0,
                    dim_input.1,
                    dim_input.2,
                    display_save_key(save_key)
                )
            }
            LayerDraft::Attention {
                dim_input,
                save_key,
            } => {
                format!(
                    "Attention {}x{}x{} ({} positions){}",
                    dim_input.0,
                    dim_input.1,
                    dim_input.2,
                    dim_input.0 * dim_input.1,
                    display_save_key(save_key)
                )
            }
            LayerDraft::FullyConnected {
                dim_input,
                nb_neurons,
                method,
                save_key,
            } => {
                format!(
                    "Perceptron({}) {}x{}x{} -> 1x1x{}{}",
                    method,
                    dim_input.0,
                    dim_input.1,
                    dim_input.2,
                    nb_neurons,
                    display_save_key(save_key)
                )
            }
            LayerDraft::UpsampleConv {
                dim_input,
                scale_factor,
                nb_kernel,
                dim_kernel,
                padding,
                save_key,
            } => {
                let (ow, oh, oz) = compute_out_upsample_conv(
                    *dim_input,
                    *scale_factor,
                    *dim_kernel,
                    *nb_kernel,
                    padding,
                );
                format!(
                    "Upsamplex{}+Conv {}x{}x{} -> {}x{}x{}{}",
                    scale_factor,
                    dim_input.0,
                    dim_input.1,
                    dim_input.2,
                    ow,
                    oh,
                    oz,
                    display_save_key(save_key)
                )
            }
            LayerDraft::Concat {
                dim_input,
                dim_skip,
                skip_key,
                save_key,
            } => {
                format!(
                    "Concat({}) {}x{}x{} + {}x{}x{} -> {}x{}x{}{}",
                    skip_key,
                    dim_input.0,
                    dim_input.1,
                    dim_input.2,
                    dim_skip.0,
                    dim_skip.1,
                    dim_skip.2,
                    dim_input.0,
                    dim_input.1,
                    dim_input.2 + dim_skip.2,
                    display_save_key(save_key)
                )
            }
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            LayerDraft::Convolution { .. } => "Conv",
            LayerDraft::Activation { .. } => "Activation",
            LayerDraft::GroupNorm { .. } => "GroupNorm",
            LayerDraft::Attention { .. } => "Attention",
            LayerDraft::FullyConnected { .. } => "Perceptron",
            LayerDraft::UpsampleConv { .. } => "UpsampleConv",
            LayerDraft::Concat { .. } => "Concat",
        }
    }

    pub fn input_dim_str(&self) -> String {
        let (x, y, z) = match self {
            LayerDraft::Convolution { dim_input, .. }
            | LayerDraft::Activation { dim_input, .. }
            | LayerDraft::GroupNorm { dim_input, .. }
            | LayerDraft::Attention { dim_input, .. }
            | LayerDraft::FullyConnected { dim_input, .. }
            | LayerDraft::UpsampleConv { dim_input, .. }
            | LayerDraft::Concat { dim_input, .. } => *dim_input,
        };
        format!("{}x{}x{}", x, y, z)
    }

    pub fn output_dim_str(&self) -> String {
        let (x, y, z) = self.output_dims();
        format!("{}x{}x{}", x, y, z)
    }
}

// ---------------------------------------------------------------------------
// Architecture summary — reading a config_file as an architecture
// ---------------------------------------------------------------------------

/// One layer of the stack, as the front door reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureRow {
    /// Position in `layers`, the same index the layer builder shows.
    pub index: usize,
    /// The one-line description [`LayerDraft::display`] already produces.
    pub display: String,
    /// Trainable scalars this layer owns — weights **and** biases.
    pub parameters: u64,
    /// Receptive field in input pixels **after** this layer, when this layer
    /// type moves it. Pointwise layers leave it `None` rather than repeating
    /// the previous value, so the column reads as "here is where the field
    /// grew".
    pub receptive_field: Option<f32>,
    /// Why this layer is worth pointing at, if it is: attention, an upsample,
    /// a skip fusion. Everything else is `None` — a marker on every row marks
    /// nothing.
    pub notable: Option<&'static str>,
}

/// A model's architecture, computed from its `config_file` alone.
///
/// No GPU, no checkpoint, no filesystem: everything here comes out of the
/// layer list the config already carries, which is what makes it affordable to
/// compute for every model in the list on every keystroke.
///
/// The receptive field follows the same recurrence as
/// `tools/receptive_field.py`, and for the same reason: a diffusion model whose
/// output pixel cannot see the whole image cannot choose a *global* content,
/// and drifts towards the dataset mean. `Concat` is skipped by both — it
/// re-injects a skip whose field is smaller, so the main path stays the bound.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureSummary {
    pub rows: Vec<ArchitectureRow>,
    pub layer_count: usize,
    /// Trainable scalars over the whole stack.
    pub parameters: u64,
    /// Receptive field of the last layer that moves it — `None` for a stack
    /// with no convolution at all, where the question has no answer.
    pub receptive_field: Option<f32>,
    /// The input edge the field is compared against (`input_size.0`).
    pub input_edge: u32,
    pub attention_layers: usize,
    pub upsample_layers: usize,
    pub concat_layers: usize,
}

impl ArchitectureSummary {
    /// Whether one output pixel can see the whole input image.
    ///
    /// `None` when there is no receptive field to speak of.
    pub fn covers_the_image(&self) -> Option<bool> {
        self.receptive_field
            .map(|field| field >= self.input_edge as f32)
    }
}

impl LayerDraft {
    /// Trainable scalars this layer owns, weights and biases together.
    ///
    /// Mirrors the buffer sizes the layer types allocate: a convolution's
    /// `kernel_bytes` plus one bias per kernel, GroupNorm's `gamma`/`beta`,
    /// attention's four packed `C×C` projections plus their four bias vectors.
    /// `Activation` and `Concat` own nothing.
    pub fn parameter_count(&self) -> u64 {
        let product = |(x, y, z): (u32, u32, u32)| x as u64 * y as u64 * z as u64;
        match self {
            LayerDraft::Convolution {
                nb_kernel,
                dim_kernel,
                ..
            }
            | LayerDraft::UpsampleConv {
                nb_kernel,
                dim_kernel,
                ..
            } => product(*dim_kernel) * *nb_kernel as u64 + *nb_kernel as u64,
            LayerDraft::GroupNorm { dim_input, .. } => 2 * dim_input.2 as u64,
            // Q, K, V, O — one C×C matrix and one C-vector of bias each.
            LayerDraft::Attention { dim_input, .. } => {
                let channels = dim_input.2 as u64;
                4 * channels * channels + 4 * channels
            }
            LayerDraft::FullyConnected {
                dim_input,
                nb_neurons,
                ..
            } => product(*dim_input) * *nb_neurons as u64 + *nb_neurons as u64,
            LayerDraft::Activation { .. } | LayerDraft::Concat { .. } => 0,
        }
    }

    /// The short word that says why this layer is worth noticing, if it is.
    pub fn notable_marker(&self) -> Option<&'static str> {
        match self {
            LayerDraft::Attention { .. } => Some("attention"),
            LayerDraft::UpsampleConv { .. } => Some("upsample"),
            LayerDraft::Concat { .. } => Some("skip"),
            _ => None,
        }
    }
}

/// Reads a layer stack as an architecture: the pile, what it costs, how far it
/// sees.
pub fn summarize_architecture(
    layers: &[LayerDraft],
    input_size: (u32, u32, u32),
) -> ArchitectureSummary {
    // The recurrence of `tools/receptive_field.py`: a kernel `k` at stride `s`
    // grows the field by `(k - 1) * jump` and multiplies the jump by `s`; an
    // upsample by `f` divides the jump by `f` *before* its convolution reads.
    let (mut field, mut jump) = (1.0f32, 1.0f32);
    let mut last_field = None;
    let mut rows = Vec::with_capacity(layers.len());
    let (mut attention_layers, mut upsample_layers, mut concat_layers) = (0, 0, 0);
    let mut parameters = 0u64;

    for (index, layer) in layers.iter().enumerate() {
        let moved_field = match layer {
            LayerDraft::Convolution {
                dim_kernel, stride, ..
            } => {
                field += (dim_kernel.0.max(1) - 1) as f32 * jump;
                jump *= (*stride).max(1) as f32;
                Some(field)
            }
            LayerDraft::UpsampleConv {
                scale_factor,
                dim_kernel,
                ..
            } => {
                jump /= (*scale_factor).max(1) as f32;
                field += (dim_kernel.0.max(1) - 1) as f32 * jump;
                Some(field)
            }
            _ => None,
        };
        if moved_field.is_some() {
            last_field = Some(field);
        }
        match layer {
            LayerDraft::Attention { .. } => attention_layers += 1,
            LayerDraft::UpsampleConv { .. } => upsample_layers += 1,
            LayerDraft::Concat { .. } => concat_layers += 1,
            _ => {}
        }
        let layer_parameters = layer.parameter_count();
        parameters += layer_parameters;
        rows.push(ArchitectureRow {
            index,
            display: layer.display(),
            parameters: layer_parameters,
            receptive_field: moved_field,
            notable: layer.notable_marker(),
        });
    }

    ArchitectureSummary {
        rows,
        layer_count: layers.len(),
        parameters,
        receptive_field: last_field,
        input_edge: input_size.0,
        attention_layers,
        upsample_layers,
        concat_layers,
    }
}

/// Return a copy of `layer` with `dim_input` replaced by `new_input`.
pub fn update_layer_dim_input(layer: &LayerDraft, new_input: (u32, u32, u32)) -> LayerDraft {
    match layer {
        LayerDraft::Convolution {
            nb_kernel,
            dim_kernel,
            stride,
            padding,
            save_key,
            ..
        } => {
            let kc = new_input.2;
            LayerDraft::Convolution {
                dim_input: new_input,
                nb_kernel: *nb_kernel,
                dim_kernel: (dim_kernel.0, dim_kernel.1, kc),
                stride: *stride,
                padding: padding.clone(),
                save_key: save_key.clone(),
            }
        }
        LayerDraft::Activation {
            method, save_key, ..
        } => LayerDraft::Activation {
            dim_input: new_input,
            method: method.clone(),
            save_key: save_key.clone(),
        },
        LayerDraft::GroupNorm {
            num_groups,
            save_key,
            ..
        } => LayerDraft::GroupNorm {
            dim_input: new_input,
            num_groups: *num_groups,
            save_key: save_key.clone(),
        },
        LayerDraft::Attention { save_key, .. } => LayerDraft::Attention {
            dim_input: new_input,
            save_key: save_key.clone(),
        },
        LayerDraft::FullyConnected {
            nb_neurons,
            method,
            save_key,
            ..
        } => LayerDraft::FullyConnected {
            dim_input: new_input,
            nb_neurons: *nb_neurons,
            method: method.clone(),
            save_key: save_key.clone(),
        },
        LayerDraft::UpsampleConv {
            scale_factor,
            nb_kernel,
            dim_kernel,
            padding,
            save_key,
            ..
        } => {
            let kc = new_input.2;
            LayerDraft::UpsampleConv {
                dim_input: new_input,
                scale_factor: *scale_factor,
                nb_kernel: *nb_kernel,
                dim_kernel: (dim_kernel.0, dim_kernel.1, kc),
                padding: padding.clone(),
                save_key: save_key.clone(),
            }
        }
        LayerDraft::Concat {
            dim_skip,
            skip_key,
            save_key,
            ..
        } => LayerDraft::Concat {
            dim_input: new_input,
            dim_skip: *dim_skip,
            skip_key: skip_key.clone(),
            save_key: save_key.clone(),
        },
    }
}

fn display_save_key(save_key: &Option<String>) -> String {
    save_key
        .as_ref()
        .map(|key| format!(" [save:{key}]"))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// LayerKind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Convolution,
    GroupNorm,
    Attention,
    Activation,
    FullyConnected,
    UpsampleConv,
    Concat,
}

impl fmt::Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerKind::Convolution => write!(f, "Conv"),
            LayerKind::GroupNorm => write!(f, "GNorm"),
            LayerKind::Attention => write!(f, "Attn"),
            LayerKind::Activation => write!(f, "Activ"),
            LayerKind::FullyConnected => write!(f, "Perceptron"),
            LayerKind::UpsampleConv => write!(f, "UpConv"),
            LayerKind::Concat => write!(f, "Concat"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelTemplate {
    pub key: String,
    pub name: String,
    pub description: String,
    pub input_size: (u32, u32, u32),
    pub layers: Vec<LayerDraft>,
    pub default_lr: f32,
    pub default_batch_size: u32,
    pub default_steps: usize,
}

/// Draft builder for the U-Net both templates share.
///
/// Every dimension is *derived* from the layer before it — in particular
/// `dim_kernel.z`, which must equal the running channel count. This is the
/// discipline of `tools/gen_unet_config.py`, ported here for the same reason it
/// exists there: hand-kept dims drift silently. `Convolution` at least rejects a
/// mismatched kernel depth at build time; `UpsampleConv` does **not** — its
/// shader indexes the weights with `IC = dim_input.z` and quietly corrupts.
struct UNetDraft {
    dim: (u32, u32, u32),
    layers: Vec<LayerDraft>,
    skips: HashMap<String, (u32, u32, u32)>,
}

impl UNetDraft {
    fn new(input: (u32, u32, u32)) -> Self {
        Self {
            dim: input,
            layers: Vec::new(),
            skips: HashMap::new(),
        }
    }

    fn push(&mut self, layer: LayerDraft) {
        self.dim = layer.output_dims();
        if let Some(key) = layer.save_key() {
            self.skips.insert(key.to_string(), self.dim);
        }
        self.layers.push(layer);
    }

    fn conv(&mut self, nb_kernel: u32, stride: u32, save_key: Option<&str>) {
        self.push(LayerDraft::Convolution {
            dim_input: self.dim,
            nb_kernel,
            dim_kernel: (3, 3, self.dim.2),
            stride,
            padding: PaddingMode::Same,
            save_key: save_key.map(str::to_string),
        });
    }

    fn group_norm(&mut self, num_groups: u32) {
        self.push(LayerDraft::GroupNorm {
            dim_input: self.dim,
            num_groups,
            save_key: None,
        });
    }

    fn silu(&mut self) {
        self.push(LayerDraft::Activation {
            dim_input: self.dim,
            method: ActivationMethod::Silu,
            save_key: None,
        });
    }

    fn upsample(&mut self, nb_kernel: u32, scale_factor: u32) {
        self.push(LayerDraft::UpsampleConv {
            dim_input: self.dim,
            scale_factor,
            nb_kernel,
            dim_kernel: (3, 3, self.dim.2),
            padding: PaddingMode::Same,
            save_key: None,
        });
    }

    fn concat(&mut self, skip_key: &str) {
        let dim_skip = self.skips[skip_key];
        self.push(LayerDraft::Concat {
            dim_input: self.dim,
            dim_skip,
            skip_key: skip_key.to_string(),
            save_key: None,
        });
    }
}

/// The 12-layer diffusion U-Net both templates share, given its channel budget.
///
/// `time_channels` is the whole point of the split: a diffusion model **must**
/// be conditioned on the timestep, which it is by carrying more input channels
/// than it emits — the excess receives the time embedding. With
/// `time_channels = 0` the input and output depths match, ε̂ degenerates and
/// sampling saturates to white. That is not hypothetical: it is the geometry
/// the repository keeps as `Models/Greyscale_Diffusion_broken`, and it is what
/// both templates shipped until a blind test read their config off disk.
fn diffusion_unet(signal_channels: u32, time_channels: u32) -> ((u32, u32, u32), Vec<LayerDraft>) {
    let input = (32, 32, signal_channels + time_channels);
    let mut net = UNetDraft::new(input);
    net.conv(16, 1, Some("enc1"));
    net.group_norm(4);
    net.silu();
    net.conv(32, 2, None);
    net.group_norm(8);
    net.silu();
    net.conv(32, 1, None);
    net.upsample(16, 2);
    net.concat("enc1");
    net.group_norm(8);
    net.silu();
    // The head emits the signal alone: the time channels are an input, never an
    // output. This is the asymmetry that makes the model conditionable.
    net.conv(signal_channels, 1, None);
    (input, net.layers)
}

pub fn diffusion_template() -> ModelTemplate {
    let (input_size, layers) = diffusion_unet(3, 4);
    ModelTemplate {
        key: "Stable_Diffusion".to_string(),
        name: "Stable Diffusion".to_string(),
        description:
            "RGB diffusion backbone — 3 signal channels out, 7 in (4 carry the time embedding)."
                .to_string(),
        input_size,
        layers,
        default_lr: 0.01,
        default_batch_size: 1,
        default_steps: 200,
    }
}

pub fn greyscale_diffusion_template() -> ModelTemplate {
    let (input_size, layers) = diffusion_unet(1, 2);
    ModelTemplate {
        key: "Greyscale_Diffusion".to_string(),
        name: "Greyscale Diffusion".to_string(),
        description:
            "Greyscale diffusion backbone — 1 signal channel out, 3 in (2 carry the time embedding)."
                .to_string(),
        input_size,
        layers,
        default_lr: 0.01,
        default_batch_size: 1,
        default_steps: 200,
    }
}

pub fn built_in_templates() -> Vec<ModelTemplate> {
    vec![diffusion_template(), greyscale_diffusion_template()]
}

// ---------------------------------------------------------------------------
// Run / model configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub lr: f32,
    pub batch_size: u32,
    pub steps: usize,
    pub dataset_path: String,
    pub loss: LossMethod,
    #[serde(default)]
    pub checkpoint_path: Option<String>,
    #[serde(default)]
    pub load_checkpoint: bool,
    /// Weight-update rule. Absent from configs written before the optimiser
    /// was selectable, which therefore keep SGD.
    #[serde(default)]
    pub optimizer: OptimizerKind,
    /// Weight-initialisation scheme. Same story as `optimizer`: absent from
    /// older configs, which therefore keep the uniform draw.
    #[serde(default)]
    pub weight_init: WeightInit,
    /// Per-timestep loss weighting for diffusion runs, applied by biasing the
    /// timestep draw (see `training::weighting`). Absent from older configs,
    /// which therefore keep the uniform draw and the unweighted ε-MSE.
    #[serde(default)]
    pub loss_weighting: LossWeighting,
    /// Decay of the exponential moving average kept over the weights, or `None`
    /// for no average at all — which is the default, and what every config
    /// written before averaging existed deserialises to.
    ///
    /// A run that keeps one writes a V3 checkpoint holding **both** weight sets,
    /// and sampling then uses the average unless it is told otherwise. See
    /// `model::ema`.
    #[serde(default)]
    pub ema_decay: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default = "InferenceConfig::default_random_seed")]
    pub random_seed: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "InferenceConfig::default_denoising_paths")]
    pub denoising_paths: usize,
    #[serde(default = "InferenceConfig::default_denoise_magnitude")]
    pub denoise_magnitude: f32,
    /// Weights to sample from. `None` falls back to the model's `latest.ckpt`,
    /// which is what every config written before the weight choice was honoured
    /// implicitly meant — so older files keep their behaviour.
    #[serde(default)]
    pub checkpoint: Option<String>,
}

impl InferenceConfig {
    const fn default_random_seed() -> bool {
        true
    }

    const fn default_denoising_paths() -> usize {
        1
    }

    pub const fn default_denoise_magnitude() -> f32 {
        1.0
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            random_seed: Self::default_random_seed(),
            seed: None,
            denoising_paths: Self::default_denoising_paths(),
            denoise_magnitude: Self::default_denoise_magnitude(),
            checkpoint: None,
        }
    }
}

/// An inference that never finishes: descend the chain, re-noise, descend
/// again. See [`crate::model::training::perpetual`] for the itinerary itself —
/// this is only what the TUI asks for and what the config file remembers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpetualConfig {
    #[serde(default = "PerpetualConfig::default_random_seed")]
    pub random_seed: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "PerpetualConfig::default_denoise_magnitude")]
    pub denoise_magnitude: f32,
    /// `t_r` — how far back up the schedule each cycle throws the image.
    #[serde(default = "PerpetualConfig::default_renoise_depth")]
    pub renoise_depth: usize,
    #[serde(default)]
    pub regime: PerpetualRegime,
    /// Reverse steps per second the run is paced to. The sampler runs an order
    /// of magnitude faster than this; see [`PerpetualConfig::default_tempo`].
    #[serde(default = "PerpetualConfig::default_tempo")]
    pub tempo: f32,
    #[serde(default)]
    pub checkpoint: Option<String>,
    /// The dataset a run drifts away from — `Models/<name>/config_file` may
    /// name one, otherwise it is derived from the model's output channels
    /// (grey → `cifar10_grey.batraw`, colour → `cifar10_rgb.batraw`).
    ///
    /// A run whose dataset cannot be found falls back on pure noise at the top
    /// of the schedule, which is what perpetual runs did before this existed.
    #[serde(default)]
    pub seed_dataset: Option<String>,
}

impl PerpetualConfig {
    const fn default_random_seed() -> bool {
        true
    }

    pub const fn default_denoise_magnitude() -> f32 {
        1.0
    }

    /// A quarter of the 256-step schedule: deep enough to recompose the image,
    /// shallow enough that its lineage survives the cycle.
    pub const fn default_renoise_depth() -> usize {
        64
    }

    /// Reverse steps per second.
    ///
    /// The sampler measures ~300 steps/s on this model, which would run a
    /// `t_r = 64` cycle to completion five times a second — a flicker, not a
    /// drift. 30 steps/s puts one denoising step on each visualiser frame (the
    /// window renders at ~28 fps, `INFER_VIZ.md` §4) and makes a full cycle a
    /// couple of seconds long.
    pub const fn default_tempo() -> f32 {
        30.0
    }

    /// Bounds on the tempo dial. The floor keeps a stalled-looking run
    /// distinguishable from a paused one; the ceiling is past what the sampler
    /// can sustain, so it means "as fast as it goes".
    pub const MIN_TEMPO: f32 = 2.0;
    pub const MAX_TEMPO: f32 = 400.0;
    /// One press of the tempo key. Multiplicative, because the interesting
    /// range spans two orders of magnitude.
    pub const TEMPO_FACTOR: f32 = 1.5;
}

impl Default for PerpetualConfig {
    fn default() -> Self {
        Self {
            random_seed: Self::default_random_seed(),
            seed: None,
            denoise_magnitude: Self::default_denoise_magnitude(),
            renoise_depth: Self::default_renoise_depth(),
            regime: PerpetualRegime::default(),
            tempo: Self::default_tempo(),
            checkpoint: None,
            seed_dataset: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunMode {
    Infer,
    Train(TrainingConfig),
    Perpetual(PerpetualConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub mode: RunMode,
}

/// Commands the monitor sends down to whichever worker is running.
///
/// Named for training because that is the only worker that used to accept any,
/// but the channel is the run-control channel: the perpetual worker listens on
/// the same one, which is why `SetPaused` needs no perpetual twin.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingControlCommand {
    SetPaused(bool),
    SaveCheckpoint,
    UpdateParams {
        lr: f32,
        batch_size: u32,
        total_steps: usize,
    },
    /// Move `t_r` by `delta` notches (perpetual runs).
    NudgeRenoiseDepth(i32),
    /// Move the pace by `delta` notches (perpetual runs).
    NudgeTempo(i32),
    /// Restart the drift from fresh noise (perpetual runs).
    Reseed,
    /// Swap wander ↔ breathe (perpetual runs).
    ToggleRegime,
    /// Write the frame currently on screen to a PNG (perpetual runs).
    SaveImage,
    /// Swap the visualiser between x̂₀ alone and the double x_t | x̂₀ view
    /// (perpetual runs). Re-registers the source, so the window comes back at
    /// the new aspect ratio.
    ToggleView,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_name: Option<String>,
    pub input_size: (u32, u32, u32),
    pub layers: Vec<LayerDraft>,
    #[serde(default)]
    pub inference: InferenceConfig,
    pub run: RunConfig,
}

impl ModelConfig {
    pub fn input_elem_count(&self) -> usize {
        let (w, h, c) = self.input_size;
        (w * h * c) as usize
    }

    pub fn output_elem_count(&self) -> usize {
        let out = compute_inferred_input(&self.layers, self.input_size);
        (out.0 * out.1 * out.2) as usize
    }
}


impl ModelConfig {
    /// Parse a `config_file` from its bytes.
    ///
    /// The engine never opens the file itself: the CLI hands it `fs::read(...)`,
    /// a browser would hand it a `fetch` response body. Keeping the filesystem
    /// out of this path is what makes the inference chain portable to wasm.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Serialise back to the on-disk `config_file` form (pretty-printed, as the
    /// TUI has always written it).
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A diffusion model must be conditionable on the timestep: it has to carry
    /// more input channels than it emits, the excess receiving the time
    /// embedding. Without it ε̂ degenerates and sampling saturates to white.
    ///
    /// Both built-in templates violated this — 1→1 and 3→3 — which is exactly
    /// the geometry the repository preserves as
    /// `Models/Greyscale_Diffusion_broken`. Found by a blind test reading the
    /// config the "New model (from template)" flow writes to disk, not by
    /// reading this file.
    #[test]
    fn every_template_is_conditionable_on_the_timestep() {
        for template in built_in_templates() {
            let output = compute_inferred_input(&template.layers, template.input_size);
            assert!(
                template.input_size.2 > output.2,
                "template '{}' emits {} channels for {} in — a model that cannot be \
                 conditioned on t (the geometry of Models/Greyscale_Diffusion_broken)",
                template.key,
                output.2,
                template.input_size.2,
            );
            assert_eq!(
                (template.input_size.0, template.input_size.1),
                (output.0, output.1),
                "template '{}' does not return to its input resolution",
                template.key,
            );
        }
    }

    /// The kernel depth of every convolution must equal the channel count
    /// reaching it. `Convolution` rejects a mismatch at build time, but
    /// `UpsampleConv` does not — it indexes its weights with `dim_input.z` and
    /// corrupts silently. This is why the templates derive their dims.
    #[test]
    fn every_template_kernel_is_as_deep_as_its_input() {
        for template in built_in_templates() {
            for (index, layer) in template.layers.iter().enumerate() {
                let (dim_input, dim_kernel) = match layer {
                    LayerDraft::Convolution {
                        dim_input,
                        dim_kernel,
                        ..
                    }
                    | LayerDraft::UpsampleConv {
                        dim_input,
                        dim_kernel,
                        ..
                    } => (dim_input, dim_kernel),
                    _ => continue,
                };
                assert_eq!(
                    dim_kernel.2, dim_input.2,
                    "template '{}', layer {index}: kernel depth {} for {} input channels",
                    template.key, dim_kernel.2, dim_input.2,
                );
            }
        }
    }

    /// The first layer must consume the model input itself: a template whose
    /// stack starts on a different depth than `input_size` is inconsistent no
    /// matter what the rest does.
    #[test]
    fn every_template_stack_starts_on_its_declared_input() {
        for template in built_in_templates() {
            let first = template.layers.first().expect("a template has layers");
            let LayerDraft::Convolution { dim_input, .. } = first else {
                panic!("template '{}' does not open on a convolution", template.key);
            };
            assert_eq!(*dim_input, template.input_size, "template '{}'", template.key);
        }
    }

    // -- Architecture summary ------------------------------------------------

    /// The receptive field, walked by hand against the recurrence, on the stack
    /// both templates share.
    ///
    /// `diffusion_unet` is conv3(s=1) · conv3(s=2) · conv3(s=1) · up×2+conv3 ·
    /// conv3(s=1), so:
    ///
    /// ```text
    ///   start                 rf = 1     jump = 1
    ///   conv k3 s1            rf = 3     jump = 1
    ///   conv k3 s2            rf = 5     jump = 2
    ///   conv k3 s1            rf = 9     jump = 2
    ///   up ×2 then k3         rf = 11    jump = 1
    ///   conv k3 s1            rf = 13    jump = 1
    /// ```
    ///
    /// 13 px against a 32 px image: the template does **not** cover it, which
    /// is the fact the panel exists to put in front of whoever picks a model.
    #[test]
    fn the_receptive_field_follows_the_recurrence_the_python_tool_walks() {
        for template in built_in_templates() {
            let summary = summarize_architecture(&template.layers, template.input_size);
            assert_eq!(
                summary.receptive_field,
                Some(13.0),
                "template '{}'",
                template.key
            );
            assert_eq!(summary.input_edge, 32);
            assert_eq!(
                summary.covers_the_image(),
                Some(false),
                "template '{}' would be claimed to see the whole image",
                template.key
            );
        }
    }

    /// A stack with nothing that moves the field has no field to report — `1`
    /// would read as "one pixel of context", which is a claim, where the honest
    /// answer is that the question does not apply.
    #[test]
    fn a_stack_with_no_convolution_reports_no_receptive_field() {
        let layers = vec![LayerDraft::GroupNorm {
            dim_input: (8, 8, 4),
            num_groups: 2,
            save_key: None,
        }];
        let summary = summarize_architecture(&layers, (8, 8, 4));
        assert_eq!(summary.receptive_field, None);
        assert_eq!(summary.covers_the_image(), None);
        assert_eq!(summary.parameters, 8, "gamma and beta, one pair per channel");
    }

    /// Every layer type's parameter count, written out term by term. The total
    /// is cross-checked against the *real* buffers a built model allocates by
    /// `the_parameter_count_matches_the_scalars_a_checkpoint_holds`, over in the
    /// binary crate where a model can be built — this one pins the arithmetic
    /// so a wrong term can be read off without a GPU.
    #[test]
    fn each_layer_type_counts_its_weights_and_its_biases() {
        let conv = LayerDraft::Convolution {
            dim_input: (32, 32, 3),
            nb_kernel: 16,
            dim_kernel: (3, 3, 3),
            stride: 1,
            padding: PaddingMode::Same,
            save_key: None,
        };
        assert_eq!(conv.parameter_count(), 3 * 3 * 3 * 16 + 16);

        let upsample = LayerDraft::UpsampleConv {
            dim_input: (16, 16, 32),
            scale_factor: 2,
            nb_kernel: 16,
            dim_kernel: (3, 3, 32),
            padding: PaddingMode::Same,
            save_key: None,
        };
        assert_eq!(upsample.parameter_count(), 3 * 3 * 32 * 16 + 16);

        let attention = LayerDraft::Attention {
            dim_input: (8, 8, 64),
            save_key: None,
        };
        assert_eq!(attention.parameter_count(), 4 * 64 * 64 + 4 * 64);

        let perceptron = LayerDraft::FullyConnected {
            dim_input: (4, 4, 2),
            nb_neurons: 10,
            method: ActivationMethod::Relu,
            save_key: None,
        };
        assert_eq!(perceptron.parameter_count(), 4 * 4 * 2 * 10 + 10);

        // The two that own nothing. A summary that credited them with
        // parameters would inflate every model that uses skips.
        assert_eq!(
            LayerDraft::Activation {
                dim_input: (8, 8, 4),
                method: ActivationMethod::Silu,
                save_key: None,
            }
            .parameter_count(),
            0
        );
        assert_eq!(
            LayerDraft::Concat {
                dim_input: (8, 8, 4),
                dim_skip: (8, 8, 4),
                skip_key: "enc1".to_string(),
                save_key: None,
            }
            .parameter_count(),
            0
        );
    }

    /// The summary is the stack, one row per layer, in order — and the total is
    /// the sum of the rows. A panel that dropped a layer would be lying about
    /// the architecture it is there to describe.
    #[test]
    fn the_summary_holds_one_row_per_layer_and_totals_them() {
        let template = greyscale_diffusion_template();
        let summary = summarize_architecture(&template.layers, template.input_size);

        assert_eq!(summary.layer_count, template.layers.len());
        assert_eq!(summary.rows.len(), template.layers.len());
        assert_eq!(
            summary.rows.iter().map(|row| row.index).collect::<Vec<_>>(),
            (0..template.layers.len()).collect::<Vec<_>>()
        );
        assert_eq!(
            summary.parameters,
            summary.rows.iter().map(|row| row.parameters).sum::<u64>()
        );
        assert!(summary.parameters > 0);

        // The notable layers, counted — the template has one upsample and one
        // skip fusion, and no attention.
        assert_eq!(summary.upsample_layers, 1);
        assert_eq!(summary.concat_layers, 1);
        assert_eq!(summary.attention_layers, 0);
        assert_eq!(
            summary
                .rows
                .iter()
                .filter_map(|row| row.notable)
                .collect::<Vec<_>>(),
            vec!["upsample", "skip"]
        );
    }
}
