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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationMethod {
    Relu,
    Silu,
    Linear,
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

pub fn diffusion_template() -> ModelTemplate {
    ModelTemplate {
        key: "Stable_Diffusion".to_string(),
        name: "Stable Diffusion".to_string(),
        description: "RGB diffusion backbone (32x32x3) with optional pretrained checkpoints."
            .to_string(),
        input_size: (32, 32, 3),
        layers: vec![
            LayerDraft::Convolution {
                dim_input: (32, 32, 3),
                nb_kernel: 16,
                dim_kernel: (3, 3, 3),
                stride: 1,
                padding: PaddingMode::Same,
                save_key: Some("enc1".to_string()),
            },
            LayerDraft::GroupNorm {
                dim_input: (32, 32, 16),
                num_groups: 4,
                save_key: None,
            },
            LayerDraft::Activation {
                dim_input: (32, 32, 16),
                method: ActivationMethod::Silu,
                save_key: None,
            },
            LayerDraft::Convolution {
                dim_input: (32, 32, 16),
                nb_kernel: 32,
                dim_kernel: (3, 3, 16),
                stride: 2,
                padding: PaddingMode::Same,
                save_key: None,
            },
            LayerDraft::GroupNorm {
                dim_input: (16, 16, 32),
                num_groups: 8,
                save_key: None,
            },
            LayerDraft::Activation {
                dim_input: (16, 16, 32),
                method: ActivationMethod::Silu,
                save_key: None,
            },
            LayerDraft::Convolution {
                dim_input: (16, 16, 32),
                nb_kernel: 32,
                dim_kernel: (3, 3, 32),
                stride: 1,
                padding: PaddingMode::Same,
                save_key: None,
            },
            LayerDraft::UpsampleConv {
                dim_input: (16, 16, 32),
                scale_factor: 2,
                nb_kernel: 16,
                dim_kernel: (3, 3, 32),
                padding: PaddingMode::Same,
                save_key: None,
            },
            LayerDraft::Concat {
                dim_input: (32, 32, 16),
                dim_skip: (32, 32, 16),
                skip_key: "enc1".to_string(),
                save_key: None,
            },
            LayerDraft::GroupNorm {
                dim_input: (32, 32, 32),
                num_groups: 8,
                save_key: None,
            },
            LayerDraft::Activation {
                dim_input: (32, 32, 32),
                method: ActivationMethod::Silu,
                save_key: None,
            },
            LayerDraft::Convolution {
                dim_input: (32, 32, 32),
                nb_kernel: 3,
                dim_kernel: (3, 3, 32),
                stride: 1,
                padding: PaddingMode::Same,
                save_key: None,
            },
        ],
        default_lr: 0.01,
        default_batch_size: 1,
        default_steps: 200,
    }
}

pub fn greyscale_diffusion_template() -> ModelTemplate {
    ModelTemplate {
        key: "Greyscale_Diffusion".to_string(),
        name: "Greyscale Diffusion".to_string(),
        description: "Greyscale diffusion backbone (32x32x1) — same U-Net structure as RGB, single channel in/out."
            .to_string(),
        input_size: (32, 32, 1),
        layers: vec![
            LayerDraft::Convolution {
                dim_input: (32, 32, 1),
                nb_kernel: 16,
                dim_kernel: (3, 3, 1),
                stride: 1,
                padding: PaddingMode::Same,
                save_key: Some("enc1".to_string()),
            },
            LayerDraft::GroupNorm {
                dim_input: (32, 32, 16),
                num_groups: 4,
                save_key: None,
            },
            LayerDraft::Activation {
                dim_input: (32, 32, 16),
                method: ActivationMethod::Silu,
                save_key: None,
            },
            LayerDraft::Convolution {
                dim_input: (32, 32, 16),
                nb_kernel: 32,
                dim_kernel: (3, 3, 16),
                stride: 2,
                padding: PaddingMode::Same,
                save_key: None,
            },
            LayerDraft::GroupNorm {
                dim_input: (16, 16, 32),
                num_groups: 8,
                save_key: None,
            },
            LayerDraft::Activation {
                dim_input: (16, 16, 32),
                method: ActivationMethod::Silu,
                save_key: None,
            },
            LayerDraft::Convolution {
                dim_input: (16, 16, 32),
                nb_kernel: 32,
                dim_kernel: (3, 3, 32),
                stride: 1,
                padding: PaddingMode::Same,
                save_key: None,
            },
            LayerDraft::UpsampleConv {
                dim_input: (16, 16, 32),
                scale_factor: 2,
                nb_kernel: 16,
                dim_kernel: (3, 3, 32),
                padding: PaddingMode::Same,
                save_key: None,
            },
            LayerDraft::Concat {
                dim_input: (32, 32, 16),
                dim_skip: (32, 32, 16),
                skip_key: "enc1".to_string(),
                save_key: None,
            },
            LayerDraft::GroupNorm {
                dim_input: (32, 32, 32),
                num_groups: 8,
                save_key: None,
            },
            LayerDraft::Activation {
                dim_input: (32, 32, 32),
                method: ActivationMethod::Silu,
                save_key: None,
            },
            LayerDraft::Convolution {
                dim_input: (32, 32, 32),
                nb_kernel: 1,
                dim_kernel: (3, 3, 32),
                stride: 1,
                padding: PaddingMode::Same,
                save_key: None,
            },
        ],
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
