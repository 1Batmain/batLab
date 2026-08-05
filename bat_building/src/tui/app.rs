//! File purpose: Implements app behavior for the terminal user interface flow.

use super::storage::{self, SavedModelEntry};
use crate::model::training::{LossWeighting, MIN_RENOISE_DEPTH, PerpetualRegime};
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
fn update_layer_dim_input(layer: &LayerDraft, new_input: (u32, u32, u32)) -> LayerDraft {
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

fn diffusion_template() -> ModelTemplate {
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

fn greyscale_diffusion_template() -> ModelTemplate {
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

fn built_in_templates() -> Vec<ModelTemplate> {
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

    const fn default_denoise_magnitude() -> f32 {
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

    const fn default_denoise_magnitude() -> f32 {
        1.0
    }

    /// A quarter of the 256-step schedule: deep enough to recompose the image,
    /// shallow enough that its lineage survives the cycle.
    const fn default_renoise_depth() -> usize {
        64
    }

    /// Reverse steps per second.
    ///
    /// The sampler measures ~300 steps/s on this model, which would run a
    /// `t_r = 64` cycle to completion five times a second — a flicker, not a
    /// drift. 30 steps/s puts one denoising step on each visualiser frame (the
    /// window renders at ~28 fps, `INFER_VIZ.md` §4) and makes a full cycle a
    /// couple of seconds long.
    const fn default_tempo() -> f32 {
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

// ---------------------------------------------------------------------------
// Screens
// ---------------------------------------------------------------------------

pub enum Screen {
    Home,
    LoadPath,
    TemplateSelector,
    WeightSelector,
    InputSize,
    LayerBuilder,
    ModeSelector,
    InferenceParams,
    PerpetualParams,
    TrainingParams,
    DatasetSelector,
    Monitor,
    TrainingControl,
}

// ---------------------------------------------------------------------------
// Per-screen state
// ---------------------------------------------------------------------------

pub struct HomeState {
    pub selected: usize, // 0 = Load saved, 1 = Use template
}

/// The two routes out of [`Screen::Home`], in the order they are drawn.
pub const HOME_CHOICES: [&str; 2] = ["Load Saved Model", "Select Model Template"];

pub struct LoadPathState {
    pub models: Vec<SavedModelEntry>,
    pub selected: usize,
    pub error: Option<String>,
}

pub struct TemplateSelectorState {
    pub templates: Vec<ModelTemplate>,
    pub selected: usize,
    pub error: Option<String>,
}

pub struct WeightSelectorState {
    pub checkpoints: Vec<storage::CheckpointEntry>,
    pub selected: usize, // 0 = random init, 1.. = existing checkpoints
    pub error: Option<String>,
}

pub struct InputSizeState {
    pub fields: Vec<String>, // [width, height, channels]
    pub field_idx: usize,
    pub error: Option<String>,
}

pub const INPUT_SIZE_FIELD_NAMES: [&str; 3] = ["Width", "Height", "Channels"];

pub struct LayerBuilderState {
    pub layers: Vec<LayerDraft>,
    pub current_kind: LayerKind,
    pub fields: Vec<String>,
    pub field_idx: usize,
    pub model_input: (u32, u32, u32),
    pub error: Option<String>,
    /// Whether we are adding, browsing, or editing a layer.
    pub mode: LayerBuilderMode,
    /// The currently selected (highlighted) layer index when in Browse/Edit mode.
    pub browse_selected: usize,
}

/// The operational mode of the layer builder screen.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerBuilderMode {
    Add,
    Browse,
    Edit,
}

pub struct ModeSelectorState {
    pub selected: usize, // 0 = Infer, 1 = Train
}

pub struct TrainingParamsState {
    pub fields: Vec<String>, // [lr, batch_size, steps, dataset_path]
    pub field_idx: usize,
    pub error: Option<String>,
    pub datasets: Vec<String>,
    pub selected_dataset: usize,
}

pub const TRAINING_PARAM_FIELD_NAMES: [&str; 3] = ["Learning Rate", "Batch Size", "Steps"];

pub struct InferenceParamsState {
    pub random_seed: bool,
    pub fields: Vec<String>, // [seed, denoising_paths, denoise_magnitude]
    pub field_idx: usize,    // 0=random toggle, 1=seed, 2=paths, 3=magnitude
    pub error: Option<String>,
}

pub const INFERENCE_PARAM_FIELD_NAMES: [&str; 4] =
    ["Random Seed", "Seed", "Denoising Paths", "Magnitude"];

pub struct PerpetualParamsState {
    pub random_seed: bool,
    pub regime: PerpetualRegime,
    /// [seed, magnitude, renoise depth, tempo]
    pub fields: Vec<String>,
    /// 0=random toggle, 1=seed, 2=magnitude, 3=depth, 4=tempo, 5=regime toggle
    pub field_idx: usize,
    pub error: Option<String>,
}

pub const PERPETUAL_PARAM_FIELD_NAMES: [&str; 6] = [
    "Random Seed",
    "Seed",
    "Magnitude",
    "Renoise Depth (t_r)",
    "Steps / second",
    "Regime",
];

pub struct TrainingControlState {
    pub fields: Vec<String>, // [lr, batch_size, total_steps]
    pub field_idx: usize,
    pub error: Option<String>,
}

pub const TRAINING_CONTROL_FIELD_NAMES: [&str; 3] = ["Learning Rate", "Batch Size", "Total Steps"];

#[derive(Debug, Clone)]
pub struct MonitorImage {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub pixels: Vec<u8>,
}

/// Live read-out of a perpetual run, republished by the worker on every change.
///
/// The worker owns the drift — the TUI only ever *asks* for a change and is
/// told the result. Mirroring `t_r` in the UI and hoping the two agree is how a
/// footer starts lying about what the sampler is doing.
#[derive(Debug, Clone)]
pub struct PerpetualStatus {
    pub regime: String,
    pub depth: usize,
    pub min_depth: usize,
    pub max_depth: usize,
    pub cycle: usize,
    pub diffusion_step: usize,
    /// Reverse steps walked since the run started.
    pub steps: usize,
    /// Measured pace, as opposed to the requested `tempo`.
    pub steps_per_sec: f32,
    pub tempo: f32,
    pub paused: bool,
}

#[derive(Debug, Clone)]
pub struct LoadingProgress {
    pub label: String,
    pub current: usize,
    pub total: usize,
}

#[derive(Default)]
pub struct MonitorState {
    pub step: usize,
    pub loss_history: Vec<f64>,
    pub done: bool,
    pub total_steps: usize,
    pub last_sample_path: Option<String>,
    pub error: Option<String>,
    /// Set to `true` when the user requests a new training run.
    pub restart_training: bool,
    /// Set after the user successfully saves the model config.
    pub save_status: Option<String>,
    /// The model config currently being monitored (used when saving).
    pub model_config: Option<ModelConfig>,
    /// Device limit proxy for largest single GPU buffer allocation.
    pub max_buffer_bytes: Option<u64>,
    /// Device limit proxy for largest storage-binding allocation.
    pub max_storage_binding_bytes: Option<u64>,
    /// Best-effort estimate of current model+training GPU allocation.
    pub estimated_training_bytes: Option<u64>,
    /// Inference preview image rendered in the monitor.
    pub inference_image: Option<MonitorImage>,
    /// Checkpoint used for the latest inference run.
    pub inference_checkpoint_path: Option<String>,
    /// Seed used for the latest inference sample.
    pub inference_seed: Option<u64>,
    /// Generic loading/progress state for long-running GPU preparation/sampling.
    pub loading_progress: Option<LoadingProgress>,
    /// Whether the training worker is currently paused.
    pub is_training_paused: bool,
    /// Current runtime learning rate (may differ from initial config).
    pub current_lr: Option<f32>,
    /// Current runtime batch size (may differ from initial config).
    pub current_batch_size: Option<u32>,
    /// Commands queued from UI to training worker.
    pub pending_control_commands: Vec<TrainingControlCommand>,
    /// Live state of a perpetual run, as reported by its worker.
    pub perpetual: Option<PerpetualStatus>,
}

// App
// ---------------------------------------------------------------------------

pub struct App {
    pub screen: Screen,
    pub home: HomeState,
    pub load_path: LoadPathState,
    pub template_selector: TemplateSelectorState,
    pub weight_selector: WeightSelectorState,
    pub input_size: InputSizeState,
    pub layer_builder: LayerBuilderState,
    pub mode_selector: ModeSelectorState,
    pub inference_params: InferenceParamsState,
    pub perpetual_params: PerpetualParamsState,
    pub training_params: TrainingParamsState,
    pub training_control: TrainingControlState,
    pub monitor: MonitorState,
    pub run_config: Option<RunConfig>,
    pub active_model_name: Option<String>,
    pub selected_checkpoint_path: Option<String>,
    pub load_checkpoint_on_start: bool,
    pub should_quit: bool,
}

// Field helpers

fn conv_field_names() -> Vec<&'static str> {
    vec![
        "Num Kernels",
        "Kernel W",
        "Kernel H",
        "Stride",
        "Padding",
        "Save As",
    ]
}

fn conv_field_defaults() -> Vec<String> {
    vec![
        "4".into(),
        "3".into(),
        "3".into(),
        "1".into(),
        "Valid".into(),
        "".into(),
    ]
}

fn activation_field_names() -> Vec<&'static str> {
    vec!["Method", "Save As"]
}

fn activation_field_defaults() -> Vec<String> {
    vec!["ReLU".into(), "".into()]
}

fn group_norm_field_names() -> Vec<&'static str> {
    vec!["Groups", "Save As"]
}

fn group_norm_field_defaults() -> Vec<String> {
    vec!["1".into(), "".into()]
}

fn fully_connected_field_names() -> Vec<&'static str> {
    vec!["Neurons", "Method", "Save As"]
}

fn fully_connected_field_defaults() -> Vec<String> {
    vec!["10".into(), "ReLU".into(), "".into()]
}

fn upsample_conv_field_names() -> Vec<&'static str> {
    vec![
        "Scale",
        "Num Kernels",
        "Kernel W",
        "Kernel H",
        "Padding",
        "Save As",
    ]
}

fn upsample_conv_field_defaults() -> Vec<String> {
    vec![
        "2".into(),
        "8".into(),
        "3".into(),
        "3".into(),
        "Same".into(),
        "".into(),
    ]
}

fn concat_field_names() -> Vec<&'static str> {
    vec!["Skip Key", "Save As"]
}

fn concat_field_defaults() -> Vec<String> {
    vec!["".into(), "".into()]
}

fn is_toggle_field(name: &str) -> bool {
    matches!(name, "Padding" | "Method")
}

fn is_numeric_field(name: &str) -> bool {
    matches!(
        name,
        "Groups" | "Scale" | "Num Kernels" | "Kernel W" | "Kernel H" | "Stride" | "Neurons"
    )
}

fn normalize_key(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

impl App {
    pub fn new() -> Self {
        let models = storage::list_models().unwrap_or_default();
        let templates = built_in_templates();
        let (default_lr, default_batch, default_steps) = templates
            .first()
            .map(|template| {
                (
                    template.default_lr,
                    template.default_batch_size,
                    template.default_steps,
                )
            })
            .unwrap_or((0.01, 1, 50));
        let datasets = storage::list_datasets().unwrap_or_default();
        let dataset_path = datasets.first().cloned().unwrap_or_default();
        let mut app = Self {
            // Home, not TemplateSelector: the template route builds a *fresh*
            // architecture and rewrites the target model's `config_file`, so
            // starting there made every saved model unreachable and lossy to
            // reach anyway. Home is the fork — load, or build new.
            screen: Screen::Home,
            home: HomeState { selected: 0 },
            load_path: LoadPathState {
                models,
                selected: 0,
                error: None,
            },
            template_selector: TemplateSelectorState {
                templates,
                selected: 0,
                error: None,
            },
            weight_selector: WeightSelectorState {
                checkpoints: Vec::new(),
                selected: 0,
                error: None,
            },
            input_size: InputSizeState {
                fields: vec!["28".into(), "28".into(), "1".into()],
                field_idx: 0,
                error: None,
            },
            layer_builder: LayerBuilderState {
                layers: Vec::new(),
                current_kind: LayerKind::Convolution,
                fields: Vec::new(),
                field_idx: 0,
                model_input: (28, 28, 1),
                error: None,
                mode: LayerBuilderMode::Add,
                browse_selected: 0,
            },
            mode_selector: ModeSelectorState { selected: 1 },
            inference_params: InferenceParamsState {
                random_seed: true,
                fields: vec!["0".into(), "1".into(), "1.0".into()],
                field_idx: 0,
                error: None,
            },
            perpetual_params: PerpetualParamsState {
                random_seed: true,
                regime: PerpetualRegime::default(),
                fields: vec![
                    "0".into(),
                    PerpetualConfig::default_denoise_magnitude().to_string(),
                    PerpetualConfig::default_renoise_depth().to_string(),
                    PerpetualConfig::default_tempo().to_string(),
                ],
                field_idx: 0,
                error: None,
            },
            training_params: TrainingParamsState {
                fields: vec![
                    default_lr.to_string(),
                    default_batch.to_string(),
                    default_steps.to_string(),
                    dataset_path,
                ],
                field_idx: 0,
                error: None,
                datasets,
                selected_dataset: 0,
            },
            training_control: TrainingControlState {
                fields: vec!["0.01".into(), "1".into(), "1".into()],
                field_idx: 0,
                error: None,
            },
            monitor: MonitorState {
                step: 0,
                loss_history: Vec::new(),
                done: false,
                total_steps: 0,
                last_sample_path: None,
                error: None,
                restart_training: false,
                save_status: None,
                model_config: None,
                max_buffer_bytes: None,
                max_storage_binding_bytes: None,
                estimated_training_bytes: None,
                inference_image: None,
                inference_checkpoint_path: None,
                inference_seed: None,
                loading_progress: None,
                is_training_paused: false,
                current_lr: None,
                current_batch_size: None,
                pending_control_commands: Vec::new(),
                perpetual: None,
            },
            run_config: None,
            active_model_name: None,
            selected_checkpoint_path: None,
            load_checkpoint_on_start: false,
            should_quit: false,
        };
        app.refresh_templates();
        app.sync_selected_dataset_from_field();
        app.sync_inference_params_from_config(&InferenceConfig::default());
        app.reset_layer_form();
        app
    }

    /// Re-scans `Models/` so the load list reflects the disk, not the snapshot
    /// taken when the app was constructed.
    fn refresh_load_path(&mut self) {
        self.load_path.models = storage::list_models().unwrap_or_default();
        if self.load_path.selected >= self.load_path.models.len() {
            self.load_path.selected = 0;
        }
        self.load_path.error = None;
    }

    fn refresh_templates(&mut self) {
        self.template_selector.templates = built_in_templates();
        if self.template_selector.selected >= self.template_selector.templates.len() {
            self.template_selector.selected =
                self.template_selector.templates.len().saturating_sub(1);
        }
        self.template_selector.error = None;
    }

    fn refresh_weight_selector(&mut self) {
        let Some(model_name) = self.active_model_name.clone() else {
            self.weight_selector.checkpoints.clear();
            self.weight_selector.selected = 0;
            self.weight_selector.error = Some("No model selected.".to_string());
            return;
        };
        match storage::list_model_checkpoints(&model_name) {
            Ok(checkpoints) => {
                self.weight_selector.checkpoints = checkpoints;
                if self.weight_selector.selected > self.weight_selector.checkpoints.len() {
                    self.weight_selector.selected = 0;
                }
                self.weight_selector.error = None;
            }
            Err(err) => {
                self.weight_selector.checkpoints.clear();
                self.weight_selector.selected = 0;
                self.weight_selector.error =
                    Some(format!("Failed to read pretrained weights: {err}"));
            }
        }
    }

    fn refresh_datasets(&mut self) {
        self.training_params.datasets = storage::list_datasets().unwrap_or_default();
        if self.training_params.datasets.is_empty() {
            self.training_params.selected_dataset = 0;
            return;
        }
        if self.training_params.fields[3].trim().is_empty() {
            self.training_params.fields[3] = self.training_params.datasets[0].clone();
            self.training_params.selected_dataset = 0;
        } else {
            self.sync_selected_dataset_from_field();
        }
    }

    pub fn sync_selected_dataset_from_field(&mut self) {
        if let Some(index) = self
            .training_params
            .datasets
            .iter()
            .position(|dataset| dataset == &self.training_params.fields[3])
        {
            self.training_params.selected_dataset = index;
        }
    }

    fn apply_template(&mut self, template: &ModelTemplate) -> Result<(), String> {
        self.active_model_name = Some(template.key.clone());
        self.load_checkpoint_on_start = false;
        self.selected_checkpoint_path = Some(
            storage::default_model_checkpoint_path(&template.key)
                .map_err(|err| format!("Failed to prepare model folder: {err}"))?
                .to_string_lossy()
                .to_string(),
        );
        self.layer_builder.model_input = template.input_size;
        self.input_size.fields = vec![
            template.input_size.0.to_string(),
            template.input_size.1.to_string(),
            template.input_size.2.to_string(),
        ];
        self.layer_builder.layers = template.layers.clone();
        self.layer_builder.error = None;
        self.layer_builder.mode = LayerBuilderMode::Add;
        self.layer_builder.browse_selected = self.layer_builder.layers.len().saturating_sub(1);
        self.input_size.error = None;

        self.training_params.fields[0] = template.default_lr.to_string();
        self.training_params.fields[1] = template.default_batch_size.to_string();
        self.training_params.fields[2] = template.default_steps.to_string();
        self.training_params.error = None;
        self.training_params.field_idx = 0;
        self.refresh_datasets();
        if self.training_params.datasets.is_empty() {
            self.training_params.fields[3].clear();
            self.training_params.selected_dataset = 0;
        } else {
            self.training_params.selected_dataset = 0;
            self.training_params.fields[3] = self.training_params.datasets[0].clone();
        }

        self.mode_selector.selected = 1;
        let config = ModelConfig {
            model_name: Some(template.key.clone()),
            input_size: template.input_size,
            layers: template.layers.clone(),
            inference: InferenceConfig::default(),
            run: RunConfig {
                mode: RunMode::Infer,
            },
        };
        storage::write_model_config(&template.key, &config)
            .map_err(|err| format!("Failed to write template config_file: {err}"))?;
        self.refresh_weight_selector();
        self.weight_selector.selected = 0;
        self.screen = Screen::WeightSelector;
        Ok(())
    }

    fn apply_loaded_model(&mut self, config: ModelConfig) {
        self.active_model_name = config.model_name.clone();
        self.layer_builder.model_input = config.input_size;
        self.input_size.fields = vec![
            config.input_size.0.to_string(),
            config.input_size.1.to_string(),
            config.input_size.2.to_string(),
        ];
        self.layer_builder.layers = config.layers;
        self.layer_builder.error = None;
        self.input_size.error = None;
        self.refresh_datasets();
        self.sync_inference_params_from_config(&config.inference);

        match config.run.mode {
            RunMode::Infer => {
                self.mode_selector.selected = 0;
                self.selected_checkpoint_path = config.inference.checkpoint.clone();
                self.load_checkpoint_on_start = config.inference.checkpoint.is_some();
            }
            RunMode::Train(train) => {
                self.mode_selector.selected = 1;
                self.training_params.fields = vec![
                    train.lr.to_string(),
                    train.batch_size.to_string(),
                    train.steps.to_string(),
                    train.dataset_path,
                ];
                self.selected_checkpoint_path = train.checkpoint_path.clone();
                self.load_checkpoint_on_start = train.load_checkpoint;
                self.sync_selected_dataset_from_field();
            }
            RunMode::Perpetual(perpetual) => {
                self.mode_selector.selected = 2;
                self.selected_checkpoint_path = perpetual.checkpoint.clone();
                self.load_checkpoint_on_start = perpetual.checkpoint.is_some();
                self.sync_perpetual_params_from_config(&perpetual);
            }
        }

        // The weight list is the point of loading a saved model: its trained
        // checkpoints live beside the config and are otherwise unreachable.
        // Note what this path does *not* do — unlike `apply_template`, it never
        // calls `write_model_config`, so the model's own `inference` block
        // survives being opened.
        self.refresh_weight_selector();
        let preselected = self
            .selected_checkpoint_path
            .as_deref()
            .and_then(|path| {
                self.weight_selector
                    .checkpoints
                    .iter()
                    .position(|entry| entry.path == path)
            })
            .map_or(0, |index| index + 1);
        self.weight_selector.selected = preselected;
        self.screen = Screen::WeightSelector;
    }

    fn sync_inference_params_from_config(&mut self, inference: &InferenceConfig) {
        self.inference_params.random_seed = inference.random_seed;
        self.inference_params.fields[0] = inference.seed.unwrap_or(0).to_string();
        self.inference_params.fields[1] = inference.denoising_paths.max(1).to_string();
        self.inference_params.fields[2] = inference.denoise_magnitude.to_string();
        self.inference_params.field_idx = 0;
        self.inference_params.error = None;
    }

    pub fn cycle_dataset_forward(&mut self) {
        if self.training_params.datasets.is_empty() {
            return;
        }
        self.training_params.selected_dataset =
            (self.training_params.selected_dataset + 1) % self.training_params.datasets.len();
        self.training_params.fields[3] =
            self.training_params.datasets[self.training_params.selected_dataset].clone();
        self.training_params.error = None;
    }

    pub fn cycle_dataset_backward(&mut self) {
        if self.training_params.datasets.is_empty() {
            return;
        }
        if self.training_params.selected_dataset == 0 {
            self.training_params.selected_dataset = self.training_params.datasets.len() - 1;
        } else {
            self.training_params.selected_dataset -= 1;
        }
        self.training_params.fields[3] =
            self.training_params.datasets[self.training_params.selected_dataset].clone();
        self.training_params.error = None;
    }

    pub fn select_dataset(&mut self, index: usize) {
        if self.training_params.datasets.is_empty() {
            self.training_params.selected_dataset = 0;
            self.training_params.fields[3].clear();
            return;
        }
        let clamped = index.min(self.training_params.datasets.len() - 1);
        self.training_params.selected_dataset = clamped;
        self.training_params.fields[3] = self.training_params.datasets[clamped].clone();
        self.training_params.error = None;
    }

    pub fn layer_field_names(&self) -> Vec<&'static str> {
        match self.layer_builder.current_kind {
            LayerKind::Convolution => conv_field_names(),
            LayerKind::GroupNorm => group_norm_field_names(),
            LayerKind::Activation => activation_field_names(),
            LayerKind::FullyConnected => fully_connected_field_names(),
            LayerKind::UpsampleConv => upsample_conv_field_names(),
            LayerKind::Concat => concat_field_names(),
        }
    }

    pub fn reset_layer_form(&mut self) {
        self.layer_builder.fields = match self.layer_builder.current_kind {
            LayerKind::Convolution => conv_field_defaults(),
            LayerKind::GroupNorm => group_norm_field_defaults(),
            LayerKind::Activation => activation_field_defaults(),
            LayerKind::FullyConnected => fully_connected_field_defaults(),
            LayerKind::UpsampleConv => upsample_conv_field_defaults(),
            LayerKind::Concat => concat_field_defaults(),
        };
        self.layer_builder.field_idx = 0;
        self.layer_builder.error = None;
    }

    pub fn inferred_input(&self) -> (u32, u32, u32) {
        compute_inferred_input(&self.layer_builder.layers, self.layer_builder.model_input)
    }

    fn saved_output_dims(&self) -> HashMap<String, (u32, u32, u32)> {
        self.layer_builder
            .layers
            .iter()
            .filter_map(|layer| {
                layer
                    .save_key()
                    .map(|key| (key.to_string(), layer.output_dims()))
            })
            .collect()
    }

    /// Live preview of output dims based on current form values (best-effort).
    pub fn preview_output(&self) -> Option<(u32, u32, u32)> {
        let lb = &self.layer_builder;
        let inferred = self.inferred_input();
        match lb.current_kind {
            LayerKind::Convolution => {
                let names = self.layer_field_names();
                let get = |name: &str| -> Option<u32> {
                    let idx = names.iter().position(|&n| n == name)?;
                    lb.fields.get(idx)?.parse().ok()
                };
                let nb_kernel = get("Num Kernels")?;
                let kw = get("Kernel W")?;
                let kh = get("Kernel H")?;
                let kc = inferred.2; // inferred from input depth
                let stride = get("Stride")?;
                if stride == 0 {
                    return None;
                }
                let pidx = names.iter().position(|&n| n == "Padding")?;
                let padding = if lb.fields.get(pidx)? == "Same" {
                    PaddingMode::Same
                } else {
                    PaddingMode::Valid
                };
                Some(compute_out_conv(
                    inferred,
                    (kw, kh, kc),
                    stride,
                    nb_kernel,
                    &padding,
                ))
            }
            LayerKind::GroupNorm => {
                let names = self.layer_field_names();
                let idx = names.iter().position(|&n| n == "Groups")?;
                let groups = lb.fields.get(idx)?.parse::<u32>().ok()?;
                if groups == 0 || inferred.2 == 0 || inferred.2 % groups != 0 {
                    return None;
                }
                Some(inferred)
            }
            LayerKind::Activation => Some(inferred),
            LayerKind::FullyConnected => {
                let names = self.layer_field_names();
                let idx = names.iter().position(|&n| n == "Neurons")?;
                let nb_neurons = lb.fields.get(idx)?.parse().ok()?;
                Some((1, 1, nb_neurons))
            }
            LayerKind::UpsampleConv => {
                let names = self.layer_field_names();
                let get = |name: &str| -> Option<u32> {
                    let idx = names.iter().position(|&n| n == name)?;
                    lb.fields.get(idx)?.parse().ok()
                };
                let scale_factor = get("Scale")?;
                if scale_factor == 0 {
                    return None;
                }
                let nb_kernel = get("Num Kernels")?;
                let kw = get("Kernel W")?;
                let kh = get("Kernel H")?;
                let padding_idx = names.iter().position(|&n| n == "Padding")?;
                let padding = if lb.fields.get(padding_idx)? == "Same" {
                    PaddingMode::Same
                } else {
                    PaddingMode::Valid
                };
                Some(compute_out_upsample_conv(
                    inferred,
                    scale_factor,
                    (kw, kh, inferred.2),
                    nb_kernel,
                    &padding,
                ))
            }
            LayerKind::Concat => {
                let names = self.layer_field_names();
                let idx = names.iter().position(|&n| n == "Skip Key")?;
                let skip_key = normalize_key(lb.fields.get(idx)?)?;
                let dim_skip = self.saved_output_dims().get(&skip_key).copied()?;
                if dim_skip.0 != inferred.0 || dim_skip.1 != inferred.1 {
                    return None;
                }
                Some((inferred.0, inferred.1, inferred.2 + dim_skip.2))
            }
        }
    }

    pub fn try_add_layer(&mut self) -> Result<(), String> {
        let inferred = self.inferred_input();
        let draft = self.build_draft_from_form(inferred, None)?;
        self.layer_builder.layers.push(draft);
        self.load_checkpoint_on_start = false;
        if let Some(model_name) = self.active_model_name.as_deref() {
            self.selected_checkpoint_path = storage::default_model_checkpoint_path(model_name)
                .ok()
                .map(|path| path.to_string_lossy().to_string());
        }
        self.layer_builder.error = None;
        self.reset_layer_form();
        Ok(())
    }

    pub fn delete_last_layer(&mut self) {
        self.layer_builder.layers.pop();
        self.load_checkpoint_on_start = false;
        if let Some(model_name) = self.active_model_name.as_deref() {
            self.selected_checkpoint_path = storage::default_model_checkpoint_path(model_name)
                .ok()
                .map(|path| path.to_string_lossy().to_string());
        }
        self.reset_layer_form();
    }

    // --- Layer editing helpers ---

    /// Compute the inferred input dimensions for a layer at a given index
    /// (i.e. the output dims of the previous layer, or model_input for index 0).
    pub fn inferred_input_for(&self, idx: usize) -> (u32, u32, u32) {
        if idx == 0 {
            self.layer_builder.model_input
        } else {
            compute_inferred_input(
                &self.layer_builder.layers[..idx],
                self.layer_builder.model_input,
            )
        }
    }

    /// Reconstruct a `LayerDraft` from the current form fields using the provided
    /// `inferred` input dims and optionally excluding a save key from the duplicate check.
    fn build_draft_from_form(
        &self,
        inferred: (u32, u32, u32),
        exclude_save_key: Option<&str>,
    ) -> Result<LayerDraft, String> {
        let names = self.layer_field_names();
        let fields = self.layer_builder.fields.clone();

        let parse_u32 = |name: &str| -> Result<u32, String> {
            let idx = names
                .iter()
                .position(|&n| n == name)
                .expect("field name mismatch");
            fields[idx]
                .parse::<u32>()
                .map_err(|_| format!("'{name}' must be a positive integer"))
        };
        let existing_saved = self.saved_output_dims();
        let parse_save_key = || -> Result<Option<String>, String> {
            let Some(idx) = names.iter().position(|&n| n == "Save As") else {
                return Ok(None);
            };
            let save_key = normalize_key(&fields[idx]);
            if let Some(ref key) = save_key {
                // Allow reusing the same key that was already on this layer (editing).
                if Some(key.as_str()) != exclude_save_key && existing_saved.contains_key(key) {
                    return Err(format!("Save key '{key}' already exists"));
                }
            }
            Ok(save_key)
        };

        match self.layer_builder.current_kind {
            LayerKind::Convolution => {
                let nb_kernel = parse_u32("Num Kernels")?;
                let kw = parse_u32("Kernel W")?;
                let kh = parse_u32("Kernel H")?;
                let kc = inferred.2;
                let stride = parse_u32("Stride")?;
                if stride == 0 {
                    return Err("Stride must be > 0".into());
                }
                let pidx = names.iter().position(|&n| n == "Padding").unwrap();
                let padding = if fields[pidx] == "Same" {
                    PaddingMode::Same
                } else {
                    PaddingMode::Valid
                };
                Ok(LayerDraft::Convolution {
                    dim_input: inferred,
                    nb_kernel,
                    dim_kernel: (kw, kh, kc),
                    stride,
                    padding,
                    save_key: parse_save_key()?,
                })
            }
            LayerKind::GroupNorm => {
                let num_groups = parse_u32("Groups")?;
                if num_groups == 0 || inferred.2 == 0 || inferred.2 % num_groups != 0 {
                    return Err(format!(
                        "Groups must be > 0 and divide the input channels ({})",
                        inferred.2
                    ));
                }
                Ok(LayerDraft::GroupNorm {
                    dim_input: inferred,
                    num_groups,
                    save_key: parse_save_key()?,
                })
            }
            LayerKind::Activation => {
                let method = ActivationMethod::from_label(&fields[0])
                    .ok_or_else(|| format!("Unknown activation method '{}'", fields[0]))?;
                Ok(LayerDraft::Activation {
                    dim_input: inferred,
                    method,
                    save_key: parse_save_key()?,
                })
            }
            LayerKind::FullyConnected => {
                let nb_neurons = parse_u32("Neurons")?;
                if nb_neurons == 0 {
                    return Err("Neurons must be > 0".into());
                }
                let method_idx = names.iter().position(|&n| n == "Method").unwrap();
                let method = ActivationMethod::from_label(&fields[method_idx])
                    .ok_or_else(|| format!("Unknown activation method '{}'", fields[method_idx]))?;
                Ok(LayerDraft::FullyConnected {
                    dim_input: inferred,
                    nb_neurons,
                    method,
                    save_key: parse_save_key()?,
                })
            }
            LayerKind::UpsampleConv => {
                let scale_factor = parse_u32("Scale")?;
                if scale_factor == 0 {
                    return Err("Scale must be > 0".into());
                }
                let nb_kernel = parse_u32("Num Kernels")?;
                let kw = parse_u32("Kernel W")?;
                let kh = parse_u32("Kernel H")?;
                let kc = inferred.2;
                let pidx = names.iter().position(|&n| n == "Padding").unwrap();
                let padding = if fields[pidx] == "Same" {
                    PaddingMode::Same
                } else {
                    PaddingMode::Valid
                };
                Ok(LayerDraft::UpsampleConv {
                    dim_input: inferred,
                    scale_factor,
                    nb_kernel,
                    dim_kernel: (kw, kh, kc),
                    padding,
                    save_key: parse_save_key()?,
                })
            }
            LayerKind::Concat => {
                let skip_idx = names.iter().position(|&n| n == "Skip Key").unwrap();
                let skip_key = normalize_key(&fields[skip_idx])
                    .ok_or_else(|| "Skip Key must not be empty".to_string())?;
                let dim_skip = existing_saved
                    .get(&skip_key)
                    .copied()
                    .ok_or_else(|| format!("Unknown skip key '{skip_key}'"))?;
                if dim_skip.0 != inferred.0 || dim_skip.1 != inferred.1 {
                    return Err(format!(
                        "Concat requires matching spatial dims, got input {}x{} and skip {}x{}",
                        inferred.0, inferred.1, dim_skip.0, dim_skip.1
                    ));
                }
                Ok(LayerDraft::Concat {
                    dim_input: inferred,
                    dim_skip,
                    skip_key,
                    save_key: parse_save_key()?,
                })
            }
        }
    }

    /// Rebuild `dim_input` for all layers starting from `start_idx` so that the
    /// chain stays consistent after an edit.  Invalid Concat spatial mismatches are
    /// left in place so the user can see and correct them.
    fn rebuild_layer_dims_from(&mut self, start_idx: usize) {
        for i in start_idx..self.layer_builder.layers.len() {
            let new_input = if i == 0 {
                self.layer_builder.model_input
            } else {
                self.layer_builder.layers[i - 1].output_dims()
            };
            self.layer_builder.layers[i] =
                update_layer_dim_input(&self.layer_builder.layers[i], new_input);
        }
    }

    /// Populate the form fields from an existing layer so the user can edit it.
    pub fn populate_form_from_layer(&mut self, idx: usize) {
        let layer = self.layer_builder.layers[idx].clone();
        self.layer_builder.current_kind = match &layer {
            LayerDraft::Convolution { .. } => LayerKind::Convolution,
            LayerDraft::Activation { .. } => LayerKind::Activation,
            LayerDraft::GroupNorm { .. } => LayerKind::GroupNorm,
            LayerDraft::FullyConnected { .. } => LayerKind::FullyConnected,
            LayerDraft::UpsampleConv { .. } => LayerKind::UpsampleConv,
            LayerDraft::Concat { .. } => LayerKind::Concat,
        };
        self.layer_builder.fields = match &layer {
            LayerDraft::Convolution {
                nb_kernel,
                dim_kernel,
                stride,
                padding,
                save_key,
                ..
            } => vec![
                nb_kernel.to_string(),
                dim_kernel.0.to_string(),
                dim_kernel.1.to_string(),
                stride.to_string(),
                padding.to_string(),
                save_key.as_deref().unwrap_or("").to_string(),
            ],
            LayerDraft::Activation {
                method, save_key, ..
            } => vec![
                method.to_string(),
                save_key.as_deref().unwrap_or("").to_string(),
            ],
            LayerDraft::GroupNorm {
                num_groups,
                save_key,
                ..
            } => vec![
                num_groups.to_string(),
                save_key.as_deref().unwrap_or("").to_string(),
            ],
            LayerDraft::FullyConnected {
                nb_neurons,
                method,
                save_key,
                ..
            } => vec![
                nb_neurons.to_string(),
                method.to_string(),
                save_key.as_deref().unwrap_or("").to_string(),
            ],
            LayerDraft::UpsampleConv {
                scale_factor,
                nb_kernel,
                dim_kernel,
                padding,
                save_key,
                ..
            } => vec![
                scale_factor.to_string(),
                nb_kernel.to_string(),
                dim_kernel.0.to_string(),
                dim_kernel.1.to_string(),
                padding.to_string(),
                save_key.as_deref().unwrap_or("").to_string(),
            ],
            LayerDraft::Concat {
                skip_key, save_key, ..
            } => vec![
                skip_key.clone(),
                save_key.as_deref().unwrap_or("").to_string(),
            ],
        };
        self.layer_builder.field_idx = 0;
        self.layer_builder.error = None;
    }

    // --- Browse / Edit mode transitions ---

    pub fn enter_browse_mode(&mut self) {
        if self.layer_builder.layers.is_empty() {
            return;
        }
        self.layer_builder.mode = LayerBuilderMode::Browse;
        self.layer_builder.browse_selected = self
            .layer_builder
            .browse_selected
            .min(self.layer_builder.layers.len().saturating_sub(1));
        self.layer_builder.error = None;
    }

    pub fn exit_browse_mode(&mut self) {
        self.layer_builder.mode = LayerBuilderMode::Add;
        self.reset_layer_form();
    }

    pub fn enter_edit_mode(&mut self) {
        if self.layer_builder.layers.is_empty() {
            return;
        }
        let idx = self.layer_builder.browse_selected;
        self.populate_form_from_layer(idx);
        self.layer_builder.mode = LayerBuilderMode::Edit;
    }

    pub fn cancel_edit(&mut self) {
        self.layer_builder.mode = LayerBuilderMode::Browse;
        self.layer_builder.error = None;
    }

    pub fn confirm_layer_edit(&mut self) -> Result<(), String> {
        let idx = self.layer_builder.browse_selected;
        let inferred = self.inferred_input_for(idx);
        // Find the current save key of the layer being edited so we don't flag it as duplicate.
        let existing_key = self.layer_builder.layers[idx]
            .save_key()
            .map(|s| s.to_string());
        let draft = self.build_draft_from_form(inferred, existing_key.as_deref())?;
        self.layer_builder.layers[idx] = draft;
        self.load_checkpoint_on_start = false;
        if let Some(model_name) = self.active_model_name.as_deref() {
            self.selected_checkpoint_path = storage::default_model_checkpoint_path(model_name)
                .ok()
                .map(|path| path.to_string_lossy().to_string());
        }
        self.rebuild_layer_dims_from(idx + 1);
        self.layer_builder.error = None;
        self.layer_builder.mode = LayerBuilderMode::Browse;
        Ok(())
    }

    pub fn browse_move_up(&mut self) {
        if self.layer_builder.browse_selected > 0 {
            self.layer_builder.browse_selected -= 1;
        }
    }

    pub fn browse_move_down(&mut self) {
        if self.layer_builder.layers.is_empty() {
            return;
        }
        let last = self.layer_builder.layers.len() - 1;
        if self.layer_builder.browse_selected < last {
            self.layer_builder.browse_selected += 1;
        }
    }

    pub fn delete_selected_layer(&mut self) {
        if self.layer_builder.layers.is_empty() {
            return;
        }
        let idx = self.layer_builder.browse_selected;
        self.layer_builder.layers.remove(idx);
        self.load_checkpoint_on_start = false;
        if let Some(model_name) = self.active_model_name.as_deref() {
            self.selected_checkpoint_path = storage::default_model_checkpoint_path(model_name)
                .ok()
                .map(|path| path.to_string_lossy().to_string());
        }
        self.rebuild_layer_dims_from(idx);
        if self.layer_builder.layers.is_empty() {
            self.exit_browse_mode();
        } else {
            self.layer_builder.browse_selected =
                idx.min(self.layer_builder.layers.len().saturating_sub(1));
        }
    }

    pub fn trigger_monitor_save(&mut self) -> Result<(), String> {
        let mut config = self
            .monitor
            .model_config
            .as_ref()
            .ok_or_else(|| "No model config available".to_string())?
            .clone();
        let model_name = config
            .model_name
            .clone()
            .or_else(|| self.active_model_name.clone())
            .unwrap_or_else(|| {
                storage::next_model_name().unwrap_or_else(|_| "model-001".to_string())
            });
        config.model_name = Some(model_name.clone());
        let config_path = storage::write_model_config(&model_name, &config)
            .map_err(|err| format!("failed to save model config: {err}"))?;
        self.active_model_name = Some(model_name.clone());
        self.selected_checkpoint_path = storage::default_model_checkpoint_path(&model_name)
            .ok()
            .map(|path| path.to_string_lossy().to_string());
        self.load_checkpoint_on_start = false;
        self.monitor.model_config = Some(config);
        self.monitor.error = None;

        if self.monitor.current_lr.is_some() && !self.monitor.done {
            self.monitor
                .pending_control_commands
                .push(TrainingControlCommand::SaveCheckpoint);
            self.monitor.save_status = Some(format!(
                "Saved config → {} | checkpoint requested",
                config_path.display()
            ));
        } else {
            self.monitor
                .save_status
                .replace(format!("Saved config → {}", config_path.display()));
        }
        Ok(())
    }

    // --- Monitor restart ---

    pub fn request_restart(&mut self) {
        self.monitor.restart_training = true;
    }

    pub fn toggle_training_pause(&mut self) {
        let next = !self.monitor.is_training_paused;
        self.monitor.is_training_paused = next;
        self.monitor
            .pending_control_commands
            .push(TrainingControlCommand::SetPaused(next));
    }

    pub fn toggle_visualise(&mut self) {
        if let Err(err) = super::visualiser_control::toggle_visualiser() {
            self.monitor.error = Some(err);
        }
    }

    pub fn open_training_control(&mut self) -> Result<(), String> {
        let lr = self
            .monitor
            .current_lr
            .ok_or_else(|| "Training controls are available only in training mode.".to_string())?;
        let batch_size = self
            .monitor
            .current_batch_size
            .ok_or_else(|| "Training controls are available only in training mode.".to_string())?;
        self.training_control.fields = vec![
            lr.to_string(),
            batch_size.to_string(),
            self.monitor.total_steps.to_string(),
        ];
        self.training_control.field_idx = 0;
        self.training_control.error = None;
        self.screen = Screen::TrainingControl;
        Ok(())
    }

    pub fn finish_training_control(&mut self) -> Result<(), String> {
        let lr = self.training_control.fields[0]
            .parse::<f32>()
            .map_err(|_| "Learning rate must be a number".to_string())?;
        let batch_size = self.training_control.fields[1]
            .parse::<u32>()
            .map_err(|_| "Batch size must be an integer".to_string())?;
        let total_steps = self.training_control.fields[2]
            .parse::<usize>()
            .map_err(|_| "Total steps must be an integer".to_string())?;
        if lr <= 0.0 {
            return Err("Learning rate must be > 0".to_string());
        }
        if batch_size == 0 {
            return Err("Batch size must be > 0".to_string());
        }
        if total_steps == 0 {
            return Err("Total steps must be > 0".to_string());
        }

        self.monitor.current_lr = Some(lr);
        self.monitor.current_batch_size = Some(batch_size);
        self.monitor.total_steps = total_steps;
        self.training_control.error = None;
        self.monitor
            .pending_control_commands
            .push(TrainingControlCommand::UpdateParams {
                lr,
                batch_size,
                total_steps,
            });

        if let Some(config) = self.monitor.model_config.as_mut()
            && let RunMode::Train(ref mut train) = config.run.mode
        {
            train.lr = lr;
            train.batch_size = batch_size;
            train.steps = total_steps;
        }

        self.screen = Screen::Monitor;
        Ok(())
    }

    pub fn drain_monitor_control_commands(&mut self) -> Vec<TrainingControlCommand> {
        std::mem::take(&mut self.monitor.pending_control_commands)
    }

    pub fn cycle_kind_forward(&mut self) {
        self.layer_builder.current_kind = match self.layer_builder.current_kind {
            LayerKind::Convolution => LayerKind::GroupNorm,
            LayerKind::GroupNorm => LayerKind::Activation,
            LayerKind::Activation => LayerKind::FullyConnected,
            LayerKind::FullyConnected => LayerKind::UpsampleConv,
            LayerKind::UpsampleConv => LayerKind::Concat,
            LayerKind::Concat => LayerKind::Convolution,
        };
        self.reset_layer_form();
    }

    pub fn cycle_kind_backward(&mut self) {
        self.layer_builder.current_kind = match self.layer_builder.current_kind {
            LayerKind::Convolution => LayerKind::Concat,
            LayerKind::GroupNorm => LayerKind::Convolution,
            LayerKind::Activation => LayerKind::GroupNorm,
            LayerKind::FullyConnected => LayerKind::Activation,
            LayerKind::UpsampleConv => LayerKind::FullyConnected,
            LayerKind::Concat => LayerKind::UpsampleConv,
        };
        self.reset_layer_form();
    }

    pub fn handle_char_layer(&mut self, c: char) {
        let names = self.layer_field_names();
        let idx = self.layer_builder.field_idx;
        if idx >= names.len() {
            return;
        }
        if is_toggle_field(names[idx]) {
            self.toggle_layer_field();
            return;
        }
        if is_numeric_field(names[idx]) {
            if c.is_ascii_digit() {
                self.layer_builder.fields[idx].push(c);
            }
        } else if !c.is_control() {
            self.layer_builder.fields[idx].push(c);
        }
    }

    pub fn handle_backspace_layer(&mut self) {
        let names = self.layer_field_names();
        let idx = self.layer_builder.field_idx;
        if idx >= names.len() || is_toggle_field(names[idx]) {
            return;
        }
        self.layer_builder.fields[idx].pop();
    }

    pub fn toggle_layer_field(&mut self) {
        let names = self.layer_field_names();
        let idx = self.layer_builder.field_idx;
        if idx >= names.len() {
            return;
        }
        match names[idx] {
            "Padding" => {
                let cur = if self.layer_builder.fields[idx] == "Same" {
                    PaddingMode::Same
                } else {
                    PaddingMode::Valid
                };
                self.layer_builder.fields[idx] = cur.toggle().to_string();
            }
            "Method" => {
                let cur = ActivationMethod::from_label(&self.layer_builder.fields[idx])
                    .expect("invalid activation method label");
                self.layer_builder.fields[idx] = cur.toggle().to_string();
            }
            _ => {}
        }
    }

    // --- Screen transitions ---

    pub fn finish_home(&mut self) {
        if self.home.selected == 0 {
            self.refresh_load_path();
            self.screen = Screen::LoadPath;
        } else {
            self.refresh_templates();
            self.screen = Screen::TemplateSelector;
        }
    }

    pub fn finish_template_selector(&mut self) {
        let Some(template) = self
            .template_selector
            .templates
            .get(self.template_selector.selected)
            .cloned()
        else {
            self.template_selector.error = Some("No templates are available.".to_string());
            return;
        };
        self.template_selector.error = None;
        if let Err(err) = self.apply_template(&template) {
            self.template_selector.error = Some(err);
        }
    }

    pub fn finish_weight_selector(&mut self) {
        let Some(model_name) = self.active_model_name.clone() else {
            self.weight_selector.error = Some("No model selected.".to_string());
            return;
        };

        if self.weight_selector.selected == 0 {
            self.load_checkpoint_on_start = false;
            match storage::default_model_checkpoint_path(&model_name) {
                Ok(path) => {
                    self.selected_checkpoint_path = Some(path.to_string_lossy().to_string());
                    self.weight_selector.error = None;
                    self.screen = Screen::ModeSelector;
                }
                Err(err) => {
                    self.weight_selector.error =
                        Some(format!("Failed to prepare default checkpoint path: {err}"));
                }
            }
            return;
        }

        let checkpoint_index = self.weight_selector.selected - 1;
        let Some(entry) = self.weight_selector.checkpoints.get(checkpoint_index) else {
            self.weight_selector.error = Some("Selected checkpoint is invalid.".to_string());
            return;
        };
        self.load_checkpoint_on_start = true;
        self.selected_checkpoint_path = Some(entry.path.clone());
        self.weight_selector.error = None;
        self.screen = Screen::ModeSelector;
    }

    pub fn finish_load_path(&mut self) {
        let Some(model) = self.load_path.models.get(self.load_path.selected) else {
            self.load_path.error = Some("No model configs found in Models/".into());
            return;
        };
        match storage::load_model_config(&model.path) {
            Ok(config) => self.apply_loaded_model(config),
            Err(err) => {
                self.load_path.error = Some(format!("Failed to load {}: {err}", model.name));
            }
        }
    }

    pub fn finish_input_size(&mut self) -> Result<(), String> {
        let w = self.input_size.fields[0]
            .parse::<u32>()
            .map_err(|_| "Width must be a positive integer".to_string())?;
        let h = self.input_size.fields[1]
            .parse::<u32>()
            .map_err(|_| "Height must be a positive integer".to_string())?;
        let c = self.input_size.fields[2]
            .parse::<u32>()
            .map_err(|_| "Channels must be a positive integer".to_string())?;
        if w == 0 || h == 0 || c == 0 {
            return Err("Dimensions must be > 0".into());
        }
        self.load_checkpoint_on_start = false;
        if let Some(model_name) = self.active_model_name.as_deref() {
            self.selected_checkpoint_path = storage::default_model_checkpoint_path(model_name)
                .ok()
                .map(|path| path.to_string_lossy().to_string());
        }
        self.layer_builder.model_input = (w, h, c);
        self.layer_builder.layers.clear();
        self.reset_layer_form();
        self.refresh_datasets();
        self.screen = Screen::LayerBuilder;
        Ok(())
    }

    pub fn finish_layer_builder(&mut self) {
        if self.layer_builder.layers.is_empty() {
            self.layer_builder.error = Some("Add at least one layer before proceeding.".into());
            return;
        }
        self.layer_builder.error = None;
        self.screen = Screen::ModeSelector;
    }

    pub fn finish_mode_selector(&mut self) {
        match self.mode_selector.selected {
            0 => {
                self.inference_params.error = None;
                self.inference_params.field_idx = 0;
                self.screen = Screen::InferenceParams;
            }
            1 => {
                self.training_params.error = None;
                self.training_params.field_idx = 0;
                self.screen = Screen::TrainingParams;
            }
            _ => {
                self.perpetual_params.error = None;
                self.perpetual_params.field_idx = 0;
                self.screen = Screen::PerpetualParams;
            }
        }
    }

    pub fn toggle_inference_seed_mode(&mut self) {
        self.inference_params.random_seed = !self.inference_params.random_seed;
        self.inference_params.error = None;
    }

    pub fn finish_inference_params(&mut self) -> Result<(), String> {
        let seed = if self.inference_params.random_seed {
            None
        } else {
            let parsed = self.inference_params.fields[0]
                .parse::<u64>()
                .map_err(|_| "Seed must be an unsigned integer".to_string())?;
            Some(parsed)
        };
        let denoising_paths = self.inference_params.fields[1]
            .parse::<usize>()
            .map_err(|_| "Denoising paths must be a positive integer".to_string())?;
        if denoising_paths == 0 {
            return Err("Denoising paths must be > 0".to_string());
        }
        let denoise_magnitude = self.inference_params.fields[2]
            .parse::<f32>()
            .map_err(|_| "Denoise magnitude must be a number".to_string())?;
        if denoise_magnitude <= 0.0 {
            return Err("Denoise magnitude must be > 0".to_string());
        }

        let inference = InferenceConfig {
            random_seed: self.inference_params.random_seed,
            seed,
            denoising_paths,
            denoise_magnitude,
            checkpoint: self.selected_checkpoint_path.clone(),
        };

        self.inference_params.error = None;
        self.run_config = Some(RunConfig {
            mode: RunMode::Infer,
        });
        if let Some(config) = self.monitor.model_config.as_mut() {
            config.inference = inference.clone();
        }
        self.inference_params.fields[1] = denoising_paths.to_string();
        self.inference_params.fields[2] = denoise_magnitude.to_string();
        Ok(())
    }

    // --- Perpetual ---

    pub fn toggle_perpetual_seed_mode(&mut self) {
        self.perpetual_params.random_seed = !self.perpetual_params.random_seed;
        self.perpetual_params.error = None;
    }

    pub fn toggle_perpetual_regime(&mut self) {
        self.perpetual_params.regime = self.perpetual_params.regime.toggle();
        self.perpetual_params.error = None;
    }

    fn sync_perpetual_params_from_config(&mut self, cfg: &PerpetualConfig) {
        self.perpetual_params.random_seed = cfg.random_seed;
        self.perpetual_params.regime = cfg.regime;
        self.perpetual_params.fields[0] = cfg.seed.unwrap_or(0).to_string();
        self.perpetual_params.fields[1] = cfg.denoise_magnitude.to_string();
        self.perpetual_params.fields[2] = cfg.renoise_depth.to_string();
        self.perpetual_params.fields[3] = cfg.tempo.to_string();
        self.perpetual_params.field_idx = 0;
        self.perpetual_params.error = None;
    }

    pub fn finish_perpetual_params(&mut self) -> Result<(), String> {
        let seed = if self.perpetual_params.random_seed {
            None
        } else {
            Some(
                self.perpetual_params.fields[0]
                    .parse::<u64>()
                    .map_err(|_| "Seed must be an unsigned integer".to_string())?,
            )
        };
        let denoise_magnitude = self.perpetual_params.fields[1]
            .parse::<f32>()
            .map_err(|_| "Magnitude must be a number".to_string())?;
        if denoise_magnitude <= 0.0 {
            return Err("Magnitude must be > 0".to_string());
        }
        let renoise_depth = self.perpetual_params.fields[2]
            .parse::<usize>()
            .map_err(|_| "Renoise depth must be a positive integer".to_string())?;
        if renoise_depth < MIN_RENOISE_DEPTH {
            return Err(format!("Renoise depth must be >= {MIN_RENOISE_DEPTH}"));
        }
        let tempo = self.perpetual_params.fields[3]
            .parse::<f32>()
            .map_err(|_| "Steps/second must be a number".to_string())?;
        if !(PerpetualConfig::MIN_TEMPO..=PerpetualConfig::MAX_TEMPO).contains(&tempo) {
            return Err(format!(
                "Steps/second must be between {} and {}",
                PerpetualConfig::MIN_TEMPO,
                PerpetualConfig::MAX_TEMPO
            ));
        }

        let perpetual = PerpetualConfig {
            random_seed: self.perpetual_params.random_seed,
            seed,
            denoise_magnitude,
            renoise_depth,
            regime: self.perpetual_params.regime,
            tempo,
            checkpoint: self.selected_checkpoint_path.clone(),
        };

        self.perpetual_params.error = None;
        self.perpetual_params.fields[1] = denoise_magnitude.to_string();
        self.perpetual_params.fields[2] = renoise_depth.to_string();
        self.perpetual_params.fields[3] = tempo.to_string();
        self.run_config = Some(RunConfig {
            mode: RunMode::Perpetual(perpetual.clone()),
        });
        if let Some(config) = self.monitor.model_config.as_mut() {
            config.run.mode = RunMode::Perpetual(perpetual);
        }
        Ok(())
    }

    pub fn handle_char_perpetual(&mut self, c: char) {
        let idx = self.perpetual_params.field_idx;
        let accepted = match idx {
            1 | 3 => c.is_ascii_digit(),
            2 | 4 => c.is_ascii_digit() || c == '.',
            _ => false,
        };
        if accepted {
            self.perpetual_params.fields[idx - 1].push(c);
            self.perpetual_params.error = None;
        }
    }

    pub fn handle_backspace_perpetual(&mut self) {
        let idx = self.perpetual_params.field_idx;
        if (1..=4).contains(&idx) {
            self.perpetual_params.fields[idx - 1].pop();
        }
    }

    /// Whether the monitor is watching a perpetual run — the gate on every
    /// perpetual key, so they cannot fire during a training or inference run
    /// where they would mean something else (or nothing).
    pub fn is_perpetual_run(&self) -> bool {
        self.monitor
            .model_config
            .as_ref()
            .is_some_and(|config| matches!(config.run.mode, RunMode::Perpetual(_)))
    }

    /// Queues a perpetual control command. The worker owns the drift, so
    /// nothing here anticipates the result: the footer updates when the run
    /// says so.
    pub fn send_perpetual_command(&mut self, command: TrainingControlCommand) {
        self.monitor.pending_control_commands.push(command);
    }

    pub fn toggle_perpetual_pause(&mut self) {
        let next = !self.monitor.is_training_paused;
        self.monitor.is_training_paused = next;
        self.send_perpetual_command(TrainingControlCommand::SetPaused(next));
    }

    pub fn enter_layer_builder_from_mode(&mut self) {
        self.layer_builder.error = None;
        if self.layer_builder.layers.is_empty() {
            self.layer_builder.mode = LayerBuilderMode::Add;
        } else {
            self.layer_builder.mode = LayerBuilderMode::Browse;
            if self.layer_builder.browse_selected >= self.layer_builder.layers.len() {
                self.layer_builder.browse_selected = self.layer_builder.layers.len() - 1;
            }
        }
        self.screen = Screen::LayerBuilder;
    }

    pub fn finish_dataset_selector(&mut self) -> Result<(), String> {
        let lr = self.training_params.fields[0]
            .parse::<f32>()
            .map_err(|_| "Learning rate must be a number".to_string())?;
        let batch = self.training_params.fields[1]
            .parse::<u32>()
            .map_err(|_| "Batch size must be an integer".to_string())?;
        let steps = self.training_params.fields[2]
            .parse::<usize>()
            .map_err(|_| "Steps must be an integer".to_string())?;
        if lr <= 0.0 {
            return Err("Learning rate must be > 0".into());
        }
        if batch == 0 {
            return Err("Batch size must be > 0".into());
        }
        if steps == 0 {
            return Err("Steps must be > 0".into());
        }
        if self.training_params.datasets.is_empty() {
            return Err("No datasets found in datasets/".into());
        }
        self.select_dataset(self.training_params.selected_dataset);
        let dataset_path = self.training_params.fields[3].clone();
        if dataset_path.trim().is_empty() {
            return Err("Dataset path must not be empty".into());
        }
        if !std::path::Path::new(&dataset_path).exists() {
            return Err("Dataset path does not exist".into());
        }
        self.monitor.total_steps = steps;
        self.monitor.last_sample_path = None;
        self.monitor.error = None;
        self.run_config = Some(RunConfig {
            mode: RunMode::Train(TrainingConfig {
                lr,
                batch_size: batch,
                steps,
                dataset_path,
                loss: LossMethod::MeanSquared,
                checkpoint_path: self.selected_checkpoint_path.clone(),
                load_checkpoint: self.load_checkpoint_on_start,
                optimizer: OptimizerKind::default(),
                weight_init: WeightInit::default(),
                loss_weighting: LossWeighting::default(),
            }),
        });
        Ok(())
    }

    pub fn finish_training_params(&mut self) -> Result<(), String> {
        let lr = self.training_params.fields[0]
            .parse::<f32>()
            .map_err(|_| "Learning rate must be a number".to_string())?;
        let batch = self.training_params.fields[1]
            .parse::<u32>()
            .map_err(|_| "Batch size must be an integer".to_string())?;
        let steps = self.training_params.fields[2]
            .parse::<usize>()
            .map_err(|_| "Steps must be an integer".to_string())?;
        if lr <= 0.0 {
            return Err("Learning rate must be > 0".into());
        }
        if batch == 0 {
            return Err("Batch size must be > 0".into());
        }
        if steps == 0 {
            return Err("Steps must be > 0".into());
        }
        self.monitor.total_steps = steps;
        self.training_params.error = None;
        self.training_params.fields[0] = lr.to_string();
        self.training_params.fields[1] = batch.to_string();
        self.training_params.fields[2] = steps.to_string();
        self.screen = Screen::DatasetSelector;
        Ok(())
    }

    // --- Character input helpers ---

    pub fn handle_char_input_size(&mut self, c: char) {
        if c.is_ascii_digit() {
            let idx = self.input_size.field_idx;
            self.input_size.fields[idx].push(c);
            self.input_size.error = None;
        }
    }

    pub fn handle_backspace_input_size(&mut self) {
        let idx = self.input_size.field_idx;
        self.input_size.fields[idx].pop();
    }

    pub fn handle_char_training(&mut self, c: char) {
        let idx = self.training_params.field_idx;
        let accepted = match idx {
            0 => c.is_ascii_digit() || c == '.',
            1 | 2 => c.is_ascii_digit(),
            _ => false,
        };
        if accepted {
            self.training_params.fields[idx].push(c);
            self.training_params.error = None;
        }
    }

    pub fn handle_backspace_training(&mut self) {
        let idx = self.training_params.field_idx;
        self.training_params.fields[idx].pop();
    }

    pub fn handle_char_inference(&mut self, c: char) {
        let idx = self.inference_params.field_idx;
        let accepted = match idx {
            1 => c.is_ascii_digit(),
            2 => c.is_ascii_digit(),
            3 => c.is_ascii_digit() || c == '.',
            _ => false,
        };
        if accepted {
            self.inference_params.fields[idx - 1].push(c);
            self.inference_params.error = None;
        }
    }

    pub fn handle_backspace_inference(&mut self) {
        let idx = self.inference_params.field_idx;
        if (1..=3).contains(&idx) {
            self.inference_params.fields[idx - 1].pop();
        }
    }

    pub fn handle_char_training_control(&mut self, c: char) {
        let idx = self.training_control.field_idx;
        let accepted = match idx {
            0 => c.is_ascii_digit() || c == '.',
            1 | 2 => c.is_ascii_digit(),
            _ => false,
        };
        if accepted {
            self.training_control.fields[idx].push(c);
            self.training_control.error = None;
        }
    }

    pub fn handle_backspace_training_control(&mut self) {
        let idx = self.training_control.field_idx;
        self.training_control.fields[idx].pop();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        App, InferenceConfig, LayerKind, LossMethod, LossWeighting, MIN_RENOISE_DEPTH, ModelConfig,
        OptimizerKind, PerpetualRegime, RunConfig, RunMode, Screen, TrainingConfig,
        TrainingControlCommand, WeightInit,
    };

    #[test]
    fn cycle_kind_backward_moves_in_reverse_order() {
        let mut app = App::new();
        app.layer_builder.current_kind = LayerKind::Convolution;

        app.cycle_kind_backward();
        assert_eq!(app.layer_builder.current_kind, LayerKind::Concat);

        app.cycle_kind_backward();
        assert_eq!(app.layer_builder.current_kind, LayerKind::UpsampleConv);
    }

    /// `Screen::LoadPath` existed, was drawn, and handled its keys, but nothing
    /// ever assigned it — the app opened straight on the template selector, so a
    /// trained model under `Models/` could not be reached at all. Home is the
    /// fork that makes it reachable, and it must default to the load route.
    #[test]
    fn app_starts_on_home_with_the_load_route_selected() {
        let app = App::new();
        assert!(matches!(app.screen, Screen::Home));
        assert_eq!(app.home.selected, 0);
    }

    #[test]
    fn home_routes_to_load_path_and_to_the_template_selector() {
        let mut app = App::new();

        app.home.selected = 0;
        app.finish_home();
        assert!(matches!(app.screen, Screen::LoadPath));

        app.home.selected = 1;
        app.finish_home();
        assert!(matches!(app.screen, Screen::TemplateSelector));
    }

    /// Opening a saved model must not touch what it says. The template route
    /// calls `write_model_config` with `InferenceConfig::default()`, which is
    /// how the previous mission silently reset two `config_file`s; the load
    /// route has to carry the stored inference block into the form instead.
    #[test]
    fn loading_a_model_preserves_its_inference_block() {
        let mut app = App::new();
        let config = ModelConfig {
            model_name: Some("unit-test-load".to_string()),
            input_size: (32, 32, 5),
            layers: Vec::new(),
            inference: InferenceConfig {
                random_seed: false,
                seed: Some(4242),
                denoising_paths: 7,
                denoise_magnitude: 0.35,
                checkpoint: None,
            },
            run: RunConfig {
                mode: RunMode::Infer,
            },
        };

        app.apply_loaded_model(config);

        assert!(!app.inference_params.random_seed);
        assert_eq!(app.inference_params.fields[0], "4242");
        assert_eq!(app.inference_params.fields[1], "7");
        assert_eq!(app.inference_params.fields[2], "0.35");
        assert_eq!(app.active_model_name.as_deref(), Some("unit-test-load"));
        // The weights of a loaded model are the point of loading it.
        assert!(matches!(app.screen, Screen::WeightSelector));
    }

    #[test]
    fn selecting_template_prefills_architecture_and_advances_to_weight_selector() {
        let mut app = App::new();

        app.finish_template_selector();

        assert!(matches!(app.screen, Screen::WeightSelector));
        assert!(!app.layer_builder.layers.is_empty());
        assert_eq!(app.layer_builder.model_input, (32, 32, 3));
    }

    #[test]
    fn selecting_random_weights_advances_to_mode_selector() {
        let mut app = App::new();
        app.finish_template_selector();
        app.weight_selector.selected = 0;

        app.finish_weight_selector();

        assert!(matches!(app.screen, Screen::ModeSelector));
        assert!(!app.load_checkpoint_on_start);
        assert!(app.selected_checkpoint_path.is_some());
    }

    #[test]
    fn inference_selection_builds_infer_run_config() {
        let mut app = App::new();
        app.finish_template_selector();
        app.finish_weight_selector();
        app.mode_selector.selected = 0;

        app.finish_mode_selector();
        assert!(matches!(app.screen, Screen::InferenceParams));

        app.inference_params.random_seed = false;
        app.inference_params.fields[0] = "123".to_string();
        app.inference_params.fields[1] = "2".to_string();
        app.inference_params.fields[2] = "0.8".to_string();
        app.finish_inference_params()
            .expect("inference params should be accepted");

        let Some(run) = app.run_config.as_ref() else {
            panic!("run config should be set");
        };
        assert!(matches!(run.mode, RunMode::Infer));
    }

    #[test]
    fn perpetual_selection_builds_a_perpetual_run_config() {
        let mut app = App::new();
        app.finish_template_selector();
        app.finish_weight_selector();
        app.mode_selector.selected = 2;

        app.finish_mode_selector();
        assert!(matches!(app.screen, Screen::PerpetualParams));

        app.perpetual_params.random_seed = false;
        app.perpetual_params.fields[0] = "99".to_string();
        app.perpetual_params.fields[1] = "0.9".to_string();
        app.perpetual_params.fields[2] = "48".to_string();
        app.perpetual_params.fields[3] = "20".to_string();
        app.toggle_perpetual_regime();
        app.finish_perpetual_params()
            .expect("perpetual params should be accepted");

        let Some(run) = app.run_config.as_ref() else {
            panic!("run config should be set");
        };
        match &run.mode {
            RunMode::Perpetual(cfg) => {
                assert_eq!(cfg.seed, Some(99));
                assert_eq!(cfg.denoise_magnitude, 0.9);
                assert_eq!(cfg.renoise_depth, 48);
                assert_eq!(cfg.tempo, 20.0);
                assert_eq!(cfg.regime, PerpetualRegime::Breathe);
            }
            _ => panic!("expected a perpetual run mode"),
        }
    }

    /// `t_r` below the drift's own minimum would make a cycle a single step and
    /// freeze the piece; the form must refuse it rather than let the drift
    /// silently clamp something the user cannot see.
    #[test]
    fn perpetual_params_reject_a_depth_below_the_drift_minimum() {
        let mut app = App::new();
        app.perpetual_params.fields[2] = (MIN_RENOISE_DEPTH - 1).to_string();
        assert!(app.finish_perpetual_params().is_err());
    }

    #[test]
    fn training_selection_advances_to_training_params() {
        let mut app = App::new();
        app.finish_template_selector();
        app.finish_weight_selector();
        app.mode_selector.selected = 1;

        app.finish_mode_selector();

        assert!(matches!(app.screen, Screen::TrainingParams));
    }

    #[test]
    fn dataset_selector_requires_available_dataset() {
        let mut app = App::new();
        app.finish_template_selector();
        app.finish_weight_selector();
        app.mode_selector.selected = 1;
        app.finish_mode_selector();
        app.training_params.fields[0] = "0.01".to_string();
        app.training_params.fields[1] = "1".to_string();
        app.training_params.fields[2] = "10".to_string();
        app.finish_training_params()
            .expect("training params should be valid");
        app.training_params.datasets.clear();
        app.training_params.fields[3].clear();

        let result = app.finish_dataset_selector();

        assert!(result.is_err());
    }

    #[test]
    fn training_params_and_dataset_build_train_run_config() {
        let mut app = App::new();
        app.finish_template_selector();
        app.finish_weight_selector();
        app.mode_selector.selected = 1;
        app.finish_mode_selector();

        app.training_params.fields[0] = "0.005".to_string();
        app.training_params.fields[1] = "3".to_string();
        app.training_params.fields[2] = "77".to_string();
        app.finish_training_params()
            .expect("training params should be accepted");
        assert!(matches!(app.screen, Screen::DatasetSelector));

        app.training_params.datasets = vec![".".to_string()];
        app.training_params.selected_dataset = 0;
        app.training_params.fields[3] = ".".to_string();
        app.finish_dataset_selector()
            .expect("dataset selector should produce run config");

        let Some(run) = app.run_config.as_ref() else {
            panic!("run config should be set");
        };

        match &run.mode {
            RunMode::Train(train) => {
                assert_eq!(train.lr, 0.005);
                assert_eq!(train.batch_size, 3);
                assert_eq!(train.steps, 77);
                assert_eq!(train.dataset_path, ".");
                assert_eq!(train.load_checkpoint, false);
                assert!(train.checkpoint_path.is_some());
            }
            _ => panic!("expected training run mode"),
        }
    }

    #[test]
    fn toggle_training_pause_enqueues_set_paused_command() {
        let mut app = App::new();
        app.monitor.current_lr = Some(0.01);
        app.monitor.current_batch_size = Some(1);

        app.toggle_training_pause();
        let commands = app.drain_monitor_control_commands();

        assert_eq!(commands, vec![TrainingControlCommand::SetPaused(true)]);
        assert!(app.monitor.is_training_paused);
    }

    #[test]
    fn finishing_training_control_enqueues_update_command() {
        let mut app = App::new();
        app.monitor.current_lr = Some(0.01);
        app.monitor.current_batch_size = Some(2);
        app.monitor.total_steps = 100;
        app.monitor.model_config = Some(ModelConfig {
            model_name: Some("unit-test-model".to_string()),
            input_size: (32, 32, 3),
            layers: Vec::new(),
            inference: InferenceConfig::default(),
            run: RunConfig {
                mode: RunMode::Train(TrainingConfig {
                    lr: 0.01,
                    batch_size: 2,
                    steps: 100,
                    dataset_path: ".".to_string(),
                    loss: LossMethod::MeanSquared,
                    checkpoint_path: None,
                    load_checkpoint: false,
                    optimizer: OptimizerKind::default(),
                    weight_init: WeightInit::default(),
                    loss_weighting: LossWeighting::default(),
                }),
            },
        });

        app.open_training_control()
            .expect("training control should open in training mode");
        app.training_control.fields[0] = "0.005".to_string();
        app.training_control.fields[1] = "4".to_string();
        app.training_control.fields[2] = "250".to_string();
        app.finish_training_control()
            .expect("valid training control update should succeed");

        let commands = app.drain_monitor_control_commands();
        assert_eq!(
            commands,
            vec![TrainingControlCommand::UpdateParams {
                lr: 0.005,
                batch_size: 4,
                total_steps: 250
            }]
        );
        assert!(matches!(app.screen, Screen::Monitor));
        assert_eq!(app.monitor.current_lr, Some(0.005));
        assert_eq!(app.monitor.current_batch_size, Some(4));
        assert_eq!(app.monitor.total_steps, 250);
    }
}
