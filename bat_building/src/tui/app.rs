use super::storage::{self, SavedModelEntry};
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunMode {
    Infer,
    Train(TrainingConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub mode: RunMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_name: Option<String>,
    pub input_size: (u32, u32, u32),
    pub layers: Vec<LayerDraft>,
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
    InputSize,
    LayerBuilder,
    ModeSelector,
    TrainingParams,
    Monitor,
    SaveModel,
}

// ---------------------------------------------------------------------------
// Per-screen state
// ---------------------------------------------------------------------------

pub struct HomeState {
    pub selected: usize, // 0 = Load, 1 = Build
}

pub struct LoadPathState {
    pub models: Vec<SavedModelEntry>,
    pub selected: usize,
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

pub const TRAINING_PARAM_FIELD_NAMES: [&str; 4] =
    ["Learning Rate", "Batch Size", "Steps", "Dataset Path"];

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
}

// ---------------------------------------------------------------------------
// Save Model screen state
// ---------------------------------------------------------------------------

pub struct SaveModelState {
    pub name: String,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

pub struct App {
    pub screen: Screen,
    pub home: HomeState,
    pub load_path: LoadPathState,
    pub input_size: InputSizeState,
    pub layer_builder: LayerBuilderState,
    pub mode_selector: ModeSelectorState,
    pub training_params: TrainingParamsState,
    pub monitor: MonitorState,
    pub save_model: SaveModelState,
    pub run_config: Option<RunConfig>,
    pub active_model_name: Option<String>,
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
        let models = storage::list_saved_models().unwrap_or_default();
        let datasets = storage::list_datasets().unwrap_or_default();
        let dataset_path = datasets.first().cloned().unwrap_or_default();
        let mut app = Self {
            screen: Screen::Home,
            home: HomeState { selected: 1 },
            load_path: LoadPathState {
                models,
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
            training_params: TrainingParamsState {
                fields: vec!["0.01".into(), "1".into(), "50".into(), dataset_path],
                field_idx: 0,
                error: None,
                datasets,
                selected_dataset: 0,
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
            },
            save_model: SaveModelState {
                name: String::new(),
                error: None,
            },
            run_config: None,
            active_model_name: None,
            should_quit: false,
        };
        app.sync_selected_dataset_from_field();
        app.reset_layer_form();
        app
    }

    fn refresh_saved_models(&mut self) {
        self.load_path.models = storage::list_saved_models().unwrap_or_default();
        if self.load_path.selected >= self.load_path.models.len() {
            self.load_path.selected = self.load_path.models.len().saturating_sub(1);
        }
        self.load_path.error = None;
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

    fn sync_selected_dataset_from_field(&mut self) {
        if let Some(index) = self
            .training_params
            .datasets
            .iter()
            .position(|dataset| dataset == &self.training_params.fields[3])
        {
            self.training_params.selected_dataset = index;
        }
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

        match config.run.mode {
            RunMode::Infer => {
                self.mode_selector.selected = 0;
            }
            RunMode::Train(train) => {
                self.mode_selector.selected = 1;
                self.training_params.fields = vec![
                    train.lr.to_string(),
                    train.batch_size.to_string(),
                    train.steps.to_string(),
                    train.dataset_path,
                ];
                self.sync_selected_dataset_from_field();
            }
        }

        self.screen = Screen::ModeSelector;
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
        self.active_model_name = None;
        self.layer_builder.error = None;
        self.reset_layer_form();
        Ok(())
    }

    pub fn delete_last_layer(&mut self) {
        self.layer_builder.layers.pop();
        self.active_model_name = None;
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
        self.active_model_name = None;
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
        self.active_model_name = None;
        self.rebuild_layer_dims_from(idx);
        if self.layer_builder.layers.is_empty() {
            self.exit_browse_mode();
        } else {
            self.layer_builder.browse_selected =
                idx.min(self.layer_builder.layers.len().saturating_sub(1));
        }
    }

    // --- Save model ---

    pub fn open_save_model(&mut self) {
        let default_name = storage::next_model_name().unwrap_or_else(|_| "model-001".to_string());
        self.save_model.name = default_name;
        self.save_model.error = None;
        self.screen = Screen::SaveModel;
    }

    /// Save the current model config with the name stored in `save_model.name`.
    /// Returns the saved path as a string on success.
    pub fn finish_save_model(&mut self) -> Result<String, String> {
        let mut config = self
            .monitor
            .model_config
            .as_ref()
            .ok_or_else(|| "No model config available".to_string())?
            .clone();
        let name = self.save_model.name.trim().to_string();
        if name.is_empty() {
            return Err("Name must not be empty".to_string());
        }
        if name.contains('/') || name.contains('\\') {
            return Err("Name must not contain path separators".to_string());
        }
        config.model_name = Some(name.clone());
        storage::save_model_config_named(&config, &name)
            .map_err(|err| err.to_string())
            .map(|path| {
                self.active_model_name = Some(name.clone());
                self.monitor.model_config = Some(config);
                path.display().to_string()
            })
    }

    pub fn handle_char_save_model(&mut self, c: char) {
        if !c.is_control() {
            self.save_model.name.push(c);
            self.save_model.error = None;
        }
    }

    pub fn handle_backspace_save_model(&mut self) {
        self.save_model.name.pop();
    }

    // --- Monitor restart ---

    pub fn request_restart(&mut self) {
        self.monitor.restart_training = true;
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
        self.screen = match self.home.selected {
            0 => {
                self.refresh_saved_models();
                Screen::LoadPath
            }
            _ => Screen::InputSize,
        };
    }

    pub fn finish_load_path(&mut self) {
        let Some(model) = self.load_path.models.get(self.load_path.selected) else {
            self.load_path.error = Some("No saved models found in saved_models/".into());
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
        self.active_model_name = None;
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
                self.run_config = Some(RunConfig {
                    mode: RunMode::Infer,
                })
            }
            _ => self.screen = Screen::TrainingParams,
        }
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
        let dataset_path = self.training_params.fields[3].clone();
        if lr <= 0.0 {
            return Err("Learning rate must be > 0".into());
        }
        if batch == 0 {
            return Err("Batch size must be > 0".into());
        }
        if steps == 0 {
            return Err("Steps must be > 0".into());
        }
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
            }),
        });
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
            3 => !c.is_control(),
            _ => false,
        };
        if accepted {
            self.training_params.fields[idx].push(c);
            self.training_params.error = None;
            if idx == 3 {
                self.sync_selected_dataset_from_field();
            }
        }
    }

    pub fn handle_backspace_training(&mut self) {
        let idx = self.training_params.field_idx;
        self.training_params.fields[idx].pop();
        if idx == 3 {
            self.sync_selected_dataset_from_field();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{App, LayerKind};

    #[test]
    fn cycle_kind_backward_moves_in_reverse_order() {
        let mut app = App::new();
        app.layer_builder.current_kind = LayerKind::Convolution;

        app.cycle_kind_backward();
        assert_eq!(app.layer_builder.current_kind, LayerKind::Concat);

        app.cycle_kind_backward();
        assert_eq!(app.layer_builder.current_kind, LayerKind::UpsampleConv);
    }
}
