use std::fmt;

// ---------------------------------------------------------------------------
// Mirror types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone)]
pub enum LayerDraft {
    Convolution {
        dim_input: (u32, u32, u32),
        nb_kernel: u32,
        dim_kernel: (u32, u32, u32),
        stride: u32,
        padding: PaddingMode,
    },
    Activation {
        dim_input: (u32, u32, u32),
        method: ActivationMethod,
    },
    FullyConnected {
        dim_input: (u32, u32, u32),
        nb_neurons: u32,
        method: ActivationMethod,
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

pub fn compute_inferred_input(
    layers: &[LayerDraft],
    model_input: (u32, u32, u32),
) -> (u32, u32, u32) {
    let mut current = model_input;
    for layer in layers {
        current = match layer {
            LayerDraft::Convolution {
                nb_kernel,
                dim_kernel,
                stride,
                padding,
                ..
            } => compute_out_conv(current, *dim_kernel, *stride, *nb_kernel, padding),
            LayerDraft::Activation { .. } => current,
            LayerDraft::FullyConnected { nb_neurons, .. } => (1, 1, *nb_neurons),
        };
    }
    current
}

impl LayerDraft {
    pub fn display(&self) -> String {
        match self {
            LayerDraft::Convolution {
                dim_input,
                nb_kernel,
                dim_kernel,
                stride,
                padding,
            } => {
                let (ow, oh, oz) =
                    compute_out_conv(*dim_input, *dim_kernel, *stride, *nb_kernel, padding);
                format!(
                    "Conv  {}x{}x{} -> {}x{}x{}",
                    dim_input.0, dim_input.1, dim_input.2, ow, oh, oz
                )
            }
            LayerDraft::Activation { dim_input, method } => {
                format!(
                    "{:<6} {}x{}x{}",
                    method.to_string(),
                    dim_input.0,
                    dim_input.1,
                    dim_input.2
                )
            }
            LayerDraft::FullyConnected {
                dim_input,
                nb_neurons,
                method,
            } => {
                format!(
                    "Perceptron({}) {}x{}x{} -> 1x1x{}",
                    method, dim_input.0, dim_input.1, dim_input.2, nb_neurons
                )
            }
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            LayerDraft::Convolution { .. } => "Conv",
            LayerDraft::Activation { .. } => "Activation",
            LayerDraft::FullyConnected { .. } => "Perceptron",
        }
    }

    pub fn input_dim_str(&self) -> String {
        let (x, y, z) = match self {
            LayerDraft::Convolution { dim_input, .. }
            | LayerDraft::Activation { dim_input, .. }
            | LayerDraft::FullyConnected { dim_input, .. } => *dim_input,
        };
        format!("{}x{}x{}", x, y, z)
    }

    pub fn output_dim_str(&self) -> String {
        match self {
            LayerDraft::Convolution {
                dim_input,
                nb_kernel,
                dim_kernel,
                stride,
                padding,
            } => {
                let (ow, oh, oz) =
                    compute_out_conv(*dim_input, *dim_kernel, *stride, *nb_kernel, padding);
                format!("{}x{}x{}", ow, oh, oz)
            }
            LayerDraft::Activation { dim_input, .. } => {
                format!("{}x{}x{}", dim_input.0, dim_input.1, dim_input.2)
            }
            LayerDraft::FullyConnected { nb_neurons, .. } => {
                format!("1x1x{}", nb_neurons)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LayerKind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Convolution,
    Activation,
    FullyConnected,
}

impl fmt::Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerKind::Convolution => write!(f, "Conv"),
            LayerKind::Activation => write!(f, "Activ"),
            LayerKind::FullyConnected => write!(f, "Perceptron"),
        }
    }
}

// ---------------------------------------------------------------------------
// Run / model configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub lr: f32,
    pub batch_size: u32,
    pub steps: usize,
    pub dataset_path: String,
    pub loss: LossMethod,
}

#[derive(Debug, Clone)]
pub enum RunMode {
    Infer,
    Train(TrainingConfig),
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub mode: RunMode,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
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
}

// ---------------------------------------------------------------------------
// Per-screen state
// ---------------------------------------------------------------------------

pub struct HomeState {
    pub selected: usize, // 0 = Load, 1 = Build
}

pub struct LoadPathState {
    pub path: String,
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
}

pub struct ModeSelectorState {
    pub selected: usize, // 0 = Infer, 1 = Train
}

pub struct TrainingParamsState {
    pub fields: Vec<String>, // [lr, batch_size, steps, dataset_path]
    pub field_idx: usize,
    pub error: Option<String>,
}

pub const TRAINING_PARAM_FIELD_NAMES: [&str; 4] =
    ["Learning Rate", "Batch Size", "Steps", "Dataset Path"];

pub struct MonitorState {
    pub step: usize,
    pub loss_history: Vec<f64>,
    pub done: bool,
    pub total_steps: usize,
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
    pub run_config: Option<RunConfig>,
    pub should_quit: bool,
}

// Field helpers

fn conv_field_names() -> Vec<&'static str> {
    vec!["Num Kernels", "Kernel W", "Kernel H", "Stride", "Padding"]
}

fn conv_field_defaults() -> Vec<String> {
    vec![
        "4".into(),
        "3".into(),
        "3".into(),
        "1".into(),
        "Valid".into(),
    ]
}

fn activation_field_names() -> Vec<&'static str> {
    vec!["Method"]
}

fn activation_field_defaults() -> Vec<String> {
    vec!["ReLU".into()]
}

fn fully_connected_field_names() -> Vec<&'static str> {
    vec!["Neurons", "Method"]
}

fn fully_connected_field_defaults() -> Vec<String> {
    vec!["10".into(), "ReLU".into()]
}

fn is_toggle_field(name: &str) -> bool {
    matches!(name, "Padding" | "Method")
}

impl App {
    pub fn new() -> Self {
        let mut app = Self {
            screen: Screen::Home,
            home: HomeState { selected: 1 },
            load_path: LoadPathState {
                path: String::new(),
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
            },
            mode_selector: ModeSelectorState { selected: 1 },
            training_params: TrainingParamsState {
                fields: vec!["0.01".into(), "1".into(), "50".into(), "".into()],
                field_idx: 0,
                error: None,
            },
            monitor: MonitorState {
                step: 0,
                loss_history: Vec::new(),
                done: false,
                total_steps: 0,
            },
            run_config: None,
            should_quit: false,
        };
        app.reset_layer_form();
        app
    }

    pub fn layer_field_names(&self) -> Vec<&'static str> {
        match self.layer_builder.current_kind {
            LayerKind::Convolution => conv_field_names(),
            LayerKind::Activation => activation_field_names(),
            LayerKind::FullyConnected => fully_connected_field_names(),
        }
    }

    pub fn reset_layer_form(&mut self) {
        self.layer_builder.fields = match self.layer_builder.current_kind {
            LayerKind::Convolution => conv_field_defaults(),
            LayerKind::Activation => activation_field_defaults(),
            LayerKind::FullyConnected => fully_connected_field_defaults(),
        };
        self.layer_builder.field_idx = 0;
        self.layer_builder.error = None;
    }

    pub fn inferred_input(&self) -> (u32, u32, u32) {
        compute_inferred_input(&self.layer_builder.layers, self.layer_builder.model_input)
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
            LayerKind::Activation => Some(inferred),
            LayerKind::FullyConnected => {
                let names = self.layer_field_names();
                let idx = names.iter().position(|&n| n == "Neurons")?;
                let nb_neurons = lb.fields.get(idx)?.parse().ok()?;
                Some((1, 1, nb_neurons))
            }
        }
    }

    pub fn try_add_layer(&mut self) -> Result<(), String> {
        let names = self.layer_field_names();
        let fields = self.layer_builder.fields.clone();
        let inferred = self.inferred_input();

        let parse_u32 = |name: &str| -> Result<u32, String> {
            let idx = names
                .iter()
                .position(|&n| n == name)
                .expect("field name mismatch");
            fields[idx]
                .parse::<u32>()
                .map_err(|_| format!("'{name}' must be a positive integer"))
        };

        let draft = match self.layer_builder.current_kind {
            LayerKind::Convolution => {
                let nb_kernel = parse_u32("Num Kernels")?;
                let kw = parse_u32("Kernel W")?;
                let kh = parse_u32("Kernel H")?;
                let kc = inferred.2; // always inferred from input depth
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
                LayerDraft::Convolution {
                    dim_input: inferred,
                    nb_kernel,
                    dim_kernel: (kw, kh, kc),
                    stride,
                    padding,
                }
            }
            LayerKind::Activation => {
                let method = ActivationMethod::from_label(&fields[0])
                    .ok_or_else(|| format!("Unknown activation method '{}'", fields[0]))?;
                LayerDraft::Activation {
                    dim_input: inferred,
                    method,
                }
            }
            LayerKind::FullyConnected => {
                let nb_neurons = parse_u32("Neurons")?;
                if nb_neurons == 0 {
                    return Err("Neurons must be > 0".into());
                }
                let method_idx = names.iter().position(|&n| n == "Method").unwrap();
                let method = ActivationMethod::from_label(&fields[method_idx])
                    .ok_or_else(|| format!("Unknown activation method '{}'", fields[method_idx]))?;
                LayerDraft::FullyConnected {
                    dim_input: inferred,
                    nb_neurons,
                    method,
                }
            }
        };

        self.layer_builder.layers.push(draft);
        self.layer_builder.error = None;
        self.reset_layer_form();
        Ok(())
    }

    pub fn delete_last_layer(&mut self) {
        self.layer_builder.layers.pop();
        self.reset_layer_form();
    }

    pub fn cycle_kind_forward(&mut self) {
        self.layer_builder.current_kind = match self.layer_builder.current_kind {
            LayerKind::Convolution => LayerKind::Activation,
            LayerKind::Activation => LayerKind::FullyConnected,
            LayerKind::FullyConnected => LayerKind::Convolution,
        };
        self.reset_layer_form();
    }

    pub fn cycle_kind_backward(&mut self) {
        self.cycle_kind_forward();
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
        if c.is_ascii_digit() {
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
            0 => Screen::LoadPath,
            _ => Screen::InputSize,
        };
    }

    pub fn finish_load_path(&mut self) {
        // Load not yet implemented — proceed to InputSize
        self.screen = Screen::InputSize;
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
        self.layer_builder.model_input = (w, h, c);
        self.layer_builder.layers.clear();
        self.reset_layer_form();
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
        self.monitor.total_steps = steps;
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
        }
    }

    pub fn handle_backspace_training(&mut self) {
        let idx = self.training_params.field_idx;
        self.training_params.fields[idx].pop();
    }

    pub fn handle_char_load_path(&mut self, c: char) {
        if !c.is_control() {
            self.load_path.path.push(c);
        }
    }

    pub fn handle_backspace_load_path(&mut self) {
        self.load_path.path.pop();
    }
}
