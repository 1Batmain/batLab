//! File purpose: Terminal UI state machine — the screens, the forms, and the
//! keyboard-driven model builder.
//!
//! The serialisable half of what used to live here (the model/run schema) is
//! `batlab_core::config`: it is engine-side, because a wasm build needs to
//! describe a model without a terminal anywhere in sight. What is left here is
//! genuinely terminal state.

use crate::storage::{self, SavedModelEntry, Storage, default_seed_dataset_name};
pub use batlab_core::config::*;
use batlab_core::model::training::{LossWeighting, MIN_RENOISE_DEPTH, PerpetualRegime};
use batlab_core::model::{OptimizerKind, WeightInit};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Screens
// ---------------------------------------------------------------------------

/// The screens of the terminal UI.
///
/// The flow is model-first: you open batlab onto [`Screen::ModelList`], you pick
/// a model, and only then do you pick what to do with it
/// ([`Screen::ModelActions`]). Every screen below has exactly one parent, and
/// `Esc` walks back to it — `nav_tests.rs` holds the executable copy of that
/// claim, because a screen that is drawn and handled but never *assigned* is
/// this codebase's recurring bug (`docs/reports/PERPETUAL_INFERENCE.md` §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Screen {
    ModelList,
    TemplateSelector,
    ModelActions,
    RenameModel,
    DuplicateModel,
    DeleteConfirm,
    WeightSelector,
    InputSize,
    LayerBuilder,
    InferenceParams,
    PerpetualParams,
    TrainingParams,
    DatasetSelector,
    /// The dataset a perpetual drift sets out from — a chooser, not a typed
    /// path, so the geometry of each candidate can be shown next to it and an
    /// incompatible one refused before the run rather than resized in silence.
    SeedDatasetSelector,
    Monitor,
    TrainingControl,
    /// What this model costs on the GPU — see [`ResourcesState`].
    Resources,
}

/// The flow, as the five steps a run actually walks through.
///
/// The screens are the implementation; this is the *path*, and it is what the
/// breadcrumb draws at the bottom of the terminal. Several screens map to one
/// step — the four parameter forms are all "Parameters", because from the
/// user's side there is one step there whatever the run mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathStep {
    Model,
    Action,
    Weights,
    Parameters,
    Run,
}

impl PathStep {
    /// The path, in order. The breadcrumb draws exactly this.
    pub const ALL: [PathStep; 5] = [
        PathStep::Model,
        PathStep::Action,
        PathStep::Weights,
        PathStep::Parameters,
        PathStep::Run,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            PathStep::Model => "Model",
            PathStep::Action => "Action",
            PathStep::Weights => "Weights",
            PathStep::Parameters => "Parameters",
            PathStep::Run => "Run",
        }
    }

    /// How far along the path this step sits.
    pub fn position(self) -> usize {
        Self::ALL
            .iter()
            .position(|step| *step == self)
            .expect("PathStep::ALL must list every step")
    }
}

impl Screen {
    /// Which step of the path this screen belongs to, or `None` for the screens
    /// that sit off it.
    ///
    /// The match is exhaustive on purpose: a new screen cannot be added without
    /// someone deciding whether it is a step of the flow or a detour. The
    /// detours are the manager forms, the architecture editor and its input
    /// geometry, and the training-control popup — reached *from* a step, but
    /// not a step, and drawing a breadcrumb on them would claim a position in
    /// the path that pressing `←` would not honour.
    pub const fn path_step(self) -> Option<PathStep> {
        match self {
            Screen::ModelList | Screen::TemplateSelector => Some(PathStep::Model),
            Screen::ModelActions => Some(PathStep::Action),
            Screen::WeightSelector => Some(PathStep::Weights),
            Screen::TrainingParams
            | Screen::DatasetSelector
            | Screen::InferenceParams
            | Screen::PerpetualParams
            | Screen::SeedDatasetSelector => Some(PathStep::Parameters),
            Screen::Monitor => Some(PathStep::Run),
            Screen::RenameModel
            | Screen::DuplicateModel
            | Screen::DeleteConfirm
            | Screen::InputSize
            | Screen::LayerBuilder
            | Screen::Resources
            | Screen::TrainingControl => None,
        }
    }

    /// Whether `←`/`→` walk the path on this screen.
    ///
    /// Only the pure selectors qualify. Everywhere else those two keys already
    /// mean something to a field or a cursor — the layer kind, the tempo of a
    /// perpetual run, the dataset, a seed toggle — and a breadcrumb that took
    /// them would break bindings people already use. The parameter forms are
    /// therefore *shown* on the path but not navigable by arrow: `Esc` remains
    /// the way back out of a form.
    pub const fn walks_the_path_by_arrow(self) -> bool {
        matches!(
            self,
            Screen::ModelList
                | Screen::TemplateSelector
                | Screen::ModelActions
                | Screen::WeightSelector
        )
    }

    /// Every screen there is. A test walks the whole flow and asserts it visited
    /// all of these, so a new variant stays failing until something actually
    /// routes to it.
    pub const ALL: [Screen; 17] = [
        Screen::ModelList,
        Screen::TemplateSelector,
        Screen::ModelActions,
        Screen::RenameModel,
        Screen::DuplicateModel,
        Screen::DeleteConfirm,
        Screen::WeightSelector,
        Screen::InputSize,
        Screen::LayerBuilder,
        Screen::InferenceParams,
        Screen::PerpetualParams,
        Screen::TrainingParams,
        Screen::DatasetSelector,
        Screen::SeedDatasetSelector,
        Screen::Monitor,
        Screen::TrainingControl,
        Screen::Resources,
    ];
}

// ---------------------------------------------------------------------------
// Per-screen state
// ---------------------------------------------------------------------------

/// The front door: every saved model, plus one row that opens the template
/// flow.
pub struct ModelListState {
    pub models: Vec<SavedModelEntry>,
    /// Index into the rows. `models.len()` is the "new model" row — it moves
    /// with the list rather than sitting at a fixed index, so deleting the last
    /// model cannot strand the cursor past the end.
    pub selected: usize,
    pub error: Option<String>,
    /// What the last manager operation did. Kept on screen so a rename or a
    /// delete is acknowledged where its effect is visible.
    pub status: Option<String>,
}

/// Label of the row that leads to the template flow.
pub const NEW_MODEL_ENTRY: &str = "New model (from template)";

impl ModelListState {
    pub fn entry_count(&self) -> usize {
        self.models.len() + 1
    }

    pub fn is_new_model_selected(&self) -> bool {
        self.selected >= self.models.len()
    }

    pub fn selected_model(&self) -> Option<&SavedModelEntry> {
        self.models.get(self.selected)
    }
}

/// What can be done to the model that was just picked. The three run modes and
/// the three manager operations, in one menu — the model is chosen first, the
/// action second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelAction {
    Train,
    Infer,
    Perpetual,
    Rename,
    Duplicate,
    Delete,
}

/// The actions, in the order they are drawn. The key handler bounds its cursor
/// on this list, so adding an action here is enough to make it selectable.
///
/// `Delete` stays last: the cursor arriving there has to be a deliberate walk
/// past everything else.
pub const MODEL_ACTIONS: [&str; 6] = [
    "Train",
    "Infer",
    "Perpetual",
    "Rename",
    "Duplicate",
    "Delete",
];

impl ModelAction {
    pub const fn index(self) -> usize {
        match self {
            ModelAction::Train => 0,
            ModelAction::Infer => 1,
            ModelAction::Perpetual => 2,
            ModelAction::Rename => 3,
            ModelAction::Duplicate => 4,
            ModelAction::Delete => 5,
        }
    }

    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(ModelAction::Train),
            1 => Some(ModelAction::Infer),
            2 => Some(ModelAction::Perpetual),
            3 => Some(ModelAction::Rename),
            4 => Some(ModelAction::Duplicate),
            5 => Some(ModelAction::Delete),
            _ => None,
        }
    }

    /// Whether the action starts a run (as opposed to managing the model).
    pub const fn is_run(self) -> bool {
        matches!(
            self,
            ModelAction::Train | ModelAction::Infer | ModelAction::Perpetual
        )
    }
}

pub struct ModelActionsState {
    pub selected: usize,
    pub error: Option<String>,
}

pub struct RenameModelState {
    pub input: String,
    pub error: Option<String>,
}

/// What a duplicate carries, in the order the two rows are drawn.
///
/// Weights first, and selected by default: duplicating exists so a foundation
/// model can be fine-tuned under another name, and a fine-tune with no weights
/// to start from is a new model, not a copy.
pub const DUPLICATE_CONTENT_CHOICES: [&str; 2] = [
    "config + weights — fine-tune the copy, the original keeps its own",
    "config only     — same architecture, fresh weights",
];

pub struct DuplicateModelState {
    /// The copy's name. Opens on [`Storage::suggest_copy_name`], so the common
    /// case is Enter.
    pub input: String,
    /// Cursor over [`DUPLICATE_CONTENT_CHOICES`].
    pub selected: usize,
    pub error: Option<String>,
}

impl DuplicateModelState {
    pub fn copies_weights(&self) -> bool {
        self.selected == 0
    }
}

/// What the model list says a duplication just did.
///
/// It names the files, and it does not round up: asking for weights from a
/// model that has none copies the config alone, and the line says *that* rather
/// than leaving the user to discover an empty `pretrained_weights/` two screens
/// later.
pub fn duplicate_status(from: &str, to: &str, outcome: &storage::DuplicateOutcome) -> String {
    if outcome.copied_weights.is_empty() {
        format!("Duplicated '{from}' → '{to}' — config only, no weights copied")
    } else {
        format!(
            "Duplicated '{from}' → '{to}' — config + {}",
            outcome.copied_weights.join(", ")
        )
    }
}

/// Deleting is irreversible, so the confirmation is not a keystroke: the model's
/// name has to be typed back exactly. A `[y]` on a menu is one fat finger away
/// from a lost training run.
pub struct DeleteConfirmState {
    pub typed: String,
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

/// The checkpoint a model is continued from when nothing more specific was
/// asked for. It is the file every training run writes, so "open a model and
/// train" means "keep training the model", not "throw the weights away".
pub const PREFERRED_CHECKPOINT_NAME: &str = storage::LATEST_CHECKPOINT_NAME;

impl WeightSelectorState {
    /// The row the cursor should sit on given the checkpoints on disk and the
    /// path already chosen, if any.
    ///
    /// Row 0 is "start from random weights" and it is now the *fallback*, not
    /// the default: a model with weights opens ready to continue from them.
    /// Row `i + 1` is `checkpoints[i]`.
    pub fn preferred_row(&self, chosen: Option<&str>) -> usize {
        if let Some(index) = chosen.and_then(|path| {
            self.checkpoints
                .iter()
                .position(|entry| entry.path == path)
        }) {
            return index + 1;
        }
        if let Some(index) = self
            .checkpoints
            .iter()
            .position(|entry| entry.name == PREFERRED_CHECKPOINT_NAME)
        {
            return index + 1;
        }
        if self.checkpoints.is_empty() { 0 } else { 1 }
    }

    /// The checkpoint currently under the cursor, or `None` on row 0.
    pub fn selected_checkpoint(&self) -> Option<&storage::CheckpointEntry> {
        self.selected
            .checked_sub(1)
            .and_then(|index| self.checkpoints.get(index))
    }
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

/// The Resources page: what this model puts on the GPU, and whether it fits.
///
/// The page carries almost no state, and that is deliberate — the inventory is
/// recomputed from the architecture on every draw (a walk over ~30 layers, well
/// under a millisecond) rather than cached. A cached inventory is a number that
/// can go stale behind an architecture edit, which is exactly the class of bug
/// the weight selector's derived cursor exists to avoid.
///
/// What *is* state is the question being asked: at which batch, and about which
/// workload. `←`/`→` move the batch, which is the whole simulator: the answer to
/// "what would this cost on a bigger machine" is a number that moves.
pub struct ResourcesState {
    /// The batch the page is sizing for. Starts at the model's own.
    pub batch: u32,
    /// Whether the page is sizing inference instead of training.
    pub inference: bool,
    /// First line drawn, for `↑`/`↓` on a page longer than the terminal.
    pub scroll: usize,
}

impl ResourcesState {
    /// Batches the `←`/`→` keys step through.
    ///
    /// Powers of two rather than `+1`: nobody trains at batch 37, the
    /// interesting range spans two orders of magnitude, and one keypress per
    /// sample would make the top of the range unreachable. Same reasoning as
    /// the perpetual tempo dial, which is multiplicative for the same reason.
    pub const STOPS: [u32; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    /// The next stop above `batch`, or `batch` itself at the top.
    pub fn larger(batch: u32) -> u32 {
        Self::STOPS
            .iter()
            .copied()
            .find(|stop| *stop > batch)
            .unwrap_or(batch)
    }

    /// The next stop below `batch`, or `batch` itself at the bottom.
    pub fn smaller(batch: u32) -> u32 {
        Self::STOPS
            .iter()
            .copied()
            .rev()
            .find(|stop| *stop < batch)
            .unwrap_or(batch)
    }
}

pub struct TrainingParamsState {
    /// `[lr, batch_size, steps, ema_decay, dataset_path]`.
    ///
    /// The first four are indexed by `field_idx` directly — one row of the form,
    /// one slot. The dataset trails behind them with no row of its own: the
    /// dataset *selector* is a screen of its own, and this slot is only where
    /// its answer is parked.
    pub fields: Vec<String>,
    pub field_idx: usize,
    pub error: Option<String>,
    pub datasets: Vec<String>,
    pub selected_dataset: usize,
}

/// The training form, in the order it is walked.
///
/// The last entry is a toggle, not a typed value: it is the explicit opt-out
/// from the pretrained weights the flow now defaults to. It sits last because
/// it is the rare choice — the common case is "keep training this model", and
/// the common case should be the one you can reach by pressing Enter.
pub const TRAINING_PARAM_FIELD_NAMES: [&str; 5] = [
    "Learning Rate",
    "Batch Size",
    "Steps",
    "EMA decay",
    "Start from random",
];

/// Index of the EMA decay entry, in the form AND in `fields` — the two agree
/// for every typed row, which is what stops the panel from explaining one
/// field while the keystrokes edit another.
pub const TRAINING_EMA_FIELD: usize = 3;

/// Index of the random-weights toggle inside [`TRAINING_PARAM_FIELD_NAMES`].
pub const TRAINING_RANDOM_WEIGHTS_FIELD: usize = 4;

/// What the EMA row shows for a configured decay. **Empty means off** — the
/// field is the presence of an average, not a number that is always there.
pub fn ema_decay_field(decay: Option<f32>) -> String {
    decay.map(|d| d.to_string()).unwrap_or_default()
}

/// Read the EMA row back: blank (or whitespace) is `None`, anything else has to
/// be a decay strictly inside `(0, 1)`.
///
/// `1.0` freezes the average on the initial weights for ever and `0.0` makes it
/// a copy of the weights — both are silently useless rather than loudly wrong,
/// which is exactly the setting that costs a night of GPU before anyone looks.
pub fn parse_ema_decay_field(raw: &str) -> Result<Option<f32>, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    let decay: f32 = raw
        .parse()
        .map_err(|_| "EMA decay must be a number, or empty for no average".to_string())?;
    if !(decay.is_finite() && decay > 0.0 && decay < 1.0) {
        return Err("EMA decay must be strictly between 0 and 1 (e.g. 0.999)".to_string());
    }
    Ok(Some(decay))
}

/// Where the dataset path is parked inside `TrainingParamsState::fields`.
///
/// It is NOT a form row: it shares its number with the toggle by accident of
/// arithmetic, not by design. Named because it used to be a bare `[3]` in
/// fifteen places, and inserting a row in front of it silently repointed every
/// one of them at the EMA field.
pub const TRAINING_DATASET_FIELD: usize = 4;

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
    /// 0=random toggle, 1=seed, 2=magnitude, 3=depth, 4=tempo,
    /// 5=seed dataset (opens the chooser), 6=regime toggle
    pub field_idx: usize,
    pub error: Option<String>,
}

pub const PERPETUAL_PARAM_FIELD_NAMES: [&str; 7] = [
    "Random Seed",
    "Seed",
    "Magnitude",
    "Renoise Depth (t_r)",
    "Steps / second",
    "Seed Dataset",
    "Regime",
];

/// The row that opens [`Screen::SeedDatasetSelector`]. Not typed: a path is
/// long, and the chooser can show each candidate's geometry — which is the
/// whole point, since a mismatched one is refused rather than resized.
pub const PERPETUAL_SEED_DATASET_FIELD: usize = 5;

/// The chooser behind the "Seed Dataset" row.
///
/// Row 0 is always "(défaut — selon les canaux de sortie)", which stores `None`
/// and hands the choice back to the convention. The rest are the files
/// `datasets/` holds, each with the geometry read from its header.
#[derive(Default)]
pub struct SeedDatasetSelectorState {
    /// `(path, geometry)` — geometry `None` when the header could not be read
    /// (a directory of images, say), which is not a reason to hide the entry.
    pub datasets: Vec<(String, Option<(u32, u32, u32)>)>,
    /// 0 = the default row; `index - 1` indexes `datasets`.
    pub selected: usize,
    pub error: Option<String>,
}

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
    /// Which way the run is going — "descente" or "remontée". Half of a cycle
    /// is spent dissolving the image on purpose, and the panel has to say so.
    pub phase: String,
    pub depth: usize,
    /// What the depth dial is called in this regime — `t_r` where it is a
    /// renoise depth, `t*` in flux where it is the level the run lives on.
    /// Published by the worker rather than derived in the UI, so the name and
    /// the number on screen cannot disagree about which regime is running.
    pub depth_label: String,
    pub min_depth: usize,
    pub max_depth: usize,
    pub cycle: usize,
    /// What `cycle` counts in this regime — cycles, or stationary frames in
    /// flux, which has none.
    pub cycle_label: String,
    pub diffusion_step: usize,
    /// Reverse steps walked since the run started.
    pub steps: usize,
    /// Measured pace, as opposed to the requested `tempo`.
    pub steps_per_sec: f32,
    /// What the `[v]` window is showing — x̂₀ alone, or both panes. Published by
    /// the worker, which owns the `LiveFrame`, so the legend and the window
    /// cannot disagree about which layout is up.
    pub view: String,
    /// What the run set out from: a dataset image, or pure noise when none
    /// could be found. Worth saying, because a run that silently fell back on
    /// noise looks exactly like one that was asked to.
    pub origin: String,
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

/// Open an adapter just to read its limits, and let it go.
///
/// The Resources page is about *this* machine, so the limits have to come off
/// the machine. Nothing else here needs the device — the context is dropped as
/// soon as its numbers are read, and a run creates its own later.
///
/// A machine with no adapter is not an error: the page falls back on the WebGPU
/// default limits and says the profile is hypothetical. Someone reading a config
/// on a headless box still deserves an answer.
fn probe_device_profile() -> batlab_core::DeviceProfile {
    let context = std::panic::catch_unwind(|| {
        pollster::block_on(batlab_core::GpuContext::new_headless())
    });
    match context {
        Ok(gpu) => batlab_core::DeviceProfile::from_gpu(&gpu),
        Err(_) => batlab_core::DeviceProfile::hypothetical(
            "no adapter available — WebGPU default limits",
            None,
        ),
    }
}

pub struct App {
    pub screen: Screen,
    /// Where `Models/` and `datasets/` live for this session. Injected rather
    /// than deduced, so a test — or a throwaway `BATLAB_ROOT` session — can
    /// rename and delete models without any of it landing in the repository.
    pub storage: Storage,
    pub model_list: ModelListState,
    pub template_selector: TemplateSelectorState,
    pub model_actions: ModelActionsState,
    pub rename_model: RenameModelState,
    pub duplicate_model: DuplicateModelState,
    pub delete_confirm: DeleteConfirmState,
    pub weight_selector: WeightSelectorState,
    pub input_size: InputSizeState,
    pub layer_builder: LayerBuilderState,
    pub inference_params: InferenceParamsState,
    pub perpetual_params: PerpetualParamsState,
    pub seed_dataset_selector: SeedDatasetSelectorState,
    pub training_params: TrainingParamsState,
    pub training_control: TrainingControlState,
    pub monitor: MonitorState,
    pub resources: ResourcesState,
    /// The device the Resources page judges this model against.
    ///
    /// Injected for the same reason [`Storage`] is: the default probes a real
    /// adapter, and a test that walked the flow would then need a GPU to press
    /// a key. `App::with_device_profile` hands it a stated one instead.
    pub device_profile: batlab_core::DeviceProfile,
    /// The dataset the page accounts for — resolved from the model's training
    /// config when one is loaded, `None` otherwise.
    pub resources_dataset: Option<batlab_core::DatasetSpec>,
    /// The optimiser and the weight average the model's config asks for. They
    /// triple the parameter posts between them, so the page reads them from the
    /// file rather than assuming.
    pub resources_optimizer: batlab_core::OptimizerKind,
    /// The three training knobs the model's `config_file` carries and no form
    /// field shows. Held so that launching a run **preserves** them: they are
    /// read back into the `TrainingConfig` the run is given, rather than reset
    /// to their defaults on the way past.
    pub training_optimizer: batlab_core::OptimizerKind,
    pub training_weight_init: batlab_core::WeightInit,
    pub training_loss_weighting: LossWeighting,
    pub resources_ema: bool,
    pub run_config: Option<RunConfig>,
    /// The model's own seed dataset, as its `config_file` holds it — a property
    /// of the model, not of one run, which is why it lives here beside
    /// `active_model_name` and not in `perpetual_params`. Written back into
    /// every `ModelConfig` this TUI builds, so a training run does not erase
    /// what a perpetual run set. `None` = fall back on the channel convention.
    pub seed_dataset: Option<String>,
    pub active_model_name: Option<String>,
    /// The model this process currently has a run on, if any. The manager
    /// refuses to rename or delete it — see [`App::model_run_in_progress`] for
    /// what that does and does not cover.
    pub running_model: Option<String>,
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

/// Attention has no shape parameter of its own: the sequence is the spatial
/// grid and the feature dimension the channel count, both inherited.
fn attention_field_names() -> Vec<&'static str> {
    vec!["Save As"]
}

fn attention_field_defaults() -> Vec<String> {
    vec!["".into()]
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

// Add takes the same two fields as Concat — a skip to fetch and an optional key
// to save under. The difference is in what it does with the skip, not in what it
// asks the builder for.
fn add_field_names() -> Vec<&'static str> {
    vec!["Skip Key", "Save As"]
}

fn add_field_defaults() -> Vec<String> {
    vec!["".into(), "".into()]
}

// TimeBias reads the timestep embedding from a saved tensor (Time Key) and
// projects channels [Embed Offset, Embed Offset + Embed Channels) into a
// per-channel bias.
fn time_bias_field_names() -> Vec<&'static str> {
    vec!["Time Key", "Embed Offset", "Embed Channels", "Save As"]
}

fn time_bias_field_defaults() -> Vec<String> {
    vec!["".into(), "1".into(), "3".into(), "".into()]
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
    /// An app on the default storage root (the workspace, or `BATLAB_ROOT`).
    pub fn new() -> Self {
        let mut app = Self::with_storage(Storage::default());
        // Probed once, at startup, and only on the real entry point: the limits
        // do not change while the process runs, and `with_storage` — the
        // constructor the tests use — must not need an adapter to press a key.
        app.device_profile = probe_device_profile();
        app
    }

    /// An app that judges the Resources page against `profile` instead of
    /// probing an adapter.
    ///
    /// Two uses, and the second is why it exists: answering "would this fit on
    /// an 8 GiB card?" about a machine that is not here, and letting the
    /// navigation tests press every key without a GPU.
    pub fn with_device_profile(
        storage: Storage,
        profile: batlab_core::DeviceProfile,
    ) -> Self {
        let mut app = Self::with_storage(storage);
        app.device_profile = profile;
        app
    }

    /// An app on an explicit storage root. This is the constructor tests use:
    /// the flow writes, renames and deletes model directories, and none of that
    /// belongs in the repository's own `Models/`.
    pub fn with_storage(storage: Storage) -> Self {
        let models = storage.list_models().unwrap_or_default();
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
        let datasets = storage.list_datasets().unwrap_or_default();
        let dataset_path = datasets.first().cloned().unwrap_or_default();
        let mut app = Self {
            // The model list is the front door: open batlab, choose a model,
            // then choose what to do with it. Opening on the template selector
            // (as this once did) made every saved model unreachable, and
            // opening on a fork between "load" and "new" made the common case
            // — I have models, show me them — cost a keystroke and a decision.
            screen: Screen::ModelList,
            storage,
            model_list: ModelListState {
                models,
                selected: 0,
                error: None,
                status: None,
            },
            template_selector: TemplateSelectorState {
                templates,
                selected: 0,
                error: None,
            },
            model_actions: ModelActionsState {
                selected: ModelAction::Train.index(),
                error: None,
            },
            rename_model: RenameModelState {
                input: String::new(),
                error: None,
            },
            duplicate_model: DuplicateModelState {
                input: String::new(),
                selected: 0,
                error: None,
            },
            delete_confirm: DeleteConfirmState {
                typed: String::new(),
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
            seed_dataset_selector: SeedDatasetSelectorState::default(),
            training_params: TrainingParamsState {
                fields: vec![
                    default_lr.to_string(),
                    default_batch.to_string(),
                    default_steps.to_string(),
                    // Empty means no averaging — the default, and the run this
                    // binary did before averaging existed.
                    String::new(),
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
            resources: ResourcesState {
                batch: default_batch.max(1),
                inference: false,
                scroll: 0,
            },
            // Probed once, here, rather than on every draw: opening an adapter
            // costs tens of milliseconds and its limits do not change while the
            // process runs. A machine with no adapter at all falls back on the
            // WebGPU defaults, which is the honest answer — those are the limits
            // a browser would grant — and the page says the profile is
            // hypothetical.
            // Until `App::new` probes a real one. On a machine with no adapter
            // this stays, and it is the honest answer rather than a blank: the
            // WebGPU defaults are the limits a browser grants the visitor's GPU,
            // and the page says the profile is hypothetical.
            device_profile: batlab_core::DeviceProfile::hypothetical(
                "no adapter probed — WebGPU default limits",
                None,
            ),
            resources_dataset: None,
            resources_optimizer: batlab_core::OptimizerKind::default(),
            training_optimizer: batlab_core::OptimizerKind::default(),
            training_weight_init: batlab_core::WeightInit::default(),
            training_loss_weighting: LossWeighting::default(),
            resources_ema: false,
            run_config: None,
            seed_dataset: None,
            active_model_name: None,
            running_model: None,
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

    /// Re-scans `Models/` so the list reflects the disk, not the snapshot taken
    /// when the app was constructed — a rename or a delete has to be visible
    /// the moment it lands. Leaves `error`/`status` alone: the caller owns what
    /// the screen is saying.
    pub fn refresh_model_list(&mut self) {
        self.model_list.models = self.storage.list_models().unwrap_or_default();
        let last_row = self.model_list.entry_count() - 1;
        if self.model_list.selected > last_row {
            self.model_list.selected = last_row;
        }
    }

    /// Puts the cursor on a model by name, if it is still there.
    fn select_model_in_list(&mut self, name: &str) {
        if let Some(index) = self
            .model_list
            .models
            .iter()
            .position(|model| model.name == name)
        {
            self.model_list.selected = index;
        }
    }

    fn refresh_templates(&mut self) {
        self.template_selector.templates = built_in_templates();
        if self.template_selector.selected >= self.template_selector.templates.len() {
            self.template_selector.selected =
                self.template_selector.templates.len().saturating_sub(1);
        }
        self.template_selector.error = None;
    }

    /// Re-reads the model's checkpoints and puts the cursor back where the
    /// current choice says it belongs.
    ///
    /// The cursor is *derived*, never remembered: `load_checkpoint_on_start`
    /// plus `selected_checkpoint_path` are the single source of truth, and every
    /// path that invalidates the weights (editing a layer, changing the input
    /// geometry) clears them. Deriving is what keeps a stale row from
    /// re-selecting a checkpoint that no longer matches the architecture.
    fn refresh_weight_selector(&mut self) {
        let Some(model_name) = self.active_model_name.clone() else {
            self.weight_selector.checkpoints.clear();
            self.weight_selector.selected = 0;
            self.weight_selector.error = Some("No model selected.".to_string());
            return;
        };
        match self.storage.list_model_checkpoints(&model_name) {
            Ok(checkpoints) => {
                self.weight_selector.checkpoints = checkpoints;
                self.weight_selector.selected = if self.load_checkpoint_on_start {
                    self.weight_selector
                        .preferred_row(self.selected_checkpoint_path.as_deref())
                } else {
                    0
                };
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

    /// Whether the open model has any weights to continue from.
    ///
    /// The two screens that offer the choice both need it: the weight selector
    /// says so where the list would otherwise just be empty, and the training
    /// form uses it to pin its toggle on — "start from random" is not a choice
    /// when it is the only option, and a form that pretends otherwise is
    /// lying.
    pub fn has_pretrained_weights(&self) -> bool {
        !self.weight_selector.checkpoints.is_empty()
    }

    /// Points the flow at the model's pretrained weights, which is what opening
    /// a model now means.
    ///
    /// `keep` is the path the model's own `config_file` recorded, honoured when
    /// it still exists on disk; otherwise the preferred checkpoint
    /// ([`PREFERRED_CHECKPOINT_NAME`], else the first) is taken. With no
    /// checkpoints at all this falls back to random weights — the only case
    /// where it does.
    fn preselect_pretrained_weights(&mut self, keep: Option<String>) {
        self.load_checkpoint_on_start = false;
        self.selected_checkpoint_path = keep.clone();
        self.refresh_weight_selector();

        let row = self.weight_selector.preferred_row(keep.as_deref());
        self.weight_selector.selected = row;
        match self.weight_selector.selected_checkpoint() {
            Some(entry) => {
                self.selected_checkpoint_path = Some(entry.path.clone());
                self.load_checkpoint_on_start = true;
            }
            None => {
                // No weights yet: the run starts from random, and the
                // checkpoint path is where this run will *write*.
                self.selected_checkpoint_path = self.default_checkpoint_for_active_model();
                self.load_checkpoint_on_start = false;
            }
        }
    }

    /// The training form's toggle. Turning it off is only possible when there
    /// are weights to turn it off *to*.
    pub fn toggle_start_from_random(&mut self) {
        if !self.has_pretrained_weights() {
            return;
        }
        if self.load_checkpoint_on_start {
            self.load_checkpoint_on_start = false;
            self.selected_checkpoint_path = self.default_checkpoint_for_active_model();
        } else {
            self.preselect_pretrained_weights(None);
        }
        self.weight_selector.selected = if self.load_checkpoint_on_start {
            self.weight_selector
                .preferred_row(self.selected_checkpoint_path.as_deref())
        } else {
            0
        };
        self.training_params.error = None;
    }

    /// What the training form shows for its toggle: `true` means this run
    /// throws the weights away and starts over.
    pub fn start_from_random_weights(&self) -> bool {
        !self.load_checkpoint_on_start
    }

    fn refresh_datasets(&mut self) {
        self.training_params.datasets = self.storage.list_datasets().unwrap_or_default();
        if self.training_params.datasets.is_empty() {
            self.training_params.selected_dataset = 0;
            return;
        }
        if self.training_params.fields[TRAINING_DATASET_FIELD].trim().is_empty() {
            self.training_params.fields[TRAINING_DATASET_FIELD] = self.training_params.datasets[0].clone();
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
            .position(|dataset| dataset == &self.training_params.fields[TRAINING_DATASET_FIELD])
        {
            self.training_params.selected_dataset = index;
        }
    }

    /// The checkpoint a run defaults to for the model currently open —
    /// `Models/<name>/pretrained_weights/latest.ckpt`. `None` only when no model
    /// is open, or when the folder cannot be prepared.
    fn default_checkpoint_for_active_model(&self) -> Option<String> {
        let model_name = self.active_model_name.as_deref()?;
        self.storage
            .default_model_checkpoint_path(model_name)
            .ok()
            .map(|path| path.to_string_lossy().to_string())
    }

    fn apply_template(&mut self, template: &ModelTemplate) -> Result<(), String> {
        self.active_model_name = Some(template.key.clone());
        self.load_checkpoint_on_start = false;
        self.selected_checkpoint_path = Some(
            self.storage
                .default_model_checkpoint_path(&template.key)
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
            self.training_params.fields[TRAINING_DATASET_FIELD].clear();
            self.training_params.selected_dataset = 0;
        } else {
            self.training_params.selected_dataset = 0;
            self.training_params.fields[TRAINING_DATASET_FIELD] = self.training_params.datasets[0].clone();
        }

        self.model_actions.selected = ModelAction::Train.index();
        self.model_actions.error = None;
        let config = ModelConfig {
            model_name: Some(template.key.clone()),
            input_size: template.input_size,
            layers: template.layers.clone(),
            inference: InferenceConfig::default(),
            // A template names no dataset of its own: a fresh model falls back
            // on the channel convention until someone chooses otherwise.
            seed_dataset: None,
            run: RunConfig {
                mode: RunMode::Infer,
            },
        };
        self.seed_dataset = None;
        self.storage
            .write_model_config(&template.key, &config)
            .map_err(|err| format!("Failed to write template config_file: {err}"))?;
        self.refresh_weight_selector();
        self.weight_selector.selected = 0;
        // The template *created* a model; the flow rejoins the common path —
        // model in hand, now pick what to do with it.
        self.refresh_model_list();
        self.select_model_in_list(&template.key);
        self.screen = Screen::ModelActions;
        Ok(())
    }

    fn apply_loaded_model(&mut self, config: ModelConfig) {
        self.active_model_name = config.model_name.clone();
        // Reset before the match below fills them in, so a model whose config
        // is not a training config cannot inherit the previous model's
        // optimiser. Adam is the default for such a model rather than SGD:
        // `OPTIMIZER_ADAM.md` makes it the project's standard, and sizing an
        // Adam run as an SGD one under-reports the parameter posts by 3×.
        self.resources_optimizer = batlab_core::OptimizerKind::Adam;
        self.training_optimizer = batlab_core::OptimizerKind::default();
        self.training_weight_init = batlab_core::WeightInit::default();
        self.training_loss_weighting = LossWeighting::default();
        self.resources_ema = false;
        self.resources_dataset = None;
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
        // Unconditional, and it has to be: the seed dataset is a property of
        // the model, so it is read whatever `run.mode` the file last recorded.
        // Read only under `RunMode::Perpetual` it would have done two wrong
        // things at once — lost the setting the moment a training run rewrote
        // the mode, and left the *previous* model's dataset in place when the
        // one being opened has none.
        self.seed_dataset = config.seed_dataset.clone();

        // Whichever path the model's own `config_file` last recorded. It is a
        // preference, not a verdict: `preselect_pretrained_weights` honours it
        // only if that file is still on disk.
        let recorded = match config.run.mode {
            RunMode::Infer => {
                self.model_actions.selected = ModelAction::Infer.index();
                config.inference.checkpoint.clone()
            }
            RunMode::Train(train) => {
                self.model_actions.selected = ModelAction::Train.index();
                // What the Resources page needs from a training config, and
                // could not honestly guess: Adam doubles the parameter posts,
                // an EMA adds another copy, and the dataset is the one post
                // that is streamed rather than resident.
                self.resources_optimizer = train.optimizer;
                self.training_optimizer = train.optimizer;
                self.training_weight_init = train.weight_init;
                self.training_loss_weighting = train.loss_weighting;
                self.resources_ema = train.ema_decay.is_some();
                self.resources_dataset = self.storage.dataset_spec(&train.dataset_path);
                self.resources.batch = train.batch_size.max(1);
                self.training_params.fields = vec![
                    train.lr.to_string(),
                    train.batch_size.to_string(),
                    train.steps.to_string(),
                    ema_decay_field(train.ema_decay),
                    train.dataset_path,
                ];
                self.sync_selected_dataset_from_field();
                train.checkpoint_path.clone()
            }
            RunMode::Perpetual(perpetual) => {
                self.model_actions.selected = ModelAction::Perpetual.index();
                let checkpoint = perpetual.checkpoint.clone();
                self.sync_perpetual_params_from_config(&perpetual);
                checkpoint
            }
        };

        // The checkpoint list is prepared here even though the weight screen
        // comes later, so the action menu can say how many there are — and so
        // that opening a model already points at its weights.
        // Note what this path does *not* do — unlike `apply_template`, it never
        // calls `write_model_config`, so the model's own `inference` block
        // survives being opened.
        self.preselect_pretrained_weights(recorded);
        self.model_actions.error = None;
        self.screen = Screen::ModelActions;
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
        self.training_params.fields[TRAINING_DATASET_FIELD] =
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
        self.training_params.fields[TRAINING_DATASET_FIELD] =
            self.training_params.datasets[self.training_params.selected_dataset].clone();
        self.training_params.error = None;
    }

    pub fn select_dataset(&mut self, index: usize) {
        if self.training_params.datasets.is_empty() {
            self.training_params.selected_dataset = 0;
            self.training_params.fields[TRAINING_DATASET_FIELD].clear();
            return;
        }
        let clamped = index.min(self.training_params.datasets.len() - 1);
        self.training_params.selected_dataset = clamped;
        self.training_params.fields[TRAINING_DATASET_FIELD] = self.training_params.datasets[clamped].clone();
        self.training_params.error = None;
    }

    pub fn layer_field_names(&self) -> Vec<&'static str> {
        match self.layer_builder.current_kind {
            LayerKind::Convolution => conv_field_names(),
            LayerKind::GroupNorm => group_norm_field_names(),
            LayerKind::Attention => attention_field_names(),
            LayerKind::Activation => activation_field_names(),
            LayerKind::FullyConnected => fully_connected_field_names(),
            LayerKind::UpsampleConv => upsample_conv_field_names(),
            LayerKind::Concat => concat_field_names(),
            LayerKind::Add => add_field_names(),
            LayerKind::TimeBias => time_bias_field_names(),
        }
    }

    pub fn reset_layer_form(&mut self) {
        self.layer_builder.fields = match self.layer_builder.current_kind {
            LayerKind::Convolution => conv_field_defaults(),
            LayerKind::GroupNorm => group_norm_field_defaults(),
            LayerKind::Attention => attention_field_defaults(),
            LayerKind::Activation => activation_field_defaults(),
            LayerKind::FullyConnected => fully_connected_field_defaults(),
            LayerKind::UpsampleConv => upsample_conv_field_defaults(),
            LayerKind::Concat => concat_field_defaults(),
            LayerKind::Add => add_field_defaults(),
            LayerKind::TimeBias => time_bias_field_defaults(),
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
            // Shape-preserving: the preview is the inferred input itself.
            LayerKind::Attention => Some(inferred),
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
            LayerKind::Add => {
                let names = self.layer_field_names();
                let idx = names.iter().position(|&n| n == "Skip Key")?;
                let skip_key = normalize_key(lb.fields.get(idx)?)?;
                let dim_skip = self.saved_output_dims().get(&skip_key).copied()?;
                // Add is shape-preserving, but ONLY when both sides match on
                // every axis. A mismatch previews nothing rather than a wrong
                // shape — the builder refuses it at add time with a message.
                if dim_skip != inferred {
                    return None;
                }
                Some(inferred)
            }
            // Shape-preserving as long as the referenced tensor exists.
            LayerKind::TimeBias => {
                let names = self.layer_field_names();
                let idx = names.iter().position(|&n| n == "Time Key")?;
                let time_key = normalize_key(lb.fields.get(idx)?)?;
                self.saved_output_dims().get(&time_key)?;
                Some(inferred)
            }
        }
    }

    pub fn try_add_layer(&mut self) -> Result<(), String> {
        let inferred = self.inferred_input();
        let draft = self.build_draft_from_form(inferred, None)?;
        self.layer_builder.layers.push(draft);
        self.load_checkpoint_on_start = false;
        self.selected_checkpoint_path = self.default_checkpoint_for_active_model();
        self.layer_builder.error = None;
        self.reset_layer_form();
        Ok(())
    }

    pub fn delete_last_layer(&mut self) {
        self.layer_builder.layers.pop();
        self.load_checkpoint_on_start = false;
        self.selected_checkpoint_path = self.default_checkpoint_for_active_model();
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
            LayerKind::Attention => Ok(LayerDraft::Attention {
                dim_input: inferred,
                save_key: parse_save_key()?,
            }),
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
            LayerKind::Add => {
                let skip_idx = names.iter().position(|&n| n == "Skip Key").unwrap();
                let skip_key = normalize_key(&fields[skip_idx])
                    .ok_or_else(|| "Skip Key must not be empty".to_string())?;
                let dim_skip = existing_saved
                    .get(&skip_key)
                    .copied()
                    .ok_or_else(|| format!("Unknown skip key '{skip_key}'"))?;
                // The whole point of Add over Concat: it needs the same shape on
                // both sides. Refused here, naming both widths, rather than
                // corrupting a buffer at build (the UpsampleConv trap).
                if dim_skip != inferred {
                    return Err(format!(
                        "Add requires identical dims, got input {}x{}x{} and skip {}x{}x{} \
                         — put a 1x1 conv on the shortcut to realign the width",
                        inferred.0, inferred.1, inferred.2, dim_skip.0, dim_skip.1, dim_skip.2
                    ));
                }
                Ok(LayerDraft::Add {
                    dim_input: inferred,
                    dim_skip,
                    skip_key,
                    save_key: parse_save_key()?,
                })
            }
            LayerKind::TimeBias => {
                let key_idx = names.iter().position(|&n| n == "Time Key").unwrap();
                let time_key = normalize_key(&fields[key_idx])
                    .ok_or_else(|| "Time Key must not be empty".to_string())?;
                let time_dim = existing_saved
                    .get(&time_key)
                    .copied()
                    .ok_or_else(|| format!("Unknown time key '{time_key}'"))?;
                let embed_offset = parse_u32("Embed Offset")?;
                let embed_channels = parse_u32("Embed Channels")?;
                if embed_channels == 0 {
                    return Err("Embed Channels must be > 0".into());
                }
                if embed_offset + embed_channels > time_dim.2 {
                    return Err(format!(
                        "Embed Offset+Channels ({}) exceeds the {}-channel tensor '{time_key}'",
                        embed_offset + embed_channels,
                        time_dim.2
                    ));
                }
                Ok(LayerDraft::TimeBias {
                    dim_input: inferred,
                    time_key,
                    embed_offset,
                    embed_channels,
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
            LayerDraft::Attention { .. } => LayerKind::Attention,
            LayerDraft::FullyConnected { .. } => LayerKind::FullyConnected,
            LayerDraft::UpsampleConv { .. } => LayerKind::UpsampleConv,
            LayerDraft::Concat { .. } => LayerKind::Concat,
            LayerDraft::Add { .. } => LayerKind::Add,
            LayerDraft::TimeBias { .. } => LayerKind::TimeBias,
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
            LayerDraft::Attention { save_key, .. } => {
                vec![save_key.as_deref().unwrap_or("").to_string()]
            }
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
            LayerDraft::Add {
                skip_key, save_key, ..
            } => vec![
                skip_key.clone(),
                save_key.as_deref().unwrap_or("").to_string(),
            ],
            LayerDraft::TimeBias {
                time_key,
                embed_offset,
                embed_channels,
                save_key,
                ..
            } => vec![
                time_key.clone(),
                embed_offset.to_string(),
                embed_channels.to_string(),
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
        self.selected_checkpoint_path = self.default_checkpoint_for_active_model();
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
        self.selected_checkpoint_path = self.default_checkpoint_for_active_model();
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
                self.storage
                    .next_model_name()
                    .unwrap_or_else(|_| "model-001".to_string())
            });
        config.model_name = Some(model_name.clone());
        let config_path = self
            .storage
            .write_model_config(&model_name, &config)
            .map_err(|err| format!("failed to save model config: {err}"))?;
        self.active_model_name = Some(model_name.clone());
        self.selected_checkpoint_path = self.default_checkpoint_for_active_model();
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
            LayerKind::GroupNorm => LayerKind::Attention,
            LayerKind::Attention => LayerKind::Activation,
            LayerKind::Activation => LayerKind::FullyConnected,
            LayerKind::FullyConnected => LayerKind::UpsampleConv,
            LayerKind::UpsampleConv => LayerKind::Concat,
            LayerKind::Concat => LayerKind::Add,
            LayerKind::Add => LayerKind::TimeBias,
            LayerKind::TimeBias => LayerKind::Convolution,
        };
        self.reset_layer_form();
    }

    pub fn cycle_kind_backward(&mut self) {
        self.layer_builder.current_kind = match self.layer_builder.current_kind {
            LayerKind::Convolution => LayerKind::TimeBias,
            LayerKind::GroupNorm => LayerKind::Convolution,
            LayerKind::Attention => LayerKind::GroupNorm,
            LayerKind::Activation => LayerKind::Attention,
            LayerKind::FullyConnected => LayerKind::Activation,
            LayerKind::UpsampleConv => LayerKind::FullyConnected,
            LayerKind::Concat => LayerKind::UpsampleConv,
            LayerKind::Add => LayerKind::Concat,
            LayerKind::TimeBias => LayerKind::Add,
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

    /// Enter on the model list: open the selected model, or take the last row
    /// into the template flow.
    pub fn finish_model_list(&mut self) {
        self.model_list.status = None;
        if self.model_list.is_new_model_selected() {
            self.model_list.error = None;
            self.refresh_templates();
            self.screen = Screen::TemplateSelector;
            return;
        }
        let Some(model) = self.model_list.selected_model() else {
            self.model_list.error = Some("No model configs found in Models/".into());
            return;
        };
        let (name, path) = (model.name.clone(), model.path.clone());
        match storage::load_model_config(&path) {
            Ok(mut config) => {
                // The directory is the name. Rename keeps the two in step, and
                // if a hand-edited `config_file` disagrees the directory wins —
                // it is what the manager and every checkpoint path key off.
                config.model_name = Some(name);
                self.model_list.error = None;
                self.apply_loaded_model(config);
            }
            Err(err) => {
                self.model_list.error = Some(format!("Failed to load {name}: {err}"));
            }
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

    /// Opens the action menu on the model already in hand, re-reading its
    /// checkpoints from disk first.
    ///
    /// The host calls this on the restart path. Without the re-read the menu
    /// showed "no checkpoints" for a model that plainly had some, because
    /// `run_monitor` builds a fresh `App` and never fills the checkpoint list —
    /// found by driving the real TUI, where a run had just been saved.
    pub fn enter_model_actions(&mut self) {
        // The run that just ended most likely wrote a checkpoint, and "run
        // again" almost always means "carry on from it" — so the re-read also
        // re-points the flow at the weights it just found.
        let recorded = self.selected_checkpoint_path.clone();
        self.preselect_pretrained_weights(recorded);
        self.model_actions.error = None;
        self.screen = Screen::ModelActions;
    }

    /// The action currently highlighted in the action menu.
    pub fn selected_action(&self) -> Option<ModelAction> {
        ModelAction::from_index(self.model_actions.selected)
    }

    // --- Walking the path: `←` and `→`, and the `Esc` they share a spine with ---

    /// One step back up the path.
    ///
    /// `Esc` and `←` both go through here on the selector screens, which is the
    /// point: two keys that mean "back" and are implemented twice drift, and the
    /// drift is invisible until someone uses the one that was not maintained.
    ///
    /// Returns `false` when there is nothing above — only on the model list,
    /// the front door. `Esc` turns that into a quit; `←` turns it into nothing,
    /// because an arrow key must never be the thing that ends the session.
    pub fn path_back(&mut self) -> bool {
        match self.screen {
            Screen::ModelList => false,
            Screen::TemplateSelector => {
                self.refresh_model_list();
                self.screen = Screen::ModelList;
                true
            }
            Screen::ModelActions => {
                self.model_actions.error = None;
                self.refresh_model_list();
                self.screen = Screen::ModelList;
                true
            }
            Screen::WeightSelector => {
                self.screen = Screen::ModelActions;
                true
            }
            _ => false,
        }
    }

    /// One step forward along the path, for the user who went back with `←` and
    /// wants to return without re-deciding anything.
    ///
    /// It only ever moves where moving is *navigation*. The template selector is
    /// the deliberate hole in the sequence: going forward from it writes a new
    /// model's `config_file` to disk, and an arrow key is not an instruction to
    /// create something. `Enter` remains the way to do that. The same reasoning
    /// keeps `Rename` and `Delete` inert here — they are not steps of the path,
    /// they are what the action menu also happens to offer.
    pub fn path_forward(&mut self) {
        match self.screen {
            Screen::ModelList => {
                let returning_to_the_same_model = !self.model_list.is_new_model_selected()
                    && self
                        .model_list
                        .selected_model()
                        .map(|model| model.name.as_str())
                        == self.active_model_name.as_deref();
                if returning_to_the_same_model {
                    // Re-opening would re-read the `config_file` and reset the
                    // action and weight choices from it. The whole promise of
                    // `→` is that going back and forward costs nothing, so the
                    // model already in hand is simply picked back up.
                    self.model_actions.error = None;
                    self.screen = Screen::ModelActions;
                } else {
                    self.finish_model_list();
                }
            }
            Screen::ModelActions => {
                if self.selected_action().is_some_and(ModelAction::is_run) {
                    self.finish_model_actions();
                }
            }
            Screen::WeightSelector => self.finish_weight_selector(),
            _ => {}
        }
    }

    /// Whether this process is running the named model right now.
    ///
    /// The state consulted is the TUI's own: `running_model` is set when a run
    /// starts and stays set until the run reports itself done. **It does not
    /// cross the process boundary** — a second batlab, or a `--headless-train`
    /// in another shell, is invisible here, and renaming a model out from under
    /// one of those will send its next checkpoint write to a directory that no
    /// longer exists. A lockfile under the model directory is the fix, and is
    /// deliberately not in this mission.
    pub fn model_run_in_progress(&self, model_name: &str) -> bool {
        self.running_model.as_deref() == Some(model_name) && !self.monitor.done
    }

    fn guard_model_not_running(&self, model_name: &str) -> Result<(), String> {
        if self.model_run_in_progress(model_name) {
            return Err(format!(
                "'{model_name}' has a run in progress in this session — stop it first."
            ));
        }
        Ok(())
    }

    /// Enter on the action menu. The three run modes go on to pick weights; the
    /// two manager operations open their own screen, and refuse outright while
    /// the model is running.
    pub fn finish_model_actions(&mut self) {
        let Some(action) = self.selected_action() else {
            return;
        };
        let Some(model_name) = self.active_model_name.clone() else {
            self.model_actions.error = Some("No model selected.".to_string());
            return;
        };
        if !action.is_run()
            && let Err(err) = self.guard_model_not_running(&model_name)
        {
            self.model_actions.error = Some(err);
            return;
        }
        self.model_actions.error = None;

        match action {
            ModelAction::Train | ModelAction::Infer | ModelAction::Perpetual => {
                self.refresh_weight_selector();
                self.screen = Screen::WeightSelector;
            }
            ModelAction::Rename => {
                self.rename_model.input = model_name;
                self.rename_model.error = None;
                self.screen = Screen::RenameModel;
            }
            ModelAction::Duplicate => {
                self.duplicate_model.input = self.storage.suggest_copy_name(&model_name);
                // Weights, every time the form opens. The choice is not
                // remembered between visits: "duplicate to fine-tune" is what
                // the feature is for, and a leftover "config only" from a
                // previous visit would silently make a copy with no weights.
                self.duplicate_model.selected = 0;
                self.duplicate_model.error = None;
                self.screen = Screen::DuplicateModel;
            }
            ModelAction::Delete => {
                self.delete_confirm.typed.clear();
                self.delete_confirm.error = None;
                self.screen = Screen::DeleteConfirm;
            }
        }
    }

    /// Renames the open model on disk, then returns to the list with the
    /// renamed entry selected — the list is where the new name is visible, so
    /// it is where the operation reports back.
    pub fn finish_rename(&mut self) {
        let Some(current) = self.active_model_name.clone() else {
            self.rename_model.error = Some("No model selected.".to_string());
            return;
        };
        if let Err(err) = self.guard_model_not_running(&current) {
            self.rename_model.error = Some(err);
            return;
        }
        let target = self.rename_model.input.trim().to_string();
        match self.storage.rename_model(&current, &target) {
            Ok(_) => {
                self.active_model_name = Some(target.clone());
                // The weights moved with the directory; whatever the forms were
                // pointing at is stale until the model is opened again.
                self.selected_checkpoint_path = None;
                self.load_checkpoint_on_start = false;
                self.weight_selector.checkpoints.clear();
                self.weight_selector.selected = 0;
                self.rename_model.error = None;
                self.refresh_model_list();
                self.select_model_in_list(&target);
                self.model_list.error = None;
                self.model_list.status = Some(format!("Renamed '{current}' → '{target}'"));
                self.screen = Screen::ModelList;
            }
            Err(err) => self.rename_model.error = Some(err.to_string()),
        }
    }

    /// Copies the open model under a new name, then returns to the list with the
    /// **copy** selected — the copy is the model the user now wants to work on,
    /// which is the whole reason for having duplicated.
    ///
    /// The original is left open in `active_model_name`: nothing about it
    /// changed, so nothing about the session's grip on it needs to.
    pub fn finish_duplicate(&mut self) {
        let Some(current) = self.active_model_name.clone() else {
            self.duplicate_model.error = Some("No model selected.".to_string());
            return;
        };
        if let Err(err) = self.guard_model_not_running(&current) {
            self.duplicate_model.error = Some(err);
            return;
        }
        let target = self.duplicate_model.input.trim().to_string();
        let weights = if self.duplicate_model.copies_weights() {
            storage::WeightsToCopy::Newest
        } else {
            storage::WeightsToCopy::None
        };
        match self.storage.duplicate_model(&current, &target, weights) {
            Ok(outcome) => {
                self.duplicate_model.error = None;
                self.refresh_model_list();
                self.select_model_in_list(&target);
                self.model_list.error = None;
                self.model_list.status = Some(duplicate_status(&current, &target, &outcome));
                self.screen = Screen::ModelList;
            }
            Err(err) => self.duplicate_model.error = Some(err.to_string()),
        }
    }

    /// Deletes the open model, but only once its name has been typed back
    /// exactly. Anything else leaves the directory alone and says so.
    pub fn finish_delete(&mut self) {
        let Some(current) = self.active_model_name.clone() else {
            self.delete_confirm.error = Some("No model selected.".to_string());
            return;
        };
        if let Err(err) = self.guard_model_not_running(&current) {
            self.delete_confirm.error = Some(err);
            return;
        }
        if self.delete_confirm.typed != current {
            self.delete_confirm.error = Some(format!(
                "Type the model name exactly — '{current}' — to confirm."
            ));
            return;
        }
        match self.storage.delete_model(&current) {
            Ok(()) => {
                self.active_model_name = None;
                self.selected_checkpoint_path = None;
                self.load_checkpoint_on_start = false;
                self.weight_selector.checkpoints.clear();
                self.weight_selector.selected = 0;
                self.delete_confirm.typed.clear();
                self.delete_confirm.error = None;
                self.refresh_model_list();
                self.model_list.selected = 0;
                self.model_list.error = None;
                self.model_list.status = Some(format!("Deleted '{current}'"));
                self.screen = Screen::ModelList;
            }
            Err(err) => self.delete_confirm.error = Some(err.to_string()),
        }
    }

    pub fn finish_weight_selector(&mut self) {
        let Some(model_name) = self.active_model_name.clone() else {
            self.weight_selector.error = Some("No model selected.".to_string());
            return;
        };

        if self.weight_selector.selected == 0 {
            self.load_checkpoint_on_start = false;
            match self.storage.default_model_checkpoint_path(&model_name) {
                Ok(path) => {
                    self.selected_checkpoint_path = Some(path.to_string_lossy().to_string());
                    self.weight_selector.error = None;
                    self.enter_params_for_selected_action();
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
        self.enter_params_for_selected_action();
    }

    /// Opens the parameter form of whichever run mode was chosen in the action
    /// menu. Only reached from the weight selector, which only run actions
    /// reach — a manager action never lands here.
    fn enter_params_for_selected_action(&mut self) {
        match self.selected_action() {
            Some(ModelAction::Infer) => {
                self.inference_params.error = None;
                self.inference_params.field_idx = 0;
                self.screen = Screen::InferenceParams;
            }
            Some(ModelAction::Perpetual) => {
                self.perpetual_params.error = None;
                self.perpetual_params.field_idx = 0;
                self.screen = Screen::PerpetualParams;
            }
            _ => {
                self.training_params.error = None;
                self.training_params.field_idx = 0;
                self.screen = Screen::TrainingParams;
            }
        }
    }

    /// Opens the input-size form from the layer builder. Confirming it clears
    /// the layer list — the geometry is the chain's first link, and the layers
    /// after it were sized against the old one.
    pub fn open_input_size(&mut self) {
        self.input_size.fields = vec![
            self.layer_builder.model_input.0.to_string(),
            self.layer_builder.model_input.1.to_string(),
            self.layer_builder.model_input.2.to_string(),
        ];
        self.input_size.field_idx = 0;
        self.input_size.error = None;
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
        self.load_checkpoint_on_start = false;
        self.selected_checkpoint_path = self.default_checkpoint_for_active_model();
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
        self.screen = Screen::ModelActions;
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
            config.seed_dataset = self.seed_dataset.clone();
        }
        Ok(())
    }

    /// The `ModelConfig` a run is launched with — and, since
    /// `normalize_config_for_models_layout` writes it back, the one that lands
    /// in `Models/<name>/config_file`.
    ///
    /// On [`App`] rather than inline in the event loop so that what a run is
    /// given, and what the file keeps, can be asserted without a terminal.
    /// That matters here: `seed_dataset` is carried through **whatever the run
    /// mode is**, which is a claim worth a test rather than a comment.
    pub fn compose_run_config(&self, run: RunConfig) -> ModelConfig {
        let denoising_paths = self
            .inference_params
            .fields
            .get(1)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let denoise_magnitude = self
            .inference_params
            .fields
            .get(2)
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(1.0)
            .max(1e-6);
        let inference = InferenceConfig {
            random_seed: self.inference_params.random_seed,
            seed: if self.inference_params.random_seed {
                None
            } else {
                self.inference_params
                    .fields
                    .first()
                    .and_then(|value| value.parse::<u64>().ok())
            },
            denoising_paths,
            denoise_magnitude,
            checkpoint: self.selected_checkpoint_path.clone(),
        };
        ModelConfig {
            model_name: self.active_model_name.clone(),
            input_size: self.layer_builder.model_input,
            layers: self.layer_builder.layers.clone(),
            inference,
            // Whatever the mode. It belongs to the model, so a training run has
            // to carry it through rather than drop it: this is the line that
            // keeps a specialised model's seed from being erased by the next
            // night of training.
            seed_dataset: self.seed_dataset.clone(),
            run,
        }
    }

    // --- The seed dataset ---

    /// The model's own output geometry — what a seed dataset has to agree with.
    pub fn seed_dataset_target(&self) -> (u32, u32, u32) {
        self.inferred_input()
    }

    /// Whether `path` can seed a drift of this model, and why not when it
    /// cannot.
    ///
    /// **Channels only.** Width and height are resampled on the host on the way
    /// in, which is a deliberate and visible thing to do to a picture — a 64×64
    /// photograph scaled down to 32×32 is still that photograph. Channels are
    /// not: a greyscale file handed to a colour model is replicated across R, G
    /// and B, and a colour one handed to a greyscale model is flattened. Both
    /// *succeed*, which is exactly what makes them worth refusing — the repo
    /// already lost time to that silence once.
    pub fn seed_dataset_verdict(&self, path: &str) -> Result<(), String> {
        let Some((_, _, _, channels, _)) = storage::read_batraw_header(Path::new(path)) else {
            // Not a `.batraw`: a directory of images has no header to check,
            // and its channels are decided by the loader on purpose.
            return Ok(());
        };
        let wanted = self.seed_dataset_target().2;
        if channels == wanted {
            return Ok(());
        }
        Err(format!(
            "{} has {channels} channel(s), the model emits {wanted} — it would be \
             replicated or flattened without a word, not resized",
            Path::new(path)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string())
        ))
    }

    /// What the channel convention would pick if nothing else named a dataset —
    /// spelled out rather than left as "default", because "default" is exactly
    /// the word that let a colour model drift away from CIFAR trucks without
    /// anyone noticing.
    pub fn seed_dataset_default_label(&self) -> String {
        match default_seed_dataset_name(self.seed_dataset_target().2) {
            Some(name) => name.to_string(),
            None => "aucun — bruit pur".to_string(),
        }
    }

    /// What the "Seed Dataset" row shows: the chosen file, or what the
    /// convention will pick in its place.
    pub fn seed_dataset_label(&self) -> String {
        match self.seed_dataset.as_deref() {
            Some(path) => Path::new(path)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string()),
            None => format!("(défaut — {})", self.seed_dataset_default_label()),
        }
    }

    pub fn open_seed_dataset_selector(&mut self) {
        self.seed_dataset_selector.datasets = self
            .storage
            .list_datasets()
            .unwrap_or_default()
            .into_iter()
            .map(|path| {
                let geometry = storage::read_batraw_header(Path::new(&path))
                    .map(|header| (header.1, header.2, header.3));
                (path, geometry)
            })
            .collect();
        // The cursor opens on what the model already holds, so confirming
        // without moving is a no-op rather than a silent change of dataset.
        self.seed_dataset_selector.selected = match self.seed_dataset.as_deref() {
            Some(current) => self
                .seed_dataset_selector
                .datasets
                .iter()
                .position(|(path, _)| path == current)
                .map(|index| index + 1)
                .unwrap_or(0),
            None => 0,
        };
        self.seed_dataset_selector.error = None;
        self.screen = Screen::SeedDatasetSelector;
    }

    /// One past the last row: row 0 is the default, then one row per dataset.
    pub fn seed_dataset_row_count(&self) -> usize {
        self.seed_dataset_selector.datasets.len() + 1
    }

    pub fn move_seed_dataset_cursor(&mut self, delta: isize) {
        let last = self.seed_dataset_row_count().saturating_sub(1);
        let next = self.seed_dataset_selector.selected as isize + delta;
        self.seed_dataset_selector.selected = next.clamp(0, last as isize) as usize;
        self.seed_dataset_selector.error = None;
    }

    /// Takes the row under the cursor. An incompatible dataset is **refused
    /// here**, on the screen that chose it — not swallowed and turned into a
    /// mangled opening picture three screens later.
    pub fn finish_seed_dataset_selector(&mut self) -> Result<(), String> {
        let selected = self.seed_dataset_selector.selected;
        let chosen = match selected {
            0 => None,
            index => match self.seed_dataset_selector.datasets.get(index - 1) {
                Some((path, _)) => Some(path.clone()),
                None => return Err("That row is not a dataset.".to_string()),
            },
        };
        if let Some(path) = chosen.as_deref() {
            self.seed_dataset_verdict(path)?;
        }
        self.seed_dataset = chosen;
        self.seed_dataset_selector.error = None;
        self.screen = Screen::PerpetualParams;
        // Onto the next row, so walking the form with Enter walks *through*
        // this one instead of reopening the chooser it just closed.
        self.perpetual_params.field_idx = PERPETUAL_SEED_DATASET_FIELD + 1;
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

    pub fn enter_layer_builder(&mut self) {
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

    /// Open the Resources page on the model in hand.
    ///
    /// The batch it opens at is the one the model's own training form carries,
    /// not a constant: the first number someone wants is what their *current*
    /// run costs, and only then what a different batch would.
    pub fn enter_resources(&mut self) {
        self.resources.scroll = 0;
        // Slot 1 of the training form is the batch size — see the field's own
        // comment on `TrainingParamsState::fields`.
        if let Some(batch) = self
            .training_params
            .fields
            .get(1)
            .and_then(|value| value.parse::<u32>().ok())
        {
            self.resources.batch = batch.max(1);
        }
        self.screen = Screen::Resources;
    }

    /// The inventory the Resources page draws, for the batch and workload it is
    /// currently asking about.
    ///
    /// Recomputed rather than stored — see [`ResourcesState`]. It reads the
    /// architecture from the layer builder's own list, which is the *edited*
    /// stack: add a layer and the page's numbers move with it, without a save.
    pub fn resources_inventory(
        &self,
        batch: u32,
    ) -> Result<batlab_core::GpuInventory, batlab_core::ModelError> {
        batlab_core::inventory(&self.resources_request(batch), &self.device_profile)
    }

    pub fn resources_request(&self, batch: u32) -> batlab_core::InventoryRequest {
        let workload = if self.resources.inference {
            batlab_core::Workload::Inference
        } else {
            batlab_core::Workload::training(self.resources_optimizer, self.resources_ema)
        };
        batlab_core::InventoryRequest {
            layers: self.layer_builder.layers.clone(),
            input_size: self.layer_builder.model_input,
            batch: batch.max(1),
            workload,
            // Inference streams nothing: the sampler holds one latent, and the
            // dataset is not in the graph at all.
            dataset: if self.resources.inference {
                None
            } else {
                self.resources_dataset
            },
            live_frame: false,
        }
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
        let dataset_path = self.training_params.fields[TRAINING_DATASET_FIELD].clone();
        if dataset_path.trim().is_empty() {
            return Err("Dataset path must not be empty".into());
        }
        if !std::path::Path::new(&dataset_path).exists() {
            return Err("Dataset path does not exist".into());
        }
        let ema_decay = parse_ema_decay_field(&self.training_params.fields[TRAINING_EMA_FIELD])?;
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
                // Carried from the model's own config, not reset to the
                // defaults. These three have no form field, so `::default()`
                // here meant that starting a run from the TUI silently
                // downgraded `--optimizer adam` to SGD — and then wrote that
                // back to `config_file`, making the loss permanent. Worse, the
                // Resources page one screen earlier reads the *file's* value,
                // so it sized an Adam run while launching an SGD one.
                optimizer: self.training_optimizer,
                weight_init: self.training_weight_init,
                loss_weighting: self.training_loss_weighting,
                ema_decay,
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
        parse_ema_decay_field(&self.training_params.fields[TRAINING_EMA_FIELD])?;
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
            0 | TRAINING_EMA_FIELD => c.is_ascii_digit() || c == '.',
            1 | 2 => c.is_ascii_digit(),
            _ => false,
        };
        if accepted {
            self.training_params.fields[idx].push(c);
            self.training_params.error = None;
        }
    }

    /// Backspace deletes a character of the *typed* fields only.
    ///
    /// The guard is not decoration: `fields[TRAINING_DATASET_FIELD]` is the
    /// dataset path, which the dataset selector owns and this form never shows.
    /// With the toggle sharing its index, an unguarded `pop()` would have eaten
    /// that path one character per keystroke, from a screen where nothing
    /// appears to change.
    pub fn handle_backspace_training(&mut self) {
        let idx = self.training_params.field_idx;
        if idx >= TRAINING_RANDOM_WEIGHTS_FIELD {
            return;
        }
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

    /// Both manager forms take a model name, so every printable key is text —
    /// including `q`, which quits on most other screens. `Esc` is the only way
    /// out, or a model called `q-experiment` could not be typed at all.
    pub fn handle_char_rename(&mut self, c: char) {
        if !c.is_control() {
            self.rename_model.input.push(c);
            self.rename_model.error = None;
        }
    }

    pub fn handle_backspace_rename(&mut self) {
        self.rename_model.input.pop();
        self.rename_model.error = None;
    }

    pub fn handle_char_duplicate(&mut self, c: char) {
        if !c.is_control() {
            self.duplicate_model.input.push(c);
            self.duplicate_model.error = None;
        }
    }

    pub fn handle_backspace_duplicate(&mut self) {
        self.duplicate_model.input.pop();
        self.duplicate_model.error = None;
    }

    pub fn handle_char_delete_confirm(&mut self, c: char) {
        if !c.is_control() {
            self.delete_confirm.typed.push(c);
            self.delete_confirm.error = None;
        }
    }

    pub fn handle_backspace_delete_confirm(&mut self) {
        self.delete_confirm.typed.pop();
        self.delete_confirm.error = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        App, DUPLICATE_CONTENT_CHOICES, InferenceConfig, LayerKind, LossMethod, LossWeighting,
        MIN_RENOISE_DEPTH, ModelAction, ModelConfig, OptimizerKind, PERPETUAL_SEED_DATASET_FIELD,
        PerpetualRegime, RunConfig,
        RunMode, Screen, TRAINING_DATASET_FIELD, TRAINING_EMA_FIELD, TrainingConfig,
        TrainingControlCommand, WeightInit, parse_ema_decay_field, storage,
    };
    use crate::storage::TempRoot;
    use crate::tui::events::handle_key;
    use crossterm::event::KeyCode;
    use batlab_core::config::{built_in_templates, compute_inferred_input};

    /// Every app under test lives on its own throwaway storage root. The
    /// template route writes a `config_file`, and rename/delete move and remove
    /// directories: on the deduced root all of that landed in the repository's
    /// own `Models/` (`docs/reports/PERPETUAL_INFERENCE.md` §5).
    fn test_app(tag: &str) -> (TempRoot, App) {
        let temp = TempRoot::new(tag);
        let app = App::with_storage(temp.storage());
        (temp, app)
    }

    #[test]
    fn cycle_kind_backward_moves_in_reverse_order() {
        let (_temp, mut app) = test_app("cycle-kind");
        app.layer_builder.current_kind = LayerKind::Convolution;

        // TimeBias is the last kind in the ring, so stepping back from the first
        // (Convolution) lands on it, then on Add.
        app.cycle_kind_backward();
        assert_eq!(app.layer_builder.current_kind, LayerKind::TimeBias);

        app.cycle_kind_backward();
        assert_eq!(app.layer_builder.current_kind, LayerKind::Add);
    }

    /// The front door is the model list. `Screen::LoadPath` — its ancestor —
    /// existed, was drawn, and handled its keys while nothing ever assigned it,
    /// so a trained model under `Models/` could not be reached at all. Opening
    /// straight onto the list is what makes "I have models, show me them" cost
    /// nothing.
    #[test]
    fn app_opens_on_the_model_list() {
        let (_temp, app) = test_app("opens-on-list");
        assert_eq!(app.screen, Screen::ModelList);
        assert_eq!(app.model_list.selected, 0);
    }

    /// The last row of the list is the template flow, and its index moves with
    /// the list — on an empty root it is row 0, which is why a fresh install is
    /// not a dead end.
    #[test]
    fn the_last_row_of_an_empty_list_opens_the_template_flow() {
        let (_temp, mut app) = test_app("empty-list");
        assert!(app.model_list.models.is_empty());
        assert!(app.model_list.is_new_model_selected());

        app.finish_model_list();

        assert_eq!(app.screen, Screen::TemplateSelector);
    }

    /// Opening a model goes straight to the action menu: model first, action
    /// second. Nothing in between.
    #[test]
    fn opening_a_saved_model_lands_on_the_action_menu() {
        let (_temp, mut app) = test_app("open-model");
        app.finish_template_selector();
        let created = app.active_model_name.clone().expect("template made a model");
        app.refresh_model_list();
        app.screen = Screen::ModelList;
        app.select_model_in_list(&created);

        app.finish_model_list();

        assert_eq!(app.screen, Screen::ModelActions);
        assert_eq!(app.active_model_name.as_deref(), Some(created.as_str()));
    }

    /// Opening a saved model must not touch what it says. The template route
    /// calls `write_model_config` with `InferenceConfig::default()`, which is
    /// how the previous mission silently reset two `config_file`s; the load
    /// route has to carry the stored inference block into the form instead.
    #[test]
    fn loading_a_model_preserves_its_inference_block() {
        let (_temp, mut app) = test_app("preserve-inference");
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
            seed_dataset: None,
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
        // The model's own run mode preselects the action, so re-opening a model
        // offers what it was last used for.
        assert_eq!(app.model_actions.selected, ModelAction::Infer.index());
        assert_eq!(app.screen, Screen::ModelActions);
    }

    /// Starting a training run from the TUI must **preserve** the three knobs
    /// the config file carries and no form shows.
    ///
    /// Same defect class as the seed dataset, one step over: the field is read
    /// at run time, but the only interactive way to launch destroyed it first.
    /// `--optimizer adam` converges ~20× faster per step
    /// (`OPTIMIZER_ADAM.md`), and a run started from the TUI silently became
    /// SGD — then wrote that back to `config_file`, so the loss was permanent
    /// and invisible.
    #[test]
    fn starting_a_run_from_the_form_keeps_the_optimizer_the_config_asked_for() {
        let (_temp, mut app) = test_app("keep-training-knobs");
        let dataset = a_dataset(&app, "train.batraw", (32, 32, 3));
        let config = ModelConfig {
            model_name: Some("unit-test-load".to_string()),
            input_size: (32, 32, 7),
            layers: Vec::new(),
            inference: InferenceConfig::default(),
            seed_dataset: None,
            run: RunConfig {
                mode: RunMode::Train(TrainingConfig {
                    lr: 0.001,
                    batch_size: 4,
                    steps: 100,
                    dataset_path: dataset.clone(),
                    loss: LossMethod::MeanSquared,
                    checkpoint_path: None,
                    load_checkpoint: false,
                    optimizer: OptimizerKind::Adam,
                    weight_init: WeightInit::He,
                    ema_decay: None,
                    loss_weighting: LossWeighting::Snr { gamma: 1.0 },
                }),
            },
        };
        app.apply_loaded_model(config);

        app.training_params.fields[TRAINING_DATASET_FIELD] = dataset;
        app.sync_selected_dataset_from_field();
        app.finish_dataset_selector()
            .expect("the form should validate");

        let RunMode::Train(train) = app
            .run_config
            .clone()
            .expect("finishing arms a run")
            .mode
        else {
            panic!("the training form must arm a training run");
        };
        assert_eq!(
            train.optimizer,
            OptimizerKind::Adam,
            "the run must use the optimiser the config asked for, not SGD"
        );
        assert_eq!(train.weight_init, WeightInit::He);
        assert_eq!(train.loss_weighting, LossWeighting::Snr { gamma: 1.0 });
    }

    // -- The seed dataset ----------------------------------------------------

    /// Writes a `.batraw` of the given geometry under the test root and returns
    /// its path. Only the header is ever read by the chooser.
    fn a_dataset(app: &App, name: &str, geometry: (u32, u32, u32)) -> String {
        let dir = app
            .storage
            .datasets_dir()
            .expect("the test root must have a datasets/ dir");
        let path = dir.join(name);
        let mut bytes = b"BATRAW3\0".to_vec();
        for word in [1u32, geometry.0, geometry.1, geometry.2] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.resize(
            bytes.len() + (geometry.0 * geometry.1 * geometry.2) as usize,
            0,
        );
        std::fs::write(&path, bytes).expect("writing a test dataset should work");
        path.to_string_lossy().into_owned()
    }

    /// **The form sets it, the `config_file` keeps it, reopening reads it
    /// back** — the round trip the whole feature is for. A specialised model
    /// that has to be told its own dataset at every launch has not been fixed.
    #[test]
    fn a_seed_dataset_chosen_in_the_form_survives_being_written_and_reopened() {
        let (_temp, mut app, name) = app_on_a_model("seed-roundtrip");
        // The template model emits 3 channels, so a 3-channel dataset fits.
        let chosen = a_dataset(&app, "elephants.batraw", (32, 32, 3));

        app.model_actions.selected = ModelAction::Perpetual.index();
        app.finish_model_actions();
        app.finish_weight_selector();
        assert_eq!(app.screen, Screen::PerpetualParams);

        app.perpetual_params.field_idx = PERPETUAL_SEED_DATASET_FIELD;
        app.open_seed_dataset_selector();
        assert_eq!(app.screen, Screen::SeedDatasetSelector);
        let row = app
            .seed_dataset_selector
            .datasets
            .iter()
            .position(|(path, _)| path == &chosen)
            .expect("the chooser must list the datasets/ directory");
        app.seed_dataset_selector.selected = row + 1;
        app.finish_seed_dataset_selector()
            .expect("a fitting dataset must be accepted");
        assert_eq!(app.seed_dataset.as_deref(), Some(chosen.as_str()));

        // What a run would be launched with, and what gets written to disk.
        app.finish_perpetual_params()
            .expect("the form should validate");
        let run = app
            .run_config
            .clone()
            .expect("finishing the form arms a run");
        let written = app.compose_run_config(run);
        app.storage
            .write_model_config(&name, &written)
            .expect("writing the config should work");

        // Reopen it the way the model list does.
        let reloaded = app
            .storage
            .load_model_config_for_model(&name)
            .expect("the config should load back");
        assert_eq!(
            reloaded.seed_dataset.as_deref(),
            Some(chosen.as_str()),
            "the seed dataset must survive the round trip through the config_file"
        );
        let (_temp2, mut fresh) = test_app("seed-roundtrip-reopen");
        fresh.storage = app.storage.clone();
        fresh.apply_loaded_model(reloaded);
        assert_eq!(
            fresh.seed_dataset.as_deref(),
            Some(chosen.as_str()),
            "reopening the model must point the drift back at its own dataset"
        );
    }

    /// It is a property of the **model**, so a training run must carry it
    /// through rather than drop it.
    ///
    /// Parked inside `RunMode::Perpetual` — where it was declared — a single
    /// night of training rewrote `run.mode` and the setting was gone, with
    /// nothing on screen to say so. That is why it sits at the top level of
    /// `ModelConfig`, beside `inference`.
    #[test]
    fn a_training_run_does_not_erase_the_models_seed_dataset() {
        let (_temp, mut app) = test_app("seed-survives-training");
        let config = ModelConfig {
            model_name: Some("unit-test-load".to_string()),
            input_size: (32, 32, 7),
            layers: Vec::new(),
            inference: InferenceConfig::default(),
            seed_dataset: Some("datasets/elephants256.batraw".to_string()),
            run: RunConfig {
                mode: RunMode::Train(TrainingConfig {
                    lr: 0.01,
                    batch_size: 2,
                    steps: 10,
                    dataset_path: ".".to_string(),
                    loss: LossMethod::MeanSquared,
                    checkpoint_path: None,
                    load_checkpoint: false,
                    optimizer: OptimizerKind::default(),
                    weight_init: WeightInit::default(),
                    ema_decay: None,
                    loss_weighting: LossWeighting::default(),
                }),
            },
        };

        app.apply_loaded_model(config);
        assert_eq!(
            app.seed_dataset.as_deref(),
            Some("datasets/elephants256.batraw"),
            "a model whose last run was training still has a seed dataset, and \
             the form has to find it"
        );

        // And launching a *training* run writes it straight back, which is the
        // half that was lost when it lived inside `RunMode::Perpetual`.
        for mode in [
            RunMode::Infer,
            RunMode::Train(TrainingConfig {
                lr: 0.01,
                batch_size: 2,
                steps: 10,
                dataset_path: ".".to_string(),
                loss: LossMethod::MeanSquared,
                checkpoint_path: None,
                load_checkpoint: false,
                optimizer: OptimizerKind::default(),
                weight_init: WeightInit::default(),
                ema_decay: None,
                loss_weighting: LossWeighting::default(),
            }),
        ] {
            let composed = app.compose_run_config(RunConfig { mode });
            assert_eq!(
                composed.seed_dataset.as_deref(),
                Some("datasets/elephants256.batraw"),
                "every run mode has to carry the model's seed dataset through"
            );
        }
    }

    /// Opening a model that names **no** dataset must not leave the previous
    /// model's in place. The state is per-model, so it is read outside the
    /// `run.mode` match — read inside it, a stale value would survive.
    #[test]
    fn opening_a_model_without_a_seed_dataset_clears_the_previous_ones() {
        let (_temp, mut app) = test_app("seed-not-sticky");
        let mut config = ModelConfig {
            model_name: Some("first".to_string()),
            input_size: (32, 32, 7),
            layers: Vec::new(),
            inference: InferenceConfig::default(),
            seed_dataset: Some("datasets/elephants256.batraw".to_string()),
            run: RunConfig {
                mode: RunMode::Infer,
            },
        };
        app.apply_loaded_model(config.clone());
        assert!(app.seed_dataset.is_some());

        config.model_name = Some("second".to_string());
        config.seed_dataset = None;
        app.apply_loaded_model(config);
        assert_eq!(
            app.seed_dataset, None,
            "the second model names none, so it must drift from the convention — \
             not from the first model's elephants"
        );
    }

    /// A dataset whose **channels** disagree with the model is refused by the
    /// chooser, with the reason on screen, and the setting is left alone.
    ///
    /// Refusing here rather than at run time is the point: the loader would
    /// have replicated a greyscale file across R, G and B and succeeded, and
    /// the only symptom would have been a drift away from a grey picture.
    #[test]
    fn the_chooser_refuses_a_dataset_of_the_wrong_channels() {
        let (_temp, mut app, _name) = app_on_a_model("seed-wrong-channels");
        let grey = a_dataset(&app, "grey.batraw", (32, 32, 1));
        assert_eq!(app.seed_dataset_target().2, 3, "the template emits colour");

        app.open_seed_dataset_selector();
        let row = app
            .seed_dataset_selector
            .datasets
            .iter()
            .position(|(path, _)| path == &grey)
            .expect("the chooser lists it, marked");
        app.seed_dataset_selector.selected = row + 1;

        let err = app
            .finish_seed_dataset_selector()
            .expect_err("a 1-channel dataset must not seed a 3-channel model");
        assert!(
            err.contains('1') && err.contains('3'),
            "the refusal must name both counts: {err}"
        );
        assert_eq!(app.seed_dataset, None, "a refused row must change nothing");
        assert_eq!(
            app.screen,
            Screen::SeedDatasetSelector,
            "a refusal stays on the screen that can act on it"
        );
    }

    /// The first row hands the choice back to the convention, and says which
    /// file that is. "Default" without naming the file is how a colour model
    /// drifted away from CIFAR trucks with nothing on screen looking wrong.
    #[test]
    fn the_default_row_clears_the_choice_and_names_what_takes_over() {
        let (_temp, mut app, _name) = app_on_a_model("seed-default-row");
        let chosen = a_dataset(&app, "mine.batraw", (32, 32, 3));
        app.seed_dataset = Some(chosen);

        app.open_seed_dataset_selector();
        assert!(
            app.seed_dataset_selector.selected > 0,
            "the chooser must open on what the model already holds"
        );
        app.seed_dataset_selector.selected = 0;
        app.finish_seed_dataset_selector()
            .expect("the default row is always available");

        assert_eq!(app.seed_dataset, None);
        assert_eq!(app.seed_dataset_default_label(), "cifar10_rgb.batraw");
        assert!(
            app.seed_dataset_label().contains("cifar10_rgb.batraw"),
            "the form row has to name the file the convention will pick: {}",
            app.seed_dataset_label()
        );
    }

    #[test]
    fn selecting_template_prefills_architecture_and_advances_to_the_action_menu() {
        let (_temp, mut app) = test_app("template-prefill");

        app.finish_template_selector();

        assert_eq!(app.screen, Screen::ModelActions);
        assert!(!app.layer_builder.layers.is_empty());
        // 3 signal channels + 4 carrying the time embedding.
        assert_eq!(app.layer_builder.model_input, (32, 32, 7));
    }

    /// Every model the "New model (from template)" flow writes to disk must be
    /// conditionable on the timestep — `input_size.z > output.z`.
    ///
    /// The invariant is checked on the **config_file as written**, not on the
    /// template in memory, because that file is what training and inference
    /// will read back. Both templates shipped a model that failed this (1→1 and
    /// 3→3, the geometry of `Models/Greyscale_Diffusion_broken`); a blind test
    /// caught it by reading the file, which is why the assertion lives here too
    /// and not only in `batlab_core`.
    #[test]
    fn every_model_created_from_a_template_is_conditionable_on_the_timestep() {
        for (index, template) in built_in_templates().into_iter().enumerate() {
            let (temp, mut app) = test_app(&format!("template-geometry-{index}"));
            app.template_selector.selected = index;

            app.finish_template_selector();

            assert_eq!(
                app.template_selector.error, None,
                "template '{}' failed to apply",
                template.key
            );
            let written = temp
                .storage()
                .load_model_config_for_model(&template.key)
                .unwrap_or_else(|err| panic!("template '{}' wrote no config: {err}", template.key));
            let output = compute_inferred_input(&written.layers, written.input_size);
            assert!(
                written.input_size.2 > output.2,
                "the model written for '{}' emits {} channels for {} in — it cannot be \
                 conditioned on t, and sampling from it saturates to white",
                template.key,
                output.2,
                written.input_size.2,
            );
        }
    }

    /// The weight choice sits between the action and its parameters, so which
    /// parameter form comes next is decided by the action that was picked.
    #[test]
    fn the_weight_choice_leads_to_the_form_of_the_chosen_action() {
        let (_temp, mut app) = test_app("weights-route");
        app.finish_template_selector();

        for (action, expected) in [
            (ModelAction::Train, Screen::TrainingParams),
            (ModelAction::Infer, Screen::InferenceParams),
            (ModelAction::Perpetual, Screen::PerpetualParams),
        ] {
            app.model_actions.selected = action.index();
            app.finish_model_actions();
            assert_eq!(app.screen, Screen::WeightSelector, "{action:?}");

            app.weight_selector.selected = 0;
            app.finish_weight_selector();

            assert_eq!(app.screen, expected, "{action:?}");
            assert!(!app.load_checkpoint_on_start);
            assert!(app.selected_checkpoint_path.is_some());
        }
    }

    // -----------------------------------------------------------------------
    // Pretrained weights are the default; random is the opt-out
    // -----------------------------------------------------------------------

    /// Drops `names` into the model's `pretrained_weights/`.
    fn write_checkpoints(app: &App, model: &str, names: &[&str]) {
        let dir = app.storage.model_weights_dir(model).expect("weights dir");
        for name in names {
            std::fs::write(dir.join(name), b"weights").expect("checkpoint write");
        }
    }

    /// The point of the whole item: open a model that has weights, and the
    /// flow is already pointing at them. Before this, every run started from
    /// random unless the user walked the weight list by hand — so "train some
    /// more" silently threw away everything the model had learned.
    #[test]
    fn opening_a_model_with_weights_defaults_to_continuing_from_them() {
        let (_temp, mut app, name) = app_on_a_model("default-pretrained");
        write_checkpoints(&app, &name, &["latest.ckpt"]);

        app.enter_model_actions();

        assert!(app.load_checkpoint_on_start, "the default went to random");
        assert!(
            app.selected_checkpoint_path
                .as_deref()
                .expect("a checkpoint should be selected")
                .ends_with("latest.ckpt")
        );
        assert_eq!(
            app.weight_selector.selected, 1,
            "the cursor must sit on the checkpoint it defaulted to, or the \
             screen and the state disagree"
        );
        assert!(!app.start_from_random_weights());
    }

    /// `latest.ckpt` is what every run writes, so it wins over any other file
    /// in the folder — alphabetical order would have picked `epoch-0010.ckpt`.
    #[test]
    fn the_default_checkpoint_is_the_one_training_keeps_writing() {
        let (_temp, mut app, name) = app_on_a_model("default-latest");
        write_checkpoints(
            &app,
            &name,
            &["aaa-first-alphabetically.ckpt", "latest.ckpt", "zzz.ckpt"],
        );

        app.enter_model_actions();

        assert!(
            app.selected_checkpoint_path
                .as_deref()
                .expect("a checkpoint should be selected")
                .ends_with("latest.ckpt")
        );
    }

    /// A model with nothing to load falls back to random — and the fallback is
    /// coherent: the flag is off, and the path points at where this run will
    /// *write*, not at a file that does not exist.
    #[test]
    fn a_model_with_no_weights_falls_back_to_random() {
        let (_temp, mut app, _name) = app_on_a_model("default-no-weights");

        app.enter_model_actions();

        assert!(!app.load_checkpoint_on_start);
        assert!(app.start_from_random_weights());
        assert!(!app.has_pretrained_weights());
        assert_eq!(app.weight_selector.selected, 0);
        assert!(
            app.selected_checkpoint_path.is_some(),
            "the run still needs somewhere to save"
        );
    }

    /// The opt-out, from the training form. Off by default, and it round-trips:
    /// turning it on and off again must land back on the same checkpoint.
    #[test]
    fn the_training_form_can_opt_out_of_the_pretrained_weights_and_back() {
        let (_temp, mut app, name) = app_on_a_model("random-toggle");
        write_checkpoints(&app, &name, &["latest.ckpt"]);
        app.enter_model_actions();
        let chosen = app.selected_checkpoint_path.clone();

        assert!(!app.start_from_random_weights(), "the box starts unchecked");

        app.toggle_start_from_random();
        assert!(app.start_from_random_weights());
        assert!(!app.load_checkpoint_on_start);
        assert_eq!(
            app.weight_selector.selected, 0,
            "the weight screen must agree with the form"
        );

        app.toggle_start_from_random();
        assert!(!app.start_from_random_weights());
        assert!(app.load_checkpoint_on_start);
        assert_eq!(app.selected_checkpoint_path, chosen);
    }

    /// With no weights on disk the toggle is pinned on: there is nothing to
    /// turn it off *to*, and a checkbox that silently refuses is worse than one
    /// that is honestly stuck.
    #[test]
    fn the_random_toggle_is_pinned_when_there_is_nothing_to_load() {
        let (_temp, mut app, _name) = app_on_a_model("random-pinned");
        app.enter_model_actions();

        app.toggle_start_from_random();

        assert!(app.start_from_random_weights());
        assert!(!app.load_checkpoint_on_start);
    }

    /// Editing the architecture invalidates the weights, and the default must
    /// not quietly bring them back: a checkpoint from another geometry would be
    /// refused by the engine mid-run, after the GPU work of building the model.
    #[test]
    fn editing_a_layer_drops_the_weights_and_the_default_does_not_restore_them() {
        let (_temp, mut app, name) = app_on_a_model("edit-drops-weights");
        write_checkpoints(&app, &name, &["latest.ckpt"]);
        app.enter_model_actions();
        assert!(app.load_checkpoint_on_start, "precondition");

        app.delete_last_layer();

        assert!(!app.load_checkpoint_on_start);
        app.model_actions.selected = ModelAction::Train.index();
        app.finish_model_actions();
        assert_eq!(
            app.weight_selector.selected, 0,
            "the weight screen re-selected a checkpoint for an architecture \
             that no longer matches it"
        );
        assert!(!app.load_checkpoint_on_start);
    }

    /// The run config the form finally produces is what the engine acts on, so
    /// the default has to survive all the way to it.
    #[test]
    fn the_default_reaches_the_training_run_config() {
        let (_temp, mut app, name) = app_on_a_model("default-in-run-config");
        write_checkpoints(&app, &name, &["latest.ckpt"]);
        app.enter_model_actions();
        app.model_actions.selected = ModelAction::Train.index();
        app.finish_model_actions();
        app.finish_weight_selector();
        app.finish_training_params()
            .expect("training params should be accepted");
        app.training_params.datasets = vec![".".to_string()];
        app.training_params.selected_dataset = 0;
        app.training_params.fields[TRAINING_DATASET_FIELD] = ".".to_string();
        app.finish_dataset_selector()
            .expect("dataset selector should produce run config");

        match &app.run_config.as_ref().expect("run config").mode {
            RunMode::Train(train) => {
                assert!(train.load_checkpoint, "the run starts from random anyway");
                assert!(
                    train
                        .checkpoint_path
                        .as_deref()
                        .expect("checkpoint path")
                        .ends_with("latest.ckpt")
                );
            }
            other => panic!("expected a training run mode, got {other:?}"),
        }
    }

    /// The dataset path lives at `fields[3]`, which is also the toggle's index.
    /// Backspace on the toggle used to eat that path one character at a time,
    /// from a screen that shows neither.
    #[test]
    fn backspace_on_the_toggle_does_not_eat_the_dataset_path() {
        let (_temp, mut app, _name) = app_on_a_model("backspace-toggle");
        app.training_params.fields[TRAINING_DATASET_FIELD] = "datasets/cifar10_grey.batraw".to_string();
        app.training_params.field_idx = super::TRAINING_RANDOM_WEIGHTS_FIELD;

        for _ in 0..5 {
            app.handle_backspace_training();
        }

        assert_eq!(app.training_params.fields[TRAINING_DATASET_FIELD], "datasets/cifar10_grey.batraw");
    }

    #[test]
    fn inference_selection_builds_infer_run_config() {
        let (_temp, mut app) = test_app("infer-run-config");
        app.finish_template_selector();
        app.model_actions.selected = ModelAction::Infer.index();
        app.finish_model_actions();
        app.finish_weight_selector();
        assert_eq!(app.screen, Screen::InferenceParams);

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
        let (_temp, mut app) = test_app("perpetual-run-config");
        app.finish_template_selector();
        app.model_actions.selected = ModelAction::Perpetual.index();
        app.finish_model_actions();
        app.finish_weight_selector();
        assert_eq!(app.screen, Screen::PerpetualParams);

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
        let (_temp, mut app) = test_app("perpetual-min-depth");
        app.perpetual_params.fields[2] = (MIN_RENOISE_DEPTH - 1).to_string();
        assert!(app.finish_perpetual_params().is_err());
    }


    /// The EMA row is empty by default, and empty means no average — the run
    /// the framework did before averaging existed.
    #[test]
    fn the_ema_row_starts_empty_and_an_empty_row_means_no_average() {
        let (_temp, mut app) = test_app("ema-default-off");
        assert_eq!(app.training_params.fields[TRAINING_EMA_FIELD], "");

        app.finish_template_selector();
        app.model_actions.selected = ModelAction::Train.index();
        app.finish_model_actions();
        app.finish_weight_selector();
        app.training_params.fields[0] = "0.001".to_string();
        app.training_params.fields[1] = "16".to_string();
        app.training_params.fields[2] = "500".to_string();
        app.finish_training_params().expect("form accepted");
        app.training_params.datasets = vec![".".to_string()];
        app.training_params.selected_dataset = 0;
        app.training_params.fields[TRAINING_DATASET_FIELD] = ".".to_string();
        app.finish_dataset_selector().expect("run config built");

        match &app.run_config.as_ref().expect("run config").mode {
            RunMode::Train(train) => assert_eq!(train.ema_decay, None),
            other => panic!("expected a training run, got {other:?}"),
        }
    }

    /// A decay typed into the form reaches the run configuration.
    #[test]
    fn a_decay_typed_on_the_form_reaches_the_run_configuration() {
        let (_temp, mut app) = test_app("ema-typed");
        app.finish_template_selector();
        app.model_actions.selected = ModelAction::Train.index();
        app.finish_model_actions();
        app.finish_weight_selector();
        app.training_params.fields[0] = "0.001".to_string();
        app.training_params.fields[1] = "16".to_string();
        app.training_params.fields[2] = "500".to_string();
        app.training_params.fields[TRAINING_EMA_FIELD] = "0.999".to_string();
        app.finish_training_params().expect("form accepted");
        app.training_params.datasets = vec![".".to_string()];
        app.training_params.selected_dataset = 0;
        app.training_params.fields[TRAINING_DATASET_FIELD] = ".".to_string();
        app.finish_dataset_selector().expect("run config built");

        match &app.run_config.as_ref().expect("run config").mode {
            RunMode::Train(train) => assert_eq!(train.ema_decay, Some(0.999)),
            other => panic!("expected a training run, got {other:?}"),
        }
    }

    /// The form refuses the two decays that are silently useless rather than
    /// loudly wrong — and it refuses them on the form, not six hours later.
    #[test]
    fn the_form_refuses_a_decay_outside_the_open_unit_interval() {
        assert_eq!(parse_ema_decay_field(""), Ok(None));
        assert_eq!(parse_ema_decay_field("   "), Ok(None));
        assert_eq!(parse_ema_decay_field("0.999"), Ok(Some(0.999)));
        for refused in ["1", "1.0", "0", "0.0", "-0.5", "2", "abc"] {
            assert!(
                parse_ema_decay_field(refused).is_err(),
                "'{refused}' should be refused"
            );
        }

        let (_temp, mut app) = test_app("ema-refused");
        app.finish_template_selector();
        app.model_actions.selected = ModelAction::Train.index();
        app.finish_model_actions();
        app.finish_weight_selector();
        app.training_params.fields[0] = "0.001".to_string();
        app.training_params.fields[1] = "16".to_string();
        app.training_params.fields[2] = "500".to_string();
        app.training_params.fields[TRAINING_EMA_FIELD] = "1.0".to_string();
        assert!(app.finish_training_params().is_err());
    }

    /// Typing on the EMA row must not eat the dataset path.
    ///
    /// This is the exact bug the old `fields[3]`-is-both-things layout produced
    /// once already: the toggle shared its index with the dataset, and
    /// backspace on it deleted the path one character per keystroke from a
    /// screen where nothing appeared to change. Inserting a row in front of the
    /// dataset is precisely the edit that re-creates it.
    #[test]
    fn typing_on_the_ema_row_leaves_the_dataset_path_alone() {
        let (_temp, mut app) = test_app("ema-no-clobber");
        app.training_params.fields[TRAINING_DATASET_FIELD] =
            "datasets/cifar10_grey.batraw".to_string();
        app.training_params.field_idx = TRAINING_EMA_FIELD;
        for c in "0.999".chars() {
            app.handle_char_training(c);
        }
        assert_eq!(app.training_params.fields[TRAINING_EMA_FIELD], "0.999");
        for _ in 0..10 {
            app.handle_backspace_training();
        }
        assert_eq!(app.training_params.fields[TRAINING_EMA_FIELD], "");
        assert_eq!(
            app.training_params.fields[TRAINING_DATASET_FIELD],
            "datasets/cifar10_grey.batraw",
            "the dataset path is not what this row edits"
        );

        // …and the toggle row, which is past every typed field, must not edit
        // anything at all.
        app.training_params.field_idx = super::TRAINING_RANDOM_WEIGHTS_FIELD;
        for _ in 0..10 {
            app.handle_backspace_training();
        }
        app.handle_char_training('7');
        assert_eq!(
            app.training_params.fields[TRAINING_DATASET_FIELD],
            "datasets/cifar10_grey.batraw"
        );
    }

    #[test]
    fn dataset_selector_requires_available_dataset() {
        let (_temp, mut app) = test_app("dataset-required");
        app.finish_template_selector();
        app.model_actions.selected = ModelAction::Train.index();
        app.finish_model_actions();
        app.finish_weight_selector();
        app.training_params.fields[0] = "0.01".to_string();
        app.training_params.fields[1] = "1".to_string();
        app.training_params.fields[2] = "10".to_string();
        app.finish_training_params()
            .expect("training params should be valid");
        app.training_params.datasets.clear();
        app.training_params.fields[TRAINING_DATASET_FIELD].clear();

        let result = app.finish_dataset_selector();

        assert!(result.is_err());
    }

    #[test]
    fn training_params_and_dataset_build_train_run_config() {
        let (_temp, mut app) = test_app("train-run-config");
        app.finish_template_selector();
        app.model_actions.selected = ModelAction::Train.index();
        app.finish_model_actions();
        app.finish_weight_selector();

        app.training_params.fields[0] = "0.005".to_string();
        app.training_params.fields[1] = "3".to_string();
        app.training_params.fields[2] = "77".to_string();
        app.finish_training_params()
            .expect("training params should be accepted");
        assert!(matches!(app.screen, Screen::DatasetSelector));

        app.training_params.datasets = vec![".".to_string()];
        app.training_params.selected_dataset = 0;
        app.training_params.fields[TRAINING_DATASET_FIELD] = ".".to_string();
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
        let (_temp, mut app) = test_app("toggle-pause");
        app.monitor.current_lr = Some(0.01);
        app.monitor.current_batch_size = Some(1);

        app.toggle_training_pause();
        let commands = app.drain_monitor_control_commands();

        assert_eq!(commands, vec![TrainingControlCommand::SetPaused(true)]);
        assert!(app.monitor.is_training_paused);
    }

    #[test]
    fn finishing_training_control_enqueues_update_command() {
        let (_temp, mut app) = test_app("training-control");
        app.monitor.current_lr = Some(0.01);
        app.monitor.current_batch_size = Some(2);
        app.monitor.total_steps = 100;
        app.monitor.model_config = Some(ModelConfig {
            model_name: Some("unit-test-model".to_string()),
            input_size: (32, 32, 3),
            layers: Vec::new(),
            inference: InferenceConfig::default(),
            seed_dataset: None,
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
                    ema_decay: None,
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

    // -----------------------------------------------------------------------
    // The manager: rename, duplicate and delete, as the UI drives them
    // -----------------------------------------------------------------------

    /// Puts the app on the action menu of a freshly created template model.
    fn app_on_a_model(tag: &str) -> (TempRoot, App, String) {
        let (temp, mut app) = test_app(tag);
        app.finish_template_selector();
        let name = app
            .active_model_name
            .clone()
            .expect("the template route names the model it creates");
        assert_eq!(app.screen, Screen::ModelActions);
        (temp, app, name)
    }

    #[test]
    fn rename_moves_the_directory_updates_the_config_and_reports_in_the_list() {
        let (_temp, mut app, original) = app_on_a_model("ui-rename");
        app.model_actions.selected = ModelAction::Rename.index();
        app.finish_model_actions();
        assert_eq!(app.screen, Screen::RenameModel);
        assert_eq!(
            app.rename_model.input, original,
            "the form should open on the current name, so a small edit is a small edit"
        );

        app.rename_model.input = "renamed-model".to_string();
        app.finish_rename();

        assert_eq!(app.screen, Screen::ModelList);
        assert_eq!(app.active_model_name.as_deref(), Some("renamed-model"));
        assert!(!app.storage.models_root().join(&original).exists());
        assert!(app.storage.models_root().join("renamed-model").is_dir());
        assert_eq!(
            app.storage
                .load_model_config_for_model("renamed-model")
                .expect("the renamed config should load")
                .model_name
                .as_deref(),
            Some("renamed-model")
        );
        assert!(
            app.model_list
                .models
                .iter()
                .any(|model| model.name == "renamed-model"),
            "the list must show the rename it just performed"
        );
        assert!(app.model_list.status.is_some(), "the rename went unreported");
    }

    #[test]
    fn rename_refuses_an_unsafe_name_and_changes_nothing() {
        let (_temp, mut app, original) = app_on_a_model("ui-rename-unsafe");
        app.model_actions.selected = ModelAction::Rename.index();
        app.finish_model_actions();

        for bad in ["", "../escape", "with/slash", "."] {
            app.rename_model.input = bad.to_string();
            app.finish_rename();

            assert_eq!(
                app.screen,
                Screen::RenameModel,
                "rename to {bad:?} left the form"
            );
            assert!(
                app.rename_model.error.is_some(),
                "rename to {bad:?} was silent"
            );
            assert!(app.storage.models_root().join(&original).is_dir());
            assert_eq!(app.active_model_name.as_deref(), Some(original.as_str()));
        }
    }

    #[test]
    fn delete_requires_the_name_typed_back_exactly() {
        let (_temp, mut app, original) = app_on_a_model("ui-delete-confirm");
        app.model_actions.selected = ModelAction::Delete.index();
        app.finish_model_actions();
        assert_eq!(app.screen, Screen::DeleteConfirm);
        assert!(
            app.delete_confirm.typed.is_empty(),
            "the confirmation must start empty — a prefilled name is not a confirmation"
        );

        for wrong in ["", "y", "yes", &original.to_lowercase(), &original[..2]] {
            if wrong == original {
                continue;
            }
            app.delete_confirm.typed = wrong.to_string();
            app.finish_delete();

            assert_eq!(app.screen, Screen::DeleteConfirm, "{wrong:?} got through");
            assert!(app.delete_confirm.error.is_some());
            assert!(
                app.storage.models_root().join(&original).is_dir(),
                "{wrong:?} deleted the model"
            );
        }

        app.delete_confirm.typed = original.clone();
        app.finish_delete();

        assert_eq!(app.screen, Screen::ModelList);
        assert!(!app.storage.models_root().join(&original).exists());
        assert!(app.active_model_name.is_none());
        assert!(app.model_list.status.is_some(), "the delete went unreported");
    }

    /// Writes weights into a model, the way a finished run leaves them: a dated
    /// run with `latest.ckpt` linked onto it.
    fn weights_on_disk(app: &App, model: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = app.storage.model_weights_dir(model).expect("weights dir");
        let dated = dir.join("run-2026-08-08_1120.ckpt");
        std::fs::write(&dated, bytes).expect("write");
        storage::point_latest_at(&dated).expect("latest link");
        dated
    }

    /// The flow the feature exists for: open the foundation, duplicate it with
    /// its weights, and end up on the *copy* — with the original untouched.
    #[test]
    fn duplicating_lands_on_the_copy_and_leaves_the_original_alone() {
        let (_temp, mut app, original) = app_on_a_model("ui-duplicate");
        let source_weights = weights_on_disk(&app, &original, b"the foundation's weights");

        app.model_actions.selected = ModelAction::Duplicate.index();
        app.finish_model_actions();
        assert_eq!(app.screen, Screen::DuplicateModel);
        assert_eq!(
            app.duplicate_model.input,
            format!("{original}-copy"),
            "the form should open on a free name, so the common case is one keystroke"
        );
        assert!(
            app.duplicate_model.copies_weights(),
            "weights are the default — a fine-tune with no weights is a new model"
        );

        app.duplicate_model.input = "Elephants_XL".to_string();
        app.finish_duplicate();

        assert_eq!(app.screen, Screen::ModelList);
        assert_eq!(
            app.model_list.selected_model().map(|model| model.name.as_str()),
            Some("Elephants_XL"),
            "the cursor must land on the copy — it is the model to work on now"
        );
        let status = app.model_list.status.clone().expect("the copy went unreported");
        assert!(
            status.contains("run-2026-08-08_1120.ckpt") && status.contains("latest.ckpt"),
            "the list must say what was copied, got: {status}"
        );

        // The copy is a model of its own, and it carries the weights.
        assert_eq!(
            app.storage
                .load_model_config_for_model("Elephants_XL")
                .expect("the copy's config should load")
                .model_name
                .as_deref(),
            Some("Elephants_XL")
        );
        let copied = app
            .storage
            .list_model_checkpoints("Elephants_XL")
            .expect("listing");
        assert_eq!(copied.len(), 2, "expected the dated run and latest.ckpt");

        // And the foundation is exactly where it was.
        assert!(app.storage.models_root().join(&original).is_dir());
        assert_eq!(
            std::fs::read(&source_weights).expect("read"),
            b"the foundation's weights"
        );
        assert_eq!(
            app.storage
                .list_model_checkpoints(&original)
                .expect("listing")
                .len(),
            2,
            "the original's weights directory changed"
        );
    }

    /// The other row of the form: same architecture, no weights — and the list
    /// says so instead of letting an empty `pretrained_weights/` be a surprise.
    #[test]
    fn a_duplicate_can_be_asked_for_without_the_weights() {
        let (_temp, mut app, original) = app_on_a_model("ui-duplicate-config-only");
        weights_on_disk(&app, &original, b"weights that stay behind");

        app.model_actions.selected = ModelAction::Duplicate.index();
        app.finish_model_actions();
        handle_key(&mut app, KeyCode::Down);
        assert!(!app.duplicate_model.copies_weights());
        app.duplicate_model.input = "Fresh_XL".to_string();
        app.finish_duplicate();

        assert_eq!(app.screen, Screen::ModelList);
        assert!(
            app.storage
                .list_model_checkpoints("Fresh_XL")
                .expect("listing")
                .is_empty()
        );
        let status = app.model_list.status.clone().expect("unreported");
        assert!(
            status.contains("config only"),
            "the list must say no weights were copied, got: {status}"
        );

        // `↑` comes back to the default; the cursor cannot run off either end.
        app.model_actions.selected = ModelAction::Duplicate.index();
        app.finish_model_actions();
        handle_key(&mut app, KeyCode::Up);
        assert!(app.duplicate_model.copies_weights());
        handle_key(&mut app, KeyCode::Up);
        assert_eq!(app.duplicate_model.selected, 0);
        handle_key(&mut app, KeyCode::Down);
        handle_key(&mut app, KeyCode::Down);
        assert_eq!(app.duplicate_model.selected, DUPLICATE_CONTENT_CHOICES.len() - 1);
    }

    /// A refused name leaves the form up with a reason, and writes nothing.
    #[test]
    fn duplicating_under_an_unusable_name_changes_nothing() {
        let (_temp, mut app, original) = app_on_a_model("ui-duplicate-refusals");
        app.model_actions.selected = ModelAction::Duplicate.index();
        app.finish_model_actions();

        for bad in ["", "../evil", "a/b", ".hidden", &original] {
            app.duplicate_model.input = bad.to_string();
            app.finish_duplicate();

            assert_eq!(
                app.screen,
                Screen::DuplicateModel,
                "duplicate to {bad:?} left the form"
            );
            assert!(
                app.duplicate_model.error.is_some(),
                "duplicate to {bad:?} was silent"
            );
        }
        let models: Vec<String> = app
            .storage
            .list_models()
            .expect("listing")
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        assert_eq!(models, vec![original], "a refused duplicate created a model");
    }

    /// Both manager operations refuse a model this process is running. The guard
    /// is TUI state and stops at the process boundary — see
    /// [`App::model_run_in_progress`].
    #[test]
    fn the_manager_refuses_a_model_with_a_run_in_progress() {
        let (_temp, mut app, original) = app_on_a_model("ui-running-guard");
        app.running_model = Some(original.clone());
        app.monitor.done = false;

        for action in [
            ModelAction::Rename,
            ModelAction::Duplicate,
            ModelAction::Delete,
        ] {
            app.model_actions.selected = action.index();
            app.finish_model_actions();

            assert_eq!(
                app.screen,
                Screen::ModelActions,
                "{action:?} opened while the model was running"
            );
            assert!(app.model_actions.error.is_some(), "{action:?} was silent");
        }
        assert!(app.storage.models_root().join(&original).is_dir());

        // A run that has reported itself finished no longer holds the model.
        app.monitor.done = true;
        app.model_actions.selected = ModelAction::Delete.index();
        app.finish_model_actions();
        assert_eq!(app.screen, Screen::DeleteConfirm);
    }

    /// Re-entering the action menu re-reads the model's checkpoints. On the
    /// restart path the app is rebuilt from scratch, so a menu that trusted its
    /// own state announced "no checkpoints" for a model that had just written
    /// one.
    #[test]
    fn re_entering_the_action_menu_re_reads_the_checkpoints() {
        let (_temp, mut app, name) = app_on_a_model("reenter-actions");
        std::fs::write(
            app.storage
                .model_weights_dir(&name)
                .expect("weights dir")
                .join("latest.ckpt"),
            b"weights written by the run that just finished",
        )
        .expect("checkpoint write");
        app.weight_selector.checkpoints.clear();

        app.enter_model_actions();

        assert_eq!(app.screen, Screen::ModelActions);
        assert_eq!(
            app.weight_selector
                .checkpoints
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["latest.ckpt"]
        );
    }

    /// A run action is never blocked by the guard — only the two manager
    /// operations are.
    #[test]
    fn a_run_in_progress_does_not_block_starting_another_run() {
        let (_temp, mut app, original) = app_on_a_model("ui-running-run");
        app.running_model = Some(original);

        app.model_actions.selected = ModelAction::Infer.index();
        app.finish_model_actions();

        assert_eq!(app.screen, Screen::WeightSelector);
    }
}
