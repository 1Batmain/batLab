//! File purpose: Implements events behavior for the terminal user interface flow.

use super::app::{
    App, DUPLICATE_CONTENT_CHOICES, INPUT_SIZE_FIELD_NAMES, LayerBuilderMode, MODEL_ACTIONS,
    PERPETUAL_PARAM_FIELD_NAMES, PerpetualStatus, ResourcesState, Screen,
    TRAINING_PARAM_FIELD_NAMES,
    TRAINING_RANDOM_WEIGHTS_FIELD, TrainingControlCommand,
};
use crossterm::event::KeyCode;

#[derive(Debug)]
pub enum TrainingEvent {
    ResourceReport {
        max_buffer_bytes: u64,
        max_storage_binding_bytes: u64,
        estimated_training_bytes: u64,
    },
    Step {
        step: usize,
        loss: Option<f32>,
        sample_path: Option<String>,
    },
    InferenceImage {
        width: u32,
        height: u32,
        channels: u32,
        pixels: Vec<u8>,
        checkpoint_path: String,
        seed: u64,
    },
    InferenceProgress {
        label: String,
        current: usize,
        total: usize,
    },
    TrainingState {
        paused: bool,
        lr: f32,
        batch_size: u32,
        total_steps: usize,
    },
    /// Live read-out of a perpetual run. Republished on every change, so the
    /// footer always shows what the worker actually did rather than what the
    /// UI asked for.
    PerpetualState(PerpetualStatus),
    SaveStatus {
        message: String,
        is_error: bool,
    },
    Error {
        message: String,
    },
    Done,
}

pub fn handle_key(app: &mut App, code: KeyCode) {
    match app.screen {
        Screen::ModelList => handle_model_list(app, code),
        Screen::TemplateSelector => handle_template_selector(app, code),
        Screen::ModelActions => handle_model_actions(app, code),
        Screen::RenameModel => handle_rename_model(app, code),
        Screen::DuplicateModel => handle_duplicate_model(app, code),
        Screen::DeleteConfirm => handle_delete_confirm(app, code),
        Screen::WeightSelector => handle_weight_selector(app, code),
        Screen::InputSize => handle_input_size(app, code),
        Screen::LayerBuilder => handle_layer_builder(app, code),
        Screen::InferenceParams => handle_inference_params(app, code),
        Screen::PerpetualParams => handle_perpetual_params(app, code),
        Screen::TrainingParams => handle_training_params(app, code),
        Screen::DatasetSelector => handle_dataset_selector(app, code),
        Screen::Monitor => handle_monitor(app, code),
        Screen::TrainingControl => handle_training_control(app, code),
        Screen::Resources => handle_resources(app, code),
    }
}

/// The Resources page.
///
/// `←`/`→` are the simulator — they move the batch through its stops and the
/// totals move with them. That is why this screen is NOT on the breadcrumb's
/// arrow path (`walks_the_path_by_arrow` is false for it): the two keys already
/// mean something here, and the repository's rule is that the breadcrumb never
/// steals a binding a screen already uses.
///
/// `[i]` swaps between the training and the inference graph — the difference is
/// not a scale factor but a different graph, and seeing both is most of the
/// point. `Esc` goes back up to the action menu, like every other detour.
fn handle_resources(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.screen = Screen::ModelActions,
        KeyCode::Left => {
            app.resources.batch = ResourcesState::smaller(app.resources.batch);
        }
        KeyCode::Right => {
            app.resources.batch = ResourcesState::larger(app.resources.batch);
        }
        KeyCode::Char('i') => {
            app.resources.inference = !app.resources.inference;
            app.resources.scroll = 0;
        }
        KeyCode::Down => app.resources.scroll = app.resources.scroll.saturating_add(1),
        KeyCode::Up => app.resources.scroll = app.resources.scroll.saturating_sub(1),
        KeyCode::Char('q') => app.should_quit = true,
        _ => {}
    }
}

/// The front door. `Esc` quits because there is nowhere above it to go — and
/// `←` does *not*, which is the one place the two "go back" keys differ.
fn handle_model_list(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Left => {
            app.path_back();
        }
        KeyCode::Right => app.path_forward(),
        KeyCode::Up => {
            if app.model_list.selected > 0 {
                app.model_list.selected -= 1;
            }
        }
        KeyCode::Down => {
            // Bounded on the row count, which includes the "new model" row —
            // a bound taken from `models.len()` would leave the template flow
            // drawn but unselectable, which is exactly the bug this codebase
            // keeps producing.
            if app.model_list.selected + 1 < app.model_list.entry_count() {
                app.model_list.selected += 1;
            }
        }
        KeyCode::Char('r') => app.refresh_model_list(),
        KeyCode::Enter => app.finish_model_list(),
        _ => {}
    }
}

fn handle_model_actions(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Left => {
            app.path_back();
        }
        KeyCode::Right => app.path_forward(),
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('e') => app.enter_layer_builder(),
        KeyCode::Char('r') => app.enter_resources(),
        KeyCode::Up => {
            if app.model_actions.selected > 0 {
                app.model_actions.selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.model_actions.selected + 1 < MODEL_ACTIONS.len() {
                app.model_actions.selected += 1;
            }
        }
        KeyCode::Enter => app.finish_model_actions(),
        _ => {}
    }
}

/// Every printable key is text here — a model name may contain a `q`, so `q`
/// cannot also mean quit. `Esc` is the way back.
fn handle_rename_model(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.rename_model.error = None;
            app.screen = Screen::ModelActions;
        }
        KeyCode::Backspace => app.handle_backspace_rename(),
        KeyCode::Enter => app.finish_rename(),
        KeyCode::Char(c) => app.handle_char_rename(c),
        _ => {}
    }
}

/// The name field always has focus, so every printable key is text here too —
/// the copy of `q-experiment` has to be nameable. What the copy *carries* is
/// picked with `↑`/`↓`, which are the only keys the field does not want.
fn handle_duplicate_model(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.duplicate_model.error = None;
            app.screen = Screen::ModelActions;
        }
        KeyCode::Up => {
            if app.duplicate_model.selected > 0 {
                app.duplicate_model.selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.duplicate_model.selected + 1 < DUPLICATE_CONTENT_CHOICES.len() {
                app.duplicate_model.selected += 1;
            }
        }
        KeyCode::Backspace => app.handle_backspace_duplicate(),
        KeyCode::Enter => app.finish_duplicate(),
        KeyCode::Char(c) => app.handle_char_duplicate(c),
        _ => {}
    }
}

/// Same rule as rename, and the same reason: what is typed here is compared
/// against the model's name, so it has to be able to *be* the model's name.
fn handle_delete_confirm(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.delete_confirm.typed.clear();
            app.delete_confirm.error = None;
            app.screen = Screen::ModelActions;
        }
        KeyCode::Backspace => app.handle_backspace_delete_confirm(),
        KeyCode::Enter => app.finish_delete(),
        KeyCode::Char(c) => app.handle_char_delete_confirm(c),
        _ => {}
    }
}

fn handle_template_selector(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Left => {
            app.path_back();
        }
        // `→` is deliberately inert here: forward from a template *creates a
        // model on disk*. `Enter` stays the only key that writes.
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => {
            if app.template_selector.selected > 0 {
                app.template_selector.selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.template_selector.selected + 1 < app.template_selector.templates.len() {
                app.template_selector.selected += 1;
            }
        }
        KeyCode::Enter => app.finish_template_selector(),
        _ => {}
    }
}

fn handle_weight_selector(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Left => {
            app.path_back();
        }
        KeyCode::Right => app.path_forward(),
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => {
            if app.weight_selector.selected > 0 {
                app.weight_selector.selected -= 1;
            }
        }
        KeyCode::Down => {
            let max = app.weight_selector.checkpoints.len();
            if app.weight_selector.selected < max {
                app.weight_selector.selected += 1;
            }
        }
        KeyCode::Enter => app.finish_weight_selector(),
        _ => {}
    }
}

fn handle_input_size(app: &mut App, code: KeyCode) {
    let max_field = INPUT_SIZE_FIELD_NAMES.len() - 1;
    match code {
        KeyCode::Esc => app.screen = Screen::LayerBuilder,
        KeyCode::Up => {
            if app.input_size.field_idx > 0 {
                app.input_size.field_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.input_size.field_idx < max_field {
                app.input_size.field_idx += 1;
            }
        }
        KeyCode::Backspace => app.handle_backspace_input_size(),
        KeyCode::Enter => {
            if app.input_size.field_idx < max_field {
                app.input_size.field_idx += 1;
            } else {
                if let Err(e) = app.finish_input_size() {
                    app.input_size.error = Some(e);
                }
            }
        }
        KeyCode::Char(c) => app.handle_char_input_size(c),
        _ => {}
    }
}

fn handle_layer_builder(app: &mut App, code: KeyCode) {
    match app.layer_builder.mode {
        LayerBuilderMode::Add => handle_lb_add(app, code),
        LayerBuilderMode::Browse => handle_lb_browse(app, code),
        LayerBuilderMode::Edit => handle_lb_edit(app, code),
    }
}

fn handle_lb_add(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.screen = Screen::ModelActions,
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('b') => app.finish_layer_builder(),
        KeyCode::Char('d') => app.delete_last_layer(),
        KeyCode::Char('e') => app.enter_browse_mode(),
        // The model's input geometry — the one thing the layer form cannot
        // express, and the screen that used to be drawn with nothing routing
        // to it.
        KeyCode::Char('i') => app.open_input_size(),
        KeyCode::Left => app.cycle_kind_backward(),
        KeyCode::Right => app.cycle_kind_forward(),
        KeyCode::Up => {
            if app.layer_builder.field_idx > 0 {
                app.layer_builder.field_idx -= 1;
            }
        }
        KeyCode::Down => {
            let max = app.layer_field_names().len().saturating_sub(1);
            if app.layer_builder.field_idx < max {
                app.layer_builder.field_idx += 1;
            }
        }
        KeyCode::Char(' ') => app.toggle_layer_field(),
        KeyCode::Enter => {
            if let Err(e) = app.try_add_layer() {
                app.layer_builder.error = Some(e);
            }
        }
        KeyCode::Backspace => app.handle_backspace_layer(),
        KeyCode::Char(c) => app.handle_char_layer(c),
        _ => {}
    }
}

fn handle_lb_browse(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('e') => app.exit_browse_mode(),
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => app.browse_move_up(),
        KeyCode::Down => app.browse_move_down(),
        KeyCode::Enter => app.enter_edit_mode(),
        KeyCode::Char('d') => app.delete_selected_layer(),
        _ => {}
    }
}

fn handle_lb_edit(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.cancel_edit(),
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Left => app.cycle_kind_backward(),
        KeyCode::Right => app.cycle_kind_forward(),
        KeyCode::Up => {
            if app.layer_builder.field_idx > 0 {
                app.layer_builder.field_idx -= 1;
            }
        }
        KeyCode::Down => {
            let max = app.layer_field_names().len().saturating_sub(1);
            if app.layer_builder.field_idx < max {
                app.layer_builder.field_idx += 1;
            }
        }
        KeyCode::Char(' ') => app.toggle_layer_field(),
        KeyCode::Enter => {
            if let Err(e) = app.confirm_layer_edit() {
                app.layer_builder.error = Some(e);
            }
        }
        KeyCode::Backspace => app.handle_backspace_layer(),
        KeyCode::Char(c) => app.handle_char_layer(c),
        _ => {}
    }
}

fn handle_dataset_selector(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.screen = Screen::TrainingParams,
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Left => app.cycle_dataset_backward(),
        KeyCode::Right => app.cycle_dataset_forward(),
        KeyCode::Up => {
            if app.training_params.selected_dataset > 0 {
                let next = app.training_params.selected_dataset - 1;
                app.select_dataset(next);
            }
        }
        KeyCode::Down => {
            if app.training_params.selected_dataset + 1 < app.training_params.datasets.len() {
                let next = app.training_params.selected_dataset + 1;
                app.select_dataset(next);
            }
        }
        KeyCode::Enter => {
            if let Err(e) = app.finish_dataset_selector() {
                app.training_params.error = Some(e);
            }
        }
        _ => {}
    }
}

fn handle_inference_params(app: &mut App, code: KeyCode) {
    let max_field = 3;
    match code {
        KeyCode::Esc => app.screen = Screen::WeightSelector,
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => {
            if app.inference_params.field_idx > 0 {
                app.inference_params.field_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.inference_params.field_idx < max_field {
                app.inference_params.field_idx += 1;
            }
        }
        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ')
            if app.inference_params.field_idx == 0 =>
        {
            app.toggle_inference_seed_mode();
        }
        KeyCode::Backspace => app.handle_backspace_inference(),
        KeyCode::Enter => {
            if app.inference_params.field_idx < max_field {
                app.inference_params.field_idx += 1;
            } else if let Err(e) = app.finish_inference_params() {
                app.inference_params.error = Some(e);
            }
        }
        KeyCode::Char(c) => app.handle_char_inference(c),
        _ => {}
    }
}

fn handle_perpetual_params(app: &mut App, code: KeyCode) {
    let max_field = PERPETUAL_PARAM_FIELD_NAMES.len() - 1;
    // The two toggles are the first and last field; the four in between are typed.
    let seed_toggle = 0;
    let regime_toggle = max_field;
    match code {
        KeyCode::Esc => app.screen = Screen::WeightSelector,
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => {
            if app.perpetual_params.field_idx > 0 {
                app.perpetual_params.field_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.perpetual_params.field_idx < max_field {
                app.perpetual_params.field_idx += 1;
            }
        }
        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ')
            if app.perpetual_params.field_idx == seed_toggle =>
        {
            app.toggle_perpetual_seed_mode();
        }
        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ')
            if app.perpetual_params.field_idx == regime_toggle =>
        {
            app.toggle_perpetual_regime();
        }
        KeyCode::Backspace => app.handle_backspace_perpetual(),
        KeyCode::Enter => {
            if app.perpetual_params.field_idx < max_field {
                app.perpetual_params.field_idx += 1;
            } else if let Err(e) = app.finish_perpetual_params() {
                app.perpetual_params.error = Some(e);
            }
        }
        KeyCode::Char(c) => app.handle_char_perpetual(c),
        _ => {}
    }
}

fn handle_training_params(app: &mut App, code: KeyCode) {
    let max_field = TRAINING_PARAM_FIELD_NAMES.len() - 1;
    match code {
        KeyCode::Esc => app.screen = Screen::WeightSelector,
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => {
            if app.training_params.field_idx > 0 {
                app.training_params.field_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.training_params.field_idx < max_field {
                app.training_params.field_idx += 1;
            }
        }
        // The random-weights opt-out is a toggle, so it answers to the toggle
        // keys the other forms already use — and to nothing else. `←`/`→` are
        // free on this screen precisely because it is a form: the breadcrumb
        // never claims them here (see `path_nav`).
        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ')
            if app.training_params.field_idx == TRAINING_RANDOM_WEIGHTS_FIELD =>
        {
            app.toggle_start_from_random();
        }
        KeyCode::Backspace => app.handle_backspace_training(),
        KeyCode::Enter => {
            if app.training_params.field_idx < max_field {
                app.training_params.field_idx += 1;
            } else if let Err(e) = app.finish_training_params() {
                app.training_params.error = Some(e);
            }
        }
        KeyCode::Char(c) => app.handle_char_training(c),
        _ => {}
    }
}

fn handle_monitor(app: &mut App, code: KeyCode) {
    // A perpetual run rebinds most of the monitor: `s` writes a PNG of what is
    // on screen rather than the model config, `r` re-seeds instead of asking
    // for a new run, and space pauses. These arms sit first and are gated on
    // the run mode, so every other mode keeps its keys untouched.
    if app.is_perpetual_run() && !app.monitor.done {
        match code {
            KeyCode::Up => {
                app.send_perpetual_command(TrainingControlCommand::NudgeRenoiseDepth(1));
                return;
            }
            KeyCode::Down => {
                app.send_perpetual_command(TrainingControlCommand::NudgeRenoiseDepth(-1));
                return;
            }
            KeyCode::Right => {
                app.send_perpetual_command(TrainingControlCommand::NudgeTempo(1));
                return;
            }
            KeyCode::Left => {
                app.send_perpetual_command(TrainingControlCommand::NudgeTempo(-1));
                return;
            }
            KeyCode::Char(' ') => {
                app.toggle_perpetual_pause();
                return;
            }
            KeyCode::Char('r') => {
                app.send_perpetual_command(TrainingControlCommand::Reseed);
                return;
            }
            KeyCode::Char('m') => {
                app.send_perpetual_command(TrainingControlCommand::ToggleRegime);
                return;
            }
            KeyCode::Char('s') => {
                app.send_perpetual_command(TrainingControlCommand::SaveImage);
                return;
            }
            KeyCode::Char('x') => {
                app.send_perpetual_command(TrainingControlCommand::ToggleView);
                return;
            }
            _ => {}
        }
    }

    match code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Char('s') => {
            if let Err(e) = app.trigger_monitor_save() {
                app.monitor.error = Some(e);
            }
        }
        KeyCode::Char('p') if !app.monitor.done && app.monitor.current_lr.is_some() => {
            app.toggle_training_pause();
        }
        KeyCode::Char('t') if !app.monitor.done && app.monitor.current_lr.is_some() => {
            if let Err(e) = app.open_training_control() {
                app.monitor.error = Some(e);
            }
        }
        KeyCode::Char('r') if app.monitor.done => app.request_restart(),
        // Gated on there being a source to show, not on the run mode: training
        // registers the model output buffer, inference registers its live
        // denoising frame, and `[v]` means the same thing in both.
        KeyCode::Char('v')
            if !app.monitor.done && super::visualiser_control::has_visualiser_source() =>
        {
            app.toggle_visualise()
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TempRoot;
    use crate::tui::app::{
        InferenceConfig, ModelAction, ModelConfig, PerpetualConfig, RunConfig, RunMode,
        TrainingConfig,
    };

    fn test_app(tag: &str) -> (TempRoot, App) {
        let temp = TempRoot::new(tag);
        let app = App::with_storage(temp.storage());
        (temp, app)
    }

    fn monitor_app(mode: RunMode) -> (TempRoot, App) {
        let (temp, mut app) = test_app("monitor");
        app.screen = Screen::Monitor;
        app.monitor.model_config = Some(ModelConfig {
            model_name: Some("unit-test-monitor".to_string()),
            input_size: (32, 32, 5),
            layers: Vec::new(),
            inference: InferenceConfig::default(),
            run: RunConfig { mode },
        });
        (temp, app)
    }

    /// Found by driving the real TUI: `Perpetual` was drawn as a third option
    /// but the cursor stopped at index 1, so it could never be selected. The
    /// bound now comes from the choice list itself — and the list has since
    /// grown the two manager actions, which the cursor must reach too.
    #[test]
    fn the_action_cursor_reaches_every_action() {
        let (_temp, mut app) = test_app("action-cursor");
        app.screen = Screen::ModelActions;
        app.model_actions.selected = 0;

        for expected in 1..MODEL_ACTIONS.len() {
            handle_key(&mut app, KeyCode::Down);
            assert_eq!(app.model_actions.selected, expected);
        }
        handle_key(&mut app, KeyCode::Down);
        assert_eq!(
            app.model_actions.selected,
            MODEL_ACTIONS.len() - 1,
            "cursor ran past the last action"
        );
        assert_eq!(
            ModelAction::from_index(app.model_actions.selected),
            Some(ModelAction::Delete)
        );
    }

    /// The cursor has to reach the "new model" row, which sits *after* the
    /// models — a bound taken from `models.len()` would draw it and never let
    /// it be selected.
    #[test]
    fn the_model_list_cursor_reaches_the_new_model_row() {
        let (_temp, mut app) = test_app("list-cursor");
        app.finish_template_selector();
        app.refresh_model_list();
        app.screen = Screen::ModelList;
        app.model_list.selected = 0;
        assert_eq!(app.model_list.models.len(), 1);

        handle_key(&mut app, KeyCode::Down);

        assert!(app.model_list.is_new_model_selected());
        handle_key(&mut app, KeyCode::Down);
        assert_eq!(
            app.model_list.selected,
            app.model_list.entry_count() - 1,
            "cursor ran past the last row"
        );
    }

    /// A model name may contain any of these; on the two manager screens they
    /// are text, not commands. `q` in particular quits everywhere else.
    #[test]
    fn the_manager_forms_treat_every_printable_key_as_text() {
        let (_temp, mut app) = test_app("manager-typing");
        app.screen = Screen::RenameModel;
        for c in "q-model.2".chars() {
            handle_key(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.rename_model.input, "q-model.2");
        assert!(!app.should_quit, "[q] must be typable in a model name");

        app.screen = Screen::DuplicateModel;
        for c in "quiet-copy".chars() {
            handle_key(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.duplicate_model.input, "quiet-copy");
        assert!(!app.should_quit, "[q] must be typable in a copy's name");
        handle_key(&mut app, KeyCode::Backspace);
        assert_eq!(app.duplicate_model.input, "quiet-cop");
        // …and `e` is a letter here, not the shortcut to the layer builder.
        assert_eq!(app.screen, Screen::DuplicateModel);

        app.screen = Screen::DeleteConfirm;
        for c in "quiet".chars() {
            handle_key(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.delete_confirm.typed, "quiet");
        assert!(!app.should_quit);

        handle_key(&mut app, KeyCode::Backspace);
        assert_eq!(app.delete_confirm.typed, "quie");
    }

    /// Leaving the delete screen must not leave the typed confirmation behind:
    /// coming back to it half-confirmed is one keystroke from a deletion the
    /// user never asked for twice.
    #[test]
    fn leaving_the_delete_screen_clears_what_was_typed() {
        let (_temp, mut app) = test_app("delete-esc");
        app.screen = Screen::DeleteConfirm;
        app.delete_confirm.typed = "almost".to_string();

        handle_key(&mut app, KeyCode::Esc);

        assert_eq!(app.screen, Screen::ModelActions);
        assert!(app.delete_confirm.typed.is_empty());
    }

    #[test]
    fn perpetual_monitor_keys_steer_the_drift() {
        let (_temp, mut app) = monitor_app(RunMode::Perpetual(PerpetualConfig::default()));

        for (key, expected) in [
            (KeyCode::Up, TrainingControlCommand::NudgeRenoiseDepth(1)),
            (KeyCode::Down, TrainingControlCommand::NudgeRenoiseDepth(-1)),
            (KeyCode::Right, TrainingControlCommand::NudgeTempo(1)),
            (KeyCode::Left, TrainingControlCommand::NudgeTempo(-1)),
            (KeyCode::Char('r'), TrainingControlCommand::Reseed),
            (KeyCode::Char('m'), TrainingControlCommand::ToggleRegime),
            (KeyCode::Char('s'), TrainingControlCommand::SaveImage),
            (KeyCode::Char('x'), TrainingControlCommand::ToggleView),
        ] {
            handle_key(&mut app, key);
            assert_eq!(
                app.drain_monitor_control_commands(),
                vec![expected],
                "key {key:?} did not reach the drift"
            );
        }

        handle_key(&mut app, KeyCode::Char(' '));
        assert_eq!(
            app.drain_monitor_control_commands(),
            vec![TrainingControlCommand::SetPaused(true)]
        );

        handle_key(&mut app, KeyCode::Char('q'));
        assert!(app.should_quit, "[q] must still leave a perpetual run");
    }

    /// The perpetual bindings rebind keys that mean something else everywhere
    /// else — `r` asks for a new run, `s` saves the model config. They are gated
    /// on the run mode, and this is what proves the gate holds.
    #[test]
    fn a_training_monitor_keeps_its_own_bindings() {
        let (_temp, mut app) = monitor_app(RunMode::Train(TrainingConfig {
            lr: 0.01,
            batch_size: 1,
            steps: 10,
            dataset_path: ".".to_string(),
            loss: crate::tui::app::LossMethod::MeanSquared,
            checkpoint_path: None,
            load_checkpoint: false,
            optimizer: Default::default(),
            weight_init: Default::default(),
            loss_weighting: Default::default(),
            ema_decay: None,
        }));

        for key in [
            KeyCode::Up,
            KeyCode::Down,
            KeyCode::Left,
            KeyCode::Right,
            KeyCode::Char('m'),
            KeyCode::Char('r'),
            KeyCode::Char('x'),
        ] {
            handle_key(&mut app, key);
            assert!(
                app.drain_monitor_control_commands().is_empty(),
                "key {key:?} leaked a perpetual command into a training run"
            );
        }
        assert!(!app.should_quit);
    }
}

fn handle_training_control(app: &mut App, code: KeyCode) {
    let max_field = 2;
    match code {
        KeyCode::Esc => {
            app.screen = Screen::Monitor;
        }
        KeyCode::Up => {
            if app.training_control.field_idx > 0 {
                app.training_control.field_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.training_control.field_idx < max_field {
                app.training_control.field_idx += 1;
            }
        }
        KeyCode::Backspace => app.handle_backspace_training_control(),
        KeyCode::Enter => {
            if app.training_control.field_idx < max_field {
                app.training_control.field_idx += 1;
            } else if let Err(e) = app.finish_training_control() {
                app.training_control.error = Some(e);
            }
        }
        KeyCode::Char(c) => app.handle_char_training_control(c),
        _ => {}
    }
}
