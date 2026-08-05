//! File purpose: Implements events behavior for the terminal user interface flow.

use super::app::{
    App, HOME_CHOICES, INPUT_SIZE_FIELD_NAMES, LayerBuilderMode, PERPETUAL_PARAM_FIELD_NAMES,
    PerpetualStatus, RUN_MODE_CHOICES, Screen, TrainingControlCommand,
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
        Screen::Home => handle_home(app, code),
        Screen::LoadPath => handle_load_path(app, code),
        Screen::TemplateSelector => handle_template_selector(app, code),
        Screen::WeightSelector => handle_weight_selector(app, code),
        Screen::InputSize => handle_input_size(app, code),
        Screen::LayerBuilder => handle_layer_builder(app, code),
        Screen::ModeSelector => handle_mode_selector(app, code),
        Screen::InferenceParams => handle_inference_params(app, code),
        Screen::PerpetualParams => handle_perpetual_params(app, code),
        Screen::TrainingParams => handle_training_params(app, code),
        Screen::DatasetSelector => handle_dataset_selector(app, code),
        Screen::Monitor => handle_monitor(app, code),
        Screen::TrainingControl => handle_training_control(app, code),
    }
}

fn handle_home(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => {
            if app.home.selected > 0 {
                app.home.selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.home.selected + 1 < HOME_CHOICES.len() {
                app.home.selected += 1;
            }
        }
        KeyCode::Enter => app.finish_home(),
        _ => {}
    }
}

fn handle_load_path(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.screen = Screen::Home,
        KeyCode::Up => {
            if app.load_path.selected > 0 {
                app.load_path.selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.load_path.selected + 1 < app.load_path.models.len() {
                app.load_path.selected += 1;
            }
        }
        KeyCode::Enter => app.finish_load_path(),
        _ => {}
    }
}

fn handle_template_selector(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.screen = Screen::Home,
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
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
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
        KeyCode::Esc => app.should_quit = true,
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
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('b') => app.finish_layer_builder(),
        KeyCode::Char('d') => app.delete_last_layer(),
        KeyCode::Char('e') => app.enter_browse_mode(),
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

fn handle_mode_selector(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('e') => app.enter_layer_builder_from_mode(),
        KeyCode::Up => {
            if app.mode_selector.selected > 0 {
                app.mode_selector.selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.mode_selector.selected + 1 < RUN_MODE_CHOICES.len() {
                app.mode_selector.selected += 1;
            }
        }
        KeyCode::Enter => app.finish_mode_selector(),
        _ => {}
    }
}

fn handle_dataset_selector(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
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
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
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
        KeyCode::Esc => app.screen = Screen::ModeSelector,
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
    let max_field = 2;
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
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
    use crate::tui::app::{
        InferenceConfig, ModelConfig, PerpetualConfig, RunConfig, RunMode, TrainingConfig,
    };

    fn monitor_app(mode: RunMode) -> App {
        let mut app = App::new();
        app.screen = Screen::Monitor;
        app.monitor.model_config = Some(ModelConfig {
            model_name: Some("unit-test-monitor".to_string()),
            input_size: (32, 32, 5),
            layers: Vec::new(),
            inference: InferenceConfig::default(),
            run: RunConfig { mode },
        });
        app
    }

    /// Found by driving the real TUI: `Perpetual` was drawn as a third option
    /// but the cursor stopped at index 1, so it could never be selected. The
    /// bound now comes from the choice list itself.
    #[test]
    fn the_mode_cursor_reaches_every_run_mode() {
        let mut app = App::new();
        app.screen = Screen::ModeSelector;
        app.mode_selector.selected = 0;

        for expected in 1..RUN_MODE_CHOICES.len() {
            handle_key(&mut app, KeyCode::Down);
            assert_eq!(app.mode_selector.selected, expected);
        }
        handle_key(&mut app, KeyCode::Down);
        assert_eq!(
            app.mode_selector.selected,
            RUN_MODE_CHOICES.len() - 1,
            "cursor ran past the last mode"
        );
    }

    #[test]
    fn perpetual_monitor_keys_steer_the_drift() {
        let mut app = monitor_app(RunMode::Perpetual(PerpetualConfig::default()));

        for (key, expected) in [
            (KeyCode::Up, TrainingControlCommand::NudgeRenoiseDepth(1)),
            (KeyCode::Down, TrainingControlCommand::NudgeRenoiseDepth(-1)),
            (KeyCode::Right, TrainingControlCommand::NudgeTempo(1)),
            (KeyCode::Left, TrainingControlCommand::NudgeTempo(-1)),
            (KeyCode::Char('r'), TrainingControlCommand::Reseed),
            (KeyCode::Char('m'), TrainingControlCommand::ToggleRegime),
            (KeyCode::Char('s'), TrainingControlCommand::SaveImage),
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
        let mut app = monitor_app(RunMode::Train(TrainingConfig {
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
        }));

        for key in [
            KeyCode::Up,
            KeyCode::Down,
            KeyCode::Left,
            KeyCode::Right,
            KeyCode::Char('m'),
            KeyCode::Char('r'),
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
