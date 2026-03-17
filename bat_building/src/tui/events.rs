//! File purpose: Implements events behavior for the terminal user interface flow.

use super::app::{App, INPUT_SIZE_FIELD_NAMES, LayerBuilderMode, Screen};
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
    TrainingState {
        paused: bool,
        lr: f32,
        batch_size: u32,
        total_steps: usize,
    },
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
        Screen::TrainingParams => handle_training_params(app, code),
        Screen::DatasetSelector => handle_dataset_selector(app, code),
        Screen::Monitor => handle_monitor(app, code),
        Screen::TrainingControl => handle_training_control(app, code),
    }
}

fn handle_home(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Enter => app.finish_home(),
        _ => {}
    }
}

fn handle_load_path(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.should_quit = true,
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
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
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
            if app.mode_selector.selected < 1 {
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
        KeyCode::Char('v') if !app.monitor.done && app.monitor.current_lr.is_some() => {
            app.toggle_visualise()
        }
        _ => {}
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
