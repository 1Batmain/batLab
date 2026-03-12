use crate::app::{App, INPUT_SIZE_FIELD_NAMES, Screen, TRAINING_PARAM_FIELD_NAMES};
use crossterm::event::KeyCode;

#[derive(Debug)]
pub enum TrainingEvent {
    Step { step: usize, loss: f32 },
    Done,
}

pub fn handle_key(app: &mut App, code: KeyCode) {
    match app.screen {
        Screen::Home => handle_home(app, code),
        Screen::LoadPath => handle_load_path(app, code),
        Screen::InputSize => handle_input_size(app, code),
        Screen::LayerBuilder => handle_layer_builder(app, code),
        Screen::ModeSelector => handle_mode_selector(app, code),
        Screen::TrainingParams => handle_training_params(app, code),
        Screen::Monitor => handle_monitor(app, code),
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
            if app.home.selected < 1 {
                app.home.selected += 1;
            }
        }
        KeyCode::Enter => app.finish_home(),
        _ => {}
    }
}

fn handle_load_path(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.should_quit = true,
        KeyCode::Enter => app.finish_load_path(),
        KeyCode::Backspace => app.handle_backspace_load_path(),
        KeyCode::Char(c) => app.handle_char_load_path(c),
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
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('b') => app.finish_layer_builder(),
        KeyCode::Char('d') => app.delete_last_layer(),
        KeyCode::Char('a') => {
            if let Err(e) = app.try_add_layer() {
                app.layer_builder.error = Some(e);
            }
        }
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

fn handle_mode_selector(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.should_quit = true,
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

fn handle_training_params(app: &mut App, code: KeyCode) {
    let max_field = TRAINING_PARAM_FIELD_NAMES.len() - 1;
    match code {
        KeyCode::Esc => app.should_quit = true,
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
            } else {
                if let Err(e) = app.finish_training_params() {
                    app.training_params.error = Some(e);
                }
            }
        }
        KeyCode::Char(c) => app.handle_char_training(c),
        _ => {}
    }
}

fn handle_monitor(app: &mut App, code: KeyCode) {
    if matches!(code, KeyCode::Char('q') | KeyCode::Esc) {
        app.should_quit = true;
    }
}
