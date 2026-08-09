//! File purpose: Module entry point for tui; wires submodules and shared exports.

pub mod app;
pub mod events;
pub mod help;
#[cfg(test)]
mod nav_tests;
pub mod ui;
pub mod visualiser_control;

pub use app::{App, ModelAction, MonitorImage, PerpetualStatus, Screen};
// Re-exported for call sites that already speak in terms of `tui::…`; the types
// themselves are engine-side (`batlab_core::config`) because a model
// description must survive without a terminal.
pub use batlab_core::config::{
    ActivationMethod, InferenceConfig, LayerDraft, LayerKind, LossMethod, ModelConfig, PaddingMode,
    PerpetualConfig, RunConfig, RunMode, TrainingConfig, TrainingControlCommand,
};
pub use events::TrainingEvent;
pub use visualiser_control::{
    clear_visualiser_source, has_visualiser_source, register_visualiser_source,
    set_visualiser_visible, toggle_visualiser, warmup_visualiser,
};

use std::io;
use std::sync::mpsc::{Receiver, Sender};

/// Outcome returned by [`run_monitor`].
pub enum MonitorOutcome {
    /// The user quit without requesting a new training run.
    Done,
    /// The user pressed `[r]` and configured a new training run.
    /// The returned [`ModelConfig`] should be used to start the next run.
    Restart(ModelConfig),
}

pub fn run() -> Result<ModelConfig, Box<dyn std::error::Error>> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = App::new();
    let result = run_builder_loop(&mut terminal, &mut app);

    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    result
}

fn run_builder_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<ModelConfig, Box<dyn std::error::Error>> {
    use crossterm::event::{Event, KeyEventKind, poll, read};
    use std::time::Duration;

    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        if poll(Duration::from_millis(16))? {
            if let Event::Key(key) = read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                events::handle_key(app, key.code);
            }
        }

        if app.should_quit {
            return Err("quit".into());
        }

        if let Some(run) = app.run_config.take() {
            return Ok(app.compose_run_config(run));
        }
    }
}

pub fn run_monitor(
    config: ModelConfig,
    rx: Receiver<TrainingEvent>,
    control_tx: Option<Sender<TrainingControlCommand>>,
) -> Result<MonitorOutcome, Box<dyn std::error::Error>> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = App::new();
    app.screen = Screen::Monitor;
    app.layer_builder.layers = config.layers.clone();
    app.layer_builder.model_input = config.input_size;
    app.active_model_name = config.model_name.clone();
    // What the manager refuses to rename or delete for as long as this run is
    // alive. Set here rather than in the builder: this is the moment a worker
    // actually holds the model's directory.
    app.running_model = config.model_name.clone();
    app.monitor.model_config = Some(config.clone());
    // Model-level, so it is read whatever the run mode — same reason
    // `apply_loaded_model` reads it outside the match.
    app.seed_dataset = config.seed_dataset.clone();
    match &config.run.mode {
        RunMode::Infer => {
            app.model_actions.selected = ModelAction::Infer.index();
            app.inference_params.random_seed = config.inference.random_seed;
            app.inference_params.fields[0] = config.inference.seed.unwrap_or(0).to_string();
            app.inference_params.fields[1] = config.inference.denoising_paths.max(1).to_string();
            app.inference_params.fields[2] = config.inference.denoise_magnitude.to_string();
            app.inference_params.field_idx = 0;
            app.inference_params.error = None;
        }
        RunMode::Perpetual(pc) => {
            app.model_actions.selected = ModelAction::Perpetual.index();
            app.perpetual_params.random_seed = pc.random_seed;
            app.perpetual_params.regime = pc.regime;
            app.perpetual_params.fields[0] = pc.seed.unwrap_or(0).to_string();
            app.perpetual_params.fields[1] = pc.denoise_magnitude.to_string();
            app.perpetual_params.fields[2] = pc.renoise_depth.to_string();
            app.perpetual_params.fields[3] = pc.tempo.to_string();
            app.perpetual_params.field_idx = 0;
            app.perpetual_params.error = None;
            app.selected_checkpoint_path = pc.checkpoint.clone();
        }
        RunMode::Train(tc) => {
            app.model_actions.selected = ModelAction::Train.index();
            app.monitor.total_steps = tc.steps;
            app.monitor.current_lr = Some(tc.lr);
            app.monitor.current_batch_size = Some(tc.batch_size);
            app.monitor.is_training_paused = false;
            app.training_params.fields = vec![
                tc.lr.to_string(),
                tc.batch_size.to_string(),
                tc.steps.to_string(),
                crate::tui::app::ema_decay_field(tc.ema_decay),
                tc.dataset_path.clone(),
            ];
            app.selected_checkpoint_path = tc.checkpoint_path.clone();
            app.load_checkpoint_on_start = tc.load_checkpoint;
            app.sync_selected_dataset_from_field();
        }
    }

    let outcome = run_monitor_session(&mut terminal, &mut app, rx, control_tx);

    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    outcome
}

fn run_monitor_session(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    app: &mut App,
    rx: Receiver<TrainingEvent>,
    control_tx: Option<Sender<TrainingControlCommand>>,
) -> Result<MonitorOutcome, Box<dyn std::error::Error>> {
    visualiser_control::warmup_visualiser();
    run_monitor_loop(terminal, app, rx, control_tx)?;
    visualiser_control::clear_visualiser_source();

    if app.monitor.restart_training {
        // Back to the action menu with the same model in hand, so the user can
        // pick train/infer/perpetual again — or now rename or delete it, which
        // the guard allowed the moment the run reported itself done. The
        // checkpoint list is re-read: the run may well have just written one.
        app.monitor = Default::default();
        app.running_model = None;
        app.should_quit = false;
        app.enter_model_actions();

        match run_builder_loop(terminal, app) {
            Ok(new_config) => return Ok(MonitorOutcome::Restart(new_config)),
            Err(_) => {}
        }
    }

    Ok(MonitorOutcome::Done)
}

fn run_monitor_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    app: &mut App,
    rx: Receiver<TrainingEvent>,
    control_tx: Option<Sender<TrainingControlCommand>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use crossterm::event::{Event, KeyEventKind, poll, read};
    use std::time::Duration;

    loop {
        while let Ok(event) = rx.try_recv() {
            match event {
                TrainingEvent::ResourceReport {
                    max_buffer_bytes,
                    max_storage_binding_bytes,
                    estimated_training_bytes,
                } => {
                    app.monitor.max_buffer_bytes = Some(max_buffer_bytes);
                    app.monitor.max_storage_binding_bytes = Some(max_storage_binding_bytes);
                    app.monitor.estimated_training_bytes = Some(estimated_training_bytes);
                }
                TrainingEvent::Step {
                    step,
                    loss,
                    sample_path,
                } => {
                    app.monitor.step = step;
                    if let Some(loss) = loss {
                        app.monitor.loss_history.push(loss as f64);
                    }
                    if let Some(path) = sample_path {
                        app.monitor.last_sample_path = Some(path);
                    }
                }
                TrainingEvent::InferenceImage {
                    width,
                    height,
                    channels,
                    pixels,
                    checkpoint_path,
                    seed,
                } => {
                    app.monitor.inference_image = Some(MonitorImage {
                        width,
                        height,
                        channels,
                        pixels,
                    });
                    app.monitor.inference_checkpoint_path = Some(checkpoint_path);
                    app.monitor.inference_seed = Some(seed);
                    app.monitor.loading_progress = None;
                }
                TrainingEvent::InferenceProgress {
                    label,
                    current,
                    total,
                } => {
                    app.monitor.loading_progress = Some(app::LoadingProgress {
                        label,
                        current,
                        total,
                    });
                }
                TrainingEvent::TrainingState {
                    paused,
                    lr,
                    batch_size,
                    total_steps,
                } => {
                    app.monitor.is_training_paused = paused;
                    app.monitor.current_lr = Some(lr);
                    app.monitor.current_batch_size = Some(batch_size);
                    app.monitor.total_steps = total_steps;
                    if let Some(config) = app.monitor.model_config.as_mut()
                        && let RunMode::Train(ref mut train) = config.run.mode
                    {
                        train.lr = lr;
                        train.batch_size = batch_size;
                        train.steps = total_steps;
                    }
                }
                TrainingEvent::PerpetualState(status) => {
                    app.monitor.is_training_paused = status.paused;
                    app.monitor.perpetual = Some(status);
                }
                TrainingEvent::SaveStatus { message, is_error } => {
                    if is_error {
                        app.monitor.error = Some(message);
                    } else {
                        app.monitor.save_status = Some(message);
                    }
                }
                TrainingEvent::Error { message } => {
                    app.monitor.error = Some(message);
                    app.monitor.loading_progress = None;
                    app.monitor.done = true;
                }
                TrainingEvent::Done => {
                    app.monitor.loading_progress = None;
                    app.monitor.done = true;
                }
            }
        }

        terminal.draw(|f| ui::draw(f, app))?;

        if poll(Duration::from_millis(16))? {
            if let Event::Key(key) = read()? {
                if key.kind == KeyEventKind::Press {
                    events::handle_key(app, key.code);
                }
            }
        }

        for command in app.drain_monitor_control_commands() {
            if let Some(tx) = control_tx.as_ref() {
                if tx.send(command).is_err() {
                    app.monitor.error = Some("training control channel disconnected".to_string());
                }
            }
        }

        if app.should_quit || app.monitor.restart_training {
            break;
        }
    }

    Ok(())
}
