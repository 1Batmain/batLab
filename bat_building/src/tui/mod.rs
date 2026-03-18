//! File purpose: Module entry point for tui; wires submodules and shared exports.

pub mod app;
pub mod events;
pub mod storage;
pub mod ui;
pub mod visualiser_control;

pub use app::{
    ActivationMethod, App, InferenceConfig, LayerDraft, LayerKind, LossMethod, ModelConfig,
    MonitorImage, PaddingMode, RunConfig, RunMode, Screen, TrainingConfig, TrainingControlCommand,
};
pub use events::TrainingEvent;
pub use visualiser_control::{
    clear_visualiser_source, register_visualiser_source, set_visualiser_visible, toggle_visualiser,
    warmup_visualiser,
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
            let denoising_paths = app
                .inference_params
                .fields
                .get(1)
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(1)
                .max(1);
            let denoise_magnitude = app
                .inference_params
                .fields
                .get(2)
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(1.0)
                .max(1e-6);
            let inference = InferenceConfig {
                random_seed: app.inference_params.random_seed,
                seed: if app.inference_params.random_seed {
                    None
                } else {
                    app.inference_params
                        .fields
                        .first()
                        .and_then(|value| value.parse::<u64>().ok())
                },
                denoising_paths,
                denoise_magnitude,
            };
            return Ok(ModelConfig {
                model_name: app.active_model_name.clone(),
                input_size: app.layer_builder.model_input,
                layers: app.layer_builder.layers.clone(),
                inference,
                run,
            });
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
    app.monitor.model_config = Some(config.clone());
    match &config.run.mode {
        RunMode::Infer => {
            app.mode_selector.selected = 0;
            app.inference_params.random_seed = config.inference.random_seed;
            app.inference_params.fields[0] = config.inference.seed.unwrap_or(0).to_string();
            app.inference_params.fields[1] = config.inference.denoising_paths.max(1).to_string();
            app.inference_params.fields[2] = config.inference.denoise_magnitude.to_string();
            app.inference_params.field_idx = 0;
            app.inference_params.error = None;
        }
        RunMode::Train(tc) => {
            app.mode_selector.selected = 1;
            app.monitor.total_steps = tc.steps;
            app.monitor.current_lr = Some(tc.lr);
            app.monitor.current_batch_size = Some(tc.batch_size);
            app.monitor.is_training_paused = false;
            app.training_params.fields = vec![
                tc.lr.to_string(),
                tc.batch_size.to_string(),
                tc.steps.to_string(),
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
        // Reset to mode selection with the same architecture so the user can
        // quickly pick infer/train again and choose a dataset for training.
        app.screen = Screen::ModeSelector;
        app.monitor = Default::default();
        app.should_quit = false;

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
