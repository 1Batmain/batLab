pub mod app;
pub mod events;
pub mod storage;
pub mod ui;

pub use app::{
    ActivationMethod, App, LayerDraft, LayerKind, LossMethod, ModelConfig, PaddingMode, RunConfig,
    RunMode, Screen, TrainingConfig,
};
pub use events::TrainingEvent;

use std::io;
use std::sync::mpsc::Receiver;

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
            return Ok(ModelConfig {
                model_name: app.active_model_name.clone(),
                input_size: app.layer_builder.model_input,
                layers: app.layer_builder.layers.clone(),
                run,
            });
        }
    }
}

pub fn run_monitor(
    config: ModelConfig,
    rx: Receiver<TrainingEvent>,
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
        }
        RunMode::Train(tc) => {
            app.mode_selector.selected = 1;
            app.monitor.total_steps = tc.steps;
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

    let outcome = run_monitor_session(&mut terminal, &mut app, rx);

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
) -> Result<MonitorOutcome, Box<dyn std::error::Error>> {
    run_monitor_loop(terminal, app, rx)?;

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
                TrainingEvent::Error { message } => {
                    app.monitor.error = Some(message);
                    app.monitor.done = true;
                }
                TrainingEvent::Done => {
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

        if app.should_quit || app.monitor.restart_training {
            break;
        }
    }

    Ok(())
}
