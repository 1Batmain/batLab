pub mod app;
pub mod events;
pub mod ui;

pub use app::{
    ActivationMethod, App, LayerDraft, LayerKind, LossMethod, ModelConfig, PaddingMode, RunConfig,
    RunMode, Screen, TrainingConfig,
};
pub use events::TrainingEvent;

use std::io;
use std::sync::mpsc::Receiver;

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
    use crossterm::event::{poll, read, Event, KeyEventKind};
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
) -> Result<(), Box<dyn std::error::Error>> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = App::new();
    app.screen = Screen::Monitor;
    app.layer_builder.layers = config.layers;
    app.layer_builder.model_input = config.input_size;
    if let RunMode::Train(ref tc) = config.run.mode {
        app.monitor.total_steps = tc.steps;
    }

    let result = run_monitor_loop(&mut terminal, &mut app, rx);

    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    result
}

fn run_monitor_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    app: &mut App,
    rx: Receiver<TrainingEvent>,
) -> Result<(), Box<dyn std::error::Error>> {
    use crossterm::event::{poll, read, Event, KeyEventKind};
    use std::time::Duration;

    loop {
        while let Ok(event) = rx.try_recv() {
            match event {
                TrainingEvent::Step { step, loss } => {
                    app.monitor.step = step;
                    app.monitor.loss_history.push(loss as f64);
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

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
