use super::app::{
    App, INPUT_SIZE_FIELD_NAMES, LayerBuilderMode, LayerKind, RunMode, Screen,
    TRAINING_PARAM_FIELD_NAMES,
};
use ratatui::{prelude::*, widgets::*};

pub fn draw(f: &mut Frame, app: &App) {
    match app.screen {
        Screen::Home => draw_home(f, app),
        Screen::LoadPath => draw_load_path(f, app),
        Screen::TemplateSelector => draw_template_selector(f, app),
        Screen::InputSize => draw_input_size(f, app),
        Screen::LayerBuilder => draw_layer_builder(f, app),
        Screen::ModeSelector => draw_mode_selector(f, app),
        Screen::TrainingParams => draw_training_params(f, app),
        Screen::DatasetSelector => draw_dataset_selector(f, app),
        Screen::Monitor => draw_monitor(f, app),
        Screen::SaveModel => {
            draw_monitor(f, app);
            draw_save_model(f, app);
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(r);

    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(popup_layout[1])[1]
}

fn hint_bar<'a>(text: &'a str) -> Paragraph<'a> {
    Paragraph::new(text).style(Style::default().fg(Color::DarkGray))
}

fn focused_label(focused: bool) -> Style {
    if focused {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Gray)
    }
}

fn focused_value(focused: bool) -> Style {
    if focused {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    }
}

/// Generic "choose one of N options" screen.
fn draw_choice_screen(f: &mut Frame, title: &str, choices: &[&str], selected: usize, hint: &str) {
    let area = f.area();
    let popup = centered_rect(44, 50, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {title} "))
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, choice) in choices.iter().enumerate() {
        let (prefix, style) = if i == selected {
            (
                "  > ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            ("    ", Style::default().fg(Color::Gray))
        };
        lines.push(Line::from(Span::styled(
            format!("  {}{}", prefix, choice),
            style,
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  {hint}"),
        Style::default().fg(Color::DarkGray),
    )));

    f.render_widget(Paragraph::new(lines), inner);
}

/// Generic form screen (text fields).
fn draw_form_screen(
    f: &mut Frame,
    title: &str,
    field_names: &[&str],
    fields: &[String],
    field_idx: usize,
    error: Option<&str>,
    hint: &str,
) {
    let area = f.area();
    let popup = centered_rect(56, 70, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {title} "))
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, name) in field_names.iter().enumerate() {
        let focused = i == field_idx;
        let cursor = if focused { "\u{2588}" } else { "" };
        lines.push(Line::from(vec![
            Span::styled(format!("  {:>16} : ", name), focused_label(focused)),
            Span::styled(
                format!(
                    "{}{}",
                    fields.get(i).map(String::as_str).unwrap_or(""),
                    cursor
                ),
                focused_value(focused),
            ),
        ]));
    }

    lines.push(Line::from(""));
    if let Some(err) = error {
        lines.push(Line::from(Span::styled(
            format!("  \u{2717} {err}"),
            Style::default().fg(Color::Red),
        )));
        lines.push(Line::from(""));
    }
    lines.push(Line::from(Span::styled(
        format!("  {hint}"),
        Style::default().fg(Color::DarkGray),
    )));

    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Screen: Home
// ---------------------------------------------------------------------------

fn draw_home(f: &mut Frame, app: &App) {
    draw_choice_screen(
        f,
        "batBuilder",
        &["Load Saved Model", "Use Template"],
        app.home.selected,
        "[arrow] select  [Enter] confirm  [q] quit",
    );
}

// ---------------------------------------------------------------------------
// Screen: Load Path
// ---------------------------------------------------------------------------

fn draw_load_path(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup = centered_rect(60, 55, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Load Model ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines = vec![Line::from("")];
    if app.load_path.models.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No saved models found in saved_models/",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (index, model) in app.load_path.models.iter().enumerate() {
            let style = if index == app.load_path.selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            lines.push(Line::from(Span::styled(
                format!(
                    "  {} {}  ({}x{}x{}, {} layers)",
                    if index == app.load_path.selected {
                        ">"
                    } else {
                        " "
                    },
                    model.name,
                    model.input_size.0,
                    model.input_size.1,
                    model.input_size.2,
                    model.layer_count
                ),
                style,
            )));
        }
    }
    if let Some(error) = &app.load_path.error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ {error}"),
            Style::default().fg(Color::Red),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  [arrow] select  [Enter] load  [Esc] quit",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_template_selector(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup = centered_rect(70, 62, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Model Templates ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines = vec![Line::from("")];
    if app.template_selector.templates.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No templates available",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (index, template) in app.template_selector.templates.iter().enumerate() {
            let selected = index == app.template_selector.selected;
            let style = if selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            let marker = if selected { ">" } else { " " };
            lines.push(Line::from(Span::styled(
                format!("  {} {}", marker, template.name),
                style,
            )));
            lines.push(Line::from(Span::styled(
                format!("      {}", template.description),
                Style::default().fg(Color::DarkGray),
            )));
            lines.push(Line::from(""));
        }
    }

    if let Some(error) = app.template_selector.error.as_deref() {
        lines.push(Line::from(Span::styled(
            format!("  ✗ {error}"),
            Style::default().fg(Color::Red),
        )));
        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        "  [arrow] select  [Enter] continue  [Esc] quit",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Screen: Input Size
// ---------------------------------------------------------------------------

fn draw_input_size(f: &mut Frame, app: &App) {
    draw_form_screen(
        f,
        "Model Input Size",
        &INPUT_SIZE_FIELD_NAMES,
        &app.input_size.fields,
        app.input_size.field_idx,
        app.input_size.error.as_deref(),
        "[arrow] field  [0-9] type  [Enter] next/confirm  [Backspace] del  [Esc] quit",
    );
}

// ---------------------------------------------------------------------------
// Screen: Layer Builder
// ---------------------------------------------------------------------------

fn draw_layer_builder(f: &mut Frame, app: &App) {
    let area = f.area();

    match app.layer_builder.mode {
        LayerBuilderMode::Browse => draw_lb_browse_mode(f, app, area),
        LayerBuilderMode::Add | LayerBuilderMode::Edit => draw_lb_add_edit_mode(f, app, area),
    }
}

fn draw_lb_browse_mode(f: &mut Frame, app: &App, area: Rect) {
    let vertical = Layout::vertical([Constraint::Min(0), Constraint::Length(2)]).split(area);

    let arch_area = vertical[0];
    let hint_area = vertical[1];

    draw_lb_architecture(f, app, arch_area);

    let block = Block::default().borders(Borders::TOP);
    let inner = block.inner(hint_area);
    f.render_widget(block, hint_area);
    f.render_widget(
        hint_bar(" [up/down] navigate  [Enter] edit selected  [d] delete selected  [e/Esc] back to add  [q] quit"),
        inner,
    );
}

fn draw_lb_add_edit_mode(f: &mut Frame, app: &App, area: Rect) {
    let vertical = Layout::vertical([
        Constraint::Percentage(30),
        Constraint::Min(0),
        Constraint::Length(2),
    ])
    .split(area);

    let arch_area = vertical[0];
    let form_area = vertical[1];
    let hint_area = vertical[2];

    draw_lb_architecture(f, app, arch_area);
    draw_lb_form(f, app, form_area);

    let block = Block::default().borders(Borders::TOP);
    let inner = block.inner(hint_area);
    f.render_widget(block, hint_area);

    let hint = if app.layer_builder.mode == LayerBuilderMode::Edit {
        " [left/right] type  [up/down] field  [Space] toggle  [type] value  [Enter] save  [Esc] cancel  [q] quit"
    } else {
        " [left/right] type  [up/down] field  [Space] toggle  [type] value  [Enter] add  [d] del last  [e] edit layers  [b] done  [q] quit"
    };
    f.render_widget(hint_bar(hint), inner);
}

fn draw_lb_architecture(f: &mut Frame, app: &App, area: Rect) {
    let lb = &app.layer_builder;
    let title = format!(
        " Architecture — input {}x{}x{} ({} layers) ",
        lb.model_input.0,
        lb.model_input.1,
        lb.model_input.2,
        lb.layers.len()
    );
    let block = Block::default().borders(Borders::ALL).title(title);

    let in_browse_or_edit =
        lb.mode == LayerBuilderMode::Browse || lb.mode == LayerBuilderMode::Edit;
    let items: Vec<ListItem> = lb
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let selected = in_browse_or_edit && i == lb.browse_selected;
            let prefix = if selected { "▶ " } else { "  " };
            let style = if selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Span::styled(
                format!("{}{}: {}", prefix, i, l.display()),
                style,
            ))
        })
        .collect();

    let list = List::new(items).block(block);
    f.render_widget(list, area);
}

fn draw_lb_form(f: &mut Frame, app: &App, area: Rect) {
    let lb = &app.layer_builder;
    let (inferred, preview) = if lb.mode == LayerBuilderMode::Edit {
        let idx = lb.browse_selected;
        let inf = app.inferred_input_for(idx);
        let prev = app.preview_output();
        (inf, prev)
    } else {
        (app.inferred_input(), app.preview_output())
    };

    let inferred_str = format!("{}x{}x{}", inferred.0, inferred.1, inferred.2);
    let preview_str = match preview {
        Some((w, h, c)) => format!("{}x{}x{}", w, h, c),
        None => "?".to_string(),
    };

    let title = if lb.mode == LayerBuilderMode::Edit {
        format!(" Edit Layer {} ", lb.browse_selected)
    } else {
        format!(" Add Layer {} ", lb.layers.len())
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 3 {
        return;
    }

    // Row 0: type selector
    let type_area = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: 1,
    };
    let kinds = [
        LayerKind::Convolution,
        LayerKind::GroupNorm,
        LayerKind::Activation,
        LayerKind::FullyConnected,
        LayerKind::UpsampleConv,
        LayerKind::Concat,
    ];
    let mut kind_spans: Vec<Span> = kinds
        .iter()
        .flat_map(|k| {
            let style = if *k == lb.current_kind {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            vec![Span::styled(format!("[{}]", k), style), Span::raw("  ")]
        })
        .collect();
    kind_spans.push(Span::styled(
        format!("  Input: {} -> {}", inferred_str, preview_str),
        Style::default().fg(Color::Cyan),
    ));
    f.render_widget(Paragraph::new(Line::from(kind_spans)), type_area);

    // Row 1: separator
    if inner.height < 3 {
        return;
    }
    let sep_area = Rect {
        x: inner.x,
        y: inner.y + 1,
        width: inner.width,
        height: 1,
    };
    f.render_widget(
        Paragraph::new("\u{2500}".repeat(inner.width as usize))
            .style(Style::default().fg(Color::DarkGray)),
        sep_area,
    );

    // Rows 2+: fields
    let fields_area = Rect {
        x: inner.x,
        y: inner.y + 2,
        width: inner.width,
        height: inner.height.saturating_sub(2),
    };

    let names = app.layer_field_names();
    let mut lines: Vec<Line> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let value = lb.fields.get(i).map(String::as_str).unwrap_or("");
            let focused = i == lb.field_idx;
            let cursor = if focused { "\u{2588}" } else { "" };
            Line::from(vec![
                Span::styled(format!("  {:>14} : ", name), focused_label(focused)),
                Span::styled(format!("{}{}", value, cursor), focused_value(focused)),
            ])
        })
        .collect();

    if let Some(ref err) = lb.error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  \u{2717} {}", err),
            Style::default().fg(Color::Red),
        )));
    }

    f.render_widget(Paragraph::new(lines), fields_area);
}

// ---------------------------------------------------------------------------
// Screen: Mode Selector
// ---------------------------------------------------------------------------

fn draw_mode_selector(f: &mut Frame, app: &App) {
    draw_choice_screen(
        f,
        "Run Mode",
        &["Inference", "Training"],
        app.mode_selector.selected,
        "[arrow] select  [Enter] confirm  [e] edit layers  [q] quit",
    );
}

// ---------------------------------------------------------------------------
// Screen: Training Params
// ---------------------------------------------------------------------------

fn draw_training_params(f: &mut Frame, app: &App) {
    let training_fields = &app.training_params.fields[..3];
    draw_form_screen(
        f,
        "Training Parameters",
        &TRAINING_PARAM_FIELD_NAMES,
        training_fields,
        app.training_params.field_idx,
        app.training_params.error.as_deref(),
        "[arrow] field  [type] edit  [Enter] next/confirm  [Backspace] del  [Esc] quit",
    );
}

// ---------------------------------------------------------------------------
// Screen: Dataset Selector
// ---------------------------------------------------------------------------

fn draw_dataset_selector(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup = centered_rect(66, 70, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Training Dataset ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines: Vec<Line> = vec![Line::from("")];
    let lr = app
        .training_params
        .fields
        .first()
        .map(String::as_str)
        .unwrap_or("?");
    let batch = app
        .training_params
        .fields
        .get(1)
        .map(String::as_str)
        .unwrap_or("?");
    let steps = app
        .training_params
        .fields
        .get(2)
        .map(String::as_str)
        .unwrap_or("?");

    lines.push(Line::from(Span::styled(
        format!("  Configured params: lr={lr}, batch={batch}, steps={steps}"),
        Style::default().fg(Color::DarkGray),
    )));

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Available datasets:",
        Style::default().fg(Color::Cyan),
    )));
    if app.training_params.datasets.is_empty() {
        lines.push(Line::from(Span::styled(
            "    (none found in datasets/)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (index, dataset) in app.training_params.datasets.iter().enumerate() {
            let style = if index == app.training_params.selected_dataset {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Gray)
            };
            let marker = if index == app.training_params.selected_dataset {
                ">"
            } else {
                " "
            };
            lines.push(Line::from(Span::styled(
                format!("    {} {}", marker, dataset),
                style,
            )));
        }
    }

    if let Some(err) = app.training_params.error.as_deref() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ {err}"),
            Style::default().fg(Color::Red),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  [arrow] select dataset  [<- / ->] cycle  [Enter] start training  [Esc] quit",
        Style::default().fg(Color::DarkGray),
    )));

    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Screen: Monitor
// ---------------------------------------------------------------------------

fn draw_monitor(f: &mut Frame, app: &App) {
    let area = f.area();

    let vertical = Layout::vertical([Constraint::Min(0), Constraint::Length(2)]).split(area);
    let main_area = vertical[0];
    let hint_area = vertical[1];

    let horizontal = Layout::horizontal([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(main_area);

    let right = Layout::vertical([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(horizontal[1]);

    draw_monitor_architecture(f, app, horizontal[0]);
    draw_sparkline(f, app, right[0]);
    draw_analytics(f, app, right[1]);

    let block = Block::default().borders(Borders::TOP);
    let inner = block.inner(hint_area);
    f.render_widget(block, hint_area);
    let hint_text = if let Some(save_status) = &app.monitor.save_status {
        format!(" ✓ {save_status} | [s] save  [q] quit")
    } else if let Some(error) = &app.monitor.error {
        format!(" error: {error} | [s] save  [q] quit")
    } else if app.monitor.done {
        " [r] new training  [s] save model  [q] quit".to_string()
    } else if let Some(sample_path) = &app.monitor.last_sample_path {
        format!(" sample: {sample_path} | [s] save  [q] quit")
    } else {
        " [s] save model  [q] quit".to_string()
    };
    f.render_widget(hint_bar(&hint_text), inner);
}

fn draw_monitor_architecture(f: &mut Frame, app: &App, area: Rect) {
    let lb = &app.layer_builder;
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Architecture ");
    let lines: Vec<Line> = lb
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| Line::from(format!("  {}: {}", i, l.display())))
        .collect();
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_sparkline(f: &mut Frame, app: &App, area: Rect) {
    let current_loss = app.monitor.loss_history.last().copied();
    let loss_str = current_loss.map_or_else(|| "—".to_string(), |l| format!("{:.6}", l));

    let title = if app.monitor.done {
        if app.monitor.error.is_some() {
            format!(
                " Loss: {}  \u{2717} Stopped at step {} ",
                loss_str,
                app.monitor.step + 1
            )
        } else {
            format!(
                " Loss: {}  \u{2713} Done ({} steps) ",
                loss_str,
                app.monitor.step + 1
            )
        }
    } else if app.monitor.total_steps > 0 {
        format!(
            " Loss: {}  step {}/{} ",
            loss_str,
            app.monitor.step + 1,
            app.monitor.total_steps
        )
    } else {
        format!(" Loss: {}  step {} ", loss_str, app.monitor.step)
    };

    let block = Block::default().borders(Borders::ALL).title(title);
    // Reserve space for the borders (2 columns) when computing the visible window.
    let max_points = (area.width.saturating_sub(2)) as usize;

    let data: Vec<u64> = if app.monitor.loss_history.is_empty() {
        vec![0]
    } else {
        let history = &app.monitor.loss_history;
        // Show only the most-recent `max_points` values so the chart scrolls
        // to keep the latest iterations visible once the width is exceeded.
        let start = history.len().saturating_sub(max_points);
        let window = &history[start..];

        let max = window
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-10);
        window.iter().map(|v| ((v / max) * 100.0) as u64).collect()
    };

    let sparkline = Sparkline::default()
        .block(block)
        .data(&data)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, area);
}

fn draw_analytics(f: &mut Frame, app: &App, area: Rect) {
    fn format_bytes(value: Option<u64>) -> String {
        let Some(bytes) = value else {
            return "—".to_string();
        };
        const KIB: f64 = 1024.0;
        const MIB: f64 = KIB * 1024.0;
        const GIB: f64 = MIB * 1024.0;
        let bytes_f = bytes as f64;
        if bytes_f >= GIB {
            format!("{:.2} GiB", bytes_f / GIB)
        } else if bytes_f >= MIB {
            format!("{:.2} MiB", bytes_f / MIB)
        } else if bytes_f >= KIB {
            format!("{:.2} KiB", bytes_f / KIB)
        } else {
            format!("{bytes} B")
        }
    }

    let block = Block::default().borders(Borders::ALL).title(" Analytics ");

    let history = &app.monitor.loss_history;
    let current_loss = history.last().copied();
    let best_loss = history.iter().cloned().reduce(f64::min);
    let worst_loss = history.iter().cloned().reduce(f64::max);

    // Trend: compare average of the last 10 % of samples to the first 10 %.
    let trend = if history.len() >= 10 {
        let window = (history.len() / 10).max(1);
        let recent: f64 = history[history.len() - window..].iter().sum::<f64>() / window as f64;
        let early: f64 = history[..window].iter().sum::<f64>() / window as f64;
        if recent < early * 0.99 {
            "\u{2193} Improving"
        } else if recent > early * 1.01 {
            "\u{2191} Worsening"
        } else {
            "\u{2192} Stable"
        }
    } else {
        "—"
    };

    // Extract hyper-parameters from the stored config when available.
    let (lr_str, batch_str, loss_fn_str) = if let Some(config) = &app.monitor.model_config {
        if let RunMode::Train(ref tc) = config.run.mode {
            (
                format!("{}", tc.lr),
                format!("{}", tc.batch_size),
                tc.loss.to_string(),
            )
        } else {
            ("—".into(), "—".into(), "—".into())
        }
    } else {
        ("—".into(), "—".into(), "—".into())
    };

    let step_str = if app.monitor.total_steps > 0 {
        format!("{}/{}", app.monitor.step + 1, app.monitor.total_steps)
    } else {
        format!("{}", app.monitor.step + 1)
    };

    let progress_str = if app.monitor.total_steps > 0 {
        let pct = ((app.monitor.step + 1) * 100)
            .checked_div(app.monitor.total_steps)
            .unwrap_or(0)
            .min(100);
        // Progress bar width in character cells.
        const BAR_WIDTH: usize = 10;
        let filled = pct * BAR_WIDTH / 100;
        let bar: String = (0..BAR_WIDTH)
            .map(|i| if i < filled { '\u{2588}' } else { '\u{2591}' })
            .collect();
        format!("{bar} {pct}%")
    } else {
        "—".to_string()
    };

    let format_loss = |v: Option<f64>| v.map_or_else(|| "—".to_string(), |x| format!("{:.6}", x));

    let rows: &[(&str, String)] = &[
        ("Current Loss", format_loss(current_loss)),
        ("Best Loss   ", format_loss(best_loss)),
        ("Worst Loss  ", format_loss(worst_loss)),
        ("Trend       ", trend.to_string()),
        ("Step        ", step_str),
        ("Progress    ", progress_str),
        ("Learning Rt ", lr_str),
        ("Batch Size  ", batch_str),
        ("Loss Fn     ", loss_fn_str),
        ("Max Buffer  ", format_bytes(app.monitor.max_buffer_bytes)),
        (
            "Max Storage ",
            format_bytes(app.monitor.max_storage_binding_bytes),
        ),
        (
            "Est. Usage  ",
            format_bytes(app.monitor.estimated_training_bytes),
        ),
    ];

    let label_style = Style::default().fg(Color::DarkGray);
    let value_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);

    let lines: Vec<Line> = rows
        .iter()
        .map(|(label, value)| {
            Line::from(vec![
                Span::styled(format!("  {} : ", label), label_style),
                Span::styled(value.clone(), value_style),
            ])
        })
        .collect();

    f.render_widget(Paragraph::new(lines).block(block), area);
}

// ---------------------------------------------------------------------------
// Screen: Save Model (popup over Monitor)
// ---------------------------------------------------------------------------

fn draw_save_model(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup = centered_rect(54, 30, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Save Model Config ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let cursor = "\u{2588}";
    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Name : ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{}{}", app.save_model.name, cursor),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
    ];

    if let Some(err) = &app.save_model.error {
        lines.push(Line::from(Span::styled(
            format!("  \u{2717} {err}"),
            Style::default().fg(Color::Red),
        )));
        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        "  [type] edit name  [Enter] save  [Esc] cancel",
        Style::default().fg(Color::DarkGray),
    )));

    f.render_widget(Paragraph::new(lines), inner);
}
