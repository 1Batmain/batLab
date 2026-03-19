//! File purpose: Implements ui behavior for the terminal user interface flow.

use super::app::{
    App, INFERENCE_PARAM_FIELD_NAMES, INPUT_SIZE_FIELD_NAMES, LayerBuilderMode, LayerKind,
    MonitorImage, RunMode, Screen, TRAINING_CONTROL_FIELD_NAMES, TRAINING_PARAM_FIELD_NAMES,
};
use ratatui::{prelude::*, widgets::*};

pub fn draw(f: &mut Frame, app: &App) {
    match app.screen {
        Screen::Home => draw_home(f, app),
        Screen::LoadPath => draw_load_path(f, app),
        Screen::TemplateSelector => draw_template_selector(f, app),
        Screen::WeightSelector => draw_weight_selector(f, app),
        Screen::InputSize => draw_input_size(f, app),
        Screen::LayerBuilder => draw_layer_builder(f, app),
        Screen::ModeSelector => draw_mode_selector(f, app),
        Screen::InferenceParams => draw_inference_params(f, app),
        Screen::TrainingParams => draw_training_params(f, app),
        Screen::DatasetSelector => draw_dataset_selector(f, app),
        Screen::Monitor => draw_monitor(f, app),
        Screen::TrainingControl => {
            draw_monitor(f, app);
            draw_training_control(f, app);
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

fn draw_home(f: &mut Frame, _app: &App) {
    draw_choice_screen(
        f,
        "batBuilder",
        &["Select Model Template"],
        0,
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
            "  No model configs found in Models/",
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

fn draw_weight_selector(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup = centered_rect(70, 66, area);
    f.render_widget(Clear, popup);

    let model_name = app.active_model_name.as_deref().unwrap_or("Selected Model");
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {model_name} — Weights "))
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);
    let mut lines = vec![Line::from("")];
    let random_selected = app.weight_selector.selected == 0;
    lines.push(Line::from(Span::styled(
        format!(
            "  {} Start from random weights (new run)",
            if random_selected { ">" } else { " " }
        ),
        if random_selected {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        },
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Load existing pretrained weights:",
        Style::default().fg(Color::Cyan),
    )));

    if app.weight_selector.checkpoints.is_empty() {
        lines.push(Line::from(Span::styled(
            "    (none found in Models/<model>/pretrained_weights/)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (idx, checkpoint) in app.weight_selector.checkpoints.iter().enumerate() {
            let selected = app.weight_selector.selected == idx + 1;
            lines.push(Line::from(Span::styled(
                format!(
                    "    {} {}",
                    if selected { ">" } else { " " },
                    checkpoint.name
                ),
                if selected {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Gray)
                },
            )));
        }
    }

    if let Some(error) = app.weight_selector.error.as_deref() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ {error}"),
            Style::default().fg(Color::Red),
        )));
    }

    lines.push(Line::from(""));
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
        "[arrow] select  [Enter] configure/run  [e] edit layers  [q] quit",
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

fn draw_inference_params(f: &mut Frame, app: &App) {
    let seed_mode = if app.inference_params.random_seed {
        "Random"
    } else {
        "Manual"
    };
    let values = vec![
        seed_mode.to_string(),
        app.inference_params.fields[0].clone(),
        app.inference_params.fields[1].clone(),
        app.inference_params.fields[2].clone(),
    ];
    draw_form_screen(
        f,
        "Inference Parameters",
        &INFERENCE_PARAM_FIELD_NAMES,
        &values,
        app.inference_params.field_idx,
        app.inference_params.error.as_deref(),
        "[up/down] field  [left/right/space] toggle random seed  [type] edit  [Enter] next/run",
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

    draw_monitor_architecture(f, app, horizontal[0]);
    if is_inference_mode(app) {
        let right = Layout::vertical([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(horizontal[1]);
        draw_inference_image(f, app, right[0]);
        draw_analytics(f, app, right[1]);
    } else {
        let right = Layout::vertical([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(horizontal[1]);
        draw_sparkline(f, app, right[0]);
        draw_analytics(f, app, right[1]);
    }

    let block = Block::default().borders(Borders::TOP);
    let inner = block.inner(hint_area);
    f.render_widget(block, hint_area);
    let hint_text = if let Some(save_status) = &app.monitor.save_status {
        format!(" ✓ {save_status} | [s] save  [q] quit")
    } else if let Some(error) = &app.monitor.error {
        format!(" error: {error} | [s] save  [q] quit")
    } else if app.monitor.done {
        " [r] new run  [s] save config  [q] quit".to_string()
    } else if is_inference_mode(app) {
        if let Some(progress) = app.monitor.loading_progress.as_ref() {
            format!(
                " inference: {} ({}/{}) | [s] save  [q] quit",
                progress.label, progress.current, progress.total
            )
        } else {
            " inference: running | [s] save  [q] quit".to_string()
        }
    } else if is_training_mode(app) {
        let status = if app.monitor.is_training_paused {
            "paused"
        } else {
            "running"
        };
        format!(
            " training: {status} | [p] pause/resume  [t] tune params  [v] visualise  [s] save snapshot  [q] quit"
        )
    } else if let Some(checkpoint_path) = &app.monitor.inference_checkpoint_path {
        format!(" checkpoint: {checkpoint_path} | [s] save  [q] quit")
    } else if let Some(sample_path) = &app.monitor.last_sample_path {
        format!(" sample: {sample_path} | [s] save  [q] quit")
    } else {
        " [s] save config  [q] quit".to_string()
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

fn is_inference_mode(app: &App) -> bool {
    app.monitor
        .model_config
        .as_ref()
        .is_some_and(|config| matches!(config.run.mode, RunMode::Infer))
}

fn is_training_mode(app: &App) -> bool {
    app.monitor
        .model_config
        .as_ref()
        .is_some_and(|config| matches!(config.run.mode, RunMode::Train(_)))
}

fn monitor_image_rgb(image: &MonitorImage, x: u32, y: u32) -> (u8, u8, u8) {
    let idx = ((y * image.width + x) * 3) as usize;
    let r = *image.pixels.get(idx).unwrap_or(&0);
    let g = *image.pixels.get(idx + 1).unwrap_or(&0);
    let b = *image.pixels.get(idx + 2).unwrap_or(&0);
    (r, g, b)
}

fn draw_inference_image(f: &mut Frame, app: &App, area: Rect) {
    let title = app
        .monitor
        .inference_image
        .as_ref()
        .map(|image| {
            format!(
                " Inference Preview ({}x{}x{}) ",
                image.width, image.height, image.channels
            )
        })
        .unwrap_or_else(|| " Inference Preview ".to_string());
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(image) = app.monitor.inference_image.as_ref() else {
        if let Some(progress) = app.monitor.loading_progress.as_ref() {
            let sections = Layout::vertical([
                Constraint::Length(2),
                Constraint::Length(3),
                Constraint::Min(0),
            ])
            .split(inner);
            let status = Paragraph::new(format!(
                "  {} ({}/{})",
                progress.label, progress.current, progress.total
            ));
            f.render_widget(status, sections[0]);
            let ratio = if progress.total == 0 {
                0.0
            } else {
                (progress.current as f64 / progress.total as f64).clamp(0.0, 1.0)
            };
            let gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(" Loading "))
                .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
                .ratio(ratio)
                .label(format!(
                    "{:.0}%",
                    if progress.total == 0 {
                        0.0
                    } else {
                        (progress.current as f64 / progress.total as f64) * 100.0
                    }
                ));
            f.render_widget(gauge, sections[1]);
        } else {
            let placeholder = Paragraph::new("  Waiting for inference output...");
            f.render_widget(placeholder, inner);
        }
        return;
    };
    if inner.width == 0 || inner.height == 0 || image.width == 0 || image.height == 0 {
        return;
    }

    let max_w = inner.width as u32;
    let max_h_px = (inner.height as u32).saturating_mul(2);
    if max_w == 0 || max_h_px == 0 {
        return;
    }

    let scale_w = max_w as f32 / image.width as f32;
    let scale_h = max_h_px as f32 / image.height as f32;
    let scale = scale_w.min(scale_h).min(1.0);
    let dst_w = ((image.width as f32 * scale).floor() as u32).max(1);
    let dst_h_px = ((image.height as f32 * scale).floor() as u32).max(1);
    let dst_h_cells = (dst_h_px + 1) / 2;

    let mut lines = Vec::with_capacity(dst_h_cells as usize);
    for y_cell in 0..dst_h_cells {
        let y_top_px = y_cell * 2;
        let y_bottom_px = (y_top_px + 1).min(dst_h_px - 1);
        let src_y_top = (y_top_px * image.height / dst_h_px).min(image.height - 1);
        let src_y_bottom = (y_bottom_px * image.height / dst_h_px).min(image.height - 1);

        let mut spans = Vec::with_capacity(dst_w as usize);
        for x in 0..dst_w {
            let src_x = (x * image.width / dst_w).min(image.width - 1);
            let (tr, tg, tb) = monitor_image_rgb(image, src_x, src_y_top);
            let (br, bg, bb) = monitor_image_rgb(image, src_x, src_y_bottom);
            spans.push(Span::styled(
                "▀",
                Style::default()
                    .fg(Color::Rgb(tr, tg, tb))
                    .bg(Color::Rgb(br, bg, bb)),
            ));
        }
        lines.push(Line::from(spans));
    }

    let paragraph = Paragraph::new(lines).alignment(Alignment::Center);
    f.render_widget(paragraph, inner);
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

    // Extract hyper-parameters from live monitor state with config fallback.
    let lr_str = app
        .monitor
        .current_lr
        .map(|value| value.to_string())
        .or_else(|| {
            app.monitor.model_config.as_ref().and_then(|config| {
                if let RunMode::Train(ref tc) = config.run.mode {
                    Some(tc.lr.to_string())
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| "—".to_string());
    let batch_str = app
        .monitor
        .current_batch_size
        .map(|value| value.to_string())
        .or_else(|| {
            app.monitor.model_config.as_ref().and_then(|config| {
                if let RunMode::Train(ref tc) = config.run.mode {
                    Some(tc.batch_size.to_string())
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| "—".to_string());
    let loss_fn_str = app
        .monitor
        .model_config
        .as_ref()
        .and_then(|config| {
            if let RunMode::Train(ref tc) = config.run.mode {
                Some(tc.loss.to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "—".to_string());

    let (step_str, progress_str) = if let Some(progress) = app.monitor.loading_progress.as_ref() {
        let total = progress.total.max(1);
        let current = progress.current.min(total);
        let pct = (current * 100 / total).min(100);
        const BAR_WIDTH: usize = 10;
        let filled = pct * BAR_WIDTH / 100;
        let bar: String = (0..BAR_WIDTH)
            .map(|i| if i < filled { '\u{2588}' } else { '\u{2591}' })
            .collect();
        (format!("{current}/{total}"), format!("{bar} {pct}%"))
    } else if app.monitor.total_steps > 0 {
        (
            format!("{}/{}", app.monitor.step + 1, app.monitor.total_steps),
            {
                let pct = ((app.monitor.step + 1) * 100)
                    .checked_div(app.monitor.total_steps)
                    .unwrap_or(0)
                    .min(100);
                const BAR_WIDTH: usize = 10;
                let filled = pct * BAR_WIDTH / 100;
                let bar: String = (0..BAR_WIDTH)
                    .map(|i| if i < filled { '\u{2588}' } else { '\u{2591}' })
                    .collect();
                format!("{bar} {pct}%")
            },
        )
    } else {
        (format!("{}", app.monitor.step + 1), "—".to_string())
    };

    let format_loss = |v: Option<f64>| v.map_or_else(|| "—".to_string(), |x| format!("{:.6}", x));
    let mode_str = if is_inference_mode(app) {
        "Inference".to_string()
    } else {
        "Training".to_string()
    };
    let preview_dims = app
        .monitor
        .inference_image
        .as_ref()
        .map(|image| format!("{}x{}x{}", image.width, image.height, image.channels))
        .unwrap_or_else(|| "—".to_string());
    let seed_str = app
        .monitor
        .inference_seed
        .map(|seed| seed.to_string())
        .unwrap_or_else(|| "—".to_string());
    let status_str = if app.monitor.done {
        "Done".to_string()
    } else if is_inference_mode(app) && app.monitor.loading_progress.is_some() {
        "Loading".to_string()
    } else if is_training_mode(app) {
        if app.monitor.is_training_paused {
            "Paused".to_string()
        } else {
            "Running".to_string()
        }
    } else {
        "Running".to_string()
    };

    let rows: &[(&str, String)] = &[
        ("Mode        ", mode_str),
        ("Status      ", status_str),
        ("Current Loss", format_loss(current_loss)),
        ("Best Loss   ", format_loss(best_loss)),
        ("Worst Loss  ", format_loss(worst_loss)),
        ("Trend       ", trend.to_string()),
        ("Step        ", step_str),
        ("Progress    ", progress_str),
        ("Learning Rt ", lr_str),
        ("Batch Size  ", batch_str),
        ("Loss Fn     ", loss_fn_str),
        ("Preview     ", preview_dims),
        ("Seed        ", seed_str),
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
// Screen: Training Control (popup over Monitor)
// ---------------------------------------------------------------------------

fn draw_training_control(f: &mut Frame, app: &App) {
    draw_form_screen(
        f,
        "Training Controls",
        &TRAINING_CONTROL_FIELD_NAMES,
        &app.training_control.fields,
        app.training_control.field_idx,
        app.training_control.error.as_deref(),
        "[up/down] field  [type] edit  [Enter] apply  [Esc] cancel",
    );
}
