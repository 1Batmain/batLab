use crate::app::{
    App, LayerKind, Screen, INPUT_SIZE_FIELD_NAMES, TRAINING_PARAM_FIELD_NAMES,
};
use ratatui::{prelude::*, widgets::*};

pub fn draw(f: &mut Frame, app: &App) {
    match app.screen {
        Screen::Home => draw_home(f, app),
        Screen::LoadPath => draw_load_path(f, app),
        Screen::InputSize => draw_input_size(f, app),
        Screen::LayerBuilder => draw_layer_builder(f, app),
        Screen::ModeSelector => draw_mode_selector(f, app),
        Screen::TrainingParams => draw_training_params(f, app),
        Screen::Monitor => draw_monitor(f, app),
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
fn draw_choice_screen(
    f: &mut Frame,
    title: &str,
    choices: &[&str],
    selected: usize,
    hint: &str,
) {
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
                format!("{}{}", fields.get(i).map(String::as_str).unwrap_or(""), cursor),
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
        &["Load Model", "Build Model"],
        app.home.selected,
        "[arrow] select  [Enter] confirm  [q] quit",
    );
}

// ---------------------------------------------------------------------------
// Screen: Load Path
// ---------------------------------------------------------------------------

fn draw_load_path(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup = centered_rect(56, 40, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Load Model ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let cursor = "\u{2588}";
    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Path : ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{}{}", app.load_path.path, cursor),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  [any] type  [Backspace] delete  [Enter] continue  [Esc] quit",
            Style::default().fg(Color::DarkGray),
        )),
    ];
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
    f.render_widget(
        hint_bar(" [<>] type  [up/dn] field  [Space] toggle  [Enter] add  [d] del last  [b] done  [q] quit"),
        inner,
    );
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

    let items: Vec<ListItem> = lb
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| ListItem::new(format!("  {}: {}", i, l.display())))
        .collect();

    let list = List::new(items).block(block);
    f.render_widget(list, area);
}

fn draw_lb_form(f: &mut Frame, app: &App, area: Rect) {
    let lb = &app.layer_builder;
    let inferred = app.inferred_input();
    let preview = app.preview_output();

    let inferred_str = format!("{}x{}x{}", inferred.0, inferred.1, inferred.2);
    let preview_str = match preview {
        Some((w, h, c)) => format!("{}x{}x{}", w, h, c),
        None => "?".to_string(),
    };

    let title = format!(" Add Layer {} ", lb.layers.len());
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
    let kinds = [LayerKind::Convolution, LayerKind::Activation];
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
        format!(
            "  Input: {} -> {}",
            inferred_str, preview_str
        ),
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
        "[arrow] select  [Enter] confirm  [q] quit",
    );
}

// ---------------------------------------------------------------------------
// Screen: Training Params
// ---------------------------------------------------------------------------

fn draw_training_params(f: &mut Frame, app: &App) {
    draw_form_screen(
        f,
        "Training Parameters",
        &TRAINING_PARAM_FIELD_NAMES,
        &app.training_params.fields,
        app.training_params.field_idx,
        app.training_params.error.as_deref(),
        "[arrow] field  [any] type  [Enter] next/start  [Backspace] del  [Esc] quit",
    );
}

// ---------------------------------------------------------------------------
// Screen: Monitor
// ---------------------------------------------------------------------------

fn draw_monitor(f: &mut Frame, app: &App) {
    let area = f.area();

    let vertical = Layout::vertical([Constraint::Min(0), Constraint::Length(2)]).split(area);
    let main_area = vertical[0];
    let hint_area = vertical[1];

    let horizontal = Layout::horizontal([
        Constraint::Percentage(35),
        Constraint::Percentage(65),
    ])
    .split(main_area);

    let right = Layout::vertical([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(horizontal[1]);

    draw_monitor_architecture(f, app, horizontal[0]);
    draw_sparkline(f, app, right[0]);
    draw_stats_table(f, app, right[1]);

    let block = Block::default().borders(Borders::TOP);
    let inner = block.inner(hint_area);
    f.render_widget(block, hint_area);
    f.render_widget(hint_bar(" [q] quit"), inner);
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
    let title = if app.monitor.done {
        format!(" Loss \u{2713} Done ({} steps) ", app.monitor.step + 1)
    } else if app.monitor.total_steps > 0 {
        format!(
            " Loss  step {}/{} ",
            app.monitor.step + 1,
            app.monitor.total_steps
        )
    } else {
        format!(" Loss  step {} ", app.monitor.step)
    };

    let block = Block::default().borders(Borders::ALL).title(title);

    let data: Vec<u64> = if app.monitor.loss_history.is_empty() {
        vec![0]
    } else {
        let max = app
            .monitor
            .loss_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-10);
        app.monitor
            .loss_history
            .iter()
            .map(|v| ((v / max) * 100.0) as u64)
            .collect()
    };

    let sparkline = Sparkline::default()
        .block(block)
        .data(&data)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, area);
}

fn draw_stats_table(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Layer Stats ");

    let header = Row::new(vec!["#", "Type", "Input", "Output"])
        .style(Style::default().add_modifier(Modifier::BOLD))
        .height(1);

    let rows: Vec<Row> = app
        .layer_builder
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            Row::new(vec![
                i.to_string(),
                l.type_name().to_string(),
                l.input_dim_str(),
                l.output_dim_str(),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(3),
        Constraint::Length(10),
        Constraint::Min(10),
        Constraint::Min(10),
    ];

    f.render_widget(
        Table::new(rows, widths)
            .header(header)
            .block(block)
            .column_spacing(1),
        area,
    );
}
