//! File purpose: Implements ui behavior for the terminal user interface flow.

use super::app::{
    App, INFERENCE_PARAM_FIELD_NAMES, INPUT_SIZE_FIELD_NAMES, LayerBuilderMode, LayerKind,
    MODEL_ACTIONS, ModelAction, MonitorImage, NEW_MODEL_ENTRY, PERPETUAL_PARAM_FIELD_NAMES,
    PathStep, RunMode, Screen, TRAINING_CONTROL_FIELD_NAMES, TRAINING_PARAM_FIELD_NAMES,
};
use super::help;
use ratatui::{prelude::*, widgets::*};

pub fn draw(f: &mut Frame, app: &App) {
    let full = f.area();

    // Most screens are a popup over nothing, so the cells they do not touch
    // keep whatever the *previous* screen left there. That is invisible while a
    // screen redraws itself — consecutive frames are identical — and glaring
    // the moment the flow moves from a full-screen one to a popup one: coming
    // back from the monitor, half the architecture panel and the analytics
    // table stayed behind the weight selector, interleaved with it. Clearing
    // the frame first is one line, and ratatui still diffs before writing, so
    // it costs nothing on a still screen.
    f.render_widget(Clear, full);

    // The breadcrumb takes the last row of the terminal, and the screen gets
    // what is left — reserved rather than drawn over, because the monitor uses
    // its full area right down to the bottom border.
    let (body, trail) = match app.screen.path_step() {
        Some(_) if full.height > 1 => {
            let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(full);
            (rows[0], Some(rows[1]))
        }
        _ => (full, None),
    };

    match app.screen {
        Screen::ModelList => draw_model_list(f, app, body),
        Screen::TemplateSelector => draw_template_selector(f, app, body),
        Screen::ModelActions => draw_model_actions(f, app, body),
        Screen::RenameModel => draw_rename_model(f, app, body),
        Screen::DeleteConfirm => draw_delete_confirm(f, app, body),
        Screen::WeightSelector => draw_weight_selector(f, app, body),
        Screen::InputSize => draw_input_size(f, app, body),
        Screen::LayerBuilder => draw_layer_builder(f, app, body),
        Screen::InferenceParams => draw_inference_params(f, app, body),
        Screen::PerpetualParams => draw_perpetual_params(f, app, body),
        Screen::TrainingParams => draw_training_params(f, app, body),
        Screen::DatasetSelector => draw_dataset_selector(f, app, body),
        Screen::Monitor => draw_monitor(f, app, body),
        Screen::TrainingControl => {
            draw_monitor(f, app, body);
            draw_training_control(f, app, body);
        }
    }

    if let Some(trail) = trail {
        draw_breadcrumb(f, app, trail);
    }
}

// ---------------------------------------------------------------------------
// The breadcrumb — the path, at the bottom of the terminal
// ---------------------------------------------------------------------------

/// Below this width the breadcrumb drops its key hint, and below the narrower
/// bound it drops the step names too and keeps only a position — a wrapped
/// breadcrumb is worse than none.
const BREADCRUMB_FULL_WIDTH: u16 = 72;
const BREADCRUMB_MIN_WIDTH: u16 = 30;

/// The flow drawn as the path it is: `Model › Action › Weights › Parameters ›
/// Run`, current step lit, steps already walked kept legible, steps ahead
/// dimmed.
///
/// It answers the question the screens themselves never did — *where am I, and
/// how did I get here* — which is why it is drawn on every step and on none of
/// the detours.
fn draw_breadcrumb(f: &mut Frame, app: &App, area: Rect) {
    let Some(current) = app.screen.path_step() else {
        return;
    };
    if area.width < BREADCRUMB_MIN_WIDTH {
        return;
    }
    let here = current.position();

    if area.width < BREADCRUMB_FULL_WIDTH {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(" {} ({}/{})", current.label(), here + 1, PathStep::ALL.len()),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ))),
            area,
        );
        return;
    }

    let mut spans = vec![Span::raw(" ")];
    for (index, step) in PathStep::ALL.iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled(
                " › ",
                Style::default().fg(Color::DarkGray),
            ));
        }
        let style = if index == here {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else if index < here {
            // Already chosen. Kept readable rather than dimmed: these are the
            // decisions the current screen is standing on.
            Style::default().fg(Color::Gray)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let label = if index == here {
            format!("[{}]", step.label())
        } else {
            step.label().to_string()
        };
        spans.push(Span::styled(label, style));
    }

    // The hint is offered only where the keys actually do something. Advertising
    // `←/→` on a form that hands them to a field is how a footer starts lying.
    if app.screen.walks_the_path_by_arrow() {
        spans.push(Span::styled(
            "   ← back  → forward",
            Style::default().fg(Color::DarkGray),
        ));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ---------------------------------------------------------------------------
// The help panel — what the focused parameter actually does
// ---------------------------------------------------------------------------

/// How wide the panel is when it is shown.
const HELP_PANEL_WIDTH: u16 = 46;
/// Below this total width the panel is dropped entirely. The bound leaves the
/// form the ~60 columns its longest hint line needs: squeezing both is how a
/// help panel turns into two unreadable columns.
const HELP_MIN_TOTAL_WIDTH: u16 = 106;

/// Splits a screen's area into `(body, help)`, or hands the whole thing back
/// when the terminal is too narrow to carry both.
fn split_for_help(area: Rect, has_help: bool) -> (Rect, Option<Rect>) {
    if !has_help || area.width < HELP_MIN_TOTAL_WIDTH {
        return (area, None);
    }
    let columns =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(HELP_PANEL_WIDTH)]).split(area);
    (columns[0], Some(columns[1]))
}

/// The panel's frame, and where its text goes. Shared by the two things that
/// claim that column — a parameter's explanation, and the selected model's
/// architecture — so they cannot drift apart into two panels that merely look
/// alike.
///
/// Returns the rectangle the caller may write into, already inset.
fn open_side_panel(f: &mut Frame, title: &str, area: Rect) -> Rect {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(format!(" {title} "))
        .title_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    f.render_widget(Clear, area);
    f.render_widget(block, area);
    inner.inner(Margin {
        horizontal: 1,
        vertical: 0,
    })
}

fn draw_help_panel(f: &mut Frame, entry: &help::HelpEntry, area: Rect) {
    let inner = open_side_panel(f, entry.title, area);

    let mut lines: Vec<Line> = vec![Line::from("")];
    for paragraph in entry.body {
        lines.push(Line::from(Span::styled(
            *paragraph,
            Style::default().fg(Color::Gray),
        )));
    }
    if let Some(source) = entry.source {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("→ docs/reports/{source}"),
            Style::default().fg(Color::DarkGray),
        )));
    }

    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

// ---------------------------------------------------------------------------
// The architecture panel — what the model under the cursor actually is
// ---------------------------------------------------------------------------

/// A parameter count as a human reads it: three significant figures and a
/// magnitude, because `1216032` says nothing at a glance and `1,2 M` says the
/// whole thing.
fn format_parameters(count: u64) -> String {
    match count {
        0 => "0 paramètre".to_string(),
        n if n < 10_000 => format!("{n} paramètres"),
        n if n < 1_000_000 => format!("{:.0} k paramètres", n as f64 / 1e3),
        n => format!("{:.1} M paramètres", n as f64 / 1e6).replace('.', ","),
    }
}

/// The one-character mark a notable layer carries in the stack.
///
/// A mark on every row marks nothing, so only three layer kinds get one: the
/// ones that change what the network can *do* rather than how big it is.
fn notable_mark(notable: Option<&str>) -> &'static str {
    match notable {
        Some("attention") => "✳",
        Some("upsample") => "↑",
        Some("skip") => "⊕",
        _ => " ",
    }
}

/// Cuts a line to `width` columns, ending it in `…` when something was lost.
///
/// Counted in `chars`, not bytes: the marks above and the model names both go
/// through here, and slicing a multi-byte character in half panics.
fn clip(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let keep = width.saturating_sub(1);
    text.chars().take(keep).chain(['…']).collect()
}

/// The selected model's architecture, laid out for a column `width` wide and
/// `height` rows tall.
///
/// The stack is the part that does not fit: a 28-layer model against a 20-row
/// panel. Rather than scroll something the user cannot scroll, the middle is
/// elided and *says* how many rows it swallowed — a silently truncated stack
/// reads as a shorter network.
fn architecture_lines(
    entry: &crate::storage::SavedModelEntry,
    width: usize,
    height: usize,
) -> Vec<Line<'static>> {
    let summary = &entry.architecture;
    let grey = Style::default().fg(Color::Gray);
    let dim = Style::default().fg(Color::DarkGray);
    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from(Span::styled(
        clip(
            &format!(
                "{}×{}×{} · {} couches · {}",
                summary.input_edge,
                entry.input_size.1,
                entry.input_size.2,
                summary.layer_count,
                format_parameters(summary.parameters),
            ),
            width,
        ),
        grey,
    )));

    // The receptive field, with the verdict attached. The number alone is a
    // number; "does not cover the image" is the reason the number is here —
    // an output pixel that cannot see the whole frame cannot choose a global
    // content, and the model drifts to the dataset mean.
    let field = match (summary.receptive_field, summary.covers_the_image()) {
        // Short enough that the *verdict* survives the 42-column gutter: the
        // number clipped to "13/32 px — NE COUVRE …" would lose exactly the
        // half that matters.
        (Some(field), Some(covers)) => format!(
            "champ réceptif {:.0}/{} px — {}",
            field,
            summary.input_edge,
            if covers { "couvre" } else { "NE COUVRE PAS" }
        ),
        _ => "champ réceptif : aucune convolution".to_string(),
    };
    lines.push(Line::from(Span::styled(
        clip(&field, width),
        if summary.covers_the_image() == Some(false) {
            Style::default().fg(Color::Yellow)
        } else {
            grey
        },
    )));

    let mut notable = Vec::new();
    if summary.attention_layers > 0 {
        notable.push(format!("✳ {} attention", summary.attention_layers));
    }
    if summary.upsample_layers > 0 {
        notable.push(format!("↑ {} upsample", summary.upsample_layers));
    }
    if summary.concat_layers > 0 {
        notable.push(format!("⊕ {} skip", summary.concat_layers));
    }
    if !notable.is_empty() {
        lines.push(Line::from(Span::styled(
            clip(&notable.join("  "), width),
            grey,
        )));
    }
    lines.push(Line::from(""));

    // Whatever rows are left after the header go to the stack.
    let budget = height.saturating_sub(lines.len());
    let row_line = |row: &batlab_core::ArchitectureRow| {
        Line::from(Span::styled(
            clip(
                &format!(
                    "{:>3}{} {}",
                    row.index,
                    notable_mark(row.notable),
                    row.display
                ),
                width,
            ),
            if row.notable.is_some() { grey } else { dim },
        ))
    };
    let rows = &summary.rows;
    if rows.len() <= budget {
        lines.extend(rows.iter().map(row_line));
    } else if budget >= 3 {
        // Head and tail, because a diffusion stack is read from both ends: the
        // first convolution says what it consumes, the last what it emits.
        let head = budget.div_ceil(2) - 1;
        let tail = budget - head - 1;
        lines.extend(rows[..head].iter().map(row_line));
        lines.push(Line::from(Span::styled(
            clip(&format!("    … {} couches …", rows.len() - head - tail), width),
            dim,
        )));
        lines.extend(rows[rows.len() - tail..].iter().map(row_line));
    }
    lines
}

fn draw_architecture_panel(f: &mut Frame, entry: &crate::storage::SavedModelEntry, area: Rect) {
    let inner = open_side_panel(f, &entry.name, area);
    let lines = architecture_lines(entry, inner.width as usize, inner.height as usize);
    f.render_widget(Paragraph::new(lines), inner);
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

/// Generic form screen (text fields), with the focused field's explanation
/// beside it when the terminal is wide enough to carry one.
fn draw_form_screen(
    f: &mut Frame,
    area: Rect,
    screen: Screen,
    title: &str,
    field_names: &[&str],
    fields: &[String],
    field_idx: usize,
    error: Option<&str>,
    hint: &str,
) {
    let entry = help::help_for(screen, field_idx);
    let (area, help_area) = split_for_help(area, entry.is_some());
    if let (Some(entry), Some(help_area)) = (entry, help_area) {
        draw_help_panel(f, entry, help_area);
    }

    let popup = centered_rect(56, 70, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {title} "))
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    // The label column is measured, not assumed. It was pinned at 16, and the
    // perpetual form's "Renoise Depth (t_r)" is 19 — so that form's colons
    // walked out of line long before the training form grew a longer label.
    let label_width = field_names.iter().map(|name| name.len()).max().unwrap_or(0);

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, name) in field_names.iter().enumerate() {
        let focused = i == field_idx;
        let cursor = if focused { "\u{2588}" } else { "" };
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {name:>label_width$} : "),
                focused_label(focused),
            ),
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
// Screen: Model List — the front door
// ---------------------------------------------------------------------------

fn selected_style(selected: bool) -> Style {
    if selected {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    }
}

/// How a model's checkpoints are summarised in the list. The count first,
/// because "does this model have trained weights at all" is the question; the
/// names after it, because picking between them is the next one.
fn checkpoint_summary(checkpoints: &[String]) -> String {
    match checkpoints.len() {
        0 => "no checkpoints".to_string(),
        1 => format!("1 checkpoint: {}", checkpoints[0]),
        n if n <= 3 => format!("{n} checkpoints: {}", checkpoints.join(", ")),
        n => format!(
            "{n} checkpoints: {}, …",
            checkpoints[..2].join(", ")
        ),
    }
}

fn draw_model_list(f: &mut Frame, app: &App, area: Rect) {
    // The panel is offered exactly when there is a model under the cursor to
    // describe: the "new model" row has no architecture yet, and neither has an
    // empty `Models/`.
    let selected = app.model_list.selected_model();
    let (area, panel_area) = split_for_help(area, selected.is_some());
    if let (Some(entry), Some(panel_area)) = (selected, panel_area) {
        draw_architecture_panel(f, entry, panel_area);
    }

    let popup = centered_rect(72, 66, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" batlab — Models ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines = vec![Line::from("")];
    if app.model_list.models.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No models yet in Models/ — start from a template below.",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
    }
    for (index, model) in app.model_list.models.iter().enumerate() {
        let selected = index == app.model_list.selected;
        lines.push(Line::from(Span::styled(
            format!("  {} {}", if selected { ">" } else { " " }, model.name),
            selected_style(selected),
        )));
        lines.push(Line::from(Span::styled(
            format!(
                "      {}x{}x{} · {} layers · {}",
                model.input_size.0,
                model.input_size.1,
                model.input_size.2,
                model.layer_count,
                checkpoint_summary(&model.checkpoints),
            ),
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines.push(Line::from(""));
    let new_selected = app.model_list.is_new_model_selected();
    lines.push(Line::from(Span::styled(
        format!(
            "  {} {}",
            if new_selected { ">" } else { " " },
            NEW_MODEL_ENTRY
        ),
        selected_style(new_selected),
    )));

    if let Some(status) = app.model_list.status.as_deref() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✓ {status}"),
            Style::default().fg(Color::Green),
        )));
    }
    if let Some(error) = app.model_list.error.as_deref() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ {error}"),
            Style::default().fg(Color::Red),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  [arrow] select  [Enter] open  [r] refresh  [Esc/q] quit",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Screen: Model Actions — what to do with the model that was just picked
// ---------------------------------------------------------------------------

fn draw_model_actions(f: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(56, 62, area);
    f.render_widget(Clear, popup);

    let model_name = app.active_model_name.as_deref().unwrap_or("(no model)");
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {model_name} "))
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!(
                "  {}x{}x{} · {} layers · {}",
                app.layer_builder.model_input.0,
                app.layer_builder.model_input.1,
                app.layer_builder.model_input.2,
                app.layer_builder.layers.len(),
                checkpoint_summary(
                    &app.weight_selector
                        .checkpoints
                        .iter()
                        .map(|entry| entry.name.clone())
                        .collect::<Vec<_>>()
                ),
            ),
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];

    for (index, label) in MODEL_ACTIONS.iter().enumerate() {
        let selected = index == app.model_actions.selected;
        // The destructive half of the menu is coloured as such even when it is
        // not under the cursor.
        let destructive = ModelAction::from_index(index) == Some(ModelAction::Delete);
        let style = if selected {
            selected_style(true)
        } else if destructive {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::Gray)
        };
        if index == 3 {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            format!("  {} {}", if selected { ">" } else { " " }, label),
            style,
        )));
    }

    if let Some(error) = app.model_actions.error.as_deref() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ {error}"),
            Style::default().fg(Color::Red),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  [arrow] select  [Enter] confirm  [e] edit layers  [Esc] back  [q] quit",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Screens: the manager — rename and delete
// ---------------------------------------------------------------------------

fn draw_rename_model(f: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(60, 40, area);
    f.render_widget(Clear, popup);

    let current = app.active_model_name.as_deref().unwrap_or("(no model)");
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Rename Model ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  Current name : {current}"),
            Style::default().fg(Color::Gray),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  New name     : ", focused_label(true)),
            Span::styled(
                format!("{}\u{2588}", app.rename_model.input),
                focused_value(true),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Renames Models/<name>/ and rewrites the config_file to match.",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    if let Some(error) = app.rename_model.error.as_deref() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ {error}"),
            Style::default().fg(Color::Red),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  [type] edit  [Backspace] del  [Enter] rename  [Esc] cancel",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_delete_confirm(f: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(64, 46, area);
    f.render_widget(Clear, popup);

    let current = app.active_model_name.as_deref().unwrap_or("(no model)");
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red))
        .title(" Delete Model ")
        .title_alignment(Alignment::Center);
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let matches = app.delete_confirm.typed == current;
    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  This deletes Models/{current}/ and everything in it —"),
            Style::default().fg(Color::Red),
        )),
        Line::from(Span::styled(
            "  config, checkpoints, metrics. It cannot be undone.",
            Style::default().fg(Color::Red),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  Type '{current}' to confirm:"),
            Style::default().fg(Color::Gray),
        )),
        Line::from(vec![
            Span::styled("  > ", focused_label(true)),
            Span::styled(
                format!("{}\u{2588}", app.delete_confirm.typed),
                if matches {
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                } else {
                    focused_value(true)
                },
            ),
        ]),
    ];

    if let Some(error) = app.delete_confirm.error.as_deref() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  ✗ {error}"),
            Style::default().fg(Color::Red),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        if matches {
            "  [Enter] DELETE — no further prompt  [Esc] cancel"
        } else {
            "  [type] the name  [Backspace] del  [Esc] cancel"
        },
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_template_selector(f: &mut Frame, app: &App, area: Rect) {
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
        "  [arrow] select  [Enter] create  [Esc] back",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_weight_selector(f: &mut Frame, app: &App, area: Rect) {
    // The weight choice is a parameter like any other now that it has a
    // default, so it gets the same panel as the forms.
    let entry = help::help_for(Screen::WeightSelector, app.weight_selector.selected);
    let (area, help_area) = split_for_help(area, entry.is_some());
    if let (Some(entry), Some(help_area)) = (entry, help_area) {
        draw_help_panel(f, entry, help_area);
    }

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
        // The default is the pretrained weights — so when there are none, say
        // which way the flow fell back rather than leaving the cursor sitting
        // on a row the user never chose.
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "    This model has no weights yet: the run starts from random.",
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
        "  [arrow] select  [Enter] continue  [Esc] back  [q] quit",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Screen: Input Size
// ---------------------------------------------------------------------------

fn draw_input_size(f: &mut Frame, app: &App, area: Rect) {
    draw_form_screen(
        f,
        area,
        Screen::InputSize,
        "Model Input Size",
        &INPUT_SIZE_FIELD_NAMES,
        &app.input_size.fields,
        app.input_size.field_idx,
        app.input_size.error.as_deref(),
        "[arrow] field  [0-9] type  [Enter] next/confirm (clears layers)  [Backspace] del  [Esc] back",
    );
}

// ---------------------------------------------------------------------------
// Screen: Layer Builder
// ---------------------------------------------------------------------------

fn draw_layer_builder(f: &mut Frame, app: &App, area: Rect) {

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
        " [left/right] type  [up/down] field  [Space] toggle  [Enter] add  [d] del last  [e] edit layers  [i] input size  [b] done  [Esc] back  [q] quit"
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
// Screen: Training Params
// ---------------------------------------------------------------------------

/// How the random-weights toggle reads.
///
/// Three states, not two: "No" (continuing from a checkpoint), "Yes" (the
/// deliberate opt-out), and "Yes — no checkpoint found", which is the *forced*
/// case. Collapsing the third into the second would show a checkbox the user
/// cannot uncheck, with nothing on screen to say why.
fn random_weights_value(app: &App) -> String {
    if !app.has_pretrained_weights() {
        return "Yes — no checkpoint found".to_string();
    }
    if app.start_from_random_weights() {
        "Yes".to_string()
    } else {
        let name = app
            .weight_selector
            .selected_checkpoint()
            .map(|entry| entry.name.as_str())
            .unwrap_or("pretrained weights");
        format!("No — continue from {name}")
    }
}

fn draw_training_params(f: &mut Frame, app: &App, area: Rect) {
    let mut values: Vec<String> = app.training_params.fields[..3].to_vec();
    values.push(random_weights_value(app));
    draw_form_screen(
        f,
        area,
        Screen::TrainingParams,
        "Training Parameters",
        &TRAINING_PARAM_FIELD_NAMES,
        &values,
        app.training_params.field_idx,
        app.training_params.error.as_deref(),
        "[arrow] field  [type] edit  [space] toggle  [Enter] next/confirm  [Esc] back",
    );
}

fn draw_inference_params(f: &mut Frame, app: &App, area: Rect) {
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
        area,
        Screen::InferenceParams,
        "Inference Parameters",
        &INFERENCE_PARAM_FIELD_NAMES,
        &values,
        app.inference_params.field_idx,
        app.inference_params.error.as_deref(),
        "[up/down] field  [left/right/space] toggle random seed  [type] edit  [Enter] next/run  [Esc] back",
    );
}

fn draw_perpetual_params(f: &mut Frame, app: &App, area: Rect) {
    let seed_mode = if app.perpetual_params.random_seed {
        "Random"
    } else {
        "Manual"
    };
    let values = vec![
        seed_mode.to_string(),
        app.perpetual_params.fields[0].clone(),
        app.perpetual_params.fields[1].clone(),
        app.perpetual_params.fields[2].clone(),
        app.perpetual_params.fields[3].clone(),
        app.perpetual_params.regime.label().to_string(),
    ];
    draw_form_screen(
        f,
        area,
        Screen::PerpetualParams,
        "Perpetual Inference",
        &PERPETUAL_PARAM_FIELD_NAMES,
        &values,
        app.perpetual_params.field_idx,
        app.perpetual_params.error.as_deref(),
        "[up/down] field  [left/right/space] toggle  [type] edit  [Enter] next/run  [Esc] back",
    );
}

// ---------------------------------------------------------------------------
// Screen: Dataset Selector
// ---------------------------------------------------------------------------

fn draw_dataset_selector(f: &mut Frame, app: &App, area: Rect) {
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
        "  [arrow] select dataset  [<- / ->] cycle  [Enter] start training  [Esc] back  [q] quit",
        Style::default().fg(Color::DarkGray),
    )));

    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Screen: Monitor
// ---------------------------------------------------------------------------

fn draw_monitor(f: &mut Frame, app: &App, area: Rect) {

    let vertical = Layout::vertical([Constraint::Min(0), Constraint::Length(2)]).split(area);
    let main_area = vertical[0];
    let hint_area = vertical[1];

    let horizontal = Layout::horizontal([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(main_area);

    draw_monitor_architecture(f, app, horizontal[0]);
    if is_perpetual_mode(app) {
        let right = Layout::vertical([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(horizontal[1]);
        draw_perpetual_panel(f, app, right[0]);
        draw_analytics(f, app, right[1]);
    } else if is_inference_mode(app) {
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
    // A perpetual run has no end state to fall through to, and its keys differ
    // from every other mode's, so it claims the footer first.
    let hint_text = if is_perpetual_mode(app) && !app.monitor.done {
        perpetual_hint(app)
    } else if let Some(save_status) = &app.monitor.save_status {
        format!(" ✓ {save_status} | [s] save  [q] quit")
    } else if let Some(error) = &app.monitor.error {
        format!(" error: {error} | [s] save  [q] quit")
    } else if app.monitor.done {
        " [r] new run  [s] save config  [q] quit".to_string()
    } else if is_inference_mode(app) {
        // `[v]` is only advertised once the denoising frame is registered —
        // offering it during model build would be a key that does nothing.
        let visualise = if super::visualiser_control::has_visualiser_source() {
            "  [v] visualise"
        } else {
            ""
        };
        // Same legend as the perpetual panel: the window shows two panes, and
        // a user with no caption reads them as one image that has split.
        let legend = if super::visualiser_control::has_visualiser_source() {
            "  (fenêtre : gauche x_t, droite x̂₀)"
        } else {
            ""
        };
        if let Some(progress) = app.monitor.loading_progress.as_ref() {
            format!(
                " inference: {} ({}/{}) |{visualise}  [s] save  [q] quit{legend}",
                progress.label, progress.current, progress.total
            )
        } else {
            format!(" inference: running |{visualise}  [s] save  [q] quit{legend}")
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

fn is_perpetual_mode(app: &App) -> bool {
    app.monitor
        .model_config
        .as_ref()
        .is_some_and(|config| matches!(config.run.mode, RunMode::Perpetual(_)))
}

/// The perpetual footer: what the drift is doing, then every key that steers it.
///
/// Built from `monitor.perpetual`, which only the worker writes — so `t_r` and
/// the regime on screen are the ones the sampler is using, not the ones the UI
/// last asked for.
fn perpetual_hint(app: &App) -> String {
    const KEYS: &str = "[↑↓] niveau  [←→] tempo  [espace] pause  [r] re-seed  \
                        [m] regime  [x] vue  [s] PNG  [v] visualise  [q] quit";

    // The last PNG write (or failure) is worth a word, but must not cost the
    // user the key list — it rides as a prefix instead of replacing the line.
    let notice = app
        .monitor
        .error
        .as_deref()
        .map(|error| format!(" ✗ {error} |"))
        .or_else(|| {
            app.monitor
                .save_status
                .as_deref()
                .map(|status| format!(" ✓ {status} |"))
        })
        .unwrap_or_default();

    let Some(state) = app.monitor.perpetual.as_ref() else {
        return format!("{notice} perpetual: starting… | {KEYS}");
    };

    let status = if state.paused { "en pause" } else { "en cours" };
    format!(
        "{notice} perpetual · {} · {} · {status} · {}={} (max {}) · t={} · {} {} · {:.0}/{:.0} pas/s | {KEYS}",
        state.regime,
        state.phase,
        state.depth_label,
        state.depth,
        state.max_depth,
        state.diffusion_step,
        state.cycle_label,
        state.cycle,
        state.steps_per_sec,
        state.tempo,
    )
}

/// The drift's read-out. The image itself lives in the `[v]` window — this pane
/// is the instrument panel beside it.
fn draw_perpetual_panel(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Perpetual Drift ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(state) = app.monitor.perpetual.as_ref() else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  building the model and loading weights…",
                    Style::default().fg(Color::DarkGray),
                )),
            ]),
            inner,
        );
        return;
    };

    let row = |label: &str, value: String, key: &str| {
        Line::from(vec![
            Span::styled(format!("  {label:<22}"), Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{value:<20}"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(key.to_string(), Style::default().fg(Color::DarkGray)),
        ])
    };

    let lines = vec![
        Line::from(""),
        row("regime", state.regime.clone(), "[m]"),
        row("phase", state.phase.clone(), ""),
        row(
            &format!("niveau {}", state.depth_label),
            format!("{} / {}", state.depth, state.max_depth),
            "[↑ / ↓]",
        ),
        row("timestep t", state.diffusion_step.to_string(), ""),
        row(&state.cycle_label, state.cycle.to_string(), "[r] re-seed"),
        row("appels modèle", state.steps.to_string(), ""),
        row(
            "pace",
            format!("{:.1} / {:.0} steps/s", state.steps_per_sec, state.tempo),
            "[← / →]",
        ),
        row(
            "state",
            if state.paused { "paused" } else { "running" }.to_string(),
            "[space]",
        ),
        row("origine", state.origin.clone(), ""),
        row("fenêtre [v]", state.view.clone(), "[x]"),
        Line::from(""),
        Line::from(Span::styled(
            "  La dérive part d'une image réelle du dataset et s'en éloigne ; [r] en tire une autre.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "  En remontée x_t se dissout pas à pas, et le modèle prédit toujours : x̂₀ rêve, jamais figé.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "  En flux, le niveau t* ne bouge plus : un cran débruité, un cran rebruité par frame.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "  [s] writes the current x̂₀ to perpetual_samples/.",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    f.render_widget(Paragraph::new(lines), inner);
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

fn draw_training_control(f: &mut Frame, app: &App, area: Rect) {
    draw_form_screen(
        f,
        area,
        Screen::TrainingControl,
        "Training Controls",
        &TRAINING_CONTROL_FIELD_NAMES,
        &app.training_control.fields,
        app.training_control.field_idx,
        app.training_control.error.as_deref(),
        "[up/down] field  [type] edit  [Enter] apply  [Esc] cancel",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::SavedModelEntry;

    /// A stack of `count` identical convolutions, wrapped as the list entry the
    /// panel is handed.
    fn entry_with(layers: Vec<batlab_core::LayerDraft>, input: (u32, u32, u32)) -> SavedModelEntry {
        SavedModelEntry {
            name: "Test_Model".to_string(),
            path: std::path::PathBuf::from("Models/Test_Model/config_file"),
            input_size: input,
            layer_count: layers.len(),
            checkpoints: Vec::new(),
            architecture: batlab_core::summarize_architecture(&layers, input),
        }
    }

    fn template_entry() -> SavedModelEntry {
        let template = batlab_core::config::greyscale_diffusion_template();
        entry_with(template.layers, template.input_size)
    }

    /// The panel lives in a fixed 46-column gutter, and the layer descriptions
    /// it prints are not bounded by anything: a `Concat(enc1) 32x32x64 +
    /// 32x32x32 -> 32x32x96 [save:d2]` is 55 characters. Ratatui does not wrap
    /// a plain `Paragraph`, so an over-long line is silently cut at the border —
    /// but a line cut by the *renderer* loses its right edge without a mark,
    /// where one cut here ends in `…` and says so.
    #[test]
    fn no_line_of_the_panel_is_wider_than_the_column_it_lives_in() {
        let entry = template_entry();
        for width in [20usize, 30, 42, 44] {
            for line in architecture_lines(&entry, width, 24) {
                let rendered: String = line.spans.iter().map(|span| span.content.as_ref()).collect();
                assert!(
                    rendered.chars().count() <= width,
                    "at width {width}, {rendered:?} is {} columns",
                    rendered.chars().count()
                );
            }
        }
    }

    /// With room for the whole stack, every layer is on screen, in order.
    #[test]
    fn a_panel_with_room_shows_every_layer_in_order() {
        let entry = template_entry();
        let lines = architecture_lines(&entry, 60, 40);
        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        for (index, row) in entry.architecture.rows.iter().enumerate() {
            assert!(
                rendered.iter().any(|line| line.trim_start().starts_with(&format!("{index}"))
                    && line.contains(row.display.split_whitespace().next().expect("a kind"))),
                "layer {index} ({}) is missing from the panel: {rendered:#?}",
                row.display
            );
        }
        assert!(!rendered.iter().any(|line| line.contains("couches …")));
    }

    /// A stack taller than the panel is **elided**, not truncated, and the
    /// elision says how many rows it swallowed. A stack quietly cut at row 12
    /// reads as a twelve-layer network, which is a different model.
    #[test]
    fn a_stack_taller_than_the_panel_says_what_it_hid() {
        let layers: Vec<batlab_core::LayerDraft> = (0..40)
            .map(|_| batlab_core::LayerDraft::Convolution {
                dim_input: (32, 32, 8),
                nb_kernel: 8,
                dim_kernel: (3, 3, 8),
                stride: 1,
                padding: batlab_core::config::PaddingMode::Same,
                save_key: None,
            })
            .collect();
        let entry = entry_with(layers, (32, 32, 8));

        let height = 20;
        let lines = architecture_lines(&entry, 44, height);
        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        assert!(lines.len() <= height, "the panel overflowed its own height");
        let elision = rendered
            .iter()
            .find(|line| line.contains("couches …"))
            .expect("an over-long stack must be marked as elided");
        // Head + tail + the elision line account for the whole stack.
        let shown = rendered.iter().filter(|line| line.contains("Conv")).count();
        let hidden: usize = elision
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .expect("the elision states a count");
        assert_eq!(
            shown + hidden,
            40,
            "the panel accounts for {shown} shown and {hidden} hidden of 40"
        );
        // Both ends survive: the first layer says what the model consumes, the
        // last what it emits.
        assert!(rendered.iter().any(|line| line.trim_start().starts_with('0')));
        assert!(rendered.iter().any(|line| line.trim_start().starts_with("39")));
    }

    /// The verdict, not just the number. 13 px of receptive field on a 32 px
    /// image is the single most consequential fact about the built-in
    /// templates, and it has to be legible without doing the comparison in
    /// one's head.
    #[test]
    fn the_panel_states_whether_the_field_covers_the_image() {
        let rendered: String = architecture_lines(&template_entry(), 60, 40)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("13/32 px"), "{rendered}");
        assert!(rendered.contains("NE COUVRE PAS"), "{rendered}");
        // The verdict must survive the real gutter, not just a wide test bench:
        // clipped to "NE COUVRE …" the panel says the opposite of nothing.
        let narrow: String = architecture_lines(&template_entry(), 42, 40)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(narrow.contains("NE COUVRE PAS"), "{narrow}");
        // …and the notable layers of that stack, counted.
        assert!(rendered.contains("↑ 1 upsample"), "{rendered}");
        assert!(rendered.contains("⊕ 1 skip"), "{rendered}");
    }

    #[test]
    fn parameters_are_written_at_the_magnitude_a_reader_needs() {
        assert_eq!(format_parameters(0), "0 paramètre");
        assert_eq!(format_parameters(432), "432 paramètres");
        assert_eq!(format_parameters(121_000), "121 k paramètres");
        assert_eq!(format_parameters(1_216_032), "1,2 M paramètres");
    }

    /// `clip` counts characters, never bytes — the panel's own marks (`✳ ↑ ⊕`)
    /// and any accented model name are multi-byte, and slicing one in half
    /// panics the whole TUI.
    #[test]
    fn clipping_never_splits_a_character() {
        assert_eq!(clip("abcdef", 6), "abcdef");
        assert_eq!(clip("abcdef", 4), "abc…");
        assert_eq!(clip("✳↑⊕ modèle", 4), "✳↑⊕…");
        assert_eq!(clip("é", 0), "…");
    }
}
