//! File purpose: The navigation invariants of the TUI, as tests.
//!
//! Two claims, and this file is what makes them checkable rather than merely
//! stated:
//!
//! 1. **Every screen is reachable by keys.** Twice now a screen has been
//!    written, drawn, given a key handler — and never assigned, so it did not
//!    exist as far as a user was concerned: `Screen::LoadPath` and the
//!    `Perpetual` run mode (`docs/reports/PERPETUAL_INFERENCE.md` §1). Both were
//!    found by driving the real TUI, not by reading the code, because reading
//!    the code shows you the screen. The walk below presses keys from the front
//!    door and asserts it visited all of [`Screen::ALL`]; a new variant fails it
//!    until something routes to it.
//!
//! 2. **`Esc` always goes up, and never nowhere.** Every screen has exactly one
//!    parent, or is a root where `Esc` leaves the app. Nothing is a dead end.

use super::app::{App, ModelAction, Screen};
use super::events::handle_key;
use crate::storage::TempRoot;
use crossterm::event::KeyCode;
use std::collections::HashSet;

fn test_app(tag: &str) -> (TempRoot, App) {
    let temp = TempRoot::new(tag);
    let app = App::with_storage(temp.storage());
    (temp, app)
}

/// Presses `key` and asserts where it landed, recording the screen as visited.
fn step(app: &mut App, visited: &mut HashSet<Screen>, key: KeyCode, expected: Screen) {
    let from = app.screen;
    handle_key(app, key);
    assert_eq!(
        app.screen, expected,
        "{key:?} from {from:?} should reach {expected:?}"
    );
    assert!(!app.should_quit, "{key:?} from {from:?} quit the app");
    visited.insert(app.screen);
}

/// Walks the builder half of the flow with nothing but key presses, from the
/// front door, and reports every screen it saw.
fn walk_the_builder(app: &mut App) -> HashSet<Screen> {
    let mut visited = HashSet::new();
    visited.insert(app.screen);
    assert_eq!(app.screen, Screen::ModelList, "the app must open on the list");

    // An empty root: row 0 is the "new model" row, which is the template flow.
    step(app, &mut visited, KeyCode::Enter, Screen::TemplateSelector);
    // Picking a template creates the model and rejoins the common path.
    step(app, &mut visited, KeyCode::Enter, Screen::ModelActions);

    // The architecture, and the input geometry behind it.
    step(app, &mut visited, KeyCode::Char('e'), Screen::LayerBuilder);
    handle_key(app, KeyCode::Esc); // browse mode → add mode, same screen
    step(app, &mut visited, KeyCode::Char('i'), Screen::InputSize);
    step(app, &mut visited, KeyCode::Esc, Screen::LayerBuilder);
    step(app, &mut visited, KeyCode::Esc, Screen::ModelActions);

    // Each run action, down to its parameter form.
    select_action(app, ModelAction::Train);
    step(app, &mut visited, KeyCode::Enter, Screen::WeightSelector);
    step(app, &mut visited, KeyCode::Enter, Screen::TrainingParams);
    handle_key(app, KeyCode::Enter); // learning rate → batch size
    handle_key(app, KeyCode::Enter); // batch size → steps
    step(app, &mut visited, KeyCode::Enter, Screen::DatasetSelector);
    step(app, &mut visited, KeyCode::Esc, Screen::TrainingParams);
    step(app, &mut visited, KeyCode::Esc, Screen::WeightSelector);
    step(app, &mut visited, KeyCode::Esc, Screen::ModelActions);

    select_action(app, ModelAction::Infer);
    step(app, &mut visited, KeyCode::Enter, Screen::WeightSelector);
    step(app, &mut visited, KeyCode::Enter, Screen::InferenceParams);
    step(app, &mut visited, KeyCode::Esc, Screen::WeightSelector);
    step(app, &mut visited, KeyCode::Esc, Screen::ModelActions);

    select_action(app, ModelAction::Perpetual);
    step(app, &mut visited, KeyCode::Enter, Screen::WeightSelector);
    step(app, &mut visited, KeyCode::Enter, Screen::PerpetualParams);
    step(app, &mut visited, KeyCode::Esc, Screen::WeightSelector);
    step(app, &mut visited, KeyCode::Esc, Screen::ModelActions);

    // And the manager.
    select_action(app, ModelAction::Rename);
    step(app, &mut visited, KeyCode::Enter, Screen::RenameModel);
    step(app, &mut visited, KeyCode::Esc, Screen::ModelActions);

    select_action(app, ModelAction::Delete);
    step(app, &mut visited, KeyCode::Enter, Screen::DeleteConfirm);
    step(app, &mut visited, KeyCode::Esc, Screen::ModelActions);

    step(app, &mut visited, KeyCode::Esc, Screen::ModelList);
    visited
}

/// Moves the action cursor with arrow keys only — the point is that the cursor
/// can *get* there, not that the field can be assigned.
fn select_action(app: &mut App, action: ModelAction) {
    assert_eq!(app.screen, Screen::ModelActions);
    while app.model_actions.selected > action.index() {
        handle_key(app, KeyCode::Up);
    }
    while app.model_actions.selected < action.index() {
        handle_key(app, KeyCode::Down);
    }
    assert_eq!(
        app.model_actions.selected,
        action.index(),
        "the cursor cannot reach {action:?}"
    );
}

/// The monitor half. It is a second root: the host enters it after a run has
/// started, not by a key press from the builder.
fn walk_the_monitor(app: &mut App) -> HashSet<Screen> {
    let mut visited = HashSet::new();
    app.screen = Screen::Monitor;
    app.monitor.current_lr = Some(0.001);
    app.monitor.current_batch_size = Some(1);
    visited.insert(app.screen);

    step(app, &mut visited, KeyCode::Char('t'), Screen::TrainingControl);
    step(app, &mut visited, KeyCode::Esc, Screen::Monitor);
    visited
}

#[test]
fn every_screen_is_reachable_by_pressing_keys() {
    let (_temp, mut app) = test_app("nav-reachability");

    let mut visited = walk_the_builder(&mut app);
    visited.extend(walk_the_monitor(&mut app));

    let missing: Vec<Screen> = Screen::ALL
        .iter()
        .copied()
        .filter(|screen| !visited.contains(screen))
        .collect();
    assert!(
        missing.is_empty(),
        "these screens are drawn but no sequence of keys reaches them: {missing:?}"
    );
}

/// Where `Esc` goes from each screen. `None` means the screen is a root and
/// `Esc` leaves the application — there are exactly two, and both are honest
/// about it: the model list is the front door, and the monitor is a run in
/// progress that quitting ends.
fn esc_parent(screen: Screen) -> Option<Screen> {
    match screen {
        Screen::ModelList => None,
        Screen::TemplateSelector => Some(Screen::ModelList),
        Screen::ModelActions => Some(Screen::ModelList),
        Screen::RenameModel => Some(Screen::ModelActions),
        Screen::DeleteConfirm => Some(Screen::ModelActions),
        Screen::WeightSelector => Some(Screen::ModelActions),
        Screen::InputSize => Some(Screen::LayerBuilder),
        Screen::LayerBuilder => Some(Screen::ModelActions),
        Screen::InferenceParams => Some(Screen::WeightSelector),
        Screen::PerpetualParams => Some(Screen::WeightSelector),
        Screen::TrainingParams => Some(Screen::WeightSelector),
        Screen::DatasetSelector => Some(Screen::TrainingParams),
        Screen::Monitor => None,
        Screen::TrainingControl => Some(Screen::Monitor),
    }
}

/// `Esc` goes up one step from every screen, and there is no screen it does
/// nothing on. Before the model-first flow, `Esc` quit outright from six of
/// these — a mistyped learning rate meant restarting the whole session.
#[test]
fn esc_walks_up_one_step_from_every_screen_and_is_never_inert() {
    for screen in Screen::ALL {
        let (_temp, mut app) = test_app("nav-esc");
        app.screen = screen;

        handle_key(&mut app, KeyCode::Esc);

        match esc_parent(screen) {
            Some(parent) => {
                assert_eq!(
                    app.screen, parent,
                    "Esc on {screen:?} should go up to {parent:?}"
                );
                assert!(!app.should_quit, "Esc on {screen:?} quit instead of going up");
            }
            None => assert!(
                app.should_quit,
                "{screen:?} is a root, so Esc must leave the app"
            ),
        }
    }
}

/// The parent relation has to actually terminate at a root. A cycle
/// (`A → B → A`) would satisfy "Esc goes somewhere" while trapping the user.
#[test]
fn following_esc_from_anywhere_reaches_a_root() {
    for screen in Screen::ALL {
        let mut current = screen;
        let mut hops = 0;
        while let Some(parent) = esc_parent(current) {
            current = parent;
            hops += 1;
            assert!(
                hops <= Screen::ALL.len(),
                "Esc from {screen:?} never reaches a root — the parent chain loops"
            );
        }
    }
}
