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

use super::app::{App, ModelAction, PathStep, Screen, TrainingControlCommand};
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
    handle_key(app, KeyCode::Enter); // steps → EMA decay
    handle_key(app, KeyCode::Enter); // EMA decay → start-from-random toggle
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

// ---------------------------------------------------------------------------
// The path: the breadcrumb, and `←` / `→`
// ---------------------------------------------------------------------------

/// The breadcrumb only means something if the position it shows moves the way
/// the user does. Every screen on the path has to sit at a step, every step has
/// to be reachable, and the order has to match the order they are walked in.
#[test]
fn every_step_of_the_path_is_reached_in_order() {
    let (_temp, mut app) = test_app("path-steps");
    app.finish_template_selector();

    let walked: Vec<Option<PathStep>> = {
        let mut seen = vec![Screen::ModelActions.path_step()];
        app.model_actions.selected = ModelAction::Train.index();
        handle_key(&mut app, KeyCode::Enter);
        seen.push(app.screen.path_step());
        handle_key(&mut app, KeyCode::Enter);
        seen.push(app.screen.path_step());
        app.screen = Screen::Monitor;
        seen.push(app.screen.path_step());
        seen
    };
    assert_eq!(
        walked,
        vec![
            Some(PathStep::Action),
            Some(PathStep::Weights),
            Some(PathStep::Parameters),
            Some(PathStep::Run),
        ]
    );
    assert_eq!(Screen::ModelList.path_step(), Some(PathStep::Model));

    // And the positions are the order the labels are drawn in.
    let positions: Vec<usize> = PathStep::ALL.iter().map(|step| step.position()).collect();
    assert_eq!(positions, vec![0, 1, 2, 3, 4]);
}

/// Every step of the path must be the step of at least one screen, or the
/// breadcrumb would draw a position nothing can occupy.
#[test]
fn no_step_of_the_path_is_unreachable() {
    for step in PathStep::ALL {
        assert!(
            Screen::ALL
                .iter()
                .any(|screen| screen.path_step() == Some(step)),
            "no screen sits at {step:?}, so the breadcrumb draws a step the user \
             can never be on"
        );
    }
}

/// `←` is `Esc` without the quit. On the front door — the one root of the
/// builder half — it must do nothing at all: an arrow key ending the session is
/// exactly the accident this guards.
#[test]
fn left_goes_back_everywhere_esc_does_and_never_quits() {
    for screen in Screen::ALL.iter().filter(|s| s.walks_the_path_by_arrow()) {
        let (_temp, mut app) = test_app("path-left");
        app.screen = *screen;

        handle_key(&mut app, KeyCode::Left);

        assert!(!app.should_quit, "← quit the app from {screen:?}");
        match esc_parent(*screen) {
            Some(parent) => assert_eq!(app.screen, parent, "← from {screen:?}"),
            None => assert_eq!(
                app.screen, *screen,
                "{screen:?} is a root, so ← must stay put"
            ),
        }
    }
}

/// The promise of `→`: going back and forward costs nothing. Pick an action and
/// a checkpoint, walk all the way up with `←`, walk back down with `→`, and
/// land on the same form with the same choices.
#[test]
fn right_walks_back_down_the_path_without_losing_the_choices() {
    let (_temp, mut app) = test_app("path-right");
    app.finish_template_selector();
    let model = app.active_model_name.clone().expect("template made a model");
    std::fs::write(
        app.storage
            .model_weights_dir(&model)
            .expect("weights dir")
            .join("latest.ckpt"),
        b"weights",
    )
    .expect("checkpoint write");
    app.enter_model_actions();

    // Down: pick Perpetual, then confirm the weights it defaulted to.
    app.model_actions.selected = ModelAction::Perpetual.index();
    handle_key(&mut app, KeyCode::Right);
    assert_eq!(app.screen, Screen::WeightSelector);
    let weights = app.selected_checkpoint_path.clone();
    handle_key(&mut app, KeyCode::Right);
    assert_eq!(app.screen, Screen::PerpetualParams);

    // Up, all the way to the front door.
    handle_key(&mut app, KeyCode::Esc);
    handle_key(&mut app, KeyCode::Left);
    assert_eq!(app.screen, Screen::ModelActions);
    handle_key(&mut app, KeyCode::Left);
    assert_eq!(app.screen, Screen::ModelList);
    handle_key(&mut app, KeyCode::Left);
    assert_eq!(app.screen, Screen::ModelList, "← must not leave the root");

    // And down again, by arrow alone.
    handle_key(&mut app, KeyCode::Right);
    assert_eq!(app.screen, Screen::ModelActions);
    assert_eq!(
        app.model_actions.selected,
        ModelAction::Perpetual.index(),
        "coming back re-read the config and lost the chosen action"
    );
    handle_key(&mut app, KeyCode::Right);
    assert_eq!(app.screen, Screen::WeightSelector);
    assert_eq!(app.selected_checkpoint_path, weights);
    assert!(app.load_checkpoint_on_start);
    handle_key(&mut app, KeyCode::Right);
    assert_eq!(app.screen, Screen::PerpetualParams);
}

/// `→` never writes. From the template selector, forward would create a model
/// directory and a `config_file`; that stays on `Enter`, where the user asked
/// for it. Same for the two manager actions, which are not steps of the path.
#[test]
fn right_never_creates_anything() {
    let (temp, mut app) = test_app("path-right-inert");
    app.screen = Screen::TemplateSelector;

    handle_key(&mut app, KeyCode::Right);

    assert_eq!(app.screen, Screen::TemplateSelector);
    assert!(
        temp.storage().list_models().unwrap_or_default().is_empty(),
        "→ on the template selector created a model"
    );

    let (_temp, mut app, _name) = {
        let (temp, mut app) = test_app("path-right-manager");
        app.finish_template_selector();
        let name = app.active_model_name.clone().expect("model");
        (temp, app, name)
    };
    for action in [ModelAction::Rename, ModelAction::Delete] {
        app.screen = Screen::ModelActions;
        app.model_actions.selected = action.index();
        handle_key(&mut app, KeyCode::Right);
        assert_eq!(
            app.screen,
            Screen::ModelActions,
            "→ opened {action:?}, which is not a step of the path"
        );
    }
}

/// The breadcrumb is not allowed to take `←`/`→` from anyone who already had
/// them. These are the four places they were bound before, and each is checked
/// by its *effect*, not by where the screen went.
#[test]
fn the_existing_arrow_bindings_still_do_what_they_did() {
    use crate::tui::app::{
        InferenceConfig, ModelConfig, PerpetualConfig, RunConfig, RunMode,
    };

    // 1. The layer builder cycles the layer kind.
    let (_temp, mut app) = test_app("arrows-layer-kind");
    app.finish_template_selector();
    handle_key(&mut app, KeyCode::Char('e'));
    handle_key(&mut app, KeyCode::Esc); // browse → add
    let kind = app.layer_builder.current_kind.clone();
    handle_key(&mut app, KeyCode::Right);
    assert_ne!(app.layer_builder.current_kind, kind, "→ stopped cycling kinds");
    handle_key(&mut app, KeyCode::Left);
    assert_eq!(app.layer_builder.current_kind, kind);
    assert_eq!(app.screen, Screen::LayerBuilder, "the arrows moved screen");

    // 2. The dataset selector cycles datasets.
    app.screen = Screen::DatasetSelector;
    app.training_params.datasets = vec!["a".into(), "b".into(), "c".into()];
    app.select_dataset(0);
    handle_key(&mut app, KeyCode::Right);
    assert_eq!(app.training_params.selected_dataset, 1, "→ stopped cycling datasets");
    handle_key(&mut app, KeyCode::Left);
    assert_eq!(app.training_params.selected_dataset, 0);
    assert_eq!(app.screen, Screen::DatasetSelector);

    // 3. The seed toggle on the two inference forms.
    app.screen = Screen::InferenceParams;
    app.inference_params.field_idx = 0;
    let before = app.inference_params.random_seed;
    handle_key(&mut app, KeyCode::Left);
    assert_ne!(app.inference_params.random_seed, before, "← stopped toggling the seed");
    assert_eq!(app.screen, Screen::InferenceParams, "the arrows moved screen");

    app.screen = Screen::PerpetualParams;
    app.perpetual_params.field_idx = 0;
    let before = app.perpetual_params.random_seed;
    handle_key(&mut app, KeyCode::Right);
    assert_ne!(app.perpetual_params.random_seed, before, "→ stopped toggling the seed");
    assert_eq!(app.screen, Screen::PerpetualParams, "the arrows moved screen");

    // …and the regime toggle at the far end of the same form.
    app.perpetual_params.field_idx = super::app::PERPETUAL_PARAM_FIELD_NAMES.len() - 1;
    let regime = app.perpetual_params.regime;
    handle_key(&mut app, KeyCode::Left);
    assert_ne!(app.perpetual_params.regime, regime, "← stopped toggling the regime");

    // 4. A perpetual run steers its tempo from the monitor.
    let (_temp, mut app) = test_app("arrows-perpetual-monitor");
    app.screen = Screen::Monitor;
    app.monitor.model_config = Some(ModelConfig {
        model_name: Some("arrows".to_string()),
        input_size: (32, 32, 5),
        layers: Vec::new(),
        inference: InferenceConfig::default(),
        run: RunConfig {
            mode: RunMode::Perpetual(PerpetualConfig::default()),
        },
    });
    handle_key(&mut app, KeyCode::Right);
    assert_eq!(
        app.drain_monitor_control_commands(),
        vec![TrainingControlCommand::NudgeTempo(1)],
        "→ stopped steering the drift"
    );
    assert_eq!(app.screen, Screen::Monitor);
}
