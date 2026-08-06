//! File purpose: The on-disk layout — where `Models/`, `datasets/` and the
//! checkpoints live, how a `config_file` is read and written, and the two
//! destructive model-manager operations (rename, delete).
//!
//! This is deliberately *not* in `batlab_core`. The engine takes bytes
//! ([`ModelConfig::from_json_bytes`], [`batlab_core::Model::load_checkpoint_bytes`])
//! and knows nothing of paths, so that the same inference chain can run in a
//! browser where there is no filesystem at all. Deciding that a model's config
//! sits at `Models/<name>/config_file` is a host decision, and it is made here.
//!
//! ## The root is injected, not deduced
//!
//! Every path below hangs off a [`Storage`] root. The root used to be a global
//! deduced from the workspace, which meant a test had no way to say "not the
//! real `Models/`": a plain `cargo test` rewrote `Models/Stable_Diffusion/config_file`
//! in the repository (`docs/reports/PERPETUAL_INFERENCE.md` §5). Tests now build
//! a `Storage` on a temporary directory and the workspace root is only ever the
//! *default*.

use batlab_core::config::ModelConfig;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct SavedModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub input_size: (u32, u32, u32),
    pub layer_count: usize,
    /// Checkpoint file names found under `pretrained_weights/`, sorted. The
    /// model list shows them because "which weights does this thing have?" is
    /// the question that decides whether a model is worth opening at all.
    pub checkpoints: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    pub name: String,
    pub path: String,
}

fn serde_to_io(err: serde_json::Error) -> io::Error {
    io::Error::other(err)
}

// ---------------------------------------------------------------------------
// Model names
// ---------------------------------------------------------------------------

/// Characters a model name may contain, beyond ASCII alphanumerics.
///
/// The set is deliberately tiny: a model name becomes a directory name, and
/// `rename`/`delete` act on that directory. Excluding `/`, `\` and everything
/// else means a traversal (`../../etc`) cannot be *spelled*, let alone walked —
/// the path guard in [`Storage::model_path_guarded`] is the second lock, not the
/// first.
const NAME_EXTRA_CHARS: [char; 3] = ['-', '_', '.'];

/// Longest accepted model name. Well under every filesystem's limit; the point
/// is to keep the list readable, not to test the kernel.
pub const MAX_MODEL_NAME_LEN: usize = 64;

/// Why a model name was refused. The message is user-facing: it is shown in the
/// rename form under the field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameError {
    Empty,
    TooLong,
    Hidden,
    BadChar(char),
}

impl fmt::Display for NameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NameError::Empty => write!(f, "the name must not be empty"),
            NameError::TooLong => write!(f, "the name must be at most {MAX_MODEL_NAME_LEN} characters"),
            NameError::Hidden => write!(f, "the name must not start with '.'"),
            NameError::BadChar(c) => write!(
                f,
                "'{c}' is not allowed — use letters, digits, '-', '_' or '.'"
            ),
        }
    }
}

/// Accepts exactly the names that are safe to use as a directory under
/// `Models/`. Called by every path-producing entry point, so an invalid name
/// never reaches the filesystem.
pub fn validate_model_name(name: &str) -> Result<(), NameError> {
    if name.is_empty() {
        return Err(NameError::Empty);
    }
    if name.len() > MAX_MODEL_NAME_LEN {
        return Err(NameError::TooLong);
    }
    if name.starts_with('.') {
        // Rules out "." and ".." along with hidden directories, which
        // `list_models` skips anyway — a model you cannot see is a model you
        // cannot manage.
        return Err(NameError::Hidden);
    }
    if let Some(bad) = name
        .chars()
        .find(|c| !c.is_ascii_alphanumeric() && !NAME_EXTRA_CHARS.contains(c))
    {
        return Err(NameError::BadChar(bad));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Manager errors
// ---------------------------------------------------------------------------

/// What can go wrong in [`Storage::rename_model`] / [`Storage::delete_model`].
#[derive(Debug)]
pub enum ManagerError {
    /// The name would not be a safe directory name.
    InvalidName(NameError),
    /// No such model under `Models/`.
    NotFound(String),
    /// The destination name is taken.
    AlreadyExists(String),
    /// Rename to the name it already has.
    Unchanged(String),
    /// The resolved path is not a direct child of `Models/` — a symlinked model
    /// directory, or a name that resolved somewhere it has no business being.
    /// Nothing is touched.
    OutsideModelsRoot(PathBuf),
    Io(io::Error),
}

impl fmt::Display for ManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManagerError::InvalidName(err) => write!(f, "invalid model name: {err}"),
            ManagerError::NotFound(name) => write!(f, "no model named '{name}' under Models/"),
            ManagerError::AlreadyExists(name) => write!(f, "a model named '{name}' already exists"),
            ManagerError::Unchanged(name) => write!(f, "'{name}' is already the name of this model"),
            ManagerError::OutsideModelsRoot(path) => write!(
                f,
                "refusing to touch {} — it is not a direct child of Models/",
                path.display()
            ),
            ManagerError::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ManagerError {}

impl From<NameError> for ManagerError {
    fn from(err: NameError) -> Self {
        ManagerError::InvalidName(err)
    }
}

impl From<io::Error> for ManagerError {
    fn from(err: io::Error) -> Self {
        ManagerError::Io(err)
    }
}

// ---------------------------------------------------------------------------
// Storage root
// ---------------------------------------------------------------------------

/// A storage root: the directory holding `Models/`, `datasets/` and the
/// generated sample folders.
///
/// Cheap to clone, holds nothing but the path. Construct one with
/// [`Storage::at`] to point somewhere else — that is what tests do, and what
/// `BATLAB_ROOT` does for a throwaway end-to-end session.
#[derive(Debug, Clone)]
pub struct Storage {
    root: PathBuf,
}

impl Default for Storage {
    fn default() -> Self {
        Self::at(default_root())
    }
}

impl Storage {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    // --- Directory layout (pure path arithmetic, no I/O) ---

    /// `<root>/Models` — computed, not created. Listing must not have side
    /// effects: `list_models` on a fresh root should report nothing, not
    /// conjure directories.
    pub fn models_root(&self) -> PathBuf {
        self.root.join("Models")
    }

    pub fn datasets_root(&self) -> PathBuf {
        self.root.join("datasets")
    }

    // --- Directory layout (creating) ---

    pub fn datasets_dir(&self) -> io::Result<PathBuf> {
        let dir = self.datasets_root();
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    pub fn models_dir(&self) -> io::Result<PathBuf> {
        let dir = self.models_root();
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    pub fn model_dir(&self, model_name: &str) -> io::Result<PathBuf> {
        validate_model_name(model_name).map_err(|err| io::Error::other(err.to_string()))?;
        let dir = self.models_dir()?.join(model_name);
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    pub fn model_weights_dir(&self, model_name: &str) -> io::Result<PathBuf> {
        let dir = self.model_dir(model_name)?.join("pretrained_weights");
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    pub fn default_model_checkpoint_path(&self, model_name: &str) -> io::Result<PathBuf> {
        Ok(self.model_weights_dir(model_name)?.join("latest.ckpt"))
    }

    pub fn model_config_path(&self, model_name: &str) -> io::Result<PathBuf> {
        Ok(self.model_dir(model_name)?.join("config_file"))
    }

    pub fn next_model_name(&self) -> io::Result<String> {
        let root = self.models_dir()?;
        let mut next_index = 1usize;
        loop {
            let name = format!("model-{next_index:03}");
            if !root.join(&name).exists() {
                return Ok(name);
            }
            next_index += 1;
        }
    }

    // --- Config I/O ---

    pub fn write_model_config(
        &self,
        model_name: &str,
        config: &ModelConfig,
    ) -> io::Result<PathBuf> {
        let path = self.model_config_path(model_name)?;
        let mut persisted = config.clone();
        persisted.model_name = Some(model_name.to_string());
        // Via l'API bytes du moteur, pas serde_json directement : c'est la même
        // porte que prendra un build wasm, elle doit rester la seule.
        let bytes = persisted.to_json_bytes().map_err(serde_to_io)?;
        fs::write(&path, bytes)?;
        Ok(path)
    }

    pub fn load_model_config_for_model(&self, model_name: &str) -> io::Result<ModelConfig> {
        let path = self.model_config_path(model_name)?;
        load_model_config(&path)
    }

    // --- Listing ---

    /// Checkpoint files of a model, sorted by name. Read-only: a model with no
    /// `pretrained_weights/` yet reports an empty list rather than growing one.
    fn checkpoint_names(&self, model_name: &str) -> Vec<String> {
        let dir = self.models_root().join(model_name).join("pretrained_weights");
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|entry| entry.path().is_file())
            .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
            .filter(|name| !name.starts_with('.'))
            .collect();
        names.sort();
        names
    }

    pub fn list_model_checkpoints(&self, model_name: &str) -> io::Result<Vec<CheckpointEntry>> {
        let dir = self.model_weights_dir(model_name)?;
        let mut entries = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.'))
            {
                continue;
            }
            let display_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("checkpoint")
                .to_string();
            entries.push(CheckpointEntry {
                name: display_name,
                path: path.to_string_lossy().to_string(),
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    pub fn list_models(&self) -> io::Result<Vec<SavedModelEntry>> {
        let mut models = Vec::new();
        let root = self.models_root();
        let entries = match fs::read_dir(&root) {
            Ok(entries) => entries,
            // A root with no `Models/` holds no models. Saying so is the honest
            // answer, and it keeps listing free of side effects.
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(models),
            Err(err) => return Err(err),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            let config_path = path.join("config_file");
            if !config_path.is_file() {
                continue;
            }
            let config = match load_model_config(&config_path) {
                Ok(config) => config,
                Err(_) => continue,
            };
            models.push(SavedModelEntry {
                name: name.to_string(),
                path: config_path,
                input_size: config.input_size,
                layer_count: config.layers.len(),
                checkpoints: self.checkpoint_names(name),
            });
        }
        models.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(models)
    }

    pub fn list_datasets(&self) -> io::Result<Vec<String>> {
        let mut datasets = Vec::new();
        let dir = self.datasets_dir()?;
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.'))
            {
                continue;
            }
            datasets.push(path.display().to_string());
        }

        let legacy_cifar = self.root.join("cifar");
        if legacy_cifar.exists() {
            let legacy = legacy_cifar.display().to_string();
            if !datasets.iter().any(|path| path == &legacy) {
                datasets.push(legacy);
            }
        }

        datasets.sort();
        Ok(datasets)
    }

    // --- The manager: rename and delete ---

    /// Resolve `<Models>/<name>` and prove it is a directory that `Models/`
    /// directly owns, before anything destructive happens to it.
    ///
    /// Both sides are canonicalised, so a model directory that is a *symlink*
    /// to somewhere else fails the check and is refused rather than followed:
    /// `remove_dir_all` on a resolved symlink target would delete the target's
    /// contents. Refusing is the right answer — the manager's blast radius is
    /// one direct child of `Models/`, and nothing else.
    fn model_path_guarded(&self, name: &str) -> Result<PathBuf, ManagerError> {
        validate_model_name(name)?;
        let models_root = self
            .models_root()
            .canonicalize()
            .map_err(|_| ManagerError::NotFound(name.to_string()))?;
        let candidate = self.models_root().join(name);
        let resolved = candidate
            .canonicalize()
            .map_err(|_| ManagerError::NotFound(name.to_string()))?;
        if resolved.parent() != Some(models_root.as_path()) {
            return Err(ManagerError::OutsideModelsRoot(resolved));
        }
        if !resolved.is_dir() {
            return Err(ManagerError::NotFound(name.to_string()));
        }
        Ok(resolved)
    }

    /// Rename `Models/<from>/` to `Models/<to>/` **and** rewrite the model's
    /// `config_file` so the two agree.
    ///
    /// The directory name and the config's `model_name` are two copies of one
    /// fact; a rename that moved only the directory would leave the config
    /// claiming the old name, and every checkpoint path stored in it pointing at
    /// a directory that no longer exists. So this also retargets the three
    /// places a checkpoint path is persisted (`inference.checkpoint`, a training
    /// run's `checkpoint_path`, a perpetual run's `checkpoint`) when they point
    /// inside the model's own directory.
    ///
    /// If the config rewrite fails the directory rename is undone, so the pair
    /// is never left half-renamed.
    pub fn rename_model(&self, from: &str, to: &str) -> Result<PathBuf, ManagerError> {
        validate_model_name(to)?;
        let from_path = self.model_path_guarded(from)?;
        if from == to {
            return Err(ManagerError::Unchanged(to.to_string()));
        }
        let to_path = self.models_root().join(to);
        // `exists()` is case-insensitive on macOS, so a pure case change would
        // look like a collision with itself. Compare the resolved paths: only a
        // *different* directory is a real collision.
        if to_path.exists()
            && to_path
                .canonicalize()
                .map(|resolved| resolved != from_path)
                .unwrap_or(true)
        {
            return Err(ManagerError::AlreadyExists(to.to_string()));
        }

        fs::rename(&from_path, &to_path)?;

        match self.retarget_config(from, &from_path, to, &to_path) {
            Ok(()) => Ok(to_path),
            Err(err) => {
                // Undo, so the caller sees "nothing happened" rather than a
                // directory whose config disagrees with its name.
                let _ = fs::rename(&to_path, &from_path);
                Err(err)
            }
        }
    }

    /// Rewrite the renamed model's config so `model_name` and every checkpoint
    /// path inside its own directory follow the move.
    fn retarget_config(
        &self,
        from: &str,
        from_path: &Path,
        to: &str,
        to_path: &Path,
    ) -> Result<(), ManagerError> {
        let config_path = to_path.join("config_file");
        if !config_path.is_file() {
            // A directory with no config is not a model the UI can show, but a
            // rename of it is still a legitimate no-op on the config side.
            return Ok(());
        }
        let mut config = load_model_config(&config_path)?;
        config.model_name = Some(to.to_string());

        // The stored paths were written against the *uncanonicalised* root
        // (`<root>/Models/<from>/…`), while `from_path` is canonical
        // (`/private/tmp/…` on macOS). Try both spellings.
        let old_dirs = [self.models_root().join(from), from_path.to_path_buf()];
        let retarget = |stored: &mut Option<String>| {
            if let Some(path) = stored.as_deref()
                && let Some(moved) = old_dirs
                    .iter()
                    .find_map(|old| retarget_under(path, old, to_path))
            {
                *stored = Some(moved);
            }
        };

        retarget(&mut config.inference.checkpoint);
        match &mut config.run.mode {
            batlab_core::config::RunMode::Train(train) => retarget(&mut train.checkpoint_path),
            batlab_core::config::RunMode::Perpetual(perpetual) => {
                retarget(&mut perpetual.checkpoint)
            }
            batlab_core::config::RunMode::Infer => {}
        }

        let bytes = config.to_json_bytes().map_err(serde_to_io)?;
        fs::write(&config_path, bytes)?;
        Ok(())
    }

    /// Delete `Models/<name>/` and everything under it. Irreversible.
    ///
    /// The path is resolved and proven to be a direct child of `Models/` first
    /// (see [`Storage::model_path_guarded`]); anything else is refused with
    /// nothing touched. Confirming the user meant it is the UI's job — this
    /// function assumes the question was already asked.
    pub fn delete_model(&self, name: &str) -> Result<(), ManagerError> {
        let path = self.model_path_guarded(name)?;
        fs::remove_dir_all(path)?;
        Ok(())
    }
}

/// If `stored` sits inside `old_dir`, return it rebased onto `new_dir`.
fn retarget_under(stored: &str, old_dir: &Path, new_dir: &Path) -> Option<String> {
    let relative = Path::new(stored).strip_prefix(old_dir).ok()?;
    Some(new_dir.join(relative).to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// The default root
// ---------------------------------------------------------------------------

/// Where [`Storage::default`] points.
///
/// `BATLAB_ROOT`, if set, wins: that is how a session can be pointed at a
/// throwaway directory without touching the repository's own `Models/`.
/// Otherwise the workspace root, found by walking up until a `Cargo.toml` that
/// declares `[workspace]` — not by a fixed number of `parent()` hops. This code
/// moved twice during the restructure — `bat_building/` → `crates/batlab-core/`
/// → `crates/batlab-ui/` — and each hop would have silently resolved `Models/`
/// to `crates/Models/` with a hard-coded count: every model and dataset
/// invisible, no error until first use.
pub fn default_root() -> PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        if let Some(root) = std::env::var_os("BATLAB_ROOT") {
            let root = PathBuf::from(root);
            if !root.as_os_str().is_empty() {
                return root;
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .find(|dir| {
                fs::read_to_string(dir.join("Cargo.toml"))
                    .is_ok_and(|manifest| manifest.contains("[workspace]"))
            })
            .expect("batlab_ui should live under a cargo workspace root")
            .to_path_buf()
    })
    .clone()
}

// ---------------------------------------------------------------------------
// Free functions over the default root
//
// The CLI has exactly one storage root for the life of the process, so it calls
// these rather than threading a `Storage` through every headless entry point.
// Anything that needs to be *pointed* somewhere — the TUI, every test — holds a
// `Storage` instead.
// ---------------------------------------------------------------------------

pub fn project_root() -> PathBuf {
    default_root()
}

pub fn datasets_dir() -> io::Result<PathBuf> {
    Storage::default().datasets_dir()
}

pub fn models_dir() -> io::Result<PathBuf> {
    Storage::default().models_dir()
}

pub fn model_dir(model_name: &str) -> io::Result<PathBuf> {
    Storage::default().model_dir(model_name)
}

pub fn model_weights_dir(model_name: &str) -> io::Result<PathBuf> {
    Storage::default().model_weights_dir(model_name)
}

pub fn default_model_checkpoint_path(model_name: &str) -> io::Result<PathBuf> {
    Storage::default().default_model_checkpoint_path(model_name)
}

pub fn model_config_path(model_name: &str) -> io::Result<PathBuf> {
    Storage::default().model_config_path(model_name)
}

pub fn next_model_name() -> io::Result<String> {
    Storage::default().next_model_name()
}

pub fn write_model_config(model_name: &str, config: &ModelConfig) -> io::Result<PathBuf> {
    Storage::default().write_model_config(model_name, config)
}

/// Reads a `config_file` from an explicit path — no storage root involved, so
/// it stays a free function.
pub fn load_model_config(path: &Path) -> io::Result<ModelConfig> {
    let bytes = fs::read(path)?;
    let mut config = ModelConfig::from_json_bytes(&bytes).map_err(serde_to_io)?;
    if config.model_name.is_none() {
        config.model_name = path
            .parent()
            .and_then(|dir| dir.file_name())
            .and_then(|name| name.to_str())
            .map(|name| name.to_string())
            .or_else(|| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(|stem| stem.to_string())
            });
    }
    Ok(config)
}

pub fn load_model_config_for_model(model_name: &str) -> io::Result<ModelConfig> {
    Storage::default().load_model_config_for_model(model_name)
}

pub fn list_model_checkpoints(model_name: &str) -> io::Result<Vec<CheckpointEntry>> {
    Storage::default().list_model_checkpoints(model_name)
}

pub fn list_models() -> io::Result<Vec<SavedModelEntry>> {
    Storage::default().list_models()
}

pub fn list_datasets() -> io::Result<Vec<String>> {
    Storage::default().list_datasets()
}

// ---------------------------------------------------------------------------
// Test sandbox
// ---------------------------------------------------------------------------

/// A storage root under the system temp directory, removed on drop.
///
/// Every test that touches storage goes through this. Without it a test either
/// writes into the repository's own `Models/` or asserts nothing at all — the
/// first is what actually happened before this type existed.
#[cfg(test)]
pub(crate) struct TempRoot {
    path: PathBuf,
}

#[cfg(test)]
impl TempRoot {
    pub(crate) fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "batlab-test-{}-{tag}-{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("temp root should be creatable");
        Self { path }
    }

    pub(crate) fn storage(&self) -> Storage {
        Storage::at(&self.path)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batlab_core::config::{InferenceConfig, RunConfig, RunMode, TrainingConfig};

    fn a_config() -> ModelConfig {
        ModelConfig {
            model_name: None,
            input_size: (8, 8, 1),
            layers: Vec::new(),
            inference: InferenceConfig::default(),
            run: RunConfig {
                mode: RunMode::Infer,
            },
        }
    }

    fn seed_model(storage: &Storage, name: &str) {
        storage
            .write_model_config(name, &a_config())
            .expect("seeding a model should work");
    }

    /// Nothing errors when the default root lands on the wrong directory — the
    /// TUI just reports an empty `Models/`. Anchor it on the directories that
    /// must be there, so a future move fails loudly instead of silently.
    ///
    /// Read-only by construction: it asserts, it does not create.
    #[test]
    fn project_root_is_the_workspace_that_holds_models_and_datasets() {
        // Only meaningful for the deduced root; a session pointed elsewhere by
        // BATLAB_ROOT is exactly the case this test does not describe.
        if std::env::var_os("BATLAB_ROOT").is_some() {
            return;
        }
        let root = project_root();
        assert!(
            root.join("Cargo.toml").exists(),
            "project_root() = {} has no Cargo.toml",
            root.display()
        );
        assert!(
            root.join("Models").is_dir(),
            "project_root() = {} does not hold Models/",
            root.display()
        );
        assert!(
            root.join("crates").is_dir(),
            "project_root() = {} does not hold crates/",
            root.display()
        );
    }

    /// Listing must not create anything. `list_models` used to go through
    /// `models_dir()`, which calls `create_dir_all` — so merely *looking* at a
    /// root left `Models/` behind in it.
    #[test]
    fn listing_a_fresh_root_creates_nothing() {
        let temp = TempRoot::new("listing");
        let storage = temp.storage();

        assert!(storage.list_models().expect("listing should work").is_empty());

        assert!(
            !temp.path().join("Models").exists(),
            "listing conjured a Models/ directory"
        );
    }

    #[test]
    fn a_listed_model_reports_its_geometry_layers_and_checkpoints() {
        let temp = TempRoot::new("entry");
        let storage = temp.storage();
        seed_model(&storage, "alpha");
        fs::write(
            storage
                .model_weights_dir("alpha")
                .expect("weights dir")
                .join("latest.ckpt"),
            b"not really a checkpoint",
        )
        .expect("checkpoint write");

        let models = storage.list_models().expect("listing should work");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "alpha");
        assert_eq!(models[0].input_size, (8, 8, 1));
        assert_eq!(models[0].layer_count, 0);
        assert_eq!(models[0].checkpoints, vec!["latest.ckpt".to_string()]);
    }

    // --- Name validation ---

    #[test]
    fn model_names_reject_everything_that_is_not_a_safe_directory_name() {
        assert!(validate_model_name("Greyscale_Diffusion_L-2").is_ok());
        assert!(validate_model_name("model.001").is_ok());

        assert_eq!(validate_model_name(""), Err(NameError::Empty));
        assert_eq!(validate_model_name("."), Err(NameError::Hidden));
        assert_eq!(validate_model_name(".."), Err(NameError::Hidden));
        assert_eq!(validate_model_name(".hidden"), Err(NameError::Hidden));
        assert_eq!(validate_model_name("a/b"), Err(NameError::BadChar('/')));
        assert_eq!(
            validate_model_name("../../etc"),
            Err(NameError::Hidden),
            "a traversal must not even be spellable"
        );
        assert_eq!(validate_model_name("a b"), Err(NameError::BadChar(' ')));
        assert_eq!(validate_model_name("a\0b"), Err(NameError::BadChar('\0')));
        assert_eq!(
            validate_model_name(&"x".repeat(MAX_MODEL_NAME_LEN + 1)),
            Err(NameError::TooLong)
        );
    }

    // --- Rename ---

    #[test]
    fn rename_moves_the_directory_and_the_config_name_together() {
        let temp = TempRoot::new("rename");
        let storage = temp.storage();
        seed_model(&storage, "before");

        storage
            .rename_model("before", "after")
            .expect("rename should succeed");

        assert!(!storage.models_root().join("before").exists());
        assert!(storage.models_root().join("after").is_dir());
        let config = storage
            .load_model_config_for_model("after")
            .expect("renamed config should load");
        assert_eq!(
            config.model_name.as_deref(),
            Some("after"),
            "the config kept the old name — the two copies of the name disagree"
        );
    }

    /// The name is persisted twice over: as the directory, and inside every
    /// checkpoint path the config remembers. A rename that fixed only the first
    /// leaves the model pointing at weights that are no longer there.
    #[test]
    fn rename_retargets_checkpoint_paths_that_lived_in_the_model_directory() {
        let temp = TempRoot::new("rename-ckpt");
        let storage = temp.storage();
        let checkpoint = storage
            .default_model_checkpoint_path("before")
            .expect("checkpoint path")
            .to_string_lossy()
            .to_string();
        let elsewhere = temp.path().join("shared.ckpt").to_string_lossy().to_string();
        let mut config = a_config();
        config.inference.checkpoint = Some(checkpoint.clone());
        config.run.mode = RunMode::Train(TrainingConfig {
            lr: 0.001,
            batch_size: 1,
            steps: 1,
            dataset_path: String::new(),
            loss: batlab_core::config::LossMethod::MeanSquared,
            checkpoint_path: Some(elsewhere.clone()),
            load_checkpoint: false,
            optimizer: Default::default(),
            weight_init: Default::default(),
            loss_weighting: Default::default(),
        });
        storage
            .write_model_config("before", &config)
            .expect("seed write");

        storage
            .rename_model("before", "after")
            .expect("rename should succeed");

        let moved = storage
            .load_model_config_for_model("after")
            .expect("renamed config should load");
        assert_eq!(
            moved.inference.checkpoint.as_deref(),
            Some(
                storage
                    .models_root()
                    .join("after")
                    .join("pretrained_weights")
                    .join("latest.ckpt")
                    .to_string_lossy()
                    .as_ref()
            ),
            "a checkpoint inside the model directory must follow it"
        );
        match moved.run.mode {
            RunMode::Train(train) => assert_eq!(
                train.checkpoint_path.as_deref(),
                Some(elsewhere.as_str()),
                "a checkpoint outside the model directory must be left alone"
            ),
            _ => panic!("expected the training run mode to survive the rename"),
        }
    }

    #[test]
    fn rename_refuses_a_collision_an_unchanged_name_and_an_unknown_model() {
        let temp = TempRoot::new("rename-refusals");
        let storage = temp.storage();
        seed_model(&storage, "alpha");
        seed_model(&storage, "beta");

        assert!(matches!(
            storage.rename_model("alpha", "beta"),
            Err(ManagerError::AlreadyExists(_))
        ));
        assert!(matches!(
            storage.rename_model("alpha", "alpha"),
            Err(ManagerError::Unchanged(_))
        ));
        assert!(matches!(
            storage.rename_model("ghost", "gamma"),
            Err(ManagerError::NotFound(_))
        ));
        assert!(matches!(
            storage.rename_model("alpha", "not a name"),
            Err(ManagerError::InvalidName(_))
        ));

        // Nothing moved.
        assert!(storage.models_root().join("alpha").is_dir());
        assert!(storage.models_root().join("beta").is_dir());
    }

    // --- Delete ---

    #[test]
    fn delete_removes_the_model_directory_and_nothing_else() {
        let temp = TempRoot::new("delete");
        let storage = temp.storage();
        seed_model(&storage, "doomed");
        seed_model(&storage, "spared");

        storage.delete_model("doomed").expect("delete should work");

        assert!(!storage.models_root().join("doomed").exists());
        assert!(storage.models_root().join("spared").is_dir());
    }

    /// The guard-rail. A model name is the only thing the manager will delete,
    /// and only where `Models/` directly owns it.
    #[test]
    fn delete_refuses_every_path_that_is_not_a_direct_child_of_models() {
        let temp = TempRoot::new("delete-guard");
        let storage = temp.storage();
        seed_model(&storage, "kept");
        // A bystander directory next to the storage root, of the kind a
        // traversal would be aiming at.
        let bystander = temp.path().join("precious");
        fs::create_dir_all(bystander.join("data")).expect("bystander");

        for pathological in [
            "..",
            "../..",
            "../precious",
            "/",
            "/etc",
            "Models/kept",
            "kept/pretrained_weights",
            "",
            ".",
        ] {
            let result = storage.delete_model(pathological);
            assert!(
                result.is_err(),
                "delete_model({pathological:?}) was accepted — it must not be"
            );
        }

        assert!(bystander.is_dir(), "a bystander directory was deleted");
        assert!(storage.models_root().join("kept").is_dir());
        assert!(temp.path().is_dir(), "the storage root itself was deleted");
    }

    /// A model directory that is really a symlink elsewhere is refused rather
    /// than followed — `remove_dir_all` on the resolved target would delete
    /// somebody else's files.
    #[cfg(unix)]
    #[test]
    fn delete_refuses_a_symlinked_model_directory() {
        let temp = TempRoot::new("delete-symlink");
        let storage = temp.storage();
        seed_model(&storage, "real");
        let outside = temp.path().join("outside");
        fs::create_dir_all(&outside).expect("outside dir");
        fs::write(outside.join("keep-me"), b"x").expect("outside file");
        std::os::unix::fs::symlink(&outside, storage.models_root().join("linked"))
            .expect("symlink");

        let result = storage.delete_model("linked");

        assert!(
            matches!(result, Err(ManagerError::OutsideModelsRoot(_))),
            "a symlinked model directory must be refused, got {result:?}"
        );
        assert!(outside.join("keep-me").exists(), "the symlink was followed");
    }

    #[test]
    fn delete_refuses_an_unknown_model() {
        let temp = TempRoot::new("delete-unknown");
        let storage = temp.storage();
        assert!(matches!(
            storage.delete_model("ghost"),
            Err(ManagerError::NotFound(_))
        ));
    }
}
