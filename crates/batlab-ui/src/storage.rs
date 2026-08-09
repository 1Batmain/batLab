//! File purpose: The on-disk layout — where `Models/`, `datasets/` and the
//! checkpoints live, how a `config_file` is read and written, and the
//! model-manager operations (rename, duplicate, delete).
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

use crate::clock;
use batlab_core::config::ModelConfig;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

#[derive(Debug, Clone)]
pub struct SavedModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub input_size: (u32, u32, u32),
    pub layer_count: usize,
    /// Checkpoints found under `pretrained_weights/`, newest first. The model
    /// list shows them because "which weights does this thing have?" is the
    /// question that decides whether a model is worth opening at all — and
    /// "when were they written?" is the one right behind it.
    pub checkpoints: Vec<CheckpointEntry>,
    /// The stack, its cost and its reach — read off the same `config_file` the
    /// fields above come from, so the panel beside the list never re-opens a
    /// file to say what the row already knows.
    pub architecture: batlab_core::ArchitectureSummary,
}

#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    pub name: String,
    pub path: String,
    /// When the file was last written, if the filesystem will say. `None` is
    /// shown as such — an invented date is worse than a missing one, since the
    /// whole point of showing it is to pick "the one from last night".
    pub modified: Option<SystemTime>,
    pub size_bytes: u64,
}

impl CheckpointEntry {
    /// Read one checkpoint's identity off disk. `None` when the path is not a
    /// name we can show.
    fn of(path: &Path) -> Option<Self> {
        let name = path.file_name().and_then(|name| name.to_str())?.to_string();
        let metadata = fs::metadata(path).ok();
        Some(Self {
            name,
            path: path.to_string_lossy().to_string(),
            modified: metadata.as_ref().and_then(|meta| meta.modified().ok()),
            size_bytes: metadata.map(|meta| meta.len()).unwrap_or(0),
        })
    }

    /// The date and size, as one line for a listing: `2026-08-08 10:41 · 14.2 MB`.
    pub fn detail(&self) -> String {
        let when = self
            .modified
            .and_then(clock::short)
            .unwrap_or_else(|| "date unknown".to_string());
        format!("{when} · {}", clock::human_bytes(self.size_bytes))
    }
}

// ---------------------------------------------------------------------------
// Checkpoint naming
// ---------------------------------------------------------------------------

/// The name every run's checkpoint still answers to: the most recent one.
///
/// A run no longer *overwrites* it — it writes `run-<stamp>.ckpt` and then
/// points this name at that file (see [`point_latest_at`]) — but the name
/// stays, because "open a model and train" means "keep training the model" and
/// that default is spelled `latest.ckpt` in the weight selector, in every
/// model's `config_file`, and in three reports.
pub const LATEST_CHECKPOINT_NAME: &str = "latest.ckpt";

/// What a run's own checkpoint is called: `run-<YYYY-MM-DD_HHMM>.ckpt`.
///
/// The stamp is local time, zero-padded, biggest unit first, so **sorting the
/// names is sorting the runs** — `pretrained_weights/` is listed by name in
/// more than one place and this is what keeps "last is newest" true there.
///
/// Two runs in the same minute (a 20-step smoke test, twice) would collide, so
/// a taken name gains an `_02`, `_03`… suffix. The separator is an underscore
/// and not a dash on purpose: `-` sorts *before* `.`, so `run-…_1041-02.ckpt`
/// would come out ahead of `run-…_1041.ckpt` and break the one property the
/// stamp exists for. `_` sorts after `.`, so the pair stays in run order.
pub fn new_run_checkpoint_path_in(dir: &Path, at: SystemTime) -> PathBuf {
    let stamp = clock::stamp(at);
    let first = dir.join(format!("run-{stamp}.ckpt"));
    if !first.exists() {
        return first;
    }
    for nth in 2..=99u32 {
        let candidate = dir.join(format!("run-{stamp}_{nth:02}.ckpt"));
        if !candidate.exists() {
            return candidate;
        }
    }
    dir.join(format!("run-{stamp}_99.ckpt"))
}

/// Point `latest.ckpt` at the checkpoint just written, beside it.
///
/// **A hard link, not a copy.** The alternative — writing the bytes twice —
/// doubles the cost of every save, and a save is 14 MB for the XL model and
/// happens on every `--checkpoint-every` rotation; a night of training would
/// pay for a second copy of every partial for no reason. A link costs a
/// directory entry, `latest.ckpt` stays a *real file* that `fs::read` opens and
/// `--resume` loads (unlike a symlink, nothing can dangle), and the two names
/// simply describe the same bytes.
///
/// Written through a hidden scratch name and a rename, so `latest.ckpt` is
/// never observed missing or half-linked. The scratch name starts with a dot
/// precisely so a listing racing this call cannot offer it as weights.
///
/// Falls back to a copy when the filesystem refuses to link (a `Models/` spread
/// across devices, an exotic mount): the name matters more than the trick.
pub fn point_latest_at(written: &Path) -> io::Result<PathBuf> {
    let parent = written.parent().ok_or_else(|| {
        io::Error::other(format!(
            "checkpoint {} has no parent directory",
            written.display()
        ))
    })?;
    let target = parent.join(LATEST_CHECKPOINT_NAME);
    if written.file_name() == Some(std::ffi::OsStr::new(LATEST_CHECKPOINT_NAME)) {
        // The run wrote `latest.ckpt` itself; there is nothing to point.
        return Ok(target);
    }
    let scratch = parent.join(".latest.ckpt.tmp");
    let _ = fs::remove_file(&scratch);
    if fs::hard_link(written, &scratch).is_err() {
        fs::copy(written, &scratch)?;
    }
    match fs::rename(&scratch, &target) {
        Ok(()) => Ok(target),
        Err(err) => {
            let _ = fs::remove_file(&scratch);
            Err(err)
        }
    }
}

/// Newest first, ties broken by name.
///
/// The selector opens on the most recent run, which is what "continue training"
/// means; `latest.ckpt` and the dated file it points at share an mtime, and the
/// tie-break puts `latest.ckpt` first — the row the default already sits on.
fn sort_newest_first(entries: &mut [CheckpointEntry]) {
    entries.sort_by(|a, b| b.modified.cmp(&a.modified).then_with(|| a.name.cmp(&b.name)));
}

/// Every checkpoint in `dir`, newest first. A directory that cannot be read
/// holds no checkpoints — listing must not have side effects.
fn checkpoint_entries_in(dir: &Path) -> Vec<CheckpointEntry> {
    let Ok(read) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut entries: Vec<CheckpointEntry> = read
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| is_checkpoint_file(path))
        .filter_map(|path| CheckpointEntry::of(&path))
        .collect();
    sort_newest_first(&mut entries);
    entries
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
/// rename and duplicate forms under the field.
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

/// What can go wrong in [`Storage::rename_model`], [`Storage::duplicate_model`]
/// or [`Storage::delete_model`].
#[derive(Debug)]
pub enum ManagerError {
    /// The name would not be a safe directory name.
    InvalidName(NameError),
    /// No such model under `Models/`.
    NotFound(String),
    /// The destination name is taken.
    AlreadyExists(String),
    /// Rename — or duplicate — to the name it already has.
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
// Duplicating a model
// ---------------------------------------------------------------------------

/// How much of a model travels with its duplicate.
///
/// Weights are the default at the call site that matters: duplicating exists so
/// a foundation model can be *fine-tuned* under another name, and a fine-tune
/// with no weights to start from is just a new model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsToCopy {
    /// The architecture alone — the copy starts from fresh weights.
    None,
    /// The weights the model currently answers to, and nothing else.
    ///
    /// **Not the history.** A model trained overnight with `--checkpoint-every`
    /// holds dozens of `run-*.ckpt`, 14 MB each for the XL; copying the lot to
    /// make one fine-tune would cost half a gigabyte to carry runs the copy has
    /// no claim to. What travels is the newest run — the one `latest.ckpt`
    /// designates — under its own dated name, with `latest.ckpt` re-linked onto
    /// it inside the copy.
    Newest,
}

/// What a duplication actually did, so the UI can *say* it instead of implying
/// it. `copied_weights` is empty when nothing was carried — including when
/// weights were asked for and the source model had none yet.
#[derive(Debug, Clone)]
pub struct DuplicateOutcome {
    pub path: PathBuf,
    /// Checkpoint file names written into the copy, in the order they appeared:
    /// the dated run first, then `latest.ckpt` pointing at it.
    pub copied_weights: Vec<String>,
}

/// The `(device, inode)` pair of a file, or `None` if it cannot be read.
///
/// Used to answer one question: *which* dated run is `latest.ckpt`? The two are
/// one set of bytes under two names (`point_latest_at`), and comparing identity
/// is how that pairing is recovered without reading 14 MB twice.
fn file_identity(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::metadata(path).ok()?;
    Some((meta.dev(), meta.ino()))
}

/// The single checkpoint that carries a model's current weights, or `None` for
/// a model that has never been trained.
///
/// `latest.ckpt` is the answer to "which weights?", but it is a *name*: copying
/// it alone would drop the date the weights were made on, which is the whole
/// point of the dated naming. So the dated file it is hard-linked to is
/// preferred, and `latest.ckpt` is only carried on its own when there is no
/// such sibling (a checkpoint from before the convention, or a copy of one).
fn weights_in_hand(dir: &Path) -> Option<PathBuf> {
    let entries = checkpoint_entries_in(dir);
    let latest = dir.join(LATEST_CHECKPOINT_NAME);
    match file_identity(&latest) {
        Some(identity) => Some(
            entries
                .iter()
                .filter(|entry| entry.name != LATEST_CHECKPOINT_NAME)
                .map(|entry| PathBuf::from(&entry.path))
                .find(|path| file_identity(path) == Some(identity))
                .unwrap_or(latest),
        ),
        // No `latest.ckpt` at all: the newest checkpoint by mtime is the best
        // answer there is, and the copy gains the `latest.ckpt` the source
        // never had — which is what makes it openable on its weights.
        None => entries.first().map(|entry| PathBuf::from(&entry.path)),
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
        Ok(self
            .model_weights_dir(model_name)?
            .join(LATEST_CHECKPOINT_NAME))
    }

    /// Where a run starting at `at` writes its own checkpoint:
    /// `Models/<name>/pretrained_weights/run-<stamp>.ckpt`.
    ///
    /// A *new* file every run — that is the whole feature. The previous run's
    /// weights are still there afterwards, and `latest.ckpt` is pointed at this
    /// one once it has been written ([`point_latest_at`]).
    pub fn new_run_checkpoint_path(
        &self,
        model_name: &str,
        at: SystemTime,
    ) -> io::Result<PathBuf> {
        let dir = self.model_weights_dir(model_name)?;
        Ok(new_run_checkpoint_path_in(&dir, at))
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

    /// Checkpoints of a model, newest first. Read-only: a model with no
    /// `pretrained_weights/` yet reports an empty list rather than growing one.
    fn checkpoint_entries(&self, model_name: &str) -> Vec<CheckpointEntry> {
        let dir = self.models_root().join(model_name).join("pretrained_weights");
        checkpoint_entries_in(&dir)
    }

    /// The same listing the model list shows, for the weight selector — newest
    /// first, so the cursor's default and the first row agree.
    pub fn list_model_checkpoints(&self, model_name: &str) -> io::Result<Vec<CheckpointEntry>> {
        let dir = self.model_weights_dir(model_name)?;
        Ok(checkpoint_entries_in(&dir))
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
                checkpoints: self.checkpoint_entries(name),
                // Computed once, here, off the config that was just parsed —
                // the panel beside the list then costs nothing per keystroke.
                architecture: batlab_core::summarize_architecture(
                    &config.layers,
                    config.input_size,
                ),
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

    /// A dataset's shape, for the resource inventory, from its header alone.
    ///
    /// `None` when the file is missing or is not a `.batraw` — the inventory
    /// then simply has no streamed post, which is the truth rather than a zero.
    pub fn dataset_spec(&self, path: &str) -> Option<batlab_core::DatasetSpec> {
        let header = read_batraw_header(Path::new(path))?;
        Some(batlab_core::DatasetSpec {
            sample_count: header.0,
            // Times the payload's own width, not a hard-coded 4. A BATRAW3
            // sample is a quarter of the f32 one it replaces, and the residency
            // budget is decided on this number.
            sample_bytes: header.1 as u64 * header.2 as u64 * header.3 as u64 * header.4,
        })
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

    /// A free name for a copy of `from`: `<from>-copy`, then `-copy-2`,
    /// `-copy-3`… The form opens on it, so the common case — duplicate, press
    /// Enter — never has to be typed.
    ///
    /// Kept inside [`MAX_MODEL_NAME_LEN`] by trimming the *base*, not the
    /// suffix: a name that says nothing about being a copy would be the one
    /// thing worth keeping if only one could fit.
    pub fn suggest_copy_name(&self, from: &str) -> String {
        let root = self.models_root();
        let with_suffix = |suffix: &str| -> String {
            let room = MAX_MODEL_NAME_LEN.saturating_sub(suffix.len());
            let base: String = from.chars().take(room).collect();
            format!("{base}{suffix}")
        };
        let first = with_suffix("-copy");
        if !root.join(&first).exists() {
            return first;
        }
        for nth in 2..=99u32 {
            let candidate = with_suffix(&format!("-copy-{nth}"));
            if !root.join(&candidate).exists() {
                return candidate;
            }
        }
        with_suffix("-copy-99")
    }

    /// Copy `Models/<from>/` to `Models/<to>/` — a model of its own, with its
    /// own `config_file`, that the original knows nothing about.
    ///
    /// This is what makes fine-tuning safe. Training a foundation model under a
    /// second name used to mean training it *in place*: the run would write into
    /// the same `pretrained_weights/`, and after enough steps on a narrow
    /// dataset the general model was gone. Duplicate first, fine-tune the copy,
    /// and the foundation is still there tomorrow — **the original is not opened
    /// for writing at any point below**.
    ///
    /// The copy's config is rewritten by the same code the rename uses
    /// ([`Storage::retarget_config`]): `model_name` becomes `to`, and every
    /// checkpoint path that pointed inside the source directory is rebased onto
    /// the copy. A config that still named the original would send the copy's
    /// first run back into the directory this whole operation exists to protect.
    ///
    /// What `weights` carries is [`WeightsToCopy`]. Anything that fails leaves
    /// no half-made model: the destination directory is removed and the error
    /// returned.
    pub fn duplicate_model(
        &self,
        from: &str,
        to: &str,
        weights: WeightsToCopy,
    ) -> Result<DuplicateOutcome, ManagerError> {
        validate_model_name(to)?;
        let from_path = self.model_path_guarded(from)?;
        if from == to {
            return Err(ManagerError::Unchanged(to.to_string()));
        }
        let to_path = self.models_root().join(to);
        // Plain `exists()`: unlike the rename, there is no "it is the same
        // directory" case to allow through. On a case-insensitive filesystem a
        // name that differs from an existing model only in case *is* taken, and
        // saying so is the honest answer.
        if to_path.exists() {
            return Err(ManagerError::AlreadyExists(to.to_string()));
        }

        match self.fill_duplicate(from, &from_path, to, &to_path, weights) {
            Ok(copied_weights) => Ok(DuplicateOutcome {
                path: to_path,
                copied_weights,
            }),
            Err(err) => {
                let _ = fs::remove_dir_all(&to_path);
                Err(err)
            }
        }
    }

    /// Everything `duplicate_model` writes, in one place so its failure path is
    /// one `remove_dir_all` on a directory nothing else has touched.
    fn fill_duplicate(
        &self,
        from: &str,
        from_path: &Path,
        to: &str,
        to_path: &Path,
        weights: WeightsToCopy,
    ) -> Result<Vec<String>, ManagerError> {
        let config_source = from_path.join("config_file");
        if !config_source.is_file() {
            // A directory with no config is not a model, and copying it would
            // produce something the list cannot even show.
            return Err(ManagerError::NotFound(from.to_string()));
        }
        fs::create_dir_all(to_path)?;
        fs::copy(&config_source, to_path.join("config_file"))?;
        self.retarget_config(from, from_path, to, to_path)?;

        // Created either way: a model directory has a `pretrained_weights/`,
        // and the copy is a model.
        let target_dir = to_path.join("pretrained_weights");
        fs::create_dir_all(&target_dir)?;

        let mut copied = Vec::new();
        if weights == WeightsToCopy::Newest
            && let Some(source) = weights_in_hand(&from_path.join("pretrained_weights"))
        {
            let name = source
                .file_name()
                .ok_or_else(|| io::Error::other("checkpoint has no file name"))?;
            let written = target_dir.join(name);
            // A real copy, not a link into the source: the copy has to survive
            // the original being deleted, and its disk cost has to be visible.
            fs::copy(&source, &written)?;
            copied.push(name.to_string_lossy().to_string());
            // And `latest.ckpt` inside the copy designates the copy's newest
            // run, exactly as it does for a model that trained here.
            let latest = point_latest_at(&written)?;
            if latest != written {
                copied.push(LATEST_CHECKPOINT_NAME.to_string());
            }
        }
        Ok(copied)
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

/// Whether a file in `pretrained_weights/` is weights someone can load.
///
/// The extension is the whole test, and it has to be: every training run drops
/// a `*_metrics.jsonl` next to its checkpoint, so a listing that takes any file
/// offers that JSONL as loadable weights and counts it in "N checkpoints".
/// Observed end-to-end — a 40-step run turned one checkpoint into two.
///
/// Dot-files stay excluded on top of the extension: macOS writes AppleDouble
/// siblings (`._latest.ckpt`) whose extension passes the test but whose bytes
/// are not a checkpoint.
fn is_checkpoint_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    !name.starts_with('.') && path.extension().is_some_and(|ext| ext == "ckpt")
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

pub fn new_run_checkpoint_path(model_name: &str, at: SystemTime) -> io::Result<PathBuf> {
    Storage::default().new_run_checkpoint_path(model_name, at)
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

/// `(sample_count, width, height, channels, value_bytes)` from a `.batraw`
/// header.
///
/// The header only: CIFAR-10 is 195 MiB on disk, and reading it whole to answer
/// "how many chunks does this become on the GPU?" would make the answer cost
/// more than the run it describes.
///
/// All three magics are accepted. `BATRAW1` differs from `BATRAW2` in the
/// *range* of its payload, not in its shape; `BATRAW3` differs in its **width**
/// — one byte per value instead of four — which is why the width is part of the
/// answer rather than assumed. Left out, this function returned `None` for every
/// file the current converters write, and the Resources page then reported no
/// streamed dataset at all for them.
pub fn read_batraw_header(path: &Path) -> Option<(u64, u32, u32, u32, u64)> {
    use std::io::Read;
    let mut file = fs::File::open(path).ok()?;
    let mut header = [0u8; 24];
    file.read_exact(&mut header).ok()?;
    let value_bytes: u64 = match &header[..8] {
        b"BATRAW3\0" => 1,
        b"BATRAW2\0" | b"BATRAW1\0" => 4,
        _ => return None,
    };
    let word = |i: usize| {
        u32::from_le_bytes([
            header[8 + i * 4],
            header[9 + i * 4],
            header[10 + i * 4],
            header[11 + i * 4],
        ])
    };
    Some((word(0) as u64, word(1), word(2), word(3), value_bytes))
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

    /// A `.batraw` of `count` samples of the given geometry, under `magic`.
    /// Only the header is ever read back, but the payload is written full size
    /// so a reader that checked the length would still be satisfied.
    fn write_batraw(path: &Path, magic: &[u8; 8], count: u32, geometry: (u32, u32, u32)) {
        let value_bytes = if magic == b"BATRAW3\0" { 1 } else { 4 };
        let mut bytes = magic.to_vec();
        for word in [count, geometry.0, geometry.1, geometry.2] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let values = (count * geometry.0 * geometry.1 * geometry.2) as usize;
        bytes.resize(bytes.len() + values * value_bytes, 0);
        fs::write(path, bytes).expect("writing a test dataset should work");
    }

    /// `BATRAW3` is what every converter in `tools/` writes today, and this
    /// reader refused it — so `dataset_spec` answered `None` for every current
    /// dataset and the Resources page reported no streamed post at all where
    /// there were gigabytes of one.
    ///
    /// The second half is the reason the payload width is returned rather than
    /// assumed: the same 32×32×3 images are a quarter of the bytes in 8-bit,
    /// and that number is what the residency budget is decided on.
    #[test]
    fn the_header_reader_accepts_every_magic_and_reports_the_payload_width() {
        let temp = TempRoot::new("batraw-magics");
        let geometry = (32, 32, 3);
        let expected_f32 = 32 * 32 * 3 * 4;

        for (magic, value_bytes) in [
            (b"BATRAW3\0", 1u64),
            (b"BATRAW2\0", 4),
            (b"BATRAW1\0", 4),
        ] {
            let path = temp.path().join(format!(
                "{}.batraw",
                std::str::from_utf8(&magic[..7]).unwrap()
            ));
            write_batraw(&path, magic, 7, geometry);

            let header = read_batraw_header(&path).unwrap_or_else(|| {
                panic!(
                    "{} was refused — every dataset in this format would report no \
                     size at all",
                    std::str::from_utf8(&magic[..7]).unwrap()
                )
            });
            assert_eq!((header.0, header.1, header.2, header.3), (7, 32, 32, 3));
            assert_eq!(header.4, value_bytes);

            let spec = Storage::at(temp.path())
                .dataset_spec(&path.to_string_lossy())
                .expect("a readable header must yield a spec");
            assert_eq!(spec.sample_count, 7);
            assert_eq!(
                spec.sample_bytes,
                expected_f32 / (4 / value_bytes),
                "an 8-bit sample must not be accounted as an f32 one"
            );
        }

        let bogus = temp.path().join("not-a-dataset.batraw");
        fs::write(&bogus, b"BATRAWX\0            ").expect("write");
        assert!(
            read_batraw_header(&bogus).is_none(),
            "an unknown magic must stay refused"
        );
    }

    fn seed_model(storage: &Storage, name: &str) {
        storage
            .write_model_config(name, &a_config())
            .expect("seeding a model should work");
    }

    fn checkpoint_names(entry: &SavedModelEntry) -> Vec<String> {
        entry
            .checkpoints
            .iter()
            .map(|ckpt| ckpt.name.clone())
            .collect()
    }

    /// A `SystemTime` for a given wall-clock instant is not something `std`
    /// offers, so tests that need *distinct, ordered* instants build them by
    /// offsetting a fixed epoch — which is all the naming rule needs.
    fn at_minutes(minutes: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(minutes * 60)
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
        assert_eq!(checkpoint_names(&models[0]), vec!["latest.ckpt".to_string()]);
    }

    /// Training writes `latest_metrics.jsonl` beside `latest.ckpt`, so both
    /// listings see it on every trained model. Neither may call it weights: the
    /// list would say "2 checkpoints" and the weight selector would offer the
    /// JSONL as something to load.
    #[test]
    fn the_metrics_journal_beside_the_weights_is_not_a_checkpoint() {
        let temp = TempRoot::new("metrics-sibling");
        let storage = temp.storage();
        seed_model(&storage, "alpha");
        let weights = storage.model_weights_dir("alpha").expect("weights dir");
        fs::write(weights.join("latest.ckpt"), b"weights").expect("checkpoint write");
        fs::write(weights.join("latest_metrics.jsonl"), b"{}\n").expect("metrics write");
        fs::write(weights.join("._latest.ckpt"), b"apple double").expect("sibling write");

        let models = storage.list_models().expect("listing should work");
        assert_eq!(checkpoint_names(&models[0]), vec!["latest.ckpt".to_string()]);

        let offered = storage
            .list_model_checkpoints("alpha")
            .expect("checkpoint listing")
            .into_iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>();
        assert_eq!(offered, vec!["latest.ckpt".to_string()]);
    }

    // --- Dated checkpoints ---

    /// The naming rule's whole purpose: a directory listing sorted by name is
    /// the runs in the order they happened. Checked on names alone — no
    /// filesystem, no mtime — because that is the property external tools
    /// (`ls`, a glob in a shell script) get for free.
    #[test]
    fn dated_run_names_sort_by_name_in_the_order_the_runs_happened() {
        let dir = Path::new("/nonexistent-so-nothing-is-taken");
        let minute = 60;
        let hour = 60 * minute;
        let day = 24 * hour;
        let instants = [
            at_minutes(0),
            at_minutes(59),
            at_minutes(hour),
            at_minutes(9 * hour),
            at_minutes(day),
            at_minutes(40 * day),
            at_minutes(400 * day),
        ];
        let names: Vec<String> = instants
            .iter()
            .map(|at| {
                new_run_checkpoint_path_in(dir, *at)
                    .file_name()
                    .expect("a file name")
                    .to_string_lossy()
                    .to_string()
            })
            .collect();

        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(
            names, sorted,
            "dated names must sort chronologically: {names:?}"
        );
        assert!(
            names.iter().all(|name| name.starts_with("run-")
                && name.ends_with(".ckpt")),
            "unexpected shape: {names:?}"
        );
    }

    /// Two runs in the same minute is not a hypothetical — it is two 20-step
    /// smoke tests in a row. The second must not overwrite the first, and its
    /// name must still sort after it.
    #[test]
    fn two_runs_in_the_same_minute_get_two_files() {
        let temp = TempRoot::new("same-minute");
        let storage = temp.storage();
        seed_model(&storage, "alpha");
        let at = at_minutes(12_345);

        let first = storage
            .new_run_checkpoint_path("alpha", at)
            .expect("run path");
        fs::write(&first, b"first").expect("write");
        let second = storage
            .new_run_checkpoint_path("alpha", at)
            .expect("run path");
        fs::write(&second, b"second").expect("write");

        assert_ne!(first, second, "the second run reused the first's name");
        assert_eq!(fs::read(&first).expect("first still there"), b"first");
        assert!(
            second.file_name().unwrap().to_string_lossy()
                > first.file_name().unwrap().to_string_lossy(),
            "{second:?} should sort after {first:?}"
        );
    }

    /// `latest.ckpt` still exists, still holds the newest weights, and costs no
    /// second copy on disk: it is a hard link, so the two names share one inode.
    #[test]
    fn latest_holds_the_newest_bytes_without_a_second_copy() {
        let temp = TempRoot::new("latest-link");
        let storage = temp.storage();
        seed_model(&storage, "alpha");

        let first = storage
            .new_run_checkpoint_path("alpha", at_minutes(1))
            .expect("run path");
        fs::write(&first, b"weights of the first run").expect("write");
        let latest = point_latest_at(&first).expect("pointing latest");
        assert_eq!(latest.file_name().unwrap(), "latest.ckpt");
        assert_eq!(
            fs::read(&latest).expect("latest readable"),
            b"weights of the first run"
        );

        let second = storage
            .new_run_checkpoint_path("alpha", at_minutes(2))
            .expect("run path");
        fs::write(&second, b"weights of the second run").expect("write");
        point_latest_at(&second).expect("pointing latest again");

        assert_eq!(
            fs::read(&latest).expect("latest readable"),
            b"weights of the second run",
            "latest.ckpt did not follow the newest run"
        );
        assert_eq!(
            fs::read(&first).expect("the first run's file survived"),
            b"weights of the first run",
            "the previous run's weights were clobbered — that is the whole bug"
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let inode = |path: &Path| fs::metadata(path).expect("metadata").ino();
            assert_eq!(
                inode(&latest),
                inode(&second),
                "latest.ckpt should be a link to the newest run, not a copy"
            );
            assert_ne!(inode(&latest), inode(&first));
        }

        // The scratch name used while linking must never survive a call, and
        // must never be listable as weights.
        assert!(!temp.path().join(".latest.ckpt.tmp").exists());
        let offered = storage
            .list_model_checkpoints("alpha")
            .expect("listing")
            .into_iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>();
        assert!(!offered.iter().any(|name| name.starts_with('.')));
    }

    /// The selector opens on the newest, so the listing has to be ordered by
    /// *time*, not by name — a legacy `night_run.ckpt` written yesterday sits
    /// below today's run, whatever the alphabet says.
    #[test]
    fn checkpoints_are_listed_newest_first_and_carry_their_date_and_size() {
        let temp = TempRoot::new("newest-first");
        let storage = temp.storage();
        seed_model(&storage, "alpha");
        let weights = storage.model_weights_dir("alpha").expect("weights dir");

        // Written oldest → newest, with a pause so the filesystem records
        // distinct modification times.
        for name in ["night_run.ckpt", "zzz_old.ckpt", "run-2026-08-08_1041.ckpt"] {
            fs::write(weights.join(name), vec![0u8; 2048]).expect("write");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        let listed = storage.list_model_checkpoints("alpha").expect("listing");
        let names: Vec<&str> = listed.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["run-2026-08-08_1041.ckpt", "zzz_old.ckpt", "night_run.ckpt"],
            "newest must come first"
        );
        assert!(listed[0].modified.is_some(), "no modification time read");
        assert_eq!(listed[0].size_bytes, 2048);
        assert!(
            listed[0].detail().contains("2 kB"),
            "the row should carry its size: {}",
            listed[0].detail()
        );

        // The model list is fed by the same order, so both screens agree.
        let models = storage.list_models().expect("model listing");
        assert_eq!(checkpoint_names(&models[0])[0], "run-2026-08-08_1041.ckpt");
    }

    /// Checkpoints written before this convention existed — `latest.ckpt`,
    /// `night_run.ckpt`, `one.ckpt` — are ordinary checkpoints. Nothing about
    /// the dated naming may hide them or require a migration.
    #[test]
    fn checkpoints_that_predate_the_naming_convention_are_still_offered() {
        let temp = TempRoot::new("legacy-names");
        let storage = temp.storage();
        seed_model(&storage, "alpha");
        let weights = storage.model_weights_dir("alpha").expect("weights dir");
        for name in ["latest.ckpt", "night_run.ckpt", "night2.ckpt"] {
            fs::write(weights.join(name), b"legacy weights").expect("write");
        }

        let mut offered = storage
            .list_model_checkpoints("alpha")
            .expect("listing")
            .into_iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>();
        offered.sort();
        assert_eq!(offered, vec!["latest.ckpt", "night2.ckpt", "night_run.ckpt"]);
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
            ema_decay: None,
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

    // --- Duplicate ---

    /// A model that has trained twice: two dated runs, `latest.ckpt` linked onto
    /// the newer, and a config that records where its weights are — the shape
    /// the foundation model to be fine-tuned actually has on disk.
    fn seed_trained_model(storage: &Storage, name: &str) -> PathBuf {
        let weights = storage.model_weights_dir(name).expect("weights dir");
        fs::write(weights.join("run-2026-08-01_0900.ckpt"), b"the older run").expect("write");
        std::thread::sleep(std::time::Duration::from_millis(20));
        let newest = weights.join("run-2026-08-08_1120.ckpt");
        fs::write(&newest, b"the weights that matter").expect("write");
        point_latest_at(&newest).expect("latest link");

        let mut config = a_config();
        config.inference.checkpoint = Some(
            weights
                .join(LATEST_CHECKPOINT_NAME)
                .to_string_lossy()
                .to_string(),
        );
        storage.write_model_config(name, &config).expect("seed");
        newest
    }

    /// Everything about a file that must not move: its bytes, its size and when
    /// it was last written.
    fn fingerprint(path: &Path) -> (Vec<u8>, u64, Option<SystemTime>) {
        let meta = fs::metadata(path).expect("metadata");
        (
            fs::read(path).expect("read"),
            meta.len(),
            meta.modified().ok(),
        )
    }

    fn checkpoint_names_in(storage: &Storage, model: &str) -> Vec<String> {
        let mut names = storage
            .list_model_checkpoints(model)
            .expect("listing")
            .into_iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    /// The property the whole feature exists for: after duplicating, the
    /// original is exactly what it was. Fine-tuning the copy is only safe
    /// because nothing here writes to the source.
    #[test]
    fn duplicating_a_model_leaves_the_original_untouched() {
        let temp = TempRoot::new("duplicate-intact");
        let storage = temp.storage();
        let newest = seed_trained_model(&storage, "Foundation");
        let source_dir = storage.models_root().join("Foundation");
        let before_config = fingerprint(&source_dir.join("config_file"));
        let before_weights = fingerprint(&newest);
        let before_names = checkpoint_names_in(&storage, "Foundation");

        storage
            .duplicate_model("Foundation", "Elephants", WeightsToCopy::Newest)
            .expect("duplicate should succeed");

        assert_eq!(
            fingerprint(&source_dir.join("config_file")),
            before_config,
            "the original's config_file was rewritten"
        );
        assert_eq!(
            fingerprint(&newest),
            before_weights,
            "the original's weights were touched"
        );
        assert_eq!(
            checkpoint_names_in(&storage, "Foundation"),
            before_names,
            "the original's pretrained_weights/ gained or lost a file"
        );
    }

    /// The copy is a model in its own right: its own directory, its own
    /// `config_file` naming *it*, and every checkpoint path rebased onto it. A
    /// config still pointing at the source would send the copy's first run back
    /// into the directory this operation exists to protect.
    #[test]
    fn a_duplicate_is_a_model_of_its_own_and_its_config_says_so() {
        let temp = TempRoot::new("duplicate-config");
        let storage = temp.storage();
        seed_trained_model(&storage, "Foundation");

        let outcome = storage
            .duplicate_model("Foundation", "Elephants", WeightsToCopy::Newest)
            .expect("duplicate should succeed");

        assert_eq!(outcome.path, storage.models_root().join("Elephants"));
        assert!(outcome.path.is_dir());
        let config = storage
            .load_model_config_for_model("Elephants")
            .expect("the copy's config should load");
        assert_eq!(config.model_name.as_deref(), Some("Elephants"));
        let recorded = config.inference.checkpoint.expect("checkpoint recorded");
        assert!(
            recorded.contains("Elephants") && !recorded.contains("Foundation"),
            "the copy's config still points into the original: {recorded}"
        );
        assert!(
            Path::new(&recorded).is_file(),
            "the path the copy's config records does not exist: {recorded}"
        );

        // And the list shows two models, both loadable.
        let listed: Vec<String> = storage
            .list_models()
            .expect("listing")
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        assert_eq!(listed, vec!["Elephants", "Foundation"]);
    }

    /// What travels is the newest run — under its own dated name — and
    /// `latest.ckpt` re-linked onto it *inside the copy*. Not the history: a
    /// model with dozens of 14 MB runs would cost half a gigabyte per copy.
    #[test]
    fn the_newest_run_travels_with_the_copy_and_latest_points_at_it_there() {
        let temp = TempRoot::new("duplicate-weights");
        let storage = temp.storage();
        let newest = seed_trained_model(&storage, "Foundation");

        let outcome = storage
            .duplicate_model("Foundation", "Elephants", WeightsToCopy::Newest)
            .expect("duplicate should succeed");

        assert_eq!(
            outcome.copied_weights,
            vec!["run-2026-08-08_1120.ckpt".to_string(), "latest.ckpt".to_string()],
            "the outcome must name exactly what was written"
        );
        assert_eq!(
            checkpoint_names_in(&storage, "Elephants"),
            vec!["latest.ckpt", "run-2026-08-08_1120.ckpt"],
            "the older run must not have been carried along"
        );

        let copied = outcome.path.join("pretrained_weights");
        let dated = copied.join("run-2026-08-08_1120.ckpt");
        let latest = copied.join(LATEST_CHECKPOINT_NAME);
        assert_eq!(
            fs::read(&dated).expect("read"),
            fs::read(&newest).expect("read"),
            "the copied weights are not the original's bytes"
        );
        assert_eq!(fs::read(&latest).expect("read"), fs::read(&dated).expect("read"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let inode = |path: &Path| fs::metadata(path).expect("metadata").ino();
            assert_eq!(
                inode(&latest),
                inode(&dated),
                "latest.ckpt in the copy must be a hard link onto the run, not a second copy"
            );
            assert_ne!(
                inode(&dated),
                inode(&newest),
                "the copy is linked into the original — deleting one would surprise the other"
            );
        }
    }

    /// The other half of the choice: same architecture, fresh weights. The copy
    /// gets a `pretrained_weights/`, and it is empty.
    #[test]
    fn a_duplicate_can_be_taken_without_the_weights() {
        let temp = TempRoot::new("duplicate-config-only");
        let storage = temp.storage();
        seed_trained_model(&storage, "Foundation");

        let outcome = storage
            .duplicate_model("Foundation", "Fresh", WeightsToCopy::None)
            .expect("duplicate should succeed");

        assert!(outcome.copied_weights.is_empty());
        assert!(outcome.path.join("pretrained_weights").is_dir());
        assert!(checkpoint_names_in(&storage, "Fresh").is_empty());
        assert_eq!(
            storage
                .load_model_config_for_model("Fresh")
                .expect("config")
                .model_name
                .as_deref(),
            Some("Fresh")
        );
    }

    /// A model that has never trained duplicates as config alone even when
    /// weights were asked for. The outcome says so rather than implying weights
    /// arrived.
    #[test]
    fn asking_for_weights_a_model_does_not_have_copies_the_config_alone() {
        let temp = TempRoot::new("duplicate-untrained");
        let storage = temp.storage();
        seed_model(&storage, "Untrained");

        let outcome = storage
            .duplicate_model("Untrained", "Untrained-copy", WeightsToCopy::Newest)
            .expect("duplicate should succeed");

        assert!(outcome.copied_weights.is_empty());
        assert!(checkpoint_names_in(&storage, "Untrained-copy").is_empty());
    }

    /// A checkpoint from before the dated convention — a bare `latest.ckpt` with
    /// no dated sibling — travels under the only name it has.
    #[test]
    fn a_lone_latest_travels_under_its_own_name() {
        let temp = TempRoot::new("duplicate-legacy");
        let storage = temp.storage();
        seed_model(&storage, "Legacy");
        let weights = storage.model_weights_dir("Legacy").expect("weights dir");
        fs::write(weights.join(LATEST_CHECKPOINT_NAME), b"legacy weights").expect("write");

        let outcome = storage
            .duplicate_model("Legacy", "Legacy-copy", WeightsToCopy::Newest)
            .expect("duplicate should succeed");

        assert_eq!(outcome.copied_weights, vec!["latest.ckpt".to_string()]);
        assert_eq!(checkpoint_names_in(&storage, "Legacy-copy"), vec!["latest.ckpt"]);
        assert_eq!(
            fs::read(outcome.path.join("pretrained_weights").join(LATEST_CHECKPOINT_NAME))
                .expect("read"),
            b"legacy weights"
        );
    }

    #[test]
    fn duplicating_onto_a_name_that_is_taken_changes_nothing() {
        let temp = TempRoot::new("duplicate-collision");
        let storage = temp.storage();
        seed_trained_model(&storage, "Foundation");
        seed_model(&storage, "Elephants");
        let victim = fingerprint(&storage.models_root().join("Elephants").join("config_file"));

        assert!(matches!(
            storage.duplicate_model("Foundation", "Elephants", WeightsToCopy::Newest),
            Err(ManagerError::AlreadyExists(_))
        ));
        assert!(matches!(
            storage.duplicate_model("Foundation", "Foundation", WeightsToCopy::Newest),
            Err(ManagerError::Unchanged(_))
        ));
        assert!(matches!(
            storage.duplicate_model("ghost", "Elephants-2", WeightsToCopy::Newest),
            Err(ManagerError::NotFound(_))
        ));

        assert_eq!(
            fingerprint(&storage.models_root().join("Elephants").join("config_file")),
            victim,
            "a refused duplicate overwrote the model already holding the name"
        );
        assert!(checkpoint_names_in(&storage, "Elephants").is_empty());
        assert!(!storage.models_root().join("Elephants-2").exists());
    }

    /// The same guard the delete has: a name that is not a plain child of
    /// `Models/` cannot even be spelled, and a refusal writes nothing anywhere.
    #[test]
    fn a_pathological_duplicate_name_is_refused_and_writes_nothing() {
        let temp = TempRoot::new("duplicate-guard");
        let storage = temp.storage();
        seed_trained_model(&storage, "Foundation");
        let bystander = temp.path().join("precious");
        fs::create_dir_all(&bystander).expect("bystander");
        fs::write(bystander.join("keep-me"), b"x").expect("bystander file");

        for pathological in [
            "../evil",
            "..",
            "../precious",
            "/etc",
            "a/b",
            "",
            ".",
            ".hidden",
            "Models/Foundation",
        ] {
            let result = storage.duplicate_model("Foundation", pathological, WeightsToCopy::Newest);
            assert!(
                result.is_err(),
                "duplicate_model(.., {pathological:?}) was accepted — it must not be"
            );
        }

        assert_eq!(
            fs::read(bystander.join("keep-me")).expect("read"),
            b"x",
            "a refused duplicate wrote outside Models/"
        );
        assert!(!temp.path().join("evil").exists());
        let listed: Vec<String> = storage
            .list_models()
            .expect("listing")
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        assert_eq!(listed, vec!["Foundation"], "a refused duplicate left a model behind");
    }

    /// The form opens on this name, so it has to be free — and it has to stay
    /// inside the length a model name is allowed.
    #[test]
    fn a_suggested_copy_name_is_free_and_short_enough() {
        let temp = TempRoot::new("duplicate-suggest");
        let storage = temp.storage();
        seed_model(&storage, "Foundation");
        assert_eq!(storage.suggest_copy_name("Foundation"), "Foundation-copy");

        seed_model(&storage, "Foundation-copy");
        assert_eq!(storage.suggest_copy_name("Foundation"), "Foundation-copy-2");
        seed_model(&storage, "Foundation-copy-2");
        assert_eq!(storage.suggest_copy_name("Foundation"), "Foundation-copy-3");

        let long = "x".repeat(MAX_MODEL_NAME_LEN);
        let suggested = storage.suggest_copy_name(&long);
        assert!(suggested.ends_with("-copy"));
        assert!(
            validate_model_name(&suggested).is_ok(),
            "the suggested name is not a usable model name: {suggested}"
        );
    }
}
