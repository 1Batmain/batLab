use super::app::ModelConfig;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SavedModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub input_size: (u32, u32, u32),
    pub layer_count: usize,
}

#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    pub name: String,
    pub path: String,
}

fn serde_to_io(err: serde_json::Error) -> io::Error {
    io::Error::other(err)
}

pub fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bat_building crate should live under workspace root")
        .to_path_buf()
}

pub fn saved_models_dir() -> io::Result<PathBuf> {
    let dir = project_root().join("saved_models");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn datasets_dir() -> io::Result<PathBuf> {
    let dir = project_root().join("datasets");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn save_model_config(config: &ModelConfig) -> io::Result<PathBuf> {
    let name = next_model_name()?;
    save_model_config_named(config, &name)
}

/// Returns the next unused default model name stem, e.g. "model-003".
pub fn next_model_name() -> io::Result<String> {
    let dir = saved_models_dir()?;
    let mut next_index = 1usize;
    loop {
        let name = format!("model-{next_index:03}");
        let path = dir.join(format!("{name}.json"));
        if !path.exists() {
            return Ok(name);
        }
        next_index += 1;
    }
}

/// Saves a model config with an explicit file name stem (no extension required).
/// Overwrites any existing file with the same name.
pub fn save_model_config_named(config: &ModelConfig, name: &str) -> io::Result<PathBuf> {
    let dir = saved_models_dir()?;
    let path = dir.join(format!("{name}.json"));
    let mut persisted = config.clone();
    persisted.model_name = Some(name.to_string());
    let bytes = serde_json::to_vec_pretty(&persisted).map_err(serde_to_io)?;
    fs::write(&path, bytes)?;
    Ok(path)
}

pub fn load_model_config(path: &Path) -> io::Result<ModelConfig> {
    let bytes = fs::read(path)?;
    let mut config: ModelConfig = serde_json::from_slice(&bytes).map_err(serde_to_io)?;
    if config.model_name.is_none() {
        config.model_name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(|stem| stem.to_string());
    }
    Ok(config)
}

pub fn checkpoint_path_for_model_name(name: &str) -> io::Result<PathBuf> {
    Ok(saved_models_dir()?.join(format!("{name}.ckpt")))
}

pub fn models_dir() -> io::Result<PathBuf> {
    let dir = project_root().join("Models");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn model_dir(model_name: &str) -> io::Result<PathBuf> {
    let dir = models_dir()?.join(model_name);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn model_weights_dir(model_name: &str) -> io::Result<PathBuf> {
    let dir = model_dir(model_name)?.join("pretrained_weights");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn default_model_checkpoint_path(model_name: &str) -> io::Result<PathBuf> {
    Ok(model_weights_dir(model_name)?.join("latest.ckpt"))
}

pub fn model_config_path(model_name: &str) -> io::Result<PathBuf> {
    Ok(model_dir(model_name)?.join("config_file"))
}

pub fn write_model_template_config(model_name: &str, config: &ModelConfig) -> io::Result<PathBuf> {
    let path = model_config_path(model_name)?;
    let mut persisted = config.clone();
    persisted.model_name = Some(model_name.to_string());
    let bytes = serde_json::to_vec_pretty(&persisted).map_err(serde_to_io)?;
    fs::write(&path, bytes)?;
    Ok(path)
}

pub fn list_model_checkpoints(model_name: &str) -> io::Result<Vec<CheckpointEntry>> {
    let dir = model_weights_dir(model_name)?;
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

pub fn list_saved_models() -> io::Result<Vec<SavedModelEntry>> {
    let dir = saved_models_dir()?;
    let mut models = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let config = match load_model_config(&path) {
            Ok(config) => config,
            Err(_) => continue,
        };
        models.push(SavedModelEntry {
            name: path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("model")
                .to_string(),
            path,
            input_size: config.input_size,
            layer_count: config.layers.len(),
        });
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

pub fn list_datasets() -> io::Result<Vec<String>> {
    let mut datasets = Vec::new();
    let dir = datasets_dir()?;
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

    let legacy_cifar = project_root().join("cifar");
    if legacy_cifar.exists() {
        let legacy = legacy_cifar.display().to_string();
        if !datasets.iter().any(|path| path == &legacy) {
            datasets.push(legacy);
        }
    }

    datasets.sort();
    Ok(datasets)
}
