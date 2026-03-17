//! File purpose: Implements storage behavior for the terminal user interface flow.

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

pub fn datasets_dir() -> io::Result<PathBuf> {
    let dir = project_root().join("datasets");
    fs::create_dir_all(&dir)?;
    Ok(dir)
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

pub fn next_model_name() -> io::Result<String> {
    let root = models_dir()?;
    let mut next_index = 1usize;
    loop {
        let name = format!("model-{next_index:03}");
        if !root.join(&name).exists() {
            return Ok(name);
        }
        next_index += 1;
    }
}

pub fn write_model_config(model_name: &str, config: &ModelConfig) -> io::Result<PathBuf> {
    let path = model_config_path(model_name)?;
    let mut persisted = config.clone();
    persisted.model_name = Some(model_name.to_string());
    let bytes = serde_json::to_vec_pretty(&persisted).map_err(serde_to_io)?;
    fs::write(&path, bytes)?;
    Ok(path)
}

pub fn load_model_config(path: &Path) -> io::Result<ModelConfig> {
    let bytes = fs::read(path)?;
    let mut config: ModelConfig = serde_json::from_slice(&bytes).map_err(serde_to_io)?;
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
    let path = model_config_path(model_name)?;
    load_model_config(&path)
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

pub fn list_models() -> io::Result<Vec<SavedModelEntry>> {
    let mut models = Vec::new();
    let root = models_dir()?;
    for entry in fs::read_dir(root)? {
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
