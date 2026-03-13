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
    let bytes = serde_json::to_vec_pretty(config).map_err(serde_to_io)?;
    fs::write(&path, bytes)?;
    Ok(path)
}

pub fn load_model_config(path: &Path) -> io::Result<ModelConfig> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(serde_to_io)
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
