//! File purpose: Application entry point that orchestrates training, inference, and TUI workflows.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use bat_building::tui::{
    self, ActivationMethod, LayerDraft, ModelConfig, MonitorOutcome, PaddingMode, RunMode,
    TrainingConfig,
};
use bat_building::{
    ActivationMethod as PActivation, ActivationType, ConvolutionType, DiffusionTask, Dim3,
    FullyConnectedType, GpuContext, GpuDataset, GroupNormType, LayerTypes, LinearNoiseSchedule,
    LossMethod as PLoss, Model, PaddingMode as PPadding, Trainer, UpsampleConvType,
    model::Training,
};
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};

/// Magic header for the raw binary dataset format produced by the pre-processing scripts.
/// Format: magic(8) | count(u32le) | width(u32le) | height(u32le) | channels(u32le) | f32 data…
const RAW_DATASET_MAGIC: &[u8; 8] = b"BATRAW1\0";

const DIFFUSION_SCHEDULE_STEPS: usize = 256;
const DIFFUSION_BETA_START: f32 = 1e-4;
const DIFFUSION_BETA_END: f32 = 2e-2;
const LOSS_REPORT_INTERVAL_STEPS: usize = 25;
const INFERENCE_RUNTIME_LR: f32 = 0.01;
const INFERENCE_RUNTIME_BATCH_SIZE: u32 = 1;

#[derive(Debug, Clone)]
struct ImageSample {
    target: Vec<f32>,
}

fn main() {
    let config = match tui::run() {
        Ok(c) => c,
        Err(_) => return,
    };

    run_execution_loop(config);
}

fn run_execution_loop(mut config: ModelConfig) {
    loop {
        if let Err(err) = normalize_config_for_models_layout(&mut config) {
            eprintln!("failed to prepare model persistence: {err}");
            break;
        }

        let (tx, rx) = std::sync::mpsc::channel::<tui::TrainingEvent>();
        let (control_tx, control_rx) = std::sync::mpsc::channel::<tui::TrainingControlCommand>();
        let is_training_run = matches!(&config.run.mode, RunMode::Train(_));
        let config_clone = config.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(async {
                let run_result = match config_clone.run.mode.clone() {
                    RunMode::Train(train_cfg) => {
                        run_training(config_clone, train_cfg, &tx, control_rx).await
                    }
                    RunMode::Infer => run_inference(config_clone, &tx).await,
                };
                if let Err(message) = run_result {
                    let _ = tx.send(tui::TrainingEvent::Error { message });
                    let _ = tx.send(tui::TrainingEvent::Done);
                }
            });
        });

        let maybe_control_tx = if is_training_run {
            Some(control_tx)
        } else {
            None
        };
        match tui::run_monitor(config.clone(), rx, maybe_control_tx) {
            Ok(MonitorOutcome::Restart(new_config)) => {
                config = new_config;
            }
            _ => break,
        }
    }
}

fn normalize_config_for_models_layout(config: &mut ModelConfig) -> Result<(), String> {
    let model_name = match config.model_name.clone() {
        Some(name) => name,
        None => {
            let generated = tui::storage::next_model_name()
                .map_err(|err| format!("failed to allocate model name: {err}"))?;
            config.model_name = Some(generated.clone());
            generated
        }
    };

    if let RunMode::Train(train) = &mut config.run.mode
        && train.checkpoint_path.is_none()
    {
        train.checkpoint_path = Some(
            tui::storage::default_model_checkpoint_path(&model_name)
                .map_err(|err| format!("failed to resolve checkpoint path: {err}"))?
                .to_string_lossy()
                .to_string(),
        );
    }

    tui::storage::write_model_config(&model_name, config)
        .map_err(|err| format!("failed to write model config for '{model_name}': {err}"))?;
    Ok(())
}

async fn build_execution_model(
    config: &ModelConfig,
    lr: f32,
    batch_size: u32,
) -> Result<(Arc<GpuContext>, Model<Training>), String> {
    let gpu = Arc::new(GpuContext::new_headless().await);
    let mut model = Model::new_training(gpu.clone(), lr, batch_size, PLoss::MeanSquared).await;
    for draft in &config.layers {
        append_layer(&mut model, draft).map_err(|err| err.to_string())?;
    }
    model.build().map_err(|err| err.to_string())?;
    Ok((gpu, model))
}

async fn run_training(
    config: ModelConfig,
    train_cfg: TrainingConfig,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
    control_rx: Receiver<tui::TrainingControlCommand>,
) -> Result<(), String> {
    let (gpu, mut model) =
        build_execution_model(&config, train_cfg.lr, train_cfg.batch_size).await?;
    let checkpoint_path = match train_cfg.checkpoint_path.as_deref() {
        Some(path) => Some(PathBuf::from(path)),
        None => config
            .model_name
            .as_deref()
            .map(tui::storage::default_model_checkpoint_path)
            .transpose()
            .map_err(|err| format!("failed to resolve checkpoint path: {err}"))?,
    };
    if let Some(path) = checkpoint_path.as_ref() {
        if train_cfg.load_checkpoint {
            if !path.exists() {
                return Err(format!(
                    "selected checkpoint does not exist: {}",
                    path.display()
                ));
            }
            model
                .load_checkpoint(path)
                .map_err(|err| format!("failed to load checkpoint {}: {err}", path.display()))?;
        } else if train_cfg.checkpoint_path.is_none() && path.exists() {
            // Backward compatibility for legacy configs without explicit load mode.
            model
                .load_checkpoint(path)
                .map_err(|err| format!("failed to load checkpoint {}: {err}", path.display()))?;
        }
    }

    let input_dims = model
        .input_dim()
        .ok_or_else(|| "model has no input dimensions".to_string())?;
    let output_dims = model
        .output_dim()
        .ok_or_else(|| "model has no output dimensions".to_string())?;
    let input_size = (input_dims.x, input_dims.y, input_dims.z);
    let output_size = (output_dims.x, output_dims.y, output_dims.z);

    let schedule = LinearNoiseSchedule::new_linear(
        DIFFUSION_SCHEDULE_STEPS,
        DIFFUSION_BETA_START,
        DIFFUSION_BETA_END,
    );
    let mut trainer = Trainer::new(DiffusionTask::new(schedule));
    trainer
        .configure_for_model(&model)
        .map_err(|err| format!("failed to configure diffusion task: {err}"))?;
    let diffusion = trainer.task().schedule().clone();

    let dataset = load_dataset(&train_cfg.dataset_path, output_size)?;
    let sample_len = (output_size.0 * output_size.1 * output_size.2) as usize;
    let gpu_samples: Vec<Vec<f32>> = dataset.into_iter().map(|sample| sample.target).collect();
    let mut gpu_dataset = GpuDataset::from_samples(gpu.as_ref(), gpu_samples, sample_len)
        .map_err(|err| format!("failed to upload dataset to GPU: {err}"))?;
    let limits = gpu.device().limits();
    let estimated_training_bytes = model
        .estimated_gpu_bytes()
        .saturating_add(gpu_dataset.gpu_buffer_bytes())
        .saturating_add(trainer.task().estimated_prepare_gpu_bytes(output_dims));
    let _ = tx.send(tui::TrainingEvent::ResourceReport {
        max_buffer_bytes: limits.max_buffer_size,
        max_storage_binding_bytes: limits.max_storage_buffer_binding_size as u64,
        estimated_training_bytes,
    });
    let sample_dir = prepare_sample_dir(&train_cfg.dataset_path)?;
    let mut current_lr = train_cfg.lr;
    let mut current_batch_size = train_cfg.batch_size.max(1);
    let mut total_steps = train_cfg.steps.max(1);
    let mut paused = false;
    let _ = tx.send(tui::TrainingEvent::TrainingState {
        paused,
        lr: current_lr,
        batch_size: current_batch_size,
        total_steps,
    });

    if let Some(output_buf) = model.last_output_buffer() {
        let title = format!(
            "Model Output  ({}×{}×{} channels)",
            output_size.0, output_size.1, output_size.2
        );
        tui::register_visualiser_source(
            model.gpu_context(),
            output_buf,
            output_size.0,
            output_size.1,
            output_size.2,
            title,
        );
    } else {
        eprintln!("[visualiser] model has no output buffer yet");
    }

    let mut step = 0usize;
    while step < total_steps {
        while let Ok(command) = control_rx.try_recv() {
            let channel_open = apply_and_publish_training_state(
                command,
                &mut model,
                &mut paused,
                &mut current_lr,
                &mut current_batch_size,
                &mut total_steps,
                checkpoint_path.as_deref(),
                tx,
            );
            if !channel_open {
                return Ok(());
            }
        }
        if paused {
            match control_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(command) => {
                    let channel_open = apply_and_publish_training_state(
                        command,
                        &mut model,
                        &mut paused,
                        &mut current_lr,
                        &mut current_batch_size,
                        &mut total_steps,
                        checkpoint_path.as_deref(),
                        tx,
                    );
                    if !channel_open {
                        return Ok(());
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
            continue;
        }

        let should_report_loss = step % LOSS_REPORT_INTERVAL_STEPS == 0 || step + 1 == total_steps;
        let loss = if should_report_loss {
            trainer
                .task_mut()
                .train_step_report_batch(
                    &mut model,
                    &mut gpu_dataset,
                    step,
                    current_batch_size as usize,
                    (step as u64) << 32,
                )
                .map_err(|err| format!("failed diffusion GPU batch step: {err}"))?
        } else {
            trainer
                .task_mut()
                .train_step_batch(
                    &mut model,
                    &mut gpu_dataset,
                    step,
                    current_batch_size as usize,
                    (step as u64) << 32,
                )
                .map_err(|err| format!("failed diffusion GPU batch step: {err}"))?;
            None
        };

        if tx
            .send(tui::TrainingEvent::Step {
                step,
                loss,
                sample_path: None,
            })
            .is_err()
        {
            return Ok(());
        }

        step += 1;
    }

    let final_step = step.saturating_sub(1);
    let output = sample_diffusion_image(
        &mut model,
        input_size,
        output_size,
        &diffusion,
        final_step as u64,
    );
    let sample_path = save_tensor_as_image(&output, output_size, &sample_dir, final_step)
        .map(|path| path.display().to_string())?;
    let _ = tx.send(tui::TrainingEvent::Step {
        step: final_step,
        loss: None,
        sample_path: Some(sample_path),
    });

    if let Some(path) = checkpoint_path.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create checkpoint directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        model
            .save_checkpoint(path)
            .map_err(|err| format!("failed to save checkpoint {}: {err}", path.display()))?;
    }

    let _ = tx.send(tui::TrainingEvent::Done);
    Ok(())
}

fn apply_and_publish_training_state(
    command: tui::TrainingControlCommand,
    model: &mut Model<Training>,
    paused: &mut bool,
    current_lr: &mut f32,
    current_batch_size: &mut u32,
    total_steps: &mut usize,
    checkpoint_path: Option<&Path>,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
) -> bool {
    apply_training_control_command(
        command,
        model,
        paused,
        current_lr,
        current_batch_size,
        total_steps,
        checkpoint_path,
        tx,
    );
    tx.send(tui::TrainingEvent::TrainingState {
        paused: *paused,
        lr: *current_lr,
        batch_size: *current_batch_size,
        total_steps: *total_steps,
    })
    .is_ok()
}

fn apply_training_control_command(
    command: tui::TrainingControlCommand,
    model: &mut Model<Training>,
    paused: &mut bool,
    current_lr: &mut f32,
    current_batch_size: &mut u32,
    total_steps: &mut usize,
    checkpoint_path: Option<&Path>,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
) {
    match command {
        tui::TrainingControlCommand::SetPaused(next) => {
            *paused = next;
        }
        tui::TrainingControlCommand::SaveCheckpoint => {
            let Some(path) = checkpoint_path else {
                let _ = tx.send(tui::TrainingEvent::SaveStatus {
                    message: "cannot save checkpoint: no checkpoint path configured".to_string(),
                    is_error: true,
                });
                return;
            };
            if let Some(parent) = path.parent()
                && let Err(err) = fs::create_dir_all(parent)
            {
                let _ = tx.send(tui::TrainingEvent::SaveStatus {
                    message: format!(
                        "failed to create checkpoint directory {}: {err}",
                        parent.display()
                    ),
                    is_error: true,
                });
                return;
            }
            match model.save_checkpoint(path) {
                Ok(()) => {
                    let _ = tx.send(tui::TrainingEvent::SaveStatus {
                        message: format!("checkpoint saved → {}", path.display()),
                        is_error: false,
                    });
                }
                Err(err) => {
                    let _ = tx.send(tui::TrainingEvent::SaveStatus {
                        message: format!("failed to save checkpoint {}: {err}", path.display()),
                        is_error: true,
                    });
                }
            }
        }
        tui::TrainingControlCommand::UpdateParams {
            lr,
            batch_size,
            total_steps: new_total_steps,
        } => {
            *current_lr = lr;
            *current_batch_size = batch_size.max(1);
            *total_steps = new_total_steps.max(1);
            model.set_learning_rate(*current_lr);
            model.set_batch_size(*current_batch_size);
        }
    }
}

async fn run_inference(
    config: ModelConfig,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
) -> Result<(), String> {
    let (gpu, mut model) =
        build_execution_model(&config, INFERENCE_RUNTIME_LR, INFERENCE_RUNTIME_BATCH_SIZE).await?;

    let checkpoint_path = resolve_inference_checkpoint_path(&config)?;
    model.load_checkpoint(&checkpoint_path).map_err(|err| {
        format!(
            "failed to load inference checkpoint {}: {err}",
            checkpoint_path.display()
        )
    })?;

    let limits = gpu.device().limits();
    let _ = tx.send(tui::TrainingEvent::ResourceReport {
        max_buffer_bytes: limits.max_buffer_size,
        max_storage_binding_bytes: limits.max_storage_buffer_binding_size as u64,
        estimated_training_bytes: model.estimated_gpu_bytes(),
    });

    let input_dims = model
        .input_dim()
        .ok_or_else(|| "model has no input dimensions".to_string())?;
    let output_dims = model
        .output_dim()
        .ok_or_else(|| "model has no output dimensions".to_string())?;
    let input_size = (input_dims.x, input_dims.y, input_dims.z);
    let output_size = (output_dims.x, output_dims.y, output_dims.z);
    let diffusion = LinearNoiseSchedule::new_linear(
        DIFFUSION_SCHEDULE_STEPS,
        DIFFUSION_BETA_START,
        DIFFUSION_BETA_END,
    );

    let inference = &config.inference;
    let seed = if inference.random_seed {
        random_seed()
    } else {
        inference.seed.unwrap_or(0)
    };

    let total_steps = diffusion
        .len()
        .saturating_mul(inference.denoising_paths.max(1));
    let _ = tx.send(tui::TrainingEvent::InferenceProgress {
        label: "Preparing inference path".to_string(),
        current: 0,
        total: total_steps.max(1),
    });

    let output = sample_diffusion_image_with_controls(
        &mut model,
        input_size,
        output_size,
        &diffusion,
        seed,
        inference.denoising_paths,
        inference.denoise_magnitude,
        |current, total| {
            let _ = tx.send(tui::TrainingEvent::InferenceProgress {
                label: "Running denoising paths".to_string(),
                current,
                total,
            });
        },
    );
    let pixels = tensor_to_rgb_pixels(&output, output_size)?;

    if tx
        .send(tui::TrainingEvent::InferenceImage {
            width: output_size.0,
            height: output_size.1,
            channels: output_size.2,
            pixels,
            checkpoint_path: checkpoint_path.display().to_string(),
            seed,
        })
        .is_err()
    {
        return Ok(());
    }

    let _ = tx.send(tui::TrainingEvent::Done);
    Ok(())
}

fn resolve_inference_checkpoint_path(config: &ModelConfig) -> Result<PathBuf, String> {
    let model_name = config
        .model_name
        .as_deref()
        .ok_or_else(|| "inference requires a named model configuration".to_string())?;
    let checkpoint = tui::storage::default_model_checkpoint_path(model_name).map_err(|err| {
        format!("failed to resolve inference checkpoint path for '{model_name}': {err}")
    })?;
    if !checkpoint.exists() {
        return Err(format!(
            "inference checkpoint not found: {} (train the model first or place weights there)",
            checkpoint.display()
        ));
    }
    Ok(checkpoint)
}

fn load_dataset(
    dataset_path: &str,
    output_size: (u32, u32, u32),
) -> Result<Vec<ImageSample>, String> {
    let canonical_path = Path::new(dataset_path)
        .canonicalize()
        .map_err(|err| format!("failed to resolve dataset path '{}': {err}", dataset_path))?;

    if let Some(dataset) = try_load_raw_dataset(&canonical_path, output_size)? {
        return Ok(dataset);
    }

    if let Some(dataset) = try_load_cifar_dataset(&canonical_path, output_size)? {
        return Ok(dataset);
    }

    let mut image_paths = Vec::new();
    let mut visited_dirs = HashSet::new();
    collect_image_paths(&canonical_path, &mut image_paths, &mut visited_dirs)?;
    image_paths.sort();

    if image_paths.is_empty() {
        return Err(format!("no images found at '{}'", dataset_path));
    }

    image_paths
        .into_iter()
        .map(|path| {
            let image = image::open(&path)
                .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
            Ok(ImageSample {
                target: image_to_tensor(&image, output_size),
            })
        })
        .collect()
}

/// Tries to load a dataset from a raw binary file (`*.batraw`) or a directory that contains such
/// files.  Returns `Ok(None)` when the path does not look like a raw-binary dataset so the caller
/// can fall back to other loaders.
///
/// # Binary format (produced by the Python pre-processing scripts)
/// ```text
/// [0..8]   magic: b"BATRAW1\0"
/// [8..12]  count:    u32 LE – number of samples
/// [12..16] width:    u32 LE – image width in pixels
/// [16..20] height:   u32 LE – image height in pixels
/// [20..24] channels: u32 LE – number of channels per pixel
/// [24..]   data:     count * width * height * channels × f32 LE values, normalised [0, 1]
/// ```
fn try_load_raw_dataset(
    dataset_path: &Path,
    output_size: (u32, u32, u32),
) -> Result<Option<Vec<ImageSample>>, String> {
    // Collect candidate .batraw files.
    let mut raw_files: Vec<PathBuf> = Vec::new();

    if dataset_path.is_file() {
        if dataset_path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("batraw"))
        {
            raw_files.push(dataset_path.to_path_buf());
        } else {
            return Ok(None);
        }
    } else if dataset_path.is_dir() {
        for entry in fs::read_dir(dataset_path)
            .map_err(|err| format!("failed to read directory {}: {err}", dataset_path.display()))?
        {
            let path = entry
                .map_err(|err| {
                    format!(
                        "failed to read directory entry in {}: {err}",
                        dataset_path.display()
                    )
                })?
                .path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("batraw"))
            {
                raw_files.push(path);
            }
        }
        if raw_files.is_empty() {
            return Ok(None);
        }
        raw_files.sort();
    } else {
        return Ok(None);
    }

    let mut dataset: Vec<ImageSample> = Vec::new();
    for raw_file in &raw_files {
        let bytes = fs::read(raw_file)
            .map_err(|err| format!("failed to read {}: {err}", raw_file.display()))?;

        if bytes.len() < RAW_DATASET_MAGIC.len() + 16 {
            return Err(format!(
                "raw dataset file too short to contain a valid header: {}",
                raw_file.display()
            ));
        }
        if &bytes[..RAW_DATASET_MAGIC.len()] != RAW_DATASET_MAGIC {
            return Err(format!(
                "invalid magic in raw dataset file: {}",
                raw_file.display()
            ));
        }

        let mut offset = RAW_DATASET_MAGIC.len();
        // count is a u32 value from a validated header produced by our own tooling, so it
        // safely fits in usize on all supported 32- and 64-bit targets.
        let count = read_u32_le_bytes(&bytes, &mut offset)? as usize;
        let width = read_u32_le_bytes(&bytes, &mut offset)?;
        let height = read_u32_le_bytes(&bytes, &mut offset)?;
        let channels = read_u32_le_bytes(&bytes, &mut offset)?;

        let sample_floats = (width * height * channels) as usize;
        let expected_bytes = offset + count * sample_floats * 4;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "raw dataset file size mismatch in {}: expected {expected_bytes} bytes, got {}",
                raw_file.display(),
                bytes.len()
            ));
        }

        for _ in 0..count {
            let raw: Vec<f32> = bytes[offset..offset + sample_floats * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            offset += sample_floats * 4;

            // Rescale to the model's output dimensions if they differ.
            let target = if (width, height, channels) == output_size {
                raw
            } else {
                let image = raw_floats_to_dynamic_image(&raw, width, height, channels)?;
                image_to_tensor(&image, output_size)
            };
            dataset.push(ImageSample { target });
        }
    }

    Ok(Some(dataset))
}

/// Decode a flat `[0, 1]` f32 slice back into a [`DynamicImage`] for rescaling.
fn raw_floats_to_dynamic_image(
    data: &[f32],
    width: u32,
    height: u32,
    channels: u32,
) -> Result<DynamicImage, String> {
    let pixels: Vec<u8> = data.iter().map(|v| to_u8(*v)).collect();
    match channels {
        1 => {
            let img = GrayImage::from_raw(width, height, pixels)
                .ok_or_else(|| "failed to reconstruct greyscale image from raw data".to_string())?;
            Ok(DynamicImage::ImageLuma8(img))
        }
        3 => {
            let img = RgbImage::from_raw(width, height, pixels)
                .ok_or_else(|| "failed to reconstruct RGB image from raw data".to_string())?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        c => Err(format!(
            "unsupported channel count {c} in raw dataset (expected 1 or 3)"
        )),
    }
}

/// Read a little-endian `u32` from `bytes` at `*offset`, advancing the offset by 4.
fn read_u32_le_bytes(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = *offset + 4;
    if end > bytes.len() {
        return Err(format!(
            "unexpected end of data reading u32 at offset {offset}"
        ));
    }
    let value = u32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    *offset = end;
    Ok(value)
}

fn try_load_cifar_dataset(
    dataset_path: &Path,
    output_size: (u32, u32, u32),
) -> Result<Option<Vec<ImageSample>>, String> {
    let cifar_dir = if dataset_path.join("cifar-10-batches-bin").is_dir() {
        Some(dataset_path.join("cifar-10-batches-bin"))
    } else if dataset_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "cifar-10-batches-bin")
    {
        Some(dataset_path.to_path_buf())
    } else {
        None
    };

    let Some(cifar_dir) = cifar_dir else {
        return Ok(None);
    };

    let mut batch_files = Vec::new();
    for entry in fs::read_dir(&cifar_dir).map_err(|err| {
        format!(
            "failed to read CIFAR directory {}: {err}",
            cifar_dir.display()
        )
    })? {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read directory entry in {}: {err}",
                cifar_dir.display()
            )
        })?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("data_batch_") && name.ends_with(".bin"))
        {
            batch_files.push(path);
        }
    }
    batch_files.sort();

    if batch_files.is_empty() {
        let test_batch = cifar_dir.join("test_batch.bin");
        if test_batch.is_file() {
            batch_files.push(test_batch);
        }
    }

    if batch_files.is_empty() {
        return Err(format!(
            "no CIFAR batch files found in {}",
            cifar_dir.display()
        ));
    }

    let mut dataset = Vec::new();
    for batch_file in batch_files {
        let bytes = fs::read(&batch_file)
            .map_err(|err| format!("failed to read {}: {err}", batch_file.display()))?;
        if bytes.len() % 3073 != 0 {
            return Err(format!(
                "unexpected CIFAR batch size in {}: {} bytes",
                batch_file.display(),
                bytes.len()
            ));
        }

        for record in bytes.chunks_exact(3073) {
            let image = cifar_record_to_image(record)?;
            dataset.push(ImageSample {
                target: image_to_tensor(&image, output_size),
            });
        }
    }

    Ok(Some(dataset))
}

fn cifar_record_to_image(record: &[u8]) -> Result<DynamicImage, String> {
    if record.len() != 3073 {
        return Err(format!(
            "invalid CIFAR record length: expected 3073, got {}",
            record.len()
        ));
    }

    let channels = &record[1..];
    let mut pixels = Vec::with_capacity(32 * 32 * 3);
    for index in 0..1024 {
        pixels.push(channels[index]);
        pixels.push(channels[1024 + index]);
        pixels.push(channels[2048 + index]);
    }
    let image = RgbImage::from_raw(32, 32, pixels)
        .ok_or_else(|| "failed to build CIFAR RGB image".to_string())?;
    Ok(DynamicImage::ImageRgb8(image))
}

fn collect_image_paths(
    path: &Path,
    files: &mut Vec<PathBuf>,
    visited_dirs: &mut HashSet<PathBuf>,
) -> Result<(), String> {
    if path.is_file() {
        if is_supported_image(path) {
            files.push(path.to_path_buf());
            return Ok(());
        }
        return Err(format!(
            "'{}' is not a supported image file",
            path.display()
        ));
    }

    if !path.is_dir() {
        return Err(format!("'{}' is not a file or directory", path.display()));
    }

    let canonical_dir = path
        .canonicalize()
        .map_err(|err| format!("failed to resolve {}: {err}", path.display()))?;
    if !visited_dirs.insert(canonical_dir) {
        return Ok(());
    }

    for entry in fs::read_dir(path)
        .map_err(|err| format!("failed to read directory {}: {err}", path.display()))?
    {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read directory entry in {}: {err}",
                path.display()
            )
        })?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_image_paths(&entry_path, files, visited_dirs)?;
        } else if is_supported_image(&entry_path) {
            files.push(entry_path);
        }
    }

    Ok(())
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp"
            )
        })
        .unwrap_or(false)
}

fn image_to_tensor(image: &DynamicImage, dims: (u32, u32, u32)) -> Vec<f32> {
    let (width, height, channels) = dims;
    if width == 0 || height == 0 || channels == 0 {
        return Vec::new();
    }

    if channels == 1 {
        return image
            .resize_exact(width, height, FilterType::Triangle)
            .to_luma8()
            .pixels()
            .map(|pixel| pixel.0[0] as f32 / 255.0)
            .collect();
    }

    let resized = image
        .resize_exact(width, height, FilterType::Triangle)
        .to_rgb8();
    let mut tensor = Vec::with_capacity((width * height * channels) as usize);
    for pixel in resized.pixels() {
        let rgb = pixel.0;
        for channel in 0..channels as usize {
            let value = match channel {
                0 => rgb[0],
                1 => rgb[1],
                2 => rgb[2],
                _ => rgb[2],
            };
            tensor.push(value as f32 / 255.0);
        }
    }
    tensor
}

fn compose_diffusion_input(
    signal: &[f32],
    input_dims: (u32, u32, u32),
    output_dims: (u32, u32, u32),
    timestep_features: &[f32],
) -> Vec<f32> {
    if signal.is_empty() || input_dims.2 == 0 {
        return signal.to_vec();
    }

    let pixel_count = (output_dims.0 * output_dims.1) as usize;
    let input_channels = input_dims.2 as usize;
    let signal_channels = output_dims.2 as usize;
    let mut packed = vec![0.0f32; pixel_count * input_channels];

    for (pixel_idx, pixel) in signal.chunks(signal_channels).enumerate() {
        let dst = &mut packed[pixel_idx * input_channels..(pixel_idx + 1) * input_channels];
        dst[..signal_channels].copy_from_slice(pixel);
        for (value, feature) in dst
            .iter_mut()
            .skip(signal_channels)
            .zip(timestep_features.iter())
        {
            *value = *feature;
        }
    }
    packed
}

fn sample_diffusion_image<State>(
    model: &mut Model<State>,
    input_dims: (u32, u32, u32),
    output_dims: (u32, u32, u32),
    schedule: &LinearNoiseSchedule,
    seed: u64,
) -> Vec<f32> {
    sample_diffusion_image_with_controls(
        model,
        input_dims,
        output_dims,
        schedule,
        seed,
        1,
        1.0,
        |_, _| {},
    )
}

fn sample_diffusion_image_with_controls<State, F>(
    model: &mut Model<State>,
    input_dims: (u32, u32, u32),
    output_dims: (u32, u32, u32),
    schedule: &LinearNoiseSchedule,
    seed: u64,
    denoising_paths: usize,
    denoise_magnitude: f32,
    mut progress: F,
) -> Vec<f32>
where
    F: FnMut(usize, usize),
{
    let output_len = (output_dims.0 * output_dims.1 * output_dims.2) as usize;
    let path_count = denoising_paths.max(1);
    let steps = schedule.len().max(1);
    let total_work = path_count.saturating_mul(steps);
    let mut accumulated = vec![0.0f32; output_len];

    for path_idx in 0..path_count {
        let path_seed = seed ^ ((path_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut latent = schedule.sample_noise(output_len, path_seed ^ 0xa5a5_5a5a_0123_4567);

        for (step_idx, diffusion_step) in (0..schedule.len()).rev().enumerate() {
            let timestep_features = schedule.timestep_embedding(
                diffusion_step,
                input_dims.2.saturating_sub(output_dims.2) as usize,
            );
            let model_input =
                compose_diffusion_input(&latent, input_dims, output_dims, &timestep_features);
            let predicted_noise = model.predict(&model_input);
            latent = schedule.denoise_step_with_magnitude(
                &latent,
                &predicted_noise,
                diffusion_step,
                path_seed ^ diffusion_step as u64,
                denoise_magnitude,
            );
            progress(path_idx * steps + step_idx + 1, total_work);
        }

        for (acc, value) in accumulated.iter_mut().zip(latent.iter()) {
            *acc += *value;
        }
    }

    for value in &mut accumulated {
        *value /= path_count as f32;
    }
    accumulated
}

fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x5eed_u64);
    nanos ^ (nanos.rotate_left(17)).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn prepare_sample_dir(dataset_path: &str) -> Result<PathBuf, String> {
    let dataset_path = Path::new(dataset_path);
    let sample_dir = if dataset_path.is_dir() {
        dataset_path.join("generated_samples")
    } else {
        dataset_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("generated_samples")
    };

    fs::create_dir_all(&sample_dir).map_err(|err| {
        format!(
            "failed to create sample directory {}: {err}",
            sample_dir.display()
        )
    })?;
    Ok(sample_dir)
}

fn save_tensor_as_image(
    tensor: &[f32],
    dims: (u32, u32, u32),
    sample_dir: &Path,
    step: usize,
) -> Result<PathBuf, String> {
    let (width, height, channels) = dims;
    let expected_len = (width * height * channels) as usize;
    if tensor.len() != expected_len {
        return Err(format!(
            "output tensor length mismatch: expected {expected_len}, got {}",
            tensor.len()
        ));
    }

    let path = sample_dir.join(format!("step_{step:04}.png"));
    if channels == 1 {
        let pixels: Vec<u8> = tensor.iter().map(|value| to_u8(*value)).collect();
        let image = GrayImage::from_raw(width, height, pixels)
            .ok_or_else(|| format!("failed to build grayscale image for {}", path.display()))?;
        image
            .save(&path)
            .map_err(|err| format!("failed to save {}: {err}", path.display()))?;
        return Ok(path);
    }

    let pixels = tensor_to_rgb_pixels(tensor, dims)?;
    let image = RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| format!("failed to build RGB image for {}", path.display()))?;
    image
        .save(&path)
        .map_err(|err| format!("failed to save {}: {err}", path.display()))?;
    Ok(path)
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn tensor_to_rgb_pixels(tensor: &[f32], dims: (u32, u32, u32)) -> Result<Vec<u8>, String> {
    let (width, height, channels) = dims;
    let expected_len = (width * height * channels) as usize;
    if tensor.len() != expected_len {
        return Err(format!(
            "output tensor length mismatch: expected {expected_len}, got {}",
            tensor.len()
        ));
    }
    if channels == 0 {
        return Err("output channels must be > 0".to_string());
    }

    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for pixel in tensor.chunks(channels as usize) {
        let fallback = *pixel.first().unwrap_or(&0.0);
        pixels.push(to_u8(fallback));
        pixels.push(to_u8(*pixel.get(1).unwrap_or(&fallback)));
        pixels.push(to_u8(*pixel.get(2).unwrap_or(&fallback)));
    }
    Ok(pixels)
}

fn append_layer<State>(
    model: &mut Model<State>,
    draft: &LayerDraft,
) -> Result<(), bat_building::ModelError> {
    match draft {
        LayerDraft::Convolution {
            dim_input,
            nb_kernel,
            dim_kernel,
            stride,
            padding,
            save_key,
        } => {
            model.add_layer(LayerTypes::Convolution(ConvolutionType::new(
                Dim3::new(*dim_input),
                *nb_kernel,
                Dim3::new(*dim_kernel),
                *stride,
                convert_padding(padding),
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::Activation {
            dim_input,
            method,
            save_key,
        } => {
            model.add_layer(LayerTypes::Activation(ActivationType::new(
                convert_activation(method),
                Dim3::new(*dim_input),
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::GroupNorm {
            dim_input,
            num_groups,
            save_key,
        } => {
            model.add_layer(LayerTypes::GroupNorm(GroupNormType::new(
                Dim3::new(*dim_input),
                *num_groups,
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::FullyConnected {
            dim_input,
            nb_neurons,
            method,
            save_key,
            ..
        } => {
            model.add_layer(LayerTypes::FullyConnected(FullyConnectedType::new(
                Dim3::new(*dim_input),
                *nb_neurons,
                convert_activation(method),
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::UpsampleConv {
            dim_input,
            scale_factor,
            nb_kernel,
            dim_kernel,
            padding,
            save_key,
            ..
        } => {
            model.add_layer(LayerTypes::UpsampleConv(UpsampleConvType::new(
                Dim3::new(*dim_input),
                *scale_factor,
                *nb_kernel,
                Dim3::new(*dim_kernel),
                convert_padding(padding),
            )))?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
        LayerDraft::Concat {
            skip_key, save_key, ..
        } => {
            model.add_concat(skip_key.clone())?;
            if let Some(key) = save_key {
                model.mark_output(key.clone())?;
            }
            Ok(())
        }
    }
}

fn convert_padding(p: &PaddingMode) -> PPadding {
    match p {
        PaddingMode::Valid => PPadding::Valid,
        PaddingMode::Same => PPadding::Same,
    }
}

fn convert_activation(a: &ActivationMethod) -> PActivation {
    match a {
        ActivationMethod::Relu => PActivation::Relu,
        ActivationMethod::Silu => PActivation::Silu,
        ActivationMethod::Linear => PActivation::Linear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_batraw(
        path: &std::path::Path,
        count: u32,
        width: u32,
        height: u32,
        channels: u32,
        samples: &[Vec<f32>],
    ) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(RAW_DATASET_MAGIC).unwrap();
        for v in [count, width, height, channels] {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        for sample in samples {
            for &v in sample {
                file.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("batlab_test_{name}"));
        dir
    }

    #[test]
    fn raw_dataset_greyscale_round_trip() {
        let out = tmp_path("grey.batraw");
        // One 2×2 greyscale sample.
        let sample = vec![0.0_f32, 0.25, 0.5, 1.0];
        write_batraw(&out, 1, 2, 2, 1, &[sample.clone()]);

        let dataset = try_load_raw_dataset(&out, (2, 2, 1))
            .expect("load should succeed")
            .expect("should detect .batraw file");

        let _ = std::fs::remove_file(&out);
        assert_eq!(dataset.len(), 1);
        for (a, b) in dataset[0].target.iter().zip(sample.iter()) {
            assert!((a - b).abs() < 1e-6, "value mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn raw_dataset_rgb_round_trip() {
        let out = tmp_path("rgb.batraw");
        // One 2×2 RGB sample (4 pixels × 3 channels).
        let sample: Vec<f32> = (0..12).map(|i| i as f32 / 11.0).collect();
        write_batraw(&out, 1, 2, 2, 3, &[sample.clone()]);

        let dataset = try_load_raw_dataset(&out, (2, 2, 3))
            .expect("load should succeed")
            .expect("should detect .batraw file");

        let _ = std::fs::remove_file(&out);
        assert_eq!(dataset.len(), 1);
        for (a, b) in dataset[0].target.iter().zip(sample.iter()) {
            assert!((a - b).abs() < 1e-6, "value mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn raw_dataset_non_batraw_file_returns_none() {
        let out = tmp_path("image.png");
        std::fs::write(&out, b"notabatraw").unwrap();

        let result = try_load_raw_dataset(&out, (32, 32, 1)).unwrap();
        let _ = std::fs::remove_file(&out);
        assert!(result.is_none(), "non-.batraw file should return None");
    }

    #[test]
    fn raw_dataset_wrong_magic_returns_error() {
        let out = tmp_path("bad.batraw");
        // Header with correct structure but wrong magic.
        let mut data = b"WRONGMAG".to_vec();
        for v in [1u32, 2, 2, 1] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..4u32 {
            data.extend_from_slice(&0.5f32.to_le_bytes());
        }
        std::fs::write(&out, &data).unwrap();

        let result = try_load_raw_dataset(&out, (2, 2, 1));
        let _ = std::fs::remove_file(&out);
        assert!(result.is_err(), "wrong magic should return an error");
    }
}
