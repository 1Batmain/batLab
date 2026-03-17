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
    model::{Infer, Training},
};
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};

const DIFFUSION_SCHEDULE_STEPS: usize = 256;
const DIFFUSION_BETA_START: f32 = 1e-4;
const DIFFUSION_BETA_END: f32 = 2e-2;
const LOSS_REPORT_INTERVAL_STEPS: usize = 25;

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

async fn run_training(
    config: ModelConfig,
    train_cfg: TrainingConfig,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
    control_rx: Receiver<tui::TrainingControlCommand>,
) -> Result<(), String> {
    let gpu = Arc::new(GpuContext::new_headless().await);
    let mut model = Model::new_training(
        gpu.clone(),
        train_cfg.lr,
        train_cfg.batch_size,
        PLoss::MeanSquared,
    )
    .await;

    for draft in &config.layers {
        append_layer(&mut model, draft).map_err(|err| err.to_string())?;
    }
    model.build().map_err(|err| err.to_string())?;
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

    // Handle for the live-visualiser window.  `None` until the user presses
    // [v].  Dropping the handle closes the window.
    let mut window_handle: Option<bat_building::visualiser::VisualiserHandle> = None;

    let mut step = 0usize;
    while step < total_steps {
        // If the user closed the visualiser window via its close button, clear
        // the stale handle so that pressing [v] opens a fresh window.
        if window_handle.as_ref().is_some_and(|h| h.is_closed()) {
            window_handle = None;
        }

        while let Ok(command) = control_rx.try_recv() {
            if command == tui::TrainingControlCommand::ToggleWindow {
                window_handle = toggle_visualiser_window(window_handle, &model, output_size);
            } else {
                apply_training_control_command(
                    command,
                    &mut model,
                    &mut paused,
                    &mut current_lr,
                    &mut current_batch_size,
                    &mut total_steps,
                    checkpoint_path.as_deref(),
                    tx,
                );
            }
            let _ = tx.send(tui::TrainingEvent::TrainingState {
                paused,
                lr: current_lr,
                batch_size: current_batch_size,
                total_steps,
            });
        }
        if paused {
            match control_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(command) => {
                    if command == tui::TrainingControlCommand::ToggleWindow {
                        window_handle =
                            toggle_visualiser_window(window_handle, &model, output_size);
                    } else {
                        apply_training_control_command(
                            command,
                            &mut model,
                            &mut paused,
                            &mut current_lr,
                            &mut current_batch_size,
                            &mut total_steps,
                            checkpoint_path.as_deref(),
                            tx,
                        );
                    }
                    let _ = tx.send(tui::TrainingEvent::TrainingState {
                        paused,
                        lr: current_lr,
                        batch_size: current_batch_size,
                        total_steps,
                    });
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

/// Open the visualiser window (or close it if it is already running).
///
/// The window binds the model's GPU output buffer directly – no CPU readback.
/// Dropping the returned [`bat_building::visualiser::VisualiserHandle`] closes the window.
fn toggle_visualiser_window(
    existing: Option<bat_building::visualiser::VisualiserHandle>,
    model: &Model<Training>,
    output_size: (u32, u32, u32),
) -> Option<bat_building::visualiser::VisualiserHandle> {
    if existing.is_some() {
        // Drop the handle → AtomicBool signals the window thread to exit.
        return None;
    }

    let Some(output_buf) = model.last_output_buffer() else {
        eprintln!("[visualiser] model has no output buffer yet");
        return None;
    };

    let title = format!(
        "Model Output  ({}×{}×{} channels)",
        output_size.0, output_size.1, output_size.2
    );

    Some(bat_building::visualiser::spawn_window(
        model.gpu_context(),
        output_buf,
        output_size.0,
        output_size.1,
        output_size.2,
        title,
    ))
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
        // ToggleWindow is handled in the training loop before this function is called.
        tui::TrainingControlCommand::ToggleWindow => {}
    }
}

async fn run_inference(
    config: ModelConfig,
    tx: &std::sync::mpsc::Sender<tui::TrainingEvent>,
) -> Result<(), String> {
    let gpu = Arc::new(GpuContext::new_headless().await);
    let mut model = Model::<Infer>::new(gpu.clone()).await;

    for draft in &config.layers {
        append_layer(&mut model, draft).map_err(|err| err.to_string())?;
    }
    model.build_model().map_err(|err| err.to_string())?;

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

    let seed = 0;
    let output = sample_diffusion_image(&mut model, input_size, output_size, &diffusion, seed);
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
    let output_len = (output_dims.0 * output_dims.1 * output_dims.2) as usize;
    let mut latent = schedule.sample_noise(output_len, seed ^ 0xa5a5_5a5a_0123_4567);

    for diffusion_step in (0..schedule.len()).rev() {
        let timestep_features = schedule.timestep_embedding(
            diffusion_step,
            input_dims.2.saturating_sub(output_dims.2) as usize,
        );
        let model_input =
            compose_diffusion_input(&latent, input_dims, output_dims, &timestep_features);
        let predicted_noise = model.predict(&model_input);
        latent = schedule.denoise_step(
            &latent,
            &predicted_noise,
            diffusion_step,
            seed ^ diffusion_step as u64,
        );
    }

    latent
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
