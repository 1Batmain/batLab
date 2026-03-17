//! File purpose: Implements model functionality for model execution, state, or diagnostics.

use crate::gpu_context::GpuContext;
use crate::model::debug::{LayerDebugView, read_back_f32};
use crate::model::error::ModelError;
use crate::model::layer::Layer;
use crate::model::layer_types::{ConcatType, LayerType, LayerTypes, LossMethod, LossType};
use crate::model::types::Dim3;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use wgpu::Buffer;

const CHECKPOINT_MAGIC: &[u8; 7] = b"BBCKPT1";

// ---------------------------------------------------------------------------
// State markers
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Infer;

#[derive(Debug)]
pub struct Training {
    pub lr: f32,
    pub batch_size: u32,
    pub(crate) loss_method: LossMethod,
}

#[derive(Debug)]
pub struct ModelState {
    pub(crate) is_build: bool,
}

struct PendingLossReadback {
    staging: wgpu::Buffer,
    rx: futures::channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Model<State = Infer> {
    pub(crate) gpu: Arc<GpuContext>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) loss_layer: Option<Layer>,
    pub(crate) training: Option<State>,
    pub(crate) state: ModelState,
    pub(crate) saved_outputs: HashMap<String, usize>,
    pending_loss_readback: Option<PendingLossReadback>,
    last_reported_loss: Option<f32>,
    loss_readback_disabled: bool,
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

impl Model<Infer> {
    pub fn build_model(&mut self) -> Result<(), ModelError> {
        if self.state.is_build {
            self.clear();
        }
        self.build_forwards()?;
        self.state.is_build = true;
        Ok(())
    }

    pub async fn infer_batch(&mut self, input: Vec<f32>) -> Vec<f32> {
        let expected = self
            .layers
            .first()
            .expect("at least one layer required")
            .ty
            .get_dim_input()
            .length() as usize;

        if input.len() == expected {
            return self.predict(&input);
        }
        if input.len() % expected == 0 {
            let count = input.len() / expected;
            println!("running sequential inference for {count} samples");
            let mut out = Vec::new();
            for chunk in input.chunks(expected) {
                out.extend(self.predict(chunk));
            }
            return out;
        }
        panic!(
            "invalid input length: expected {expected} (or a multiple), got {}",
            input.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

impl Model<Training> {
    /// Create a model ready for training.
    pub async fn new_training(
        gpu: Arc<GpuContext>,
        lr: f32,
        batch_size: u32,
        loss_method: LossMethod,
    ) -> Self {
        Self {
            gpu,
            layers: Vec::new(),
            loss_layer: None,
            training: Some(Training {
                lr,
                batch_size,
                loss_method,
            }),
            state: ModelState { is_build: false },
            saved_outputs: HashMap::new(),
            pending_loss_readback: None,
            last_reported_loss: None,
            loss_readback_disabled: false,
        }
    }

    /// Build all forward + backward + SGD passes in the correct order.
    pub fn build(&mut self) -> Result<(), ModelError> {
        if self.state.is_build {
            self.clear();
        }

        // 1. Forward passes for all network layers.
        self.build_forwards()?;

        // 2. Build the loss layer.
        //    Its input (binding 0) is shared from the last forward layer's output.
        let last_fwd_output = self
            .layers
            .last()
            .expect("at least one layer required for training")
            .buffers
            .forward
            .last()
            .unwrap()
            .clone();
        let last_dim = self.layers.last().unwrap().ty.get_dim_output();

        let loss_method = self.training.as_ref().unwrap().loss_method;
        let loss_spec = LossType::new(loss_method, last_dim);
        let mut loss_layer = Layer::new(&self.gpu.device, LayerTypes::Loss(loss_spec), None)
            .expect("failed to create loss layer");

        // create_buffers shares last_fwd_output as binding 0 (model_result),
        // allocates binding 1 (target) and binding 2 (grad_output), returns grad_output.
        let empty_saved_outputs = HashMap::new();
        let loss_grad_out =
            loss_layer.create_buffers(&self.gpu, Some(last_fwd_output), &empty_saved_outputs)?;
        loss_layer.set_pipeline(&self.gpu.device);
        loss_layer.set_bind_group(&self.gpu.device);

        // 3. Backward passes in reverse layer order.
        //    Each layer receives the previous layer's grad_input as its grad_output.
        let lr = self.training.as_ref().unwrap().lr;
        let mut incoming_grad = loss_grad_out;
        let mut pending_saved_grads: HashMap<String, Arc<Buffer>> = HashMap::new();

        for layer in self.layers.iter_mut().rev() {
            let grad_for_layer = if let Some(key) = layer.saved_output_key().map(str::to_string) {
                if let Some(skip_grad) = pending_saved_grads.remove(&key) {
                    layer.create_merge_pass(&self.gpu, Arc::clone(&incoming_grad), skip_grad)
                } else {
                    Arc::clone(&incoming_grad)
                }
            } else {
                Arc::clone(&incoming_grad)
            };

            incoming_grad = layer.create_back_buffers(&self.gpu, Some(grad_for_layer));
            layer.init_back_shader(&self.gpu.device);
            layer.set_back_pipeline(&self.gpu.device);
            layer.set_back_bind_group(&self.gpu.device);
            for (key, grad) in layer.saved_gradient_buffers() {
                if pending_saved_grads.insert(key.clone(), grad).is_some() {
                    return Err(ModelError::DuplicateSavedGradient { key });
                }
            }
            if layer.ty.has_weights() {
                layer.create_opt_pass(&self.gpu, lr);
            }
        }

        self.loss_layer = Some(loss_layer);
        self.state.is_build = true;
        Ok(())
    }

    /// Run one training step: forward → loss/gradient → backward → SGD update.
    /// Call build() before the first train_step.
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) {
        self.run_train_step(input, target);
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.training
            .as_mut()
            .expect("training config unavailable")
            .lr = lr;
    }

    pub fn set_batch_size(&mut self, batch_size: u32) {
        self.training
            .as_mut()
            .expect("training config unavailable")
            .batch_size = batch_size;
    }

    pub fn training_hyperparameters(&self) -> (f32, u32) {
        let training = self.training.as_ref().expect("training config unavailable");
        (training.lr, training.batch_size)
    }

    /// Run one training step and read the resulting loss back to the CPU.
    pub fn train_step_report(&mut self, input: &[f32], target: &[f32]) -> f32 {
        self.run_train_step(input, target);
        self.read_last_loss()
    }

    pub(crate) fn train_step_report_with_prepass<F>(&mut self, prepass: F) -> f32
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        debug_assert!(self.state.is_build, "call build() before train_step()");
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        prepass(&mut encoder);
        self.encode_zero_optimizer_gradients(&mut encoder);
        self.encode_train_graph(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.read_last_loss()
    }

    pub(crate) fn train_step_report_with_prepass_no_opt<F>(&mut self, prepass: F) -> Option<f32>
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        debug_assert!(self.state.is_build, "call build() before train_step()");
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        prepass(&mut encoder);
        self.encode_train_graph_without_opt(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.read_last_loss_optional()
    }

    pub(crate) fn train_step_with_prepass_no_opt<F>(&mut self, prepass: F)
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        debug_assert!(self.state.is_build, "call build() before train_step()");
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        prepass(&mut encoder);
        self.encode_train_graph_without_opt(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
    }

    pub(crate) fn begin_batch_accumulation(&mut self) {
        debug_assert!(
            self.state.is_build,
            "call build() before begin_batch_accumulation()"
        );
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_zero_optimizer_gradients(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
    }

    pub(crate) fn finish_batch_accumulation(&mut self, batch_size: usize) {
        debug_assert!(
            self.state.is_build,
            "call build() before finish_batch_accumulation()"
        );
        let batch_size = batch_size.max(1) as f32;
        let base_lr = self
            .training
            .as_ref()
            .expect("training config unavailable")
            .lr;
        let scaled_lr = base_lr / batch_size;
        for layer in &self.layers {
            layer.set_opt_learning_rate(self.gpu.as_ref(), scaled_lr);
        }
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_train_optimizer_graph(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
    }

    fn run_train_step(&mut self, input: &[f32], target: &[f32]) {
        debug_assert!(self.state.is_build, "call build() before train_step()");

        // Write CPU data before any GPU work is encoded.
        self.gpu.queue.write_buffer(
            self.layers.first().unwrap().buffers.forward[0].as_ref(),
            0,
            bytemuck::cast_slice(input),
        );
        // loss forward buffers: [0]=model_result (shared), [1]=target, [2]=loss_terms, [3]=grad_output
        let loss_buf = self
            .loss_layer
            .as_ref()
            .expect("call build() before train_step()")
            .buffers
            .forward[1]
            .clone();
        self.gpu
            .queue
            .write_buffer(loss_buf.as_ref(), 0, bytemuck::cast_slice(target));

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_zero_optimizer_gradients(&mut encoder);
        self.encode_train_graph(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
    }

    fn encode_train_graph(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_train_graph_without_opt(encoder);
        self.encode_train_optimizer_graph(encoder);
    }

    fn encode_train_graph_without_opt(&self, encoder: &mut wgpu::CommandEncoder) {
        // Forward
        for layer in &self.layers {
            layer.encode_pass(encoder);
        }
        // Loss / initial gradient computation
        self.loss_layer.as_ref().unwrap().encode_pass(encoder);

        // Backward (reverse order)
        for layer in self.layers.iter().rev() {
            layer.encode_merge_pass(encoder);
            layer.encode_back_pass(encoder);
        }
    }

    fn encode_train_optimizer_graph(&self, encoder: &mut wgpu::CommandEncoder) {
        // SGD weight updates
        for layer in &self.layers {
            layer.encode_opt_pass(encoder);
        }
    }

    fn encode_zero_optimizer_gradients(&self, encoder: &mut wgpu::CommandEncoder) {
        for layer in &self.layers {
            layer.encode_zero_opt_gradients(encoder);
        }
    }
}

// ---------------------------------------------------------------------------
// Shared impl
// ---------------------------------------------------------------------------

impl<State> Model<State> {
    pub async fn new(gpu: Arc<GpuContext>) -> Self {
        Self {
            gpu,
            layers: Vec::new(),
            loss_layer: None,
            training: None,
            state: ModelState { is_build: false },
            saved_outputs: HashMap::new(),
            pending_loss_readback: None,
            last_reported_loss: None,
            loss_readback_disabled: false,
        }
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| l.clear());
        self.layers.clear();
        self.loss_layer = None;
        self.state.is_build = false;
        self.saved_outputs.clear();
        self.pending_loss_readback = None;
        self.last_reported_loss = None;
        self.loss_readback_disabled = false;
    }

    pub fn training_mode(&mut self, training: Option<State>) {
        self.clear();
        self.training = training;
    }

    pub fn input_dim(&self) -> Option<Dim3> {
        self.layers.first().map(|layer| layer.ty.get_dim_input())
    }

    pub fn output_dim(&self) -> Option<Dim3> {
        self.layers.last().map(|layer| layer.ty.get_dim_output())
    }

    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelError> {
        if !self.state.is_build {
            return Err(ModelError::InvalidCheckpointFormat {
                message: "model must be built before saving checkpoint".to_string(),
            });
        }

        #[derive(Debug)]
        struct Entry {
            layer_index: u32,
            weights: Vec<f32>,
            bias: Vec<f32>,
        }

        let mut entries = Vec::new();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let Some(bindings) = layer.ty.get_optimizer_bindings() else {
                continue;
            };
            let weights_buf = layer
                .buffers
                .forward
                .get(bindings.weights_forward_index)
                .ok_or_else(|| ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: "missing weights buffer".to_string(),
                })?;
            let bias_buf = layer
                .buffers
                .forward
                .get(bindings.bias_forward_index)
                .ok_or_else(|| ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: "missing bias buffer".to_string(),
                })?;

            let weights = read_back_f32(self.gpu.as_ref(), weights_buf, weights_buf.size())
                .ok_or_else(|| ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: "weights buffer is not readable from GPU".to_string(),
                })?;
            let bias =
                read_back_f32(self.gpu.as_ref(), bias_buf, bias_buf.size()).ok_or_else(|| {
                    ModelError::CheckpointLayerMismatch {
                        layer_index,
                        message: "bias buffer is not readable from GPU".to_string(),
                    }
                })?;
            entries.push(Entry {
                layer_index: layer_index as u32,
                weights,
                bias,
            });
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(CHECKPOINT_MAGIC);
        bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for entry in entries {
            bytes.extend_from_slice(&entry.layer_index.to_le_bytes());
            bytes.extend_from_slice(&(entry.weights.len() as u32).to_le_bytes());
            bytes.extend_from_slice(bytemuck::cast_slice(&entry.weights));
            bytes.extend_from_slice(&(entry.bias.len() as u32).to_le_bytes());
            bytes.extend_from_slice(bytemuck::cast_slice(&entry.bias));
        }

        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| ModelError::CheckpointIo {
                path: parent.display().to_string(),
                message: err.to_string(),
            })?;
        }
        fs::write(path, bytes).map_err(|err| ModelError::CheckpointIo {
            path: path.display().to_string(),
            message: err.to_string(),
        })
    }

    pub fn load_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelError> {
        if !self.state.is_build {
            return Err(ModelError::InvalidCheckpointFormat {
                message: "model must be built before loading checkpoint".to_string(),
            });
        }

        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|err| ModelError::CheckpointIo {
            path: path.display().to_string(),
            message: err.to_string(),
        })?;
        if bytes.len() < CHECKPOINT_MAGIC.len() + 4
            || &bytes[..CHECKPOINT_MAGIC.len()] != CHECKPOINT_MAGIC
        {
            return Err(ModelError::InvalidCheckpointFormat {
                message: "missing or invalid checkpoint magic".to_string(),
            });
        }

        let mut offset = CHECKPOINT_MAGIC.len();
        let entry_count = read_u32_le(&bytes, &mut offset)? as usize;
        for _ in 0..entry_count {
            let layer_index = read_u32_le(&bytes, &mut offset)? as usize;
            let weight_len = read_u32_le(&bytes, &mut offset)? as usize;
            let weights = read_f32_vec_le(&bytes, &mut offset, weight_len)?;
            let bias_len = read_u32_le(&bytes, &mut offset)? as usize;
            let bias = read_f32_vec_le(&bytes, &mut offset, bias_len)?;

            let layer = self.layers.get(layer_index).ok_or_else(|| {
                ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: "layer index not found in model".to_string(),
                }
            })?;
            let bindings = layer.ty.get_optimizer_bindings().ok_or_else(|| {
                ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: "layer has no trainable parameters".to_string(),
                }
            })?;
            let weights_buf = layer
                .buffers
                .forward
                .get(bindings.weights_forward_index)
                .ok_or_else(|| ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: "missing weights buffer".to_string(),
                })?;
            let bias_buf = layer
                .buffers
                .forward
                .get(bindings.bias_forward_index)
                .ok_or_else(|| ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: "missing bias buffer".to_string(),
                })?;

            let expected_weights = (weights_buf.size() as usize) / std::mem::size_of::<f32>();
            let expected_bias = (bias_buf.size() as usize) / std::mem::size_of::<f32>();
            if weights.len() != expected_weights {
                return Err(ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: format!(
                        "weights length mismatch (expected {expected_weights}, got {})",
                        weights.len()
                    ),
                });
            }
            if bias.len() != expected_bias {
                return Err(ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: format!(
                        "bias length mismatch (expected {expected_bias}, got {})",
                        bias.len()
                    ),
                });
            }

            self.gpu
                .queue
                .write_buffer(weights_buf.as_ref(), 0, bytemuck::cast_slice(&weights));
            self.gpu
                .queue
                .write_buffer(bias_buf.as_ref(), 0, bytemuck::cast_slice(&bias));
        }

        if offset != bytes.len() {
            return Err(ModelError::InvalidCheckpointFormat {
                message: "checkpoint has trailing bytes".to_string(),
            });
        }
        Ok(())
    }

    pub fn predict(&mut self, input: &[f32]) -> Vec<f32> {
        debug_assert!(
            self.state.is_build,
            "call build/build_model before predict()"
        );

        self.gpu.queue.write_buffer(
            self.layers
                .first()
                .expect("at least one layer required")
                .buffers
                .forward[0]
                .as_ref(),
            0,
            bytemuck::cast_slice(input),
        );

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        for layer in &self.layers {
            layer.encode_pass(&mut encoder);
        }
        self.gpu.queue.submit([encoder.finish()]);

        self.read_last_output()
    }

    fn build_forwards(&mut self) -> Result<(), ModelError> {
        let mut last_output: Option<Arc<Buffer>> = None;
        let mut saved_output_buffers: HashMap<String, Arc<Buffer>> = HashMap::new();
        for layer in &mut self.layers {
            last_output =
                Some(layer.create_buffers(&self.gpu, last_output, &saved_output_buffers)?);
            layer.set_pipeline(&self.gpu.device);
            layer.set_bind_group(&self.gpu.device);
            if let Some(key) = layer.saved_output_key().map(str::to_string) {
                saved_output_buffers.insert(key, Arc::clone(last_output.as_ref().unwrap()));
            }
        }
        Ok(())
    }

    pub fn add_layer(&mut self, spec: LayerTypes) -> Result<(), ModelError> {
        if matches!(spec, LayerTypes::Loss(_)) {
            panic!("Loss is not a network layer; pass LossMethod to new_training() instead");
        }
        let last_output = self.layers.last().map(|l| l.ty.get_dim_output());
        let layer = Layer::new(&self.gpu.device, spec, last_output)?;
        self.layers.push(layer);
        Ok(())
    }

    pub fn mark_output(&mut self, key: impl Into<String>) -> Result<(), ModelError> {
        let key = key.into();
        if self.layers.is_empty() {
            return Err(ModelError::NoLayersToMark);
        }
        if self.saved_outputs.contains_key(&key) {
            return Err(ModelError::DuplicateSavedOutput { key });
        }
        let layer_index = self.layers.len() - 1;
        self.layers[layer_index].mark_saved_output(key.clone());
        self.saved_outputs.insert(key, layer_index);
        Ok(())
    }

    pub fn add_concat(&mut self, key: impl Into<String>) -> Result<(), ModelError> {
        let key = key.into();
        let source_index = self
            .saved_outputs
            .get(&key)
            .copied()
            .ok_or_else(|| ModelError::MissingSavedOutput { key: key.clone() })?;
        let skip_dim = self.layers[source_index].ty.get_dim_output();
        let last_output = self.layers.last().map(|l| l.ty.get_dim_output());
        let layer = Layer::new(
            &self.gpu.device,
            LayerTypes::Concat(ConcatType::new(key, Dim3::default(), skip_dim)),
            last_output,
        )?;
        self.layers.push(layer);
        Ok(())
    }

    pub fn read_last_output(&self) -> Vec<f32> {
        let last = self.layers.last().expect("at least one layer required");
        read_back_f32(
            self.gpu.as_ref(),
            last.buffers.forward.last().expect("no output buffer"),
            last.ty.get_dim_output().bytes_size() as u64,
        )
        .expect("failed to read last output buffer")
    }

    /// Returns an `Arc` to the last layer's output buffer so that external
    /// render pipelines (e.g. `the_window`) can bind it directly on the GPU
    /// without copying data to the CPU.
    ///
    /// The buffer already carries `STORAGE | COPY_SRC | COPY_DST` usage flags.
    /// Returns `None` if no layers have been built yet.
    pub fn last_output_buffer(&self) -> Option<Arc<Buffer>> {
        self.layers
            .last()
            .and_then(|l| l.buffers.forward.last())
            .map(Arc::clone)
    }

    /// Returns the shared [`GpuContext`] so the caller can use the same
    /// wgpu device/queue/instance for rendering without creating a second one.
    pub fn gpu_context(&self) -> Arc<GpuContext> {
        Arc::clone(&self.gpu)
    }

    pub fn estimated_gpu_bytes(&self) -> u64 {
        fn add_unique(total: &mut u64, seen: &mut HashSet<usize>, buffer: &Arc<wgpu::Buffer>) {
            let key = Arc::as_ptr(buffer) as usize;
            if seen.insert(key) {
                *total = total.saturating_add(buffer.size());
            }
        }

        let mut total = 0u64;
        let mut seen = HashSet::new();

        for layer in &self.layers {
            for buffer in &layer.buffers.forward {
                add_unique(&mut total, &mut seen, buffer);
            }
            if let Some(backward) = &layer.buffers.backward {
                for buffer in backward {
                    add_unique(&mut total, &mut seen, buffer);
                }
            }
            if let Some(opt) = &layer.opt_pass {
                for buffer in &opt.buffers {
                    add_unique(&mut total, &mut seen, buffer);
                }
            }
        }

        if let Some(loss_layer) = &self.loss_layer {
            for buffer in &loss_layer.buffers.forward {
                add_unique(&mut total, &mut seen, buffer);
            }
            if let Some(backward) = &loss_layer.buffers.backward {
                for buffer in backward {
                    add_unique(&mut total, &mut seen, buffer);
                }
            }
        }

        total
    }

    fn average_loss_terms(loss_terms: &[f32]) -> f32 {
        if loss_terms.is_empty() {
            0.0
        } else {
            loss_terms.iter().sum::<f32>() / loss_terms.len() as f32
        }
    }

    fn disable_loss_readback(&mut self, reason: &str) {
        if !self.loss_readback_disabled {
            eprintln!("[training] {reason}; disabling loss readback for this run");
            self.loss_readback_disabled = true;
            self.pending_loss_readback = None;
        }
    }

    pub fn read_last_loss_optional(&mut self) -> Option<f32> {
        if self.loss_readback_disabled {
            return self.last_reported_loss;
        }

        let loss_layer = self
            .loss_layer
            .as_ref()
            .expect("loss layer is only available in training mode");
        let Some(loss_terms) = read_back_f32(
            self.gpu.as_ref(),
            &loss_layer.buffers.forward[2],
            loss_layer.ty.get_dim_output().bytes_size() as u64,
        ) else {
            self.disable_loss_readback("failed to read loss buffer");
            return self.last_reported_loss;
        };
        let loss = Self::average_loss_terms(&loss_terms);
        self.last_reported_loss = Some(loss);
        Some(loss)
    }

    pub fn read_last_loss(&mut self) -> f32 {
        self.read_last_loss_optional().unwrap_or(0.0)
    }

    /// Schedule a non-blocking loss readback from GPU.
    /// Returns false when no loss buffer is available or a prior request is still pending.
    pub fn request_loss_readback(&mut self) -> bool {
        if self.loss_readback_disabled {
            return false;
        }
        if self.pending_loss_readback.is_some() {
            return false;
        }

        let Some(loss_layer) = self.loss_layer.as_ref() else {
            return false;
        };
        let Some(loss_terms_buf) = loss_layer.buffers.forward.get(2) else {
            return false;
        };
        if !loss_terms_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC)
        {
            return false;
        }
        let size_bytes = loss_layer.ty.get_dim_output().bytes_size() as u64;
        if size_bytes == 0 {
            return false;
        }

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loss_readback_staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(loss_terms_buf.as_ref(), 0, &staging, 0, size_bytes);
        self.gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.pending_loss_readback = Some(PendingLossReadback { staging, rx });
        true
    }

    /// Poll a pending non-blocking loss readback.
    /// Returns a value only when a pending request has completed.
    pub fn poll_loss_readback(&mut self) -> Option<f32> {
        let mut pending = self.pending_loss_readback.take()?;
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);

        match pending.rx.try_recv() {
            Ok(None) => {
                self.pending_loss_readback = Some(pending);
                None
            }
            Ok(Some(Ok(()))) => {
                let loss_terms = {
                    let bytes = pending.staging.slice(..).get_mapped_range();
                    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
                };
                pending.staging.unmap();
                let loss = Self::average_loss_terms(&loss_terms);
                self.last_reported_loss = Some(loss);
                Some(loss)
            }
            Ok(Some(Err(_))) | Err(_) => {
                self.disable_loss_readback("non-blocking loss readback failed");
                None
            }
        }
    }
}

fn read_u32_le(bytes: &[u8], offset: &mut usize) -> Result<u32, ModelError> {
    let end = offset.saturating_add(4);
    if end > bytes.len() {
        return Err(ModelError::InvalidCheckpointFormat {
            message: "unexpected end of checkpoint while reading u32".to_string(),
        });
    }
    let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
    *offset = end;
    Ok(value)
}

fn read_f32_vec_le(bytes: &[u8], offset: &mut usize, len: usize) -> Result<Vec<f32>, ModelError> {
    let byte_len = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
        ModelError::InvalidCheckpointFormat {
            message: "overflow while reading f32 vector".to_string(),
        }
    })?;
    let end = offset.saturating_add(byte_len);
    if end > bytes.len() {
        return Err(ModelError::InvalidCheckpointFormat {
            message: "unexpected end of checkpoint while reading f32 vector".to_string(),
        });
    }
    let mut values = Vec::with_capacity(len);
    for chunk in bytes[*offset..end].chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    *offset = end;
    Ok(values)
}

// ---------------------------------------------------------------------------
// Custom Debug for Model<State>
// ---------------------------------------------------------------------------

impl<State> fmt::Debug for Model<State> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Model");
        s.field("built", &self.state.is_build);
        s.field("num_layers", &self.layers.len());

        let layer_views: Vec<LayerDebugView<'_>> = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| LayerDebugView {
                idx: i,
                layer: l,
                gpu: &self.gpu,
            })
            .collect();
        s.field("layers", &layer_views);

        if let Some(ref loss) = self.loss_layer {
            s.field(
                "loss_layer",
                &LayerDebugView {
                    idx: 0,
                    layer: loss,
                    gpu: &self.gpu,
                },
            );
        }

        s.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Model;
    use crate::gpu_context::GpuContext;
    use crate::model::layer_types::{
        ActivationMethod, ActivationType, GroupNormType, LayerTypes, LossMethod,
    };
    use crate::model::types::Dim3;
    use std::fs;
    use std::sync::Arc;

    #[test]
    fn model_can_concat_saved_skip_outputs() {
        pollster::block_on(async {
            let gpu = Arc::new(GpuContext::new_headless().await);
            let mut model = Model::new(gpu).await;

            model
                .add_layer(LayerTypes::Activation(ActivationType::new(
                    ActivationMethod::Linear,
                    Dim3::new((2, 2, 2)),
                )))
                .unwrap();
            model.mark_output("skip").unwrap();
            model
                .add_layer(LayerTypes::Activation(ActivationType::new(
                    ActivationMethod::Linear,
                    Dim3::default(),
                )))
                .unwrap();
            model.add_concat("skip").unwrap();
            model.build_model().unwrap();

            let output = model.infer_batch(vec![1.0; 8]).await;
            assert_eq!(output.len(), 16);
            assert!(
                output
                    .iter()
                    .all(|value| (*value - 1.0).abs() < f32::EPSILON)
            );
        });
    }

    #[test]
    fn training_model_can_build_and_run_group_norm() {
        pollster::block_on(async {
            let gpu = Arc::new(GpuContext::new_headless().await);
            let mut model = Model::new_training(gpu, 0.01, 1, LossMethod::MeanSquared).await;

            model
                .add_layer(LayerTypes::GroupNorm(GroupNormType::new(
                    Dim3::new((1, 1, 4)),
                    2,
                )))
                .unwrap();
            model.build().unwrap();

            let loss = model.train_step_report(&[1.0, 3.0, 5.0, 7.0], &[0.0, 0.0, 0.0, 0.0]);
            let output = model.read_last_output();

            assert!(loss.is_finite());
            assert_eq!(output.len(), 4);
            assert!((output[0] + 1.0).abs() < 1e-3);
            assert!((output[1] - 1.0).abs() < 1e-3);
            assert!((output[2] + 1.0).abs() < 1e-3);
            assert!((output[3] - 1.0).abs() < 1e-3);
        });
    }

    #[test]
    fn checkpoints_roundtrip_group_norm_parameters() {
        pollster::block_on(async {
            let gpu = Arc::new(GpuContext::new_headless().await);
            let checkpoint_path = std::env::temp_dir().join(format!(
                "bat-building-checkpoint-{}-{}.ckpt",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));

            let mut trained =
                Model::new_training(gpu.clone(), 0.05, 1, LossMethod::MeanSquared).await;
            trained
                .add_layer(LayerTypes::GroupNorm(GroupNormType::new(
                    Dim3::new((1, 1, 4)),
                    2,
                )))
                .unwrap();
            trained.build().unwrap();
            let input = [0.1, 0.4, -0.2, 0.8];
            let target = [0.9, -0.7, 0.2, -0.4];
            let _ = trained.train_step_report(&input, &target);
            let expected_output = trained.predict(&input);
            trained.save_checkpoint(&checkpoint_path).unwrap();

            let mut restored =
                Model::new_training(gpu.clone(), 0.05, 1, LossMethod::MeanSquared).await;
            restored
                .add_layer(LayerTypes::GroupNorm(GroupNormType::new(
                    Dim3::new((1, 1, 4)),
                    2,
                )))
                .unwrap();
            restored.build().unwrap();
            restored.load_checkpoint(&checkpoint_path).unwrap();
            let restored_output = restored.predict(&input);

            for (a, b) in expected_output.iter().zip(restored_output.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "checkpoint output mismatch: {a} vs {b}"
                );
            }

            let _ = fs::remove_file(checkpoint_path);
        });
    }
}
