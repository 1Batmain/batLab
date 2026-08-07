//! File purpose: Implements model functionality for model execution, state, or diagnostics.

use crate::gpu_context::GpuContext;
use crate::model::debug::{LayerDebugView, read_back_f32, read_back_f32_at};
use crate::model::ema::EmaConfig;
use crate::model::error::ModelError;
use crate::model::layer::Layer;
use crate::model::layer_types::{ConcatType, LayerType, LayerTypes, LossMethod, LossType};
use crate::model::optimizer::OptimizerKind;
use crate::model::types::Dim3;
use crate::model::weight_init::WeightInit;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use wgpu::Buffer;

/// Weights + biases only. Still read; never written any more.
const CHECKPOINT_MAGIC_V1: &[u8; 7] = b"BBCKPT1";
/// V1 body followed by an optimiser-state trailer (see `save_checkpoint`).
/// Still WRITTEN, by every run that keeps no weight average.
const CHECKPOINT_MAGIC_V2: &[u8; 7] = b"BBCKPT2";
/// V2 followed by an EMA trailer: the averaged copy of every weight and bias.
/// Written only by a run that has an EMA, so a run without one produces a file
/// byte-identical to the one it produced before averaging existed.
const CHECKPOINT_MAGIC_V3: &[u8; 7] = b"BBCKPT3";
const CHECKPOINT_MAGIC_LEN: usize = 7;

/// Trailer tags for the optimiser-state section of a V2 checkpoint.
const OPT_STATE_NONE: u32 = 0;
const OPT_STATE_ADAM: u32 = 1;

/// Trailer tags for the EMA section of a V3 checkpoint.
const EMA_STATE_NONE: u32 = 0;
const EMA_STATE_PRESENT: u32 = 1;

/// Which of the two weight sets a checkpoint carries lands in the model's
/// weight buffers.
///
/// A V3 checkpoint holds both the last iterate and its moving average. Training
/// must resume from the iterate — the average is not a point the optimiser ever
/// visited, and Adam's moments describe the iterate. Sampling wants the
/// average, which is the whole reason it is kept. So the choice is made by the
/// caller, at the call site, rather than guessed from a model state that reads
/// the same in both cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckpointWeights {
    /// The weights as the last optimiser step left them. The historical
    /// behaviour, and the one every training path wants.
    #[default]
    Raw,
    /// The moving average, when the file has one; falls back on `Raw` when it
    /// has not, which is what makes `--raw-weights` a comparison and not a
    /// prerequisite.
    Ema,
}

// ---------------------------------------------------------------------------
// State markers
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Infer;

#[derive(Debug)]
pub struct Training {
    pub lr: f32,
    pub batch_size: u32,
    pub optimizer: OptimizerKind,
    pub(crate) loss_method: LossMethod,
}

#[derive(Debug)]
pub struct ModelState {
    pub(crate) is_build: bool,
}

/// What a checkpoint turned out to hold, reported back to whoever loaded it.
///
/// `carries_ema` and `used_ema` differ exactly when a caller asked for the
/// average and the file has none — the case that would otherwise turn an
/// EMA-vs-raw comparison into a comparison of a thing with itself. The CLI
/// prints both.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CheckpointLoad {
    /// The file has an EMA trailer.
    pub carries_ema: bool,
    /// The averaged set is what landed in the weight buffers.
    pub used_ema: bool,
    /// The decay the file was written with, when it carries an average.
    pub ema_decay: Option<f32>,
}

/// The EMA trailer of a checkpoint, parsed but not yet applied.
struct EmaPayload {
    decay: f32,
    /// `(layer_index, [ema_weights, ema_bias])`.
    records: Vec<(usize, Vec<Vec<f32>>)>,
}

impl EmaPayload {
    fn weights_and_bias_for(&self, layer_index: usize) -> Option<(&Vec<f32>, &Vec<f32>)> {
        self.records
            .iter()
            .find(|(index, _)| *index == layer_index)
            .and_then(|(_, section)| match section.as_slice() {
                [weights, bias] => Some((weights, bias)),
                _ => None,
            })
    }
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
    /// Global optimiser step counter `t`, 1-based at the first update.
    ///
    /// Lives on the model rather than on `Training` so that checkpoint I/O —
    /// which is shared with `Model<Infer>` — can persist and restore it.
    /// Adam's bias correction is a function of `t` alone, so resuming with the
    /// wrong `t` is what makes a naive resume take a huge first step.
    optimizer_step: u64,
    /// Initialisation scheme for trainable weight buffers. On the model rather
    /// than on `Training` because `build_forwards` is shared with inference.
    weight_init: WeightInit,
    /// The weight average this run keeps, if any. `None` — the default — means
    /// no shadow buffers, no extra dispatch, and a V2 checkpoint.
    ///
    /// On the model rather than on `Training` for the same reason the step
    /// counter is: checkpoint I/O is shared with `Model<Infer>`, and it has to
    /// know whether there is a shadow to fill.
    ema: Option<EmaConfig>,
    pending_loss_readback: Option<PendingLossReadback>,
    last_reported_loss: Option<f32>,
    loss_readback_disabled: bool,
    /// Samples the built graph carries at once.
    ///
    /// Fixed at `build()` time, because it sizes every activation buffer.
    /// Inference and every non-batched entry point leave it at 1, which
    /// reproduces the pre-batching graph exactly.
    batch: u32,
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
    /// Create a model ready for training with the default optimiser (SGD).
    pub async fn new_training(
        gpu: Arc<GpuContext>,
        lr: f32,
        batch_size: u32,
        loss_method: LossMethod,
    ) -> Self {
        Self::new_training_with_optimizer(gpu, lr, batch_size, loss_method, OptimizerKind::Sgd)
            .await
    }

    /// Create a model ready for training with an explicit optimiser.
    pub async fn new_training_with_optimizer(
        gpu: Arc<GpuContext>,
        lr: f32,
        batch_size: u32,
        loss_method: LossMethod,
        optimizer: OptimizerKind,
    ) -> Self {
        Self {
            gpu,
            layers: Vec::new(),
            loss_layer: None,
            training: Some(Training {
                lr,
                batch_size,
                optimizer,
                loss_method,
            }),
            state: ModelState { is_build: false },
            saved_outputs: HashMap::new(),
            optimizer_step: 0,
            weight_init: WeightInit::default(),
            ema: None,
            pending_loss_readback: None,
            last_reported_loss: None,
            loss_readback_disabled: false,
            batch: 1,
        }
    }

    /// Choose how trainable weights are drawn at build time. Must be set
    /// before `build()`; the default is the historical uniform draw.
    pub fn set_weight_init(&mut self, init: WeightInit) {
        self.weight_init = init;
    }

    /// Keep an exponential moving average of the weights (`None` — the default
    /// — keeps none).
    ///
    /// Must be set before `build()`: the shadow buffers are allocated with the
    /// optimiser passes and seeded from the weights that exist then.
    pub fn set_ema(&mut self, ema: Option<EmaConfig>) {
        self.ema = ema;
    }

    pub fn optimizer(&self) -> OptimizerKind {
        self.training
            .as_ref()
            .map(|t| t.optimizer)
            .unwrap_or_default()
    }

    /// Build all forward + backward + SGD passes in the correct order.
    ///
    /// The training batch size is baked in here: it sizes every activation
    /// buffer and every dispatch. Changing it afterwards therefore means
    /// rebuilding — see [`Model::resize_batch`].
    pub fn build(&mut self) -> Result<(), ModelError> {
        if self.state.is_build {
            self.clear();
        }
        self.batch = self
            .training
            .as_ref()
            .map(|t| t.batch_size.max(1))
            .unwrap_or(1);

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
        loss_layer.batch = self.batch;

        // create_buffers shares last_fwd_output as binding 0 (model_result) and
        // allocates [1] target, [2] loss_terms, [3] grad_output, [4] specs.
        //
        // The loss layer is the one place where "the layer's output" is NOT its
        // last buffer: what feeds the backward chain is `grad_output`, at the
        // fixed index 3. Taking `create_buffers`' return value here would have
        // chained whatever binding happens to sit last — which, since the specs
        // uniform was appended, is a UNIFORM buffer. It fails loudly at bind
        // group creation ("does not contain required usage flags STORAGE"), but
        // only because the usages differ; a storage buffer added last would
        // have been wired in silently.
        const LOSS_GRAD_OUTPUT_INDEX: usize = 3;
        let empty_saved_outputs = HashMap::new();
        loss_layer.create_buffers(
            &self.gpu,
            Some(last_fwd_output),
            &empty_saved_outputs,
            WeightInit::default(),
            self.layers.len(),
        )?;
        let loss_grad_out = Arc::clone(&loss_layer.buffers.forward[LOSS_GRAD_OUTPUT_INDEX]);
        loss_layer.set_pipeline(&self.gpu.device);
        loss_layer.set_bind_group(&self.gpu.device);

        // 3. Backward passes in reverse layer order.
        //    Each layer receives the previous layer's grad_input as its grad_output.
        let lr = self.training.as_ref().unwrap().lr;
        let optimizer = self.training.as_ref().unwrap().optimizer;
        let ema = self.ema;
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
                layer.create_opt_pass(&self.gpu, lr, optimizer);
                if let Some(ema) = ema {
                    layer.create_ema_pass(&self.gpu, ema);
                }
            }
        }

        self.loss_layer = Some(loss_layer);
        self.state.is_build = true;
        // Freshly allocated optimiser state (Adam's m and v are zero) — the
        // step counter must restart with it. load_checkpoint() restores both
        // together afterwards.
        self.optimizer_step = 0;
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

    /// Record a new batch size. Only takes effect on the next [`Model::build`];
    /// use [`Model::resize_batch`] on an already-built model.
    pub fn set_batch_size(&mut self, batch_size: u32) {
        self.training
            .as_mut()
            .expect("training config unavailable")
            .batch_size = batch_size;
    }

    /// Change the batch size of a *built* model, preserving its training state.
    ///
    /// The batch axis lives in the size of every activation buffer, so a new
    /// batch size means new buffers, new bind groups and new dispatch counts —
    /// i.e. a rebuild. What must survive the rebuild is everything that *is*
    /// the run: weights, biases, Adam's first and second moments, and the
    /// global step counter `t` (Adam's bias correction is a function of `t`
    /// alone, so losing it would make the first step after a resize a full
    /// +/-lr on every weight). A checkpoint round-trip carries exactly that
    /// set, which is why it is used rather than a hand-rolled copy.
    ///
    /// Called from a user keystroke in the TUI, i.e. rarely.
    pub fn resize_batch(&mut self, batch_size: u32) -> Result<(), ModelError> {
        let batch_size = batch_size.max(1);
        if self.state.is_build && self.batch == batch_size {
            self.set_batch_size(batch_size);
            return Ok(());
        }
        if !self.state.is_build {
            self.set_batch_size(batch_size);
            return Ok(());
        }
        let checkpoint = self.checkpoint_bytes()?;
        self.set_batch_size(batch_size);
        self.build()?;
        self.load_checkpoint_bytes(&checkpoint)
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
        // Single-sample entry point: it writes slot 0 and reads slot 0, but the
        // graph it encodes covers `self.batch` samples. On a batched graph the
        // other slots would be computed from stale memory and their gradients
        // accumulated as if they were data.
        debug_assert_eq!(
            self.batch, 1,
            "train_step_report_with_prepass is a batch-1 entry point"
        );
        // Single sample, no accumulation: the gradient is already its own mean.
        self.publish_optimizer_specs(1.0);
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        prepass(&mut encoder);
        self.encode_zero_optimizer_gradients(&mut encoder);
        self.encode_train_graph(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.read_last_loss()
    }

    /// Sequential reference path — one submit per sample, no optimiser.
    ///
    /// This is what the training loop did before the batch axis moved into the
    /// dispatches, kept verbatim and reachable from the tests so that the
    /// equivalence checks compare against *executable code* rather than a
    /// frozen fixture. See `batch_equivalence_tests.rs`.
    #[cfg(test)]
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

    /// See [`Model::train_step_report_with_prepass_no_opt`].
    #[cfg(test)]
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

    /// One training step over the whole batch: **one encoder, one submit**.
    ///
    /// This replaces the `begin_batch_accumulation` / N x
    /// `train_step_*_with_prepass_no_opt` / `finish_batch_accumulation`
    /// sequence, which cost `2 + batch` submissions per step (18 at batch 16)
    /// and left every kernel working on a single small tensor.
    ///
    /// The order inside the encoder is the same as before, and so is the
    /// arithmetic that depends on it:
    ///
    /// 1. zero `grad_weights` / `grad_bias` — still needed, the gradient
    ///    kernels still accumulate with `+=`, they just do it once per step
    ///    now instead of once per sample;
    /// 2. the caller's prepass (composing `x_t` and the target noise);
    /// 3. forward, loss, backward for the whole batch;
    /// 4. the optimiser, with `grad_scale = 1 / batch` — unchanged, because
    ///    the gradient buffer holds the same SUM over the batch it held before.
    ///
    /// wgpu inserts an implicit barrier between compute passes in one encoder,
    /// so the sequencing that used to be enforced by separate submissions is
    /// preserved without them.
    pub(crate) fn train_step_report_batched<F>(
        &mut self,
        batch_size: usize,
        report_loss: bool,
        prepass: F,
    ) -> Option<f32>
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        debug_assert!(self.state.is_build, "call build() before train_step()");
        let batch_size = batch_size.max(1) as f32;
        self.publish_optimizer_specs(1.0 / batch_size);

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_zero_optimizer_gradients(&mut encoder);
        prepass(&mut encoder);
        self.encode_train_graph(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);

        if report_loss {
            self.read_last_loss_optional()
        } else {
            None
        }
    }

    /// See [`Model::train_step_report_with_prepass_no_opt`].
    #[cfg(test)]
    pub(crate) fn begin_batch_accumulation(&mut self) {
        debug_assert!(
            self.state.is_build,
            "call build() before begin_batch_accumulation()"
        );
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_zero_optimizer_gradients(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
    }

    /// See [`Model::train_step_report_with_prepass_no_opt`].
    #[cfg(test)]
    pub(crate) fn finish_batch_accumulation(&mut self, batch_size: usize) {
        debug_assert!(
            self.state.is_build,
            "call build() before finish_batch_accumulation()"
        );
        let batch_size = batch_size.max(1) as f32;
        self.publish_optimizer_specs(1.0 / batch_size);
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

        // Single sample, no accumulation: the gradient is already its own mean.
        self.publish_optimizer_specs(1.0);
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_zero_optimizer_gradients(&mut encoder);
        self.encode_train_graph(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
    }

    /// Advance the global step counter and push the resulting hyperparameters
    /// to every optimiser pass. Must be called exactly once per weight update,
    /// immediately before the optimiser dispatch is encoded.
    fn publish_optimizer_specs(&mut self, grad_scale: f32) {
        self.optimizer_step += 1;
        let step = self.optimizer_step;
        let lr = self
            .training
            .as_ref()
            .expect("training config unavailable")
            .lr;
        let ema_decay = self.ema.map(|ema| ema.effective_decay(step));
        for layer in &self.layers {
            layer.write_opt_specs(self.gpu.as_ref(), lr, grad_scale, step);
            if let Some(decay) = ema_decay {
                layer.write_ema_specs(self.gpu.as_ref(), decay);
            }
        }
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
        // …then the weight average, in the same encoder. wgpu's implicit
        // barrier between compute passes is what makes the EMA read the weights
        // this step produced rather than the previous step's.
        for layer in &self.layers {
            layer.encode_ema_pass(encoder);
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
            optimizer_step: 0,
            weight_init: WeightInit::default(),
            ema: None,
            pending_loss_readback: None,
            last_reported_loss: None,
            loss_readback_disabled: false,
            batch: 1,
        }
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| l.clear());
        self.layers.clear();
        self.loss_layer = None;
        self.state.is_build = false;
        self.saved_outputs.clear();
        self.optimizer_step = 0;
        self.pending_loss_readback = None;
        self.last_reported_loss = None;
        self.loss_readback_disabled = false;
        // Back to the inference default. `build()` re-reads the training batch
        // size right after clearing, so this only ever affects a model that is
        // cleared and never rebuilt for training — including
        // `Model<Infer>::build_model`, which must always be 1.
        self.batch = 1;
    }

    /// Samples the built graph carries at once (1 for inference).
    pub fn batch(&self) -> u32 {
        self.batch
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

    /// Encode the checkpoint the model would write, without touching a
    /// filesystem. This is the portable half: a wasm build has weights and no
    /// `fs`, and gets its bytes from the network.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, ModelError> {
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

        // The version is decided by what there is to write, not by what the
        // code can write: a run without an EMA still produces the exact V2 file
        // it produced before this trailer existed.
        let has_ema = self
            .layers
            .iter()
            .any(|layer| !layer.ema_state_buffers().is_empty());

        let mut bytes = Vec::new();
        bytes.extend_from_slice(if has_ema {
            CHECKPOINT_MAGIC_V3
        } else {
            CHECKPOINT_MAGIC_V2
        });
        bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for entry in entries {
            bytes.extend_from_slice(&entry.layer_index.to_le_bytes());
            bytes.extend_from_slice(&(entry.weights.len() as u32).to_le_bytes());
            bytes.extend_from_slice(bytemuck::cast_slice(&entry.weights));
            bytes.extend_from_slice(&(entry.bias.len() as u32).to_le_bytes());
            bytes.extend_from_slice(bytemuck::cast_slice(&entry.bias));
        }

        // Optimiser-state trailer.
        //
        // Adam's m and v are as much part of the training state as the weights:
        // dropping them on resume restarts the bias correction at t=1, where
        // m̂ = g and the very first update is a full ±lr on every weight, i.e. a
        // visible loss spike. They are therefore persisted, and the run's step
        // counter with them. SGD is stateless and writes the `none` tag, which
        // keeps its checkpoints byte-identical to V1 apart from the magic and
        // the 4-byte tag.
        self.append_optimizer_state(&mut bytes)?;

        // EMA trailer (V3 only).
        //
        // The averaged weights are the ones a V3 checkpoint is generated from,
        // so they are not an optional diagnostic: a file that dropped them
        // would silently sample from the raw iterate, which is precisely the
        // comparison `--raw-weights` exists to make deliberate.
        if has_ema {
            self.append_ema_state(&mut bytes)?;
        }

        Ok(bytes)
    }

    /// Write [`Model::checkpoint_bytes`] to disk, creating the parent directory.
    /// The filesystem lives in this wrapper and nowhere deeper.
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelError> {
        let bytes = self.checkpoint_bytes()?;
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

    /// Serialise the persistent optimiser state (see `save_checkpoint`).
    fn append_optimizer_state(&self, bytes: &mut Vec<u8>) -> Result<(), ModelError> {
        let stateful: Vec<(u32, &Layer)> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| !layer.opt_state_buffers().is_empty())
            .map(|(i, layer)| (i as u32, layer))
            .collect();

        if stateful.is_empty() {
            bytes.extend_from_slice(&OPT_STATE_NONE.to_le_bytes());
            return Ok(());
        }

        bytes.extend_from_slice(&OPT_STATE_ADAM.to_le_bytes());
        bytes.extend_from_slice(&self.optimizer_step.to_le_bytes());
        bytes.extend_from_slice(&(stateful.len() as u32).to_le_bytes());
        for (layer_index, layer) in stateful {
            let buffers = layer.opt_state_buffers();
            bytes.extend_from_slice(&layer_index.to_le_bytes());
            bytes.extend_from_slice(&(buffers.len() as u32).to_le_bytes());
            for buffer in buffers {
                let values =
                    read_back_f32(self.gpu.as_ref(), buffer, buffer.size()).ok_or_else(|| {
                        ModelError::CheckpointLayerMismatch {
                            layer_index: layer_index as usize,
                            message: "optimiser state buffer is not readable from GPU".to_string(),
                        }
                    })?;
                bytes.extend_from_slice(&(values.len() as u32).to_le_bytes());
                bytes.extend_from_slice(bytemuck::cast_slice(&values));
            }
        }
        Ok(())
    }

    /// Serialise the EMA trailer: the decay, then `[ema_weights, ema_bias]` per
    /// trainable layer, in the same shape as the optimiser trailer.
    fn append_ema_state(&self, bytes: &mut Vec<u8>) -> Result<(), ModelError> {
        let shadowed: Vec<(u32, &Layer)> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| !layer.ema_state_buffers().is_empty())
            .map(|(i, layer)| (i as u32, layer))
            .collect();

        if shadowed.is_empty() {
            bytes.extend_from_slice(&EMA_STATE_NONE.to_le_bytes());
            return Ok(());
        }

        bytes.extend_from_slice(&EMA_STATE_PRESENT.to_le_bytes());
        // The decay is recorded so the file says what produced it: a shadow
        // whose decay is unknown cannot be compared with another run's, and
        // `--resume` reads it back to warn when the resuming command line would
        // drop the average or average it differently.
        bytes.extend_from_slice(&self.ema.map(|e| e.decay).unwrap_or(0.0).to_le_bytes());
        bytes.extend_from_slice(&(shadowed.len() as u32).to_le_bytes());
        for (layer_index, layer) in shadowed {
            let buffers = layer.ema_state_buffers();
            bytes.extend_from_slice(&layer_index.to_le_bytes());
            bytes.extend_from_slice(&(buffers.len() as u32).to_le_bytes());
            for buffer in buffers {
                let values =
                    read_back_f32(self.gpu.as_ref(), buffer, buffer.size()).ok_or_else(|| {
                        ModelError::CheckpointLayerMismatch {
                            layer_index: layer_index as usize,
                            message: "EMA buffer is not readable from GPU".to_string(),
                        }
                    })?;
                bytes.extend_from_slice(&(values.len() as u32).to_le_bytes());
                bytes.extend_from_slice(bytemuck::cast_slice(&values));
            }
        }
        Ok(())
    }

    /// Parse the EMA trailer written by `append_ema_state`, without applying it.
    ///
    /// Parsing and applying are separated because the trailer sits *after* the
    /// weight entries in the stream while [`CheckpointWeights::Ema`] needs it
    /// *before* deciding what to write into the weight buffers.
    fn parse_ema_state(
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<Option<EmaPayload>, ModelError> {
        let tag = read_u32_le(bytes, offset)?;
        if tag == EMA_STATE_NONE {
            return Ok(None);
        }
        if tag != EMA_STATE_PRESENT {
            return Err(ModelError::InvalidCheckpointFormat {
                message: format!("unknown EMA state tag {tag}"),
            });
        }
        let decay = f32::from_le_bytes(
            bytes
                .get(*offset..offset.saturating_add(4))
                .ok_or_else(|| ModelError::InvalidCheckpointFormat {
                    message: "unexpected end of checkpoint while reading the EMA decay".to_string(),
                })?
                .try_into()
                .unwrap(),
        );
        *offset += 4;
        let layer_count = read_u32_le(bytes, offset)? as usize;
        let mut records = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let layer_index = read_u32_le(bytes, offset)? as usize;
            let buffer_count = read_u32_le(bytes, offset)? as usize;
            let mut section = Vec::with_capacity(buffer_count);
            for _ in 0..buffer_count {
                let len = read_u32_le(bytes, offset)? as usize;
                section.push(read_f32_vec_le(bytes, offset, len)?);
            }
            records.push((layer_index, section));
        }
        Ok(Some(EmaPayload { decay, records }))
    }

    /// Restore the optimiser state trailer written by `append_optimizer_state`.
    ///
    /// A mismatch between the checkpoint's optimiser and the model's is not an
    /// error: loading Adam state into an SGD run simply drops it, and loading
    /// an SGD (stateless) checkpoint into an Adam run leaves m = v = 0 and
    /// t = 0, i.e. a cold Adam start on warm weights.
    fn restore_optimizer_state(
        &mut self,
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<(), ModelError> {
        let tag = read_u32_le(bytes, offset)?;
        if tag == OPT_STATE_NONE {
            return Ok(());
        }
        if tag != OPT_STATE_ADAM {
            return Err(ModelError::InvalidCheckpointFormat {
                message: format!("unknown optimiser state tag {tag}"),
            });
        }

        let step = read_u64_le(bytes, offset)?;
        let layer_count = read_u32_le(bytes, offset)? as usize;
        let mut restored_any = false;
        for _ in 0..layer_count {
            let layer_index = read_u32_le(bytes, offset)? as usize;
            let buffer_count = read_u32_le(bytes, offset)? as usize;
            let targets: Vec<Arc<Buffer>> = self
                .layers
                .get(layer_index)
                .map(|layer| layer.opt_state_buffers().to_vec())
                .unwrap_or_default();
            // The whole record is consumed either way — a state section the
            // current model cannot use must still be walked past so the offset
            // stays aligned with the stream.
            let mut section = Vec::with_capacity(buffer_count);
            for _ in 0..buffer_count {
                let len = read_u32_le(bytes, offset)? as usize;
                section.push(read_f32_vec_le(bytes, offset, len)?);
            }
            if targets.len() != buffer_count {
                continue;
            }
            for (target, values) in targets.iter().zip(&section) {
                let expected = (target.size() as usize) / std::mem::size_of::<f32>();
                if values.len() != expected {
                    return Err(ModelError::CheckpointLayerMismatch {
                        layer_index,
                        message: format!(
                            "optimiser state length mismatch (expected {expected}, got {})",
                            values.len()
                        ),
                    });
                }
                self.gpu
                    .queue
                    .write_buffer(target.as_ref(), 0, bytemuck::cast_slice(values));
                restored_any = true;
            }
        }

        if restored_any {
            self.optimizer_step = step;
        }
        Ok(())
    }

    /// Read a checkpoint from disk. Thin wrapper over
    /// [`Model::load_checkpoint_bytes`]: `fs` is used here and nowhere deeper,
    /// so the inference path stays usable where there is no filesystem.
    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ModelError> {
        self.load_checkpoint_with(path, CheckpointWeights::Raw)
            .map(|_| ())
    }

    /// Read a checkpoint, choosing which of its weight sets lands in the model.
    ///
    /// Reports what the file held and what was used — see [`CheckpointLoad`].
    pub fn load_checkpoint_with<P: AsRef<Path>>(
        &mut self,
        path: P,
        source: CheckpointWeights,
    ) -> Result<CheckpointLoad, ModelError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|err| ModelError::CheckpointIo {
            path: path.display().to_string(),
            message: err.to_string(),
        })?;
        self.load_checkpoint_bytes_with(&bytes, source)
    }

    /// Restore weights (and the optimiser trailer, if present) from checkpoint
    /// bytes — the entry point a browser build uses, handed a `fetch` body.
    pub fn load_checkpoint_bytes(&mut self, bytes: &[u8]) -> Result<(), ModelError> {
        self.load_checkpoint_bytes_with(bytes, CheckpointWeights::Raw)
            .map(|_| ())
    }

    /// See [`Model::load_checkpoint_with`]. The whole stream is parsed before
    /// anything is written: the EMA trailer sits after the weight entries, and
    /// [`CheckpointWeights::Ema`] has to know about it before it can decide
    /// what those entries mean.
    pub fn load_checkpoint_bytes_with(
        &mut self,
        bytes: &[u8],
        source: CheckpointWeights,
    ) -> Result<CheckpointLoad, ModelError> {
        if !self.state.is_build {
            return Err(ModelError::InvalidCheckpointFormat {
                message: "model must be built before loading checkpoint".to_string(),
            });
        }

        // Version, read off the magic. V1 has no trailer at all, V2 an
        // optimiser trailer, V3 that plus the weight average.
        let (has_optimizer_trailer, has_ema_trailer) = match bytes.get(..CHECKPOINT_MAGIC_LEN) {
            Some(magic) if magic == CHECKPOINT_MAGIC_V3 => (true, true),
            Some(magic) if magic == CHECKPOINT_MAGIC_V2 => (true, false),
            Some(magic) if magic == CHECKPOINT_MAGIC_V1 => (false, false),
            _ => {
                return Err(ModelError::InvalidCheckpointFormat {
                    message: "missing or invalid checkpoint magic".to_string(),
                });
            }
        };

        struct LoadedEntry {
            layer_index: usize,
            weights: Vec<f32>,
            bias: Vec<f32>,
        }

        let mut offset = CHECKPOINT_MAGIC_LEN;
        let entry_count = read_u32_le(bytes, &mut offset)? as usize;
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let layer_index = read_u32_le(bytes, &mut offset)? as usize;
            let weight_len = read_u32_le(bytes, &mut offset)? as usize;
            let weights = read_f32_vec_le(bytes, &mut offset, weight_len)?;
            let bias_len = read_u32_le(bytes, &mut offset)? as usize;
            let bias = read_f32_vec_le(bytes, &mut offset, bias_len)?;
            entries.push(LoadedEntry {
                layer_index,
                weights,
                bias,
            });
        }

        // The optimiser trailer is applied where it is parsed — nothing
        // downstream depends on it. The EMA trailer is not, hence the split.
        let optimizer_trailer_at = offset;
        if has_optimizer_trailer {
            skip_optimizer_state(bytes, &mut offset)?;
        }
        let ema = if has_ema_trailer {
            Self::parse_ema_state(bytes, &mut offset)?
        } else {
            None
        };

        if offset != bytes.len() {
            return Err(ModelError::InvalidCheckpointFormat {
                message: "checkpoint has trailing bytes".to_string(),
            });
        }

        let use_ema = matches!(source, CheckpointWeights::Ema) && ema.is_some();

        for entry in &entries {
            let layer_index = entry.layer_index;
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
            if entry.weights.len() != expected_weights {
                return Err(ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: format!(
                        "weights length mismatch (expected {expected_weights}, got {})",
                        entry.weights.len()
                    ),
                });
            }
            if entry.bias.len() != expected_bias {
                return Err(ModelError::CheckpointLayerMismatch {
                    layer_index,
                    message: format!(
                        "bias length mismatch (expected {expected_bias}, got {})",
                        entry.bias.len()
                    ),
                });
            }

            // The averaged set, when the file has one, is validated against the
            // very same lengths — an EMA section of the wrong shape is a broken
            // file, not a reason to fall back in silence.
            let averaged = match ema.as_ref().and_then(|p| p.weights_and_bias_for(layer_index)) {
                Some((weights, bias)) => {
                    if weights.len() != expected_weights || bias.len() != expected_bias {
                        return Err(ModelError::CheckpointLayerMismatch {
                            layer_index,
                            message: format!(
                                "EMA length mismatch (expected {expected_weights}/{expected_bias}, \
                                 got {}/{})",
                                weights.len(),
                                bias.len()
                            ),
                        });
                    }
                    Some((weights, bias))
                }
                None => None,
            };

            let (weights, bias) = match (use_ema, averaged) {
                (true, Some((weights, bias))) => (weights, bias),
                _ => (&entry.weights, &entry.bias),
            };
            self.gpu
                .queue
                .write_buffer(weights_buf.as_ref(), 0, bytemuck::cast_slice(weights));
            self.gpu
                .queue
                .write_buffer(bias_buf.as_ref(), 0, bytemuck::cast_slice(bias));

            // Fill this run's shadow, if it keeps one. Whatever landed in the
            // weight buffers is what the shadow must describe: adopting the
            // average and then averaging towards the raw iterate would make the
            // next checkpoint a blend of two different runs.
            let shadow = layer.ema_state_buffers();
            if shadow.len() == 2 {
                self.gpu
                    .queue
                    .write_buffer(shadow[0].as_ref(), 0, bytemuck::cast_slice(weights));
                self.gpu
                    .queue
                    .write_buffer(shadow[1].as_ref(), 0, bytemuck::cast_slice(bias));
                if !use_ema && let Some((avg_w, avg_b)) = averaged {
                    self.gpu
                        .queue
                        .write_buffer(shadow[0].as_ref(), 0, bytemuck::cast_slice(avg_w));
                    self.gpu
                        .queue
                        .write_buffer(shadow[1].as_ref(), 0, bytemuck::cast_slice(avg_b));
                }
            }
        }

        if has_optimizer_trailer {
            let mut opt_offset = optimizer_trailer_at;
            self.restore_optimizer_state(bytes, &mut opt_offset)?;
        }

        Ok(CheckpointLoad {
            carries_ema: ema.is_some(),
            used_ema: use_ema,
            ema_decay: ema.as_ref().map(|payload| payload.decay),
        })
    }

    /// The run's global optimiser step counter `t` (0 before the first update).
    pub fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    #[cfg(test)]
    pub(crate) fn reset_optimizer_step(&mut self) {
        self.optimizer_step = 0;
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
            // One sample: `predict` writes the first slice and reads the first
            // slice back. On a training model whose buffers are sized for a
            // full batch (this is what `probe_diffusion` does), dispatching the
            // whole grid would compute 15 more samples out of stale memory and
            // throw them away.
            layer.encode_pass_with_batch(&mut encoder, 1);
        }
        self.gpu.queue.submit([encoder.finish()]);

        self.read_last_output()
    }

    fn build_forwards(&mut self) -> Result<(), ModelError> {
        let mut last_output: Option<Arc<Buffer>> = None;
        let mut saved_output_buffers: HashMap<String, Arc<Buffer>> = HashMap::new();
        let init = self.weight_init;
        let batch = self.batch;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            // Before create_buffers: it is what sizes the activation buffers.
            layer.batch = batch;
            last_output = Some(layer.create_buffers(
                &self.gpu,
                last_output,
                &saved_output_buffers,
                init,
                layer_index,
            )?);
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
                for buffer in opt.owned_buffers() {
                    add_unique(&mut total, &mut seen, buffer);
                }
            }
            if let Some(ema) = &layer.ema_pass {
                for buffer in ema.owned_buffers() {
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
        // The LAST sample of the batch, not the first.
        //
        // Before batching, the reported loss was the one left in the buffer by
        // the last sample submitted (`is_last && report_last_loss` in
        // diffusion.rs). Keeping that exact definition is what makes the paired
        // old-vs-new loss trajectories comparable at all — and since the
        // forward pass contains no reduction over the batch axis, the number
        // must come out bit-identical.
        let per_sample = loss_layer.ty.get_dim_output().bytes_size() as u64;
        let offset = per_sample * (self.batch.saturating_sub(1)) as u64;
        let Some(loss_terms) = read_back_f32_at(
            self.gpu.as_ref(),
            &loss_layer.buffers.forward[2],
            offset,
            per_sample,
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
        // Same slice as the blocking path: the batch's last sample.
        let source_offset = size_bytes * (self.batch.saturating_sub(1)) as u64;

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loss_readback_staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            loss_terms_buf.as_ref(),
            source_offset,
            &staging,
            0,
            size_bytes,
        );
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
                let loss_terms =
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let bytes = pending.staging.slice(..).get_mapped_range();
                        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
                    })) {
                        Ok(loss_terms) => loss_terms,
                        Err(_) => {
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                pending.staging.unmap();
                            }));
                            self.disable_loss_readback(
                                "non-blocking loss readback mapped range became invalid",
                            );
                            return None;
                        }
                    };
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pending.staging.unmap();
                }));
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

/// Walk past the optimiser trailer without applying it, leaving `offset` on the
/// byte after it.
///
/// The loader has to reach the EMA trailer, which sits behind this one, before
/// it can decide what to write — so the optimiser section is walked first and
/// applied afterwards from the offset it started at. Two readers of one layout
/// would drift; this one shares `restore_optimizer_state`'s field order and
/// nothing else, and the round-trip tests fail the moment they disagree.
fn skip_optimizer_state(bytes: &[u8], offset: &mut usize) -> Result<(), ModelError> {
    let tag = read_u32_le(bytes, offset)?;
    if tag == OPT_STATE_NONE {
        return Ok(());
    }
    if tag != OPT_STATE_ADAM {
        return Err(ModelError::InvalidCheckpointFormat {
            message: format!("unknown optimiser state tag {tag}"),
        });
    }
    let _step = read_u64_le(bytes, offset)?;
    let layer_count = read_u32_le(bytes, offset)? as usize;
    for _ in 0..layer_count {
        let _layer_index = read_u32_le(bytes, offset)?;
        let buffer_count = read_u32_le(bytes, offset)? as usize;
        for _ in 0..buffer_count {
            let len = read_u32_le(bytes, offset)? as usize;
            let _ = read_f32_vec_le(bytes, offset, len)?;
        }
    }
    Ok(())
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

fn read_u64_le(bytes: &[u8], offset: &mut usize) -> Result<u64, ModelError> {
    let end = offset.saturating_add(8);
    if end > bytes.len() {
        return Err(ModelError::InvalidCheckpointFormat {
            message: "unexpected end of checkpoint while reading u64".to_string(),
        });
    }
    let value = u64::from_le_bytes(bytes[*offset..end].try_into().unwrap());
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
