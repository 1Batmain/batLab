use crate::gpu_context::GpuContext;
use crate::model::debug::{LayerDebugView, read_back_f32};
use crate::model::error::ModelError;
use crate::model::layer::Layer;
use crate::model::layer_types::{ConcatType, LayerType, LayerTypes, LossMethod, LossType};
use crate::model::types::Dim3;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use wgpu::Buffer;

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
        self.encode_train_graph(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.read_last_loss()
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
        self.encode_train_graph(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
    }

    fn encode_train_graph(&self, encoder: &mut wgpu::CommandEncoder) {
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

        // SGD weight updates
        for layer in &self.layers {
            layer.encode_opt_pass(encoder);
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
        }
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| l.clear());
        self.layers.clear();
        self.loss_layer = None;
        self.state.is_build = false;
        self.saved_outputs.clear();
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

    pub fn read_last_loss(&self) -> f32 {
        let loss_layer = self
            .loss_layer
            .as_ref()
            .expect("loss layer is only available in training mode");
        let loss_terms = read_back_f32(
            self.gpu.as_ref(),
            &loss_layer.buffers.forward[2],
            loss_layer.ty.get_dim_output().bytes_size() as u64,
        )
        .expect("failed to read loss buffer");
        if loss_terms.is_empty() {
            0.0
        } else {
            loss_terms.iter().sum::<f32>() / loss_terms.len() as f32
        }
    }
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
}
