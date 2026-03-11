use crate::gpu_context::GpuContext;
use crate::model::error::ModelError;
use crate::model::layer::Layer;
use crate::model::layer_types::{LayerType, LayerTypes, LossMethod, LossType};

use std::sync::Arc;
use wgpu::Buffer;

// ---------------------------------------------------------------------------
// State markers
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Infer;

pub struct Training {
    pub lr: f32,
    pub batch_size: u32,
    pub(crate) loss_method: LossMethod,
    /// Populated during build().
    pub(crate) loss: Option<Layer>,
}

#[derive(Debug)]
pub struct ModelState {
    pub(crate) is_build: bool,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Model<State = Infer> {
    pub(crate) gpu: Arc<GpuContext>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) training: Option<State>,
    pub(crate) state: ModelState,
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

impl Model<Infer> {
    pub fn build_model(&mut self) {
        if self.state.is_build {
            self.clear();
        }
        self.build_forwards();
        self.state.is_build = true;
    }

    fn run(&mut self, input: &[f32]) -> Vec<f32> {
        // Write input data before any compute work is submitted.
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

        let last = self.layers.last().expect("at least one layer required");
        self.read_back_f32_buffer(
            last.buffers.forward.last().expect("no output buffer"),
            last.ty.get_dim_output().bytes_size() as u64,
        )
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
            return self.run(&input);
        }
        if input.len() % expected == 0 {
            let count = input.len() / expected;
            println!("running sequential inference for {count} samples");
            let mut out = Vec::new();
            for chunk in input.chunks(expected) {
                out.extend(self.run(chunk));
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
            training: Some(Training {
                lr,
                batch_size,
                loss_method,
                loss: None,
            }),
            state: ModelState { is_build: false },
        }
    }

    /// Build all forward + backward + SGD passes in the correct order.
    pub fn build(&mut self) {
        if self.state.is_build {
            self.clear();
        }

        // 1. Forward passes for all network layers.
        self.build_forwards();

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
        let loss_grad_out = loss_layer.create_buffers(&self.gpu, Some(last_fwd_output));
        loss_layer.set_pipeline(&self.gpu.device);
        loss_layer.set_bind_group(&self.gpu.device);

        // 3. Backward passes in reverse layer order.
        //    Each layer receives the previous layer's grad_input as its grad_output.
        let lr = self.training.as_ref().unwrap().lr;
        let mut incoming_grad = loss_grad_out;

        for layer in self.layers.iter_mut().rev() {
            incoming_grad = layer.create_back_buffers(&self.gpu, Some(incoming_grad));
            layer.init_back_shader(&self.gpu.device);
            layer.set_back_pipeline(&self.gpu.device);
            layer.set_back_bind_group(&self.gpu.device);
            if layer.ty.has_weights() {
                layer.create_opt_pass(&self.gpu, lr);
            }
        }

        self.training.as_mut().unwrap().loss = Some(loss_layer);
        self.state.is_build = true;
    }

    /// Run one training step: forward → loss/gradient → backward → SGD update.
    /// Call build() before the first train_step.
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) {
        debug_assert!(self.state.is_build, "call build() before train_step()");

        // Write CPU data before any GPU work is encoded.
        self.gpu.queue.write_buffer(
            self.layers.first().unwrap().buffers.forward[0].as_ref(),
            0,
            bytemuck::cast_slice(input),
        );
        // loss forward buffers: [0]=model_result (shared), [1]=target, [2]=grad_output
        let loss_buf = self
            .training
            .as_ref()
            .unwrap()
            .loss
            .as_ref()
            .expect("call build() before train_step()")
            .buffers
            .forward[1]
            .clone();
        self.gpu
            .queue
            .write_buffer(loss_buf.as_ref(), 0, bytemuck::cast_slice(target));

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());

        // Forward
        for layer in &self.layers {
            layer.encode_pass(&mut encoder);
        }
        // Loss / initial gradient computation
        self.training
            .as_ref()
            .unwrap()
            .loss
            .as_ref()
            .unwrap()
            .encode_pass(&mut encoder);

        // Backward (reverse order)
        for layer in self.layers.iter().rev() {
            layer.encode_back_pass(&mut encoder);
        }

        // SGD weight updates
        for layer in &self.layers {
            layer.encode_opt_pass(&mut encoder);
        }

        self.gpu.queue.submit([encoder.finish()]);
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
            training: None,
            state: ModelState { is_build: false },
        }
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| l.clear());
        self.layers.clear();
        self.state.is_build = false;
    }

    pub fn training_mode(&mut self, training: Option<State>) {
        self.clear();
        self.training = training;
    }

    fn build_forwards(&mut self) {
        let mut last_output: Option<Arc<Buffer>> = None;
        for layer in &mut self.layers {
            last_output = Some(layer.create_buffers(&self.gpu, last_output));
            layer.set_pipeline(&self.gpu.device);
            layer.set_bind_group(&self.gpu.device);
        }
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

    pub(crate) fn read_back_f32_buffer(&self, source: &Buffer, size_bytes: u64) -> Vec<f32> {
        if size_bytes == 0 {
            return Vec::new();
        }
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size_bytes);
        self.gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if let Err(e) = self.gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
            panic!("device poll failed during readback: {e}");
        }
        match pollster::block_on(async { rx.await }) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => panic!("buffer map failed: {e}"),
            Err(_) => panic!("map callback channel dropped"),
        }
        let values = {
            let bytes = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };
        staging.unmap();
        values
    }
}
