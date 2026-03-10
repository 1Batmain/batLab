use crate::gpu_context::GpuContext;
use crate::model::error::ModelError;
use crate::model::layer::Layer;
use crate::model::layer_types::{LayerType, LayerTypes};
use crate::model::types::Dim3;
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

#[derive(Debug)]
pub struct Infer;

pub struct Training {
    pub(crate) lr: f32,
    pub(crate) batch_size: u32,
    pub(crate) optimizer: Layer,
    pub(crate) loss: Layer,
}

#[derive(Debug)]
pub struct ModelState {
    pub(crate) is_build: bool,
}

#[derive(Debug)]
pub struct Model<State = Infer> {
    pub(crate) gpu: Arc<GpuContext>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) training: Option<State>,
    pub(crate) state: ModelState,
}

impl Model<Infer> {
    pub fn build_model(&mut self) {
        // Let each layer create :
        // - buffers (take the previous layer's output as input)
        // - pipeline
        // - bind group
        if (self.state.is_build) {
            self.clear();
        }
        self.build_forwards();
    }

    fn run(&mut self, input: &[f32]) -> Vec<f32> {
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        for layer in &self.layers {
            layer.encode_pass(&mut encoder);
        }
        self.gpu.queue.write_buffer(
            self.layers
                .first()
                .expect("input layer that takes an input, actually")
                .buffers[0]
                .as_ref(),
            0,
            bytemuck::cast_slice(input),
        );
        self.gpu.queue.submit([encoder.finish()]);

        let last_layer = self.layers.last().expect("No layers to output from");
        let result = self.read_back_f32_buffer(
            last_layer
                .buffers
                .last()
                .expect("no output buffer on last layer"),
            last_layer.ty.get_dim_output().bytes_size() as u64,
        );
        result
    }

    pub async fn infer_batch(&mut self, input: Vec<f32>) -> Vec<f32> {
        let first_layer = self
            .layers
            .first()
            .expect("infer requires at least one layer");
        let expected_input_len = first_layer.ty.get_dim_input().length() as usize;
        if input.len() == expected_input_len {
            return self.run(&input);
        }

        if input.len() % expected_input_len == 0 {
            let image_count = input.len() / expected_input_len;
            println!(
                "detected multiple images in infer input: {} images, running sequential inference",
                image_count
            );

            let mut all_outputs = Vec::new();
            for image in input.chunks(expected_input_len) {
                let output = self.run(image);
                all_outputs.extend(output);
            }
            return all_outputs;
        }

        panic!(
            "invalid input length for infer: expected {} (or a multiple), got {}",
            expected_input_len,
            input.len()
        );
    }
}

impl Model<Training> {
    pub fn build(&mut self) {
        // Let each layer create :
        // - buffers (take the previous layer's output as input)
        // - pipeline
        // - bind group
        self.build_forwards();
        let mut last_output: Option<Arc<Buffer>> =
            Some(self.layers.last().unwrap().buffers.last().unwrap().clone());
        let training = self.training.as_mut().unwrap();
        let layer = &mut training.loss;
        last_output = Some(layer.create_buffers(&self.gpu, last_output));
        layer.set_pipeline(&self.gpu.device);
        layer.set_bind_group(&self.gpu.device);

        for layer in self.layers.iter_mut().rev() {
            last_output = Some(layer.create_back_buffers(&self.gpu, last_output));
            layer.set_back_pipeline(&self.gpu.device);
            layer.set_back_bind_group(&self.gpu.device);
        }
    }
    fn run(&mut self, input: &[f32]) -> Vec<f32> {
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        for layer in &self.layers {
            layer.encode_pass(&mut encoder);
        }
        self.gpu.queue.write_buffer(
            self.layers
                .first()
                .expect("input layer that takes an input, actually")
                .buffers[0]
                .as_ref(),
            0,
            bytemuck::cast_slice(input),
        );

        let loss = &mut self.training.as_mut().unwrap().loss;
        loss.encode_pass(&mut encoder);

        for layer in &self.layers.rev() {
            layer.encode_back_pass(&mut encoder);
        }
        for layer in &self.layers {
            layer.encode_optimizer_pass(&mut encoder);
        }

        self.gpu.queue.submit([encoder.finish()]);

        let last_layer = self.layers.last().expect("No layers to output from");
        let result = self.read_back_f32_buffer(
            last_layer
                .buffers
                .last()
                .expect("no output buffer on last layer"),
            last_layer.ty.get_dim_output().bytes_size() as u64,
        );
        result
    }

impl<State> Model<State> {
    pub async fn new(gpu: Arc<GpuContext>) -> Self {
        Self {
            gpu: gpu.clone(),
            layers: Vec::new(),
            training: None,
            state: ModelState { is_build: false },
        }
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|layer| layer.clear());
        self.layers.clear();
        self.state.is_build = false;
    }

    pub fn training_mode(&mut self, training: Option<State>) {
        self.clear();
        self.training = training;
        self.state.is_build = false;
    }

    fn build_forwards(&mut self) {
        // Let each layer create :
        // - buffers (take the previous layer's output as input)
        // - pipeline
        // - bind group
        if (self.state.is_build) {
            self.clear();
        }
        let mut last_output: Option<Arc<Buffer>> = None;
        for layer in &mut self.layers {
            last_output = Some(layer.create_buffers(&self.gpu, last_output));
            layer.set_pipeline(&self.gpu.device);
            layer.set_bind_group(&self.gpu.device);
        }
    }

    pub fn add_layer(&mut self, spec: LayerTypes) -> Result<(), ModelError> {
        let mut last_output: Option<Dim3> = None;
        if self.layers.len() > 0 {
            last_output = Some(self.layers.last().unwrap().ty.get_dim_output());
        }
        let layer = Layer::new(&self.gpu.device, spec, last_output)?;
        self.layers.push(layer);
        Ok(())
    }
    fn read_back_f32_buffer(&self, source: &Buffer, size_bytes: u64) -> Vec<f32> {
        if size_bytes == 0 {
            return Vec::new();
        }

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        let staging_buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(source, 0, &staging_buffer, 0, size_bytes);
        self.gpu.queue.submit([encoder.finish()]);

        let buffer_slice = staging_buffer.slice(..);
        let (gpu, cpu) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = gpu.send(result);
        });

        if let Err(error) = self.gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
            panic!("failed to poll device while reading buffer: {}", error);
        }

        let map_result = pollster::block_on(async { cpu.await });
        match map_result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => panic!("failed to map readback buffer: {}", error),
            Err(_) => panic!("failed to receive map callback for readback buffer"),
        }

        let values = {
            let bytes = buffer_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };
        staging_buffer.unmap();
        values
    }
}
