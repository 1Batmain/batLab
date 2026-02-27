use std::sync::Arc;

use wgpu::{BindGroup, Buffer, BufferDescriptor, BufferUsages, ComputePipeline, Device, ShaderModule};

use crate::gpu_context::GpuContext;
use crate::model::layer::Layer;
use crate::model::persistence::{SavedLayer, SavedLayerArchitecture, SavedModel, SavedTrainingSpec};
use crate::model::spec::{ActivationLayerSpec, ConvolutionLayerSpec, LayerSpec};
use crate::model::types::Optimizer;

/// Snapshot of model state useful for external Visualizers or logging.
#[derive(Debug, Clone, Copy)]
pub struct ModelVisualState {
    pub layer_count: usize,
    pub has_training_spec: bool,
    pub training_initialized: bool,
    pub has_loss_pipeline: bool,
    pub infer_revision: u64,
    pub train_revision: u64,
}

#[derive(Debug)]
pub struct TrainingSpec {
    pub lr: f32,
    pub batch_size: u32,
    pub optimizer: Optimizer,
}

#[derive(Debug)]
pub struct Model {
    pub (crate) gpu: Arc<GpuContext>,
                layers: Vec<Layer>,
                specs: Vec<LayerSpec>,  // Store original layer specs for building uniforms
                training: Option<TrainingSpec>,
                loss_shader: Option<ShaderModule>,
                loss_pipeline: Option<ComputePipeline>,
                loss_bind_group: Option<BindGroup>,
                loss_target: Option<Arc<Buffer>>,
                training_initialized: bool,
                infer_revision: u64,
                train_revision: u64,
}

impl Model {

    pub async fn new(gpu: Arc<GpuContext>, training: Option<TrainingSpec>) -> Self {
        Self {
            gpu: gpu.clone(),
            layers: Vec::new(),
            specs: Vec::new(),
            training,
            loss_shader: None,
            loss_pipeline: None,
            loss_bind_group: None,
            loss_target: None,
            training_initialized: false,
            infer_revision: 0,
            train_revision: 0,
        }
    }

    /// Borrow slice of layers for external inspection.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Get a specific layer by index (panics if out of range).
    pub fn layer(&self, idx: usize) -> &Layer {
        &self.layers[idx]
    }

    /// Returns a snapshot of the model's current visual state.
    pub fn visual_state(&self) -> ModelVisualState {
        ModelVisualState {
            layer_count: self.layers.len(),
            has_training_spec: self.training.is_some(),
            training_initialized: self.training_initialized,
            has_loss_pipeline: self.loss_pipeline.is_some(),
            infer_revision: self.infer_revision,
            train_revision: self.train_revision,
        }
    }




    pub async fn load(gpu: Arc<GpuContext>, name: &str) -> Self {
        let content = std::fs::read_to_string(name).expect("failed to read model file");
        let saved_model: SavedModel =
            serde_json::from_str(&content).expect("failed to parse model JSON");

        if saved_model.version != 1 {
            panic!("unsupported model version: {}", saved_model.version);
        }

        let training = saved_model.training.as_ref().map(|training| TrainingSpec {
            lr: training.lr,
            batch_size: training.batch_size,
            optimizer: training.optimizer,
        });

        let mut model = Model::new(gpu, training).await;

        for saved_layer in &saved_model.layers {
            let spec = match &saved_layer.architecture {
                SavedLayerArchitecture::Convolution {
                    nb_kernel,
                    dim_kernel,
                    stripe,
                    mode,
                    dim_input,
                    ..
                } => LayerSpec::Convolution(ConvolutionLayerSpec {
                    nb_kernel: *nb_kernel,
                    dim_kernel: *dim_kernel,
                    stripe: *stripe,
                    mode: *mode,
                    dim_input: Some(*dim_input),
                }),
                SavedLayerArchitecture::Activation {
                    method,
                    dim_input,
                    ..
                } => LayerSpec::Activation(ActivationLayerSpec {
                    method: *method,
                    dim_input: Some(*dim_input),
                }),
            };
            model.add_layer(spec);
        }

        model.build_model();

        for (layer, saved_layer) in model.layers.iter_mut().zip(saved_model.layers.into_iter()) {
            let expected_weights = layer.weight_size();
            if saved_layer.weights.len() != expected_weights {
                panic!(
                    "invalid weights length: expected {}, got {}",
                    expected_weights,
                    saved_layer.weights.len()
                );
            }

            let expected_bias = layer.bias_size();
            if saved_layer.bias.len() != expected_bias {
                panic!(
                    "invalid bias length: expected {}, got {}",
                    expected_bias,
                    saved_layer.bias.len()
                );
            }

            layer.set_cpu_params(
                Arc::new(saved_layer.weights.clone()),
                Arc::new(saved_layer.bias.clone()),
            );

            model.gpu.queue.write_buffer(
                layer.gpu_weights(),
                0,
                bytemuck::cast_slice(&saved_layer.weights),
            );
            model.gpu.queue.write_buffer(
                layer.gpu_bias(),
                0,
                bytemuck::cast_slice(&saved_layer.bias),
            );
        }

        model
    }

    fn create_buffer(device: &Device, size: u64) -> Arc<Buffer> {
        let buffer_size = size.max(4);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("buffer"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    fn create_loss_pipeline(device: &Device, shader: &ShaderModule) -> ComputePipeline {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("loss_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("loss_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("loss_pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Default::default(),
        })
    }

    fn create_loss_bind_group(
        device: &Device,
        pipeline: &ComputePipeline,
        prediction: &Buffer,
        target: &Buffer,
        grad_output: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prediction.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_output.as_entire_binding(),
                },
            ],
        })
    }

    fn loss_pipeline(&self) -> &ComputePipeline {
        self.loss_pipeline
            .as_ref()
            .expect("loss pipeline not initialized: call train() to initialize training resources")
    }

    fn loss_bind_group(&self) -> &BindGroup {
        self.loss_bind_group
            .as_ref()
            .expect("loss bind_group not initialized: call train() to initialize training resources")
    }

    fn loss_target(&self) -> &Buffer {
        self.loss_target
            .as_ref()
            .expect("loss target buffer not initialized: call train() to initialize training resources")
            .as_ref()
    }

    fn initialize_training(&mut self) {
        if self.training_initialized {
            return;
        }

        if self.training.is_none() {
            panic!("cannot initialize training: training spec not set");
        }

        let mut grad_inputs: Vec<Arc<Buffer>> = Vec::with_capacity(self.layers.len());
        let mut grad_outputs: Vec<Arc<Buffer>> = Vec::with_capacity(self.layers.len());
        let mut grad_weights: Vec<Arc<Buffer>> = Vec::with_capacity(self.layers.len());
        let mut grad_bias: Vec<Arc<Buffer>> = Vec::with_capacity(self.layers.len());

        {
            let device = &self.gpu.device;
            for layer in self.layers.iter() {
                let output_bytes = layer.output_size_bytes();
                let input_bytes = layer.input_size_bytes();

                grad_inputs.push(Self::create_buffer(device, input_bytes));
                grad_outputs.push(Self::create_buffer(device, output_bytes));
                grad_weights.push(Self::create_buffer(device, layer.weight_size_bytes()));
                grad_bias.push(Self::create_buffer(device, layer.bias_size_bytes()));
            }
        }

        {
            let device = &self.gpu.device;
            let num_layers = self.layers.len();
            for (idx, layer) in self.layers.iter_mut().enumerate() {
                let spec = match &layer.saved_architecture() {
                    SavedLayerArchitecture::Convolution {
                        nb_kernel,
                        dim_kernel,
                        stripe,
                        mode,
                        dim_input,
                        ..
                    } => LayerSpec::Convolution(ConvolutionLayerSpec {
                        nb_kernel: *nb_kernel,
                        dim_kernel: *dim_kernel,
                        stripe: *stripe,
                        mode: *mode,
                        dim_input: Some(*dim_input),
                    }),
                    SavedLayerArchitecture::Activation {
                        method,
                        dim_input,
                        ..
                    } => LayerSpec::Activation(ActivationLayerSpec {
                        method: *method,
                        dim_input: Some(*dim_input),
                    }),
                };
                
                let grad_output = if idx + 1 < num_layers {
                    grad_inputs[idx + 1].clone()
                } else {
                    grad_outputs[idx].clone()
                };
                
                layer.set_training_buffers(
                    grad_output,
                    grad_inputs[idx].clone(),
                    grad_weights[idx].clone(),
                    grad_bias[idx].clone(),
                );
                layer.set_backprop_pipeline(device, &spec);
                layer.set_backprop_bind_group(device);
            }
        }

        let (prediction, grad_output, output_bytes) = {
            let last_layer = self
                .layers
                .last()
                .expect("initialize_training requires at least one layer");
            (
                last_layer.output_arc(),
                last_layer.grad_output_arc(),
                last_layer.output_size_bytes(),
            )
        };

        self.loss_shader = {
            let device = &self.gpu.device;
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("loss_mse"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader/loss_mse.wgsl"))),
            }))
        };

        let loss_pipeline = {
            let device = &self.gpu.device;
            Self::create_loss_pipeline(device, self.loss_shader.as_ref().expect("loss shader must be initialized"))
        };

        self.loss_target = {
            let device = &self.gpu.device;
            Some(Self::create_buffer(device, output_bytes))
        };

        let loss_bind_group = {
            let device = &self.gpu.device;
            Self::create_loss_bind_group(
                device,
                &loss_pipeline,
                prediction.as_ref(),
                self.loss_target
                    .as_ref()
                    .expect("loss target buffer must be initialized")
                    .as_ref(),
                grad_output.as_ref(),
            )
        };

        self.loss_pipeline = Some(loss_pipeline);
        self.loss_bind_group = Some(loss_bind_group);
        self.training_initialized = true;
    }

    pub fn build_model(&mut self) {
        let input_bytes = self
            .layers
            .first()
            .expect("build_model requires at least one layer")
            .input_size_bytes();

        let mut last_output = {
            let device = &self.gpu.device;
            Self::create_buffer(device, input_bytes)
        };

        {
            let device = &self.gpu.device;
            let queue = &self.gpu.queue;
            for (layer, spec) in self.layers.iter_mut().zip(self.specs.iter()) {
                let output_bytes = layer.output_size_bytes();

                let weights = Self::create_buffer(device, layer.weight_size_bytes());
                let bias = Self::create_buffer(device, layer.bias_size_bytes());
                let output = Self::create_buffer(device, output_bytes);
                layer.set_gpu_buffers(last_output, weights, bias, output);
                last_output = layer.output_arc();

                // Create spec uniform for convolution layers
                if let LayerSpec::Convolution(conv_spec) = spec {
                    layer.create_spec_uniform(device, queue, conv_spec);
                }

                layer.set_pipeline(device);
                layer.set_bind_group(device);
            }
        }
    }

    fn infer_single(&mut self, input: &[f32]) -> Vec<f32> {
        let first_layer = self
            .layers
            .first()
            .expect("infer requires at least one layer");
        let last_layer = self
            .layers
            .last()
            .expect("infer requires at least one layer");

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        for layer in self.layers.iter() {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(layer.pipeline());
            pass.set_bind_group(0, layer.bind_group(), &[]);
            pass.dispatch_workgroups(layer.num_workgroups(), 1, 1);
        }

        self.gpu.queue
            .write_buffer(first_layer.gpu_input(), 0, bytemuck::cast_slice(input));
        self.gpu.queue.submit([encoder.finish()]);

        let result = self.read_back_f32_buffer(last_layer.gpu_output(), last_layer.output_size_bytes());
        self.infer_revision = self.infer_revision.saturating_add(1);

        result
    }

    pub async fn infer(&mut self, input: Vec<f32>) -> Vec<f32> {
        let first_layer = self
            .layers
            .first()
            .expect("infer requires at least one layer");
        let expected_input_len = (first_layer.input_size_bytes() / 4) as usize;
        if input.len() == expected_input_len {
            return self.infer_single(&input);
        }

        if input.len() % expected_input_len == 0 {
            let image_count = input.len() / expected_input_len;
            println!(
                "detected multiple images in infer input: {} images, running sequential inference",
                image_count
            );

            let mut all_outputs = Vec::new();
            for image in input.chunks(expected_input_len) {
                let output = self.infer_single(image);
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

    pub fn add_layer(&mut self, mut spec: LayerSpec) {
        if !self.layers.is_empty() {
            let prev_output = self
                .layers
                .last()
                .expect("non-empty layer list must have a last element")
                .dim_output();
            match &mut spec {
                LayerSpec::Convolution(conv_spec) => {
                    conv_spec.dim_input = Some(prev_output);
                }
                LayerSpec::Activation(activation_spec) => {
                    activation_spec.dim_input = Some(prev_output);
                }
            }
        }
        self.specs.push(spec);  // Store the spec for later use
        self.layers.push(Layer::new(&self.gpu.device, spec));
    }

    pub fn train(&mut self, input: Vec<f32>, target: Vec<f32>) {
        if !self.training_initialized {
            self.initialize_training();
        }

        let training = self
            .training
            .as_ref()
            .expect("training spec must be set before calling train");

        let last_layer = self
            .layers
            .last()
            .expect("train requires at least one layer");

        if target.len() as u64 * 4 != last_layer.output_size_bytes() {
            panic!("target length does not match model output size");
        }

        self.gpu.queue.write_buffer(
            self.layers
                .first()
                .expect("train requires at least one layer")
                .gpu_input(),
            0,
            bytemuck::cast_slice(&input),
        );
        self.gpu.queue
            .write_buffer(self.loss_target(), 0, bytemuck::cast_slice(&target));

        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());

        for layer in self.layers.iter() {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(layer.pipeline());
            pass.set_bind_group(0, layer.bind_group(), &[]);
            pass.dispatch_workgroups(layer.num_workgroups(), 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(self.loss_pipeline());
            pass.set_bind_group(0, self.loss_bind_group(), &[]);
            let total = (last_layer.output_size_bytes() / 4) as u32;
            let dispatch = total.div_ceil(64);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }

        for layer in self.layers.iter().rev() {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(layer.backprop_pipeline());
            pass.set_bind_group(0, layer.backprop_bind_group(), &[]);
            pass.dispatch_workgroups(layer.num_workgroups(), 1, 1);
        }

        self.gpu.queue.submit([encoder.finish()]);
        self.train_revision = self.train_revision.saturating_add(1);

        match training.optimizer {
            Optimizer::Sgd => {
                // TODO: apply gradients with a weight update pass.
            }
        }
    }

    pub fn set_training_spec(&mut self, training: TrainingSpec) {
        self.training = Some(training);
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

    fn move_gpu_data_to_cpu(&mut self) {
        let moved_params: Vec<(Vec<f32>, Vec<f32>)> = self
            .layers
            .iter()
            .map(|layer| {
                let weights = self.read_back_f32_buffer(layer.gpu_weights(), layer.weight_size_bytes());
                let bias = self.read_back_f32_buffer(layer.gpu_bias(), layer.bias_size_bytes());
                (weights, bias)
            })
            .collect();

        for (layer, (weights, bias)) in self.layers.iter_mut().zip(moved_params.into_iter()) {
            layer.set_cpu_params(Arc::new(weights), Arc::new(bias));
        }
    }

    pub fn save(&mut self, name: &str) {
        self.move_gpu_data_to_cpu();

        let saved_layers = self
            .layers
            .iter()
            .map(|layer| {
                let weights = layer
                    .cpu_weights()
                    .expect("CPU weights should be initialized by move_gpu_data_to_cpu or load")
                    .as_ref()
                    .clone();
                let bias = layer
                    .cpu_bias()
                    .expect("CPU bias should be initialized by move_gpu_data_to_cpu or load")
                    .as_ref()
                    .clone();

                SavedLayer {
                    architecture: layer.saved_architecture(),
                    weights,
                    bias,
                }
            })
            .collect();

        let saved_model = SavedModel {
            version: 1,
            training: self.training.as_ref().map(|training| SavedTrainingSpec {
                lr: training.lr,
                batch_size: training.batch_size,
                optimizer: training.optimizer,
            }),
            layers: saved_layers,
        };

        let content =
            serde_json::to_string_pretty(&saved_model).expect("failed to serialize model to JSON");
        std::fs::write(name, content).expect("failed to write model file");
    }
}
