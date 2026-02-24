use wgpu::{Adapter, BindGroup, Buffer, BufferDescriptor, BufferUsages, ComputePipeline, Device, Instance, Queue, ShaderModule, util::{BufferInitDescriptor, DeviceExt}};
use encase::{ShaderType, rts_array::Length};
use std::sync::Arc;

// Définitions de types utils
#[derive(Debug, Clone, Copy)]
enum PaddingMode {
    Valid,
    Same,
}
#[derive(ShaderType, Debug, Clone, Copy)]
struct Dim3{
    x: u32,
    y: u32,
    z: u32,
}
impl Dim3 {
    pub fn new() -> Self {
        Self {
            x:0,
            y:0,z:0
        }
    }
    pub fn length(&self) -> u32{
        (self.x * self.y * self.z)
    }
}

// Définition des types Specs (définition d'un layer)
enum LayerSpec {
    Convolution(ConvolutionLayerSpec),
}
pub struct ConvolutionLayerSpec {
    pub nb_kernel: u32,     // Nb of convolution filter in the layer
    pub dim_kernel: Dim3,
    pub stripe: u32,        // step size to move the filter
    pub mode: PaddingMode,  // Active the layer 
    pub dim_input: Dim3
}

// Définition des types de layers (Execution layer) 
#[derive(Debug)]
struct ConvolutionLayer {
    nb_kernel: u32,
    dim_kernel: Dim3,
    stripe: u32,
    mode: PaddingMode,
    dim_input: Dim3,
    dim_output: Dim3,
}
pub trait LayerType : std::fmt::Debug {
    fn get_nb_workgroups(&self) -> u32;
    fn get_input_size(&self) -> u32;
    fn get_output_size(&self) -> u32;
    fn get_weight_size(&self) -> u32;
    fn get_bias_size(&self) -> u32;
    fn set_dim_output(&self) -> Dim3;
    // TODO get_shader_buffers -> Shaderbuffers
}
impl LayerType for ConvolutionLayer {
    fn get_nb_workgroups(&self) -> u32 {
       self.dim_output.length().div_ceil(64) as u32
    }
    fn get_input_size(&self) -> u32 {
       self.dim_input.length()
    }
    fn get_output_size(&self) -> u32 {
        self.dim_output.length()
    }
    fn get_weight_size(&self) -> u32 {
        self.dim_kernel.length() * self.nb_kernel
    }
    fn get_bias_size(&self) -> u32 {
        self.nb_kernel
    }
    fn set_dim_output(&self) -> Dim3 {
        let padding = match self.mode {
            PaddingMode::Valid  => (0, 0),
            PaddingMode::Same   => (self.dim_kernel.x - 1, self.dim_kernel.y - 1),
        };
        let x = (self.dim_input.x + 2 * padding.0 - self.dim_kernel.x / self.stripe) + 1;
        let y = (self.dim_input.y + 2 * padding.1 - self.dim_kernel.y / self.stripe) + 1;
        let z = self.nb_kernel;
        Dim3 {x, y, z}
    }
}
impl ConvolutionLayer {
    pub fn new(spec: &ConvolutionLayerSpec) -> Self
    {
        let instance = Self {
                nb_kernel: spec.nb_kernel,
                dim_kernel: spec.dim_kernel,
                stripe: spec.stripe,
                mode: spec.mode,
                dim_input: spec.dim_input,
                dim_output: Dim3::new(),
            };
        instance.set_dim_output();
        instance
    }
}

#[derive(Debug)]
struct LayerBuffers {
    input: Option<Arc<Buffer>>,
    output: Option<Arc<Buffer>>,
    weights: Option<Arc<Buffer>>,
    bias: Option<Arc<Buffer>>,
}

impl LayerBuffers {
    fn new() -> Self 
    {
        Self {
            input: None, 
            output: None,
            weights: None,
            bias: None,
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    ty: Box<dyn LayerType>,
    buffers: LayerBuffers, 
    shader: ShaderModule,
    pipeline: Option<ComputePipeline>,
    num_workgroups: u32,
    bind_group: Option<BindGroup>,
}
impl Layer {
    pub fn new(device: &Device, spec: LayerSpec) -> Self
    {
        let ty = Self::create_layer_type(&spec);
        let buffers = LayerBuffers::new();
        let shader = Self::create_shader(device, &spec);
        let num_workgroups = ty.get_nb_workgroups();
        Self {
            ty,
            buffers,
            shader,
            pipeline: None,
            num_workgroups,
            bind_group: None,
        }
    }

    fn create_layer_type(spec: &LayerSpec) -> Box<dyn LayerType>
    {
        match spec {
            LayerSpec::Convolution(l) => Box::new(ConvolutionLayer::new(l)),
//            LayerSpec::Activation(l) => Box::new(ActivationLayer::new(l)),
//            LayerSpec::Upscale(l) => Box::new(UpscaleLayer::new(l)),
//            LayerSpec::Loss(l) => Box::new(LossLayer::new(l)),
        }
    }

    fn create_shader(device: &Device, spec: &LayerSpec) -> ShaderModule {
        let (path, name): (&str, &str) = match spec {
            LayerSpec::Convolution(_) => ("shader/convolution.wgsl", "convolution"),
//            LayerSpec::Activation(_) => ("shader/activation.wgsl", "activation"),
//            LayerSpec::Upscale(_) => ("shader/upscale.wgsl", "upscale"),
//            LayerSpec::Loss(_) => ("shader/loss.wgsl", "loss"),
        };

        let shader_code = std::fs::read_to_string(path)
            .expect(&format!("Failed to read shader file: {}", path));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_code)),
        });
        shader
    }
    fn set_pipeline(&mut self, device: &Device) {
        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("piepeline"),
            layout: None,
            module: &self.shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Default::default(),
        }));
    }
    fn set_bind_group(&mut self, device: &Device) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.as_ref().unwrap().get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.input.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:1,
                    resource: self.buffers.weights.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:2,
                    resource: self.buffers.bias.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:3,
                    resource: self.buffers.output.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));
    }
}
pub struct TrainingSpec {
    lr: f32,
    batch_size: u32,

}
pub struct Model {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    layer: Vec<Layer>,
    training: Option<TrainingSpec>,
}
impl Model {
    pub async fn new(training: Option<TrainingSpec>) -> Self {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        Self {
            instance,
            adapter,
            device,
            queue,
            layer: Vec::new(),
            training: training,
        }
    }

    fn create_buffer(device: &Device, size: u32) -> Arc<Buffer> {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("buffer"),
            size: size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    pub fn build_model(&mut self) 
    {
        let device = &self.device;
        let mut last_output = Self::create_buffer(device, self.layer.first().unwrap().ty.get_input_size());
        for  layer in self.layer.iter_mut() {
            layer.buffers = LayerBuffers {
                input: Some(last_output),
                weights: Some(Self::create_buffer(device, layer.ty.get_weight_size())),
                bias: Some(Self::create_buffer(device, layer.ty.get_bias_size())),
                output: Some(Self::create_buffer(device, layer.ty.get_output_size())),
            };
            last_output = layer.buffers.output.clone().unwrap();
            layer.set_pipeline(device);
            layer.set_bind_group(device);
        }
    }


    pub async fn run (&mut self) -> Vec<f32> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for layer in self.layer.iter() {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&layer.pipeline.as_ref().unwrap());
            pass.set_bind_group(0, &layer.bind_group, &[]);
            pass.dispatch_workgroups(layer.num_workgroups, 1, 1);
        }

        let layer = self.layer.last().unwrap();
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: layer.ty.get_output_size() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&layer.buffers.output.as_ref().unwrap(), 0, &staging_buffer, 0, layer.ty.get_output_size() as u64);
        self.queue.submit([encoder.finish()]);

        let buffer_slice = staging_buffer.slice(..);

        let (gpu, cpu) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| gpu.send(result).unwrap());

        match self.device.poll(wgpu::PollType::wait_indefinitely()) {
            Ok(_) => {},
            Err(e) => println!("error on poll : {}", e),
        };

        pollster::block_on(async { 
            let _ = cpu.await.unwrap(); 
        });

        let mut result = Vec::new();
        {
            let buffers = buffer_slice.get_mapped_range();
            result = bytemuck::cast_slice(&buffers).to_vec();
        }
        staging_buffer.unmap();
        result
    }
    pub fn add_layer(&mut self, spec: LayerSpec) {
        self.layer.push(Layer::new(&self.device, spec));
    }
}
/*
impl Layer {
    pub fn new(device: &Device, spec: &LayerSpec) -> Self
    {
        let ty = Self::create_layer_type(&spec);
        let shader = Self::create_shader(device, &spec);
        let pipeline = Self::create_pipeline(device, &shader);
        let num_workgroups = ty.get_nb_workgroups();
        let (input, output) = Self::create_buffers(device);
        let bind_group = Self::create_bind_group(device, &pipeline, &input, &output);
        Self {
            ty,
            shader,
            pipeline,
            num_workgroups,
            input,
            output,
            bind_group
       } 
    }

    fn create_pipeline(device: &Device, shader: &ShaderModule) -> ComputePipeline
    {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("piepeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        pipeline
    }

    fn create_buffers(&self, device: &Device) -> (Buffer, Buffer)
    {
        let input = device.create_buffer(&BufferDescriptor {
            label: Some("input"),
            size: self.ty.get_dim_input().length() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output = device.create_buffer(&BufferDescriptor {
            label: Some("output"),
            size: self.ty.get_dim_output().length() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        (input, output)
    }

    fn create_bind_group(device: &Device, pipeline: &ComputePipeline, input: &Buffer, output: &Buffer) -> BindGroup
    {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        bind_group
    }
}
*/