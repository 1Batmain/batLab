use wgpu::{Adapter, BindGroup, Buffer, BufferDescriptor, BufferUsages, ComputePipeline, Device, Instance, Queue, ShaderModule};
use encase::ShaderType;
use std::sync::Arc;

// Définitions de types utils
#[derive(Debug, Clone, Copy)]
pub enum PaddingMode {
    Valid,
    Same,
}
#[derive(ShaderType, Debug, Clone, Copy)]
pub struct Dim3{
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
impl Dim3 {
    pub fn new() -> Self {
        Self {
            x:0,
            y:0,z:0
        }
    }
    pub fn length(&self) -> u32{
        self.x * self.y * self.z 
    }
}

// Définition des types Specs (définition d'un layer)
pub enum LayerSpec {
    Convolution(ConvolutionLayerSpec),
}
pub struct ConvolutionLayerSpec {
    pub nb_kernel: u32,     // Nb of convolution filter in the layer
    pub dim_kernel: Dim3,
    pub stripe: u32,        // step size to move the filter
    pub mode: PaddingMode,  // Active the layer 
    pub dim_input: Option<Dim3>
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
    fn get_input_size_bytes(&self) -> u64 { (self.get_input_size() * 4) as u64 }
    fn get_output_size_bytes(&self) -> u64 { (self.get_output_size() * 4) as u64 }
    fn get_weight_size_bytes(&self) -> u64 { (self.get_weight_size() * 4) as u64 }
    fn get_bias_size_bytes(&self) -> u64 { (self.get_bias_size() * 4) as u64 }
    fn set_dim_output(& mut self) -> Dim3;
    fn get_dim_output(&self) -> Dim3;
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
    fn set_dim_output(&mut self) -> Dim3 {
        let padding = match self.mode {
            PaddingMode::Valid  => (0, 0),
            PaddingMode::Same   => (self.dim_kernel.x - 1, self.dim_kernel.y - 1),
        };
        let x = ((self.dim_input.x + 2 * padding.0 - self.dim_kernel.x) / self.stripe) + 1;
        let y = ((self.dim_input.y + 2 * padding.1 - self.dim_kernel.y) / self.stripe) + 1;
        let z = self.nb_kernel;
        let res = Dim3 {x, y, z};
        self.dim_output = res;
        self.dim_output
    }
    fn get_dim_output(&self) -> Dim3 {
        self.dim_output
    }
}
impl ConvolutionLayer {
    pub fn new(spec: &ConvolutionLayerSpec) -> Self
    {
        let mut instance = Self {
                nb_kernel: spec.nb_kernel,
                dim_kernel: spec.dim_kernel,
                stripe: spec.stripe,
                mode: spec.mode,
                dim_input: spec.dim_input.unwrap(),
                dim_output: Dim3::new(),
            };
        instance.set_dim_output();
        instance
    }
}

#[derive(Debug)]
struct  LayerBuffers {
    gpu: GpuBuffers,
    cpu: CpuBuffers,
}
#[derive(Debug)]
struct CpuBuffers {
    input: Option<Arc<Vec<f32>>>,
    output: Option<Arc<Vec<f32>>>,
    weights: Option<Arc<Vec<f32>>>,
    bias: Option<Arc<Vec<f32>>>,
}
#[derive(Debug)]
struct GpuBuffers {
    input: Option<Arc<Buffer>>,
    output: Option<Arc<Buffer>>,
    weights: Option<Arc<Buffer>>,
    bias: Option<Arc<Buffer>>,
}


impl GpuBuffers {
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
impl CpuBuffers {
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
        let (code, name): (&str, &str) = match spec {
            LayerSpec::Convolution(_) => (include_str!("shader/convolution.wgsl"), "convolution"),
//            LayerSpec::Activation(_) => (include_str!("shader/activation.wgsl"), "activation"),
//            LayerSpec::Upscale(_) => (include_str!("shader/upscale.wgsl"), "upscale"),
//            LayerSpec::Loss(_) => (include_str!("shader/loss.wgsl"), "loss"),
        };
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(code)),
        });
        shader
    }
    fn set_pipeline(&mut self, device: &Device) {
        // Create bind group layout explicitly
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("piepeline"),
            layout: Some(&pipeline_layout),
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
#[derive(Debug)]
pub struct TrainingSpec {
    lr: f32,
    batch_size: u32,
}
#[derive(Debug)]
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

    fn create_buffer(device: &Device, size: u64) -> Arc<Buffer> {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("buffer"),
            size: size,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    pub fn build_model(&mut self) 
    {
        let device = &self.device;
        
        let input_bytes = self.layer.first().unwrap().ty.get_input_size_bytes();
        
        let mut last_output = Self::create_buffer(device, input_bytes);
        for  layer in self.layer.iter_mut() {
            let output_bytes = layer.ty.get_output_size_bytes();
            
            layer.buffers = LayerBuffers {
                input: Some(last_output),
                weights: Some(Self::create_buffer(device, layer.ty.get_weight_size_bytes())),
                bias: Some(Self::create_buffer(device, layer.ty.get_bias_size_bytes())),
                output: Some(Self::create_buffer(device, output_bytes)),
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
            pass.set_bind_group(0, layer.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(layer.num_workgroups, 1, 1);
        }

        let layer = self.layer.last().unwrap();
        let output_size_bytes = layer.ty.get_output_size_bytes();  // f32 is 4 bytes
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&layer.buffers.gpu.output.as_ref().unwrap(), 0, &staging_buffer, 0, output_size_bytes);
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

        let result;
        {
            let buffers = buffer_slice.get_mapped_range();
            result = bytemuck::cast_slice(&buffers).to_vec();
        }
        staging_buffer.unmap();
        result
    }
    pub fn add_layer(&mut self, mut spec: LayerSpec) {
        if !self.layer.is_empty() {
            let prev_output = self.layer.last().unwrap().ty.get_dim_output();
            match &mut spec {
                LayerSpec::Convolution(conv_spec) => {
                    conv_spec.dim_input = Some(prev_output);
                }
            }
        }
        self.layer.push(Layer::new(&self.device, spec));
    }
}