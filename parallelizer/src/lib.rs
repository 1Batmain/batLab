use wgpu::{Adapter, BindGroup, Buffer, BufferDescriptor, BufferUsages, ComputePipeline, Device, Instance, Queue, ShaderModule, util::{BufferInitDescriptor, DeviceExt}};

pub struct LayerSpec {
    pub name: String,
    pub shader_path: String,
    pub workgroup_size: usize,
    pub input: Vec<f32>,
    pub output: Vec<f32>,
}

impl LayerSpec {
    pub fn new(name: String, shader_path: String) -> Self {
        Self {
            name,
            shader_path,
            workgroup_size: 64,
            input: Vec::new(),
            output: Vec::new(),
        }
    }

    pub fn workgroup_size(mut self, size: usize) -> Self {
        self.workgroup_size = size;
        self
    }

    pub fn input(mut self, input: Vec<f32>) -> Self {
        self.input = input;
        self
    }

    pub fn output(mut self, output: Vec<f32>) -> Self {
        self.output = output;
        self
    }
}

#[derive(Debug)]
pub struct Layer {
    name: String,
    shader: ShaderModule,
    pipeline: ComputePipeline,
    num_workgroups: u32,
    input: Buffer,
    output: Buffer,
    bind_group: BindGroup,
}

impl Layer {
    pub fn new(device: &Device, spec: &LayerSpec) -> Self
    {
        let name = spec.name.to_string();
        let shader = Self::create_shader(device, &spec.shader_path);
        let pipeline = Self::create_pipeline(device, &spec.name, &shader);
        let num_workgroups = spec.input.len().div_ceil(spec.workgroup_size) as u32;
        let (input, output) = Self::create_buffers(device, &spec.input, &spec.output);
        let bind_group = Self::create_bind_group(device, &pipeline, &input, &output);
        Self {
            name,
            shader,
            pipeline,
            num_workgroups,
            input,
            output,
            bind_group
       } 
    }

    fn create_shader(device: &Device, path: &str) -> ShaderModule
    {
        let shader_code = std::fs::read_to_string(path)
            .expect(&format!("Failed to read shader file: {}", path));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_code)),
        });
        shader
    }
    fn create_pipeline(device: &Device, name: &str, shader: &ShaderModule) -> ComputePipeline
    {
        let label =  format!("{}_pipeline", name);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&label),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        pipeline
    }
    fn create_buffers<I: bytemuck::Pod, O: bytemuck::Pod>(device: &Device, input: &[I], output: &[O]) -> (Buffer, Buffer)
    {
        let input = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&BufferDescriptor {
            label: Some("output"),
            size: input.size() as u64,
            usage: BufferUsages::COPY_DST |BufferUsages::COPY_SRC | BufferUsages::STORAGE,
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

pub struct Wrapper {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    layer: Vec<Layer>,
}

impl Wrapper {
    pub async fn new() -> Self 
    {

        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        Self {
            instance,
            adapter,
            device,
            queue,
            layer: Vec::new(),
        }
    }
    pub async fn run (&mut self) -> Vec<f32>
    {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for layer in self.layer.iter() {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&layer.pipeline);
            pass.set_bind_group(0, &layer.bind_group, &[]);
            pass.dispatch_workgroups(layer.num_workgroups, 1, 1);
        }

        let layer = self.layer.last().unwrap();
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: layer.output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&layer.output, 0, &staging_buffer, 0, layer.output.size());
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
            let data = buffer_slice.get_mapped_range();
            result = bytemuck::cast_slice(&data).to_vec();
        }
        staging_buffer.unmap();
        result
    }
    pub fn add_layer(&mut self, spec: LayerSpec)
    {
        self.layer.push(Layer::new(&self.device, &spec));
    }
}