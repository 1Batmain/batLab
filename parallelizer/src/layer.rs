use std::sync::Arc;

use wgpu::{BindGroup, Buffer, ComputePipeline, Device, Queue, ShaderModule};

use crate::persistence::SavedLayerArchitecture;
use crate::spec::{ActivationLayerSpec, ConvolutionLayerSpec, LayerSpec};
use crate::types::{ActivationMethod, ConvolutionSpecUniform, Dim3, PaddingMode};

#[derive(Debug)]
struct ConvolutionLayer {
    nb_kernel: u32,
    dim_kernel: Dim3,
    stride: u32,
    mode: PaddingMode,
    dim_input: Dim3,
    dim_output: Dim3,
}

#[derive(Debug)]
struct ActivationLayer {
    method: ActivationMethod,
    dim_input: Dim3,
    dim_output: Dim3,
}

pub trait LayerType: std::fmt::Debug {
    fn get_nb_workgroups(&self) -> u32;
    fn get_input_size(&self) -> u32;
    fn get_output_size(&self) -> u32;
    fn get_dim_input(&self) -> Dim3;
    fn get_weight_size(&self) -> u32;
    fn get_bias_size(&self) -> u32;

    fn get_input_size_bytes(&self) -> u64 {
        (self.get_input_size() * 4) as u64
    }

    fn get_output_size_bytes(&self) -> u64 {
        (self.get_output_size() * 4) as u64
    }

    fn get_weight_size_bytes(&self) -> u64 {
        (self.get_weight_size() * 4) as u64
    }

    fn get_bias_size_bytes(&self) -> u64 {
        (self.get_bias_size() * 4) as u64
    }

    fn set_dim_output(&mut self) -> Dim3;
    fn get_dim_output(&self) -> Dim3;
    fn to_saved_architecture(&self) -> SavedLayerArchitecture;
}

impl LayerType for ConvolutionLayer {
    fn get_nb_workgroups(&self) -> u32 {
        self.dim_output.length().div_ceil(64)
    }

    fn get_input_size(&self) -> u32 {
        self.dim_input.length()
    }

    fn get_output_size(&self) -> u32 {
        self.dim_output.length()
    }

    fn get_dim_input(&self) -> Dim3 {
        self.dim_input
    }

    fn get_weight_size(&self) -> u32 {
        self.dim_kernel.length() * self.nb_kernel
    }

    fn get_bias_size(&self) -> u32 {
        self.nb_kernel
    }

    fn set_dim_output(&mut self) -> Dim3 {
        let padding = match self.mode {
            PaddingMode::Valid => (0, 0),
            PaddingMode::Same => (self.dim_kernel.x - 1, self.dim_kernel.y - 1),
        };
        let x = ((self.dim_input.x + 2 * padding.0 - self.dim_kernel.x) / self.stride) + 1;
        let y = ((self.dim_input.y + 2 * padding.1 - self.dim_kernel.y) / self.stride) + 1;
        let z = self.nb_kernel;
        let res = Dim3::new((x, y, z));
        self.dim_output = res;
        self.dim_output
    }

    fn get_dim_output(&self) -> Dim3 {
        self.dim_output
    }

    fn to_saved_architecture(&self) -> SavedLayerArchitecture {
        SavedLayerArchitecture::Convolution {
            nb_kernel: self.nb_kernel,
            dim_kernel: self.dim_kernel,
            stripe: self.stride,
            mode: self.mode,
            dim_input: self.dim_input,
            dim_output: self.dim_output,
        }
    }
}

impl LayerType for ActivationLayer {
    fn get_nb_workgroups(&self) -> u32 {
        self.dim_output.length().div_ceil(64)
    }

    fn get_input_size(&self) -> u32 {
        self.dim_input.length()
    }

    fn get_output_size(&self) -> u32 {
        self.dim_output.length()
    }

    fn get_dim_input(&self) -> Dim3 {
        self.dim_input
    }

    fn get_weight_size(&self) -> u32 {
        0
    }

    fn get_bias_size(&self) -> u32 {
        0
    }

    fn set_dim_output(&mut self) -> Dim3 {
        self.dim_output = self.dim_input;
        self.dim_output
    }

    fn get_dim_output(&self) -> Dim3 {
        self.dim_output
    }

    fn to_saved_architecture(&self) -> SavedLayerArchitecture {
        SavedLayerArchitecture::Activation {
            method: self.method,
            dim_input: self.dim_input,
            dim_output: self.dim_output,
        }
    }
}

impl ConvolutionLayer {
    fn new(spec: &ConvolutionLayerSpec) -> Self {
        let mut instance = Self {
            nb_kernel: spec.nb_kernel,
            dim_kernel: spec.dim_kernel,
            stride: spec.stripe,
            mode: spec.mode,
            dim_input: spec
                .dim_input
                .expect("Convolution layer requires dim_input during construction"),
            dim_output: Dim3::new((0,0,0)),
        };
        instance.set_dim_output();
        instance
    }
}

impl ActivationLayer {
    fn new(spec: &ActivationLayerSpec) -> Self {
        let mut instance = Self {
            method: spec.method,
            dim_input: spec
                .dim_input
                .expect("Activation layer requires dim_input during construction"),
            dim_output: Dim3::new((0,0,0)),
        };
        instance.set_dim_output();
        instance
    }
}

#[derive(Debug)]
struct LayerBuffers {
    gpu: GpuBuffers,
    training: GpuTrainingBuffers,
    cpu: CpuBuffers,
}

#[derive(Debug)]
struct CpuBuffers {
    weights: Option<Arc<Vec<f32>>>,
    bias: Option<Arc<Vec<f32>>>,
}

#[derive(Debug)]
struct GpuBuffers {
    input: Option<Arc<Buffer>>,
    output: Option<Arc<Buffer>>,
    weights: Option<Arc<Buffer>>,
    bias: Option<Arc<Buffer>>,
    spec_uniform: Option<Arc<Buffer>>,  // For convolution spec (dim_input, stride, mode)
}

#[derive(Debug)]
struct GpuTrainingBuffers {
    grad_output: Option<Arc<Buffer>>,
    grad_input: Option<Arc<Buffer>>,
    grad_weights: Option<Arc<Buffer>>,
    grad_bias: Option<Arc<Buffer>>,
}

impl LayerBuffers {
    fn new() -> Self {
        Self {
            gpu: GpuBuffers::new(),
            training: GpuTrainingBuffers::new(),
            cpu: CpuBuffers::new(),
        }
    }
}

impl GpuBuffers {
    fn new() -> Self {
        Self {
            input: None,
            output: None,
            weights: None,
            bias: None,
            spec_uniform: None,
        }
    }
}

impl GpuTrainingBuffers {
    fn new() -> Self {
        Self {
            grad_output: None,
            grad_input: None,
            grad_weights: None,
            grad_bias: None,
        }
    }
}

impl CpuBuffers {
    fn new() -> Self {
        Self {
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
    backprop_shader: Option<ShaderModule>,
    pipeline: Option<ComputePipeline>,
    backprop_pipeline: Option<ComputePipeline>,
    num_workgroups: u32,
    bind_group: Option<BindGroup>,
    backprop_bind_group: Option<BindGroup>,
}

impl Layer {
    pub(crate) fn new(device: &Device, spec: LayerSpec) -> Self {
        let ty = Self::create_layer_type(&spec);
        let buffers = LayerBuffers::new();
        let shader = Self::create_shader(device, &spec);
        let num_workgroups = ty.get_nb_workgroups();
        Self {
            ty,
            buffers,
            shader,
            backprop_shader: None,
            pipeline: None,
            backprop_pipeline: None,
            num_workgroups,
            bind_group: None,
            backprop_bind_group: None,
        }
    }

    fn create_layer_type(spec: &LayerSpec) -> Box<dyn LayerType> {
        match spec {
            LayerSpec::Convolution(layer_spec) => Box::new(ConvolutionLayer::new(layer_spec)),
            LayerSpec::Activation(layer_spec) => Box::new(ActivationLayer::new(layer_spec)),
        }
    }

    fn create_shader(device: &Device, spec: &LayerSpec) -> ShaderModule {
        let (code, name): (&str, &str) = match spec {
            LayerSpec::Convolution(_) => (include_str!("shader/convolution.wgsl"), "convolution"),
            LayerSpec::Activation(_) => (include_str!("shader/activation.wgsl"), "activation"),
        };

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(code)),
        })
    }

    fn create_backprop_shader(device: &Device, spec: &LayerSpec) -> ShaderModule {
        let (code, name): (&str, &str) = match spec {
            LayerSpec::Convolution(_) => (include_str!("shader/convolution.wgsl"), "convolution"),
            LayerSpec::Activation(_) => (include_str!("shader/activation.wgsl"), "activation"),
        };

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(code)),
        })
    }

    fn forward_entry_point(architecture: &SavedLayerArchitecture) -> &'static str {
        match architecture {
            SavedLayerArchitecture::Convolution { .. } => "inference",
            SavedLayerArchitecture::Activation { method, .. } => match method {
                ActivationMethod::Relu => "inference_relu",
                ActivationMethod::Linear => "inference_linear",
            },
        }
    }

    fn backprop_entry_point(spec: &LayerSpec) -> &'static str {
        match spec {
            LayerSpec::Convolution(_) => "backpropagate",
            LayerSpec::Activation(layer_spec) => match layer_spec.method {
                ActivationMethod::Relu => "backpropagate_relu",
                ActivationMethod::Linear => "backpropagate_linear",
            },
        }
    }

    pub(crate) fn set_pipeline(&mut self, device: &Device) {
        // build layout entries dynamically to support both convolution and
        // activation layers. training buffers are *not* included in the forward
        // pass layout (they belong only to the backprop pipeline) which keeps
        // the storage buffer count low and prevents runtime validation errors
        // on devices with a small limit.
        let mut entries = vec![
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        ];

        // only convolution layers need the spec uniform
        if let SavedLayerArchitecture::Convolution { .. } = self.saved_architecture() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(
                            std::mem::size_of::<ConvolutionSpecUniform>() as u64
                        ).unwrap()
                    ),
                },
                count: None,
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &self.shader,
            entry_point: Some(Self::forward_entry_point(&self.saved_architecture())),
            compilation_options: Default::default(),
            cache: Default::default(),
        }));
    }

    pub(crate) fn set_bind_group(&mut self, device: &Device) {
        // build entries corresponding to the layout we constructed earlier;
        // for activation layers the spec_uniform will be omitted so we only push
        // it when a buffer is available.
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self
                    .buffers
                    .gpu
                    .input
                    .as_ref()
                    .expect("input GPU buffer must be initialized before set_bind_group")
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self
                    .buffers
                    .gpu
                    .weights
                    .as_ref()
                    .expect("weights GPU buffer must be initialized before set_bind_group")
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self
                    .buffers
                    .gpu
                    .bias
                    .as_ref()
                    .expect("bias GPU buffer must be initialized before set_bind_group")
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self
                    .buffers
                    .gpu
                    .output
                    .as_ref()
                    .expect("output GPU buffer must be initialized before set_bind_group")
                    .as_entire_binding(),
            },
        ];

        if let Some(spec_buf) = &self.buffers.gpu.spec_uniform {
            entries.push(wgpu::BindGroupEntry {
                binding: 8,
                resource: spec_buf.as_entire_binding(),
            });
        }

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .pipeline
                .as_ref()
                .expect("pipeline must be initialized before set_bind_group")
                .get_bind_group_layout(0),
            entries: &entries,
        }));
    }

    pub(crate) fn set_backprop_pipeline(&mut self, device: &Device, spec: &LayerSpec) {
        if self.backprop_shader.is_none() {
            self.backprop_shader = Some(Self::create_backprop_shader(device, spec));
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backprop_bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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
            label: Some("backprop_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        self.backprop_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("backprop_pipeline"),
                layout: Some(&pipeline_layout),
                module: self.backprop_shader.as_ref().expect("backprop shader must be initialized"),
                entry_point: Some(Self::backprop_entry_point(spec)),
                compilation_options: Default::default(),
                cache: Default::default(),
            },
        ));
    }

    pub(crate) fn set_backprop_bind_group(&mut self, device: &Device) {
        self.backprop_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .backprop_pipeline
                .as_ref()
                .expect("backprop pipeline must be initialized before set_backprop_bind_group")
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .buffers
                        .gpu
                        .input
                        .as_ref()
                        .expect("input GPU buffer must be initialized before set_backprop_bind_group")
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .buffers
                        .gpu
                        .weights
                        .as_ref()
                        .expect("weights GPU buffer must be initialized before set_backprop_bind_group")
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .buffers
                        .gpu
                        .bias
                        .as_ref()
                        .expect("bias GPU buffer must be initialized before set_backprop_bind_group")
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self
                        .buffers
                        .gpu
                        .output
                        .as_ref()
                        .expect("output GPU buffer must be initialized before set_backprop_bind_group")
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self
                        .buffers
                        .training
                        .grad_input
                        .as_ref()
                        .expect("grad_input buffer must be initialized before set_backprop_bind_group")
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self
                        .buffers
                        .training
                        .grad_weights
                        .as_ref()
                        .expect("grad_weights buffer must be initialized before set_backprop_bind_group")
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self
                        .buffers
                        .training
                        .grad_bias
                        .as_ref()
                        .expect("grad_bias buffer must be initialized before set_backprop_bind_group")
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self
                        .buffers
                        .training
                        .grad_output
                        .as_ref()
                        .expect("grad_output buffer must be initialized before set_backprop_bind_group")
                        .as_entire_binding(),
                },
            ],
        }));
    }

    pub fn input_size_bytes(&self) -> u64 {
        self.ty.get_input_size_bytes()
    }

    pub fn output_size_bytes(&self) -> u64 {
        self.ty.get_output_size_bytes()
    }

    pub fn weight_size_bytes(&self) -> u64 {
        self.ty.get_weight_size_bytes()
    }

    pub fn bias_size_bytes(&self) -> u64 {
        self.ty.get_bias_size_bytes()
    }

    pub(crate) fn weight_size(&self) -> usize {
        self.ty.get_weight_size() as usize
    }

    pub(crate) fn bias_size(&self) -> usize {
        self.ty.get_bias_size() as usize
    }

    pub fn dim_output(&self) -> Dim3 {
        self.ty.get_dim_output()
    }

    pub fn dim_input(&self) -> Dim3 {
        self.ty.get_dim_input()
    }

    pub fn set_gpu_buffers(
        &mut self,
        input: Arc<Buffer>,
        weights: Arc<Buffer>,
        bias: Arc<Buffer>,
        output: Arc<Buffer>,
    ) {
        self.buffers.gpu = GpuBuffers {
            input: Some(input),
            output: Some(output),
            weights: Some(weights),
            bias: Some(bias),
            spec_uniform: None,  // Will be set separately
        };
    }

    /// Set the convolution spec uniform buffer containing dim_input, stride, and mode.
    pub fn set_spec_uniform(&mut self, spec_uniform: Arc<Buffer>) {
        self.buffers.gpu.spec_uniform = Some(spec_uniform);
    }

    pub fn set_training_buffers(
        &mut self,
        grad_output: Arc<Buffer>,
        grad_input: Arc<Buffer>,
        grad_weights: Arc<Buffer>,
        grad_bias: Arc<Buffer>,
    ) {
        self.buffers.training = GpuTrainingBuffers {
            grad_output: Some(grad_output),
            grad_input: Some(grad_input),
            grad_weights: Some(grad_weights),
            grad_bias: Some(grad_bias),
        };
    }

    /// Create and set the convolution spec uniform buffer from a ConvolutionLayerSpec.
    pub fn create_spec_uniform(
        &mut self,
        device: &Device,
        _queue: &Queue,
        spec: &ConvolutionLayerSpec,
    ) {
        use encase::UniformBuffer;
        
        let padding_mode = match spec.mode {
            PaddingMode::Valid => 0u32,
            PaddingMode::Same => 1u32,
        };
        
        let uniform = ConvolutionSpecUniform {
            dim_input: spec.dim_input.unwrap_or_default(),
            stride: spec.stripe,
            padding_mode,
        };
        
        // `UniformBuffer::new` takes an array and calculates size from its
        // length. using an array of `u8` ensures we end up with raw bytes and
        // keeps the borrow/repr logic simple.
        let mut buffer_contents = UniformBuffer::new([
            0u8; std::mem::size_of::<ConvolutionSpecUniform>()
        ]);
        buffer_contents.write(&uniform).expect("failed to encode uniform");

        let spec_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convolution_spec_uniform"),
            size: buffer_contents.as_ref().len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        spec_buffer.slice(..).get_mapped_range_mut().copy_from_slice(buffer_contents.as_ref());
        spec_buffer.unmap();

        self.set_spec_uniform(Arc::new(spec_buffer));
    }

    pub fn num_workgroups(&self) -> u32 {
        self.num_workgroups
    }

    pub fn pipeline(&self) -> &ComputePipeline {
        self.pipeline
            .as_ref()
            .expect("pipeline not initialized: call build_model first")
    }

    pub fn backprop_pipeline(&self) -> &ComputePipeline {
        self.backprop_pipeline
            .as_ref()
            .expect("backprop pipeline not initialized: call train() to initialize training resources")
    }

    pub fn bind_group(&self) -> &BindGroup {
        self.bind_group
            .as_ref()
            .expect("bind_group not initialized: call build_model first")
    }

    pub fn backprop_bind_group(&self) -> &BindGroup {
        self.backprop_bind_group
            .as_ref()
            .expect("backprop bind_group not initialized: call train() to initialize training resources")
    }

    pub fn gpu_input(&self) -> &Buffer {
        self.buffers
            .gpu
            .input
            .as_ref()
            .expect("input buffer not initialized")
            .as_ref()
    }

    pub fn gpu_input_arc(&self) -> Arc<Buffer> {
        self.buffers
            .gpu
            .input
            .as_ref()
            .expect("input buffer not initialized")
            .clone()
    }

    pub fn gpu_output(&self) -> &Buffer {
        self.buffers
            .gpu
            .output
            .as_ref()
            .expect("output buffer not initialized")
            .as_ref()
    }

    pub fn gpu_output_arc(&self) -> Arc<Buffer> {
        self.buffers
            .gpu
            .output
            .as_ref()
            .expect("output buffer not initialized")
            .clone()
    }

    pub fn gpu_weights(&self) -> &Buffer {
        self.buffers
            .gpu
            .weights
            .as_ref()
            .expect("weights buffer not initialized")
            .as_ref()
    }

    pub fn gpu_bias(&self) -> &Buffer {
        self.buffers
            .gpu
            .bias
            .as_ref()
            .expect("bias buffer not initialized")
            .as_ref()
    }

    pub fn grad_output_arc(&self) -> Arc<Buffer> {
        self.buffers
            .training
            .grad_output
            .as_ref()
            .expect("grad_output buffer not initialized: call train() to initialize training resources")
            .clone()
    }

    pub fn output_arc(&self) -> Arc<Buffer> {
        self.buffers
            .gpu
            .output
            .as_ref()
            .expect("output buffer not initialized")
            .clone()
    }

    pub fn cpu_weights(&self) -> Option<&Arc<Vec<f32>>> {
        self.buffers.cpu.weights.as_ref()
    }

    pub fn cpu_bias(&self) -> Option<&Arc<Vec<f32>>> {
        self.buffers.cpu.bias.as_ref()
    }

    pub fn set_cpu_params(&mut self, weights: Arc<Vec<f32>>, bias: Arc<Vec<f32>>) {
        self.buffers.cpu.weights = Some(weights);
        self.buffers.cpu.bias = Some(bias);
    }

    pub fn saved_architecture(&self) -> SavedLayerArchitecture {
        self.ty.to_saved_architecture()
    }
}
