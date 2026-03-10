use crate::gpu_context::GpuContext;
use crate::model::error::ModelError;
use crate::model::layer_types::{LayerType, LayerTypes};
use crate::model::types::Dim3;
use std::sync::Arc;
use wgpu::{
    BindGroup, Buffer, BufferDescriptor, CommandEncoder, ComputePipeline, Device, ShaderModule,
};

#[derive(Debug, Clone)]
pub struct Forward;
#[derive(Debug, Clone)]
pub struct Backward;

#[derive(Debug, Clone)]
pub(crate) struct Layer {
    pub(crate) ty: LayerTypes,
    pub(crate) buffers: Vec<Arc<Buffer>>,
    pub(crate) back_buffers: Vec<Arc<Buffer>>,
    pub(crate) shader: ShaderModule,
    pub(crate) back_shader: ShaderModule,
    pub(crate) pipeline: Option<ComputePipeline>,
    pub(crate) back_pipeline: Option<ComputePipeline>,
    pub(crate) num_workgroups: u32,
    pub(crate) bind_group: Option<BindGroup>,
    pub(crate) back_bind_group: Option<BindGroup>,
}

impl Layer {
    pub(crate) fn new(
        device: &Device,
        spec: LayerTypes,
        last_output: Option<Dim3>,
    ) -> Result<Self, ModelError> {
        let mut ty = spec;
        if let Some(input) = last_output {
            ty.set_dim_input(input);
        }
        ty.set_dim_output()?;
        let buffers = vec![];
        let back_buffers = vec![];
        let shader = Self::create_shader(device, &ty);
        let back_shader = Self::create_back_shader(device, &ty);
        let num_workgroups = ty.get_dim_output().length().div_ceil(64);
        Ok(Self {
            ty,
            shader,
            back_shader,
            buffers,
            pipeline: None,
            num_workgroups,
            bind_group: None,
        })
    }

    pub(crate) fn clear(&mut self) {
        self.buffers.clear();
        self.pipeline = None;
        self.bind_group = None;
    }

    fn create_shader(device: &Device, spec: &LayerTypes) -> ShaderModule {
        let (code, name): (&str, &str) = match spec {
            LayerTypes::Convolution(_) => (include_str!("shader/convolution.wgsl"), "convolution"),
            LayerTypes::Activation(_) => (include_str!("shader/activation.wgsl"), "activation"),
        };

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(code)),
        })
    }

    pub(crate) fn create_buffers(
        &mut self,
        gpu: &GpuContext,
        last_output: Option<Arc<Buffer>>,
    ) -> Arc<Buffer> {
        let buffers_specs = self.ty.get_buffers_specs();
        for (i, buff) in buffers_specs.iter().enumerate() {
            if i == 0
                && let Some(ref prev_buff) = last_output
            {
                self.buffers.push(Arc::clone(prev_buff));
                continue;
            }
            let new_buffer = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                label: Some(&buff.0),
                size: buff.1.size as u64,
                usage: buff.1.usage,
                mapped_at_creation: false,
            }));
            if buff.0 == "specs" {
                let uniform = self.ty.get_spec_uniform_bytes();
                gpu.queue.write_buffer(new_buffer.as_ref(), 0, &uniform);
            }
            self.buffers.push(new_buffer);
        }
        self.buffers.last().unwrap().clone()
    }
    pub(crate) fn set_pipeline(&mut self, device: &Device) {
        let buffers_specs = self.ty.get_buffers_specs();

        let mut entries = Vec::new();
        buffers_specs
            .iter()
            .enumerate()
            .for_each(|(binding, (_name, usage))| {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: usage.visibility,
                    ty: usage.ty,
                    count: None,
                });
            });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        self.pipeline = Some(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: Some(&pipeline_layout),
                module: &self.shader,
                entry_point: Some(self.ty.get_entrypoint()),
                compilation_options: Default::default(),
                cache: Default::default(),
            }),
        );
    }

    pub(crate) fn set_bind_group(&mut self, device: &Device) {
        let entries = self
            .buffers
            .iter()
            .enumerate()
            .map(|(idx, buffer)| wgpu::BindGroupEntry {
                binding: idx as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        self.bind_group = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self
                    .pipeline
                    .as_ref()
                    .expect("pipeline must be initialized before set_bind_group")
                    .get_bind_group_layout(0),
                entries: &entries,
            }),
        );
    }

    pub(crate) fn encode_pass(&self, encoder: &mut CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(
            self.pipeline
                .as_ref()
                .expect("pipeline must be initialized before encode"),
        );
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.num_workgroups, 1, 1);
    }

    // pub(crate) fn get_output_buffer(&self) -> Arc<Buffer> {
    //     self.buffers.last().unwrap().clone()
    // }

    // // // // // // // // //
    //   BACK PROPAGATION   //
    // // // // // // // // //
    pub(crate) fn encode_optimizer_pass(&self, encoder: &mut CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(
            self.back_pipeline
                .as_ref()
                .expect("pipeline must be initialized before encode"),
        );
        pass.set_bind_group(0, &self.back_bind_group, &[]);
        pass.dispatch_workgroups(self.num_workgroups, 1, 1);
    }
    pub(crate) fn encode_back_pass(&self, encoder: &mut CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(
            self.back_pipeline
                .as_ref()
                .expect("pipeline must be initialized before encode"),
        );
        pass.set_bind_group(0, &self.back_bind_group, &[]);
        pass.dispatch_workgroups(self.num_workgroups, 1, 1);
    }

    fn create_back_shader(device: &Device, spec: &LayerTypes) -> ShaderModule {
        let (code, name): (&str, &str) = match spec {
            LayerTypes::Convolution(_) => (
                include_str!("shader/back_convolution.wgsl"),
                "back_convolution",
            ),
            LayerTypes::Activation(_) => (
                include_str!("shader/back_activation.wgsl"),
                "back_activation",
            ),
        };

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(code)),
        })
    }
    pub(crate) fn create_back_buffers(
        &mut self,
        gpu: &GpuContext,
        last_output: Option<Arc<Buffer>>,
    ) -> Arc<Buffer> {
        let training_buffers_specs = self.ty.get_back_buffers_specs();
        for (i, buff) in training_buffers_specs.iter().enumerate() {
            if i == 0
                && let Some(ref prev_buff) = last_output
            {
                self.back_buffers.push(Arc::clone(prev_buff));
                continue;
            }
            let new_buffer = Arc::new(gpu.device.create_buffer(&BufferDescriptor {
                label: Some(&buff.0),
                size: buff.1.size as u64,
                usage: buff.1.usage,
                mapped_at_creation: false,
            }));
            self.back_buffers.push(new_buffer);
        }
        self.back_buffers.last().unwrap().clone()
    }

    pub(crate) fn set_back_pipeline(&mut self, device: &Device) {
        let buffers_specs = self.ty.get_back_buffers_specs();

        let mut entries = Vec::new();
        buffers_specs
            .iter()
            .enumerate()
            .for_each(|(binding, (_name, usage))| {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: usage.visibility,
                    ty: usage.ty,
                    count: None,
                });
            });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("back_bind_group_layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        self.back_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: Some(&pipeline_layout),
                module: &self.back_shader,
                entry_point: Some(self.ty.get_entrypoint()),
                compilation_options: Default::default(),
                cache: Default::default(),
            },
        ));
    }

    pub(crate) fn set_back_bind_group(&mut self, device: &Device) {
        let entries = self
            .back_buffers
            .iter()
            .enumerate()
            .map(|(idx, buffer)| wgpu::BindGroupEntry {
                binding: idx as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        self.back_bind_group = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self
                    .back_pipeline
                    .as_ref()
                    .expect("pipeline must be initialized before set_bind_group")
                    .get_bind_group_layout(0),
                entries: &entries,
            }),
        );
    }
}
