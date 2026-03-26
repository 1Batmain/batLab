//! File purpose: Implements diffusion logic used by the training pipeline.

use super::{
    GpuDataset, LinearNoiseSchedule, TaskPassSpec, TrainingTask, TrainingTaskError, Workgroups,
};
use crate::model::{Dim3, Model, Training};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DiffusionTask {
    schedule: LinearNoiseSchedule,
    timestep_channels: usize,
    pass_specs: Vec<TaskPassSpec>,
    configured_input: Option<Dim3>,
    configured_output: Option<Dim3>,
    prepare_pass: Option<DiffusionPreparePass>,
}

#[derive(Debug, Clone)]
struct DiffusionPreparePass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    clean_target: wgpu::Buffer,
    specs: wgpu::Buffer,
    model_input: Arc<wgpu::Buffer>,
    target_noise: Arc<wgpu::Buffer>,
    expected_target_len: usize,
    workgroups: u32,
}

#[derive(ShaderType, Clone, Copy)]
struct DiffusionPrepareUniform {
    alpha_bar: f32,
    step: u32,
    seed: u32,
    input_channels: u32,
    signal_channels: u32,
    timestep_channels: u32,
    pixel_count: u32,
}

impl DiffusionTask {
    pub fn new(schedule: LinearNoiseSchedule) -> Self {
        Self {
            schedule,
            timestep_channels: 0,
            pass_specs: Vec::new(),
            configured_input: None,
            configured_output: None,
            prepare_pass: None,
        }
    }

    pub fn schedule(&self) -> &LinearNoiseSchedule {
        &self.schedule
    }

    pub fn timestep_channels(&self) -> usize {
        self.timestep_channels
    }

    pub fn estimated_prepare_gpu_bytes(&self, output: Dim3) -> u64 {
        let target_bytes = (output.length() as u64 * std::mem::size_of::<f32>() as u64).max(4);
        let specs_bytes = (DiffusionPrepareUniform::SHADER_SIZE.get() as u64).max(4);
        target_bytes.saturating_add(specs_bytes)
    }

    pub fn train_step_report(
        &mut self,
        model: &mut Model<Training>,
        clean_target: &[f32],
        diffusion_step: usize,
        seed: u64,
    ) -> Result<f32, TrainingTaskError> {
        let input = model.input_dim().ok_or(TrainingTaskError::EmptyModel)?;
        let output = model.output_dim().ok_or(TrainingTaskError::EmptyModel)?;
        if !same_dims(self.configured_input, input) || !same_dims(self.configured_output, output) {
            self.configure(input, output)?;
        }

        let alpha_bar = self.schedule.alpha_bar(diffusion_step);
        let step = diffusion_step.min(self.schedule.len().saturating_sub(1));
        let seed = fold_seed(seed);
        let specs_bytes = Self::encode_prepare_uniform(DiffusionPrepareUniform {
            alpha_bar,
            step: step as u32,
            seed,
            input_channels: input.z,
            signal_channels: output.z,
            timestep_channels: self.timestep_channels as u32,
            pixel_count: output.x * output.y,
        });

        let pass = self.ensure_prepare_pass(model, input, output)?;
        if clean_target.len() != pass.expected_target_len {
            return Err(TrainingTaskError::TargetLengthMismatch {
                expected: pass.expected_target_len,
                actual: clean_target.len(),
            });
        }

        model
            .gpu
            .queue
            .write_buffer(&pass.clean_target, 0, bytemuck::cast_slice(clean_target));
        model.gpu.queue.write_buffer(&pass.specs, 0, &specs_bytes);

        let loss = model.train_step_report_with_prepass(|encoder| {
            pass.encode(encoder);
        });
        Ok(loss)
    }

    pub fn train_step_report_batch(
        &mut self,
        model: &mut Model<Training>,
        dataset: &mut GpuDataset,
        step: usize,
        batch_size: usize,
        seed: u64,
    ) -> Result<Option<f32>, TrainingTaskError> {
        self.train_step_batch_inner(model, dataset, step, batch_size, seed, true)
    }

    pub fn train_step_batch(
        &mut self,
        model: &mut Model<Training>,
        dataset: &mut GpuDataset,
        step: usize,
        batch_size: usize,
        seed: u64,
    ) -> Result<(), TrainingTaskError> {
        let _ = self.train_step_batch_inner(model, dataset, step, batch_size, seed, false)?;
        Ok(())
    }

    fn train_step_batch_inner(
        &mut self,
        model: &mut Model<Training>,
        dataset: &mut GpuDataset,
        step: usize,
        batch_size: usize,
        seed: u64,
        report_last_loss: bool,
    ) -> Result<Option<f32>, TrainingTaskError> {
        if batch_size == 0 {
            return Err(TrainingTaskError::InvalidBatchSize { batch_size });
        }
        let input = model.input_dim().ok_or(TrainingTaskError::EmptyModel)?;
        let output = model.output_dim().ok_or(TrainingTaskError::EmptyModel)?;
        if !same_dims(self.configured_input, input) || !same_dims(self.configured_output, output) {
            self.configure(input, output)?;
        }
        let schedule = self.schedule.clone();
        let schedule_len = schedule.len();
        let timestep_channels = self.timestep_channels as u32;

        let pass = self.ensure_prepare_pass(model, input, output)?;
        if dataset.sample_len() != pass.expected_target_len {
            return Err(TrainingTaskError::TargetLengthMismatch {
                expected: pass.expected_target_len,
                actual: dataset.sample_len(),
            });
        }
        let gpu = model.gpu.clone();

        let mut last_loss = None;
        let sample_count = dataset.sample_count();
        model.begin_batch_accumulation();
        for batch_offset in 0..batch_size {
            let sample_index = (step * batch_size + batch_offset) % sample_count;
            let diffusion_step = diffusion_step_for(step, batch_size, batch_offset, schedule_len);
            let alpha_bar = schedule.alpha_bar(diffusion_step);
            let step_seed = seed ^ ((batch_offset as u64) << 32) ^ sample_index as u64;
            let specs_bytes = Self::encode_prepare_uniform(DiffusionPrepareUniform {
                alpha_bar,
                step: diffusion_step as u32,
                seed: fold_seed(step_seed),
                input_channels: input.z,
                signal_channels: output.z,
                timestep_channels,
                pixel_count: output.x * output.y,
            });
            model.gpu.queue.write_buffer(&pass.specs, 0, &specs_bytes);

            let is_last = batch_offset + 1 == batch_size;
            if is_last && report_last_loss {
                last_loss = model.train_step_report_with_prepass_no_opt(|encoder| {
                    pass.encode_with_dataset(encoder, gpu.as_ref(), dataset, sample_index)
                        .expect("diffusion dataset sample copy should be valid");
                });
            } else {
                model.train_step_with_prepass_no_opt(|encoder| {
                    pass.encode_with_dataset(encoder, gpu.as_ref(), dataset, sample_index)
                        .expect("diffusion dataset sample copy should be valid");
                });
            }
        }
        model.finish_batch_accumulation(batch_size);

        Ok(last_loss)
    }

    fn ensure_prepare_pass(
        &mut self,
        model: &Model<Training>,
        input: Dim3,
        output: Dim3,
    ) -> Result<&mut DiffusionPreparePass, TrainingTaskError> {
        let model_input = model
            .layers
            .first()
            .ok_or(TrainingTaskError::EmptyModel)?
            .buffers
            .forward[0]
            .clone();
        let target_noise = model
            .loss_layer
            .as_ref()
            .ok_or(TrainingTaskError::EmptyModel)?
            .buffers
            .forward[1]
            .clone();

        let needs_rebuild = match self.prepare_pass.as_ref() {
            None => true,
            Some(state) => {
                !Arc::ptr_eq(&state.model_input, &model_input)
                    || !Arc::ptr_eq(&state.target_noise, &target_noise)
                    || state.expected_target_len != output.length() as usize
            }
        };

        if needs_rebuild {
            self.prepare_pass = Some(Self::build_prepare_pass(
                model,
                input,
                output,
                model_input,
                target_noise,
            ));
        }

        Ok(self.prepare_pass.as_mut().expect("prepare pass missing"))
    }

    fn build_prepare_pass(
        model: &Model<Training>,
        input: Dim3,
        output: Dim3,
        model_input: Arc<wgpu::Buffer>,
        target_noise: Arc<wgpu::Buffer>,
    ) -> DiffusionPreparePass {
        let device = &model.gpu.device;
        let expected_target_len = output.length() as usize;
        let target_bytes = (expected_target_len * std::mem::size_of::<f32>()) as u64;
        let specs_size = DiffusionPrepareUniform::SHADER_SIZE.get() as u64;

        let clean_target = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diffusion_clean_target"),
            size: target_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let specs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diffusion_prepare_specs"),
            size: specs_size.max(4),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("diffusion_prepare_bgl"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(specs_size.max(4)).unwrap(),
                        ),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("diffusion_prepare_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: clean_target.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: specs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: model_input.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: target_noise.as_ref().as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diffusion_prepare"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shader/diffusion_prepare.wgsl"
            ))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("diffusion_prepare_layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("diffusion_prepare_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("diffusion_prepare"),
            cache: None,
            compilation_options: Default::default(),
        });

        let workgroups = input.length().div_ceil(64);
        DiffusionPreparePass {
            pipeline,
            bind_group,
            clean_target,
            specs,
            model_input,
            target_noise,
            expected_target_len,
            workgroups,
        }
    }

    fn encode_prepare_uniform(specs: DiffusionPrepareUniform) -> Vec<u8> {
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&specs)
            .expect("failed to encode diffusion prepare uniforms");
        buffer.into_inner()
    }
}

impl DiffusionPreparePass {
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("diffusion_prepare_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.workgroups, 1, 1);
    }

    fn encode_with_dataset(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &crate::gpu_context::GpuContext,
        dataset: &mut GpuDataset,
        sample_index: usize,
    ) -> Result<(), super::GpuDatasetError> {
        dataset.copy_sample_to(gpu, encoder, sample_index, &self.clean_target)?;
        self.encode(encoder);
        Ok(())
    }
}

impl TrainingTask for DiffusionTask {
    fn name(&self) -> &'static str {
        "diffusion"
    }

    fn configure(&mut self, input: Dim3, output: Dim3) -> Result<(), TrainingTaskError> {
        if input.x != output.x || input.y != output.y {
            return Err(TrainingTaskError::InvalidLayout {
                input,
                output,
                message: "diffusion requires matching spatial input/output dims",
            });
        }
        if input.z < output.z {
            return Err(TrainingTaskError::InvalidLayout {
                input,
                output,
                message: "diffusion requires input channels >= output channels",
            });
        }
        self.timestep_channels = input.z.saturating_sub(output.z) as usize;
        self.configured_input = Some(input);
        self.configured_output = Some(output);
        self.prepare_pass = None;
        self.pass_specs = vec![TaskPassSpec {
            label: "diffusion_prepare",
            entrypoint: "diffusion_prepare",
            workgroups: Workgroups::x(input.length().div_ceil(64)),
        }];
        Ok(())
    }

    fn pass_specs(&self) -> &[TaskPassSpec] {
        &self.pass_specs
    }
}

fn fold_seed(seed: u64) -> u32 {
    let mixed = seed ^ (seed >> 32);
    (mixed as u32)
        .wrapping_mul(1664525)
        .wrapping_add(1013904223)
}

fn diffusion_step_for(
    step: usize,
    batch_size: usize,
    batch_offset: usize,
    schedule_len: usize,
) -> usize {
    if schedule_len == 0 {
        0
    } else {
        step.wrapping_mul(batch_size).wrapping_add(batch_offset) % schedule_len
    }
}

fn same_dims(saved: Option<Dim3>, target: Dim3) -> bool {
    let Some(saved) = saved else {
        return false;
    };
    saved.x == target.x && saved.y == target.y && saved.z == target.z
}

#[cfg(test)]
mod tests {
    use super::DiffusionTask;
    use crate::gpu_context::GpuContext;
    use crate::model::{ActivationMethod, ActivationType, Dim3, LayerTypes, LossMethod, Model};
    use crate::training::{GpuDataset, LinearNoiseSchedule, TrainingTask};
    use std::sync::Arc;

    #[test]
    fn diffusion_task_rejects_spatial_mismatch() {
        let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
        let mut task = DiffusionTask::new(schedule);
        let err = task
            .configure(Dim3::new((32, 32, 8)), Dim3::new((16, 16, 3)))
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("matching spatial input/output dims")
        );
    }

    #[test]
    fn diffusion_task_computes_timestep_channels() {
        let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
        let mut task = DiffusionTask::new(schedule);
        task.configure(Dim3::new((32, 32, 8)), Dim3::new((32, 32, 3)))
            .unwrap();
        assert_eq!(task.timestep_channels(), 5);
        assert_eq!(task.pass_specs().len(), 1);
        assert_eq!(task.pass_specs()[0].entrypoint, "diffusion_prepare");
        assert_eq!(
            task.pass_specs()[0].workgroups,
            super::Workgroups::x(32 * 32 * 8 / 64)
        );
    }

    #[test]
    fn diffusion_task_gpu_prepare_pass_executes() {
        pollster::block_on(async {
            let gpu = Arc::new(GpuContext::new_headless().await);
            let mut model = Model::new_training(gpu, 0.01, 1, LossMethod::MeanSquared).await;
            model
                .add_layer(LayerTypes::Activation(ActivationType::new(
                    ActivationMethod::Linear,
                    Dim3::new((4, 4, 3)),
                )))
                .unwrap();
            model.build().unwrap();

            let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
            let mut task = DiffusionTask::new(schedule);
            let samples = vec![vec![0.0f32; 4 * 4 * 3]];
            let mut dataset = GpuDataset::from_samples(model.gpu.as_ref(), samples, 4 * 4 * 3)
                .expect("failed to upload gpu dataset");
            let loss = task
                .train_step_report_batch(&mut model, &mut dataset, 0, 1, 42)
                .unwrap();
            assert!(loss.is_some());
            assert!(loss.unwrap().is_finite());
        });
    }

    #[test]
    fn diffusion_step_progression_does_not_alias_single_batch() {
        let steps: Vec<usize> = (0..8)
            .map(|step| super::diffusion_step_for(step, 1, 0, 8))
            .collect();
        assert_eq!(steps, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
