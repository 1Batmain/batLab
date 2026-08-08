//! File purpose: Creates and configures the shared headless WGPU context used by training and rendering paths.

#[derive(Debug)]
pub struct GpuSpecs {
    pub(crate) device_name: String,
    pub(crate) memory_size: u64,
}
impl GpuSpecs {
    pub fn new(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let limits = adapter.limits();
        Self {
            device_name: info.name,
            // For now we expose the adapter's max single-buffer allocation size as
            // the project-level memory capacity metric.
            memory_size: limits.max_buffer_size,
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn memory_size(&self) -> u64 {
        self.memory_size
    }
}

#[derive(Debug)]
pub struct GpuContext {
    pub(crate) _instance: wgpu::Instance,
    pub(crate) _adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) gpu_specs: GpuSpecs,
    /// What has crossed the host↔GPU boundary through this context.
    ///
    /// It lives here because this is the one object every path already shares:
    /// a counter anywhere else would have to be threaded through the sampler,
    /// the trainer and the dataset separately, and would miss whichever of them
    /// was added next.
    transfers: crate::transfers::TransferCounters,
}

impl GpuContext {
    pub async fn new_headless() -> Self {
        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        };

        // Prefer primary native backends first to avoid noisy GL/X11 probing.
        let primary_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let (instance, adapter) = match primary_instance.request_adapter(&adapter_options).await {
            Ok(adapter) => (primary_instance, adapter),
            Err(_) => {
                let fallback_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    ..Default::default()
                });
                let adapter = fallback_instance
                    .request_adapter(&adapter_options)
                    .await
                    .expect("failed to request a wgpu adapter");
                (fallback_instance, adapter)
            }
        };
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("failed to request a wgpu device");
        device.on_uncaptured_error(std::sync::Arc::new(|err| {
            eprintln!("[gpu] uncaptured wgpu error: {err}");
        }));
        let gpu_specs = GpuSpecs::new(&adapter);

        Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            gpu_specs,
            transfers: Default::default(),
        }
    }

    /// Upload bytes to a GPU buffer, and count them.
    ///
    /// Every host→device write in the engine goes through here rather than
    /// through `queue.write_buffer` directly. That is the whole discipline: the
    /// counter is only true if there is no second door, and `grep -n
    /// 'queue.write_buffer' crates/batlab-core/src` outside of tests is the way
    /// to check there is not.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.transfers.record_write(data.len() as u64);
        self.queue.write_buffer(buffer, offset, data);
    }

    /// Submit command buffers, and count the submission.
    pub fn submit<I>(&self, command_buffers: I) -> wgpu::SubmissionIndex
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        let buffers: Vec<wgpu::CommandBuffer> = command_buffers.into_iter().collect();
        self.transfers.record_submit(buffers.len() as u64);
        self.queue.submit(buffers)
    }

    /// Record a device→host readback of `bytes`.
    ///
    /// Separate from a helper that performs one, because a readback is three
    /// steps (copy into a staging buffer, `map_async`, block on a poll) spread
    /// across `model::debug` and the loss path, and the counter belongs at the
    /// point where the bytes are known.
    pub fn record_readback(&self, bytes: u64) {
        self.transfers.record_read(bytes);
    }

    /// A reading of the counters — see [`crate::transfers::TransferSnapshot`].
    pub fn transfers(&self) -> crate::transfers::TransferSnapshot {
        self.transfers.snapshot()
    }

    /// Access to the underlying wgpu device for callers outside the crate.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access to the underlying wgpu queue for callers outside the crate.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Provides access to the adapter used to create the device/queue.
    ///
    /// This is mostly useful for surface configuration, where the supported
    /// formats must be queried.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self._adapter
    }

    /// Provides access to the wgpu instance for surface creation.
    pub fn instance(&self) -> &wgpu::Instance {
        &self._instance
    }

    /// Provides access to cached GPU specification details used by the app.
    pub fn specs(&self) -> &GpuSpecs {
        &self.gpu_specs
    }
}
