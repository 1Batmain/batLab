#[derive(Debug)]
pub struct GpuContext {
    pub(crate) _instance: wgpu::Instance,
    pub(crate) _adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new_headless() -> Self {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("failed to request a wgpu adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("failed to request a wgpu device");

        Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
        }
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
}
