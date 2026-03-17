#[derive(Debug)]
pub struct GpuContext {
    pub(crate) _instance: wgpu::Instance,
    pub(crate) _adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
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
