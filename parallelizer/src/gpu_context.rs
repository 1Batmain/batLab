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

    #[cfg(feature = "visualisation")]
    pub fn create_surface<'window>(
        &self,
        window: &'window winit::window::Window,
    ) -> Result<wgpu::Surface<'window>, String> {
        self._instance
            .create_surface(window)
            .map_err(|error| format!("failed to create wgpu surface: {}", error))
    }
}
