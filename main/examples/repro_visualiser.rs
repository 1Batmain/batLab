//! Minimal reproduction of the visualiser crash: spawn a window from a non-main
//! thread exactly the way the TUI does, then observe what happens.

use bat_building::GpuContext;
use bat_building::visualiser::spawn_window;
use std::sync::Arc;
use std::time::Duration;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = Arc::new(rt.block_on(GpuContext::new_headless()));
    eprintln!("[repro] gpu ready: {}", gpu.adapter().get_info().name);

    let w = 32u32;
    let h = 32u32;
    let c = 3u32;
    let buf = Arc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("repro_output"),
        size: (w * h * c * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    eprintln!("[repro] spawning visualiser window...");
    let handle = spawn_window(gpu, buf, w, h, c, "Repro".to_string());

    for i in 0..20 {
        std::thread::sleep(Duration::from_millis(250));
        eprintln!("[repro] tick {i} closed={}", handle.is_closed());
    }
    eprintln!("[repro] done");
}
