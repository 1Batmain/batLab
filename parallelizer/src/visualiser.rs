use std::sync::Arc;
use std::{fs, path::PathBuf};

use wgpu::{Color, Device, Queue};

use crate::types::Dim3;

#[derive(Debug, Clone, Copy)]
pub struct ModelVisualState {
    pub layer_count: usize,
    pub has_training_spec: bool,
    pub training_initialized: bool,
    pub has_loss_pipeline: bool,
    pub infer_revision: u64,
    pub train_revision: u64,
}

pub struct Visualiser {
    device: Arc<Device>,
    queue: Arc<Queue>,
    clear_color: Color,
    last_input_len: usize,
    last_output_len: usize,
    last_state: Option<ModelVisualState>,
    frame_id: u64,
    output_dir: PathBuf,
}

impl std::fmt::Debug for Visualiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Visualiser")
            .field("clear_color", &self.clear_color)
            .field("last_input_len", &self.last_input_len)
            .field("last_output_len", &self.last_output_len)
            .field("last_state", &self.last_state)
            .field("frame_id", &self.frame_id)
            .field("output_dir", &self.output_dir)
            .finish()
    }
}

impl Visualiser {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            clear_color: Color::BLACK,
            last_input_len: 0,
            last_output_len: 0,
            last_state: None,
            frame_id: 0,
            output_dir: PathBuf::from("visualisation_frames"),
        }
    }

    fn channel_value(data: &[f32], dim: Dim3, x: u32, y: u32, c: u32) -> f32 {
        if x >= dim.x || y >= dim.y || c >= dim.z {
            return 0.0;
        }
        let idx = ((y * dim.x + x) * dim.z + c) as usize;
        data.get(idx).copied().unwrap_or(0.0)
    }

    fn normalize(value: f32, min_value: f32, max_value: f32) -> u8 {
        if (max_value - min_value).abs() < f32::EPSILON {
            return 0;
        }
        let scaled = ((value - min_value) / (max_value - min_value)).clamp(0.0, 1.0);
        (scaled * 255.0) as u8
    }

    fn tensor_min_max(data: &[f32]) -> (f32, f32) {
        let mut min_value = f32::INFINITY;
        let mut max_value = f32::NEG_INFINITY;
        for value in data.iter().copied() {
            if value < min_value {
                min_value = value;
            }
            if value > max_value {
                max_value = value;
            }
        }
        if min_value.is_infinite() || max_value.is_infinite() {
            (0.0, 1.0)
        } else {
            (min_value, max_value)
        }
    }

    fn pixel_rgb(data: &[f32], dim: Dim3, x: u32, y: u32, min_value: f32, max_value: f32) -> [u8; 3] {
        if dim.z == 0 {
            return [0, 0, 0];
        }

        let red = Self::channel_value(data, dim, x, y, 0);
        let green = if dim.z > 1 {
            Self::channel_value(data, dim, x, y, 1)
        } else {
            red
        };
        let blue = if dim.z > 2 {
            Self::channel_value(data, dim, x, y, 2)
        } else if dim.z > 1 {
            (red + green) * 0.5
        } else {
            red
        };

        [
            Self::normalize(red, min_value, max_value),
            Self::normalize(green, min_value, max_value),
            Self::normalize(blue, min_value, max_value),
        ]
    }

    fn write_side_by_side_ppm(&mut self, input: &[f32], input_dim: Dim3, output: &[f32], output_dim: Dim3) {
        let in_width = input_dim.x.max(1);
        let in_height = input_dim.y.max(1);
        let out_width = output_dim.x.max(1);
        let out_height = output_dim.y.max(1);
        let width = in_width + out_width;
        let height = in_height.max(out_height);

        let (in_min, in_max) = Self::tensor_min_max(input);
        let (out_min, out_max) = Self::tensor_min_max(output);

        let mut rgb = vec![0_u8; (width * height * 3) as usize];

        for y in 0..height {
            for x in 0..in_width {
                if y < in_height {
                    let pixel = Self::pixel_rgb(input, input_dim, x, y, in_min, in_max);
                    let idx = ((y * width + x) * 3) as usize;
                    rgb[idx] = pixel[0];
                    rgb[idx + 1] = pixel[1];
                    rgb[idx + 2] = pixel[2];
                }
            }

            for x in 0..out_width {
                if y < out_height {
                    let pixel = Self::pixel_rgb(output, output_dim, x, y, out_min, out_max);
                    let idx = ((y * width + (x + in_width)) * 3) as usize;
                    rgb[idx] = pixel[0];
                    rgb[idx + 1] = pixel[1];
                    rgb[idx + 2] = pixel[2];
                }
            }
        }

        let _ = fs::create_dir_all(&self.output_dir);
        let file_path = self
            .output_dir
            .join(format!("inference_{:06}.ppm", self.frame_id));

        let mut bytes = Vec::with_capacity(32 + rgb.len());
        bytes.extend_from_slice(format!("P6\n{} {}\n255\n", width, height).as_bytes());
        bytes.extend_from_slice(&rgb);

        if fs::write(&file_path, bytes).is_ok() {
            println!("visualiser frame saved to {}", file_path.display());
        }
    }

    pub fn on_inference(
        &mut self,
        input: &[f32],
        input_dim: Dim3,
        output: &[f32],
        output_dim: Dim3,
        state: ModelVisualState,
    ) {
        self.last_input_len = input.len();
        self.last_output_len = output.len();
        self.last_state = Some(state);
        self.write_side_by_side_ppm(input, input_dim, output, output_dim);
        self.frame_id = self.frame_id.saturating_add(1);
    }

    pub fn set_clear_color(&mut self, clear_color: Color) {
        self.clear_color = clear_color;
    }

    pub fn clear_color(&self) -> Color {
        self.clear_color
    }

    pub fn last_input_len(&self) -> usize {
        self.last_input_len
    }

    pub fn last_output_len(&self) -> usize {
        self.last_output_len
    }

    pub fn last_state(&self) -> Option<ModelVisualState> {
        self.last_state
    }

    pub fn device(&self) -> &Device {
        self.device.as_ref()
    }

    pub fn queue(&self) -> &Queue {
        self.queue.as_ref()
    }
}
