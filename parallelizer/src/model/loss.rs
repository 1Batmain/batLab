#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Loss {
    MeanSquare,
}

struct LossLayer {
    pub(crate) ty: Loss,
    pub(crate) buffers: Vec<Arc<Buffer>>,
    pub(crate) shader: ShaderModule,
    pub(crate) pipeline: Option<ComputePipeline>,
    pub(crate) num_workgroups: u32,
    pub(crate) bind_group: Option<BindGroup>,
}

impl Loss {
    pub(crate) fn new(ty: Loss) -> Self {
        Self { ty }
    }
}
