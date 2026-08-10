//! File purpose: Creates and configures the shared headless WGPU context used by training and rendering paths.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::profile::{PassProfiler, ProfileRun};

/// Which limits this process asks the device for.
///
/// `request_device` does not hand out what the adapter can do — it hands out
/// what you ask for, and asking for nothing means asking for
/// `wgpu::Limits::default()`, the WebGPU baseline: 256 MiB per buffer, 128 MiB
/// per storage binding. That was the engine's only mode, and it was the right
/// one for **inference**, which has to run in a visitor's browser on their card
/// (see CLAUDE.md). It was never right for **training**, which runs here: the
/// same Mac's adapter allows 28 GiB, and the baseline was capping the batch at
/// 341 on `Color_Diffusion_XL` by a *binding* limit rather than by memory, and
/// forcing the dataset to be streamed in 128 MiB chunks when the whole of
/// CIFAR-10 RGB in 8-bit is 147 MiB and would sit resident.
///
/// So the limits became a choice. The two profiles are not a performance dial:
/// they are two different questions, and both have to keep working.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuLimitsProfile {
    /// Everything the adapter reports. The default, and what training wants.
    #[default]
    Native,
    /// `wgpu::Limits::default()` — the WebGPU contract, what a browser grants
    /// without being asked. Running under it is how one checks that a model
    /// still fits the target the whole engine is shaped around.
    Web,
}

impl GpuLimitsProfile {
    pub fn label(self) -> &'static str {
        match self {
            GpuLimitsProfile::Native => "native",
            GpuLimitsProfile::Web => "web",
        }
    }

    /// Parses the `--gpu-limits` value. `Err` carries what was accepted.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "native" | "adapter" => Ok(GpuLimitsProfile::Native),
            "web" | "webgpu" | "default" => Ok(GpuLimitsProfile::Web),
            other => Err(format!(
                "unknown gpu limits profile '{other}' (expected 'native' or 'web')"
            )),
        }
    }

    fn limits(self, adapter: &wgpu::Adapter) -> wgpu::Limits {
        match self {
            GpuLimitsProfile::Native => adapter.limits(),
            GpuLimitsProfile::Web => wgpu::Limits::default(),
        }
    }

    fn code(self) -> u8 {
        match self {
            GpuLimitsProfile::Native => 0,
            GpuLimitsProfile::Web => 1,
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            1 => GpuLimitsProfile::Web,
            _ => GpuLimitsProfile::Native,
        }
    }

    /// Sets the profile every later [`GpuContext::new_headless`] will use.
    ///
    /// A process-wide setting rather than a parameter, deliberately. The limits
    /// belong to the *device*, this application opens exactly one, and the
    /// choice is made once from the command line before anything is built.
    /// Threading it instead would put a `GpuLimitsProfile` through
    /// `App::with_storage`, the run loop and every screen of the TUI, for a
    /// value that cannot change while the process lives.
    ///
    /// Tests that care about a specific profile do not touch this: they call
    /// [`GpuContext::new_headless_with`] and say which one they mean.
    pub fn set_process_default(self) {
        PROCESS_DEFAULT.store(self.code(), Ordering::Relaxed);
    }

    /// What [`GpuContext::new_headless`] will ask for.
    pub fn process_default() -> Self {
        Self::from_code(PROCESS_DEFAULT.load(Ordering::Relaxed))
    }
}

static PROCESS_DEFAULT: AtomicU8 = AtomicU8::new(0);

/// Whether [`GpuContext::new_headless`] should open a device able to time its
/// own compute passes.
///
/// A process-wide switch for the same reason the limits profile is one: the
/// `TIMESTAMP_QUERY` feature belongs to the *device*, this application opens
/// exactly one, and the choice is made from the command line before anything is
/// built. Off by default, and off means the query set is never created and
/// [`GpuContext::compute_pass`] is what it always was.
static PROFILING_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Ask every later [`GpuContext::new_headless`] for pass timing.
///
/// Must be called before the first device is opened; afterwards it has no
/// effect, because features cannot be added to a live device.
pub fn request_pass_profiling(enabled: bool) {
    PROFILING_REQUESTED.store(enabled, Ordering::Relaxed);
}

/// Whether pass profiling has been asked for. Asked for, not granted — an
/// adapter without `TIMESTAMP_QUERY` leaves [`GpuContext::can_profile`] false.
pub fn pass_profiling_requested() -> bool {
    PROFILING_REQUESTED.load(Ordering::Relaxed)
}

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
    /// The profile this device was actually opened under.
    ///
    /// The one that was *granted*, not the one that was asked for: a device
    /// request can be refused, and the fallback below then leaves the process
    /// running under limits nobody chose. A run banner that printed the
    /// intention rather than the outcome would be worse than printing nothing.
    limits_profile: GpuLimitsProfile,
    /// What has crossed the host↔GPU boundary through this context.
    ///
    /// It lives here because this is the one object every path already shares:
    /// a counter anywhere else would have to be threaded through the sampler,
    /// the trainer and the dataset separately, and would miss whichever of them
    /// was added next.
    transfers: crate::transfers::TransferCounters,
    /// Present only when the process asked for profiling *and* the adapter
    /// granted `TIMESTAMP_QUERY`. `None` is the normal, zero-cost state.
    profiler: Option<PassProfiler>,
}

impl GpuContext {
    /// A context under the process-wide profile — see
    /// [`GpuLimitsProfile::set_process_default`].
    pub async fn new_headless() -> Self {
        Self::new_headless_with(GpuLimitsProfile::process_default()).await
    }

    pub async fn new_headless_with(profile: GpuLimitsProfile) -> Self {
        Self::open(profile, pass_profiling_requested()).await
    }

    /// A context that times its compute passes, whatever the process default.
    ///
    /// Tests that need the instrument say so here rather than mutating the
    /// process-wide switch, which would leak into whatever else the harness runs
    /// in the same process.
    pub async fn new_headless_profiling() -> Self {
        Self::open(GpuLimitsProfile::process_default(), true).await
    }

    async fn open(profile: GpuLimitsProfile, want_profiling: bool) -> Self {
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
        // Asked for only when the command line asked for it, and only when the
        // adapter has it. A missing `TIMESTAMP_QUERY` is not a reason to refuse
        // to run: it is a reason to say the instrument is unavailable, which the
        // harness does, rather than to hand back an approximation dressed as a
        // measurement.
        let timestamps = want_profiling && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if want_profiling && !timestamps {
            eprintln!(
                "[gpu] this adapter does not expose TIMESTAMP_QUERY — per-pass GPU timing is \
                 unavailable on this backend."
            );
        }
        let required_features = if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let descriptor = wgpu::DeviceDescriptor {
            required_limits: profile.limits(&adapter),
            required_features,
            ..Default::default()
        };
        // The fallback is not decoration. `adapter.limits()` is what the
        // adapter *reports*, and a backend is free to refuse a device that asks
        // for all of it; a panic there would make `--gpu-limits native` a way to
        // lose the application rather than a way to ask for more. Falling back
        // to the WebGPU baseline always works — it is the floor every backend
        // wgpu supports must meet — and the profile recorded below is the one
        // that was granted, so the banner tells the truth.
        let (device, queue, limits_profile) = match adapter.request_device(&descriptor).await {
            Ok((device, queue)) => (device, queue, profile),
            Err(err) if profile != GpuLimitsProfile::Web => {
                eprintln!(
                    "[gpu] the adapter refused its own limits ({err}) — falling back to the \
                     WebGPU baseline. Batch ceilings and dataset residency will be the web ones."
                );
                let (device, queue) = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        required_limits: GpuLimitsProfile::Web.limits(&adapter),
                        required_features,
                        ..Default::default()
                    })
                    .await
                    .expect("failed to request a wgpu device");
                (device, queue, GpuLimitsProfile::Web)
            }
            Err(err) => panic!("failed to request a wgpu device: {err}"),
        };
        device.on_uncaptured_error(std::sync::Arc::new(|err| {
            eprintln!("[gpu] uncaptured wgpu error: {err}");
        }));
        let gpu_specs = GpuSpecs::new(&adapter);
        let profiler = timestamps.then(|| PassProfiler::new(&device, &queue));

        Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            gpu_specs,
            limits_profile,
            transfers: Default::default(),
            profiler,
        }
    }

    /// The profile this device was granted.
    pub fn limits_profile(&self) -> GpuLimitsProfile {
        self.limits_profile
    }

    /// Run `op` with a WebGPU **validation error scope** active, returning any
    /// validation error `op` raised as a string instead of letting it vanish.
    ///
    /// This exists for one host and one failure mode: a browser. There, a shader
    /// the browser's WGSL compiler rejects (it is stricter than the native
    /// `naga` → Metal path this engine is otherwise validated on) or a limit
    /// tripped at dispatch does **not** raise — the pipeline is quietly
    /// invalidated, its dispatches become no-ops, the output buffer keeps the
    /// zeros it was created with, and nothing is said. Wrapped in a scope, that
    /// same error is captured and handed back, so a caller (`batlab_web`) can put
    /// it on the page rather than animate a dead model. See
    /// `docs/reports/WEB_PORT.md`, "le bug de l'inférence".
    ///
    /// Pipeline-creation errors are captured synchronously as `op` runs;
    /// dispatch/submit errors are flushed with an empty submit before the scope
    /// is read. On wasm the browser drives the returned future to completion; a
    /// native caller must be pumping the device (this is only used from wasm).
    pub async fn guarded<F, T>(&self, op: F) -> Result<T, String>
    where
        F: core::future::Future<Output = T>,
    {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out = op.await;
        // Flush any queued work so a dispatch-time validation error lands in the
        // scope before we pop it. Empty is fine — it is a synchronisation point,
        // not work.
        self.queue.submit(core::iter::empty());
        match scope.pop().await {
            Some(err) => Err(err.to_string()),
            None => Ok(out),
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

    // -----------------------------------------------------------------------
    // Compute passes
    // -----------------------------------------------------------------------

    /// Begin a compute pass, time it when the profiler is armed, and run `body`
    /// inside it.
    ///
    /// Every `begin_compute_pass` in the engine goes through here, for exactly
    /// the reason every `write_buffer` does: a timing table is only true if
    /// there is no second door, and `nothing_in_the_engine_opens_an_untimed_pass`
    /// is the mechanical check that there is not.
    ///
    /// `label` is a **closure**: building `L13 Convolution · conv_back_weights`
    /// allocates a String, and a step encodes 151 passes. With profiling off the
    /// closure is never called, so an unprofiled run allocates nothing here and
    /// the descriptor is the same `Default::default()` it always was.
    ///
    /// `workgroups` is the logical dispatch count *before* [`crate::model`]'s
    /// 2-D folding — it is carried so the table can tell a pass that is slow
    /// because it is big from one that is slow because it is starved.
    pub fn compute_pass<L, F>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        workgroups: u32,
        label: L,
        body: F,
    ) where
        L: FnOnce() -> String,
        F: FnOnce(&mut wgpu::ComputePass<'_>),
    {
        match self.profiler.as_ref().and_then(|p| p.reserve(label, workgroups)) {
            Some(timestamp_writes) => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(timestamp_writes),
                });
                body(&mut pass);
            }
            None => {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                body(&mut pass);
            }
        }
    }

    /// Block until the GPU has finished everything submitted so far.
    ///
    /// The engine never needs this — a training step is fire-and-forget and only
    /// the loss readback synchronises. A *measurement* does: without it, host
    /// time around a step measures how long it took to encode, not how long it
    /// took to run.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Whether this device can time its passes at all.
    pub fn can_profile(&self) -> bool {
        self.profiler.is_some()
    }

    /// Start timing: passes encoded from now on carry timestamps.
    ///
    /// Returns `false` when the device has no timing instrument, so a caller
    /// cannot mistake "nothing was recorded" for "nothing took any time".
    pub fn arm_profiler(&self) -> bool {
        match self.profiler.as_ref() {
            Some(profiler) => {
                profiler.arm();
                true
            }
            None => false,
        }
    }

    /// Encode the query resolve. Called on the same encoder as the passes, right
    /// before `finish()`; a no-op when the profiler is off or not armed.
    pub fn resolve_profiler(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(profiler) = self.profiler.as_ref() {
            profiler.resolve(encoder);
        }
    }

    /// Read the timings back and disarm. Blocks until the GPU is done with the
    /// submission the passes were in.
    pub fn collect_profile(&self) -> Option<ProfileRun> {
        let profiler = self.profiler.as_ref()?;
        if !profiler.is_armed() {
            return None;
        }
        Some(profiler.collect(&self.device))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profile_is_parsed_from_its_name_and_nothing_else() {
        assert_eq!(
            GpuLimitsProfile::parse("native").unwrap(),
            GpuLimitsProfile::Native
        );
        assert_eq!(
            GpuLimitsProfile::parse(" WEB ").unwrap(),
            GpuLimitsProfile::Web
        );
        // A typo must not silently pick a profile: the two differ by four
        // orders of magnitude, and a run that quietly took the wrong one would
        // look like the flag had no effect.
        let err = GpuLimitsProfile::parse("naive").unwrap_err();
        assert!(err.contains("native") && err.contains("web"), "{err}");
    }

    /// Both profiles must open a device, and native must not grant *less*.
    ///
    /// The second half is the claim the whole flag rests on. It is written as
    /// `>=` rather than `>` because an adapter whose own limits are the WebGPU
    /// baseline is a legitimate machine — the flag then changes nothing, which
    /// is the correct outcome, not a failure.
    #[test]
    fn both_profiles_open_a_device_and_native_never_grants_less() {
        pollster::block_on(async {
            let web = GpuContext::new_headless_with(GpuLimitsProfile::Web).await;
            let native = GpuContext::new_headless_with(GpuLimitsProfile::Native).await;

            // What was GRANTED, not what was asked for: a refused native
            // request falls back, and this must still hold afterwards.
            assert_eq!(web.limits_profile(), GpuLimitsProfile::Web);

            let web_limits = web.device().limits();
            let native_limits = native.device().limits();
            assert!(native_limits.max_buffer_size >= web_limits.max_buffer_size);
            assert!(
                native_limits.max_storage_buffer_binding_size
                    >= web_limits.max_storage_buffer_binding_size
            );
            assert_eq!(
                web_limits.max_buffer_size,
                wgpu::Limits::default().max_buffer_size,
                "the web profile must be the WebGPU baseline exactly — it is what \
                 answers 'would this run in a browser'"
            );
        });
    }

    /// The process default is what `new_headless` picks up, and it starts at
    /// native — the profile training wants.
    #[test]
    fn the_process_default_starts_native() {
        // Not mutated here: `set_process_default` is global, and a test that
        // changed it would leak into whatever else the harness runs in this
        // process. The setter is exercised by the CLI, which is where it is
        // ever called.
        assert_eq!(GpuLimitsProfile::default(), GpuLimitsProfile::Native);
    }
}
