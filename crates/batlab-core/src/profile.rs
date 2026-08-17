//! Real GPU time, compute pass by compute pass. `TIMESTAMP_QUERY` lets a pass
//! write the GPU's own clock at its begin and end; this allocates one query set,
//! hands a slot pair per pass, and reads them back after the submit — GPU time,
//! measured by the GPU, not host time around a submit.
//!
//! Three quantities out of one armed step, all different:
//! - **`nanos` per pass** — `end − begin`, the pass's own cost.
//! - **`span`** — last `end` minus first `begin`: the whole step's GPU duration,
//!   INCLUDING what sits between passes (wgpu's implicit inter-pass barrier).
//! - **`span − Σ nanos`** — what the step spends BETWEEN passes, the "per-pass
//!   floor" `PERF_CONVOLUTION.md` §5.4 could not measure.
//!
//! It does NOT measure anything outside the profiled encoder (dataset copies,
//! uniform uploads, loss readback); those are host wall clock minus `span`.
//!
//! Off, it costs nothing: [`crate::GpuContext`] holds an `Option<PassProfiler>`,
//! so `compute_pass` is a plain `begin_compute_pass` plus one `is_some`, and the
//! label closure is never called.

use std::sync::Mutex;

/// One compute pass, as the GPU timed it.
#[derive(Debug, Clone)]
pub struct PassTiming {
    /// `L13 Convolution · conv_back_weights` — layer position, layer type, the
    /// shader entry point that ran.
    pub label: String,
    /// `end − begin`, in nanoseconds of GPU clock.
    pub nanos: u64,
    /// The logical workgroup count (before `dispatch_grid`'s 2-D fold), kept so a
    /// big pass and a starved one are not confused in the time column alone.
    pub workgroups: u32,
    /// Whether the GPU actually wrote this pass's two timestamps. Metal drops
    /// counter samples for some tiny dispatches and returns zeros instead; zero is
    /// no plausible reading of a clock at 3.76e15, so an unsampled pass is NAMED as
    /// such — counting it free would show up as time between passes, the quantity
    /// the diagnosis turns on.
    pub sampled: bool,
}

/// Everything one armed step yielded.
#[derive(Debug, Clone)]
pub struct ProfileRun {
    pub passes: Vec<PassTiming>,
    /// Last end minus first begin, GPU clock. `0` if fewer than one pass ran.
    pub span_nanos: u64,
    /// Passes that could not be timed because the query set was full. Non-zero
    /// makes every other number in the run a lower bound, so the harness must
    /// say so rather than print a total that quietly omits work.
    pub dropped: u32,
    /// Passes the backend declined to sample — see [`PassTiming::sampled`].
    pub unsampled: u32,
}

impl ProfileRun {
    /// The sum of the passes' own times.
    pub fn attributed_nanos(&self) -> u64 {
        self.passes.iter().map(|p| p.nanos).sum()
    }

    /// GPU time inside the step that is in no pass: barriers and pass setup.
    ///
    /// Saturating rather than signed: on a backend whose begin/end timestamps
    /// are not perfectly nested this could go negative by a few ticks, and a
    /// negative "unattributed" would be noise reported as a discovery.
    pub fn unattributed_nanos(&self) -> u64 {
        self.span_nanos.saturating_sub(self.attributed_nanos())
    }
}

/// Two queries per pass: one at the beginning, one at the end.
const QUERIES_PER_PASS: u32 = 2;

/// `wgpu::QUERY_SET_MAX_QUERIES` is 4096, so this is the ceiling, not a choice.
/// `Color_Diffusion_XL` encodes 151 passes per training step, so the ceiling is
/// ~13× the largest graph this engine currently builds.
const QUERY_CAPACITY: u32 = 4096;

/// `resolve_query_set`'s destination offset must be 256-aligned; we always
/// resolve at 0, but the readback buffer is sized on the same rule.
const RESOLVE_BYTES: u64 = (QUERY_CAPACITY as u64) * 8;

#[derive(Debug, Default)]
struct Recording {
    armed: bool,
    labels: Vec<(String, u32)>,
    dropped: u32,
}

#[derive(Debug)]
pub(crate) struct PassProfiler {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Nanoseconds per tick, from `Queue::get_timestamp_period`. 1.0 on Metal.
    period_ns: f32,
    state: Mutex<Recording>,
}

impl PassProfiler {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pass_profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_CAPACITY,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass_profiler_resolve"),
            size: RESOLVE_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass_profiler_readback"),
            size: RESOLVE_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve,
            readback,
            period_ns: queue.get_timestamp_period(),
            state: Mutex::new(Recording::default()),
        }
    }

    /// Start recording. The next encoded passes get timestamps; earlier ones do
    /// not. Discards whatever a previous arming left.
    pub(crate) fn arm(&self) {
        let mut state = self.state.lock().expect("profiler state poisoned");
        state.armed = true;
        state.labels.clear();
        state.dropped = 0;
    }

    pub(crate) fn is_armed(&self) -> bool {
        self.state.lock().expect("profiler state poisoned").armed
    }

    /// Reserve a slot pair for the pass about to be encoded, or `None` when the
    /// profiler is not armed (and then the pass is encoded exactly as it would
    /// be without this module).
    pub(crate) fn reserve(
        &self,
        label: impl FnOnce() -> String,
        workgroups: u32,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let mut state = self.state.lock().expect("profiler state poisoned");
        if !state.armed {
            return None;
        }
        let index = state.labels.len() as u32;
        if (index + 1) * QUERIES_PER_PASS > QUERY_CAPACITY {
            state.dropped += 1;
            return None;
        }
        state.labels.push((label(), workgroups));
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(index * QUERIES_PER_PASS),
            end_of_pass_write_index: Some(index * QUERIES_PER_PASS + 1),
        })
    }

    /// Encode the resolve and the copy to a mappable buffer. Must be called on
    /// the same encoder as the passes, after the last of them.
    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = {
            let state = self.state.lock().expect("profiler state poisoned");
            if !state.armed {
                return;
            }
            state.labels.len() as u32 * QUERIES_PER_PASS
        };
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, count as u64 * 8);
    }

    /// Read the pairs back and disarm. Blocks until the submission completes —
    /// which is exactly what a caller who wants to time one step wants anyway.
    pub(crate) fn collect(&self, device: &wgpu::Device) -> ProfileRun {
        let (labels, dropped) = {
            let mut state = self.state.lock().expect("profiler state poisoned");
            state.armed = false;
            (std::mem::take(&mut state.labels), state.dropped)
        };
        let empty = |dropped| ProfileRun {
            passes: vec![],
            span_nanos: 0,
            dropped,
            unsampled: 0,
        };
        if labels.is_empty() {
            return empty(dropped);
        }
        let bytes = labels.len() as u64 * QUERIES_PER_PASS as u64 * 8;
        let slice = self.readback.slice(0..bytes);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return empty(dropped);
        }
        match pollster::block_on(async { rx.await }) {
            Ok(Ok(())) => {}
            _ => return empty(dropped),
        }
        let ticks: Vec<u64> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        self.readback.unmap();

        let period = self.period_ns as f64;
        let mut first_begin = u64::MAX;
        let mut last_end = 0u64;
        let mut unsampled = 0u32;
        let passes: Vec<PassTiming> = labels
            .iter()
            .enumerate()
            .map(|(i, (label, workgroups))| {
                let begin = ticks[i * 2];
                let end = ticks[i * 2 + 1];
                // 0 and `u64::MAX` (Metal's failed-sample sentinel) both mean "not
                // sampled": excluded from the span rather than dragging its start to
                // zero (which once produced a 3 761 240 032 ms span). `end < begin` too.
                let sampled = begin != 0
                    && end != 0
                    && begin != u64::MAX
                    && end != u64::MAX
                    && end >= begin;
                if sampled {
                    first_begin = first_begin.min(begin);
                    last_end = last_end.max(end);
                } else {
                    unsampled += 1;
                }
                PassTiming {
                    label: label.clone(),
                    nanos: if sampled {
                        ((end - begin) as f64 * period) as u64
                    } else {
                        0
                    },
                    workgroups: *workgroups,
                    sampled,
                }
            })
            .collect();
        let span_nanos = if first_begin == u64::MAX {
            0
        } else {
            ((last_end - first_begin) as f64 * period) as u64
        };
        ProfileRun {
            passes,
            span_nanos,
            dropped,
            unsampled,
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregation across rounds
// ---------------------------------------------------------------------------

/// One pass, summarised over several armed steps.
#[derive(Debug, Clone)]
pub struct PassSummary {
    pub label: String,
    /// The house estimator (`PERF_CONVOLUTION.md` §5.1): contention can only
    /// *add* time, so the fastest round is the best estimate of the pass's own
    /// cost.
    pub min_nanos: u64,
    pub max_nanos: u64,
    pub workgroups: u32,
    /// How many dispatches carried this label per step. Two passes of one layer
    /// that share an entry point name are counted together, and the column says
    /// so.
    pub invocations: u32,
    /// Rounds in which every occurrence of this label was sampled. `0` means the
    /// backend never timed it and `min_nanos` is meaningless, not zero.
    pub rounds_sampled: u32,
}

/// Several armed steps, reduced.
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub passes: Vec<PassSummary>,
    pub rounds: usize,
    /// Per round: Σ passes and span. Kept per round rather than only reduced,
    /// because `min(Σ) ≠ Σ(min)` and mixing the two silently would make the
    /// percentages not add up — and because the three views of one step have to
    /// be read off the SAME step to mean anything.
    pub round_attributed: Vec<u64>,
    pub round_span: Vec<u64>,
    pub round_unsampled: Vec<u32>,
    pub dropped: u32,
}

impl ProfileSummary {
    /// Reduce `runs` (one per armed step) into one table, sorted by cost.
    pub fn reduce(runs: &[ProfileRun]) -> Self {
        use std::collections::HashMap;
        let mut order: Vec<String> = Vec::new();
        let mut acc: HashMap<String, PassSummary> = HashMap::new();
        // Sum a label's occurrences WITHIN a round, then take the min of the
        // per-round sums (else a twice-dispatched label reports one dispatch). A
        // round with any unsampled occurrence contributes NOTHING to that label —
        // its short sum is exactly what the min estimator would prefer, i.e. how a
        // dropped counter sample becomes a fast kernel.
        for run in runs {
            let mut per_round: HashMap<&str, (u64, u32, u32, bool)> = HashMap::new();
            for pass in &run.passes {
                let slot = per_round
                    .entry(pass.label.as_str())
                    .or_insert((0, pass.workgroups, 0, true));
                slot.0 += pass.nanos;
                slot.2 += 1;
                slot.3 &= pass.sampled;
            }
            for (label, (nanos, workgroups, invocations, sampled)) in per_round {
                match acc.get_mut(label) {
                    Some(entry) => {
                        if sampled {
                            if entry.rounds_sampled == 0 {
                                entry.min_nanos = nanos;
                                entry.max_nanos = nanos;
                            } else {
                                entry.min_nanos = entry.min_nanos.min(nanos);
                                entry.max_nanos = entry.max_nanos.max(nanos);
                            }
                            entry.rounds_sampled += 1;
                        }
                    }
                    None => {
                        order.push(label.to_string());
                        acc.insert(
                            label.to_string(),
                            PassSummary {
                                label: label.to_string(),
                                min_nanos: if sampled { nanos } else { 0 },
                                max_nanos: if sampled { nanos } else { 0 },
                                workgroups,
                                invocations,
                                rounds_sampled: u32::from(sampled),
                            },
                        );
                    }
                }
            }
        }
        let mut passes: Vec<PassSummary> = order
            .into_iter()
            .filter_map(|label| acc.remove(&label))
            .collect();
        passes.sort_by(|a, b| b.min_nanos.cmp(&a.min_nanos));
        Self {
            passes,
            rounds: runs.len(),
            round_attributed: runs.iter().map(|r| r.attributed_nanos()).collect(),
            round_span: runs.iter().map(|r| r.span_nanos).collect(),
            round_unsampled: runs.iter().map(|r| r.unsampled).collect(),
            dropped: runs.iter().map(|r| r.dropped).max().unwrap_or(0),
        }
    }

    /// Σ of the per-pass minima. A lower bound on any single round's total, and
    /// the denominator the percentage column uses.
    pub fn attributed_floor_nanos(&self) -> u64 {
        self.passes.iter().map(|p| p.min_nanos).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(passes: &[(&str, u64, u32)], span: u64) -> ProfileRun {
        ProfileRun {
            passes: passes
                .iter()
                .map(|(label, nanos, workgroups)| PassTiming {
                    label: label.to_string(),
                    nanos: *nanos,
                    workgroups: *workgroups,
                    sampled: true,
                })
                .collect(),
            span_nanos: span,
            dropped: 0,
            unsampled: 0,
        }
    }

    #[test]
    fn what_is_not_in_a_pass_is_the_span_minus_the_passes() {
        let r = run(&[("a", 10, 1), ("b", 20, 1)], 45);
        assert_eq!(r.attributed_nanos(), 30);
        assert_eq!(r.unattributed_nanos(), 15);
    }

    /// A span shorter than the sum of its passes is not evidence of negative
    /// time; it is an instrument artefact, and it must read as zero.
    #[test]
    fn an_impossible_span_reads_as_no_gap_rather_than_a_negative_one() {
        let r = run(&[("a", 10, 1), ("b", 20, 1)], 25);
        assert_eq!(r.unattributed_nanos(), 0);
    }

    /// The estimator is the minimum across rounds — the rule the rest of this
    /// repo's benchmarks use, for the reason stated in `PERF_CONVOLUTION.md`
    /// §5.1: contention can only add time.
    #[test]
    fn a_pass_is_summarised_by_its_fastest_round() {
        let runs = vec![
            run(&[("a", 100, 4), ("b", 50, 2)], 200),
            run(&[("a", 70, 4), ("b", 90, 2)], 210),
        ];
        let s = ProfileSummary::reduce(&runs);
        let a = s.passes.iter().find(|p| p.label == "a").unwrap();
        assert_eq!(a.min_nanos, 70);
        assert_eq!(a.max_nanos, 100);
        let b = s.passes.iter().find(|p| p.label == "b").unwrap();
        assert_eq!(b.min_nanos, 50);
        // Sorted by cost, most expensive first — that is what the table is for.
        assert_eq!(s.passes[0].label, "a");
    }

    /// A pass Metal declined to time (a pair of zeros) must not become a fast one:
    /// it must not enter the minimum at 0, and — worse — its missing time must not
    /// reappear as "between the passes", the quantity this instrument exists for.
    #[test]
    fn a_pass_the_backend_did_not_time_is_not_a_free_pass() {
        let mut good = run(&[("a", 100, 4), ("b", 50, 2)], 160);
        let mut bad = run(&[("a", 0, 4), ("b", 60, 2)], 70);
        bad.passes[0].sampled = false;
        bad.unsampled = 1;
        good.dropped = 0;
        bad.dropped = 0;

        let s = ProfileSummary::reduce(&[good, bad]);
        let a = s.passes.iter().find(|p| p.label == "a").unwrap();
        assert_eq!(
            a.min_nanos, 100,
            "the unsampled round must not become this pass's fastest round"
        );
        assert_eq!(a.rounds_sampled, 1, "one of the two rounds timed it");
        // The pass that WAS sampled in both rounds still takes its minimum from
        // both — the exclusion is per label, not per round.
        let b = s.passes.iter().find(|p| p.label == "b").unwrap();
        assert_eq!(b.min_nanos, 50);
        assert_eq!(b.rounds_sampled, 2);
    }

    /// No second door (like `nothing_in_the_engine_bypasses_the_counted_queue`): a
    /// timing table that omits a pass reports a smaller total and a larger
    /// "unattributed", read as dispatch overhead when it is a hole in the instrument.
    /// Checked mechanically rather than remembered.
    #[test]
    fn nothing_in_the_engine_opens_an_untimed_pass() {
        fn walk(dir: &std::path::Path, offenders: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("failed to read the engine sources") {
                let path = entry.expect("bad directory entry").path();
                if path.is_dir() {
                    walk(&path, offenders);
                    continue;
                }
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !name.ends_with(".rs") || name.contains("test") {
                    continue;
                }
                // `gpu_context.rs` is where the door is: the one file allowed to
                // begin a pass, because it is the one that times what it opens.
                if name == "gpu_context.rs" {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("failed to read a source file");
                for (number, line) in source.lines().enumerate() {
                    let code = line.trim_start();
                    if code.starts_with("//") {
                        continue;
                    }
                    // Assembled from halves so that this line is not itself an
                    // offender — the detector has to be allowed to name what it
                    // detects.
                    if code.contains(concat!("begin_compute", "_pass")) {
                        offenders.push(format!("{}:{}: {}", path.display(), number + 1, code));
                    }
                }
            }
        }

        let mut offenders = Vec::new();
        walk(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut offenders,
        );
        assert!(
            offenders.is_empty(),
            "these open a compute pass without going through GpuContext::compute_pass, \
             so their GPU time is invisible to every profile this project prints — and \
             invisible time is reported as dispatch overhead:\n{}",
            offenders.join("\n")
        );
    }

    /// A label that appears twice in one step costs what its two dispatches
    /// cost together, not what one of them cost.
    #[test]
    fn a_label_dispatched_twice_in_a_step_is_counted_twice() {
        let runs = vec![run(&[("a", 10, 1), ("a", 30, 1)], 50)];
        let s = ProfileSummary::reduce(&runs);
        assert_eq!(s.passes.len(), 1);
        assert_eq!(s.passes[0].min_nanos, 40);
        assert_eq!(s.passes[0].invocations, 2);
    }
}
