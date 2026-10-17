//! One compile serves many fuzz evaluations (chelis#3117).
//!
//! A sample's guard and predicate are evaluated as roots of a program that
//! binds the sample's values as literals, because the evaluator's entry
//! points take runtime arguments only for tensor inputs. Compiling that
//! program once per evaluation made a fuzz run pay a compile for every guard,
//! predicate and shrink trial. Here the probes of a whole run of samples are
//! declared in one program, compiled once, and each probe is evaluated as a
//! selected root of that program.
//!
//! Every probe is its own top-level declaration, independent of the others,
//! so a probe evaluates in the shared program exactly as it does alone. The
//! one thing the shared program can do that a single probe's program cannot
//! is fail as a whole: a rejection or evaluation error that one probe's
//! declaration causes reaches every root of the shared compile. So the shared
//! program's answer is used only when it is a value. A compile that fails, and
//! any evaluation that errors, is answered by the sample's own program, the
//! single-probe evaluation the fuzz tier has always run, so every verdict,
//! counterexample, shrink step and error message is that evaluation's.
//!
//! Probes are compiled ahead of their evaluation, never evaluated ahead: a
//! guard or predicate is evaluated only when the one-at-a-time run would have
//! evaluated it. Compiling ahead costs only the probes of samples a run stops
//! before reaching, so chunks start at one sample and double: a run compiles
//! the probes of at most twice the samples (or shrink trials) it reaches, in
//! a number of compiles logarithmic in them until chunks reach
//! [`MAX_SAMPLES_PER_COMPILE`].

use std::cell::OnceCell;
use std::collections::BTreeMap;

use super::{Param, Sample, bool_property_root};

/// The most samples whose probes share one compile. A run that rejects most
/// attempts compiles once per this many of them; the bound keeps one
/// program's probe count, and the per-root cost of evaluating in a program
/// that declares them all, small.
const MAX_SAMPLES_PER_COMPILE: usize = 64;

/// Chunk sizes for a run: 1, 2, 4, ... up to [`MAX_SAMPLES_PER_COMPILE`],
/// then that bound for every later chunk.
struct ChunkSizes(usize);

impl ChunkSizes {
    fn new() -> Self {
        Self(1)
    }

    fn next(&mut self) -> usize {
        let size = self.0;
        self.0 = (size * 2).min(MAX_SAMPLES_PER_COMPILE);
        size
    }
}

/// How one fuzz lane declares and evaluates sample probes.
pub(super) trait SampleLane {
    /// One sample's binder values.
    type Sample;

    /// Whether the property has a `where` guard to evaluate before its body.
    fn guarded(&self) -> bool;

    /// Evaluate one sample's guard (`precondition`) or predicate in the
    /// sample's own program.
    fn eval_alone(&self, sample: &Self::Sample, precondition: bool) -> Result<bool, String>;

    /// Declare the guard probe (when guarded) and the predicate probe of every
    /// sample in one program and compile it. `None` when that program does not
    /// compile.
    fn compile(&self, samples: &[Self::Sample]) -> Option<CompiledProbes>;
}

/// A compiled program holding the probes of a run of samples.
pub(super) struct CompiledProbes {
    prepared: chelis_compiler_api::compiler::PreparedEval,
    roots: Vec<ProbeRoots>,
}

impl CompiledProbes {
    /// `roots[i]` names sample `i`'s probes in `prepared`.
    pub(super) fn new(
        prepared: chelis_compiler_api::compiler::PreparedEval,
        roots: Vec<ProbeRoots>,
    ) -> Self {
        Self { prepared, roots }
    }
}

/// One sample's probe roots in a [`CompiledProbes`] program.
pub(super) struct ProbeRoots {
    /// The guard probe; `None` for an unguarded property.
    pub(super) pre: Option<String>,
    pub(super) probe: String,
}

/// Samples whose probes are compiled together on the first evaluation.
pub(super) struct SampleBatch<'l, L: SampleLane> {
    lane: &'l L,
    samples: Vec<L::Sample>,
    compiled: OnceCell<Option<CompiledProbes>>,
}

impl<'l, L: SampleLane> SampleBatch<'l, L> {
    pub(super) fn new(lane: &'l L, samples: Vec<L::Sample>) -> Self {
        Self {
            lane,
            samples,
            compiled: OnceCell::new(),
        }
    }

    pub(super) fn sample(&self, index: usize) -> &L::Sample {
        &self.samples[index]
    }

    fn into_sample(mut self, index: usize) -> L::Sample {
        self.samples.swap_remove(index)
    }

    /// Sample `index`'s guard or predicate, with the value its own program
    /// gives (see the module documentation).
    pub(super) fn eval(&self, index: usize, precondition: bool) -> Result<bool, String> {
        let compiled = self
            .compiled
            .get_or_init(|| self.lane.compile(&self.samples));
        if let Some(compiled) = compiled {
            let roots = &compiled.roots[index];
            let root = if precondition {
                roots.pre.as_deref()
            } else {
                Some(roots.probe.as_str())
            };
            if let Some(root) = root
                && let Ok(value) =
                    bool_property_root(compiled.prepared.eval_root(BTreeMap::new(), root))
            {
                return Ok(value);
            }
        }
        self.lane.eval_alone(&self.samples[index], precondition)
    }

    /// Whether sample `index` is admitted by every guard and falsifies the
    /// predicate: the condition a shrink trial must keep.
    fn still_fails(&self, index: usize) -> bool {
        if self.lane.guarded() && !matches!(self.eval(index, true), Ok(true)) {
            return false;
        }
        matches!(self.eval(index, false), Ok(false))
    }
}

/// What the sampler produced for one attempt.
pub(super) enum Draw<S, E> {
    Sample(S),
    /// No sample; a later attempt may still produce one.
    Skipped(E),
    /// No sample, and the run ends at this attempt, so nothing after it is
    /// drawn.
    Ends(E),
}

/// What one sampling attempt produced.
pub(super) enum Drawn<'b, 'l, L: SampleLane, E> {
    /// A sample, as `index` in `batch`.
    Sample {
        batch: &'b SampleBatch<'l, L>,
        index: usize,
    },
    /// The attempt produced no sample, for the sampler's reason.
    Failed(E),
}

/// Sampling attempts drawn ahead of their evaluation, a chunk of
/// [`ChunkSizes`] at a time, so one compile serves every sample of a chunk.
///
/// Drawing ahead does not change what is drawn: the sampler consumes the
/// random stream once per attempt, in attempt order, and nothing else reads
/// that stream. A chunk the run stops partway through leaves its later draws
/// unused, exactly as the run would never have made them, and a draw that
/// ends the run stops its chunk, so no attempt after it is drawn.
pub(super) struct AttemptDraws<'l, L: SampleLane, E> {
    lane: &'l L,
    max_attempts: usize,
    chunks: ChunkSizes,
    drawn: usize,
    pending: std::collections::VecDeque<Result<usize, E>>,
    batch: SampleBatch<'l, L>,
}

impl<'l, L: SampleLane, E> AttemptDraws<'l, L, E> {
    pub(super) fn new(lane: &'l L, max_attempts: usize) -> Self {
        Self {
            lane,
            max_attempts,
            chunks: ChunkSizes::new(),
            drawn: 0,
            pending: std::collections::VecDeque::new(),
            batch: SampleBatch::new(lane, Vec::new()),
        }
    }

    /// The next attempt's result. `draw(attempt)` samples attempt number
    /// `attempt`, counting from zero; it is called once per attempt, in order.
    pub(super) fn next(
        &mut self,
        draw: &mut impl FnMut(usize) -> Draw<L::Sample, E>,
    ) -> Drawn<'_, 'l, L, E> {
        if self.pending.is_empty() {
            let len = self
                .chunks
                .next()
                .min(self.max_attempts.saturating_sub(self.drawn))
                .max(1);
            let mut samples = Vec::new();
            for attempt in self.drawn..self.drawn + len {
                match draw(attempt) {
                    Draw::Sample(sample) => {
                        self.pending.push_back(Ok(samples.len()));
                        samples.push(sample);
                    }
                    Draw::Skipped(error) => self.pending.push_back(Err(error)),
                    Draw::Ends(error) => {
                        self.pending.push_back(Err(error));
                        break;
                    }
                }
            }
            self.drawn += len;
            self.batch = SampleBatch::new(self.lane, samples);
        }
        match self.pending.pop_front().expect("a chunk holds one attempt") {
            Ok(index) => Drawn::Sample {
                batch: &self.batch,
                index,
            },
            Err(error) => Drawn::Failed(error),
        }
    }
}

/// Shrink a counterexample one binder at a time. A step's trials (every
/// binder's candidates, in binder order) are tried in order, a chunk of
/// [`ChunkSizes`] per compile, and the first trial that still fails becomes
/// the counterexample, as the trials would have been tried one by one.
pub(super) fn shrink_with_lane<L: SampleLane<Sample = Sample>>(
    sample: Sample,
    params: &[Param],
    lane: &L,
) -> (Sample, usize) {
    super::shrink_counterexample(sample, params, |trials| {
        let mut trials = trials.into_iter().peekable();
        let mut chunks = ChunkSizes::new();
        while trials.peek().is_some() {
            let batch = SampleBatch::new(lane, trials.by_ref().take(chunks.next()).collect());
            if let Some(found) = (0..batch.samples.len()).find(|&index| batch.still_fails(index)) {
                return Some(batch.into_sample(found));
            }
        }
        None
    })
}
