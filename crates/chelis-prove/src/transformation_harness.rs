//! Soundness harness for goal transformations.
//!
//! The harness validates that a transformation preserves discharge on a corpus:
//! - A goal that IS discharged stays discharged after transformation (all output
//!   goals are dischargeable).
//! - A goal that is NOT dischargeable never BECOMES dischargeable after
//!   transformation (the critical direction — catches unsound rewrites).
//!
//! The harness is built FIRST so the agentic porting of transformations is gated
//! by a check it cannot talk past. See `spec/design/transformation_layer.md` §5.

use crate::discharge::{Discharge, Goal};
use crate::transformation::Transformation;

/// A corpus entry: a goal with its known discharge outcome.
#[derive(Debug, Clone)]
pub struct CorpusEntry {
    pub goal: Goal,
    /// `Some(discharge)` if the goal IS discharged by the current engine set;
    /// `None` if the goal is undischargeable (Unsupported/Error).
    pub discharge: Option<Discharge>,
}

/// The result of running the harness on one corpus entry.
#[derive(Debug)]
pub enum HarnessResult {
    /// The transformation preserved soundness on this entry.
    Pass,
    /// VIOLATION: a discharged goal's transformation outputs include a goal
    /// that is NOT dischargeable (the transformation broke a working proof).
    BrokeDischarged { output_index: usize },
    /// VIOLATION: an undischargeable goal became dischargeable after
    /// transformation (the transformation laundered an unprovable into a
    /// provable — this is the critical unsoundness direction).
    LaunderedUndischargeable,
    /// The transformation produced an empty output set for a non-trivially-true
    /// goal. This vacuously discharges anything and is almost always unsound.
    VacuousEmpty,
}

/// Run the soundness harness: validate `transformation` against every entry in
/// `corpus`, using `discharge_fn` to attempt discharge of each output goal.
///
/// `discharge_fn` is the oracle: given a goal, it returns `Some(discharge)` if
/// the goal can be discharged by the current engine set, or `None` if it cannot.
pub fn run_harness(
    transformation: &dyn Transformation,
    corpus: &[CorpusEntry],
    discharge_fn: impl Fn(&Goal) -> Option<Discharge>,
) -> Vec<(usize, HarnessResult)> {
    let mut results = Vec::new();
    for (i, entry) in corpus.iter().enumerate() {
        let outputs = transformation.apply(&entry.goal);

        if outputs.is_empty() {
            if entry.discharge.is_some() {
                results.push((i, HarnessResult::VacuousEmpty));
            } else {
                results.push((i, HarnessResult::LaunderedUndischargeable));
            }
            continue;
        }

        if entry.discharge.is_some() {
            for (j, output_goal) in outputs.iter().enumerate() {
                if discharge_fn(output_goal).is_none() {
                    results.push((i, HarnessResult::BrokeDischarged { output_index: j }));
                }
            }
        } else {
            let all_discharged = outputs.iter().all(|g| discharge_fn(g).is_some());
            if all_discharged {
                results.push((i, HarnessResult::LaunderedUndischargeable));
            }
        }
    }
    results
}

/// Convenience: returns true iff all entries pass (no violations).
pub fn harness_is_sound(
    transformation: &dyn Transformation,
    corpus: &[CorpusEntry],
    discharge_fn: impl Fn(&Goal) -> Option<Discharge>,
) -> bool {
    run_harness(transformation, corpus, discharge_fn)
        .iter()
        .all(|(_, r)| matches!(r, HarnessResult::Pass))
}
