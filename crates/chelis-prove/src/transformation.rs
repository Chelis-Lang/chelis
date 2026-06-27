//! Goal transformation layer: the type, application machinery, and recording.
//!
//! A transformation is a pure function `Goal -> Vec<Goal>` that is sound in the
//! proof direction: if all output goals discharge, the input goal discharges.
//! See `spec/design/transformation_layer.md`.

use crate::discharge::Goal;

/// A named, pure, deterministic goal transformation.
///
/// Properties (from the spec):
/// - Pure: same input goal yields the same output goals.
/// - Sound in the proof direction: if all outputs discharge, the input is discharged.
/// - Composable: transformations chain; a recorded sequence is replayable.
pub trait Transformation: Send + Sync {
    /// The canonical name of this transformation (for provenance recording).
    fn name(&self) -> &str;

    /// Apply the transformation to `goal`, producing zero or more sub-goals.
    ///
    /// - Returning `vec![goal.clone()]` is the identity (no transformation applied).
    /// - Returning an empty vec means the goal is vacuously discharged by this
    ///   transformation alone — this is almost always unsound and the harness
    ///   will catch it.
    fn apply(&self, goal: &Goal) -> Vec<Goal>;
}

/// A record of one transformation application, for provenance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformationRecord {
    pub name: String,
    pub output_goal_count: usize,
}

/// A composable pipeline of transformations.
#[derive(Default)]
pub struct TransformationPipeline {
    steps: Vec<Box<dyn Transformation>>,
}

impl TransformationPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, t: Box<dyn Transformation>) {
        self.steps.push(t);
    }

    /// Apply all steps in sequence. Each step's output goals become the next
    /// step's inputs. Returns the final goal set and the provenance chain.
    pub fn apply_all(&self, goal: &Goal) -> (Vec<Goal>, Vec<TransformationRecord>) {
        let mut goals = vec![goal.clone()];
        let mut records = Vec::new();
        for step in &self.steps {
            let mut next_goals = Vec::new();
            for g in &goals {
                let outputs = step.apply(g);
                records.push(TransformationRecord {
                    name: step.name().to_string(),
                    output_goal_count: outputs.len(),
                });
                next_goals.extend(outputs);
            }
            goals = next_goals;
        }
        (goals, records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discharge::{
        Discharge, IntervalBox, IrHandle, OutputRange, Qualifier, QualifierSet, Soundness,
    };
    use crate::tier_b::TierBResult;
    use crate::transformation_harness::{CorpusEntry, HarnessResult, run_harness};

    /// Identity transformation: returns the goal unchanged.
    struct Identity;
    impl Transformation for Identity {
        fn name(&self) -> &str {
            "identity"
        }
        fn apply(&self, goal: &Goal) -> Vec<Goal> {
            vec![goal.clone()]
        }
    }

    /// Bogus transformation: returns empty vec (vacuously discharges anything).
    struct VacuousEmpty;
    impl Transformation for VacuousEmpty {
        fn name(&self) -> &str {
            "vacuous_empty"
        }
        fn apply(&self, _goal: &Goal) -> Vec<Goal> {
            vec![]
        }
    }

    fn make_discharged_entry() -> CorpusEntry {
        let goal = Goal::box_range(
            IntervalBox {
                dims: vec![("x".into(), 0.0, 1.0)],
            },
            OutputRange {
                output: "out".into(),
                lo: 0.0,
                hi: 1.0,
            },
        )
        .unwrap()
        .with_ir(IrHandle::unpopulated());
        let discharge = Discharge::new(
            Soundness::SoundApproximate,
            QualifierSet::from_iter_kinds([Qualifier::SoundOverApproximation]),
            TierBResult::Proved,
            serde_json::json!({}),
        )
        .unwrap();
        CorpusEntry {
            goal,
            discharge: Some(discharge),
        }
    }

    fn make_undischargeable_entry() -> CorpusEntry {
        let goal = Goal::box_range(
            IntervalBox {
                dims: vec![("x".into(), 0.0, 1.0)],
            },
            OutputRange {
                output: "out".into(),
                lo: 100.0,
                hi: 200.0,
            },
        )
        .unwrap()
        .with_ir(IrHandle::unpopulated());
        CorpusEntry {
            goal,
            discharge: None,
        }
    }

    /// Oracle: returns the cached discharge for goals that match a corpus entry.
    fn oracle_from_corpus(corpus: &[CorpusEntry]) -> impl Fn(&Goal) -> Option<Discharge> + '_ {
        move |goal| {
            corpus
                .iter()
                .find(|e| &e.goal == goal)
                .and_then(|e| e.discharge.clone())
        }
    }

    #[test]
    fn transformation_identity_passes_harness() {
        let corpus = vec![make_discharged_entry(), make_undischargeable_entry()];
        let results = run_harness(&Identity, &corpus, oracle_from_corpus(&corpus));
        assert!(
            results
                .iter()
                .all(|(_, r)| matches!(r, HarnessResult::Pass)),
            "identity must pass: {:?}",
            results
        );
    }

    #[test]
    fn transformation_vacuous_empty_fails_on_undischargeable() {
        let corpus = vec![make_undischargeable_entry()];
        let results = run_harness(&VacuousEmpty, &corpus, oracle_from_corpus(&corpus));
        assert!(
            !results.is_empty(),
            "vacuous empty must fail on undischargeable goal"
        );
        assert!(
            results
                .iter()
                .any(|(_, r)| matches!(r, HarnessResult::LaunderedUndischargeable))
        );
    }

    #[test]
    fn transformation_vacuous_empty_flags_on_discharged() {
        let corpus = vec![make_discharged_entry()];
        let results = run_harness(&VacuousEmpty, &corpus, oracle_from_corpus(&corpus));
        assert!(
            results
                .iter()
                .any(|(_, r)| matches!(r, HarnessResult::VacuousEmpty))
        );
    }

    #[test]
    fn transformation_pipeline_chains_steps() {
        let goal = Goal::box_range(
            IntervalBox {
                dims: vec![("x".into(), 0.0, 1.0)],
            },
            OutputRange {
                output: "out".into(),
                lo: 0.0,
                hi: 1.0,
            },
        )
        .unwrap();

        let mut pipeline = TransformationPipeline::new();
        pipeline.push(Box::new(Identity));
        pipeline.push(Box::new(Identity));

        let (goals, records) = pipeline.apply_all(&goal);
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0], goal);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].name, "identity");
        assert_eq!(records[1].name, "identity");
    }
}
