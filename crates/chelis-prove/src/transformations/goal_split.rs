//! Goal splitting with lattice-aware recombination.
//!
//! Splits a conjunctive postcondition into independent sub-goals (one per
//! conjunct), each sharing the SAME variables and preconditions. The
//! recombination function re-joins sub-discharges into a composite that is
//! NEVER stronger than the weakest sub-discharge.

use crate::discharge::{Discharge, DischargeError, Goal, GoalShape, QualifierSet, Soundness};
use crate::solver::{BoolOp, SmtExpr};
use crate::tier_b::{SmtProperty, TierBResult};
use crate::transformation::Transformation;

/// Splits conjunctive SMT postconditions into independent sub-goals.
pub struct GoalSplit;

impl Transformation for GoalSplit {
    fn name(&self) -> &str {
        "goal-split"
    }

    fn apply(&self, goal: &Goal) -> Vec<Goal> {
        let GoalShape::Smt(ref prop) = goal.shape else {
            return vec![goal.clone()];
        };

        let SmtExpr::Bool(BoolOp::And, ref conjuncts) = prop.postcondition else {
            return vec![goal.clone()];
        };

        if conjuncts.len() <= 1 {
            return vec![goal.clone()];
        }

        conjuncts
            .iter()
            .map(|conjunct| {
                let sub_prop = SmtProperty {
                    variables: prop.variables.clone(),
                    preconditions: prop.preconditions.clone(),
                    postcondition: conjunct.clone(),
                };
                Goal::smt(sub_prop).with_ir(goal.ir.clone())
            })
            .collect()
    }
}

/// Recombine discharges from a split goal into a composite discharge.
///
/// INVARIANT: the composite is NEVER stronger than the weakest sub-discharge.
///
/// Rules:
/// 1. If ANY sub-discharge has result Disproved → composite is that failure,
///    carrying the failing sub-discharge's actual qualifier set.
/// 2. If ANY sub-discharge has result Error/Timeout/Unknown → composite is
///    that result at Untrusted soundness.
/// 3. Soundness = min across all sub-discharges.
/// 4. QualifierSet = union across all sub-discharges.
///
/// Non-vacuity is NOT checked here — it's a property of the shared
/// precondition, checked once before the split, not per sub-goal.
pub fn recombine_split_discharges(discharges: &[Discharge]) -> Result<Discharge, DischargeError> {
    assert!(
        !discharges.is_empty(),
        "cannot recombine empty discharge set"
    );

    // Rule 1: any Disproved dominates (carry its qualifier set)
    for d in discharges {
        if matches!(d.result(), TierBResult::Disproved(_)) {
            return Discharge::new(
                d.soundness(),
                d.qualifier_set().clone(),
                d.result().clone(),
                serde_json::json!({"recombination": "disproved_dominates"}),
            );
        }
    }

    // Rule 2: any Error/Timeout/Unknown dominates
    for d in discharges {
        match d.result() {
            TierBResult::Error(_) | TierBResult::Timeout | TierBResult::Unknown => {
                return Discharge::new(
                    Soundness::Untrusted,
                    QualifierSet::new(),
                    d.result().clone(),
                    serde_json::json!({"recombination": "error_dominates"}),
                );
            }
            _ => {}
        }
    }

    // All proved: min soundness, union qualifiers
    let min_soundness = discharges.iter().map(|d| d.soundness()).min().unwrap(); // non-empty guaranteed by assert

    let union_qualifiers = discharges
        .iter()
        .fold(QualifierSet::new(), |acc, d| acc.union(d.qualifier_set()));

    Discharge::new(
        min_soundness,
        union_qualifiers,
        TierBResult::Proved,
        serde_json::json!({"recombination": "all_proved", "sub_count": discharges.len()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discharge::{IrHandle, Qualifier};
    use crate::solver::{CmpOp, SmtSort};

    // --- Helper constructors ---

    fn make_discharge(
        soundness: Soundness,
        qualifiers: &[Qualifier],
        result: TierBResult,
    ) -> Discharge {
        Discharge::new(
            soundness,
            QualifierSet::from_iter_kinds(qualifiers.iter().copied()),
            result,
            serde_json::json!({"test": true}),
        )
        .expect("test discharge must be constructible")
    }

    fn make_conjunctive_goal(conjunct_count: usize) -> Goal {
        let conjuncts: Vec<SmtExpr> = (0..conjunct_count)
            .map(|i| {
                SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Var(format!("x{i}"))),
                    Box::new(SmtExpr::RealLit(0.0)),
                )
            })
            .collect();
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(-1.0)),
            )],
            postcondition: SmtExpr::Bool(BoolOp::And, conjuncts),
        };
        Goal::smt(prop).with_ir(IrHandle::unpopulated())
    }

    // ===================================================================
    // Row 1: All sub-discharges proved at the same soundness.
    //         Composite = Proved, min soundness, union qualifiers.
    // ===================================================================
    #[test]
    fn row1_all_proved_same_soundness_returns_proved_with_union_qualifiers() {
        let d1 = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );
        let d2 = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::SoundOverApproximation],
            TierBResult::Proved,
        );

        let composite = recombine_split_discharges(&[d1, d2]).unwrap();
        assert_eq!(*composite.result(), TierBResult::Proved);
        assert_eq!(composite.soundness(), Soundness::SoundApproximate);
        assert!(composite.qualifier_set().contains(Qualifier::RealArith));
        assert!(
            composite
                .qualifier_set()
                .contains(Qualifier::SoundOverApproximation)
        );
    }

    // ===================================================================
    // Row 2: Sub-discharges at DIFFERENT soundness levels.
    //         Composite soundness = min (the weakest link).
    //         Qualifiers must be compatible with the min soundness
    //         (Discharge::new enforces per-qualifier min-soundness floors).
    // ===================================================================
    #[test]
    fn row2_mixed_soundness_returns_min_soundness() {
        // Exact discharge with RealArith qualifier (floor = SoundApproximate, ok at Exact)
        let d1 = make_discharge(
            Soundness::Exact,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );
        // SoundApproximate discharge with RealArith qualifier
        let d2 = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );

        let composite = recombine_split_discharges(&[d1, d2]).unwrap();
        assert_eq!(*composite.result(), TierBResult::Proved);
        assert_eq!(
            composite.soundness(),
            Soundness::SoundApproximate,
            "min(Exact, SoundApproximate) = SoundApproximate"
        );
        // The union of both qualifier sets is just RealArith (both had it)
        assert!(composite.qualifier_set().contains(Qualifier::RealArith));
    }

    // ===================================================================
    // Row 3: One sub-discharge is Disproved. Composite = Disproved,
    //         carrying the failing sub-discharge's qualifier set.
    // ===================================================================
    #[test]
    fn row3_one_disproved_dominates_with_its_qualifier() {
        let proved = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );
        let disproved = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Disproved(serde_json::json!({"x": "0.5"})),
        );

        let composite = recombine_split_discharges(&[proved, disproved]).unwrap();
        assert!(matches!(composite.result(), TierBResult::Disproved(_)));
        // Carries the FAILING sub-discharge's qualifier, not the union
        assert!(composite.qualifier_set().contains(Qualifier::RealArith));
        assert_eq!(composite.soundness(), Soundness::SoundApproximate);
    }

    // ===================================================================
    // Row 4: One sub-discharge is Error/Timeout. Composite = that error at
    //         Untrusted soundness (partial green is not green).
    // ===================================================================
    #[test]
    fn row4_one_error_dominates_as_untrusted() {
        let proved = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );
        let errored = make_discharge(
            Soundness::Untrusted,
            &[],
            TierBResult::Error("solver crash".to_string()),
        );

        let composite = recombine_split_discharges(&[proved, errored]).unwrap();
        assert!(matches!(composite.result(), TierBResult::Error(_)));
        assert_eq!(composite.soundness(), Soundness::Untrusted);
        assert!(composite.qualifier_set().is_empty());
    }

    // ===================================================================
    // Row 5: Vacuity is NOT checked by recombination.
    //         If the shared precondition is unsatisfiable, sub-discharges
    //         are individually "proved" (vacuously). Recombination returns
    //         Proved — the vacuity check is the CALLER's responsibility.
    // ===================================================================
    #[test]
    fn row5_vacuity_is_callers_responsibility_not_recombinations() {
        // Both sub-discharges report Proved (vacuously, because the shared
        // precondition x > 0 AND x < 0 is unsatisfiable). Recombination
        // does NOT detect this — it just sees Proved and returns Proved.
        let d1 = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );
        let d2 = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );

        let composite = recombine_split_discharges(&[d1, d2]).unwrap();
        // Recombination does NOT magically detect vacuity.
        assert_eq!(*composite.result(), TierBResult::Proved);
        assert_eq!(composite.soundness(), Soundness::SoundApproximate);
        // The caller must run check_assumptions_satisfiable separately.
    }

    // ===================================================================
    // Row 6: A trivially-true conjunct (e.g. x == x) that discharges as
    //         Proved does NOT poison. Recombination returns Proved.
    //         Confirms no over-poisoning of trivial sub-goals.
    // ===================================================================
    #[test]
    fn row6_trivially_true_conjunct_does_not_poison() {
        // One sub-discharge is from a trivial conjunct (x == x), the other
        // from a real property. Both Proved. The trivial one must not
        // trigger any special-case rejection.
        let trivial = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );
        let real = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );

        let composite = recombine_split_discharges(&[trivial, real]).unwrap();
        assert_eq!(*composite.result(), TierBResult::Proved);
        // No special case, no poisoning — just a clean union of proved sub-goals.
    }

    // ===================================================================
    // Row 7: GoalSplit on a conjunctive postcondition splits correctly.
    //         Each sub-goal gets its own conjunct as postcondition, with
    //         the SAME variables and preconditions (shared, NOT partitioned).
    // ===================================================================
    #[test]
    fn row7_split_conjunctive_postcondition_shares_variables_and_preconditions() {
        let goal = make_conjunctive_goal(3);
        let split = GoalSplit;
        let sub_goals = split.apply(&goal);

        assert_eq!(sub_goals.len(), 3);

        // Each sub-goal has the same variables and preconditions
        let original_prop = goal.as_smt().unwrap();
        for sg in &sub_goals {
            let sp = sg.as_smt().unwrap();
            assert_eq!(sp.variables, original_prop.variables);
            assert_eq!(sp.preconditions, original_prop.preconditions);
            // Postcondition is a single conjunct, not the full conjunction
            assert!(!matches!(sp.postcondition, SmtExpr::Bool(BoolOp::And, _)));
        }

        // IR handle is preserved
        for sg in &sub_goals {
            assert_eq!(sg.ir, goal.ir);
        }
    }

    // ===================================================================
    // Row 8: GoalSplit on a non-conjunctive postcondition is identity.
    // ===================================================================
    #[test]
    fn row8_non_conjunctive_postcondition_is_identity() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let goal = Goal::smt(prop);
        let split = GoalSplit;
        let sub_goals = split.apply(&goal);

        assert_eq!(sub_goals.len(), 1);
        assert_eq!(sub_goals[0], goal);
    }

    // ===================================================================
    // Row 9: GoalSplit on a non-SMT goal (BoxRange) is identity.
    // ===================================================================
    #[test]
    fn row9_non_smt_goal_is_identity() {
        use crate::discharge::{IntervalBox, OutputRange};
        let goal = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 100.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 50.0,
            },
        )
        .unwrap();
        let split = GoalSplit;
        let sub_goals = split.apply(&goal);

        assert_eq!(sub_goals.len(), 1);
        assert_eq!(sub_goals[0], goal);
    }

    // ===================================================================
    // Row 10: Timeout sub-discharge dominates even when most sub-goals proved.
    //          Partial green is not green.
    // ===================================================================
    #[test]
    fn row10_timeout_dominates_partial_green_is_not_green() {
        let d1 = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );
        let d2 = make_discharge(
            Soundness::SoundApproximate,
            &[Qualifier::RealArith],
            TierBResult::Proved,
        );
        let timeout = make_discharge(Soundness::Untrusted, &[], TierBResult::Timeout);

        let composite = recombine_split_discharges(&[d1, d2, timeout]).unwrap();
        assert_eq!(*composite.result(), TierBResult::Timeout);
        assert_eq!(composite.soundness(), Soundness::Untrusted);
        assert!(composite.qualifier_set().is_empty());
    }

    // ===================================================================
    // Row 11: GoalSplit on a single-conjunct And (degenerate case) is identity.
    //          A single-element conjunction should NOT be split into one sub-goal.
    // ===================================================================
    #[test]
    fn row11_single_conjunct_and_is_identity() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Bool(
                BoolOp::And,
                vec![SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(0.0)),
                )],
            ),
        };
        let goal = Goal::smt(prop);
        let split = GoalSplit;
        let sub_goals = split.apply(&goal);

        assert_eq!(sub_goals.len(), 1);
        assert_eq!(sub_goals[0], goal);
    }
}
