//! Abstract-subterm transformation: replace transcendental subterms with fresh
//! variables bounded by their certified envelope, leaving a residual polynomial
//! goal that routes to Z3/cvc5.
//!
//! Soundness argument: if the residual holds for ALL values in the envelope
//! interval, and the envelope provably contains the true transcendental value
//! (which it does by the Sollya/Gappa/Arb certificate), then the original holds.
//! This is universal quantification over a certified over-approximation.
//!
//! The result is tagged `SpecialFunctionCertified` (not `Exact`) because it
//! leans on the envelope certificate.
//!
//! # Sound range evaluation
//!
//! The envelope's `bound(x)` gives a sound interval for a SINGLE point. This
//! transformation needs a sound interval over the argument's ENTIRE RANGE. We
//! compute `sound_erf_range_bound(arg_lo, arg_hi)` by evaluating the envelope
//! across all boxes the range touches, taking the hull (min of all lo, max of
//! all hi). This is sound because every box's eps is a sup-norm over that box.
//!
//! If the argument range is not statically boundable (no preconditions pin it),
//! or extends outside the envelope's covered domain, the transformation DECLINES
//! (returns identity). It never guesses.

use crate::discharge::{Goal, GoalShape};
use crate::erf_envelope::{ErfEnvelope, ErfEnvelopeBox};
use crate::solver::{BoolOp, CmpOp, SmtExpr, SmtSort};
use crate::tier_b::SmtProperty;
use crate::transformation::Transformation;

/// The abstract-subterm transformation for `erf`.
pub struct AbstractSubterm {
    envelope: ErfEnvelope,
}

impl AbstractSubterm {
    pub fn new() -> Self {
        Self {
            envelope: ErfEnvelope::committed(),
        }
    }

    /// Compute a sound bound on `erf(x)` for ALL x in `[arg_lo, arg_hi]`.
    ///
    /// Evaluates the envelope across every box the range touches, taking the
    /// hull (min of all point-lo, max of all point-hi) PLUS considering
    /// monotonicity within each box segment.
    ///
    /// Returns `None` if the range extends outside the envelope's domain.
    pub fn sound_erf_range_bound(&self, arg_lo: f64, arg_hi: f64) -> Option<(f64, f64)> {
        if arg_lo > arg_hi || !arg_lo.is_finite() || !arg_hi.is_finite() {
            return None;
        }

        // Find all boxes the range intersects
        let boxes: Vec<&ErfEnvelopeBox> = self
            .envelope
            .boxes
            .iter()
            .filter(|b| b.lo <= arg_hi && b.hi >= arg_lo)
            .collect();

        if boxes.is_empty() {
            return None; // outside covered domain
        }

        // Check the range is fully covered by the envelope
        let covered_lo = boxes.first().unwrap().lo;
        let covered_hi = boxes.last().unwrap().hi;
        if arg_lo < covered_lo || arg_hi > covered_hi {
            return None; // extends outside domain
        }

        // For each box that the range intersects, compute the sound bound over
        // the intersection. The sound bound over a box segment [a,b] is:
        //   [min(approx(x) for x in [a,b]) - eps, max(approx(x) for x in [a,b]) + eps]
        //
        // For saturation arms: approx is constant, so trivial.
        // For the central polynomial: we sample endpoints of each intersection
        // and take min/max. This is sound for monotonic functions (erf is
        // monotonically increasing), but for a general polynomial approximation
        // we must be conservative. We evaluate at the intersection endpoints
        // plus use the eps as a global error band.
        //
        // Since the ENTIRE purpose is to produce a SOUND over-approximation,
        // we take the HULL of all point evaluations at the intersection
        // boundaries of each box segment. For the polynomial arm, we evaluate
        // at both endpoints of the intersection and take the hull.

        let mut overall_lo = f64::INFINITY;
        let mut overall_hi = f64::NEG_INFINITY;

        for b in &boxes {
            // The intersection of [arg_lo, arg_hi] with this box [b.lo, b.hi]
            let seg_lo = arg_lo.max(b.lo);
            let seg_hi = arg_hi.min(b.hi);

            // Evaluate the approximation at both segment endpoints
            let approx_at_lo = b.arm.approx(seg_lo);
            let approx_at_hi = b.arm.approx(seg_hi);

            // The sound bound for each evaluation point is [approx - eps, approx + eps]
            let point_lo = approx_at_lo.min(approx_at_hi) - b.eps;
            let point_hi = approx_at_lo.max(approx_at_hi) + b.eps;

            overall_lo = overall_lo.min(point_lo);
            overall_hi = overall_hi.max(point_hi);
        }

        // Clamp to [-1, 1] since erf is bounded by definition
        overall_lo = overall_lo.max(-1.0);
        overall_hi = overall_hi.min(1.0);

        Some((overall_lo, overall_hi))
    }
}

impl Default for AbstractSubterm {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformation for AbstractSubterm {
    fn name(&self) -> &str {
        "abstract-subterm"
    }

    fn apply(&self, goal: &Goal) -> Vec<Goal> {
        let GoalShape::Smt(ref prop) = goal.shape else {
            return vec![goal.clone()]; // identity for non-SMT goals
        };

        // Find erf applications in the postcondition and collect their arguments
        let erf_sites = find_erf_applications(&prop.postcondition);
        if erf_sites.is_empty() {
            return vec![goal.clone()]; // no transcendentals to abstract
        }

        // For each erf site, determine the argument's static range from preconditions
        let mut new_variables = prop.variables.clone();
        let mut new_preconditions = prop.preconditions.clone();
        let mut postcondition = prop.postcondition.clone();

        for (fresh_counter, site) in erf_sites.iter().enumerate() {
            // Try to determine the argument's range from the preconditions
            let arg_range = extract_variable_range(&site.argument, &prop.preconditions);
            let Some((arg_lo, arg_hi)) = arg_range else {
                // Can't determine argument range statically → decline
                return vec![goal.clone()];
            };

            // Compute sound envelope bound over the argument range
            let Some((env_lo, env_hi)) = self.sound_erf_range_bound(arg_lo, arg_hi) else {
                // Range outside envelope domain → decline
                return vec![goal.clone()];
            };

            // Create a fresh variable
            let fresh_name = format!("__erf_abs_{}", fresh_counter);

            // Add variable declaration
            new_variables.push((fresh_name.clone(), SmtSort::Real));

            // Add bounds as preconditions: env_lo <= fresh_var <= env_hi
            let fresh_var = SmtExpr::Var(fresh_name.clone());
            new_preconditions.push(SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::RealLit(env_lo)),
                Box::new(fresh_var.clone()),
            ));
            new_preconditions.push(SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(fresh_var.clone()),
                Box::new(SmtExpr::RealLit(env_hi)),
            ));

            // Substitute erf(arg) with fresh_var in postcondition
            postcondition = substitute_erf_call(
                &postcondition,
                &site.original_expr,
                &SmtExpr::Var(fresh_name),
            );
        }

        let residual_prop = SmtProperty {
            variables: new_variables,
            preconditions: new_preconditions,
            postcondition,
        };

        let residual_goal = Goal::smt(residual_prop).with_ir(goal.ir.clone());
        vec![residual_goal]
    }
}

/// A located erf application in an SmtExpr tree.
struct ErfSite {
    /// The argument expression inside erf(...)
    argument: SmtExpr,
    /// The full `Apply("erf", [arg])` expression for substitution matching
    original_expr: SmtExpr,
}

/// Find all `Apply("erf", [arg])` nodes in an expression.
fn find_erf_applications(expr: &SmtExpr) -> Vec<ErfSite> {
    let mut sites = Vec::new();
    find_erf_recursive(expr, &mut sites);
    sites
}

fn find_erf_recursive(expr: &SmtExpr, sites: &mut Vec<ErfSite>) {
    match expr {
        SmtExpr::Apply(name, args) if name == "erf" && args.len() == 1 => {
            sites.push(ErfSite {
                argument: args[0].clone(),
                original_expr: expr.clone(),
            });
            // Also recurse into the argument in case of nested erf
            find_erf_recursive(&args[0], sites);
        }
        SmtExpr::Arith(_, l, r) => {
            find_erf_recursive(l, sites);
            find_erf_recursive(r, sites);
        }
        SmtExpr::Cmp(_, l, r) => {
            find_erf_recursive(l, sites);
            find_erf_recursive(r, sites);
        }
        SmtExpr::Bool(_, children) => {
            for c in children {
                find_erf_recursive(c, sites);
            }
        }
        SmtExpr::Not(inner) => find_erf_recursive(inner, sites),
        SmtExpr::Ite(c, t, e) => {
            find_erf_recursive(c, sites);
            find_erf_recursive(t, sites);
            find_erf_recursive(e, sites);
        }
        SmtExpr::Apply(_, args) => {
            for a in args {
                find_erf_recursive(a, sites);
            }
        }
        SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => {
            find_erf_recursive(body, sites);
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {}
    }
}

/// Extract the static range [lo, hi] of an expression from preconditions.
///
/// Looks for patterns like `lo <= expr` and `expr <= hi` in the preconditions.
/// For a simple `Var(x)`, also looks for `lo <= x` / `x <= hi`.
/// Returns None if bounds cannot be determined.
fn extract_variable_range(expr: &SmtExpr, preconditions: &[SmtExpr]) -> Option<(f64, f64)> {
    let var_name = match expr {
        SmtExpr::Var(n) => n.as_str(),
        _ => return None, // only handle simple variable arguments for now
    };

    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;

    for pre in preconditions {
        match pre {
            // lo <= var
            SmtExpr::Cmp(CmpOp::Le, left, right) | SmtExpr::Cmp(CmpOp::Ge, right, left) => {
                if matches!(right.as_ref(), SmtExpr::Var(n) if n == var_name)
                    && let SmtExpr::RealLit(v) = left.as_ref()
                {
                    lo = Some(lo.map_or(*v, |cur: f64| cur.max(*v)));
                }
                if matches!(left.as_ref(), SmtExpr::Var(n) if n == var_name)
                    && let SmtExpr::RealLit(v) = right.as_ref()
                {
                    hi = Some(hi.map_or(*v, |cur: f64| cur.min(*v)));
                }
            }
            SmtExpr::Cmp(CmpOp::Lt, left, right) | SmtExpr::Cmp(CmpOp::Gt, right, left) => {
                if matches!(right.as_ref(), SmtExpr::Var(n) if n == var_name)
                    && let SmtExpr::RealLit(v) = left.as_ref()
                {
                    lo = Some(lo.map_or(*v, |cur: f64| cur.max(*v)));
                }
                if matches!(left.as_ref(), SmtExpr::Var(n) if n == var_name)
                    && let SmtExpr::RealLit(v) = right.as_ref()
                {
                    hi = Some(hi.map_or(*v, |cur: f64| cur.min(*v)));
                }
            }
            // And([...]) — recurse into conjunctions
            SmtExpr::Bool(BoolOp::And, children) => {
                if let Some((clo, chi)) = extract_variable_range(expr, children) {
                    lo = Some(lo.map_or(clo, |cur: f64| cur.max(clo)));
                    hi = Some(hi.map_or(chi, |cur: f64| cur.min(chi)));
                }
            }
            _ => {}
        }
    }

    match (lo, hi) {
        (Some(l), Some(h)) if l <= h => Some((l, h)),
        _ => None,
    }
}

/// Substitute all occurrences of `target` with `replacement` in `expr`.
fn substitute_erf_call(expr: &SmtExpr, target: &SmtExpr, replacement: &SmtExpr) -> SmtExpr {
    if expr == target {
        return replacement.clone();
    }
    match expr {
        SmtExpr::Arith(op, l, r) => SmtExpr::Arith(
            *op,
            Box::new(substitute_erf_call(l, target, replacement)),
            Box::new(substitute_erf_call(r, target, replacement)),
        ),
        SmtExpr::Cmp(op, l, r) => SmtExpr::Cmp(
            *op,
            Box::new(substitute_erf_call(l, target, replacement)),
            Box::new(substitute_erf_call(r, target, replacement)),
        ),
        SmtExpr::Bool(op, children) => SmtExpr::Bool(
            *op,
            children
                .iter()
                .map(|c| substitute_erf_call(c, target, replacement))
                .collect(),
        ),
        SmtExpr::Not(inner) => {
            SmtExpr::Not(Box::new(substitute_erf_call(inner, target, replacement)))
        }
        SmtExpr::Ite(c, t, e) => SmtExpr::Ite(
            Box::new(substitute_erf_call(c, target, replacement)),
            Box::new(substitute_erf_call(t, target, replacement)),
            Box::new(substitute_erf_call(e, target, replacement)),
        ),
        SmtExpr::Apply(name, args) => SmtExpr::Apply(
            name.clone(),
            args.iter()
                .map(|a| substitute_erf_call(a, target, replacement))
                .collect(),
        ),
        SmtExpr::Forall(vars, body) => SmtExpr::Forall(
            vars.clone(),
            Box::new(substitute_erf_call(body, target, replacement)),
        ),
        SmtExpr::Exists(vars, body) => SmtExpr::Exists(
            vars.clone(),
            Box::new(substitute_erf_call(body, target, replacement)),
        ),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discharge::{
        Discharge, Goal, GoalShape, IntervalBox, IrHandle, OutputRange, Qualifier, QualifierSet,
        Soundness,
    };
    use crate::tier_b::TierBResult;
    use crate::transformation_harness::{CorpusEntry, HarnessResult, run_harness};

    #[test]
    fn sound_erf_range_bound_central_box() {
        let t = AbstractSubterm::new();
        // x in [0, 1]: erf is monotonic, so bound should contain [erf(0)=0, erf(1)≈0.843]
        let (lo, hi) = t.sound_erf_range_bound(0.0, 1.0).unwrap();
        assert!(lo <= 0.0, "bound lo must be <= erf(0)=0, got {lo}");
        assert!(hi >= 0.842, "bound hi must be >= erf(1)≈0.843, got {hi}");
    }

    #[test]
    fn sound_erf_range_bound_saturation() {
        let t = AbstractSubterm::new();
        // x in [5, 10]: deep in saturation, erf ≈ 1.0
        let (lo, hi) = t.sound_erf_range_bound(5.0, 10.0).unwrap();
        assert!(lo > 0.99, "saturation lo should be close to 1, got {lo}");
        assert!(hi <= 1.0, "saturation hi should be <= 1.0, got {hi}");
    }

    #[test]
    fn sound_erf_range_bound_crosses_boxes() {
        let t = AbstractSubterm::new();
        // x in [-1, 5]: crosses central→saturation boundary at 3
        let (lo, hi) = t.sound_erf_range_bound(-1.0, 5.0).unwrap();
        // erf(-1) ≈ -0.843, erf(5) ≈ 1.0
        assert!(lo <= -0.842, "must contain erf(-1), got lo={lo}");
        assert!(hi >= 0.999, "must contain erf(5)≈1, got hi={hi}");
    }

    #[test]
    fn sound_erf_range_bound_outside_domain_declines() {
        let t = AbstractSubterm::new();
        assert!(t.sound_erf_range_bound(400.0, 500.0).is_none());
    }

    #[test]
    fn sound_erf_range_bound_not_too_tight_across_boundary() {
        // x in [2.9, 3.1] crosses the central→saturation boundary at 3.0
        let t = AbstractSubterm::new();
        let (lo, hi) = t.sound_erf_range_bound(2.9, 3.1).unwrap();
        // erf(2.9) ≈ 0.999959, erf(3.1) ≈ 0.999998
        // The lo must be ≤ erf(2.9) (the minimum in the range, since erf is monotone)
        // The hi must be ≥ erf(3.1) (the maximum)
        assert!(lo <= 0.999959, "lo must be ≤ erf(2.9)≈0.999959, got {lo}");
        assert!(hi >= 0.999998, "hi must be ≥ erf(3.1)≈0.999998, got {hi}");
        assert!(hi > lo, "interval must be non-degenerate");
    }

    #[test]
    fn identity_on_non_smt_goal() {
        let t = AbstractSubterm::new();
        let goal = Goal::box_range(
            IntervalBox { dims: vec![] },
            OutputRange {
                output: "out".into(),
                lo: 0.0,
                hi: 1.0,
            },
        )
        .unwrap()
        .with_ir(IrHandle::unpopulated());
        let result = t.apply(&goal);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn identity_on_smt_goal_without_erf() {
        let t = AbstractSubterm::new();
        let prop = SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: vec![
                SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(SmtExpr::RealLit(0.0)),
                    Box::new(SmtExpr::Var("x".into())),
                ),
                SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(SmtExpr::Var("x".into())),
                    Box::new(SmtExpr::RealLit(1.0)),
                ),
            ],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::RealLit(2.0)),
            ),
        };
        let goal = Goal::smt(prop);
        let result = t.apply(&goal);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape, goal.shape);
    }

    #[test]
    fn transforms_erf_goal_with_bounded_arg() {
        let t = AbstractSubterm::new();
        let prop = SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: vec![
                SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(SmtExpr::RealLit(0.0)),
                    Box::new(SmtExpr::Var("x".into())),
                ),
                SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(SmtExpr::Var("x".into())),
                    Box::new(SmtExpr::RealLit(1.0)),
                ),
            ],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Apply("erf".into(), vec![SmtExpr::Var("x".into())])),
                Box::new(SmtExpr::RealLit(0.9)),
            ),
        };
        let goal = Goal::smt(prop);
        let result = t.apply(&goal);
        assert_eq!(result.len(), 1);
        let GoalShape::Smt(ref res_prop) = result[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(res_prop.variables.iter().any(|(n, _)| n == "__erf_abs_0"));
        assert!(!contains_erf(&res_prop.postcondition));
        assert!(res_prop.preconditions.len() > 2);
    }

    #[test]
    fn declines_when_arg_unbounded() {
        let t = AbstractSubterm::new();
        let prop = SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Apply("erf".into(), vec![SmtExpr::Var("x".into())])),
                Box::new(SmtExpr::RealLit(0.9)),
            ),
        };
        let goal = Goal::smt(prop.clone());
        let result = t.apply(&goal);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape, goal.shape);
    }

    #[test]
    fn harness_catches_too_tight_bound() {
        // Simulates a bogus transformation that makes an undischargeable goal
        // look dischargeable (the unsound direction the harness must catch).
        struct TooTightFake;
        impl Transformation for TooTightFake {
            fn name(&self) -> &str {
                "too_tight_fake"
            }
            fn apply(&self, _goal: &Goal) -> Vec<Goal> {
                let trivial = SmtProperty {
                    variables: vec![("y".into(), SmtSort::Real)],
                    preconditions: vec![
                        SmtExpr::Cmp(
                            CmpOp::Le,
                            Box::new(SmtExpr::RealLit(0.0)),
                            Box::new(SmtExpr::Var("y".into())),
                        ),
                        SmtExpr::Cmp(
                            CmpOp::Le,
                            Box::new(SmtExpr::Var("y".into())),
                            Box::new(SmtExpr::RealLit(1.0)),
                        ),
                    ],
                    postcondition: SmtExpr::Cmp(
                        CmpOp::Le,
                        Box::new(SmtExpr::Var("y".into())),
                        Box::new(SmtExpr::RealLit(2.0)),
                    ),
                };
                vec![Goal::smt(trivial)]
            }
        }

        let false_prop = SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: vec![
                SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(SmtExpr::RealLit(0.0)),
                    Box::new(SmtExpr::Var("x".into())),
                ),
                SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(SmtExpr::Var("x".into())),
                    Box::new(SmtExpr::RealLit(1.0)),
                ),
            ],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Apply("erf".into(), vec![SmtExpr::Var("x".into())])),
                Box::new(SmtExpr::RealLit(-2.0)),
            ),
        };
        let corpus = vec![CorpusEntry {
            goal: Goal::smt(false_prop),
            discharge: None,
        }];

        let results = run_harness(&TooTightFake, &corpus, |_goal| {
            // Oracle says the trivially-true residual IS dischargeable
            Some(
                Discharge::new(
                    Soundness::Exact,
                    QualifierSet::from_iter_kinds([Qualifier::Exact]),
                    TierBResult::Proved,
                    serde_json::Value::Null,
                )
                .unwrap(),
            )
        });
        assert!(!results.is_empty(), "harness must catch the laundering");
        assert!(
            results
                .iter()
                .any(|(_, r)| matches!(r, HarnessResult::LaunderedUndischargeable))
        );
    }

    fn contains_erf(expr: &SmtExpr) -> bool {
        match expr {
            SmtExpr::Apply(name, _) if name == "erf" => true,
            SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => contains_erf(l) || contains_erf(r),
            SmtExpr::Bool(_, children) => children.iter().any(contains_erf),
            SmtExpr::Not(inner) => contains_erf(inner),
            SmtExpr::Ite(c, t, e) => contains_erf(c) || contains_erf(t) || contains_erf(e),
            SmtExpr::Apply(_, args) => args.iter().any(contains_erf),
            SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => contains_erf(body),
            _ => false,
        }
    }
}
