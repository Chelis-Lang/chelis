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
//! # Function registry (chelis#434)
//!
//! The finder is driven by [`SpecialFnRegistry`]: it abstracts any
//! `Apply(f, [arg])` where `f` is a known special function `{erf, exp, log,
//! sqrt}` AND a certified envelope for `f` is committed. Today only `erf` has
//! committed data, so `exp`/`log`/`sqrt` sites DECLINE (identity) — the honest
//! floor until their certified envelopes land. Each function carries a
//! [`Domain`] guard (`log` needs `arg > 0`, `sqrt` needs `arg >= 0`); a site
//! whose argument range is not provably inside the domain declines.
//!
//! # Sound range evaluation
//!
//! The envelope's `bound(x)` gives a sound interval for a SINGLE point. This
//! transformation needs a sound interval over the argument's ENTIRE RANGE, from
//! [`SpecialFnEnvelope::sound_range_bound`] (the hull across every box the range
//! touches; sound because every box's eps is a certified sup-norm).
//!
//! If the argument range is not statically boundable (no preconditions pin it),
//! extends outside the envelope's covered domain, or violates the function's
//! domain guard, the transformation DECLINES (returns identity). It never guesses.

use std::collections::{BTreeMap, HashMap};

use crate::discharge::{Goal, GoalShape};
use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};
use crate::special_fn_envelope::{SpecialFnEnvelope, SpecialFnRegistry};
use crate::tier_b::SmtProperty;
use crate::transformation::Transformation;

/// The abstract-subterm transformation over the special-function registry.
pub struct AbstractSubterm {
    /// Committed certified envelopes to consult, keyed by function name. In
    /// production this is loaded from [`SpecialFnRegistry::committed`] (today:
    /// `erf` only). A function with no entry here DECLINES.
    envelopes: HashMap<String, SpecialFnEnvelope>,
}

impl AbstractSubterm {
    pub fn new() -> Self {
        let mut envelopes = HashMap::new();
        for &f in SpecialFnRegistry::known_functions() {
            if let Some(env) = SpecialFnRegistry::committed(f) {
                envelopes.insert(f.to_string(), env);
            }
        }
        Self { envelopes }
    }

    /// The committed envelope for `fn_name`, if any is loaded.
    fn envelope_for(&self, fn_name: &str) -> Option<&SpecialFnEnvelope> {
        self.envelopes.get(fn_name)
    }

    /// Test-only constructor that injects synthetic certified-shaped envelopes,
    /// so the generalized finder + domain guards can be exercised for
    /// `exp`/`log`/`sqrt` before their real certified data lands. Production only
    /// ever loads committed data via [`AbstractSubterm::new`].
    #[cfg(test)]
    fn with_envelopes(envelopes: HashMap<String, SpecialFnEnvelope>) -> Self {
        Self { envelopes }
    }

    /// Compute a sound bound on `erf(x)` for ALL x in `[arg_lo, arg_hi]`.
    ///
    /// Retained as the erf-specific entry point (the generic engine is
    /// [`SpecialFnEnvelope::sound_range_bound`]); returns `None` if erf has no
    /// loaded envelope or the range is outside its covered domain.
    pub fn sound_erf_range_bound(&self, arg_lo: f64, arg_hi: f64) -> Option<(f64, f64)> {
        self.envelope_for("erf")?.sound_range_bound(arg_lo, arg_hi)
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

        // Find special-function applications in the postcondition.
        let sites =
            find_special_fn_applications(&prop.postcondition, SpecialFnRegistry::known_functions());
        if sites.is_empty() {
            return vec![goal.clone()]; // no transcendentals to abstract
        }

        let mut new_variables = prop.variables.clone();
        let mut new_preconditions = prop.preconditions.clone();
        let mut postcondition = prop.postcondition.clone();

        for (fresh_counter, site) in sites.iter().enumerate() {
            // A certified envelope must be committed for this function, else
            // decline (the honest floor for exp/log/sqrt today).
            let Some(envelope) = self.envelope_for(&site.fn_name) else {
                return vec![goal.clone()];
            };

            // Determine the argument's static range from the preconditions.
            let Some((arg_lo, arg_hi)) =
                extract_variable_range(&site.argument, &prop.preconditions)
            else {
                // Can't determine argument range statically → decline.
                return vec![goal.clone()];
            };

            // Domain guard: the whole argument range must be provably inside the
            // function's domain (log arg>0, sqrt arg>=0), else decline.
            if !envelope.domain.covers(arg_lo, arg_hi) {
                return vec![goal.clone()];
            }

            // Sound envelope bound over the argument range.
            let Some((env_lo, env_hi)) = envelope.sound_range_bound(arg_lo, arg_hi) else {
                // Range outside envelope coverage → decline.
                return vec![goal.clone()];
            };

            // Fresh variable, named by function: `__<fn>_abs_<n>`.
            let fresh_name = format!("__{}_abs_{}", site.fn_name, fresh_counter);
            new_variables.push((fresh_name.clone(), SmtSort::Real));

            // Bounds as preconditions: env_lo <= fresh_var <= env_hi.
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

            // Substitute f(arg) with fresh_var in the postcondition.
            postcondition = substitute_call(
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

/// A located special-function application in an SmtExpr tree.
struct SpecialFnSite {
    /// The function name (`"erf"`, `"exp"`, ...).
    fn_name: String,
    /// The argument expression inside `f(...)`.
    argument: SmtExpr,
    /// The full `Apply(f, [arg])` expression for substitution matching.
    original_expr: SmtExpr,
}

/// Find all `Apply(f, [arg])` nodes whose `f` is a known special function.
fn find_special_fn_applications(expr: &SmtExpr, known: &[&str]) -> Vec<SpecialFnSite> {
    let mut sites = Vec::new();
    find_special_fn_recursive(expr, known, &mut sites);
    sites
}

fn find_special_fn_recursive(expr: &SmtExpr, known: &[&str], sites: &mut Vec<SpecialFnSite>) {
    match expr {
        SmtExpr::Apply(name, args) if args.len() == 1 && known.contains(&name.as_str()) => {
            sites.push(SpecialFnSite {
                fn_name: name.clone(),
                argument: args[0].clone(),
                original_expr: expr.clone(),
            });
            // Also recurse into the argument in case of a nested special fn.
            find_special_fn_recursive(&args[0], known, sites);
        }
        SmtExpr::Arith(_, l, r) => {
            find_special_fn_recursive(l, known, sites);
            find_special_fn_recursive(r, known, sites);
        }
        SmtExpr::Cmp(_, l, r) => {
            find_special_fn_recursive(l, known, sites);
            find_special_fn_recursive(r, known, sites);
        }
        SmtExpr::Bool(_, children) => {
            for c in children {
                find_special_fn_recursive(c, known, sites);
            }
        }
        SmtExpr::Not(inner) => find_special_fn_recursive(inner, known, sites),
        SmtExpr::Ite(c, t, e) => {
            find_special_fn_recursive(c, known, sites);
            find_special_fn_recursive(t, known, sites);
            find_special_fn_recursive(e, known, sites);
        }
        SmtExpr::Apply(_, args) => {
            for a in args {
                find_special_fn_recursive(a, known, sites);
            }
        }
        SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => {
            find_special_fn_recursive(body, known, sites);
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {}
    }
}

/// A canonical affine form `Σ coeff_i · var_i + constant` over SMT variables.
/// Each variable appears at most once (coefficients are merged), so evaluating
/// the interval by summing per-variable contributions is EXACT — there is no
/// interval-arithmetic dependency error from a repeated variable.
#[derive(Debug, Clone, PartialEq)]
struct AffineForm {
    terms: BTreeMap<String, f64>,
    constant: f64,
}

impl AffineForm {
    fn constant(c: f64) -> Self {
        Self {
            terms: BTreeMap::new(),
            constant: c,
        }
    }

    fn var(name: &str) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(name.to_string(), 1.0);
        Self {
            terms,
            constant: 0.0,
        }
    }

    /// A pure constant (no variables) — the scalar for `Mul`/`Div` scaling.
    fn as_constant(&self) -> Option<f64> {
        self.terms.is_empty().then_some(self.constant)
    }

    /// Multiply the whole form by a scalar. Fails closed (`None`) if any
    /// resulting coefficient/constant is non-finite (overflow guard).
    fn scale(mut self, k: f64) -> Option<Self> {
        if !k.is_finite() {
            return None;
        }
        self.constant *= k;
        if !self.constant.is_finite() {
            return None;
        }
        for c in self.terms.values_mut() {
            *c *= k;
            if !c.is_finite() {
                return None;
            }
        }
        // A zero coefficient drops the variable (0·x contributes nothing and
        // must NOT pull that variable's boundedness requirement in).
        self.terms.retain(|_, c| *c != 0.0);
        Some(self)
    }

    /// Add two forms (merge coefficients, add constants). Fails closed on a
    /// non-finite result.
    fn add(mut self, other: Self) -> Option<Self> {
        self.constant += other.constant;
        if !self.constant.is_finite() {
            return None;
        }
        for (name, c) in other.terms {
            let e = self.terms.entry(name).or_insert(0.0);
            *e += c;
            if !e.is_finite() {
                return None;
            }
        }
        self.terms.retain(|_, c| *c != 0.0);
        Some(self)
    }
}

/// Parse `expr` as an affine form over SMT variables, or `None` if it is not
/// affine. FAIL-CLOSED by construction: any nonlinear node (variable×variable,
/// division by a non-constant or by zero, a transcendental `Apply`, an `Ite`, a
/// comparison/boolean) returns `None`. Only `+ - unary-neg`, and `× ÷` by a
/// *nonzero constant*, are affine-preserving.
fn to_affine(expr: &SmtExpr) -> Option<AffineForm> {
    match expr {
        SmtExpr::Var(n) => Some(AffineForm::var(n)),
        SmtExpr::RealLit(v) => Some(AffineForm::constant(*v)),
        SmtExpr::IntLit(v) => Some(AffineForm::constant(*v as f64)),
        // Unary negation is lowered as Arith(Neg, inner, <dummy 0>).
        SmtExpr::Arith(ArithOp::Neg, inner, _) => to_affine(inner)?.scale(-1.0),
        SmtExpr::Arith(ArithOp::Add, l, r) => to_affine(l)?.add(to_affine(r)?),
        SmtExpr::Arith(ArithOp::Sub, l, r) => {
            let rn = to_affine(r)?.scale(-1.0)?;
            to_affine(l)?.add(rn)
        }
        SmtExpr::Arith(ArithOp::Mul, l, r) => {
            let (la, ra) = (to_affine(l)?, to_affine(r)?);
            // Affine only if at least one side is a pure constant.
            if let Some(k) = ra.as_constant() {
                la.scale(k)
            } else if let Some(k) = la.as_constant() {
                ra.scale(k)
            } else {
                None // variable × variable is nonlinear
            }
        }
        SmtExpr::Arith(ArithOp::Div, l, r) => {
            // Affine only if the denominator is a NONZERO constant. Division by
            // a variable (which could be zero) or by zero fails closed.
            let k = to_affine(r)?.as_constant()?;
            if k == 0.0 || !k.is_finite() {
                return None;
            }
            to_affine(l)?.scale(1.0 / k)
        }
        // Everything else is non-affine: transcendental Apply, Ite, Cmp, Bool,
        // Not, quantifiers, bool literals.
        _ => None,
    }
}

/// Extract the static range `[lo, hi]` of an argument expression from the
/// preconditions. Handles a bare `Var` (its precondition bounds) AND any AFFINE
/// form `a·x + b·y + … + c` over bounded variables, via sound interval
/// arithmetic on the canonical (coefficient-merged) form. Returns `None`
/// (fail-closed) if the expression is non-affine or any variable in it is not
/// bounded by the preconditions.
fn extract_variable_range(expr: &SmtExpr, preconditions: &[SmtExpr]) -> Option<(f64, f64)> {
    let form = to_affine(expr)?;
    affine_range(&form, preconditions)
}

/// Evaluate the sound interval of an affine form over the variable ranges pinned
/// by the preconditions. Each variable appears once (merged coefficient), so the
/// per-variable interval sum is exact. Fails closed if any variable is unbounded
/// or the result is non-finite / empty.
fn affine_range(form: &AffineForm, preconditions: &[SmtExpr]) -> Option<(f64, f64)> {
    let mut lo = form.constant;
    let mut hi = form.constant;
    for (var, &coeff) in &form.terms {
        let (vlo, vhi) = bare_var_range(var, preconditions)?; // unbounded → fail closed
        // coeff·[vlo,vhi]: flip endpoints for a negative coefficient.
        let (tlo, thi) = if coeff >= 0.0 {
            (coeff * vlo, coeff * vhi)
        } else {
            (coeff * vhi, coeff * vlo)
        };
        lo += tlo;
        hi += thi;
    }
    if lo.is_finite() && hi.is_finite() && lo <= hi {
        Some((lo, hi))
    } else {
        None
    }
}

/// The static range `[lo, hi]` of a single bare variable from the preconditions:
/// the tightest `lo <= x` / `x <= hi` (also under `And` conjunctions). `None` if
/// either side is missing (the variable is unbounded). This is the pre-affine
/// bare-`Var` extractor, unchanged in behavior.
fn bare_var_range(var_name: &str, preconditions: &[SmtExpr]) -> Option<(f64, f64)> {
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;

    for pre in preconditions {
        match pre {
            SmtExpr::Cmp(CmpOp::Le, left, right)
            | SmtExpr::Cmp(CmpOp::Ge, right, left)
            | SmtExpr::Cmp(CmpOp::Lt, left, right)
            | SmtExpr::Cmp(CmpOp::Gt, right, left) => {
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
            SmtExpr::Bool(BoolOp::And, children) => {
                if let Some((clo, chi)) = bare_var_range(var_name, children) {
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
fn substitute_call(expr: &SmtExpr, target: &SmtExpr, replacement: &SmtExpr) -> SmtExpr {
    if expr == target {
        return replacement.clone();
    }
    match expr {
        SmtExpr::Arith(op, l, r) => SmtExpr::Arith(
            *op,
            Box::new(substitute_call(l, target, replacement)),
            Box::new(substitute_call(r, target, replacement)),
        ),
        SmtExpr::Cmp(op, l, r) => SmtExpr::Cmp(
            *op,
            Box::new(substitute_call(l, target, replacement)),
            Box::new(substitute_call(r, target, replacement)),
        ),
        SmtExpr::Bool(op, children) => SmtExpr::Bool(
            *op,
            children
                .iter()
                .map(|c| substitute_call(c, target, replacement))
                .collect(),
        ),
        SmtExpr::Not(inner) => SmtExpr::Not(Box::new(substitute_call(inner, target, replacement))),
        SmtExpr::Ite(c, t, e) => SmtExpr::Ite(
            Box::new(substitute_call(c, target, replacement)),
            Box::new(substitute_call(t, target, replacement)),
            Box::new(substitute_call(e, target, replacement)),
        ),
        SmtExpr::Apply(name, args) => SmtExpr::Apply(
            name.clone(),
            args.iter()
                .map(|a| substitute_call(a, target, replacement))
                .collect(),
        ),
        SmtExpr::Forall(vars, body) => SmtExpr::Forall(
            vars.clone(),
            Box::new(substitute_call(body, target, replacement)),
        ),
        SmtExpr::Exists(vars, body) => SmtExpr::Exists(
            vars.clone(),
            Box::new(substitute_call(body, target, replacement)),
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

    // ─── chelis#434: generalized-finder tests (exp/log/sqrt + domain guards) ──

    use crate::erf_envelope::ProofKind;
    use crate::special_fn_envelope::{
        Domain, EnvelopeArm, SpecialFnEnvelope, SpecialFnEnvelopeBox, SpecialFnProvenance,
    };
    use std::collections::HashMap;

    /// A structurally-valid single-box synthetic envelope (a constant Saturation
    /// arm) for exercising the finder's mechanics without certified data.
    fn synthetic_env(
        fn_name: &str,
        domain: Domain,
        lo: f64,
        hi: f64,
        value: f64,
        eps: f64,
    ) -> SpecialFnEnvelope {
        SpecialFnEnvelope {
            fn_name: fn_name.to_string(),
            domain,
            output_clamp: None,
            boxes: vec![SpecialFnEnvelopeBox {
                lo,
                hi,
                arm: EnvelopeArm::Saturation { value },
                eps,
                proof_kind: ProofKind::Gappa,
            }],
            provenance: SpecialFnProvenance::default(),
        }
    }

    /// `f(x)` postcondition `Apply(f, [x])` compared `< bound`, with `lo <= x <= hi`.
    fn fn_of_bare_var_goal(f: &str, lo: f64, hi: f64, bound: f64) -> Goal {
        let prop = SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: vec![
                SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(SmtExpr::RealLit(lo)),
                    Box::new(SmtExpr::Var("x".into())),
                ),
                SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(SmtExpr::Var("x".into())),
                    Box::new(SmtExpr::RealLit(hi)),
                ),
            ],
            postcondition: SmtExpr::Cmp(
                CmpOp::Lt,
                Box::new(SmtExpr::Apply(f.into(), vec![SmtExpr::Var("x".into())])),
                Box::new(SmtExpr::RealLit(bound)),
            ),
        };
        Goal::smt(prop)
    }

    fn contains_fn(expr: &SmtExpr, name: &str) -> bool {
        match expr {
            SmtExpr::Apply(n, args) => n == name || args.iter().any(|a| contains_fn(a, name)),
            SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
                contains_fn(l, name) || contains_fn(r, name)
            }
            SmtExpr::Bool(_, children) => children.iter().any(|c| contains_fn(c, name)),
            SmtExpr::Not(inner) => contains_fn(inner, name),
            SmtExpr::Ite(c, t, e) => {
                contains_fn(c, name) || contains_fn(t, name) || contains_fn(e, name)
            }
            SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => contains_fn(body, name),
            _ => false,
        }
    }

    #[test]
    fn exp_abstracts_in_production_via_committed_envelope() {
        // exp now has committed Arb-certified data over [-2,2], so a bounded exp
        // site abstracts in PRODUCTION (AbstractSubterm::new(), no injection).
        // The emitted band is sound: exp([0,1]) = [1, e] ⊆ the __exp_abs_0 bounds.
        let t = AbstractSubterm::new();
        let goal = fn_of_bare_var_goal("exp", 0.0, 1.0, 3.0);
        let out = t.apply(&goal);
        let GoalShape::Smt(ref rp) = out[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(
            rp.variables.iter().any(|(n, _)| n == "__exp_abs_0"),
            "exp must abstract"
        );
        assert!(!contains_fn(&rp.postcondition, "exp"));
    }

    #[test]
    fn declines_log_site_without_committed_envelope() {
        // log has no committed envelope yet, so a log site is left intact (the
        // honest floor). The goal is returned unchanged.
        let t = AbstractSubterm::new();
        let goal = fn_of_bare_var_goal("log", 0.5, 2.0, 3.0);
        let result = t.apply(&goal);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].shape, goal.shape,
            "log site must decline unchanged"
        );
    }

    #[test]
    fn abstracts_exp_with_injected_envelope() {
        // With a synthetic exp envelope over [0,1], the finder abstracts exp(x)
        // into a fresh `__exp_abs_0` var and removes exp from the postcondition.
        let mut envs = HashMap::new();
        envs.insert(
            "exp".to_string(),
            synthetic_env("exp", Domain::AllReals, 0.0, 1.0, 2.0, 0.9),
        );
        let t = AbstractSubterm::with_envelopes(envs);
        let goal = fn_of_bare_var_goal("exp", 0.0, 1.0, 3.0);
        let result = t.apply(&goal);
        assert_eq!(result.len(), 1);
        let GoalShape::Smt(ref rp) = result[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(
            rp.variables.iter().any(|(n, _)| n == "__exp_abs_0"),
            "fresh var must be named by function: {:?}",
            rp.variables
        );
        assert!(
            !contains_fn(&rp.postcondition, "exp"),
            "exp must be abstracted out"
        );
        assert!(rp.preconditions.len() > 2, "envelope bounds added");
    }

    #[test]
    fn log_domain_guard_declines_nonpositive_range() {
        // log needs arg > 0. A range that dips to/below 0 must DECLINE even with
        // an envelope present; a strictly-positive range transforms.
        let mut envs = HashMap::new();
        envs.insert(
            "log".to_string(),
            synthetic_env("log", Domain::Positive, 0.25, 4.0, 0.0, 1.5),
        );
        let t = AbstractSubterm::with_envelopes(envs);

        // range [-1, 2] includes non-positive x → decline
        let bad = fn_of_bare_var_goal("log", -1.0, 2.0, 10.0);
        assert_eq!(
            t.apply(&bad)[0].shape,
            bad.shape,
            "log over [-1,2] must decline"
        );

        // range [0.5, 2] strictly positive → transform
        let good = fn_of_bare_var_goal("log", 0.5, 2.0, 10.0);
        let out = t.apply(&good);
        let GoalShape::Smt(ref rp) = out[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(rp.variables.iter().any(|(n, _)| n == "__log_abs_0"));
        assert!(!contains_fn(&rp.postcondition, "log"));
    }

    #[test]
    fn sqrt_domain_guard_allows_zero_declines_negative() {
        // sqrt needs arg >= 0: 0 is allowed, a negative lower edge declines.
        let mut envs = HashMap::new();
        envs.insert(
            "sqrt".to_string(),
            synthetic_env("sqrt", Domain::NonNegative, 0.0, 9.0, 0.0, 3.0),
        );
        let t = AbstractSubterm::with_envelopes(envs);

        let neg = fn_of_bare_var_goal("sqrt", -0.1, 4.0, 10.0);
        assert_eq!(
            t.apply(&neg)[0].shape,
            neg.shape,
            "sqrt over [-0.1,4] must decline"
        );

        let ok = fn_of_bare_var_goal("sqrt", 0.0, 4.0, 10.0);
        let out = t.apply(&ok);
        let GoalShape::Smt(ref rp) = out[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(rp.variables.iter().any(|(n, _)| n == "__sqrt_abs_0"));
    }

    #[test]
    fn declines_when_range_outside_envelope_coverage() {
        // The envelope covers only [0,1]; a bounded arg range outside it declines
        // (the sound_range_bound coverage check), even though the domain admits it.
        let mut envs = HashMap::new();
        envs.insert(
            "exp".to_string(),
            synthetic_env("exp", Domain::AllReals, 0.0, 1.0, 2.0, 0.9),
        );
        let t = AbstractSubterm::with_envelopes(envs);
        let goal = fn_of_bare_var_goal("exp", 2.0, 3.0, 100.0); // outside [0,1]
        assert_eq!(
            t.apply(&goal)[0].shape,
            goal.shape,
            "outside coverage must decline"
        );
    }

    // ─── chelis#434: affine-argument propagation (a·x + b) + adversarial ──────

    fn v(n: &str) -> SmtExpr {
        SmtExpr::Var(n.into())
    }
    fn r(x: f64) -> SmtExpr {
        SmtExpr::RealLit(x)
    }
    fn mul(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::Arith(ArithOp::Mul, Box::new(a), Box::new(b))
    }
    fn add(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::Arith(ArithOp::Add, Box::new(a), Box::new(b))
    }
    fn sub(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::Arith(ArithOp::Sub, Box::new(a), Box::new(b))
    }
    fn div(a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::Arith(ArithOp::Div, Box::new(a), Box::new(b))
    }
    fn neg(a: SmtExpr) -> SmtExpr {
        SmtExpr::Arith(ArithOp::Neg, Box::new(a), Box::new(SmtExpr::IntLit(0)))
    }
    /// preconditions `lo <= x <= hi` for each (name, lo, hi).
    fn box_pre(bounds: &[(&str, f64, f64)]) -> Vec<SmtExpr> {
        let mut out = Vec::new();
        for (n, lo, hi) in bounds {
            out.push(SmtExpr::Cmp(CmpOp::Le, Box::new(r(*lo)), Box::new(v(n))));
            out.push(SmtExpr::Cmp(CmpOp::Le, Box::new(v(n)), Box::new(r(*hi))));
        }
        out
    }

    #[test]
    fn affine_scale_and_shift_range() {
        // 2·x + 1 over x∈[0,1] → [1, 3].
        let pre = box_pre(&[("x", 0.0, 1.0)]);
        let arg = add(mul(r(2.0), v("x")), r(1.0));
        assert_eq!(extract_variable_range(&arg, &pre), Some((1.0, 3.0)));
    }

    #[test]
    fn affine_negative_coeff_flips_endpoints() {
        let pre = box_pre(&[("x", 0.0, 1.0)]);
        assert_eq!(
            extract_variable_range(&neg(v("x")), &pre),
            Some((-1.0, 0.0))
        );
        // 1 - 2x over [0,1] → [1-2, 1-0] = [-1, 1].
        let arg = sub(r(1.0), mul(r(2.0), v("x")));
        assert_eq!(extract_variable_range(&arg, &pre), Some((-1.0, 1.0)));
    }

    #[test]
    fn affine_div_by_constant() {
        let pre = box_pre(&[("x", 0.0, 4.0)]);
        assert_eq!(
            extract_variable_range(&div(v("x"), r(2.0)), &pre),
            Some((0.0, 2.0))
        );
        // x / √2 (the normal_cdf scale) over [0, √2] → [0, 1].
        let inv = crate::transformations::normal_cdf_erf::INV_SQRT2;
        let pre2 = box_pre(&[("x", 0.0, std::f64::consts::SQRT_2)]);
        let (lo, hi) = extract_variable_range(&mul(v("x"), r(inv)), &pre2).unwrap();
        assert!(
            (lo - 0.0).abs() < 1e-12 && (hi - 1.0).abs() < 1e-12,
            "[{lo},{hi}]"
        );
    }

    #[test]
    fn affine_multi_var_sums_intervals() {
        let pre = box_pre(&[("x", 0.0, 1.0), ("y", 2.0, 3.0)]);
        assert_eq!(
            extract_variable_range(&add(v("x"), v("y")), &pre),
            Some((2.0, 4.0))
        );
    }

    #[test]
    fn affine_repeated_var_merges_no_dependency_error() {
        // The dependency-error guard: x - x must be EXACTLY [0,0] (merged to 0),
        // not the naive interval [0-1, 1-0] = [-1, 1]; x + x must be 2·[0,1]=[0,2].
        let pre = box_pre(&[("x", 0.0, 1.0)]);
        assert_eq!(
            extract_variable_range(&sub(v("x"), v("x")), &pre),
            Some((0.0, 0.0))
        );
        assert_eq!(
            extract_variable_range(&add(v("x"), v("x")), &pre),
            Some((0.0, 2.0))
        );
    }

    #[test]
    fn nonaffine_fails_closed() {
        let pre = box_pre(&[("x", 1.0, 2.0), ("y", 1.0, 2.0)]);
        // var × var
        assert_eq!(extract_variable_range(&mul(v("x"), v("y")), &pre), None);
        // division by a variable (could be zero)
        assert_eq!(extract_variable_range(&div(v("x"), v("y")), &pre), None);
        // division by a zero constant
        assert_eq!(extract_variable_range(&div(v("x"), r(0.0)), &pre), None);
        // a transcendental Apply inside the argument
        let inner = SmtExpr::Apply("erf".into(), vec![v("x")]);
        assert_eq!(extract_variable_range(&add(inner, r(1.0)), &pre), None);
    }

    #[test]
    fn affine_unbounded_leaf_fails_closed() {
        // 2x+1 with x unbounded → None.
        assert_eq!(
            extract_variable_range(&add(mul(r(2.0), v("x")), r(1.0)), &[]),
            None
        );
        // x + y with only x bounded → None (y unbounded).
        let pre = box_pre(&[("x", 0.0, 1.0)]);
        assert_eq!(extract_variable_range(&add(v("x"), v("y")), &pre), None);
    }

    #[test]
    fn bare_var_still_identical_after_affine_refactor() {
        // The bare-Var path must be byte-identical to the pre-affine behavior.
        let pre = box_pre(&[("x", -2.0, 5.0)]);
        assert_eq!(extract_variable_range(&v("x"), &pre), Some((-2.0, 5.0)));
        assert_eq!(extract_variable_range(&v("x"), &[]), None);
    }

    #[test]
    fn transforms_erf_of_affine_argument_end_to_end() {
        // erf(x/2 + 1) with x∈[0,2] → arg range [1,2] (covered) → abstracts.
        let t = AbstractSubterm::new();
        let arg = add(div(v("x"), r(2.0)), r(1.0));
        let prop = SmtProperty {
            variables: vec![("x".into(), SmtSort::Real)],
            preconditions: box_pre(&[("x", 0.0, 2.0)]),
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Apply("erf".into(), vec![arg])),
                Box::new(r(0.999)),
            ),
        };
        let out = t.apply(&Goal::smt(prop));
        let GoalShape::Smt(ref rp) = out[0].shape else {
            panic!("expected Smt goal");
        };
        assert!(
            rp.variables.iter().any(|(n, _)| n == "__erf_abs_0"),
            "affine erf must abstract"
        );
        assert!(!contains_erf(&rp.postcondition), "erf must be gone");
    }
}
