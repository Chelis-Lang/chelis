//! Tier B: SMT solving via cvc5.
//!
//! Lowers property predicates to cvc5 terms and solves.
//! Returns Proved (UNSAT), Disproved (SAT with model), or Inconclusive
//! (timeout/unknown).

use serde_json::Value;

use crate::solver::{SmtExpr, SmtSort};

#[cfg(feature = "smt")]
use crate::solver::{ArithOp, BoolOp, CmpOp};

/// Tier B outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum TierBResult {
    /// Property proved (negation is UNSAT).
    Proved,
    /// Property disproved with counterexample.
    Disproved(Value),
    /// Solver timed out.
    Timeout,
    /// Solver returned unknown.
    Unknown,
    /// The property could not be lowered to a valid SMT term (e.g. a
    /// wrong-arity intrinsic application). Carries a human-facing reason.
    /// Routed to the same not-provable handling as Timeout/Unknown, never
    /// allowed to reach cvc5 as an invalid term (which would abort the
    /// solver with empty stdout).
    Error(String),
}

/// Structured property input for SMT solving.
#[derive(Debug, Clone)]
pub struct SmtProperty {
    /// Variable names and their sorts.
    pub variables: Vec<(String, SmtSort)>,
    /// Preconditions (all must hold).
    pub preconditions: Vec<SmtExpr>,
    /// Postcondition to prove (we assert its negation).
    pub postcondition: SmtExpr,
}

/// Run Tier B SMT check on a property source string.
///
/// Without the `smt` feature, returns Timeout (forces Tier C fallback).
pub fn solve(_property_source: &str, _property_name: &str, _timeout_ms: u64) -> TierBResult {
    #[cfg(feature = "smt")]
    {
        // Without a structured SmtProperty, we can't solve from raw source.
        // Use solve_property() with a structured input instead.
        TierBResult::Timeout
    }
    #[cfg(not(feature = "smt"))]
    {
        TierBResult::Timeout
    }
}

/// Solve a structured SMT property.
///
/// This is the primary entry point for Tier B when the caller has already
/// parsed the property into SmtExpr form.
pub fn solve_property(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    #[cfg(feature = "smt")]
    {
        solve_property_cvc5(property, timeout_ms)
    }
    #[cfg(not(feature = "smt"))]
    {
        let _ = (property, timeout_ms);
        TierBResult::Timeout
    }
}

/// The single source of truth for the intrinsics that [`lower_to_cvc5`]
/// can actually build a cvc5 term for, paired with their required arity.
///
/// CR2-1: three lists used to drift -- the arity guard, the lowering
/// match arms, and the transcendental-logic classifier. They are now all
/// derived from this constant (and [`CVC5_TRANSCENDENTAL`]) so a function
/// admitted by one cannot silently diverge from another.
///
/// `log` is deliberately ABSENT: cvc5 has no LOG kind, so `lower_to_cvc5`
/// cannot build a term for it. `log` is still a valid predicate intrinsic
/// (it is in [`chelis_pred::INTRINSIC_WHITELIST`]) and works at Tier C via
/// the concrete evaluator; the arity guard routes it to a clean
/// [`TierBResult::Error`] so dispatch falls through to Tier C rather than
/// reaching the lowering and panicking.
#[cfg(feature = "smt")]
const CVC5_LOWERABLE: &[(&str, usize)] = &[
    ("exp", 1),
    ("sqrt", 1),
    ("sin", 1),
    ("cos", 1),
    ("abs", 1),
    ("min", 2),
    ("max", 2),
];

/// The transcendental subset of [`CVC5_LOWERABLE`] whose presence selects
/// the `QF_NRAT` logic (vs `QF_NRA`). `abs`/`min`/`max` are algebraic and
/// stay in `QF_NRA`.
#[cfg(feature = "smt")]
const CVC5_TRANSCENDENTAL: &[&str] = &["exp", "sqrt", "sin", "cos"];

/// Required arity for a cvc5-lowerable intrinsic, or `None` if the name is
/// not in [`CVC5_LOWERABLE`].
#[cfg(feature = "smt")]
fn cvc5_lowerable_arity(name: &str) -> Option<usize> {
    CVC5_LOWERABLE
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, a)| *a)
}

/// Arity-validate every intrinsic application in an [`SmtExpr`] before it
/// reaches cvc5 (RT5-F1). Every application must name a cvc5-lowerable
/// intrinsic ([`CVC5_LOWERABLE`]) at exactly its required arity; otherwise
/// the lowering would either build an invalid cvc5 term (wrong arity) or
/// panic (a whitelisted-but-not-lowerable name such as `log`, CR2-1).
/// Returns the first offending application's reason so dispatch routes the
/// property to the same not-provable handling as Timeout/Unknown (and on
/// to Tier C, where `log` is supported by the concrete evaluator).
#[cfg(feature = "smt")]
fn validate_smt_arity(expr: &SmtExpr) -> Result<(), String> {
    match expr {
        SmtExpr::Apply(name, args) => {
            match cvc5_lowerable_arity(name) {
                Some(n) if args.len() == n => {}
                Some(n) => {
                    return Err(format!(
                        "intrinsic `{name}` expects {n} argument(s), got {}",
                        args.len()
                    ));
                }
                // Not cvc5-lowerable: either a whitelisted predicate
                // intrinsic with no cvc5 kind (e.g. `log`) or an unknown
                // name. Either way the lowering cannot handle it, so reject
                // cleanly here -- dispatch falls through to Tier C.
                None => {
                    return Err(format!(
                        "unsupported function `{name}` in cvc5 lowering (routes to Tier C)"
                    ));
                }
            }
            for a in args {
                validate_smt_arity(a)?;
            }
            Ok(())
        }
        SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
            validate_smt_arity(l)?;
            validate_smt_arity(r)
        }
        SmtExpr::Bool(_, children) => {
            for c in children {
                validate_smt_arity(c)?;
            }
            Ok(())
        }
        SmtExpr::Not(inner) | SmtExpr::Forall(_, inner) | SmtExpr::Exists(_, inner) => {
            validate_smt_arity(inner)
        }
        SmtExpr::Ite(c, t, e) => {
            validate_smt_arity(c)?;
            validate_smt_arity(t)?;
            validate_smt_arity(e)
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => Ok(()),
    }
}

#[cfg(feature = "smt")]
fn solve_property_cvc5(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    use cvc5_rs::{Kind, Solver, TermManager};
    use std::collections::HashMap;

    // Reject wrong-arity / unsupported intrinsics BEFORE building cvc5
    // terms (RT5-F1): a bad term aborts the solver with empty stdout, a
    // machine-contract violation for JSON consumers.
    if let Err(reason) = validate_smt_arity(&property.postcondition) {
        return TierBResult::Error(reason);
    }
    for pre in &property.preconditions {
        if let Err(reason) = validate_smt_arity(pre) {
            return TierBResult::Error(reason);
        }
    }

    let tm = TermManager::new();
    let mut solver = Solver::new(&tm);

    // Choose logic based on expression content
    let has_transcendentals = contains_transcendental(&property.postcondition)
        || property.preconditions.iter().any(contains_transcendental);
    if has_transcendentals {
        solver.set_logic("QF_NRAT");
    } else {
        solver.set_logic("QF_NRA");
    }
    solver.set_option("produce-models", "true");
    solver.set_option("tlimit-per", &timeout_ms.to_string());

    // 1. Declare variables
    let mut vars: HashMap<String, cvc5_rs::Term> = HashMap::new();
    for (name, sort) in &property.variables {
        let cvc5_sort = match sort {
            SmtSort::Real => tm.real_sort(),
            SmtSort::Int => tm.integer_sort(),
            SmtSort::Bool => tm.boolean_sort(),
        };
        let var = tm.mk_const(cvc5_sort, name);
        vars.insert(name.clone(), var);
    }

    // 2. Assert preconditions
    for pre in &property.preconditions {
        let term = match lower_to_cvc5(&tm, pre, &vars) {
            Ok(t) => t,
            Err(reason) => return TierBResult::Error(reason),
        };
        solver.assert_formula(term);
    }

    // 3. Assert negation of postcondition
    let post_term = match lower_to_cvc5(&tm, &property.postcondition, &vars) {
        Ok(t) => t,
        Err(reason) => return TierBResult::Error(reason),
    };
    let negated = tm.mk_term(Kind::CVC5_KIND_NOT, &[post_term]);
    solver.assert_formula(negated);

    // 4. Check satisfiability
    let result = solver.check_sat();
    if result.is_unsat() {
        // No counterexample exists → property proved
        TierBResult::Proved
    } else if result.is_sat() {
        // Counterexample found → extract model
        let mut bindings = serde_json::Map::new();
        for (name, var) in &vars {
            let val = solver.get_value(var.clone());
            bindings.insert(name.clone(), Value::String(val.to_string()));
        }
        TierBResult::Disproved(Value::Object(bindings))
    } else {
        // Unknown or timeout
        TierBResult::Unknown
    }
}

/// Lower an SmtExpr to a cvc5 Term.
///
/// Returns `Err` (rather than panicking) for any application that
/// [`validate_smt_arity`] would also reject -- a name with no cvc5 kind
/// (e.g. `log`, CR2-1) or a wrong arity. The arity guard already runs
/// before lowering in [`solve_property_cvc5`], so this is defense in depth:
/// the type system now forbids an unsupported function from silently
/// reaching cvc5 or crashing the prove process.
#[cfg(feature = "smt")]
pub fn lower_to_cvc5(
    tm: &cvc5_rs::TermManager,
    expr: &SmtExpr,
    vars: &std::collections::HashMap<String, cvc5_rs::Term>,
) -> Result<cvc5_rs::Term, String> {
    use cvc5_rs::Kind;

    Ok(match expr {
        SmtExpr::Var(name) => vars[name].clone(),
        SmtExpr::RealLit(value) => {
            // Use rational string representation for cvc5
            tm.mk_real_from_str(&format!("{value}"))
        }
        SmtExpr::IntLit(value) => tm.mk_integer(*value),
        SmtExpr::BoolLit(value) => {
            if *value {
                tm.mk_true()
            } else {
                tm.mk_false()
            }
        }
        SmtExpr::Arith(op, left, right) => {
            let l = lower_to_cvc5(tm, left, vars)?;
            let r = lower_to_cvc5(tm, right, vars)?;
            match op {
                ArithOp::Add => tm.mk_term(Kind::CVC5_KIND_ADD, &[l, r]),
                ArithOp::Sub => tm.mk_term(Kind::CVC5_KIND_SUB, &[l, r]),
                ArithOp::Mul => tm.mk_term(Kind::CVC5_KIND_MULT, &[l, r]),
                ArithOp::Div => tm.mk_term(Kind::CVC5_KIND_DIVISION, &[l, r]),
                ArithOp::Neg => tm.mk_term(Kind::CVC5_KIND_NEG, &[l]),
            }
        }
        SmtExpr::Cmp(op, left, right) => {
            let l = lower_to_cvc5(tm, left, vars)?;
            let r = lower_to_cvc5(tm, right, vars)?;
            match op {
                CmpOp::Lt => tm.mk_term(Kind::CVC5_KIND_LT, &[l, r]),
                CmpOp::Le => tm.mk_term(Kind::CVC5_KIND_LEQ, &[l, r]),
                CmpOp::Gt => tm.mk_term(Kind::CVC5_KIND_GT, &[l, r]),
                CmpOp::Ge => tm.mk_term(Kind::CVC5_KIND_GEQ, &[l, r]),
                CmpOp::Eq => tm.mk_term(Kind::CVC5_KIND_EQUAL, &[l, r]),
                CmpOp::Ne => {
                    let eq = tm.mk_term(Kind::CVC5_KIND_EQUAL, &[l, r]);
                    tm.mk_term(Kind::CVC5_KIND_NOT, &[eq])
                }
            }
        }
        SmtExpr::Bool(op, children) => {
            let terms: Vec<_> = children
                .iter()
                .map(|c| lower_to_cvc5(tm, c, vars))
                .collect::<Result<Vec<_>, _>>()?;
            match op {
                BoolOp::And => tm.mk_term(Kind::CVC5_KIND_AND, &terms),
                BoolOp::Or => tm.mk_term(Kind::CVC5_KIND_OR, &terms),
                BoolOp::Implies => tm.mk_term(Kind::CVC5_KIND_IMPLIES, &terms),
            }
        }
        SmtExpr::Not(inner) => {
            let t = lower_to_cvc5(tm, inner, vars)?;
            tm.mk_term(Kind::CVC5_KIND_NOT, &[t])
        }
        SmtExpr::Forall(bindings, body) => {
            let bound_vars: Vec<_> = bindings
                .iter()
                .map(|(name, sort)| {
                    let s = match sort {
                        SmtSort::Real => tm.real_sort(),
                        SmtSort::Int => tm.integer_sort(),
                        SmtSort::Bool => tm.boolean_sort(),
                    };
                    tm.mk_const(s, name)
                })
                .collect();
            let mut extended_vars = vars.clone();
            for (i, (name, _)) in bindings.iter().enumerate() {
                extended_vars.insert(name.clone(), bound_vars[i].clone());
            }
            let body_term = lower_to_cvc5(tm, body, &extended_vars)?;
            let bound_list = tm.mk_term(Kind::CVC5_KIND_VARIABLE_LIST, &bound_vars);
            tm.mk_term(Kind::CVC5_KIND_FORALL, &[bound_list, body_term])
        }
        SmtExpr::Exists(bindings, body) => {
            let bound_vars: Vec<_> = bindings
                .iter()
                .map(|(name, sort)| {
                    let s = match sort {
                        SmtSort::Real => tm.real_sort(),
                        SmtSort::Int => tm.integer_sort(),
                        SmtSort::Bool => tm.boolean_sort(),
                    };
                    tm.mk_const(s, name)
                })
                .collect();
            let mut extended_vars = vars.clone();
            for (i, (name, _)) in bindings.iter().enumerate() {
                extended_vars.insert(name.clone(), bound_vars[i].clone());
            }
            let body_term = lower_to_cvc5(tm, body, &extended_vars)?;
            let bound_list = tm.mk_term(Kind::CVC5_KIND_VARIABLE_LIST, &bound_vars);
            tm.mk_term(Kind::CVC5_KIND_EXISTS, &[bound_list, body_term])
        }
        SmtExpr::Apply(name, args) => {
            // Defense in depth: arity/lowerability is already enforced by
            // validate_smt_arity before any lowering runs, but re-check here
            // so this function can never build a malformed term or panic
            // (CR2-1). A non-lowerable name (e.g. `log`) returns Err.
            match cvc5_lowerable_arity(name) {
                Some(n) if args.len() == n => {}
                Some(n) => {
                    return Err(format!(
                        "intrinsic `{name}` expects {n} argument(s), got {}",
                        args.len()
                    ));
                }
                None => {
                    return Err(format!(
                        "unsupported function `{name}` in cvc5 lowering (routes to Tier C)"
                    ));
                }
            }
            let lowered_args: Vec<_> = args
                .iter()
                .map(|a| lower_to_cvc5(tm, a, vars))
                .collect::<Result<Vec<_>, _>>()?;
            match name.as_str() {
                "exp" => tm.mk_term(Kind::CVC5_KIND_EXPONENTIAL, &lowered_args),
                "sqrt" => tm.mk_term(Kind::CVC5_KIND_SQRT, &lowered_args),
                "sin" => tm.mk_term(Kind::CVC5_KIND_SINE, &lowered_args),
                "cos" => tm.mk_term(Kind::CVC5_KIND_COSINE, &lowered_args),
                "abs" => tm.mk_term(Kind::CVC5_KIND_ABS, &lowered_args),
                "min" if lowered_args.len() == 2 => {
                    let cond = tm.mk_term(
                        Kind::CVC5_KIND_LT,
                        &[lowered_args[0].clone(), lowered_args[1].clone()],
                    );
                    tm.mk_term(
                        Kind::CVC5_KIND_ITE,
                        &[cond, lowered_args[0].clone(), lowered_args[1].clone()],
                    )
                }
                "max" if lowered_args.len() == 2 => {
                    let cond = tm.mk_term(
                        Kind::CVC5_KIND_GT,
                        &[lowered_args[0].clone(), lowered_args[1].clone()],
                    );
                    tm.mk_term(
                        Kind::CVC5_KIND_ITE,
                        &[cond, lowered_args[0].clone(), lowered_args[1].clone()],
                    )
                }
                // Unreachable after the cvc5_lowerable_arity guard above;
                // returns Err rather than panicking if it is ever hit
                // (e.g. CVC5_LOWERABLE gains a name with no match arm).
                other => {
                    return Err(format!(
                        "unsupported function `{other}` in cvc5 lowering (routes to Tier C)"
                    ));
                }
            }
        }
        SmtExpr::Ite(cond, then_expr, else_expr) => {
            let c = lower_to_cvc5(tm, cond, vars)?;
            let t = lower_to_cvc5(tm, then_expr, vars)?;
            let e = lower_to_cvc5(tm, else_expr, vars)?;
            tm.mk_term(Kind::CVC5_KIND_ITE, &[c, t, e])
        }
    })
}

/// Check if an SmtExpr contains a cvc5 transcendental call (exp, sin, cos,
/// sqrt) -- the subset of [`CVC5_LOWERABLE`] that selects the `QF_NRAT`
/// logic. Derived from [`CVC5_TRANSCENDENTAL`] so it cannot drift from the
/// lowering (CR2-1). `abs`/`min`/`max` are algebraic and stay in `QF_NRA`.
#[cfg(feature = "smt")]
fn contains_transcendental(expr: &SmtExpr) -> bool {
    match expr {
        SmtExpr::Apply(name, args) => {
            CVC5_TRANSCENDENTAL.contains(&name.as_str())
                || args.iter().any(contains_transcendental)
        }
        SmtExpr::Arith(_, l, r) => contains_transcendental(l) || contains_transcendental(r),
        SmtExpr::Cmp(_, l, r) => contains_transcendental(l) || contains_transcendental(r),
        SmtExpr::Bool(_, children) => children.iter().any(contains_transcendental),
        SmtExpr::Not(inner) => contains_transcendental(inner),
        SmtExpr::Ite(c, t, e) => {
            contains_transcendental(c) || contains_transcendental(t) || contains_transcendental(e)
        }
        SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => contains_transcendental(body),
        _ => false,
    }
}

#[cfg(all(test, feature = "smt"))]
mod tests {
    use super::*;
    use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};

    #[test]
    fn proves_x_squared_non_negative() {
        // forall x: x*x >= 0
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::Var("x".to_string())),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let result = solve_property(&prop, 5000);
        assert_eq!(result, TierBResult::Proved);
    }

    #[test]
    fn disproves_x_always_positive() {
        // forall x: x > 0  (false — x=0 is a counterexample)
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let result = solve_property(&prop, 5000);
        match result {
            TierBResult::Disproved(model) => {
                assert!(model.is_object());
                assert!(model.get("x").is_some());
            }
            other => panic!("expected Disproved, got {other:?}"),
        }
    }

    #[test]
    fn proves_with_precondition() {
        // forall x where x > 0: x >= 0
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let result = solve_property(&prop, 5000);
        assert_eq!(result, TierBResult::Proved);
    }

    #[test]
    fn proves_intrinsic_value_non_negative() {
        // intrinsic_value(S, K) = if S > K then S - K else 0.0
        // Property: intrinsic_value(S, K) >= 0
        let prop = SmtProperty {
            variables: vec![
                ("S".to_string(), SmtSort::Real),
                ("K".to_string(), SmtSort::Real),
            ],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Ite(
                    Box::new(SmtExpr::Cmp(
                        CmpOp::Gt,
                        Box::new(SmtExpr::Var("S".to_string())),
                        Box::new(SmtExpr::Var("K".to_string())),
                    )),
                    Box::new(SmtExpr::Arith(
                        ArithOp::Sub,
                        Box::new(SmtExpr::Var("S".to_string())),
                        Box::new(SmtExpr::Var("K".to_string())),
                    )),
                    Box::new(SmtExpr::RealLit(0.0)),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let result = solve_property(&prop, 5000);
        assert_eq!(result, TierBResult::Proved);
    }

    fn intrinsic_ge_zero(name: &str, args: Vec<SmtExpr>) -> SmtProperty {
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(name.to_string(), args)),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        }
    }

    #[test]
    fn rt5_f1_zero_arg_transcendental_is_a_clean_error_not_a_cvc5_abort() {
        // RT5-F1: a zero-arg `exp()` previously built an invalid cvc5
        // EXPONENTIAL term (cvc5 aborts: empty stdout, bare exit 1). It
        // must lower to a clean TierBResult::Error, never an abort.
        let prop = intrinsic_ge_zero("exp", vec![]);
        match solve_property(&prop, 5000) {
            TierBResult::Error(reason) => {
                assert!(reason.contains("exp"), "names the bad intrinsic: {reason}");
            }
            other => panic!("expected Error for zero-arg exp(), got {other:?}"),
        }
    }

    #[test]
    fn rt5_f1_two_arg_transcendental_is_a_clean_error() {
        // The checker treats `exp` as variadic, so a two-arg `exp(a, b)`
        // reaches cvc5 for a "valid-looking" call. It must be a clean Error.
        let prop = intrinsic_ge_zero(
            "exp",
            vec![SmtExpr::Var("x".to_string()), SmtExpr::RealLit(1.0)],
        );
        match solve_property(&prop, 5000) {
            TierBResult::Error(reason) => assert!(reason.contains("exp"), "{reason}"),
            other => panic!("expected Error for two-arg exp(a,b), got {other:?}"),
        }
    }

    #[test]
    fn rt5_f1_wrong_arity_in_each_unary_transcendental_is_a_clean_error() {
        for name in ["exp", "log", "sqrt", "sin", "cos", "abs"] {
            // Zero args.
            assert!(
                matches!(
                    solve_property(&intrinsic_ge_zero(name, vec![]), 5000),
                    TierBResult::Error(_)
                ),
                "zero-arg {name}() must be a clean Error"
            );
            // Two args.
            let two =
                intrinsic_ge_zero(name, vec![SmtExpr::Var("x".into()), SmtExpr::RealLit(1.0)]);
            assert!(
                matches!(solve_property(&two, 5000), TierBResult::Error(_)),
                "two-arg {name}(a,b) must be a clean Error"
            );
        }
    }

    #[test]
    fn rt5_f1_correct_arity_transcendental_still_lowers_and_proves() {
        // Negative parity: a correct unary `exp(x) >= 0` still lowers and
        // proves (exp is always positive over the reals).
        let prop = intrinsic_ge_zero("exp", vec![SmtExpr::Var("x".to_string())]);
        assert_eq!(solve_property(&prop, 5000), TierBResult::Proved);
        // min/max keep their existing two-arg support.
        let max_prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(
                    "max".to_string(),
                    vec![SmtExpr::Var("x".to_string()), SmtExpr::RealLit(0.0)],
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        assert_eq!(solve_property(&max_prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn cr2_1_log_is_a_clean_unsupported_not_a_panic() {
        // CR2-1 CRITICAL: `log` is in the predicate grammar / chelis_pred
        // whitelist and works at Tier C, but cvc5 has no LOG kind, so
        // lower_to_cvc5 panics on it. A correct-arity `log(x)` must lower to
        // a clean Error (so dispatch falls to Tier C), never panic the
        // prove process. The old test suite only ever exercised exp(x).
        let prop = intrinsic_ge_zero("log", vec![SmtExpr::Var("x".to_string())]);
        match solve_property(&prop, 5000) {
            TierBResult::Error(reason) => {
                assert!(reason.contains("log"), "names the unsupported fn: {reason}");
            }
            other => panic!("expected Error for log(x), got {other:?}"),
        }
    }

    #[test]
    fn cr2_1_every_whitelisted_intrinsic_at_correct_arity_does_not_panic() {
        // Loop over EVERY whitelisted intrinsic at its correct arity so no
        // intrinsic is left untested -- that gap hid `log` twice. Each must
        // produce a determinate, panic-free outcome: a cvc5-lowerable fn
        // proves/disproves/times-out; a whitelisted-but-not-lowerable fn
        // (log) is a clean Error.
        for &name in chelis_pred::INTRINSIC_WHITELIST {
            let arity = if name == "min" || name == "max" { 2 } else { 1 };
            let args: Vec<SmtExpr> = (0..arity).map(|_| SmtExpr::Var("x".into())).collect();
            let prop = intrinsic_ge_zero(name, args);
            // The point is the absence of a panic and a determinate result.
            let result = solve_property(&prop, 5000);
            assert!(
                matches!(
                    result,
                    TierBResult::Proved
                        | TierBResult::Disproved(_)
                        | TierBResult::Timeout
                        | TierBResult::Unknown
                        | TierBResult::Error(_)
                ),
                "intrinsic `{name}` at arity {arity} must yield a determinate result, got {result:?}"
            );
        }
    }

    #[test]
    fn cr2_1_cvc5_lowerable_set_matches_lowering_and_transcendental_classification() {
        // The single source of truth (CVC5_LOWERABLE) must (a) NOT contain
        // log (cvc5 has no LOG), (b) be a subset of the predicate-grammar
        // whitelist, and (c) drive contains_transcendental so the three
        // lists cannot drift again.
        assert!(
            !CVC5_LOWERABLE.iter().any(|(n, _)| *n == "log"),
            "log is not cvc5-lowerable"
        );
        for &(f, _) in CVC5_LOWERABLE {
            assert!(
                chelis_pred::INTRINSIC_WHITELIST.contains(&f),
                "cvc5-lowerable `{f}` must be in the predicate grammar"
            );
        }
        // CVC5_TRANSCENDENTAL is a strict subset of the lowerable names.
        for &t in CVC5_TRANSCENDENTAL {
            assert!(
                CVC5_LOWERABLE.iter().any(|(n, _)| *n == t),
                "transcendental `{t}` must be cvc5-lowerable"
            );
        }
        // contains_transcendental agrees with CVC5_TRANSCENDENTAL membership
        // for every lowerable intrinsic (abs/min/max are algebraic).
        for &(f, _) in CVC5_LOWERABLE {
            let is_transcendental = CVC5_TRANSCENDENTAL.contains(&f);
            let expr = SmtExpr::Apply(f.to_string(), vec![SmtExpr::Var("x".into())]);
            assert_eq!(
                contains_transcendental(&expr),
                is_transcendental,
                "contains_transcendental disagrees on `{f}`"
            );
        }
    }
}
