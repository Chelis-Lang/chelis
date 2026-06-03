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

#[cfg(feature = "smt")]
fn solve_property_cvc5(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    use cvc5_rs::{Kind, Solver, TermManager};
    use std::collections::HashMap;

    let tm = TermManager::new();
    let mut solver = Solver::new(&tm);

    // Choose logic based on expression content
    let has_transcendentals = contains_transcendental(&property.postcondition)
        || property
            .preconditions
            .iter()
            .any(|p| contains_transcendental(p));
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
        let term = lower_to_cvc5(&tm, pre, &vars);
        solver.assert_formula(term);
    }

    // 3. Assert negation of postcondition
    let post_term = lower_to_cvc5(&tm, &property.postcondition, &vars);
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
#[cfg(feature = "smt")]
pub fn lower_to_cvc5(
    tm: &cvc5_rs::TermManager,
    expr: &SmtExpr,
    vars: &std::collections::HashMap<String, cvc5_rs::Term>,
) -> cvc5_rs::Term {
    use cvc5_rs::Kind;

    match expr {
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
            let l = lower_to_cvc5(tm, left, vars);
            let r = lower_to_cvc5(tm, right, vars);
            match op {
                ArithOp::Add => tm.mk_term(Kind::CVC5_KIND_ADD, &[l, r]),
                ArithOp::Sub => tm.mk_term(Kind::CVC5_KIND_SUB, &[l, r]),
                ArithOp::Mul => tm.mk_term(Kind::CVC5_KIND_MULT, &[l, r]),
                ArithOp::Div => tm.mk_term(Kind::CVC5_KIND_DIVISION, &[l, r]),
                ArithOp::Neg => tm.mk_term(Kind::CVC5_KIND_NEG, &[l]),
            }
        }
        SmtExpr::Cmp(op, left, right) => {
            let l = lower_to_cvc5(tm, left, vars);
            let r = lower_to_cvc5(tm, right, vars);
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
                .collect();
            match op {
                BoolOp::And => tm.mk_term(Kind::CVC5_KIND_AND, &terms),
                BoolOp::Or => tm.mk_term(Kind::CVC5_KIND_OR, &terms),
                BoolOp::Implies => tm.mk_term(Kind::CVC5_KIND_IMPLIES, &terms),
            }
        }
        SmtExpr::Not(inner) => {
            let t = lower_to_cvc5(tm, inner, vars);
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
            let body_term = lower_to_cvc5(tm, body, &extended_vars);
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
            let body_term = lower_to_cvc5(tm, body, &extended_vars);
            let bound_list = tm.mk_term(Kind::CVC5_KIND_VARIABLE_LIST, &bound_vars);
            tm.mk_term(Kind::CVC5_KIND_EXISTS, &[bound_list, body_term])
        }
        SmtExpr::Apply(name, args) => {
            let lowered_args: Vec<_> = args.iter().map(|a| lower_to_cvc5(tm, a, vars)).collect();
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
                other => panic!(
                    "unsupported function `{other}` in v0.1; only inlineable functions and supported transcendentals (exp, sqrt, sin, cos, abs, min, max) are allowed"
                ),
            }
        }
        SmtExpr::Ite(cond, then_expr, else_expr) => {
            let c = lower_to_cvc5(tm, cond, vars);
            let t = lower_to_cvc5(tm, then_expr, vars);
            let e = lower_to_cvc5(tm, else_expr, vars);
            tm.mk_term(Kind::CVC5_KIND_ITE, &[c, t, e])
        }
    }
}

/// Check if an SmtExpr contains transcendental function calls (exp, sin, cos, sqrt, abs).
#[cfg(feature = "smt")]
fn contains_transcendental(expr: &SmtExpr) -> bool {
    match expr {
        SmtExpr::Apply(name, args) => {
            matches!(name.as_str(), "exp" | "sin" | "cos" | "sqrt" | "abs")
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
}
