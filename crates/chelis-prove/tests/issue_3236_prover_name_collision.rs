//! Regression oracle for chelis#3236: a solver symbol the prover mints must
//! never alias a name the goal already uses, and a goal that declares one name
//! twice must fail closed instead of collapsing two variables into one.
//!
//! The transformation and duplicate-declaration checks are solver-free; the
//! residual solves at the bottom need cvc5.

mod support;

use chelis_prove::discharge::{Goal, GoalShape};
use chelis_prove::solver::{BoolOp, CmpOp, SmtExpr, SmtSort};
use chelis_prove::tier_b::{AssumptionSatisfiability, SmtProperty, TierBResult};
use chelis_prove::transformation::Transformation;
use chelis_prove::transformations::abstract_subterm::AbstractSubterm;
use chelis_prove::{DischargeRegistry, check_assumptions_satisfiable, render_smtlib_problem};

fn var(name: &str) -> SmtExpr {
    SmtExpr::Var(name.to_string())
}

fn real(value: f64) -> SmtExpr {
    SmtExpr::RealLit(value)
}

fn cmp(op: CmpOp, left: SmtExpr, right: SmtExpr) -> SmtExpr {
    SmtExpr::Cmp(op, Box::new(left), Box::new(right))
}

fn and(children: Vec<SmtExpr>) -> SmtExpr {
    SmtExpr::Bool(BoolOp::And, children)
}

fn erf(argument: SmtExpr) -> SmtExpr {
    SmtExpr::Apply("erf".to_string(), vec![argument])
}

/// `forall(<binders>) where x >= 0, x <= 1: <postcondition>`, every binder
/// real-sorted.
fn unit_interval_property(binders: &[&str], postcondition: SmtExpr) -> SmtProperty {
    SmtProperty {
        variables: binders
            .iter()
            .map(|name| (name.to_string(), SmtSort::Real))
            .collect(),
        preconditions: vec![
            cmp(CmpOp::Ge, var("x"), real(0.0)),
            cmp(CmpOp::Le, var("x"), real(1.0)),
        ],
        postcondition,
    }
}

/// The issue's witness: `erf(x) <= 0.9 && __erf_abs_0 <= 0.9` over a user
/// binder spelled like the first abstraction variable. False at
/// `x = 0, __erf_abs_0 = 1`.
fn forged_witness() -> SmtProperty {
    unit_interval_property(
        &["x", "__erf_abs_0"],
        and(vec![
            cmp(CmpOp::Le, erf(var("x")), real(0.9)),
            cmp(CmpOp::Le, var("__erf_abs_0"), real(0.9)),
        ]),
    )
}

/// The issue's valid control: `erf(x) <= 0.9` on `[0, 1]` (erf(1) ~ 0.8427).
fn valid_control() -> SmtProperty {
    unit_interval_property(&["x"], cmp(CmpOp::Le, erf(var("x")), real(0.9)))
}

fn abstract_residual(property: &SmtProperty) -> SmtProperty {
    let goals = AbstractSubterm::new().apply(&Goal::smt(property.clone()));
    let [goal] = goals.as_slice() else {
        panic!("abstract-subterm yields one goal: {goals:?}");
    };
    let GoalShape::Smt(residual) = &goal.shape else {
        panic!("the residual is an SMT goal");
    };
    assert_ne!(
        residual, property,
        "the erf site must be abstracted (a certified envelope covers [0, 1])"
    );
    residual.clone()
}

/// The variables the residual declares beyond the original property's.
fn minted_variables(property: &SmtProperty, residual: &SmtProperty) -> Vec<String> {
    residual.variables[property.variables.len()..]
        .iter()
        .map(|(name, _)| name.clone())
        .collect()
}

fn mentions(expr: &SmtExpr, name: &str) -> bool {
    match expr {
        SmtExpr::Var(found) => found == name,
        SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => false,
        SmtExpr::Arith(_, left, right) | SmtExpr::Cmp(_, left, right) => {
            mentions(left, name) || mentions(right, name)
        }
        SmtExpr::Bool(_, children) | SmtExpr::Apply(_, children) => {
            children.iter().any(|child| mentions(child, name))
        }
        SmtExpr::Not(inner) | SmtExpr::Forall(_, inner) | SmtExpr::Exists(_, inner) => {
            mentions(inner, name)
        }
        SmtExpr::Ite(condition, then_branch, else_branch) => {
            mentions(condition, name) || mentions(then_branch, name) || mentions(else_branch, name)
        }
    }
}

#[test]
fn abstraction_variable_avoids_a_binder_spelled_like_it() {
    crate::support::isolate();
    let property = forged_witness();
    let residual = abstract_residual(&property);
    let minted = minted_variables(&property, &residual);
    assert_eq!(minted, vec!["__erf_abs_0_1".to_string()]);
    // The envelope bounds constrain the minted variable only; the user binder
    // stays unconstrained, so the false conjunct remains false.
    let added_preconditions = &residual.preconditions[property.preconditions.len()..];
    assert!(!added_preconditions.is_empty());
    for precondition in added_preconditions {
        assert!(mentions(precondition, "__erf_abs_0_1"), "{precondition:?}");
        assert!(!mentions(precondition, "__erf_abs_0"), "{precondition:?}");
    }
    assert!(mentions(&residual.postcondition, "__erf_abs_0"));
    assert!(mentions(&residual.postcondition, "__erf_abs_0_1"));
}

#[test]
fn abstraction_variable_avoids_free_and_quantifier_bound_names() {
    crate::support::isolate();
    // `__erf_abs_0` is a free (undeclared) name and `__erf_abs_0_1` a
    // quantifier binder; the minted variable must alias neither.
    let property = unit_interval_property(
        &["x"],
        and(vec![
            cmp(CmpOp::Le, erf(var("x")), var("__erf_abs_0")),
            SmtExpr::Forall(
                vec![("__erf_abs_0_1".to_string(), SmtSort::Real)],
                Box::new(cmp(CmpOp::Ge, var("__erf_abs_0_1"), var("__erf_abs_0_1"))),
            ),
        ]),
    );
    let residual = abstract_residual(&property);
    assert_eq!(
        minted_variables(&property, &residual),
        vec!["__erf_abs_0_2".to_string()]
    );
}

#[test]
fn abstraction_variable_keeps_its_spelling_without_a_collision() {
    crate::support::isolate();
    let property = valid_control();
    let residual = abstract_residual(&property);
    assert_eq!(
        minted_variables(&property, &residual),
        vec!["__erf_abs_0".to_string()]
    );
}

#[test]
fn duplicate_declarations_fail_closed_before_any_engine() {
    crate::support::isolate();
    let duplicated = SmtProperty {
        variables: vec![
            ("x".to_string(), SmtSort::Real),
            ("x".to_string(), SmtSort::Real),
        ],
        preconditions: vec![],
        postcondition: cmp(CmpOp::Ge, var("x"), var("x")),
    };
    let reason = match chelis_prove::solve_property(&duplicated, 5_000) {
        TierBResult::Error(reason) => reason,
        other => panic!("a duplicated declaration must fail closed, got {other:?}"),
    };
    assert!(reason.contains("`x`"), "{reason}");
    assert!(reason.contains("more than once"), "{reason}");

    let discharge =
        DischargeRegistry::with_builtin_engines().dispatch(&Goal::smt(duplicated.clone()), 5_000);
    assert!(
        matches!(discharge.result(), TierBResult::Error(_)),
        "{discharge:?}"
    );
    assert!(matches!(
        check_assumptions_satisfiable(&duplicated, 5_000),
        AssumptionSatisfiability::Error(_)
    ));
    assert!(render_smtlib_problem(&duplicated).is_err());

    // Control: the same goal with distinct names is not rejected for its names.
    let distinct = SmtProperty {
        variables: vec![
            ("x".to_string(), SmtSort::Real),
            ("y".to_string(), SmtSort::Real),
        ],
        ..duplicated
    };
    assert!(render_smtlib_problem(&distinct).is_ok());
}

#[cfg(feature = "smt")]
#[test]
fn forged_residual_is_refuted_and_the_valid_control_still_proves() {
    crate::support::isolate();
    let forged = abstract_residual(&forged_witness());
    assert!(
        matches!(
            chelis_prove::solve_property(&forged, 5_000),
            TierBResult::Disproved(_)
        ),
        "the forged residual must not be proved"
    );
    let control = abstract_residual(&valid_control());
    assert_eq!(
        chelis_prove::solve_property(&control, 5_000),
        TierBResult::Proved
    );
}

#[cfg(feature = "smt")]
#[test]
fn cvc5_rejects_a_repeated_quantifier_binder() {
    crate::support::isolate();
    let property = SmtProperty {
        variables: vec![],
        preconditions: vec![],
        postcondition: SmtExpr::Forall(
            vec![
                ("k".to_string(), SmtSort::Real),
                ("k".to_string(), SmtSort::Int),
            ],
            Box::new(cmp(CmpOp::Ge, var("k"), var("k"))),
        ),
    };
    let reason = match chelis_prove::solve_property(&property, 5_000) {
        TierBResult::Error(reason) => reason,
        other => panic!("a repeated binder must fail closed, got {other:?}"),
    };
    assert!(reason.contains("more than once"), "{reason}");
}
