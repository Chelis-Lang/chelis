//! Inlineability classification for SMT expressions.

use crate::solver::SmtExpr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inlineability {
    Inlineable,
    NotInlineable(String),
}

/// Whether an expression can be evaluated by the Tier C fuzzer.
/// Quantifiers cannot be randomly sampled; they require Tier B exclusively.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fuzzability {
    Fuzzable,
    NotFuzzable(String),
}

/// The whitelisted intrinsics admitted in an inlineable SMT expression.
/// This consumes `chelis_pred::INTRINSIC_WHITELIST` -- the RFC's single
/// source of truth (D-PRED / RFC L5) -- rather than keeping a second copy
/// that could drift.
use chelis_pred::INTRINSIC_WHITELIST as SUPPORTED_FUNCTIONS;

pub fn classify_inlineability(expr: &SmtExpr) -> Inlineability {
    match expr {
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {
            Inlineability::Inlineable
        }
        SmtExpr::Arith(_, left, right) => {
            combine(classify_inlineability(left), classify_inlineability(right))
        }
        SmtExpr::Cmp(_, left, right) => {
            combine(classify_inlineability(left), classify_inlineability(right))
        }
        SmtExpr::Bool(_, children) => children
            .iter()
            .map(classify_inlineability)
            .fold(Inlineability::Inlineable, combine),
        SmtExpr::Not(inner) => classify_inlineability(inner),
        SmtExpr::Forall(_, body) => classify_inlineability(body),
        SmtExpr::Exists(_, body) => classify_inlineability(body),
        SmtExpr::Ite(cond, then_e, else_e) => combine(
            combine(classify_inlineability(cond), classify_inlineability(then_e)),
            classify_inlineability(else_e),
        ),
        SmtExpr::Apply(name, args) => {
            if SUPPORTED_FUNCTIONS.contains(&name.as_str()) {
                args.iter()
                    .map(classify_inlineability)
                    .fold(Inlineability::Inlineable, combine)
            } else {
                Inlineability::NotInlineable(format!("unsupported function: {name}"))
            }
        }
    }
}

fn combine(a: Inlineability, b: Inlineability) -> Inlineability {
    match (a, b) {
        (Inlineability::NotInlineable(reason), _) | (_, Inlineability::NotInlineable(reason)) => {
            Inlineability::NotInlineable(reason)
        }
        _ => Inlineability::Inlineable,
    }
}

/// Classify whether an expression can be evaluated by the Tier C fuzzer.
/// Quantifiers (Forall/Exists) cannot be randomly sampled.
pub fn classify_fuzzability(expr: &SmtExpr) -> Fuzzability {
    match expr {
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {
            Fuzzability::Fuzzable
        }
        SmtExpr::Arith(_, left, right) => {
            combine_fuzz(classify_fuzzability(left), classify_fuzzability(right))
        }
        SmtExpr::Cmp(_, left, right) => {
            combine_fuzz(classify_fuzzability(left), classify_fuzzability(right))
        }
        SmtExpr::Bool(_, children) => children
            .iter()
            .map(classify_fuzzability)
            .fold(Fuzzability::Fuzzable, combine_fuzz),
        SmtExpr::Not(inner) => classify_fuzzability(inner),
        SmtExpr::Forall(_, _) => {
            Fuzzability::NotFuzzable("quantified properties cannot be randomly sampled".into())
        }
        SmtExpr::Exists(_, _) => {
            Fuzzability::NotFuzzable("quantified properties cannot be randomly sampled".into())
        }
        SmtExpr::Ite(cond, then_e, else_e) => combine_fuzz(
            combine_fuzz(classify_fuzzability(cond), classify_fuzzability(then_e)),
            classify_fuzzability(else_e),
        ),
        SmtExpr::Apply(_, args) => args
            .iter()
            .map(classify_fuzzability)
            .fold(Fuzzability::Fuzzable, combine_fuzz),
    }
}

fn combine_fuzz(a: Fuzzability, b: Fuzzability) -> Fuzzability {
    match (a, b) {
        (Fuzzability::NotFuzzable(reason), _) | (_, Fuzzability::NotFuzzable(reason)) => {
            Fuzzability::NotFuzzable(reason)
        }
        _ => Fuzzability::Fuzzable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{ArithOp, CmpOp};

    #[test]
    fn cr11_supported_functions_is_the_canonical_pred_whitelist() {
        // RFC L5 / CR-11: inlineability consumes chelis_pred's single
        // source of truth, so the two cannot drift.
        assert_eq!(SUPPORTED_FUNCTIONS, chelis_pred::INTRINSIC_WHITELIST);
        // Every whitelisted intrinsic is inlineable.
        for name in SUPPORTED_FUNCTIONS {
            let e = SmtExpr::Apply((*name).into(), vec![SmtExpr::Var("x".into())]);
            assert!(
                matches!(classify_inlineability(&e), Inlineability::Inlineable),
                "intrinsic `{name}` must be inlineable"
            );
        }
    }

    #[test]
    fn polynomial_is_inlineable() {
        // x*x + 2*x + 1
        let expr = SmtExpr::Arith(
            ArithOp::Add,
            Box::new(SmtExpr::Arith(
                ArithOp::Mul,
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Var("x".into())),
            )),
            Box::new(SmtExpr::Arith(
                ArithOp::Add,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::RealLit(2.0)),
                    Box::new(SmtExpr::Var("x".into())),
                )),
                Box::new(SmtExpr::RealLit(1.0)),
            )),
        );
        assert_eq!(classify_inlineability(&expr), Inlineability::Inlineable);
    }

    #[test]
    fn supported_transcendental_is_inlineable() {
        let expr = SmtExpr::Cmp(
            CmpOp::Ge,
            Box::new(SmtExpr::Apply("exp".into(), vec![SmtExpr::Var("x".into())])),
            Box::new(SmtExpr::RealLit(0.0)),
        );
        assert_eq!(classify_inlineability(&expr), Inlineability::Inlineable);
    }

    #[test]
    fn unknown_function_not_inlineable() {
        let expr = SmtExpr::Apply("custom_fn".into(), vec![SmtExpr::Var("x".into())]);
        assert_eq!(
            classify_inlineability(&expr),
            Inlineability::NotInlineable("unsupported function: custom_fn".into())
        );
    }
}
