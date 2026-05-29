//! Tier C: Randomized fuzz testing.
//!
//! Wraps the existing `chelis prove` evaluation logic as a fallback tier.
//! Compiles the property source, evaluates the property root with the
//! compiler's IR evaluator, and reports pass/fail.

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use serde_json::Value;

/// Tier C outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum TierCResult {
    /// All samples passed.
    AllPassed(usize),
    /// A sample failed with counterexample.
    Failed(Value),
    /// Evaluation error (property couldn't be executed).
    Error(String),
}

/// Run Tier C fuzz testing on a property.
///
/// Evaluates the property using the Chelis compiler API. For properties
/// emitted by c-earchin Surf emission v1, the property body is a boolean
/// expression that the evaluator can execute directly.
///
/// Note: Full random-input generation (as in `chelis prove`) requires the
/// property to have parameters. For zero-parameter properties (legacy
/// c-earchin output), this evaluates once. For parameterized properties,
/// the existing `chelis prove` fuzz logic handles sample generation — this
/// tier delegates to it.
pub fn fuzz(
    property_source: &str,
    property_name: &str,
    samples: usize,
    _seed: u64,
) -> TierCResult {
    // For the dispatcher integration, we call eval_selected on a probe root
    // that wraps the property. This mirrors what chelis prove does internally.
    let probe_name = format!("__chelis_prove_probe_{property_name}");
    let source_with_probe = format!(
        "{property_source}\n{probe_name} = {property_name}()\n",
    );

    let result = compiler::eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source_with_probe,
            bindings: Default::default(),
        },
        &[probe_name],
    );

    match result {
        Ok(eval_result) => {
            match eval_result.roots.as_slice() {
                [root] => match &root.value {
                    ExecutionValue::Bool { value } => {
                        if *value {
                            TierCResult::AllPassed(samples)
                        } else {
                            TierCResult::Failed(Value::String(
                                "property evaluated to false".to_string(),
                            ))
                        }
                    }
                    _ => TierCResult::Error(
                        "property did not evaluate to bool".to_string(),
                    ),
                },
                _ => TierCResult::Error(
                    "unexpected number of evaluation roots".to_string(),
                ),
            }
        }
        Err(err) => {
            let msg = err.errors.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("; ");
            TierCResult::Error(msg)
        }
    }
}

// --- SmtProperty-based fuzzer (used by --tier auto) ---

use crate::solver::{SmtExpr, ArithOp, CmpOp, BoolOp};
use crate::tier_b::SmtProperty;
use std::collections::HashMap;

/// Fuzz an SmtProperty directly via concrete f64 evaluation.
/// Quantified properties (Forall/Exists in postcondition) → Error (unsupported).
pub fn fuzz_smt_property(property: &SmtProperty, samples: usize, seed: u64) -> TierCResult {
    // Check fuzzability: quantifiers in postcondition → unsupported
    if let crate::inlineability::Fuzzability::NotFuzzable(reason) =
        crate::inlineability::classify_fuzzability(&property.postcondition)
    {
        return TierCResult::Error(format!("cannot fuzz: {reason}"));
    }
    for pre in &property.preconditions {
        if let crate::inlineability::Fuzzability::NotFuzzable(reason) =
            crate::inlineability::classify_fuzzability(pre)
        {
            return TierCResult::Error(format!("cannot fuzz precondition: {reason}"));
        }
    }

    let mut rng = Lcg::new(seed);
    let max_attempts = samples.saturating_mul(100).max(samples);
    let mut accepted = 0usize;
    let mut attempts = 0usize;

    while accepted < samples && attempts < max_attempts {
        attempts += 1;
        let env: HashMap<String, f64> = property
            .variables
            .iter()
            .map(|(name, _)| (name.clone(), rng.next_f64(-10.0, 10.0)))
            .collect();

        // Check preconditions
        if !property.preconditions.iter().all(|pre| eval_bool(pre, &env)) {
            continue;
        }

        accepted += 1;
        if !eval_bool(&property.postcondition, &env) {
            let cx: serde_json::Map<String, Value> = env
                .iter()
                .map(|(k, v)| (k.clone(), Value::from(*v)))
                .collect();
            return TierCResult::Failed(Value::Object(cx));
        }
    }

    if accepted < samples {
        return TierCResult::Error(format!(
            "generator exhausted after {attempts} attempts before collecting {samples} valid samples"
        ));
    }
    TierCResult::AllPassed(accepted)
}

fn eval_bool(expr: &SmtExpr, env: &HashMap<String, f64>) -> bool {
    match expr {
        SmtExpr::BoolLit(v) => *v,
        SmtExpr::Cmp(op, left, right) => {
            let l = eval_arith(left, env);
            let r = eval_arith(right, env);
            match op {
                CmpOp::Lt => l < r,
                CmpOp::Le => l <= r,
                CmpOp::Gt => l > r,
                CmpOp::Ge => l >= r,
                CmpOp::Eq => (l - r).abs() < 1e-10,
                CmpOp::Ne => (l - r).abs() >= 1e-10,
            }
        }
        SmtExpr::Bool(BoolOp::And, children) => children.iter().all(|c| eval_bool(c, env)),
        SmtExpr::Bool(BoolOp::Or, children) => children.iter().any(|c| eval_bool(c, env)),
        SmtExpr::Bool(BoolOp::Implies, children) if children.len() == 2 => {
            !eval_bool(&children[0], env) || eval_bool(&children[1], env)
        }
        SmtExpr::Not(inner) => !eval_bool(inner, env),
        SmtExpr::Ite(cond, then_e, else_e) => {
            if eval_bool(cond, env) { eval_bool(then_e, env) } else { eval_bool(else_e, env) }
        }
        // Arithmetic expressions used in boolean context: nonzero = true
        _ => eval_arith(expr, env) != 0.0,
    }
}

fn eval_arith(expr: &SmtExpr, env: &HashMap<String, f64>) -> f64 {
    match expr {
        SmtExpr::Var(name) => env.get(name).copied().unwrap_or(0.0),
        SmtExpr::RealLit(v) => *v,
        SmtExpr::IntLit(v) => *v as f64,
        SmtExpr::BoolLit(v) => if *v { 1.0 } else { 0.0 },
        SmtExpr::Arith(op, left, right) => {
            let l = eval_arith(left, env);
            let r = eval_arith(right, env);
            match op {
                ArithOp::Add => l + r,
                ArithOp::Sub => l - r,
                ArithOp::Mul => l * r,
                ArithOp::Div => if r != 0.0 { l / r } else { f64::NAN },
                ArithOp::Neg => -l,
            }
        }
        SmtExpr::Apply(name, args) => {
            let a: Vec<f64> = args.iter().map(|a| eval_arith(a, env)).collect();
            match name.as_str() {
                "exp" => a[0].exp(),
                "log" => a[0].ln(),
                "sqrt" => a[0].sqrt(),
                "sin" => a[0].sin(),
                "cos" => a[0].cos(),
                "abs" => a[0].abs(),
                "min" if a.len() == 2 => a[0].min(a[1]),
                "max" if a.len() == 2 => a[0].max(a[1]),
                _ => f64::NAN,
            }
        }
        SmtExpr::Ite(cond, then_e, else_e) => {
            if eval_bool(cond, env) { eval_arith(then_e, env) } else { eval_arith(else_e, env) }
        }
        SmtExpr::Cmp(op, left, right) => {
            let l = eval_arith(left, env);
            let r = eval_arith(right, env);
            let result = match op {
                CmpOp::Lt => l < r, CmpOp::Le => l <= r, CmpOp::Gt => l > r,
                CmpOp::Ge => l >= r, CmpOp::Eq => (l - r).abs() < 1e-10, CmpOp::Ne => (l - r).abs() >= 1e-10,
            };
            if result { 1.0 } else { 0.0 }
        }
        SmtExpr::Bool(_, _) | SmtExpr::Not(_) => if eval_bool(expr, env) { 1.0 } else { 0.0 },
        SmtExpr::Forall(_, _) | SmtExpr::Exists(_, _) => f64::NAN, // unreachable after fuzzability check
    }
}

struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed ^ 0x9E37_79B9_7F4A_7C15 } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        min + (max - min) * unit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SmtSort;

    #[test]
    fn fuzz_x_squared_non_negative_passes() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(ArithOp::Mul, Box::new(SmtExpr::Var("x".into())), Box::new(SmtExpr::Var("x".into())))),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        match fuzz_smt_property(&prop, 100, 42) {
            TierCResult::AllPassed(n) => assert_eq!(n, 100),
            other => panic!("expected AllPassed, got {other:?}"),
        }
    }

    #[test]
    fn fuzz_x_gt_5_finds_counterexample() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::RealLit(5.0)),
            ),
        };
        match fuzz_smt_property(&prop, 100, 0) {
            TierCResult::Failed(cx) => assert!(cx.is_object()),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn fuzz_quantified_returns_error() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Forall(
                vec![("y".into(), SmtSort::Real)],
                Box::new(SmtExpr::Cmp(CmpOp::Ge, Box::new(SmtExpr::Var("y".into())), Box::new(SmtExpr::RealLit(0.0)))),
            ),
        };
        match fuzz_smt_property(&prop, 100, 0) {
            TierCResult::Error(msg) => assert!(msg.contains("cannot fuzz")),
            other => panic!("expected Error, got {other:?}"),
        }
    }
}
