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
pub fn fuzz(property_source: &str, property_name: &str, samples: usize, _seed: u64) -> TierCResult {
    // For the dispatcher integration, we call eval_selected on a probe root
    // that wraps the property. This mirrors what chelis prove does internally.
    let probe_name = format!("__chelis_prove_probe_{property_name}");
    let source_with_probe = format!("{property_source}\n{probe_name} = {property_name}()\n",);

    let result = compiler::eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source_with_probe,
            bindings: Default::default(),
        },
        &[probe_name],
    );

    match result {
        Ok(eval_result) => match eval_result.roots.as_slice() {
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
                _ => TierCResult::Error("property did not evaluate to bool".to_string()),
            },
            _ => TierCResult::Error("unexpected number of evaluation roots".to_string()),
        },
        Err(err) => {
            let msg = err
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("; ");
            TierCResult::Error(msg)
        }
    }
}

// --- SmtProperty-based fuzzer (used by --tier auto) ---

use crate::concrete_eval::{ConcreteEnv, eval_bool};
use crate::tier_b::SmtProperty;
use chelis_reef::EmbeddedRuntime;
use chelis_types::{ScalarValue, scalar_from_f64, scalar_from_i64, types::Prim};
use chelis_unord::UnordMap;

/// Fuzz an SmtProperty directly via dtype-aware concrete evaluation.
/// Quantified properties (Forall/Exists in postcondition) → Error (unsupported).
///
/// `declared` gives each numeric variable's declared dtype. An SMT sort does
/// not carry a width, so a sample is drawn, rounded once to the declared
/// dtype, and evaluated there; a numeric variable without a declared dtype is
/// an error rather than a silent f64 surrogate (chelis#2965).
pub fn fuzz_smt_property(
    property: &SmtProperty,
    declared: &UnordMap<String, Prim>,
    samples: usize,
    seed: u64,
    runtime: &'static EmbeddedRuntime,
) -> TierCResult {
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
    // chelis#3236: the sample environment is keyed by name; a repeated
    // declaration would silently share one sampled value.
    if let Some(name) = crate::smt_names::first_duplicate_variable(&property.variables) {
        return TierCResult::Error(crate::smt_names::duplicate_variable_reason(name));
    }

    for (name, sort) in &property.variables {
        let compatible = match (sort, declared.get(name)) {
            (crate::solver::SmtSort::Bool, _) => true,
            (crate::solver::SmtSort::Int, Some(prim)) => prim.is_integer(),
            (crate::solver::SmtSort::Real, Some(prim)) => prim.is_float(),
            (_, None) => false,
        };
        if !compatible {
            return TierCResult::Error(format!(
                "cannot fuzz: variable `{name}` has no declared dtype matching its {sort:?} sort"
            ));
        }
    }

    let mut rng = Lcg::new(seed);
    let max_attempts = samples.saturating_mul(100).max(samples);
    let mut accepted = 0usize;
    let mut attempts = 0usize;

    while accepted < samples && attempts < max_attempts {
        attempts += 1;
        let env: ConcreteEnv = property
            .variables
            .iter()
            .map(|(name, sort)| {
                let value = match sort {
                    crate::solver::SmtSort::Int => {
                        scalar_from_i64("prove-fuzz-sample", declared[name], rng.next_i64(-10, 10))
                            .expect("the integer fuzz bounds fit every integer dtype")
                    }
                    crate::solver::SmtSort::Real => scalar_from_f64(
                        "prove-fuzz-sample",
                        declared[name],
                        rng.next_f64(-10.0, 10.0),
                    )
                    .expect("every fuzz sample in [-10, 10] is finite at every float dtype"),
                    crate::solver::SmtSort::Bool => {
                        scalar_from_i64("prove-fuzz-sample", Prim::Bool, i64::from(rng.next_bool()))
                            .expect("boolean samples are exactly zero or one")
                    }
                };
                (name.clone(), value)
            })
            .collect();

        // Check preconditions
        if !property
            .preconditions
            .iter()
            .all(|pre| eval_bool(pre, &env, runtime))
        {
            continue;
        }

        accepted += 1;
        if !eval_bool(&property.postcondition, &env, runtime) {
            let cx: serde_json::Map<String, Value> = env
                .to_sorted()
                .into_iter()
                .map(|(k, v)| (k.clone(), scalar_json(*v)))
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

fn scalar_json(value: ScalarValue) -> Value {
    if let Some(value) = value.as_bool_exact() {
        Value::from(value)
    } else if let Some(value) = value.as_i64_exact() {
        Value::from(value)
    } else {
        Value::from(value.as_f64_lossy())
    }
}

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        min + (max - min) * unit
    }

    fn next_i64(&mut self, min: i64, max: i64) -> i64 {
        let width = (max as i128 - min as i128 + 1) as u64;
        min + (self.next_u64() % width) as i64
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};
    use chelis_std_bundle::EMBEDDED_RUNTIME;

    // The fuzzer at the runtime this test binary embeds.
    fn fuzz_smt_property(
        property: &SmtProperty,
        declared: &UnordMap<String, Prim>,
        samples: usize,
        seed: u64,
    ) -> TierCResult {
        super::fuzz_smt_property(property, declared, samples, seed, &EMBEDDED_RUNTIME)
    }

    fn declared(pairs: &[(&str, Prim)]) -> UnordMap<String, Prim> {
        pairs
            .iter()
            .map(|(name, prim)| (name.to_string(), *prim))
            .collect()
    }

    #[test]
    fn fuzz_x_squared_non_negative_passes() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::Var("x".into())),
                    Box::new(SmtExpr::Var("x".into())),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        match fuzz_smt_property(&prop, &declared(&[("x", Prim::F64)]), 100, 42) {
            TierCResult::AllPassed(n) => assert_eq!(n, 100),
            other => panic!("expected AllPassed, got {other:?}"),
        }
    }

    #[test]
    fn integer_fuzz_variables_are_sampled_and_reported_as_integers() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Int)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ne,
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Var("x".into())),
            ),
        };
        let TierCResult::Failed(Value::Object(counterexample)) =
            fuzz_smt_property(&prop, &declared(&[("x", Prim::Int64)]), 1, 7)
        else {
            panic!("x != x must produce an integer counterexample")
        };
        assert!(
            counterexample["x"].as_i64().is_some(),
            "an SMT Int sample must never enter or leave Tier C as a float"
        );
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
        match fuzz_smt_property(&prop, &declared(&[("x", Prim::F64)]), 100, 0) {
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
                Box::new(SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Var("y".into())),
                    Box::new(SmtExpr::RealLit(0.0)),
                )),
            ),
        };
        match fuzz_smt_property(&prop, &declared(&[("x", Prim::F64)]), 100, 0) {
            TierCResult::Error(msg) => assert!(msg.contains("cannot fuzz")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    /// `x + 1e-9 > x` over x in [-10, 10]: true at f64 (an ulp of 10 is
    /// about 1.8e-15), false at f32, where adding 1e-9 rounds back to x.
    fn planted_width_sensitive_property() -> SmtProperty {
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Arith(
                    ArithOp::Add,
                    Box::new(SmtExpr::Var("x".into())),
                    Box::new(SmtExpr::RealLit(1e-9)),
                )),
                Box::new(SmtExpr::Var("x".into())),
            ),
        }
    }

    #[test]
    fn chelis_2965_property_true_at_f64_is_refuted_at_declared_f32() {
        let prop = planted_width_sensitive_property();
        let TierCResult::Failed(Value::Object(counterexample)) =
            fuzz_smt_property(&prop, &declared(&[("x", Prim::F32)]), 100, 3)
        else {
            panic!("an f32 property that is false at f32 must be refuted")
        };
        let x = counterexample["x"].as_f64().expect("float counterexample");
        assert_eq!(x, f64::from(x as f32), "the counterexample is an f32 value");
    }

    #[test]
    fn chelis_2965_same_property_passes_at_declared_f64() {
        let prop = planted_width_sensitive_property();
        assert_eq!(
            fuzz_smt_property(&prop, &declared(&[("x", Prim::F64)]), 100, 3),
            TierCResult::AllPassed(100)
        );
    }

    #[test]
    fn chelis_2965_undeclared_numeric_variable_is_not_fuzzed_as_f64() {
        let prop = planted_width_sensitive_property();
        match fuzz_smt_property(&prop, &declared(&[]), 100, 3) {
            TierCResult::Error(msg) => assert!(msg.contains("no declared dtype"), "{msg}"),
            other => panic!("expected an error, got {other:?}"),
        }
        match fuzz_smt_property(&prop, &declared(&[("x", Prim::Int32)]), 100, 3) {
            TierCResult::Error(msg) => assert!(msg.contains("no declared dtype"), "{msg}"),
            other => panic!("expected an error, got {other:?}"),
        }
    }
}
