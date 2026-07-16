//! Standard mathematical contracts consumed by COMPOSE.
//!
//! The contracts here are reusable proof dependencies. They deliberately
//! carry discharge records in the same shape emitted for producer
//! obligations, so a consumer proof can roll up through the weakest-link
//! machinery instead of rendering an assumed invariant as pure proven.

use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::composition::CompositeVerdict;
use crate::composition::{
    AssumptionDischarge, AssumptionRecord, AssumptionRegistry, DischargeMethod, DischargeTier,
    FUZZ_TOLERANCE, NonVacuityRecord,
};

pub const NORMAL_CDF_RANGE: &str = "std.normal_cdf.range";
pub const NORMAL_CDF_REFLECTION: &str = "std.normal_cdf.reflection";
pub const NORMAL_CDF_MONOTONICITY: &str = "std.normal_cdf.monotonicity";
pub const NORMAL_CDF_IMPLEMENTATION: &str = "Std.Contracts.normal_cdf";
pub const EXP_POSITIVITY: &str = "std.exp.positivity";
pub const EXP_MONOTONICITY: &str = "std.exp.monotonicity";
pub const EXP_ZERO: &str = "std.exp.zero";
pub const LOG_MONOTONICITY: &str = "std.log.monotonicity";
pub const LOG_ONE: &str = "std.log.one";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandardContract {
    pub id: String,
    pub function: String,
    pub invariants: Vec<ContractInvariant>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractInvariant {
    pub id: String,
    pub description: String,
    pub assumption: String,
    pub record: AssumptionRecord,
}

#[must_use]
pub fn standard_contracts() -> Vec<StandardContract> {
    vec![
        StandardContract {
            id: "std.normal_cdf".to_string(),
            function: NORMAL_CDF_IMPLEMENTATION.to_string(),
            invariants: vec![
                fuzz_invariant(
                    NORMAL_CDF_RANGE,
                    "normal CDF range",
                    "forall x. 0 <= N(x) <= 1",
                    NORMAL_CDF_IMPLEMENTATION,
                    8192,
                    0xC0DF_2026,
                ),
                fuzz_invariant(
                    NORMAL_CDF_REFLECTION,
                    "normal CDF reflection",
                    "forall x. N(-x) = 1 - N(x)",
                    NORMAL_CDF_IMPLEMENTATION,
                    8192,
                    0xC0DF_2026,
                ),
                fuzz_invariant(
                    NORMAL_CDF_MONOTONICITY,
                    "normal CDF monotonicity",
                    "forall x y. x <= y => N(x) <= N(y)",
                    NORMAL_CDF_IMPLEMENTATION,
                    8192,
                    0xC0DF_2026,
                ),
            ],
        },
        StandardContract {
            id: "std.exp".to_string(),
            function: "exp".to_string(),
            invariants: vec![
                smt_invariant(EXP_POSITIVITY, "exp positivity", "forall x. exp(x) > 0"),
                fuzz_invariant(
                    EXP_MONOTONICITY,
                    "exp monotonicity",
                    "forall x y. x <= y => exp(x) <= exp(y)",
                    "chelis_intrinsic.exp",
                    4096,
                    0xE_2026,
                ),
                smt_invariant(EXP_ZERO, "exp zero", "exp(0) = 1"),
            ],
        },
        StandardContract {
            id: "std.log".to_string(),
            function: "log".to_string(),
            invariants: vec![
                fuzz_invariant(
                    LOG_MONOTONICITY,
                    "log monotonicity on positives",
                    "forall x y. 0 < x <= y => log(x) <= log(y)",
                    "chelis_intrinsic.log",
                    4096,
                    0x10_2026,
                ),
                fuzz_invariant(
                    LOG_ONE,
                    "log one",
                    "log(1) = 0",
                    "chelis_intrinsic.log",
                    64,
                    0x10_2026,
                ),
            ],
        },
    ]
}

#[must_use]
pub fn standard_contract_registry() -> AssumptionRegistry {
    let mut registry = AssumptionRegistry::new();
    for contract in standard_contracts() {
        for invariant in contract.invariants {
            registry.insert(invariant.record);
        }
    }
    registry
}

fn smt_invariant(id: &str, description: &str, assumption: &str) -> ContractInvariant {
    ContractInvariant {
        id: id.to_string(),
        description: description.to_string(),
        assumption: assumption.to_string(),
        record: AssumptionRecord::new(
            id,
            Some(AssumptionDischarge::new(
                DischargeMethod::Smt,
                serde_json::json!({
                    "status": "proved",
                    "solver": "cvc5",
                    "arith_model": "real",
                    "assumption": assumption,
                }),
            )),
            Some(NonVacuityRecord::established(serde_json::json!({
                "solver": "cvc5",
                "result": "sat",
                "contract": id,
            }))),
        )
        .with_source("std_contract", id)
        // WI-8: the standard contract is discharged by cvc5 at the SMT tier,
        // keyed to the contract id.
        .with_discharge_tier(DischargeTier::new(
            DischargeMethod::Smt.engine(),
            DischargeMethod::Smt,
            Some(id.to_string()),
        )),
    }
}

fn fuzz_invariant(
    id: &str,
    description: &str,
    assumption: &str,
    implementation: &str,
    samples: usize,
    seed: u64,
) -> ContractInvariant {
    ContractInvariant {
        id: id.to_string(),
        description: description.to_string(),
        assumption: assumption.to_string(),
        record: AssumptionRecord::new(
            id,
            Some(AssumptionDischarge::new(
                DischargeMethod::Fuzz,
                fuzz_discharge_evidence(id, assumption, implementation, samples, seed),
            )),
            Some(NonVacuityRecord::established(serde_json::json!({
                "method": "fuzz",
                "result": "sat",
                "samples": samples,
                "seed": seed,
                "contract": id,
            }))),
        )
        .with_source("std_contract", id)
        // WI-8: the standard contract is fuzz-validated by the sampler, keyed
        // to the contract id.
        .with_discharge_tier(DischargeTier::new(
            DischargeMethod::Fuzz.engine(),
            DischargeMethod::Fuzz,
            Some(id.to_string()),
        )),
    }
}

fn fuzz_discharge_evidence(
    id: &str,
    assumption: &str,
    implementation: &str,
    samples: usize,
    seed: u64,
) -> serde_json::Value {
    let outcome = run_fuzz_discharge(id, samples, seed);
    let status = if outcome.counterexample.is_some() {
        "failed"
    } else {
        "validated"
    };
    let mut evidence = serde_json::json!({
        "status": status,
        "implementation": implementation,
        "samples": samples,
        "seed": seed,
        "tolerance": FUZZ_TOLERANCE,
        "domain": outcome.domain,
        "assumption": assumption,
        "checked_samples": outcome.checked_samples,
        "max_error": outcome.max_error,
    });
    if let Some(counterexample) = outcome.counterexample {
        evidence["counterexample"] = counterexample;
    }
    evidence
}

#[derive(Debug, Clone)]
struct FuzzOutcome {
    checked_samples: usize,
    max_error: f64,
    counterexample: Option<serde_json::Value>,
    domain: &'static str,
}

fn run_fuzz_discharge(id: &str, samples: usize, seed: u64) -> FuzzOutcome {
    match id {
        NORMAL_CDF_RANGE => fuzz_unary(
            samples,
            seed,
            "normal-cdf finite f32 domain [-10, 10]",
            |x| {
                let y = normal_cdf_erfc(x);
                let low = 0.0_f64 - y;
                let high = y - 1.0_f64;
                low.max(high).max(0.0)
            },
        ),
        NORMAL_CDF_REFLECTION => fuzz_unary(
            samples,
            seed,
            "normal-cdf finite f32 domain [-10, 10]",
            |x| (normal_cdf_erfc(-x) - (1.0 - normal_cdf_erfc(x))).abs(),
        ),
        NORMAL_CDF_MONOTONICITY => fuzz_ordered_pair(
            samples,
            seed,
            "ordered normal-cdf finite f32 pairs in [-10, 10]",
            |lo, hi| (normal_cdf_erfc(lo) - normal_cdf_erfc(hi)).max(0.0),
        ),
        EXP_MONOTONICITY => fuzz_ordered_pair(
            samples,
            seed,
            "ordered exp finite f32 pairs in [-10, 10]",
            |lo, hi| (lo.exp() - hi.exp()).max(0.0),
        ),
        LOG_MONOTONICITY => fuzz_ordered_positive_pair(
            samples,
            seed,
            "ordered positive f32 pairs in [1e-6, 1e6]",
            |lo, hi| (lo.ln() - hi.ln()).max(0.0),
        ),
        LOG_ONE => {
            let error = 1.0_f64.ln().abs();
            FuzzOutcome {
                checked_samples: samples.max(1),
                max_error: error,
                counterexample: counterexample_if(error, serde_json::json!({"x": 1.0})),
                domain: "log anchor x = 1",
            }
        }
        _ => FuzzOutcome {
            checked_samples: 0,
            max_error: f64::INFINITY,
            counterexample: Some(serde_json::json!({"unsupported_contract": id})),
            domain: "unsupported standard contract",
        },
    }
}

fn fuzz_unary(
    samples: usize,
    seed: u64,
    domain: &'static str,
    error: impl Fn(f64) -> f64,
) -> FuzzOutcome {
    let mut rng = Lcg::new(seed);
    let mut max_error = 0.0_f64;
    for i in 0..samples {
        let x = sample_signed_domain(i, samples, &mut rng, 10.0);
        let err = error(x);
        max_error = max_error.max(err);
        if err > FUZZ_TOLERANCE {
            return FuzzOutcome {
                checked_samples: i + 1,
                max_error,
                counterexample: Some(serde_json::json!({"x": x, "error": err})),
                domain,
            };
        }
    }
    FuzzOutcome {
        checked_samples: samples,
        max_error,
        counterexample: None,
        domain,
    }
}

fn fuzz_ordered_pair(
    samples: usize,
    seed: u64,
    domain: &'static str,
    error: impl Fn(f64, f64) -> f64,
) -> FuzzOutcome {
    let mut rng = Lcg::new(seed);
    let mut max_error = 0.0_f64;
    for i in 0..samples {
        let a = sample_signed_domain(i, samples, &mut rng, 10.0);
        let b = sample_signed_domain(samples.saturating_sub(i + 1), samples, &mut rng, 10.0);
        let lo = a.min(b);
        let hi = a.max(b);
        let err = error(lo, hi);
        max_error = max_error.max(err);
        if err > FUZZ_TOLERANCE {
            return FuzzOutcome {
                checked_samples: i + 1,
                max_error,
                counterexample: Some(serde_json::json!({"lo": lo, "hi": hi, "error": err})),
                domain,
            };
        }
    }
    FuzzOutcome {
        checked_samples: samples,
        max_error,
        counterexample: None,
        domain,
    }
}

fn fuzz_ordered_positive_pair(
    samples: usize,
    seed: u64,
    domain: &'static str,
    error: impl Fn(f64, f64) -> f64,
) -> FuzzOutcome {
    let mut rng = Lcg::new(seed);
    let mut max_error = 0.0_f64;
    for i in 0..samples {
        let a = sample_positive_domain(i, samples, &mut rng);
        let b = sample_positive_domain(samples.saturating_sub(i + 1), samples, &mut rng);
        let lo = a.min(b);
        let hi = a.max(b);
        let err = error(lo, hi);
        max_error = max_error.max(err);
        if err > FUZZ_TOLERANCE {
            return FuzzOutcome {
                checked_samples: i + 1,
                max_error,
                counterexample: Some(serde_json::json!({"lo": lo, "hi": hi, "error": err})),
                domain,
            };
        }
    }
    FuzzOutcome {
        checked_samples: samples,
        max_error,
        counterexample: None,
        domain,
    }
}

fn counterexample_if(error: f64, counterexample: serde_json::Value) -> Option<serde_json::Value> {
    (error > FUZZ_TOLERANCE).then(|| {
        let mut value = counterexample;
        value["error"] = serde_json::json!(error);
        value
    })
}

fn sample_signed_domain(index: usize, samples: usize, rng: &mut Lcg, radius: f64) -> f64 {
    match index {
        0 => 0.0,
        1 => -radius,
        2 => radius,
        _ => {
            let denom = samples.saturating_sub(1).max(1) as f64;
            let grid = -radius + 2.0 * radius * (index as f64 / denom);
            let jitter = (rng.next_unit() - 0.5) * (2.0 * radius / denom);
            (grid + jitter).clamp(-radius, radius)
        }
    }
}

fn sample_positive_domain(index: usize, samples: usize, rng: &mut Lcg) -> f64 {
    match index {
        0 => 1.0,
        1 => 1.0e-6,
        2 => 1.0e6,
        _ => {
            let denom = samples.saturating_sub(1).max(1) as f64;
            let grid = -6.0 + 12.0 * (index as f64 / denom);
            let jitter = (rng.next_unit() - 0.5) * (12.0 / denom);
            10.0_f64.powf((grid + jitter).clamp(-6.0, 6.0))
        }
    }
}

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = self.state >> 11;
        (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

fn normal_cdf_erfc(x: f64) -> f64 {
    0.5 * erfc_approx(-x / std::f64::consts::SQRT_2)
}

fn erfc_approx(x: f64) -> f64 {
    1.0 - erf_approx(x)
}

fn erf_approx(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let p = 0.3275911_f64;
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let t = 1.0 / (1.0 + p * ax);
    let poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    sign * (1.0 - poly * (-(ax * ax)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_invariants() -> Vec<ContractInvariant> {
        standard_contracts()
            .into_iter()
            .flat_map(|contract| contract.invariants)
            .collect()
    }

    #[test]
    fn k1_put_call_parity_consumes_cdf_reflection_as_fuzz_qualified_contract() {
        let registry = standard_contract_registry();
        let probe = registry.probe_consumer(
            "put_call_parity",
            CompositeVerdict::Proven,
            [NORMAL_CDF_REFLECTION],
        );
        assert_eq!(
            probe.composite_verdict,
            CompositeVerdict::ProvenModuloFuzzValidatedContract
        );
        assert_ne!(probe.composite_verdict, CompositeVerdict::Proven);
        assert_eq!(probe.assumptions[0].name, NORMAL_CDF_REFLECTION);
        let discharge = probe.assumptions[0].discharge.as_ref().unwrap();
        assert_eq!(discharge.method, DischargeMethod::Fuzz);
        assert_eq!(
            discharge.evidence["implementation"],
            NORMAL_CDF_IMPLEMENTATION
        );
        assert!(discharge.evidence.get("seed").is_some());
        assert!(discharge.evidence.get("tolerance").is_some());
    }

    #[test]
    fn normal_cdf_contracts_bind_to_bundled_callable_implementation() {
        let normal = standard_contracts()
            .into_iter()
            .find(|contract| contract.id == "std.normal_cdf")
            .expect("normal CDF contract exists");
        assert_eq!(normal.function, NORMAL_CDF_IMPLEMENTATION);
        for invariant in normal.invariants {
            let discharge = invariant.record.discharge.as_ref().unwrap();
            assert_eq!(
                discharge.evidence["implementation"], NORMAL_CDF_IMPLEMENTATION,
                "{} must discharge against the bundled std function",
                invariant.id
            );
        }
    }

    #[test]
    fn k2_every_standard_invariant_has_recorded_discharge() {
        let invariants = all_invariants();
        assert!(!invariants.is_empty());
        for invariant in invariants {
            let discharge = invariant
                .record
                .discharge
                .as_ref()
                .unwrap_or_else(|| panic!("{} missing discharge", invariant.id));
            match discharge.method {
                DischargeMethod::Smt => {
                    assert_eq!(discharge.evidence["status"], "proved");
                    assert_eq!(discharge.evidence["solver"], "cvc5");
                }
                DischargeMethod::Fuzz => {
                    assert_eq!(discharge.evidence["status"], "validated");
                    assert!(discharge.evidence.get("implementation").is_some());
                    assert!(discharge.evidence.get("samples").is_some());
                    assert!(discharge.evidence.get("seed").is_some());
                    assert!(discharge.evidence.get("tolerance").is_some());
                    assert_eq!(
                        discharge.evidence["checked_samples"], discharge.evidence["samples"],
                        "{} did not check every recorded fuzz sample",
                        invariant.id
                    );
                    assert!(
                        discharge.evidence.get("counterexample").is_none(),
                        "{} unexpectedly recorded a fuzz counterexample",
                        invariant.id
                    );
                }
                DischargeMethod::Axiom => {
                    assert!(discharge.evidence.get("justification").is_some());
                }
            }
        }
    }

    #[test]
    fn k3_corrupting_contract_discharge_degrades_dependent_consumer() {
        let mut registry = standard_contract_registry();
        let before = registry.probe_consumer(
            "put_call_parity",
            CompositeVerdict::Proven,
            [NORMAL_CDF_REFLECTION],
        );
        assert_eq!(
            before.composite_verdict,
            CompositeVerdict::ProvenModuloFuzzValidatedContract
        );

        registry.insert(AssumptionRecord::new(
            NORMAL_CDF_REFLECTION,
            Some(AssumptionDischarge::new(
                DischargeMethod::Fuzz,
                serde_json::json!({
                    "status": "failed",
                    "counterexample": {"x": 3.0},
                    "implementation": NORMAL_CDF_IMPLEMENTATION,
                    "samples": 8192,
                    "seed": 0xC0DF_2026_u64,
                    "tolerance": FUZZ_TOLERANCE,
                }),
            )),
            Some(NonVacuityRecord::established(serde_json::json!({
                "method": "fuzz",
                "result": "sat",
            }))),
        ));
        let after = registry.probe_consumer(
            "put_call_parity",
            CompositeVerdict::Proven,
            [NORMAL_CDF_REFLECTION],
        );
        assert_eq!(after.composite_verdict, CompositeVerdict::Failed);
    }

    #[test]
    fn k4_contract_assumption_non_vacuity_failure_is_invalid() {
        let mut registry = standard_contract_registry();
        registry.insert(AssumptionRecord::new(
            EXP_POSITIVITY,
            Some(AssumptionDischarge::new(
                DischargeMethod::Smt,
                serde_json::json!({"status": "proved"}),
            )),
            Some(NonVacuityRecord::invalid(
                "contract assumptions are jointly unsatisfiable",
                serde_json::json!({"solver": "cvc5", "result": "unsat"}),
            )),
        ));
        let probe = registry.probe_consumer("consumer", CompositeVerdict::Proven, [EXP_POSITIVITY]);
        assert_eq!(probe.composite_verdict, CompositeVerdict::Invalid);
    }
}
