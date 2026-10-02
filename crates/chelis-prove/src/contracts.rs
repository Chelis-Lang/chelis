//! Standard mathematical contracts consumed by COMPOSE.
//!
//! The contracts here are reusable proof dependencies. They deliberately
//! carry discharge records in the same shape emitted for producer
//! obligations, so a consumer proof can roll up through the weakest-link
//! machinery instead of rendering an assumed invariant as pure proven.

use chelis_types::{
    FloatBinOp, FloatUnOp, ScalarValue, float_binop, float_unop, scalar_from_f64, types::Prim,
};
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
pub const QUANTILE_RANGE: &str = "std.quantile.range";
pub const QUANTILE_MONOTONICITY: &str = "std.quantile.monotonicity";
pub const QUANTILE_BOUNDARY: &str = "std.quantile.boundary";
pub const QUANTILE_IMPLEMENTATION: &str = "Nautilus.Stats.quantile_vec";

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
                // The float operation underflows to +0 (below about -103.97 at
                // f32), so the strict real-model `exp(x) > 0` is false for it;
                // the contract states the float property (chelis#2965).
                fuzz_invariant(
                    EXP_POSITIVITY,
                    "exp non-negativity",
                    "forall x. exp(x) >= 0",
                    "chelis_intrinsic.exp",
                    4096,
                    0xE0_2026,
                ),
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
        StandardContract {
            id: "std.quantile".to_string(),
            function: QUANTILE_IMPLEMENTATION.to_string(),
            invariants: vec![
                fuzz_invariant(
                    QUANTILE_RANGE,
                    "quantile range boundedness",
                    "forall xs q. 0 <= q <= 1 => min(xs) <= quantile(xs, q) <= max(xs)",
                    QUANTILE_IMPLEMENTATION,
                    8192,
                    0xCA_2026,
                ),
                fuzz_invariant(
                    QUANTILE_MONOTONICITY,
                    "quantile monotonicity in q",
                    "forall xs p q. 0 <= p <= q <= 1 => quantile(xs, p) <= quantile(xs, q)",
                    QUANTILE_IMPLEMENTATION,
                    8192,
                    0xCB_2026,
                ),
                fuzz_invariant(
                    QUANTILE_BOUNDARY,
                    "quantile boundary values",
                    "forall xs. quantile(xs, 0) = min(xs) and quantile(xs, 1) = max(xs)",
                    QUANTILE_IMPLEMENTATION,
                    4096,
                    0xCC_2026,
                ),
            ],
        },
    ]
}

pub fn standard_contract_registry() -> AssumptionRegistry {
    let mut registry = AssumptionRegistry::new();
    for contract in standard_contracts() {
        for invariant in contract.invariants {
            registry.insert(invariant.record);
        }
    }
    registry
}

/// Construct the standard contract registry, attempting to upgrade fuzz-discharged
/// contracts to certified-envelope discharges using the given prover.
/// Contracts the prover cannot prove stay fuzz-discharged (honest degradation).
pub fn standard_contract_registry_with_prover(
    prover: &crate::beacon_contract_prover::BeaconContractProver,
) -> AssumptionRegistry {
    let mut registry = AssumptionRegistry::new();
    for contract in standard_contracts() {
        for invariant in contract.invariants {
            // Try to upgrade fuzz-discharged contracts
            if invariant
                .record
                .discharge
                .as_ref()
                .is_some_and(|d| d.method == DischargeMethod::Fuzz)
                && let Some(discharge) = prover.prove_contract(&invariant.id)
            {
                // Successfully proved by Beacon — use the certified discharge
                let upgraded = AssumptionRecord::new(
                    &invariant.id,
                    Some(discharge),
                    invariant.record.non_vacuity.clone(),
                )
                .with_source("std_contract", &invariant.id)
                .with_discharge_tier(DischargeTier::new(
                    DischargeMethod::CertifiedEnvelope.engine(),
                    DischargeMethod::CertifiedEnvelope,
                    Some(invariant.id.clone()),
                ));
                registry.insert(upgraded);
                continue;
            }
            // Fall through: keep original discharge (fuzz, smt, axiom)
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

/// Construct a contract invariant discharged by Beacon's certified envelope.
/// Used when the BeaconContractProver successfully proves a contract obligation
/// that was previously fuzz-discharged.
pub fn certified_envelope_invariant(
    id: &str,
    description: &str,
    assumption: &str,
    evidence: serde_json::Value,
) -> ContractInvariant {
    ContractInvariant {
        id: id.to_string(),
        description: description.to_string(),
        assumption: assumption.to_string(),
        record: AssumptionRecord::new(
            id,
            Some(AssumptionDischarge::new(
                DischargeMethod::CertifiedEnvelope,
                evidence,
            )),
            Some(NonVacuityRecord::established(serde_json::json!({
                "method": "certified_envelope",
                "result": "sat",
                "contract": id,
            }))),
        )
        .with_source("std_contract", id)
        .with_discharge_tier(DischargeTier::new(
            DischargeMethod::CertifiedEnvelope.engine(),
            DischargeMethod::CertifiedEnvelope,
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
    if let Some(prim) = outcome.dtype {
        evidence["dtype"] = serde_json::json!(prim.name());
    }
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
    /// The dtype the property was evaluated at; `None` for the quantile
    /// contracts, which bind an external library's f64 implementation.
    dtype: Option<Prim>,
}

/// The dtype the float standard contracts are fuzz-discharged at, recorded as
/// the evidence's `dtype`. A fuzz verdict is a statement about the function at
/// one width: the shipped `Std.Contracts.normal_cdf` graph and the correctly
/// rounded `exp` and `log` kernels, evaluated at this dtype exactly as a Chelis
/// program evaluates them (chelis#2965). The discharge covers that width only:
/// at f32 the shipped graph does not satisfy `std.normal_cdf.reflection` within
/// [`FUZZ_TOLERANCE`].
pub const CONTRACT_FUZZ_DTYPE: Prim = Prim::F64;

fn run_fuzz_discharge(id: &str, samples: usize, seed: u64) -> FuzzOutcome {
    let prim = CONTRACT_FUZZ_DTYPE;
    match id {
        NORMAL_CDF_RANGE => fuzz_normal_cdf_range(prim, samples, seed),
        NORMAL_CDF_REFLECTION => fuzz_normal_cdf_reflection(prim, samples, seed),
        NORMAL_CDF_MONOTONICITY => fuzz_normal_cdf_monotonicity(prim, samples, seed),
        EXP_POSITIVITY => fuzz_unary(
            prim,
            samples,
            seed,
            exp_underflow_radius(prim),
            "exp over finite samples spanning the underflow to +0 and the overflow to +inf",
            |x| {
                let y = kernel("exp", x).as_f64_lossy();
                if y >= 0.0 { 0.0 } else { -y }
            },
        ),
        EXP_MONOTONICITY => fuzz_ordered_pair(
            prim,
            samples,
            seed,
            "ordered exp finite pairs in [-10, 10]",
            |lo, hi| (kernel("exp", lo).as_f64_lossy() - kernel("exp", hi).as_f64_lossy()).max(0.0),
        ),
        LOG_MONOTONICITY => fuzz_ordered_positive_pair(
            prim,
            samples,
            seed,
            "ordered positive pairs in [1e-6, 1e6]",
            |lo, hi| (kernel("log", lo).as_f64_lossy() - kernel("log", hi).as_f64_lossy()).max(0.0),
        ),
        LOG_ONE => {
            let error = kernel("log", at_dtype(prim, 1.0)).as_f64_lossy().abs();
            FuzzOutcome {
                checked_samples: samples.max(1),
                max_error: error,
                counterexample: counterexample_if(error, serde_json::json!({"x": 1.0})),
                domain: "log anchor x = 1",
                dtype: Some(prim),
            }
        }
        QUANTILE_RANGE => fuzz_quantile_range(samples, seed),
        QUANTILE_MONOTONICITY => fuzz_quantile_monotonicity(samples, seed),
        QUANTILE_BOUNDARY => fuzz_quantile_boundary(samples, seed),
        _ => FuzzOutcome {
            checked_samples: 0,
            max_error: f64::INFINITY,
            counterexample: Some(serde_json::json!({"unsupported_contract": id})),
            domain: "unsupported standard contract",
            dtype: None,
        },
    }
}

/// A sample, rounded once to the contract dtype.
fn at_dtype(prim: Prim, value: f64) -> ScalarValue {
    scalar_from_f64("prove-contract-sample", prim, value)
        .expect("contract samples are finite at every float dtype")
}

/// A correctly rounded kernel at the operand's dtype.
fn kernel(name: &str, value: ScalarValue) -> ScalarValue {
    crate::concrete_eval::correctly_rounded(name, value)
        .expect("contract kernels are float transcendentals at a float dtype")
}

/// A radius past which `exp` underflows to `+0` (and overflows to `+inf`) at
/// `prim`, so positivity samples reach the inputs where a strict `exp(x) > 0`
/// fails at that width.
fn exp_underflow_radius(prim: Prim) -> f64 {
    match prim {
        Prim::F64 => 1024.0,
        _ => 128.0,
    }
}

fn sample_json(value: ScalarValue) -> serde_json::Value {
    serde_json::json!(value.as_f64_lossy())
}

/// The shipped `normal_cdf` at every input, or the evaluation failure as a
/// counterexample: a discharge that cannot evaluate the graph fails closed.
fn normal_cdf_values(
    inputs: &[ScalarValue],
    domain: &'static str,
    prim: Prim,
) -> Result<Vec<ScalarValue>, FuzzOutcome> {
    crate::std_graph::normal_cdf_batch(inputs).map_err(|message| FuzzOutcome {
        checked_samples: 0,
        max_error: f64::INFINITY,
        counterexample: Some(serde_json::json!({"evaluation_error": message})),
        domain,
        dtype: Some(prim),
    })
}

fn signed_samples(prim: Prim, samples: usize, seed: u64, radius: f64) -> Vec<ScalarValue> {
    let mut rng = Lcg::new(seed);
    (0..samples)
        .map(|i| at_dtype(prim, sample_signed_domain(i, samples, &mut rng, radius)))
        .collect()
}

fn ordered_signed_pairs(prim: Prim, samples: usize, seed: u64) -> Vec<(ScalarValue, ScalarValue)> {
    let mut rng = Lcg::new(seed);
    (0..samples)
        .map(|i| {
            let a = sample_signed_domain(i, samples, &mut rng, 10.0);
            let b = sample_signed_domain(samples.saturating_sub(i + 1), samples, &mut rng, 10.0);
            (at_dtype(prim, a.min(b)), at_dtype(prim, a.max(b)))
        })
        .collect()
}

/// Walk per-sample errors in sample order and stop at the first one above the
/// fuzz tolerance.
fn first_failure(
    errors: impl Iterator<Item = (f64, serde_json::Value)>,
    samples: usize,
    domain: &'static str,
    prim: Prim,
) -> FuzzOutcome {
    let mut max_error = 0.0_f64;
    for (i, (err, mut witness)) in errors.enumerate() {
        max_error = max_error.max(err);
        if err > FUZZ_TOLERANCE {
            witness["error"] = serde_json::json!(err);
            return FuzzOutcome {
                checked_samples: i + 1,
                max_error,
                counterexample: Some(witness),
                domain,
                dtype: Some(prim),
            };
        }
    }
    FuzzOutcome {
        checked_samples: samples,
        max_error,
        counterexample: None,
        domain,
        dtype: Some(prim),
    }
}

fn fuzz_normal_cdf_range(prim: Prim, samples: usize, seed: u64) -> FuzzOutcome {
    const DOMAIN: &str = "normal-cdf finite domain [-10, 10]";
    let xs = signed_samples(prim, samples, seed, 10.0);
    let ys = match normal_cdf_values(&xs, DOMAIN, prim) {
        Ok(ys) => ys,
        Err(outcome) => return outcome,
    };
    let errors = xs.iter().zip(&ys).map(|(x, y)| {
        let y = y.as_f64_lossy();
        let err = if (0.0..=1.0).contains(&y) {
            0.0
        } else if y.is_nan() {
            f64::INFINITY
        } else {
            (0.0 - y).max(y - 1.0)
        };
        (err, serde_json::json!({"x": sample_json(*x)}))
    });
    first_failure(errors, samples, DOMAIN, prim)
}

fn fuzz_normal_cdf_reflection(prim: Prim, samples: usize, seed: u64) -> FuzzOutcome {
    const DOMAIN: &str = "normal-cdf finite domain [-10, 10]";
    let xs = signed_samples(prim, samples, seed, 10.0);
    let mut inputs = xs.clone();
    inputs.extend(xs.iter().map(|x| {
        float_unop(FloatUnOp::Neg, *x).expect("negation is defined at every float dtype")
    }));
    let ys = match normal_cdf_values(&inputs, DOMAIN, prim) {
        Ok(ys) => ys,
        Err(outcome) => return outcome,
    };
    let one = at_dtype(prim, 1.0);
    let errors = xs.iter().enumerate().map(|(i, x)| {
        // `N(-x) = 1 - N(x)`, with `1 - N(x)` computed at the contract dtype as
        // a Chelis program computes it.
        let complement = float_binop(FloatBinOp::Sub, one, ys[i])
            .expect("subtraction is defined at every float dtype");
        let err = (ys[samples + i].as_f64_lossy() - complement.as_f64_lossy()).abs();
        let err = if err.is_nan() { f64::INFINITY } else { err };
        (err, serde_json::json!({"x": sample_json(*x)}))
    });
    first_failure(errors, samples, DOMAIN, prim)
}

fn fuzz_normal_cdf_monotonicity(prim: Prim, samples: usize, seed: u64) -> FuzzOutcome {
    const DOMAIN: &str = "ordered normal-cdf finite pairs in [-10, 10]";
    let pairs = ordered_signed_pairs(prim, samples, seed);
    let inputs = pairs
        .iter()
        .flat_map(|(lo, hi)| [*lo, *hi])
        .collect::<Vec<_>>();
    let ys = match normal_cdf_values(&inputs, DOMAIN, prim) {
        Ok(ys) => ys,
        Err(outcome) => return outcome,
    };
    let errors = pairs.iter().enumerate().map(|(i, (lo, hi))| {
        let err = (ys[2 * i].as_f64_lossy() - ys[2 * i + 1].as_f64_lossy()).max(0.0);
        (
            err,
            serde_json::json!({"lo": sample_json(*lo), "hi": sample_json(*hi)}),
        )
    });
    first_failure(errors, samples, DOMAIN, prim)
}

fn fuzz_unary(
    prim: Prim,
    samples: usize,
    seed: u64,
    radius: f64,
    domain: &'static str,
    error: impl Fn(ScalarValue) -> f64,
) -> FuzzOutcome {
    let xs = signed_samples(prim, samples, seed, radius);
    let errors = xs
        .iter()
        .map(|x| (error(*x), serde_json::json!({"x": sample_json(*x)})));
    first_failure(errors, samples, domain, prim)
}

fn fuzz_ordered_pair(
    prim: Prim,
    samples: usize,
    seed: u64,
    domain: &'static str,
    error: impl Fn(ScalarValue, ScalarValue) -> f64,
) -> FuzzOutcome {
    let pairs = ordered_signed_pairs(prim, samples, seed);
    let errors = pairs.iter().map(|(lo, hi)| {
        (
            error(*lo, *hi),
            serde_json::json!({"lo": sample_json(*lo), "hi": sample_json(*hi)}),
        )
    });
    first_failure(errors, samples, domain, prim)
}

fn fuzz_ordered_positive_pair(
    prim: Prim,
    samples: usize,
    seed: u64,
    domain: &'static str,
    error: impl Fn(ScalarValue, ScalarValue) -> f64,
) -> FuzzOutcome {
    let mut rng = Lcg::new(seed);
    let pairs = (0..samples)
        .map(|i| {
            let a = sample_positive_domain(i, samples, &mut rng);
            let b = sample_positive_domain(samples.saturating_sub(i + 1), samples, &mut rng);
            (at_dtype(prim, a.min(b)), at_dtype(prim, a.max(b)))
        })
        .collect::<Vec<_>>();
    let errors = pairs.iter().map(|(lo, hi)| {
        (
            error(*lo, *hi),
            serde_json::json!({"lo": sample_json(*lo), "hi": sample_json(*hi)}),
        )
    });
    first_failure(errors, samples, domain, prim)
}

fn fuzz_quantile_range(samples: usize, seed: u64) -> FuzzOutcome {
    use crate::concrete_eval;
    let mut rng = Lcg::new(seed);
    let mut max_error = 0.0_f64;
    for i in 0..samples {
        // Generate a random data vector of size 3-8 and a quantile level
        let n = 3 + (rng.next_unit() * 5.0) as usize;
        let data: Vec<f64> = (0..n).map(|_| (rng.next_unit() - 0.5) * 20.0).collect();
        let q = rng.next_unit();
        let result = concrete_eval::quantile_linear_pub(&data, q);
        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let low_err = (min_val - result).max(0.0);
        let high_err = (result - max_val).max(0.0);
        let err = low_err.max(high_err);
        max_error = max_error.max(err);
        if err > FUZZ_TOLERANCE {
            return FuzzOutcome {
                checked_samples: i + 1,
                max_error,
                counterexample: Some(serde_json::json!({
                    "data": data, "q": q, "result": result,
                    "min": min_val, "max": max_val, "error": err
                })),
                domain: "quantile range bounded by [min, max]",
                dtype: None,
            };
        }
    }
    FuzzOutcome {
        checked_samples: samples,
        max_error,
        counterexample: None,
        domain: "quantile range bounded by [min, max]",
        dtype: None,
    }
}

fn fuzz_quantile_monotonicity(samples: usize, seed: u64) -> FuzzOutcome {
    use crate::concrete_eval;
    let mut rng = Lcg::new(seed);
    let mut max_error = 0.0_f64;
    for i in 0..samples {
        let n = 3 + (rng.next_unit() * 5.0) as usize;
        let data: Vec<f64> = (0..n).map(|_| (rng.next_unit() - 0.5) * 20.0).collect();
        let a = rng.next_unit();
        let b = rng.next_unit();
        let (p, q) = if a <= b { (a, b) } else { (b, a) };
        let rp = concrete_eval::quantile_linear_pub(&data, p);
        let rq = concrete_eval::quantile_linear_pub(&data, q);
        let err = (rp - rq).max(0.0);
        max_error = max_error.max(err);
        if err > FUZZ_TOLERANCE {
            return FuzzOutcome {
                checked_samples: i + 1,
                max_error,
                counterexample: Some(serde_json::json!({
                    "data": data, "p": p, "q": q,
                    "quantile_p": rp, "quantile_q": rq, "error": err
                })),
                domain: "quantile monotone in q: p <= q => quantile(p) <= quantile(q)",
                dtype: None,
            };
        }
    }
    FuzzOutcome {
        checked_samples: samples,
        max_error,
        counterexample: None,
        domain: "quantile monotone in q: p <= q => quantile(p) <= quantile(q)",
        dtype: None,
    }
}

fn fuzz_quantile_boundary(samples: usize, seed: u64) -> FuzzOutcome {
    use crate::concrete_eval;
    let mut rng = Lcg::new(seed);
    let mut max_error = 0.0_f64;
    for i in 0..samples {
        let n = 3 + (rng.next_unit() * 5.0) as usize;
        let data: Vec<f64> = (0..n).map(|_| (rng.next_unit() - 0.5) * 20.0).collect();
        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let q0 = concrete_eval::quantile_linear_pub(&data, 0.0);
        let q1 = concrete_eval::quantile_linear_pub(&data, 1.0);
        let err = (q0 - min_val).abs().max((q1 - max_val).abs());
        max_error = max_error.max(err);
        if err > FUZZ_TOLERANCE {
            return FuzzOutcome {
                checked_samples: i + 1,
                max_error,
                counterexample: Some(serde_json::json!({
                    "data": data,
                    "quantile_0": q0, "min": min_val,
                    "quantile_1": q1, "max": max_val,
                    "error": err
                })),
                domain: "quantile boundary: q=0 -> min, q=1 -> max",
                dtype: None,
            };
        }
    }
    FuzzOutcome {
        checked_samples: samples,
        max_error,
        counterexample: None,
        domain: "quantile boundary: q=0 -> min, q=1 -> max",
        dtype: None,
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
            // 10^g through the correctly rounded kernel, so the sample set does
            // not depend on the host libm.
            chelis_crmath::exp_f64((grid + jitter).clamp(-6.0, 6.0) * std::f64::consts::LN_10)
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
                DischargeMethod::CertifiedEnvelope => {
                    assert_eq!(discharge.evidence["status"], "proved");
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

    // --- chelis#2965: discharges at a declared dtype ---

    fn evidence(id: &str) -> serde_json::Value {
        all_invariants()
            .into_iter()
            .find(|invariant| invariant.id == id)
            .unwrap_or_else(|| panic!("{id} exists"))
            .record
            .discharge
            .expect("discharged")
            .evidence
    }

    #[test]
    fn chelis_2965_float_contracts_record_their_evaluation_dtype() {
        for id in [
            NORMAL_CDF_RANGE,
            NORMAL_CDF_REFLECTION,
            NORMAL_CDF_MONOTONICITY,
            EXP_POSITIVITY,
            EXP_MONOTONICITY,
            LOG_MONOTONICITY,
            LOG_ONE,
        ] {
            assert_eq!(
                evidence(id)["dtype"],
                CONTRACT_FUZZ_DTYPE.name(),
                "{id} does not record the dtype it was evaluated at"
            );
        }
        assert!(evidence(QUANTILE_RANGE).get("dtype").is_none());
    }

    #[test]
    fn chelis_2965_exp_positivity_is_the_float_property() {
        let invariant = all_invariants()
            .into_iter()
            .find(|invariant| invariant.id == EXP_POSITIVITY)
            .unwrap();
        assert_eq!(invariant.assumption, "forall x. exp(x) >= 0");
        let discharge = invariant.record.discharge.unwrap();
        assert_eq!(discharge.method, DischargeMethod::Fuzz);
        assert_eq!(discharge.evidence["status"], "validated");
    }

    #[test]
    fn chelis_2965_exp_positivity_samples_reach_underflow_where_the_strict_law_fails() {
        // The positivity sampler at each width reaches inputs where exp
        // underflows to +0, so the strict real-model law `exp(x) > 0` is
        // refuted there while the restated `exp(x) >= 0` holds.
        for prim in [Prim::F32, Prim::F64] {
            let radius = exp_underflow_radius(prim);
            let strict = fuzz_unary(prim, 4096, 0xE0_2026, radius, "strict", |x| {
                if kernel("exp", x).as_f64_lossy() > 0.0 {
                    0.0
                } else {
                    1.0
                }
            });
            let counterexample = strict
                .counterexample
                .unwrap_or_else(|| panic!("strict exp positivity not refuted at {}", prim.name()));
            let x = counterexample["x"].as_f64().unwrap();
            assert!(x < -100.0, "{}: {x}", prim.name());
            let weak = fuzz_unary(prim, 4096, 0xE0_2026, radius, "weak", |x| {
                let y = kernel("exp", x).as_f64_lossy();
                if y >= 0.0 { 0.0 } else { -y }
            });
            assert!(weak.counterexample.is_none(), "{}", prim.name());
        }
    }

    #[test]
    fn chelis_2965_shipped_normal_cdf_reflection_fails_at_f32() {
        // The reason the contracts are discharged at f64: evaluated at f32 the
        // shipped graph's `1 - N(x)` and `N(-x)` round differently by far more
        // than the fuzz tolerance.
        let outcome = fuzz_normal_cdf_reflection(Prim::F32, 256, 0xC0DF_2026);
        assert!(outcome.counterexample.is_some(), "{outcome:?}");
        assert!(outcome.max_error > FUZZ_TOLERANCE);
        let at_f64 = fuzz_normal_cdf_reflection(Prim::F64, 256, 0xC0DF_2026);
        assert!(at_f64.counterexample.is_none(), "{at_f64:?}");
    }
}
