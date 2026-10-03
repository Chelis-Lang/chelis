//! The generator of the committed standard-contract discharge table
//! (chelis#2957): every fuzz discharge the standard contracts define, run now.
//! Only the table's recomputation checks call it; the prove lane reads the
//! committed table.

use chelis_types::{
    FloatBinOp, FloatUnOp, ScalarValue, float_binop, float_unop, scalar_from_f64, types::Prim,
};

use super::*;

/// Every fuzz discharge the standard contracts define: each float contract at
/// each admitted width, and each untyped contract once.
pub(super) fn discharge_plan() -> Vec<(&'static str, usize, u64, Option<Prim>)> {
    let mut plan = Vec::new();
    for contract in STANDARD_CONTRACTS {
        for spec in contract.invariants {
            let DischargeSpec::Fuzz { samples, seed, .. } = spec.discharge else {
                continue;
            };
            if is_float_contract(spec.id) {
                for prim in CONTRACT_FLOAT_WIDTHS {
                    plan.push((spec.id, samples, seed, Some(prim)));
                }
            } else {
                plan.push((spec.id, samples, seed, None));
            }
        }
    }
    plan
}

/// Recompute the whole table.
pub(super) fn compute_discharge_table() -> Result<DischargeTable, String> {
    let rows = discharge_plan()
        .into_iter()
        .map(|(id, samples, seed, prim)| DischargeRow {
            contract: id.to_string(),
            samples,
            seed,
            outcome: recompute_fuzz_discharge(id, samples, seed, prim),
        })
        .collect();
    Ok(DischargeTable {
        schema: DISCHARGE_TABLE_SCHEMA.to_string(),
        std_graph_digest: crate::std_graph::normal_cdf_graph_digest()?,
        rows,
    })
}

pub(super) fn render_discharge_table(table: &DischargeTable) -> String {
    let mut text = serde_json::to_string_pretty(table).expect("the table serializes");
    text.push('\n');
    text
}

/// One contract's fuzz discharge at one width, run now.
pub(super) fn recompute_fuzz_discharge(
    id: &str,
    samples: usize,
    seed: u64,
    prim: Option<Prim>,
) -> FuzzOutcome {
    #[cfg(test)]
    RECOMPUTED.with(|count| count.set(count.get() + 1));
    match prim {
        Some(prim) => run_float_fuzz_discharge(id, samples, seed, prim),
        None => run_untyped_fuzz_discharge(id, samples, seed),
    }
}

fn run_untyped_fuzz_discharge(id: &str, samples: usize, seed: u64) -> FuzzOutcome {
    match id {
        QUANTILE_RANGE => fuzz_quantile_range(samples, seed),
        QUANTILE_MONOTONICITY => fuzz_quantile_monotonicity(samples, seed),
        QUANTILE_BOUNDARY => fuzz_quantile_boundary(samples, seed),
        _ => FuzzOutcome::new(
            0,
            f64::INFINITY,
            Some(serde_json::json!({"unsupported_contract": id})),
            "unsupported standard contract",
            None,
        ),
    }
}

/// The float contracts at `prim`: the shipped `Std.Contracts.normal_cdf`
/// graph and the correctly rounded `exp` and `log`, evaluated at that width
/// exactly as a Chelis program evaluates them.
fn run_float_fuzz_discharge(id: &str, samples: usize, seed: u64, prim: Prim) -> FuzzOutcome {
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
            "ordered positive pairs in [1e-6, 1e6] (1e-4..1e4 at f16)",
            |lo, hi| (kernel("log", lo).as_f64_lossy() - kernel("log", hi).as_f64_lossy()).max(0.0),
        ),
        LOG_ONE => {
            let error = kernel("log", at_dtype(prim, 1.0)).as_f64_lossy().abs();
            FuzzOutcome::new(
                samples.max(1),
                error,
                counterexample_if(error, serde_json::json!({"x": 1.0})),
                "log anchor x = 1",
                Some(prim),
            )
        }
        _ => run_untyped_fuzz_discharge(id, samples, seed),
    }
}

/// A sample, rounded once to the contract dtype.
fn at_dtype(prim: Prim, value: f64) -> ScalarValue {
    scalar_from_f64("prove-contract-sample", prim, value)
        .expect("contract samples are finite at every float dtype")
}

/// A correctly rounded kernel at the operand's dtype.
pub(super) fn kernel(name: &str, value: ScalarValue) -> ScalarValue {
    crate::concrete_eval::correctly_rounded(name, value)
        .expect("contract kernels are float transcendentals at a float dtype")
}

/// A radius past which `exp` underflows to `+0` (and overflows to `+inf`) at
/// `prim`, so positivity samples reach the inputs where a strict `exp(x) > 0`
/// fails at that width.
pub(super) fn exp_underflow_radius(prim: Prim) -> f64 {
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
    crate::std_graph::normal_cdf_batch(inputs).map_err(|message| {
        FuzzOutcome::new(
            0,
            f64::INFINITY,
            Some(serde_json::json!({"evaluation_error": message})),
            domain,
            Some(prim),
        )
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
            return FuzzOutcome::new(i + 1, max_error, Some(witness), domain, Some(prim));
        }
    }
    FuzzOutcome::new(samples, max_error, None, domain, Some(prim))
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

pub(super) fn fuzz_unary(
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
    let decades = positive_decades(prim);
    let pairs = (0..samples)
        .map(|i| {
            let a = sample_positive_domain(i, samples, &mut rng, decades);
            let b =
                sample_positive_domain(samples.saturating_sub(i + 1), samples, &mut rng, decades);
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
            return FuzzOutcome::new(
                i + 1,
                max_error,
                Some(serde_json::json!({
                    "data": data, "q": q, "result": result,
                    "min": min_val, "max": max_val, "error": err
                })),
                "quantile range bounded by [min, max]",
                None,
            );
        }
    }
    FuzzOutcome::new(
        samples,
        max_error,
        None,
        "quantile range bounded by [min, max]",
        None,
    )
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
            return FuzzOutcome::new(
                i + 1,
                max_error,
                Some(serde_json::json!({
                    "data": data, "p": p, "q": q,
                    "quantile_p": rp, "quantile_q": rq, "error": err
                })),
                "quantile monotone in q: p <= q => quantile(p) <= quantile(q)",
                None,
            );
        }
    }
    FuzzOutcome::new(
        samples,
        max_error,
        None,
        "quantile monotone in q: p <= q => quantile(p) <= quantile(q)",
        None,
    )
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
            return FuzzOutcome::new(
                i + 1,
                max_error,
                Some(serde_json::json!({
                    "data": data,
                    "quantile_0": q0, "min": min_val,
                    "quantile_1": q1, "max": max_val,
                    "error": err
                })),
                "quantile boundary: q=0 -> min, q=1 -> max",
                None,
            );
        }
    }
    FuzzOutcome::new(
        samples,
        max_error,
        None,
        "quantile boundary: q=0 -> min, q=1 -> max",
        None,
    )
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

/// A positive sample in `[10^-decades, 10^decades]`.
fn sample_positive_domain(index: usize, samples: usize, rng: &mut Lcg, decades: f64) -> f64 {
    let exponent = match index {
        0 => 0.0,
        1 => -decades,
        2 => decades,
        _ => {
            let denom = samples.saturating_sub(1).max(1) as f64;
            let grid = -decades + 2.0 * decades * (index as f64 / denom);
            let jitter = (rng.next_unit() - 0.5) * (2.0 * decades / denom);
            (grid + jitter).clamp(-decades, decades)
        }
    };
    if exponent == 0.0 {
        return 1.0;
    }
    // 10^g through the evaluator's correctly rounded exp, so the sample set
    // does not depend on the host libm.
    let exponent = at_dtype(Prim::F64, exponent * std::f64::consts::LN_10);
    kernel("exp", exponent).as_f64_lossy()
}

/// Decades of positive samples that stay finite and normal at `prim`.
fn positive_decades(prim: Prim) -> f64 {
    match prim {
        Prim::F16 => 4.0,
        _ => 6.0,
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
