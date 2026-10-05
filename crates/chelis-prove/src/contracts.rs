//! Standard mathematical contracts consumed by COMPOSE.
//!
//! The contracts here are reusable proof dependencies. They deliberately
//! carry discharge records in the same shape emitted for producer
//! obligations, so a consumer proof can roll up through the weakest-link
//! machinery instead of rendering an assumed invariant as pure proven.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use chelis_reef::EmbeddedRuntime;
use chelis_types::types::Prim;
use serde::{Deserialize, Serialize};

mod generator;

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

/// Every float width a float standard contract admits. `normal_cdf` is generic
/// over `Float`, and `exp` and `log` are primitives at every float dtype.
pub const CONTRACT_FLOAT_WIDTHS: [Prim; 4] = [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64];

/// How a standard invariant is discharged.
enum DischargeSpec {
    /// A fuzz discharge over `samples` samples drawn from `seed`, against
    /// `implementation`.
    Fuzz {
        implementation: &'static str,
        samples: usize,
        seed: u64,
    },
    /// A cvc5 proof in the real model.
    Smt,
}

struct InvariantSpec {
    id: &'static str,
    description: &'static str,
    assumption: &'static str,
    discharge: DischargeSpec,
}

struct ContractSpec {
    id: &'static str,
    function: &'static str,
    invariants: &'static [InvariantSpec],
}

const fn fuzz(
    id: &'static str,
    description: &'static str,
    assumption: &'static str,
    implementation: &'static str,
    samples: usize,
    seed: u64,
) -> InvariantSpec {
    InvariantSpec {
        id,
        description,
        assumption,
        discharge: DischargeSpec::Fuzz {
            implementation,
            samples,
            seed,
        },
    }
}

/// The standard contracts and how each invariant is discharged.
const STANDARD_CONTRACTS: &[ContractSpec] = &[
    ContractSpec {
        id: "std.normal_cdf",
        function: NORMAL_CDF_IMPLEMENTATION,
        invariants: &[
            fuzz(
                NORMAL_CDF_RANGE,
                "normal CDF range",
                "forall x. 0 <= N(x) <= 1",
                NORMAL_CDF_IMPLEMENTATION,
                8192,
                0xC0DF_2026,
            ),
            fuzz(
                NORMAL_CDF_REFLECTION,
                "normal CDF reflection",
                "forall x. N(-x) = 1 - N(x)",
                NORMAL_CDF_IMPLEMENTATION,
                8192,
                0xC0DF_2026,
            ),
            fuzz(
                NORMAL_CDF_MONOTONICITY,
                "normal CDF monotonicity",
                "forall x y. x <= y => N(x) <= N(y)",
                NORMAL_CDF_IMPLEMENTATION,
                8192,
                0xC0DF_2026,
            ),
        ],
    },
    ContractSpec {
        id: "std.exp",
        function: "exp",
        invariants: &[
            // The float operation underflows to +0 (below about -103.97 at
            // f32), so the strict real-model `exp(x) > 0` is false for it;
            // the contract states the float property (chelis#2965).
            fuzz(
                EXP_POSITIVITY,
                "exp non-negativity",
                "forall x. exp(x) >= 0",
                "chelis_intrinsic.exp",
                4096,
                0xE0_2026,
            ),
            fuzz(
                EXP_MONOTONICITY,
                "exp monotonicity",
                "forall x y. x <= y => exp(x) <= exp(y)",
                "chelis_intrinsic.exp",
                4096,
                0xE_2026,
            ),
            InvariantSpec {
                id: EXP_ZERO,
                description: "exp zero",
                assumption: "exp(0) = 1",
                discharge: DischargeSpec::Smt,
            },
        ],
    },
    ContractSpec {
        id: "std.log",
        function: "log",
        invariants: &[
            fuzz(
                LOG_MONOTONICITY,
                "log monotonicity on positives",
                "forall x y. 0 < x <= y => log(x) <= log(y)",
                "chelis_intrinsic.log",
                4096,
                0x10_2026,
            ),
            fuzz(
                LOG_ONE,
                "log one",
                "log(1) = 0",
                "chelis_intrinsic.log",
                64,
                0x10_2026,
            ),
        ],
    },
    ContractSpec {
        id: "std.quantile",
        function: QUANTILE_IMPLEMENTATION,
        invariants: &[
            fuzz(
                QUANTILE_RANGE,
                "quantile range boundedness",
                "forall xs q. 0 <= q <= 1 => min(xs) <= quantile(xs, q) <= max(xs)",
                QUANTILE_IMPLEMENTATION,
                8192,
                0xCA_2026,
            ),
            fuzz(
                QUANTILE_MONOTONICITY,
                "quantile monotonicity in q",
                "forall xs p q. 0 <= p <= q <= 1 => quantile(xs, p) <= quantile(xs, q)",
                QUANTILE_IMPLEMENTATION,
                8192,
                0xCB_2026,
            ),
            fuzz(
                QUANTILE_BOUNDARY,
                "quantile boundary values",
                "forall xs. quantile(xs, 0) = min(xs) and quantile(xs, 1) = max(xs)",
                QUANTILE_IMPLEMENTATION,
                4096,
                0xCC_2026,
            ),
        ],
    },
];

/// The standard contracts, with each float contract's discharge resolved at
/// every width in `widths` (chelis#2965). A float fuzz property is a
/// statement about the function at one width, so the discharge is per width:
/// the record fails when the property fails at any of the widths, and names
/// that width. A consumer passes the widths its operands can have; an empty
/// slice means every admitted width. The per-width outcomes come from the
/// committed discharge table, never from a fuzz run in this process.
pub fn standard_contracts_at(
    widths: &[Prim],
    runtime: &'static EmbeddedRuntime,
) -> Vec<StandardContract> {
    standard_contracts_selected(widths, |_| true, runtime)
}

/// The standard contracts reduced to the invariants `select` accepts; a
/// contract with no selected invariant is omitted.
fn standard_contracts_selected(
    widths: &[Prim],
    select: impl Fn(&str) -> bool,
    runtime: &'static EmbeddedRuntime,
) -> Vec<StandardContract> {
    let widths = consumer_widths(widths);
    STANDARD_CONTRACTS
        .iter()
        .filter_map(|contract| {
            let invariants = contract
                .invariants
                .iter()
                .filter(|spec| select(spec.id))
                .map(|spec| match spec.discharge {
                    DischargeSpec::Fuzz {
                        implementation,
                        samples,
                        seed,
                    } => fuzz_invariant(spec, implementation, samples, seed, &widths, runtime),
                    DischargeSpec::Smt => smt_invariant(spec.id, spec.description, spec.assumption),
                })
                .collect::<Vec<_>>();
            (!invariants.is_empty()).then(|| StandardContract {
                id: contract.id.to_string(),
                function: contract.function.to_string(),
                invariants,
            })
        })
        .collect()
}

pub fn standard_contract_registry(
    widths: &[Prim],
    runtime: &'static EmbeddedRuntime,
) -> AssumptionRegistry {
    standard_contract_registry_for(None, widths, None, runtime)
}

/// Construct the standard contract registry, attempting to upgrade fuzz-discharged
/// contracts to certified-envelope discharges using the given prover.
/// Contracts the prover cannot prove stay fuzz-discharged (honest degradation).
pub fn standard_contract_registry_with_prover(
    prover: &crate::beacon_contract_prover::BeaconContractProver,
    widths: &[Prim],
    runtime: &'static EmbeddedRuntime,
) -> AssumptionRegistry {
    standard_contract_registry_for(None, widths, Some(prover), runtime)
}

/// The registry over the invariants in `contracts` (every invariant when
/// `None`), so a consumer resolves only the contracts it reaches. With a
/// prover, a fuzz-validated invariant is upgraded to a certified-envelope
/// discharge when the prover proves it.
pub fn standard_contract_registry_for(
    contracts: Option<&BTreeSet<String>>,
    widths: &[Prim],
    prover: Option<&crate::beacon_contract_prover::BeaconContractProver>,
    runtime: &'static EmbeddedRuntime,
) -> AssumptionRegistry {
    let mut registry = AssumptionRegistry::new();
    let selected = |id: &str| contracts.is_none_or(|contracts| contracts.contains(id));
    for contract in standard_contracts_selected(widths, selected, runtime) {
        for invariant in contract.invariants {
            // Try to upgrade fuzz-validated contracts. A contract whose fuzz
            // discharge failed at a consumer width stays failed: a proof about
            // the real function does not make the float property true there.
            if let Some(prover) = prover
                && invariant.record.discharge.as_ref().is_some_and(|d| {
                    d.method == DischargeMethod::Fuzz && d.evidence["status"] == "validated"
                })
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
    spec: &InvariantSpec,
    implementation: &str,
    samples: usize,
    seed: u64,
    widths: &[Prim],
    runtime: &'static EmbeddedRuntime,
) -> ContractInvariant {
    let (id, description, assumption) = (spec.id, spec.description, spec.assumption);
    ContractInvariant {
        id: id.to_string(),
        description: description.to_string(),
        assumption: assumption.to_string(),
        record: AssumptionRecord::new(
            id,
            Some(AssumptionDischarge::new(
                DischargeMethod::Fuzz,
                fuzz_discharge_evidence(
                    id,
                    assumption,
                    implementation,
                    samples,
                    seed,
                    widths,
                    runtime,
                ),
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
    widths: &[Prim],
    runtime: &'static EmbeddedRuntime,
) -> serde_json::Value {
    let outcomes = if is_float_contract(id) {
        widths
            .iter()
            .map(|prim| run_fuzz_discharge(id, samples, seed, Some(*prim), runtime))
            .collect::<Vec<_>>()
    } else {
        vec![run_fuzz_discharge(id, samples, seed, None, runtime)]
    };
    // The reported outcome is the first failing width, else the first width.
    let reported = outcomes
        .iter()
        .find(|outcome| outcome.counterexample.is_some())
        .unwrap_or(&outcomes[0]);
    let status = |outcome: &FuzzOutcome| {
        if outcome.counterexample.is_some() {
            "failed"
        } else {
            "validated"
        }
    };
    let mut evidence = serde_json::json!({
        "status": status(reported),
        "implementation": implementation,
        "samples": samples,
        "seed": seed,
        "tolerance": FUZZ_TOLERANCE,
        "domain": reported.domain,
        "assumption": assumption,
        "checked_samples": reported.checked_samples,
        "max_error": reported.max_error,
    });
    if let Some(prim) = reported.dtype {
        evidence["dtype"] = serde_json::json!(prim.name());
        evidence["widths"] = outcomes
            .iter()
            .filter_map(|outcome| {
                outcome
                    .dtype
                    .map(|prim| (prim.name().to_string(), serde_json::json!(status(outcome))))
            })
            .collect::<serde_json::Map<_, _>>()
            .into();
    }
    if let Some(counterexample) = reported.counterexample.clone() {
        evidence["counterexample"] = counterexample;
    }
    evidence
}

/// The widths a discharge runs at: the consumer's, deduplicated in order, or
/// every admitted width when the consumer names none.
fn consumer_widths(widths: &[Prim]) -> Vec<Prim> {
    let mut out = Vec::new();
    for prim in widths {
        if prim.is_float() && !out.contains(prim) {
            out.push(*prim);
        }
    }
    if out.is_empty() {
        out.extend(CONTRACT_FLOAT_WIDTHS);
    }
    out
}

fn is_float_contract(id: &str) -> bool {
    matches!(
        id,
        NORMAL_CDF_RANGE
            | NORMAL_CDF_REFLECTION
            | NORMAL_CDF_MONOTONICITY
            | EXP_POSITIVITY
            | EXP_MONOTONICITY
            | LOG_MONOTONICITY
            | LOG_ONE
    )
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FuzzOutcome {
    checked_samples: usize,
    /// The largest per-sample error, as the evidence renders it (a
    /// non-finite error has no JSON number and renders as `null`).
    max_error: serde_json::Value,
    counterexample: Option<serde_json::Value>,
    domain: String,
    /// The dtype the property was evaluated at; `None` for the quantile
    /// contracts, which bind an external library's f64 implementation.
    #[serde(with = "dtype_name")]
    dtype: Option<Prim>,
}

/// A contract dtype serialized by its Chelis spelling (`f32`).
mod dtype_name {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};

    use super::{CONTRACT_FLOAT_WIDTHS, Prim};

    pub fn serialize<S: Serializer>(prim: &Option<Prim>, serializer: S) -> Result<S::Ok, S::Error> {
        prim.map(|prim| prim.name()).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Prim>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|name| {
                CONTRACT_FLOAT_WIDTHS
                    .into_iter()
                    .find(|prim| prim.name() == name)
                    .ok_or_else(|| {
                        D::Error::custom(format!("`{name}` is not a contract float width"))
                    })
            })
            .transpose()
    }
}

impl FuzzOutcome {
    fn new(
        checked_samples: usize,
        max_error: f64,
        counterexample: Option<serde_json::Value>,
        domain: &str,
        dtype: Option<Prim>,
    ) -> Self {
        Self {
            checked_samples,
            max_error: serde_json::json!(max_error),
            counterexample,
            domain: domain.to_string(),
            dtype,
        }
    }
}

/// The committed per-width fuzz discharge table (chelis#2957). Its rows are
/// a function of the contract, the sample plan, the width, the shipped
/// chelis-std graph and the compiler's kernels, all fixed per compiler build,
/// so the prove lane reads them instead of fuzzing at startup. The nightly
/// `standard_contract_discharge_table` test recomputes every row and compares
/// the rendering byte for byte (`CHELIS_PROVE_DISCHARGE_TABLE_WRITE=1`
/// rewrites the file); a per-pull-request canary recomputes one row.
pub const STANDARD_CONTRACT_DISCHARGE_TABLE_PATH: &str = DISCHARGE_TABLE_PATH;
const DISCHARGE_TABLE_PATH: &str = "data/standard_contract_discharges.json";
const DISCHARGE_TABLE_JSON: &str = include_str!("../data/standard_contract_discharges.json");
const DISCHARGE_TABLE_SCHEMA: &str = "chelis-prove-standard-contract-discharges-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DischargeTable {
    schema: String,
    /// `std_graph::normal_cdf_graph_digest` of the graph the `normal_cdf`
    /// rows were computed against.
    std_graph_digest: String,
    rows: Vec<DischargeRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DischargeRow {
    contract: String,
    samples: usize,
    seed: u64,
    #[serde(flatten)]
    outcome: FuzzOutcome,
}

/// The committed table as the prove lane reads it.
pub fn committed_standard_contract_discharge_table() -> &'static str {
    DISCHARGE_TABLE_JSON
}

/// The table recomputed now against `runtime`'s chelis-std, rendered as the
/// committed file is. This runs every fuzz discharge; nothing on the prove
/// path calls it.
pub fn recompute_standard_contract_discharge_table(
    runtime: &'static EmbeddedRuntime,
) -> Result<String, String> {
    generator::compute_discharge_table(runtime)
        .map(|table| generator::render_discharge_table(&table))
}

fn committed_discharge_table() -> Result<&'static DischargeTable, String> {
    static TABLE: OnceLock<Result<DischargeTable, String>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            let table: DischargeTable = serde_json::from_str(DISCHARGE_TABLE_JSON)
                .map_err(|err| format!("{DISCHARGE_TABLE_PATH} does not parse: {err}"))?;
            if table.schema != DISCHARGE_TABLE_SCHEMA {
                return Err(format!(
                    "{DISCHARGE_TABLE_PATH} has schema `{}`, expected `{DISCHARGE_TABLE_SCHEMA}`",
                    table.schema
                ));
            }
            Ok(table)
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(test)]
thread_local! {
    /// Table rows this thread resolved, so tests can tell which contracts a
    /// consumer reached.
    pub(crate) static RESOLVED_ROWS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Fuzz discharges this thread ran.
    pub(crate) static RECOMPUTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One contract's fuzz discharge at one width, read from the committed table.
/// A `normal_cdf` row is served only when the table's recorded graph digest is
/// the digest of `runtime`'s chelis-std.
fn run_fuzz_discharge(
    id: &str,
    samples: usize,
    seed: u64,
    prim: Option<Prim>,
    runtime: &'static EmbeddedRuntime,
) -> FuzzOutcome {
    #[cfg(test)]
    RESOLVED_ROWS.with(|rows| {
        rows.borrow_mut().push(format!(
            "{id}@{}",
            prim.map_or("untyped", |prim| prim.name())
        ));
    });
    table_outcome(
        committed_discharge_table(),
        || crate::std_graph::normal_cdf_graph_digest(runtime),
        id,
        samples,
        seed,
        prim,
    )
}

/// The row of `table` for one discharge. A missing table or row, or a
/// `normal_cdf` row recorded against a different shipped graph than
/// `shipped_digest` names, fails closed: the outcome carries the reason as
/// its counterexample, so the record is `failed`.
fn table_outcome(
    table: Result<&DischargeTable, String>,
    shipped_digest: impl FnOnce() -> Result<String, String>,
    id: &str,
    samples: usize,
    seed: u64,
    prim: Option<Prim>,
) -> FuzzOutcome {
    let unavailable = |reason: serde_json::Value| {
        FuzzOutcome::new(
            0,
            f64::INFINITY,
            Some(serde_json::json!({"discharge_table": reason})),
            "committed standard-contract discharge table",
            prim,
        )
    };
    let table = match table {
        Ok(table) => table,
        Err(message) => return unavailable(serde_json::json!(message)),
    };
    let Some(row) = table.rows.iter().find(|row| {
        row.contract == id
            && row.samples == samples
            && row.seed == seed
            && row.outcome.dtype == prim
    }) else {
        return unavailable(serde_json::json!(format!(
            "{DISCHARGE_TABLE_PATH} has no row for {id} at {} with {samples} samples and seed {seed}",
            prim.map_or("no dtype", |prim| prim.name())
        )));
    };
    if is_normal_cdf_contract(id) {
        match shipped_digest() {
            Ok(digest) if digest == table.std_graph_digest => {}
            Ok(digest) => {
                return unavailable(serde_json::json!({
                    "stale": DISCHARGE_TABLE_PATH,
                    "recorded_std_graph_digest": table.std_graph_digest,
                    "shipped_std_graph_digest": digest,
                }));
            }
            Err(message) => return unavailable(serde_json::json!(message)),
        }
    }
    row.outcome.clone()
}

fn is_normal_cdf_contract(id: &str) -> bool {
    matches!(
        id,
        NORMAL_CDF_RANGE | NORMAL_CDF_REFLECTION | NORMAL_CDF_MONOTONICITY
    )
}

#[cfg(test)]
mod tests {
    use super::generator::*;
    use super::*;

    fn all_invariants() -> Vec<ContractInvariant> {
        standard_contracts_at(&[Prim::F64], &chelis_std_bundle::EMBEDDED_RUNTIME)
            .into_iter()
            .flat_map(|contract| contract.invariants)
            .collect()
    }

    #[test]
    fn k1_put_call_parity_consumes_cdf_reflection_as_fuzz_qualified_contract() {
        let registry =
            standard_contract_registry(&[Prim::F64], &chelis_std_bundle::EMBEDDED_RUNTIME);
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
    fn chelis_2957_normal_cdf_rows_are_checked_against_the_callers_runtime() {
        // A runtime whose archive is not the one the table was computed from.
        // Its graph digest cannot match, so every normal_cdf row fails closed,
        // while a row that does not read the graph is still served.
        static OTHER: EmbeddedRuntime =
            EmbeddedRuntime::new("0.0.0", b"not the shipped archive", b"");
        let mut normal_cdf_rows = 0;
        for contract in standard_contracts_at(&[Prim::F64], &OTHER) {
            for invariant in contract.invariants {
                let evidence = &invariant.record.discharge.as_ref().unwrap().evidence;
                if is_normal_cdf_contract(&invariant.id) {
                    normal_cdf_rows += 1;
                    assert_eq!(evidence["status"], "failed", "{}", invariant.id);
                } else {
                    assert_ne!(evidence["status"], "failed", "{}", invariant.id);
                }
            }
        }
        assert_eq!(normal_cdf_rows, 3);
    }

    #[test]
    fn normal_cdf_contracts_bind_to_bundled_callable_implementation() {
        let normal = standard_contracts_at(&[Prim::F64], &chelis_std_bundle::EMBEDDED_RUNTIME)
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
        let mut registry =
            standard_contract_registry(&[Prim::F64], &chelis_std_bundle::EMBEDDED_RUNTIME);
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
        let mut registry =
            standard_contract_registry(&[Prim::F64], &chelis_std_bundle::EMBEDDED_RUNTIME);
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
            let evidence = evidence(id);
            assert_eq!(evidence["dtype"], "f64", "{id}");
            assert_eq!(
                evidence["widths"],
                serde_json::json!({"f64": "validated"}),
                "{id}"
            );
        }
        assert!(evidence(QUANTILE_RANGE).get("dtype").is_none());
    }

    #[test]
    fn chelis_2965_reflection_is_discharged_per_width_and_fails_at_f32() {
        // No consumer may be validated at a width where the property is false:
        // the f32 discharge of reflection fails on the shipped graph, so an f32
        // consumer composes to Failed while an f64 consumer stays validated.
        let at = |widths: &[Prim]| {
            standard_contract_registry(widths, &chelis_std_bundle::EMBEDDED_RUNTIME).probe_consumer(
                "put_call_parity",
                CompositeVerdict::Proven,
                [NORMAL_CDF_REFLECTION],
            )
        };
        let f64_probe = at(&[Prim::F64]);
        assert_eq!(
            f64_probe.composite_verdict,
            CompositeVerdict::ProvenModuloFuzzValidatedContract
        );
        let f32_probe = at(&[Prim::F32]);
        assert_eq!(f32_probe.composite_verdict, CompositeVerdict::Failed);
        let evidence = &f32_probe.assumptions[0]
            .discharge
            .as_ref()
            .unwrap()
            .evidence;
        assert_eq!(evidence["status"], "failed");
        assert_eq!(evidence["dtype"], "f32");
        assert!(evidence.get("counterexample").is_some());
        // A consumer that can reach both widths gets the weaker one.
        let both = at(&[Prim::F64, Prim::F32]);
        assert_eq!(both.composite_verdict, CompositeVerdict::Failed);
        let evidence = &both.assumptions[0].discharge.as_ref().unwrap().evidence;
        assert_eq!(
            evidence["widths"],
            serde_json::json!({"f64": "validated", "f32": "failed"})
        );
    }

    #[test]
    fn chelis_2965_unnamed_consumer_width_resolves_at_every_admitted_width() {
        assert_eq!(consumer_widths(&[]), CONTRACT_FLOAT_WIDTHS.to_vec());
        assert_eq!(
            consumer_widths(&[Prim::Int32]),
            CONTRACT_FLOAT_WIDTHS.to_vec()
        );
        assert_eq!(consumer_widths(&[Prim::F32, Prim::F32]), vec![Prim::F32]);
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
    fn chelis_2965_per_width_outcomes_of_the_float_contracts() {
        // Executed per-width verdicts of the shipped graph and the correctly
        // rounded kernels. Reflection holds only at f64: near zero `1 - N(x)`
        // and `N(-x)` lie on different rounding grids at the narrower widths.
        // The section 3.3 `Phi` graph is monotone at every width.
        let expected = [
            (NORMAL_CDF_RANGE, ["validated"; 4]),
            (
                NORMAL_CDF_REFLECTION,
                ["failed", "failed", "failed", "validated"],
            ),
            (NORMAL_CDF_MONOTONICITY, ["validated"; 4]),
            (EXP_POSITIVITY, ["validated"; 4]),
            (EXP_MONOTONICITY, ["validated"; 4]),
            (LOG_MONOTONICITY, ["validated"; 4]),
            (LOG_ONE, ["validated"; 4]),
        ];
        let invariants =
            standard_contracts_at(&CONTRACT_FLOAT_WIDTHS, &chelis_std_bundle::EMBEDDED_RUNTIME)
                .into_iter()
                .flat_map(|contract| contract.invariants)
                .collect::<Vec<_>>();
        for (id, statuses) in expected {
            let invariant = invariants
                .iter()
                .find(|invariant| invariant.id == id)
                .unwrap();
            let widths = &invariant.record.discharge.as_ref().unwrap().evidence["widths"];
            for (prim, status) in CONTRACT_FLOAT_WIDTHS.iter().zip(statuses) {
                assert_eq!(widths[prim.name()], status, "{id} at {}", prim.name());
            }
        }
    }

    // --- chelis#2957: the committed discharge table ---

    #[test]
    fn chelis_2957_discharge_table_canary_recomputes_the_f32_reflection_row() {
        // The per-pull-request canary: one row recomputed now, the f32
        // reflection discharge, which fails on the shipped graph. The nightly
        // `standard_contract_discharge_table` test recomputes every row.
        let committed = committed_discharge_table()
            .unwrap()
            .rows
            .iter()
            .find(|row| {
                row.contract == NORMAL_CDF_REFLECTION && row.outcome.dtype == Some(Prim::F32)
            })
            .unwrap();
        let recomputed = recompute_fuzz_discharge(
            NORMAL_CDF_REFLECTION,
            committed.samples,
            committed.seed,
            Some(Prim::F32),
            &chelis_std_bundle::EMBEDDED_RUNTIME,
        );
        assert_eq!(&recomputed, &committed.outcome);
        assert!(recomputed.counterexample.is_some());
    }

    #[test]
    fn chelis_2957_committed_table_rows_are_the_discharge_plan() {
        let table = committed_discharge_table().unwrap();
        let rows = table
            .rows
            .iter()
            .map(|row| {
                (
                    row.contract.as_str(),
                    row.samples,
                    row.seed,
                    row.outcome.dtype,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, discharge_plan());
        assert_eq!(
            table.std_graph_digest,
            crate::std_graph::normal_cdf_graph_digest(&chelis_std_bundle::EMBEDDED_RUNTIME)
                .unwrap()
        );
        assert_eq!(render_discharge_table(table), DISCHARGE_TABLE_JSON);
    }

    #[test]
    fn chelis_2957_registry_reads_the_table_and_never_fuzzes() {
        RECOMPUTED.with(|count| count.set(0));
        let registry = standard_contract_registry(
            &CONTRACT_FLOAT_WIDTHS,
            &chelis_std_bundle::EMBEDDED_RUNTIME,
        );
        let probe = registry.probe_consumer(
            "consumer",
            CompositeVerdict::Proven,
            [NORMAL_CDF_REFLECTION, EXP_POSITIVITY, QUANTILE_RANGE],
        );
        assert!(
            probe
                .assumptions
                .iter()
                .all(|record| record.discharge.is_some())
        );
        assert_eq!(RECOMPUTED.with(std::cell::Cell::get), 0);
    }

    #[test]
    fn chelis_2957_registry_resolves_only_the_reached_contracts() {
        let resolved = |contracts: &[&str], widths: &[Prim]| {
            RESOLVED_ROWS.with(|rows| rows.borrow_mut().clear());
            let reached = contracts.iter().map(|id| id.to_string()).collect();
            let registry = standard_contract_registry_for(
                Some(&reached),
                widths,
                None,
                &chelis_std_bundle::EMBEDDED_RUNTIME,
            );
            let probe = registry.probe_consumer("consumer", CompositeVerdict::Proven, contracts);
            assert!(
                probe
                    .assumptions
                    .iter()
                    .all(|record| record.discharge.is_some())
            );
            RESOLVED_ROWS.with(|rows| rows.borrow().clone())
        };
        // No float contract reached: no float discharge is resolved.
        assert_eq!(
            resolved(&[QUANTILE_RANGE], &[]),
            vec![format!("{QUANTILE_RANGE}@untyped")]
        );
        assert!(resolved(&[EXP_ZERO], &[]).is_empty());
        assert!(resolved(&[], &[]).is_empty());
        assert_eq!(
            resolved(&[NORMAL_CDF_REFLECTION], &[Prim::F32]),
            vec![format!("{NORMAL_CDF_REFLECTION}@f32")]
        );
        // The unfiltered registry resolves every row of the plan.
        RESOLVED_ROWS.with(|rows| rows.borrow_mut().clear());
        let _ = standard_contract_registry(
            &CONTRACT_FLOAT_WIDTHS,
            &chelis_std_bundle::EMBEDDED_RUNTIME,
        );
        assert_eq!(
            RESOLVED_ROWS.with(|rows| rows.borrow().len()),
            discharge_plan().len()
        );
    }

    #[test]
    fn chelis_2957_table_lookup_fails_closed() {
        let table = committed_discharge_table().unwrap();
        let shipped = || Ok(table.std_graph_digest.clone());
        let failed = |outcome: FuzzOutcome| {
            assert_eq!(outcome.checked_samples, 0);
            outcome.counterexample.expect("a failed outcome")["discharge_table"].clone()
        };
        // The committed row is served.
        let served = table_outcome(
            Ok(table),
            shipped,
            NORMAL_CDF_RANGE,
            8192,
            0xC0DF_2026,
            Some(Prim::F64),
        );
        assert!(served.counterexample.is_none());
        // A sample plan the table does not hold.
        let reason = failed(table_outcome(
            Ok(table),
            shipped,
            NORMAL_CDF_RANGE,
            8191,
            0xC0DF_2026,
            Some(Prim::F64),
        ));
        assert!(reason.as_str().unwrap().contains("has no row"), "{reason}");
        // A width the table does not hold for an untyped contract.
        failed(table_outcome(
            Ok(table),
            shipped,
            QUANTILE_RANGE,
            8192,
            0xCA_2026,
            Some(Prim::F64),
        ));
        // A normal_cdf row recorded against another shipped graph.
        let other = || Ok("0".repeat(64));
        let reason = failed(table_outcome(
            Ok(table),
            other,
            NORMAL_CDF_RANGE,
            8192,
            0xC0DF_2026,
            Some(Prim::F64),
        ));
        assert_eq!(reason["shipped_std_graph_digest"], "0".repeat(64));
        // The digest does not gate a row that does not read the graph.
        let exp = table_outcome(
            Ok(table),
            other,
            EXP_POSITIVITY,
            4096,
            0xE0_2026,
            Some(Prim::F64),
        );
        assert!(exp.counterexample.is_none());
        // An unreadable table.
        failed(table_outcome(
            Err("broken".to_string()),
            shipped,
            EXP_POSITIVITY,
            4096,
            0xE0_2026,
            Some(Prim::F64),
        ));
    }
}
