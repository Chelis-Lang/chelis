//! Composition records and weakest-link verdict rollup.
//!
//! COMPOSE keeps producer obligations as first-class assumption discharges.
//! CONTRACT can later wire concrete contract artifacts into the same
//! registry/probe surface without changing the CLI JSON shape.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::tier_b::AssumptionSatisfiability;

/// Fuzz equality tolerance used by the concrete predicate evaluator.
pub const FUZZ_TOLERANCE: f64 = 1e-10;

/// How an assumption was discharged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DischargeMethod {
    Smt,
    Fuzz,
    Axiom,
}

impl DischargeMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            DischargeMethod::Smt => "smt",
            DischargeMethod::Fuzz => "fuzz",
            DischargeMethod::Axiom => "axiom",
        }
    }
}

/// The composed proof verdict, after folding the base proof together with
/// every assumption discharge it depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompositeVerdict {
    #[default]
    Proven,
    ProvenModuloFuzzValidatedContract,
    ProvenModuloAssertedAxiom,
    Invalid,
    Unsupported,
    Failed,
}

impl CompositeVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            CompositeVerdict::Proven => "proven",
            CompositeVerdict::ProvenModuloFuzzValidatedContract => {
                "proven_modulo_fuzz_validated_contract"
            }
            CompositeVerdict::ProvenModuloAssertedAxiom => "proven_modulo_asserted_axiom",
            CompositeVerdict::Invalid => "invalid",
            CompositeVerdict::Unsupported => "unsupported",
            CompositeVerdict::Failed => "failed",
        }
    }

    fn weakness_rank(self) -> u8 {
        match self {
            CompositeVerdict::Proven => 0,
            CompositeVerdict::ProvenModuloFuzzValidatedContract => 1,
            CompositeVerdict::ProvenModuloAssertedAxiom => 2,
            CompositeVerdict::Unsupported => 3,
            CompositeVerdict::Invalid => 4,
            CompositeVerdict::Failed => 5,
        }
    }

    fn weakest(self, other: Self) -> Self {
        if other.weakness_rank() > self.weakness_rank() {
            other
        } else {
            self
        }
    }
}

/// Evidence that an assumption was discharged. The required wire shape is
/// exactly `{method, evidence}`; status/counterexample/details live inside
/// `evidence` so this remains additive as producers grow richer evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssumptionDischarge {
    pub method: DischargeMethod,
    pub evidence: serde_json::Value,
}

impl AssumptionDischarge {
    pub fn new(method: DischargeMethod, evidence: serde_json::Value) -> Self {
        Self { method, evidence }
    }

    fn verdict_contribution(&self) -> CompositeVerdict {
        match self
            .evidence
            .get("status")
            .and_then(serde_json::Value::as_str)
        {
            Some("proved") if self.method == DischargeMethod::Smt => CompositeVerdict::Proven,
            Some("validated") if self.method == DischargeMethod::Fuzz => {
                if has_fuzz_evidence(&self.evidence) {
                    CompositeVerdict::ProvenModuloFuzzValidatedContract
                } else {
                    CompositeVerdict::Unsupported
                }
            }
            Some("asserted") if self.method == DischargeMethod::Axiom => {
                if self.evidence.get("justification").is_some() {
                    CompositeVerdict::ProvenModuloAssertedAxiom
                } else {
                    CompositeVerdict::Unsupported
                }
            }
            Some("failed") => {
                if self.evidence.get("counterexample").is_some() {
                    CompositeVerdict::Failed
                } else {
                    // Failed without a counterexample is not a sound
                    // falsification. Degrade to unsupported instead.
                    CompositeVerdict::Unsupported
                }
            }
            Some("invalid") => CompositeVerdict::Invalid,
            Some("unsupported" | "error") => CompositeVerdict::Unsupported,
            _ => CompositeVerdict::Unsupported,
        }
    }
}

fn has_fuzz_evidence(evidence: &serde_json::Value) -> bool {
    evidence.get("samples").is_some()
        && evidence.get("seed").is_some()
        && evidence.get("tolerance").is_some()
}

/// Result of checking the assumptions alone for satisfiability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonVacuityStatus {
    Established,
    Invalid,
    Unsupported,
}

/// Non-vacuity evidence for an assumption set. SAT establishes
/// non-vacuity; UNSAT makes the composed proof invalid; unknown/timeout is
/// unsupported, never a green result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonVacuityRecord {
    pub status: NonVacuityStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub evidence: serde_json::Value,
}

impl NonVacuityRecord {
    pub fn established(evidence: serde_json::Value) -> Self {
        Self {
            status: NonVacuityStatus::Established,
            reason: None,
            evidence,
        }
    }

    pub fn invalid(reason: impl Into<String>, evidence: serde_json::Value) -> Self {
        Self {
            status: NonVacuityStatus::Invalid,
            reason: Some(reason.into()),
            evidence,
        }
    }

    pub fn unsupported(reason: impl Into<String>, evidence: serde_json::Value) -> Self {
        Self {
            status: NonVacuityStatus::Unsupported,
            reason: Some(reason.into()),
            evidence,
        }
    }

    fn verdict_contribution(&self) -> CompositeVerdict {
        match self.status {
            NonVacuityStatus::Established => CompositeVerdict::Proven,
            NonVacuityStatus::Invalid => CompositeVerdict::Invalid,
            NonVacuityStatus::Unsupported => CompositeVerdict::Unsupported,
        }
    }
}

impl From<AssumptionSatisfiability> for NonVacuityRecord {
    fn from(result: AssumptionSatisfiability) -> Self {
        match result {
            AssumptionSatisfiability::Sat(model) => {
                NonVacuityRecord::established(serde_json::json!({
                    "solver": "cvc5",
                    "result": "sat",
                    "model": model,
                }))
            }
            AssumptionSatisfiability::Unsat => NonVacuityRecord::invalid(
                "assumptions are jointly unsatisfiable",
                serde_json::json!({"solver": "cvc5", "result": "unsat"}),
            ),
            AssumptionSatisfiability::Timeout => NonVacuityRecord::unsupported(
                "assumption satisfiability check timed out",
                serde_json::json!({"solver": "cvc5", "result": "timeout"}),
            ),
            AssumptionSatisfiability::Unknown => NonVacuityRecord::unsupported(
                "assumption satisfiability is unknown",
                serde_json::json!({"solver": "cvc5", "result": "unknown"}),
            ),
            AssumptionSatisfiability::Error(reason) => NonVacuityRecord::unsupported(
                reason,
                serde_json::json!({"solver": "cvc5", "result": "error"}),
            ),
        }
    }
}

/// One assumption the proof depends on, with its discharge and
/// non-vacuity records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssumptionRecord {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discharge: Option<AssumptionDischarge>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_vacuity: Option<NonVacuityRecord>,
}

impl AssumptionRecord {
    pub fn new(
        name: impl Into<String>,
        discharge: Option<AssumptionDischarge>,
        non_vacuity: Option<NonVacuityRecord>,
    ) -> Self {
        Self {
            name: name.into(),
            source_type: None,
            producer: None,
            discharge,
            non_vacuity,
        }
    }

    pub fn with_source(
        mut self,
        source_type: impl Into<String>,
        producer: impl Into<String>,
    ) -> Self {
        self.source_type = Some(source_type.into());
        self.producer = Some(producer.into());
        self
    }

    pub fn missing(name: impl Into<String>) -> Self {
        Self::new(name, None, None)
    }

    fn verdict_contribution(&self) -> CompositeVerdict {
        let discharge_verdict = self
            .discharge
            .as_ref()
            .map(AssumptionDischarge::verdict_contribution)
            .unwrap_or(CompositeVerdict::Unsupported);
        let non_vacuity_verdict = self
            .non_vacuity
            .as_ref()
            .map(NonVacuityRecord::verdict_contribution)
            .unwrap_or(CompositeVerdict::Unsupported);
        discharge_verdict.weakest(non_vacuity_verdict)
    }
}

/// Roll the base proof together with all assumption records.
pub fn rollup_composite(
    base: CompositeVerdict,
    assumptions: &[AssumptionRecord],
) -> CompositeVerdict {
    assumptions
        .iter()
        .fold(base, |acc, a| acc.weakest(a.verdict_contribution()))
}

/// Dependency-sensitivity probe for CONTRACT. A consumer proof names the
/// assumption keys it depends on; removing or corrupting a registered
/// discharge must change this probe's verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositionProbe {
    pub consumer: String,
    pub assumptions: Vec<AssumptionRecord>,
    pub composite_verdict: CompositeVerdict,
}

#[derive(Debug, Clone, Default)]
pub struct AssumptionRegistry {
    records: BTreeMap<String, AssumptionRecord>,
}

impl AssumptionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, record: AssumptionRecord) {
        self.records.insert(record.name.clone(), record);
    }

    pub fn remove(&mut self, name: &str) -> Option<AssumptionRecord> {
        self.records.remove(name)
    }

    pub fn probe_consumer<I, S>(
        &self,
        consumer: impl Into<String>,
        base: CompositeVerdict,
        dependencies: I,
    ) -> CompositionProbe
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let assumptions = dependencies
            .into_iter()
            .map(|dep| {
                let key = dep.as_ref();
                self.records
                    .get(key)
                    .cloned()
                    .unwrap_or_else(|| AssumptionRecord::missing(key))
            })
            .collect::<Vec<_>>();
        let composite_verdict = rollup_composite(base, &assumptions);
        CompositionProbe {
            consumer: consumer.into(),
            assumptions,
            composite_verdict,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn established() -> NonVacuityRecord {
        NonVacuityRecord::established(json!({"solver": "cvc5", "result": "sat"}))
    }

    fn assumption(
        name: &str,
        method: DischargeMethod,
        evidence: serde_json::Value,
    ) -> AssumptionRecord {
        AssumptionRecord::new(
            name,
            Some(AssumptionDischarge::new(method, evidence)),
            Some(established()),
        )
    }

    #[test]
    fn c1_all_smt_discharges_roll_up_to_proven() {
        let assumptions = vec![
            assumption(
                "contract:a",
                DischargeMethod::Smt,
                json!({"status": "proved"}),
            ),
            assumption(
                "contract:b",
                DischargeMethod::Smt,
                json!({"status": "proved"}),
            ),
        ];
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &assumptions),
            CompositeVerdict::Proven
        );
    }

    #[test]
    fn c2_fuzz_discharge_qualifies_the_composite_with_seed_and_tolerance() {
        let assumptions = vec![assumption(
            "contract:a",
            DischargeMethod::Fuzz,
            json!({"status": "validated", "samples": 64, "seed": 7, "tolerance": FUZZ_TOLERANCE}),
        )];
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &assumptions),
            CompositeVerdict::ProvenModuloFuzzValidatedContract
        );
        let evidence = &assumptions[0].discharge.as_ref().unwrap().evidence;
        assert_eq!(evidence["seed"], 7);
        assert_eq!(evidence["tolerance"], FUZZ_TOLERANCE);
    }

    #[test]
    fn c3_removed_or_failing_discharge_degrades_the_consumer_probe() {
        let mut registry = AssumptionRegistry::new();
        registry.insert(assumption(
            "contract:a",
            DischargeMethod::Smt,
            json!({"status": "proved"}),
        ));
        assert_eq!(
            registry
                .probe_consumer("consumer", CompositeVerdict::Proven, ["contract:a"])
                .composite_verdict,
            CompositeVerdict::Proven
        );

        registry.remove("contract:a");
        assert_eq!(
            registry
                .probe_consumer("consumer", CompositeVerdict::Proven, ["contract:a"])
                .composite_verdict,
            CompositeVerdict::Unsupported
        );

        registry.insert(assumption(
            "contract:a",
            DischargeMethod::Smt,
            json!({"status": "failed", "counterexample": {"x": "2"}}),
        ));
        assert_eq!(
            registry
                .probe_consumer("consumer", CompositeVerdict::Proven, ["contract:a"])
                .composite_verdict,
            CompositeVerdict::Failed
        );
    }

    #[test]
    fn asserted_axiom_is_qualified_not_pure_proven() {
        let assumptions = vec![assumption(
            "contract:a",
            DischargeMethod::Axiom,
            json!({"status": "asserted", "justification": "mathematical axiom"}),
        )];
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &assumptions),
            CompositeVerdict::ProvenModuloAssertedAxiom
        );
    }

    #[test]
    fn malformed_discharge_evidence_is_unsupported() {
        for (method, evidence) in [
            (DischargeMethod::Smt, json!({})),
            (DischargeMethod::Smt, json!({"status": "validated"})),
            (DischargeMethod::Fuzz, json!({"status": "validated"})),
            (DischargeMethod::Axiom, json!({"status": "asserted"})),
        ] {
            let assumptions = vec![assumption("contract:bad", method, evidence)];
            assert_eq!(
                rollup_composite(CompositeVerdict::Proven, &assumptions),
                CompositeVerdict::Unsupported
            );
        }
    }

    #[test]
    fn c4_unsatisfiable_assumption_set_invalidates_the_composite() {
        let mut record = assumption(
            "contract:contradiction",
            DischargeMethod::Smt,
            json!({"status": "proved"}),
        );
        record.non_vacuity = Some(NonVacuityRecord::from(AssumptionSatisfiability::Unsat));
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &[record]),
            CompositeVerdict::Invalid
        );
    }

    #[test]
    fn non_vacuity_unknown_is_unsupported_not_sat() {
        let mut record = assumption(
            "contract:hard",
            DischargeMethod::Smt,
            json!({"status": "proved"}),
        );
        record.non_vacuity = Some(NonVacuityRecord::from(AssumptionSatisfiability::Unknown));
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &[record]),
            CompositeVerdict::Unsupported
        );
    }
}
