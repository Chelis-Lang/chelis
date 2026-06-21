//! Composition records and weakest-link verdict rollup.
//!
//! COMPOSE keeps producer obligations as first-class assumption discharges.
//! CONTRACT can later wire concrete contract artifacts into the same
//! registry/probe surface without changing the CLI JSON shape.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::discharge::{Qualifier, QualifierSet, Soundness};
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

    /// The canonical discharging-engine name for this method (WI-8). cvc5 is
    /// the SMT engine; the fuzz sampler discharges fuzz-validated assumptions;
    /// an axiom is asserted, not discharged by an engine.
    pub fn engine(self) -> &'static str {
        match self {
            DischargeMethod::Smt => "cvc5",
            DischargeMethod::Fuzz => "fuzz-sampler",
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

    /// The lattice element this badge stands for: a green badge is a point in
    /// the `(soundness, qualifier_set)` lattice; the three non-green badges are
    /// terminal outcomes off that lattice (WI-6). This is the inverse of
    /// [`VerdictRollup::badge`] on the green badges and the canonical-element
    /// representative on the rest, so a rollup that begins from a base badge
    /// (the common `rollup_composite(base, ..)` shape) re-enters the lattice
    /// through the same door every discharge does.
    fn guarantee(self) -> VerdictGuarantee {
        match self {
            CompositeVerdict::Proven => VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::new(),
            },
            CompositeVerdict::ProvenModuloFuzzValidatedContract => VerdictGuarantee::Green {
                soundness: Soundness::Empirical,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Fuzz]),
            },
            CompositeVerdict::ProvenModuloAssertedAxiom => VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Axiom]),
            },
            CompositeVerdict::Unsupported => VerdictGuarantee::Unestablished,
            CompositeVerdict::Invalid => VerdictGuarantee::Vacuous,
            CompositeVerdict::Failed => VerdictGuarantee::Disproved,
        }
    }
}

/// One discharge's guarantee as a lattice element (WI-6). A discharge that
/// supports a green verdict contributes a point in the `(soundness,
/// qualifier_set)` lattice; the three non-green contributions are terminal
/// outcomes that sit off the lattice (the property was disproved, the
/// assumption set is vacuous, or the discharge established nothing). The
/// guarantee kinds inside `qualifiers` are incomparable on one axis, so the
/// rollup is the UNION of the sets plus the MINIMUM soundness, never a
/// collapse of incomparable kinds onto a single chain.
#[derive(Debug, Clone, PartialEq, Eq)]
enum VerdictGuarantee {
    /// A green contribution: trustworthy at `soundness`, carrying `qualifiers`.
    Green {
        soundness: Soundness,
        qualifiers: QualifierSet,
    },
    /// Nothing was established (timeout, unknown, malformed evidence). Renders
    /// `Unsupported`, never a green.
    Unestablished,
    /// The assumption set is vacuous (jointly unsatisfiable). Renders
    /// `Invalid`: a proof under contradictory assumptions is unsound.
    Vacuous,
    /// The property was disproved with a counterexample. Renders `Failed`.
    Disproved,
}

/// The rolled-up verdict over a dependency set (WI-6): the union of every
/// green discharge's qualifier set and the minimum soundness across them, plus
/// the dominating terminal outcome if any discharge disproved, invalidated, or
/// failed to establish. The badge is a PROJECTION of this rollup, computed
/// from the rolled-up set, so a weak guarantee in the union cannot be
/// laundered into a strong badge.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VerdictRollup {
    /// Minimum soundness across the green contributions (weakest-link). Starts
    /// at [`Soundness::Exact`] (the identity) and only ever drops.
    soundness: Soundness,
    /// Union of every green contribution's qualifier set.
    qualifiers: QualifierSet,
    /// The strongest-dominating terminal outcome seen, if any. `Disproved`
    /// dominates `Vacuous` dominates `Unestablished`, matching the legacy
    /// `Failed > Invalid > Unsupported` precedence.
    terminal: Option<Terminal>,
}

/// A terminal (non-green) outcome, ordered weakest-dominating last so a fold
/// can keep the dominating one. `Disproved` dominates `Vacuous` dominates
/// `Unestablished`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Terminal {
    Unestablished,
    Vacuous,
    Disproved,
}

impl VerdictRollup {
    /// The neutral element: a green rollup at the strongest soundness with no
    /// qualifiers and no green contribution yet. Folding `Proven`-equivalent
    /// guarantees onto it leaves it `Proven`.
    fn identity() -> Self {
        Self {
            soundness: Soundness::Exact,
            qualifiers: QualifierSet::new(),
            terminal: None,
        }
    }

    /// Fold one guarantee into the rollup: terminal outcomes keep the
    /// dominating one; green contributions take the minimum soundness and the
    /// union of qualifiers (the WI-6 rollup primitives).
    fn fold(mut self, guarantee: VerdictGuarantee) -> Self {
        match guarantee {
            VerdictGuarantee::Green {
                soundness,
                qualifiers,
            } => {
                self.soundness = self.soundness.min(soundness);
                self.qualifiers = self.qualifiers.union(&qualifiers);
            }
            VerdictGuarantee::Unestablished => {
                self.terminal = Some(
                    self.terminal
                        .map_or(Terminal::Unestablished, |t| t.max(Terminal::Unestablished)),
                );
            }
            VerdictGuarantee::Vacuous => {
                self.terminal = Some(
                    self.terminal
                        .map_or(Terminal::Vacuous, |t| t.max(Terminal::Vacuous)),
                );
            }
            VerdictGuarantee::Disproved => {
                self.terminal = Some(
                    self.terminal
                        .map_or(Terminal::Disproved, |t| t.max(Terminal::Disproved)),
                );
            }
        }
        self
    }

    /// Project the rollup to its rendered [`CompositeVerdict`] badge. A
    /// terminal outcome dominates any green guarantee (a disproof or a vacuous
    /// assumption set is never a green); otherwise the badge is read off the
    /// UNION'd qualifier set, weakest-kind first, so a set that mixes `fuzz`
    /// (or any weak kind) with `exact` renders the WEAK badge, never `proven`.
    fn badge(&self) -> CompositeVerdict {
        if let Some(terminal) = self.terminal {
            return match terminal {
                Terminal::Disproved => CompositeVerdict::Failed,
                Terminal::Vacuous => CompositeVerdict::Invalid,
                Terminal::Unestablished => CompositeVerdict::Unsupported,
            };
        }
        // All-green. The qualifier set drives the badge, weakest kind first:
        // an asserted axiom is a weaker green than a fuzz-validated contract,
        // which is weaker than an exact proof, matching the legacy precedence.
        // Covered-or-rejected: only `Exact` soundness with no weaker qualifier
        // may render `Proven`. A green rollup at sub-`Exact` soundness carrying
        // a non-fuzz/non-axiom qualifier (`SoundOverApproximation`,
        // `DeltaComplete`, `SpecialFunctionCertified`, `CertificateBearing`)
        // has no green badge yet, so it conservatively renders `Unsupported`
        // rather than laundering into `Proven`. The proper sound-over-
        // approximation green badge and its lattice projection land with WI-9
        // when a producer first emits those qualifiers; until then any such
        // rollup is rejected from the green path here at the rendering point.
        if self.qualifiers.contains(Qualifier::Axiom) {
            CompositeVerdict::ProvenModuloAssertedAxiom
        } else if self.qualifiers.contains(Qualifier::Fuzz) {
            CompositeVerdict::ProvenModuloFuzzValidatedContract
        } else if self.soundness == Soundness::Exact {
            CompositeVerdict::Proven
        } else {
            CompositeVerdict::Unsupported
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

    /// The lattice guarantee this discharge contributes (WI-6). An SMT proof is
    /// an exact green; a validated fuzz discharge is an empirical green
    /// carrying the `fuzz` qualifier; an asserted axiom is an exact-trust green
    /// carrying the `axiom` qualifier; everything else is a terminal outcome
    /// (disproof, or nothing established). The badge is later projected from
    /// the rolled-up union, so a fuzz qualifier here cannot be laundered into a
    /// `proven` badge once it joins an otherwise-exact set.
    fn guarantee(&self) -> VerdictGuarantee {
        match self
            .evidence
            .get("status")
            .and_then(serde_json::Value::as_str)
        {
            Some("proved") if self.method == DischargeMethod::Smt => VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Exact]),
            },
            Some("validated") if self.method == DischargeMethod::Fuzz => {
                if has_fuzz_evidence(&self.evidence) {
                    VerdictGuarantee::Green {
                        soundness: Soundness::Empirical,
                        qualifiers: QualifierSet::from_iter_kinds([Qualifier::Fuzz]),
                    }
                } else {
                    VerdictGuarantee::Unestablished
                }
            }
            Some("asserted") if self.method == DischargeMethod::Axiom => {
                if self.evidence.get("justification").is_some() {
                    VerdictGuarantee::Green {
                        soundness: Soundness::Exact,
                        qualifiers: QualifierSet::from_iter_kinds([Qualifier::Axiom]),
                    }
                } else {
                    VerdictGuarantee::Unestablished
                }
            }
            Some("failed") => {
                if self.evidence.get("counterexample").is_some() {
                    VerdictGuarantee::Disproved
                } else {
                    // Failed without a counterexample is not a sound
                    // falsification. Degrade to unsupported instead.
                    VerdictGuarantee::Unestablished
                }
            }
            Some("invalid") => VerdictGuarantee::Vacuous,
            Some("unsupported" | "error") => VerdictGuarantee::Unestablished,
            _ => VerdictGuarantee::Unestablished,
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

    /// The lattice guarantee this non-vacuity record contributes (WI-6).
    /// Established non-vacuity is a neutral exact green (it adds no qualifier,
    /// only confirms the assumed domain is inhabited); an unsatisfiable
    /// assumption set is `Vacuous` (Invalid); an unknown/timeout established
    /// nothing.
    fn guarantee(&self) -> VerdictGuarantee {
        match self.status {
            NonVacuityStatus::Established => VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::new(),
            },
            NonVacuityStatus::Invalid => VerdictGuarantee::Vacuous,
            NonVacuityStatus::Unsupported => VerdictGuarantee::Unestablished,
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

/// The prover-stamped discharge provenance of an assumption (WI-8): which
/// engine discharged it and with what guarantee kind, keyed to the source
/// identity it was discharged against. This is stamped PROVER-side at discharge
/// time, where the engine and the guarantee are known. c-earchin (the bridge)
/// emits only source identity and never a tier: it cannot compute which engine
/// closed the goal or with what guarantee, so the tier is the prover's to
/// stamp, and the artifact joins this tier to the c-earchin source id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DischargeTier {
    /// The engine that discharged the assumption (e.g. `cvc5` for an SMT
    /// proof, `fuzz-sampler` for a fuzz-validated discharge).
    pub engine: String,
    /// The guarantee kind the discharge carries, the canonical
    /// [`DischargeMethod`] spelling (`smt` / `fuzz` / `axiom`).
    pub guarantee: String,
    /// The source identity this tier is keyed to: the c-earchin source id for a
    /// bridge property, or the binder / producer identity for an injected or
    /// derived assumption. The artifact joins the tier to this source. `None`
    /// when the discharge has no distinct source identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl DischargeTier {
    /// Stamp a tier from the prover side. `engine` is the discharging engine,
    /// `method` the guarantee kind, `source` the identity it is keyed to.
    pub fn new(engine: impl Into<String>, method: DischargeMethod, source: Option<String>) -> Self {
        Self {
            engine: engine.into(),
            guarantee: method.as_str().to_string(),
            source,
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
    /// Prover-stamped discharge provenance (WI-8): which engine discharged this
    /// assumption and with what guarantee, keyed to its source identity. Serde-
    /// additive (`skip_serializing_if`) so existing artifacts stay
    /// byte-identical and old payloads still deserialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discharge_tier: Option<DischargeTier>,
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
            discharge_tier: None,
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

    /// Stamp the prover-side discharge tier (WI-8). The tier records which
    /// engine discharged this assumption and with what guarantee, keyed to the
    /// given source identity.
    pub fn with_discharge_tier(mut self, tier: DischargeTier) -> Self {
        self.discharge_tier = Some(tier);
        self
    }

    pub fn missing(name: impl Into<String>) -> Self {
        Self::new(name, None, None)
    }

    /// Fold this record's two guarantees (its discharge and its non-vacuity
    /// evidence) into the rollup. A missing discharge or non-vacuity record
    /// contributes `Unestablished`: an assumption with no recorded discharge
    /// established nothing and cannot support a green.
    fn fold_into(&self, rollup: VerdictRollup) -> VerdictRollup {
        let discharge = self
            .discharge
            .as_ref()
            .map(AssumptionDischarge::guarantee)
            .unwrap_or(VerdictGuarantee::Unestablished);
        let non_vacuity = self
            .non_vacuity
            .as_ref()
            .map(NonVacuityRecord::guarantee)
            .unwrap_or(VerdictGuarantee::Unestablished);
        rollup.fold(discharge).fold(non_vacuity)
    }
}

/// Roll the base proof together with all assumption records (WI-6). The base
/// badge re-enters the `(soundness, qualifier_set)` lattice through the same
/// door every discharge does, and the rolled-up badge is PROJECTED from the
/// union of qualifiers and the minimum soundness, so a weak guarantee among
/// the dependencies cannot be laundered into a strong badge.
pub fn rollup_composite(
    base: CompositeVerdict,
    assumptions: &[AssumptionRecord],
) -> CompositeVerdict {
    let rollup = assumptions.iter().fold(
        VerdictRollup::identity().fold(base.guarantee()),
        |acc, a| a.fold_into(acc),
    );
    rollup.badge()
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

    // --- WI-6 qualifier-set verdict lattice (integrity-critical) ---

    /// The legacy precedence the lattice projection MUST reproduce, by which a
    /// fold of two badges takes the weaker. This is the byte-identity oracle:
    /// the rolled-up badge of any two badges (each re-entering the lattice via
    /// `guarantee()`) must equal the legacy weakest-of-the-two for every pair.
    fn legacy_weakness_rank(verdict: CompositeVerdict) -> u8 {
        match verdict {
            CompositeVerdict::Proven => 0,
            CompositeVerdict::ProvenModuloFuzzValidatedContract => 1,
            CompositeVerdict::ProvenModuloAssertedAxiom => 2,
            CompositeVerdict::Unsupported => 3,
            CompositeVerdict::Invalid => 4,
            CompositeVerdict::Failed => 5,
        }
    }

    const ALL_VERDICTS: [CompositeVerdict; 6] = [
        CompositeVerdict::Proven,
        CompositeVerdict::ProvenModuloFuzzValidatedContract,
        CompositeVerdict::ProvenModuloAssertedAxiom,
        CompositeVerdict::Unsupported,
        CompositeVerdict::Invalid,
        CompositeVerdict::Failed,
    ];

    #[test]
    fn wi6_lattice_projection_reproduces_legacy_weakest_for_every_badge_pair() {
        // The lattice rollup must be byte-identical to the retired scalar
        // `weakest()` fold across the whole 6x6 badge matrix, so the existing
        // prove corpus stays unchanged. We fold two guarantees directly
        // through the rollup and compare to the weaker-of-the-two legacy rank.
        for a in ALL_VERDICTS {
            for b in ALL_VERDICTS {
                let rolled = VerdictRollup::identity()
                    .fold(a.guarantee())
                    .fold(b.guarantee())
                    .badge();
                let legacy = if legacy_weakness_rank(b) > legacy_weakness_rank(a) {
                    b
                } else {
                    a
                };
                assert_eq!(
                    rolled, legacy,
                    "lattice rollup of {a:?} and {b:?} must match legacy weakest"
                );
            }
        }
    }

    #[test]
    fn wi6_homogeneous_exact_set_rolls_up_to_proven() {
        // Positive twin: an all-exact (SMT-proved) dependency set rolls up to
        // the strong `proven` badge.
        let assumptions = vec![
            assumption("a", DischargeMethod::Smt, json!({"status": "proved"})),
            assumption("b", DischargeMethod::Smt, json!({"status": "proved"})),
        ];
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &assumptions),
            CompositeVerdict::Proven
        );
    }

    #[test]
    fn wi6_mixed_fuzz_and_exact_set_does_not_launder_into_proven() {
        // The integrity test that matters most: a set mixing a fuzz-validated
        // discharge with an exact (SMT) discharge rolls up to a qualifier set
        // containing BOTH kinds with the MINIMUM soundness, and MUST render the
        // weak (modulo-fuzz) badge, NOT the strong `proven`. A weak guarantee
        // cannot be laundered into a strong one through the rollup.
        let exact = assumption("exact", DischargeMethod::Smt, json!({"status": "proved"}));
        let fuzz = assumption(
            "fuzz",
            DischargeMethod::Fuzz,
            json!({"status": "validated", "samples": 64, "seed": 1, "tolerance": FUZZ_TOLERANCE}),
        );

        let rolled = rollup_composite(CompositeVerdict::Proven, &[exact.clone(), fuzz.clone()]);
        assert_ne!(
            rolled,
            CompositeVerdict::Proven,
            "a fuzz+exact mix must not render as the strong proven badge"
        );
        assert_eq!(rolled, CompositeVerdict::ProvenModuloFuzzValidatedContract);

        // Order-independence: the union is commutative, so swapping the set
        // order cannot launder the weak guarantee away either.
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &[fuzz, exact]),
            CompositeVerdict::ProvenModuloFuzzValidatedContract
        );

        // The rolled-up lattice element itself carries BOTH qualifiers at the
        // minimum (empirical) soundness, proving the union is not collapsed.
        let rollup = VerdictRollup::identity()
            .fold(CompositeVerdict::Proven.guarantee())
            .fold(VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Exact]),
            })
            .fold(VerdictGuarantee::Green {
                soundness: Soundness::Empirical,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Fuzz]),
            });
        assert!(rollup.qualifiers.contains(Qualifier::Exact));
        assert!(rollup.qualifiers.contains(Qualifier::Fuzz));
        assert_eq!(rollup.soundness, Soundness::Empirical);
    }

    #[test]
    fn wi6_terminal_outcome_dominates_any_green_in_the_set() {
        // A disproved or vacuous contribution is never a green: it dominates
        // every green guarantee in the union regardless of order.
        let proved = assumption("p", DischargeMethod::Smt, json!({"status": "proved"}));
        let disproved = assumption(
            "d",
            DischargeMethod::Smt,
            json!({"status": "failed", "counterexample": {"x": 1}}),
        );
        assert_eq!(
            rollup_composite(
                CompositeVerdict::Proven,
                &[proved.clone(), disproved.clone()]
            ),
            CompositeVerdict::Failed
        );

        let mut vacuous = assumption("v", DischargeMethod::Smt, json!({"status": "proved"}));
        vacuous.non_vacuity = Some(NonVacuityRecord::from(AssumptionSatisfiability::Unsat));
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &[proved, vacuous]),
            CompositeVerdict::Invalid
        );
    }

    #[test]
    fn wi6_sub_exact_non_fuzz_non_axiom_green_does_not_launder_into_proven() {
        // Covered-or-rejected at the badge-rendering point: a green rollup at
        // sub-Exact soundness carrying a qualifier that is neither Fuzz nor
        // Axiom (here SoundOverApproximation) has no green badge yet, so it must
        // render Unsupported -- NOT Proven. Previously the terminal-free
        // projection special-cased only Axiom and Fuzz, then fell through to
        // Proven, laundering these kinds into the strong badge. The proper
        // green badge for these lands with WI-9; until then they are rejected
        // from the green path here.
        let rollup = VerdictRollup::identity().fold(VerdictGuarantee::Green {
            soundness: Soundness::SoundApproximate,
            qualifiers: QualifierSet::from_iter_kinds([Qualifier::SoundOverApproximation]),
        });
        assert_ne!(
            rollup.badge(),
            CompositeVerdict::Proven,
            "a sub-Exact sound-over-approximation green must not launder into proven"
        );
        assert_eq!(
            rollup.badge(),
            CompositeVerdict::Unsupported,
            "a sub-Exact non-fuzz/non-axiom green has no green badge yet and must \
             render Unsupported, not a strong proven"
        );

        // The other producer qualifiers WI-9 will emit (DeltaComplete,
        // SpecialFunctionCertified, CertificateBearing) take the same
        // conservative rejection at sub-Exact soundness rather than laundering.
        for qualifier in [
            Qualifier::DeltaComplete,
            Qualifier::SpecialFunctionCertified,
            Qualifier::CertificateBearing,
        ] {
            let rolled = VerdictRollup::identity().fold(VerdictGuarantee::Green {
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([qualifier]),
            });
            assert_eq!(
                rolled.badge(),
                CompositeVerdict::Unsupported,
                "sub-Exact green carrying {qualifier:?} must render Unsupported, not proven"
            );
        }
    }
}
