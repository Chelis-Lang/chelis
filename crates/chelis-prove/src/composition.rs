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
    /// Discharged by Beacon's certified special-function envelope (chelis#674).
    /// The proof is machine-checked (Gappa/Arb) and sound, but it is an
    /// over-approximation — the envelope bounds the function, not equals it.
    CertifiedEnvelope,
}

impl DischargeMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            DischargeMethod::Smt => "smt",
            DischargeMethod::Fuzz => "fuzz",
            DischargeMethod::Axiom => "axiom",
            DischargeMethod::CertifiedEnvelope => "certified_envelope",
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
            DischargeMethod::CertifiedEnvelope => "beacon-envelope",
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
    /// An SMT base proved over the REALS, disclosing the machine-arithmetic
    /// gap (chelis#422). Under the current real-sorted lowering every
    /// `proof_tier:"smt"` green carries this; plain `Proven` is reserved for a
    /// future exact-machine-arithmetic lowering.
    ProvenModuloRealArithmetic,
    /// A proof discharged through a certified special-function ENVELOPE
    /// (chelis#434): the transcendental subterm was abstracted to a fresh
    /// variable bounded by its Sollya/Gappa/Arb-certified envelope, and the
    /// residual proved over reals. Distinct from — and strictly weaker than —
    /// `proven_modulo_real_arithmetic`: it discloses the ADDITIONAL dependency on
    /// the certified envelope's over-approximation. The certificate is
    /// machine-checked (sound), so it is STRONGER than the trusted/empirical
    /// `proven_modulo_asserted_axiom` / `proven_modulo_fuzz_validated_contract`.
    /// An envelope-FREE over-reals proof NEVER carries this token (it stays
    /// `proven_modulo_real_arithmetic`); this token requires the
    /// `SpecialFunctionCertified` qualifier.
    ProvenModuloCertifiedEnvelope,
    ProvenModuloFuzzValidatedContract,
    ProvenModuloAssertedAxiom,
    /// A sound over-approximation backed the base (e.g. Beacon's interval
    /// engine): a green-exit result that is conservative, strictly below a
    /// proof. Distinct from the `proven_*` badges (the base was not proved).
    SoundApproximate,
    /// The base was established by fuzz sampling only -- no deductive proof
    /// underneath (chelis#422). A green-exit empirical pass that is NOT proven:
    /// it must never read as `proven_*`. Distinct from
    /// [`CompositeVerdict::ProvenModuloFuzzValidatedContract`], where an exact
    /// SMT base is proven modulo a fuzz-validated contract assumption.
    FuzzValidatedEmpirical,
    /// An SMT base DISPROVED over the REALS, disclosing the machine-arithmetic
    /// gap on the failure side (chelis#422, symmetric to
    /// [`CompositeVerdict::ProvenModuloRealArithmetic`]). The solver found a
    /// counterexample over exact rationals, but under the real-sorted lowering
    /// it does not model machine arithmetic, so that counterexample may be a
    /// FALSE counterexample at runtime (e.g. `0.1 == 1.0 / 10.0` is disproved
    /// over the reals yet holds at f64). It is a HEDGED failure: it still
    /// DOMINATES every green (a possible counterexample must never be masked by a
    /// pass) but is weaker-certainty than a definite [`CompositeVerdict::Failed`]
    /// (a definite machine-arithmetic counterexample dominates a reals-hedged
    /// one), so it never overstates a hedged disproof as a definite failure.
    DisprovedModuloRealArithmetic,
    Invalid,
    Unsupported,
    Failed,
}

impl CompositeVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            CompositeVerdict::Proven => "proven",
            CompositeVerdict::ProvenModuloRealArithmetic => "proven_modulo_real_arithmetic",
            CompositeVerdict::ProvenModuloCertifiedEnvelope => "proven_modulo_certified_envelope",
            CompositeVerdict::ProvenModuloFuzzValidatedContract => {
                "proven_modulo_fuzz_validated_contract"
            }
            CompositeVerdict::ProvenModuloAssertedAxiom => "proven_modulo_asserted_axiom",
            CompositeVerdict::SoundApproximate => "sound_approximate",
            CompositeVerdict::FuzzValidatedEmpirical => "fuzz_validated",
            CompositeVerdict::DisprovedModuloRealArithmetic => "disproved_modulo_real_arithmetic",
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
            CompositeVerdict::ProvenModuloRealArithmetic => VerdictGuarantee::Green {
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::RealArith]),
            },
            CompositeVerdict::ProvenModuloCertifiedEnvelope => VerdictGuarantee::Green {
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::SpecialFunctionCertified]),
            },
            CompositeVerdict::ProvenModuloFuzzValidatedContract => VerdictGuarantee::Green {
                soundness: Soundness::Empirical,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Fuzz]),
            },
            CompositeVerdict::ProvenModuloAssertedAxiom => VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Axiom]),
            },
            CompositeVerdict::SoundApproximate => VerdictGuarantee::Green {
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::SoundOverApproximation]),
            },
            CompositeVerdict::FuzzValidatedEmpirical => VerdictGuarantee::Green {
                soundness: Soundness::Empirical,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::FuzzBase]),
            },
            CompositeVerdict::DisprovedModuloRealArithmetic => VerdictGuarantee::DisprovedOverReals,
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
    /// The property was disproved over the REALS (a counterexample exists over
    /// exact rationals), but the counterexample may be a false counterexample at
    /// machine arithmetic (the solver does not model it). A HEDGED failure: it
    /// dominates every green (a possible counterexample must not be masked by a
    /// pass) but loses to a definite [`VerdictGuarantee::Disproved`]. Renders
    /// `DisprovedModuloRealArithmetic`.
    DisprovedOverReals,
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
    /// The strongest-dominating terminal outcome seen, if any. A definite
    /// `Disproved` dominates a reals-hedged `DisprovedOverReals`, which dominates
    /// `Vacuous`, which dominates `Unestablished` -- matching the precedence
    /// `Failed > DisprovedModuloRealArithmetic > Invalid > Unsupported`.
    terminal: Option<Terminal>,
}

/// A terminal (non-green) outcome, ordered weakest-dominating last so a fold
/// can keep the dominating one (`max`). The chain is `Unestablished < Vacuous <
/// DisprovedOverReals < Disproved`: a definite machine-arithmetic counterexample
/// (`Disproved`) is the strongest failure; a reals-hedged counterexample
/// (`DisprovedOverReals`) is a weaker-certainty failure that still dominates the
/// `Vacuous`/`Unestablished` non-failure terminals (a possible counterexample is
/// a more actionable signal than vacuous assumptions or an unestablished result)
/// and, like every terminal, dominates all greens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Terminal {
    Unestablished,
    Vacuous,
    DisprovedOverReals,
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
            VerdictGuarantee::DisprovedOverReals => {
                self.terminal = Some(self.terminal.map_or(Terminal::DisprovedOverReals, |t| {
                    t.max(Terminal::DisprovedOverReals)
                }));
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
                Terminal::DisprovedOverReals => CompositeVerdict::DisprovedModuloRealArithmetic,
                Terminal::Vacuous => CompositeVerdict::Invalid,
                Terminal::Unestablished => CompositeVerdict::Unsupported,
            };
        }
        // All-green. The qualifier set drives the badge, weakest kind first, so
        // a set that mixes a weak kind with a strong one renders the WEAK badge
        // and a weaker guarantee can never be laundered into a stronger one.
        // Precedence, most-dominating first (chelis#422):
        //   FuzzBase               -> fuzz_validated   (base never proved; dominates all)
        //   SoundOverApproximation -> sound_approximate (over-approx base, not a proof)
        //   Axiom                  -> proven_modulo_asserted_axiom
        //   Fuzz (contract)        -> proven_modulo_fuzz_validated_contract
        //   SpecialFunctionCertified -> proven_modulo_certified_envelope (chelis#434)
        //   RealArith              -> proven_modulo_real_arithmetic (proof over reals)
        //   Exact, no weaker       -> proven
        // The `FuzzBase` check is FIRST: a fuzz-only base is the weakest green
        // and can never read `proven_*`, regardless of what was discharged on
        // top of it. `SoundOverApproximation` (an over-approximating base, e.g.
        // Beacon's interval engine) is the next weakest -- a green-exit result
        // that is not a proof, so it dominates the proven_* badges. The
        // `Axiom`-before-`Fuzz` order is preserved EXACTLY from the pre-422
        // lattice so existing rollups stay byte-identical (the wi6 oracle locks
        // it). `SpecialFunctionCertified` (chelis#434) sits BELOW `Fuzz` and
        // ABOVE `RealArith`: the envelope certificate is machine-checked (sound),
        // so it is stronger than a trusted axiom or a fuzz-validated contract, but
        // it discloses the extra over-approximation dependency, so it is weaker
        // than a pure over-reals proof and MUST be checked before `RealArith` (an
        // envelope-assisted proof never launders into `proven_modulo_real_arithmetic`).
        // Covered-or-rejected: a green rollup carrying only a still-unbadged
        // qualifier (`DeltaComplete`, `CertificateBearing`) renders `Unsupported`.
        if self.qualifiers.contains(Qualifier::FuzzBase) {
            CompositeVerdict::FuzzValidatedEmpirical
        } else if self.qualifiers.contains(Qualifier::SoundOverApproximation) {
            CompositeVerdict::SoundApproximate
        } else if self.qualifiers.contains(Qualifier::Axiom) {
            CompositeVerdict::ProvenModuloAssertedAxiom
        } else if self.qualifiers.contains(Qualifier::Fuzz) {
            CompositeVerdict::ProvenModuloFuzzValidatedContract
        } else if self
            .qualifiers
            .contains(Qualifier::SpecialFunctionCertified)
        {
            CompositeVerdict::ProvenModuloCertifiedEnvelope
        } else if self.qualifiers.contains(Qualifier::RealArith) {
            CompositeVerdict::ProvenModuloRealArithmetic
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
            // chelis#422: an SMT discharge is over the REALS, so it is a sound
            // over-approximation of the machine claim -- `SoundApproximate`
            // carrying `RealArith`, NOT an exact decision. An all-SMT
            // dependency set therefore rolls up to `proven_modulo_real_arithmetic`,
            // disclosing the machine-arithmetic gap on every smt green.
            Some("proved") if self.method == DischargeMethod::Smt => VerdictGuarantee::Green {
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::RealArith]),
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
            // chelis#674: a certified-envelope discharge from Beacon. The
            // envelope is machine-checked (Gappa/Arb) and sound, so it is
            // stronger than fuzz but still an over-approximation. It injects
            // SpecialFunctionCertified, which renders as
            // ProvenModuloCertifiedEnvelope in the badge rollup.
            Some("proved") if self.method == DischargeMethod::CertifiedEnvelope => {
                VerdictGuarantee::Green {
                    soundness: Soundness::SoundApproximate,
                    qualifiers: QualifierSet::from_iter_kinds([
                        Qualifier::SpecialFunctionCertified,
                    ]),
                }
            }
            Some("failed") => {
                if self.evidence.get("counterexample").is_some() {
                    // chelis#422 (symmetric to the `proved` branch above): a
                    // disproof discharged over the REALS (`arith_model:"real"`)
                    // is a HEDGED failure -- the counterexample may be a false
                    // counterexample at machine arithmetic -- so it folds as the
                    // reals-hedged `DisprovedOverReals`, which a definite
                    // `Disproved` still dominates. A disproof WITHOUT the
                    // over-reals marker (a fuzz counterexample is a real
                    // machine-arithmetic witness) folds as the definite
                    // `Disproved`. This keeps a synthesized self-discharge record
                    // consistent with its hedged base instead of collapsing it.
                    if self
                        .evidence
                        .get("arith_model")
                        .and_then(serde_json::Value::as_str)
                        == Some("real")
                    {
                        VerdictGuarantee::DisprovedOverReals
                    } else {
                        VerdictGuarantee::Disproved
                    }
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

/// Project a deductive engine's discharge `(soundness, qualifiers)` to the
/// BASE [`CompositeVerdict`] it backs, through the SAME lattice projection
/// every rollup uses (chelis#422). This is the consumer seam: the dispatch
/// `Discharge` carries the engine's soundness and qualifiers (cvc5 over reals
/// -> `SoundApproximate` + `RealArith`; Beacon's interval engine ->
/// `SoundApproximate` + `SoundOverApproximation`), and threading them here is
/// what makes those reach the verdict as `proven_modulo_real_arithmetic` /
/// `sound_approximate` rather than being flattened to a single hardcoded
/// `proven`. A green discharge whose qualifier set has no badged kind yet
/// projects to `Unsupported` (covered-or-rejected), never a silent proof.
pub fn base_verdict_from_discharge(
    soundness: Soundness,
    qualifiers: &QualifierSet,
) -> CompositeVerdict {
    VerdictRollup::identity()
        .fold(VerdictGuarantee::Green {
            soundness,
            qualifiers: qualifiers.clone(),
        })
        .badge()
}

/// The full DISCLOSED qualifier set of a composed green verdict, as sorted
/// snake_case strings for the `qualifiers:[...]` JSON array (chelis#422, D2).
/// The single `composite_verdict` token is the WEAKEST badge; this array
/// carries every caveat in the union so a consumer sees them all -- e.g. an
/// over-reals proof modulo a fuzz contract is token
/// `proven_modulo_fuzz_validated_contract` with
/// `qualifiers:["fuzz","real_arithmetic"]`. Most terminal (non-green) rollups
/// disclose no qualifiers (an empty array): there is no green guarantee to
/// qualify. The ONE exception is a reals-hedged disproof
/// (`DisprovedModuloRealArithmetic`): it discloses `real_arithmetic`, symmetric
/// to the proof side, so a consumer sees the same machine-arithmetic caveat on
/// the failure as on the pass. The base re-enters the lattice through the same
/// door every discharge does, so the array is consistent with the rendered
/// badge.
pub fn composed_qualifier_strings(
    base_soundness: Soundness,
    base_qualifiers: &QualifierSet,
    assumptions: &[AssumptionRecord],
) -> Vec<&'static str> {
    let rollup = assumptions.iter().fold(
        VerdictRollup::identity().fold(VerdictGuarantee::Green {
            soundness: base_soundness,
            qualifiers: base_qualifiers.clone(),
        }),
        |acc, a| a.fold_into(acc),
    );
    disclosed_from_rollup(&rollup)
}

/// The disclosed qualifier strings of a rollup, shared by the green-base
/// (`composed_qualifier_strings`) and badge-base
/// (`disclosed_qualifier_strings_for_base`) entry points. A green rollup
/// discloses its union; most terminals disclose nothing; the reals-hedged
/// disproof discloses `real_arithmetic` so the machine-arithmetic caveat is
/// visible on the failure side exactly as on the proof side (chelis#422,
/// symmetric).
fn disclosed_from_rollup(rollup: &VerdictRollup) -> Vec<&'static str> {
    match rollup.terminal {
        Some(Terminal::DisprovedOverReals) => vec![Qualifier::RealArith.as_str()],
        Some(_) => Vec::new(),
        None => rollup.qualifiers.iter().map(|q| q.as_str()).collect(),
    }
}

/// The disclosed qualifier strings of a verdict that re-enters the lattice from
/// a base BADGE (rather than a green `(soundness, qualifiers)` discharge). This
/// is the failure-aware entry point: a hedged-disproof base
/// (`DisprovedModuloRealArithmetic`) re-enters as the `DisprovedOverReals`
/// terminal and discloses `real_arithmetic`, where the green-only
/// `composed_qualifier_strings` cannot express a terminal base. The base
/// re-enters through `guarantee()` -- the same door `rollup_composite` uses --
/// so the disclosed array stays consistent with the rendered badge.
pub fn disclosed_qualifier_strings_for_base(
    base: CompositeVerdict,
    assumptions: &[AssumptionRecord],
) -> Vec<&'static str> {
    let rollup = assumptions.iter().fold(
        VerdictRollup::identity().fold(base.guarantee()),
        |acc, a| a.fold_into(acc),
    );
    disclosed_from_rollup(&rollup)
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
    fn c1_all_smt_discharges_roll_up_to_proven_modulo_real_arithmetic() {
        // chelis#422: an SMT discharge is over the REALS, so an all-SMT
        // dependency set rolls up to `proven_modulo_real_arithmetic` --
        // disclosing the machine-arithmetic gap -- not plain `proven`. (Plain
        // `proven` is reserved for a future exact-machine-arithmetic lowering.)
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
            CompositeVerdict::ProvenModuloRealArithmetic
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
        // chelis#422: the SMT discharge is over reals, so the present-and-green
        // probe is `proven_modulo_real_arithmetic` (disclosing the gap), not
        // plain `proven`. The point of the test is that removing or failing the
        // discharge DEGRADES this green below; the starting green is the
        // honest over-reals badge.
        assert_eq!(
            registry
                .probe_consumer("consumer", CompositeVerdict::Proven, ["contract:a"])
                .composite_verdict,
            CompositeVerdict::ProvenModuloRealArithmetic
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
            // Greens, strongest -> weakest. `proven` is strongest; the
            // `Fuzz`-before-`Axiom` rank (fuzz stronger than axiom) is preserved
            // EXACTLY from the pre-422 lattice. The new badges insert without
            // disturbing that pair: `proven_modulo_real_arithmetic` is the
            // strongest qualifier below a plain proof (chelis#422), and
            // `sound_approximate` / `fuzz_validated` are the two weakest greens
            // (an over-approximation base, then a fuzz-only base).
            CompositeVerdict::Proven => 0,
            CompositeVerdict::ProvenModuloRealArithmetic => 1,
            // chelis#434: weaker than a pure over-reals proof (extra envelope
            // caveat), stronger than the trusted/empirical fuzz+axiom badges
            // (the certificate is machine-checked). Inserts at rank 2, shifting
            // the weaker greens down one; relative order of every pre-existing
            // badge is preserved, so the byte-identical fold matrix still holds.
            CompositeVerdict::ProvenModuloCertifiedEnvelope => 2,
            CompositeVerdict::ProvenModuloFuzzValidatedContract => 3,
            CompositeVerdict::ProvenModuloAssertedAxiom => 4,
            CompositeVerdict::SoundApproximate => 5,
            CompositeVerdict::FuzzValidatedEmpirical => 6,
            // Terminals are weaker than every green (a terminal dominates a
            // green in the fold), ordered Unsupported < Invalid <
            // DisprovedModuloRealArithmetic < Failed. The reals-hedged disproof
            // is a hedged failure: it dominates greens + the Invalid/Unsupported
            // non-failure terminals, but loses to a definite Failed (chelis#422,
            // symmetric to proven_modulo_real_arithmetic on the proof side).
            CompositeVerdict::Unsupported => 7,
            CompositeVerdict::Invalid => 8,
            CompositeVerdict::DisprovedModuloRealArithmetic => 9,
            CompositeVerdict::Failed => 10,
        }
    }

    const ALL_VERDICTS: [CompositeVerdict; 11] = [
        CompositeVerdict::Proven,
        CompositeVerdict::ProvenModuloRealArithmetic,
        CompositeVerdict::ProvenModuloCertifiedEnvelope,
        CompositeVerdict::ProvenModuloFuzzValidatedContract,
        CompositeVerdict::ProvenModuloAssertedAxiom,
        CompositeVerdict::SoundApproximate,
        CompositeVerdict::FuzzValidatedEmpirical,
        CompositeVerdict::Unsupported,
        CompositeVerdict::Invalid,
        CompositeVerdict::DisprovedModuloRealArithmetic,
        CompositeVerdict::Failed,
    ];

    #[test]
    fn wi6_lattice_projection_reproduces_legacy_weakest_for_every_badge_pair() {
        // The lattice rollup must be byte-identical to the scalar weakest-of
        // fold across the whole NxN badge matrix, so a weaker base or qualifier
        // can never launder into a stronger badge. We fold two guarantees
        // directly through the rollup and compare to the weaker-of-the-two rank.
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
    fn wi6_homogeneous_smt_set_rolls_up_to_proven_modulo_real_arithmetic() {
        // chelis#422: an all-SMT (proved over reals) dependency set rolls up to
        // `proven_modulo_real_arithmetic`, disclosing the machine-arithmetic
        // gap. Plain `proven` is reserved for a future exact lowering.
        let assumptions = vec![
            assumption("a", DischargeMethod::Smt, json!({"status": "proved"})),
            assumption("b", DischargeMethod::Smt, json!({"status": "proved"})),
        ];
        assert_eq!(
            rollup_composite(CompositeVerdict::Proven, &assumptions),
            CompositeVerdict::ProvenModuloRealArithmetic
        );
    }

    #[test]
    fn wi6_homogeneous_exact_lattice_point_still_rolls_up_to_proven() {
        // The `proven` badge remains REACHABLE in the lattice: a base and a
        // discharge that both carry the `Exact` qualifier at `Soundness::Exact`
        // (a hypothetical future exact-machine engine) roll up to plain
        // `proven`. This guards the retirement of `proven` from CURRENT output
        // against accidentally removing it from the lattice entirely.
        let rolled = VerdictRollup::identity()
            .fold(VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Exact]),
            })
            .fold(VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Exact]),
            });
        assert_eq!(rolled.badge(), CompositeVerdict::Proven);
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
    fn wi6_sub_exact_green_does_not_launder_into_proven() {
        // Covered-or-rejected at the badge-rendering point: a green rollup at
        // sub-Exact soundness must never launder into the strong `proven`
        // badge. `SoundOverApproximation` now has its own disclosed green badge
        // (`sound_approximate`, chelis#422) -- a Beacon-shaped interval
        // discharge -- and `RealArith` renders `proven_modulo_real_arithmetic`;
        // both are strictly below `proven` and disclose their approximation.
        let sound_approx = VerdictRollup::identity().fold(VerdictGuarantee::Green {
            soundness: Soundness::SoundApproximate,
            qualifiers: QualifierSet::from_iter_kinds([Qualifier::SoundOverApproximation]),
        });
        assert_ne!(
            sound_approx.badge(),
            CompositeVerdict::Proven,
            "a sub-Exact sound-over-approximation green must not launder into proven"
        );
        assert_eq!(
            sound_approx.badge(),
            CompositeVerdict::SoundApproximate,
            "a SoundOverApproximation green discloses `sound_approximate`, not proven"
        );

        let real_arith = VerdictRollup::identity().fold(VerdictGuarantee::Green {
            soundness: Soundness::SoundApproximate,
            qualifiers: QualifierSet::from_iter_kinds([Qualifier::RealArith]),
        });
        assert_ne!(
            real_arith.badge(),
            CompositeVerdict::Proven,
            "an over-reals proof must disclose the machine-arith gap, not launder into proven"
        );
        assert_eq!(
            real_arith.badge(),
            CompositeVerdict::ProvenModuloRealArithmetic,
        );

        // The still-unbadged producer qualifiers (DeltaComplete, CertificateBearing)
        // take the conservative rejection at sub-Exact soundness rather than
        // laundering into a proof.
        for qualifier in [Qualifier::DeltaComplete, Qualifier::CertificateBearing] {
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

    #[test]
    fn special_function_certified_projects_to_the_certified_envelope_tier() {
        // chelis#434: a green discharge carrying SpecialFunctionCertified now
        // projects to the distinct honest tier proven_modulo_certified_envelope
        // (NOT unsupported anymore, and NEVER proven / proven_modulo_real_arithmetic).
        let sfc = VerdictRollup::identity().fold(VerdictGuarantee::Green {
            soundness: Soundness::SoundApproximate,
            qualifiers: QualifierSet::from_iter_kinds([Qualifier::SpecialFunctionCertified]),
        });
        assert_eq!(
            sfc.badge(),
            CompositeVerdict::ProvenModuloCertifiedEnvelope,
            "SpecialFunctionCertified must project to the certified-envelope tier"
        );

        // The real envelope discharge carries BOTH SpecialFunctionCertified (the
        // envelope) AND RealArith (the residual's over-reals proof). The
        // certified-envelope tier DOMINATES: the badge discloses the envelope
        // dependency and must NEVER launder into proven_modulo_real_arithmetic.
        let both = VerdictRollup::identity().fold(VerdictGuarantee::Green {
            soundness: Soundness::SoundApproximate,
            qualifiers: QualifierSet::from_iter_kinds([
                Qualifier::SpecialFunctionCertified,
                Qualifier::RealArith,
            ]),
        });
        assert_eq!(
            both.badge(),
            CompositeVerdict::ProvenModuloCertifiedEnvelope
        );
        assert_ne!(both.badge(), CompositeVerdict::ProvenModuloRealArithmetic);
        assert_ne!(both.badge(), CompositeVerdict::Proven);

        // The reserved direction: an envelope-FREE over-reals proof (RealArith
        // alone) STAYS proven_modulo_real_arithmetic — the new token never leaks
        // onto envelope-free proofs.
        let real_only = VerdictRollup::identity().fold(VerdictGuarantee::Green {
            soundness: Soundness::SoundApproximate,
            qualifiers: QualifierSet::from_iter_kinds([Qualifier::RealArith]),
        });
        assert_eq!(
            real_only.badge(),
            CompositeVerdict::ProvenModuloRealArithmetic
        );
        assert_ne!(
            real_only.badge(),
            CompositeVerdict::ProvenModuloCertifiedEnvelope
        );

        // Serialized token.
        assert_eq!(
            CompositeVerdict::ProvenModuloCertifiedEnvelope.as_str(),
            "proven_modulo_certified_envelope"
        );
    }

    // --- chelis#422 (symmetric): DisprovedModuloRealArithmetic (failure side) ---

    #[test]
    fn disproved_over_reals_renders_the_hedged_failure_badge() {
        // The disproof-side terminal projects to the hedged-failure badge --
        // NOT a definite Failed and NOT a green.
        let rollup = VerdictRollup::identity().fold(VerdictGuarantee::DisprovedOverReals);
        assert_eq!(
            rollup.badge(),
            CompositeVerdict::DisprovedModuloRealArithmetic
        );
        assert_ne!(rollup.badge(), CompositeVerdict::Failed);
    }

    #[test]
    fn hedged_disproof_dominates_every_green_and_is_not_masked() {
        // A reals counterexample must never be masked by a pass: folded with any
        // green (including the strongest Proven), the hedged failure wins, in
        // either fold order.
        for green in [
            VerdictGuarantee::Green {
                soundness: Soundness::Exact,
                qualifiers: QualifierSet::new(),
            },
            VerdictGuarantee::Green {
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::RealArith]),
            },
            VerdictGuarantee::Green {
                soundness: Soundness::Empirical,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::Fuzz]),
            },
            VerdictGuarantee::Green {
                soundness: Soundness::Empirical,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::FuzzBase]),
            },
        ] {
            let a = VerdictRollup::identity()
                .fold(green.clone())
                .fold(VerdictGuarantee::DisprovedOverReals)
                .badge();
            let b = VerdictRollup::identity()
                .fold(VerdictGuarantee::DisprovedOverReals)
                .fold(green)
                .badge();
            assert_eq!(a, CompositeVerdict::DisprovedModuloRealArithmetic);
            assert_eq!(b, CompositeVerdict::DisprovedModuloRealArithmetic);
        }
    }

    #[test]
    fn definite_disproof_dominates_a_hedged_disproof() {
        // A definite machine-arithmetic counterexample is stronger than a
        // reals-hedged one: a set carrying both renders the definite Failed.
        let rolled = VerdictRollup::identity()
            .fold(VerdictGuarantee::DisprovedOverReals)
            .fold(VerdictGuarantee::Disproved)
            .badge();
        assert_eq!(rolled, CompositeVerdict::Failed);
        let rolled_rev = VerdictRollup::identity()
            .fold(VerdictGuarantee::Disproved)
            .fold(VerdictGuarantee::DisprovedOverReals)
            .badge();
        assert_eq!(rolled_rev, CompositeVerdict::Failed);
    }

    #[test]
    fn hedged_disproof_dominates_invalid_and_unsupported() {
        // A possible counterexample is a more actionable signal than a vacuous
        // assumption set or an unestablished result, so it dominates both.
        for weaker in [VerdictGuarantee::Vacuous, VerdictGuarantee::Unestablished] {
            let rolled = VerdictRollup::identity()
                .fold(weaker)
                .fold(VerdictGuarantee::DisprovedOverReals)
                .badge();
            assert_eq!(rolled, CompositeVerdict::DisprovedModuloRealArithmetic);
        }
    }

    #[test]
    fn disclosed_qualifiers_for_a_hedged_disproof_is_real_arithmetic() {
        // The hedged-failure rollup discloses `real_arithmetic` (the ONE
        // terminal that discloses a qualifier), symmetric to the proof side, so
        // a consumer sees the machine-arithmetic caveat on the failure too. The
        // rollup's dominating terminal is the hedged disproof (modelled by
        // re-entering the lattice from the hedged-disproof badge, exactly as the
        // production base-badge path does).
        let disclosed = disclosed_qualifier_strings_for_base(
            CompositeVerdict::DisprovedModuloRealArithmetic,
            &[],
        );
        assert_eq!(disclosed, vec!["real_arithmetic"]);
        // A definite failure terminal still discloses nothing.
        let definite = disclosed_qualifier_strings_for_base(CompositeVerdict::Failed, &[]);
        assert!(definite.is_empty());
    }

    #[test]
    fn hedged_disproof_round_trips_and_has_its_canonical_string() {
        assert_eq!(
            CompositeVerdict::DisprovedModuloRealArithmetic.as_str(),
            "disproved_modulo_real_arithmetic"
        );
        let json = serde_json::to_string(&CompositeVerdict::DisprovedModuloRealArithmetic).unwrap();
        assert_eq!(json, "\"disproved_modulo_real_arithmetic\"");
        let back: CompositeVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(back, CompositeVerdict::DisprovedModuloRealArithmetic);
    }
}
