//! Three-tier property verification dispatcher.
//!
//! This crate implements the specification-consumption layer of the Chelis
//! trust stack. It takes `@property` declarations and dispatches them through:
//!
//! - **Tier A:** Type system validation (dimension, effect, linearity discharge)
//! - **Tier B:** SMT solving via cvc5 (nonlinear real arithmetic)
//! - **Tier C:** Randomized fuzz testing (existing `chelis prove` logic)
//!
//! See `docs/trust_stack_verification.md` for architectural framing.

pub mod ad_rail;
pub mod artifact;
pub mod beacon_shim;
// WI-15 SoS certificate engine (Clarabel). The whole module is behind the
// `clarabel` feature so the default / smt / solver-free builds link none of it.
#[cfg(feature = "clarabel")]
pub mod clarabel_sos;
// WI-16 Carcara auditability: re-check cvc5's Alethe proofs. The SMT-LIB
// problem renderer and the audit-outcome type are always present (so the
// renderer is unit-testable without cvc5/carcara linked); the live
// cvc5 -> Alethe -> Carcara round-trip is gated behind the `carcara` feature.
pub mod carcara_audit;
pub mod composition;
pub mod concrete_eval;
pub mod contracts;
pub mod discharge;
pub mod dispatch;
pub mod engine_registry;
pub mod from_property_spec;
pub mod graph_extract;
pub mod inlineability;
pub mod obligation_engine;
pub mod obligations;
pub mod opaque;
pub mod property_runner;
pub mod solver;
pub mod tier_a;
pub mod tier_b;
pub mod tier_b_lower;
pub mod tier_c;
pub mod worker;
// WI-12 (WS-5): the Z3 NRA DischargeEngine over GoalShape::Smt. Gated behind
// the `z3` feature so it links Z3 only when asked; the default and smt builds
// carry no z3-named symbols.
#[cfg(feature = "z3")]
pub mod z3_engine;

pub use artifact::{ProofArtifact, ProofStatus, ProofTier};
// chelis#439: the Beacon subprocess shim (transport-only DischargeEngine for
// GoalShape::BoxRange) and its dispatch-site-owned content-addressed byte store.
pub use beacon_shim::{BEACON_BIN_ENV, BeaconShim, RequestTransport, WireDagByteStore};
pub use carcara_audit::{CarcaraAudit, render_smtlib_problem};
pub use composition::{
    AssumptionDischarge, AssumptionRecord, AssumptionRegistry, CompositeVerdict, CompositionProbe,
    DischargeMethod, DischargeTier, NonVacuityRecord, NonVacuityStatus, rollup_composite,
};
pub use concrete_eval::{eval_arith, eval_bool};
pub use contracts::{
    ContractInvariant, StandardContract, standard_contract_registry, standard_contracts,
};
pub use discharge::{
    Discharge, DischargeEngine, DischargeError, Goal, GoalError, GoalShape, IntervalBox, IrHandle,
    OutputRange, Qualifier, QualifierSet, Soundness, classify_smt_outcome,
};
// The cvc5 engine exists only when cvc5 is linked (the smt feature); the
// default build keeps zero cvc5-named symbols (the solver-free gate).
#[cfg(feature = "smt")]
pub use discharge::Cvc5Engine;
// WI-12 (WS-5): the Z3 NRA engine exists only when Z3 is linked (the z3
// feature); the default and smt builds keep zero z3-named symbols.
#[cfg(feature = "z3")]
pub use z3_engine::Z3Engine;
// WI-9: the discharge-engine registry + fitness-based dispatcher and its no-fit
// honesty floor. The solver-free SMT engine is the default-build SMT lane and
// exists only when cvc5 is absent.
#[cfg(not(feature = "smt"))]
pub use engine_registry::SolvePropertyEngine;
pub use engine_registry::{DischargeRegistry, no_fit_discharge};
// Re-export the chelis-pred predicate helpers so downstream CLI/tide
// consumers reach them through chelis-prove (RFC D-PRED consumer surface).
pub use chelis_pred::{PredAmenability, predicate_free_vars};
pub use dispatch::{DispatchOptions, dispatch_property};
pub use from_property_spec::{PropertySpecInput, to_dispatch_amenability, to_smt_property};
pub use graph_extract::{
    ExtractedGoal, GraphExtractError, box_range_goal_from_source, box_range_goal_from_wire_dag,
    box_range_goals_from_source, name_sorted_input_box,
};
pub use inlineability::{Fuzzability, Inlineability, classify_fuzzability, classify_inlineability};
pub use tier_b::{
    AssumptionSatisfiability, SmtProperty, check_assumptions_satisfiable, solve_property,
};
pub use worker::{enable_isolation, run_worker_if_requested};
