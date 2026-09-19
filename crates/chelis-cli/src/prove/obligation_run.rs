//! CLI-side rendering of derived producer obligations (RFC D-OBLIG,
//! D-PARITY).
//!
//! The collection, synthesis, Tier B lowering, and Tier C evaluation all
//! live in `chelis_prove::obligation_engine` — the SAME implementation the
//! chelis-tide MCP tool drives (D-PARITY). This module is the thin CLI
//! driver: it desugars the module, runs the checker for INFERRED return
//! types, asks the shared engine to run every obligation, then renders the
//! outcomes as the additive NDJSON `{kind:"obligation", ...}` records and
//! folds them into the prove summary.
//!
//! Gated on the `chelis-prove` optional dependency (the same gate the
//! Tier B SMT lowering uses) by the parent module declaration.

use chelis_prove::CompositeVerdict;
use chelis_prove::obligation_engine::{
    ObligationOutcome, ObligationRunOptions, ObligationRunResult, ObligationStatus, ObligationTier,
    run_surf_source_obligations,
};
use chelis_surf::ast::Decl;
use serde_json::json;

use super::{ProveOptions, Status, Summary};

/// Run every derived producer obligation discovered in `raw_decls` (the
/// FULL module-bearing Surf decls). Returns the combined obligation
/// status.
pub(super) fn run_obligations(
    raw_decls: &[Decl],
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    // Re-serialize the parsed module to Surf source for the shared engine
    // entry. (The engine re-parses + desugars + checks; this keeps the
    // CLI and tide entry identical — both feed module source.)
    let source = chelis_surf::format::format_program(raw_decls);
    let run_opts = ObligationRunOptions {
        seed: options.seed.unwrap_or(0),
        samples: options.samples.unwrap_or(100),
        smt_timeout_ms: options.smt_timeout_ms,
        tier: options.tier.to_string(),
        only: options.only.map(str::to_string),
        invariant_min_rate: options.invariant_min_rate,
    };
    let outcomes = match run_surf_source_obligations(&source, &run_opts) {
        Ok(ObligationRunResult::Ran(o)) => o,
        // A module that does not type-check cannot have its obligations
        // meaningfully verified; surface the check diagnostics and Error,
        // never silent success (RT3-F2). This also affects FlukeBall's
        // strict prove-compat admission flow.
        Ok(ObligationRunResult::CheckFailed(messages)) => {
            emit_check_failure(options, &messages, totals);
            return Status::Error;
        }
        // A genuinely unparseable module is likewise an Error, not a pass.
        Err(message) => {
            emit_check_failure(options, std::slice::from_ref(&message), totals);
            return Status::Error;
        }
    };

    let mut status = Status::Passed;
    for outcome in &outcomes {
        let s = render_outcome(outcome, options, totals);
        status = super::combine_status(status, s);
    }
    status
}

/// Type-check a bare Surf declaration set without collecting producer
/// obligations.
pub(super) fn check_unlinked_decls(
    decls: &[Decl],
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    let deep_exprs = match chelis_surf::desugar::try_desugar_program(decls) {
        Ok(deep) => deep,
        Err(error) => {
            emit_check_failure(options, &[error.to_string()], totals);
            return Status::Error;
        }
    };
    let _linked = chelis_types::install_linked_program_guard();
    if let Err(infer) = chelis_types::check_typed_program(&deep_exprs) {
        let messages = infer
            .errors
            .iter()
            .map(|err| err.message.clone())
            .collect::<Vec<_>>();
        emit_check_failure(options, &messages, totals);
        return Status::Error;
    }
    Status::Passed
}

/// Type-check an already-linked Reef declaration closure before any property
/// verdict is emitted.
///
/// Linked imports must be checked through the compiler's layered seam: it
/// installs the linker provenance guard and checks the non-stdlib declarations
/// against the stdlib signature context. Feeding the same declarations to the
/// bare `check_typed_program` path loses that context and can report imported
/// names as unbound even though the linker resolved them (chelis#923/#924).
pub(super) fn check_linked_decls(
    stdlib_decls: &[Decl],
    stdlib_source_digest: [u8; 32],
    non_stdlib_decls: &[Decl],
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    match chelis_compiler_api::check_layered(stdlib_decls, stdlib_source_digest, non_stdlib_decls) {
        Ok(Some(_)) => Status::Passed,
        Ok(None) => {
            let mut decls = stdlib_decls.to_vec();
            decls.extend_from_slice(non_stdlib_decls);
            check_unlinked_decls(&decls, options, totals)
        }
        Err(error) => {
            let mut messages = error
                .errors
                .iter()
                .map(|diagnostic| diagnostic.message.clone())
                .collect::<Vec<_>>();
            if messages.is_empty() {
                messages.push(error.stage);
            }
            emit_check_failure(options, &messages, totals);
            Status::Error
        }
    }
}

/// Emit a module type-check failure as a prove error record (RT3-F2). The
/// check diagnostics are surfaced so the failure is visible, never hidden.
fn emit_check_failure(options: &ProveOptions<'_>, messages: &[String], totals: &mut Summary) {
    totals.errors += 1;
    let joined = messages.join("; ");
    if options.json {
        println!(
            "{}",
            json!({
                "kind": "error",
                "stage": "check",
                "reason": format!("module does not type-check; obligations not verified: {joined}"),
                "diagnostics": messages,
            })
        );
    } else {
        eprintln!("prove error: module does not type-check; obligations not verified:");
        for m in messages {
            eprintln!("  - {m}");
        }
    }
}

fn render_outcome(
    outcome: &ObligationOutcome,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    match obligation_display_status(outcome) {
        "error" => {
            // A collection-time declaration error (covered-or-rejected /
            // signature rejection). Always surfaces.
            totals.errors += 1;
            emit(outcome, options);
            Status::Error
        }
        "passed" => {
            totals.total += 1;
            totals.obligations += 1;
            totals.passed += 1;
            emit(outcome, options);
            Status::Passed
        }
        "failed" => {
            totals.total += 1;
            totals.obligations += 1;
            totals.failed += 1;
            emit(outcome, options);
            Status::Failed
        }
        _ => {
            totals.total += 1;
            totals.obligations += 1;
            totals.unsupported += 1;
            emit(outcome, options);
            Status::Unsupported
        }
    }
}

fn obligation_display_status(outcome: &ObligationOutcome) -> &'static str {
    if outcome.status == ObligationStatus::Error {
        return "error";
    }
    match outcome.composite_verdict {
        // A reals-hedged disproof is a failure (the property did not hold over
        // the reals); the precise `composite_verdict` field still carries the
        // hedged badge for a consumer that wants the distinction (chelis#422).
        CompositeVerdict::Failed | CompositeVerdict::DisprovedModuloRealArithmetic => "failed",
        CompositeVerdict::Invalid | CompositeVerdict::Unsupported => "unsupported",
        // Every green badge -- proven, the proven_modulo_* disclosures, the
        // sound-over-approximation base, and the fuzz-only base -- reports
        // through the obligation status, with the badge carrying the
        // not-proven / disclosed-caveat distinction (chelis#422).
        CompositeVerdict::Proven
        | CompositeVerdict::ProvenModuloRealArithmetic
        | CompositeVerdict::ProvenModuloCertifiedEnvelope
        | CompositeVerdict::ProvenModuloFuzzValidatedContract
        | CompositeVerdict::ProvenModuloAssertedAxiom
        | CompositeVerdict::SoundApproximate
        | CompositeVerdict::FuzzValidatedEmpirical => match outcome.status {
            ObligationStatus::Passed => "passed",
            ObligationStatus::Failed => "failed",
            ObligationStatus::Unsupported => "unsupported",
            ObligationStatus::Error => "error",
        },
    }
}

fn emit(outcome: &ObligationOutcome, options: &ProveOptions<'_>) {
    let status = obligation_display_status(outcome);
    if options.json {
        if outcome.status == ObligationStatus::Error {
            // Declaration errors carry no producer/name; emit a minimal
            // record so the count of kind:"obligation" stays accurate
            // while the reason names the offending producer.
            println!(
                "{}",
                json!({
                    "kind": "obligation",
                    "obligation_kind": "invariant_producer",
                    "status": "error",
                    "composite_verdict": outcome.composite_verdict.as_str(),
                    "assumptions": &outcome.assumptions,
                    "reason": outcome.reason,
                })
            );
            return;
        }
        let mut value = json!({
            "kind": "obligation",
            "obligation_kind": outcome.meta.obligation_kind,
            "source_type": outcome.meta.source_type,
            "producer": outcome.meta.producer,
            "name": outcome.name,
            "status": status,
            "composite_verdict": outcome.composite_verdict.as_str(),
            // chelis#422 (D2): full disclosed caveat set alongside the weakest
            // `composite_verdict` token.
            "qualifiers": outcome.disclosed_qualifiers(),
            "assumptions": &outcome.assumptions,
            "proof_tier": outcome.proof_tier.as_str(),
            "samples": outcome.samples,
            "seed": outcome.seed,
        });
        // chelis#436: the discharged proposition (the invariant predicate) travels
        // with the record so a consumer displays exactly what was discharged.
        if let Some(goal) = &outcome.goal {
            value["goal"] = json!(goal);
        }
        if outcome.proof_tier == ObligationTier::Smt {
            value["arith_model"] = json!("real");
        }
        if let Some(cx) = &outcome.counterexample {
            value["counterexample"] = cx.clone();
            value["shrink_steps"] = json!(outcome.shrink_steps);
        }
        if let Some(r) = &outcome.reason {
            value["reason"] = json!(r);
        }
        println!("{value}");
    } else {
        match outcome.status {
            ObligationStatus::Passed => {
                println!(
                    "obligation: {} -- proved ({})",
                    outcome.name,
                    outcome.proof_tier.as_str()
                )
            }
            ObligationStatus::Failed => println!(
                "obligation failure: {} ({} counterexample)",
                outcome.name,
                outcome.proof_tier.as_str()
            ),
            ObligationStatus::Unsupported => println!(
                "obligation unsupported: {}: {}",
                outcome.name,
                outcome.reason.clone().unwrap_or_default()
            ),
            ObligationStatus::Error => println!(
                "obligation error: {}",
                outcome.reason.clone().unwrap_or_default()
            ),
        }
    }
}
