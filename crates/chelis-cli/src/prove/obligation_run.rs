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
//! Tier B SMT lowering uses).
#![cfg(feature = "chelis-prove")]

use chelis_prove::obligation_engine::{
    ObligationOutcome, ObligationRunOptions, ObligationStatus, ObligationTier,
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
    };
    let outcomes = match run_surf_source_obligations(&source, &run_opts) {
        Ok(o) => o,
        Err(_) => return Status::Passed,
    };

    let mut status = Status::Passed;
    for outcome in &outcomes {
        let s = render_outcome(outcome, options, totals);
        status = super::combine_status(status, s);
    }
    status
}

fn render_outcome(
    outcome: &ObligationOutcome,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    match outcome.status {
        ObligationStatus::Error => {
            // A collection-time declaration error (covered-or-rejected /
            // signature rejection). Always surfaces.
            totals.errors += 1;
            emit(outcome, options);
            Status::Error
        }
        ObligationStatus::Passed => {
            totals.total += 1;
            totals.obligations += 1;
            totals.passed += 1;
            emit(outcome, options);
            Status::Passed
        }
        ObligationStatus::Failed => {
            totals.total += 1;
            totals.obligations += 1;
            totals.failed += 1;
            emit(outcome, options);
            Status::Failed
        }
        ObligationStatus::Unsupported => {
            totals.total += 1;
            totals.obligations += 1;
            totals.unsupported += 1;
            emit(outcome, options);
            Status::Unsupported
        }
    }
}

fn emit(outcome: &ObligationOutcome, options: &ProveOptions<'_>) {
    let status = match outcome.status {
        ObligationStatus::Passed => "passed",
        ObligationStatus::Failed => "failed",
        ObligationStatus::Unsupported => "unsupported",
        ObligationStatus::Error => "error",
    };
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
            "proof_tier": outcome.proof_tier.as_str(),
            "samples": outcome.samples,
            "seed": outcome.seed,
        });
        if outcome.proof_tier == ObligationTier::Smt {
            value["arith_model"] = json!("real");
        }
        if let Some(cx) = &outcome.counterexample {
            value["counterexample"] = cx.clone();
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
