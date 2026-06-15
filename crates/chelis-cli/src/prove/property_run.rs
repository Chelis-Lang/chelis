//! CLI-side rendering of user `@property` verification (U4 / D-PARITY).
//!
//! Discovery + Tier B/C running + assumption injection all live in
//! `chelis_prove::property_runner` -- the SAME implementation the chelis-tide
//! MCP tool drives. This module is the thin CLI driver: it asks the shared
//! runner to discover and run every `@property` in the module source, then
//! renders the outcomes as the NDJSON `{kind:"property", ...}` records and
//! folds them into the prove summary, so a CLI prove and a tide prove agree
//! on the same module.
//!
//! Gated on the `chelis-prove` optional dependency (the shared runner lives
//! there). Without the capability the CLI falls back to its local
//! Tier-C-only property path in the parent module.
#![cfg(feature = "chelis-prove")]

use std::path::Path;

use chelis_prove::property_runner::{
    PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyTier,
    run_deep_source_properties, run_surf_source_properties,
};
use serde_json::json;

use super::{ProveOptions, Status, Summary};

fn run_opts(options: &ProveOptions<'_>) -> PropertyRunOptions {
    PropertyRunOptions {
        seed: options.seed.unwrap_or(0),
        samples: options.samples.unwrap_or(100),
        smt_timeout_ms: options.smt_timeout_ms,
        tier: options.tier.to_string(),
        only: options.only.map(str::to_string),
        invariant_min_rate: options.invariant_min_rate,
        max_attempts: options.max_attempts,
    }
}

/// Discover and run every user `@property` in Surf `source` through the
/// shared runner, render each outcome, and fold the verdicts into `totals`.
/// Returns the combined property status.
pub(super) fn run_surf_properties_shared(
    path: &Path,
    source: &str,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    let outcomes = match run_surf_source_properties(source, &run_opts(options)) {
        Ok(PropertyRunResult::Ran(o)) => o,
        // A parse failure here would already have surfaced upstream
        // (prove_surf_file parses first); treat as no properties.
        Err(_) => return Status::Passed,
    };
    render_all(path, &outcomes, "surf", options, totals)
}

/// Discover and run every USER `@property` in Deep `source` through the
/// SHARED property runner (F6): the SAME engine the tide MCP tool drives, so
/// a CLI prove and a tide prove of the same `.dp` module agree. The bridge
/// (`c-earchin`) deep properties are handled separately by the CLI-local
/// path with their span/requirement rendering.
pub(super) fn run_deep_properties_shared(
    path: &Path,
    source: &str,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    let outcomes = match run_deep_source_properties(source, &run_opts(options)) {
        Ok(PropertyRunResult::Ran(o)) => o,
        Err(_) => return Status::Passed,
    };
    render_all(path, &outcomes, "user", options, totals)
}

fn render_all(
    path: &Path,
    outcomes: &[PropertyOutcome],
    source_kind: &str,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    let mut status = Status::Passed;
    for outcome in outcomes {
        let s = render_property(path, outcome, source_kind, options, totals);
        status = super::combine_status(status, s);
    }
    status
}

fn render_property(
    path: &Path,
    outcome: &PropertyOutcome,
    source_kind: &str,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    totals.total += 1;
    // The display status is the shared `is_pass`-bucketed label (F8): a
    // zero-sample `Passed` sentinel is reported as "unsupported", matching
    // the tide surface exactly. The exit/summary fold buckets off the same
    // label, so CLI and tide agree on every property.
    let label = outcome.display_status();
    let status = match label {
        "passed" => {
            totals.passed += 1;
            Status::Passed
        }
        "failed" => {
            totals.failed += 1;
            Status::Failed
        }
        "unsupported" => {
            totals.unsupported += 1;
            Status::Unsupported
        }
        _ => {
            totals.errors += 1;
            Status::Error
        }
    };
    emit(path, outcome, label, source_kind, options);
    status
}

fn emit(
    path: &Path,
    outcome: &PropertyOutcome,
    status: &str,
    source_kind: &str,
    options: &ProveOptions<'_>,
) {
    if options.json {
        let mut value = json!({
            "kind": "property",
            "name": outcome.name,
            "status": status,
            "samples": outcome.samples,
            "seed": outcome.seed,
            "source": json!({ "kind": source_kind, "file": path.display().to_string() }),
        });
        if outcome.proof_tier != PropertyTier::None {
            value["proof_tier"] = json!(outcome.proof_tier.as_str());
        }
        if let Some(cx) = &outcome.counterexample {
            value["counterexample"] = cx.clone();
            value["shrink_steps"] = json!(0);
        }
        if let Some(r) = &outcome.reason {
            value["reason"] = json!(r);
        }
        println!("{value}");
    } else {
        match status {
            "passed" => {
                let suffix = if outcome.injected { " (injected)" } else { "" };
                println!(
                    "property: {} -- {}/{} passed{suffix}",
                    outcome.name, outcome.samples, outcome.samples
                )
            }
            "failed" => println!(
                "property failure: {}\n  --> {}",
                outcome.name,
                path.display()
            ),
            "unsupported" => println!(
                "property unsupported: {}: {}",
                outcome.name,
                outcome.reason.clone().unwrap_or_default()
            ),
            _ => println!(
                "property error: {}: {}",
                outcome.name,
                outcome.reason.clone().unwrap_or_default()
            ),
        }
    }
}
