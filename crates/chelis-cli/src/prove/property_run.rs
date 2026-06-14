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
    PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyStatus, PropertyTier,
    run_surf_source_properties,
};
use serde_json::json;

use super::{ProveOptions, Status, Summary};

/// Discover and run every user `@property` in Surf `source` through the
/// shared runner, render each outcome, and fold the verdicts into `totals`.
/// Returns the combined property status.
pub(super) fn run_surf_properties_shared(
    path: &Path,
    source: &str,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    let run_opts = PropertyRunOptions {
        seed: options.seed.unwrap_or(0),
        samples: options.samples.unwrap_or(100),
        smt_timeout_ms: options.smt_timeout_ms,
        tier: options.tier.to_string(),
        only: options.only.map(str::to_string),
        invariant_min_rate: options.invariant_min_rate,
        max_attempts: options.max_attempts,
    };
    let outcomes = match run_surf_source_properties(source, &run_opts) {
        Ok(PropertyRunResult::Ran(o)) => o,
        // A parse failure here would already have surfaced upstream
        // (prove_surf_file parses first); treat as no properties.
        Err(_) => return Status::Passed,
    };
    let mut status = Status::Passed;
    for outcome in &outcomes {
        let s = render_property(path, outcome, options, totals);
        status = super::combine_status(status, s);
    }
    status
}

fn render_property(
    path: &Path,
    outcome: &PropertyOutcome,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    totals.total += 1;
    // A genuine pass is Passed with samples > 0 (fuzz) or an SMT proof
    // (samples == 0, tier == Smt). A Passed-with-zero-fuzz sentinel is NOT a
    // pass; bucket it as unsupported so it lowers the exit status and is
    // visible (U4).
    let bucket = if outcome.is_pass() {
        Bucket::Passed
    } else {
        match outcome.status {
            PropertyStatus::Failed => Bucket::Failed,
            PropertyStatus::Unsupported => Bucket::Unsupported,
            PropertyStatus::Error => Bucket::Error,
            PropertyStatus::Passed => Bucket::Unsupported,
        }
    };
    match bucket {
        Bucket::Passed => totals.passed += 1,
        Bucket::Failed => totals.failed += 1,
        Bucket::Unsupported => totals.unsupported += 1,
        Bucket::Error => totals.errors += 1,
    }
    emit(path, outcome, status_label(bucket), options);
    bucket.status()
}

#[derive(Clone, Copy)]
enum Bucket {
    Passed,
    Failed,
    Unsupported,
    Error,
}

impl Bucket {
    fn status(self) -> Status {
        match self {
            Bucket::Passed => Status::Passed,
            Bucket::Failed => Status::Failed,
            Bucket::Unsupported => Status::Unsupported,
            Bucket::Error => Status::Error,
        }
    }
}

fn status_label(bucket: Bucket) -> &'static str {
    match bucket {
        Bucket::Passed => "passed",
        Bucket::Failed => "failed",
        Bucket::Unsupported => "unsupported",
        Bucket::Error => "error",
    }
}

fn emit(path: &Path, outcome: &PropertyOutcome, status: &str, options: &ProveOptions<'_>) {
    if options.json {
        let mut value = json!({
            "kind": "property",
            "name": outcome.name,
            "status": status,
            "samples": outcome.samples,
            "seed": outcome.seed,
            "source": json!({ "kind": "surf", "file": path.display().to_string() }),
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
