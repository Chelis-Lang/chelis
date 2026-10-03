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
//! Tier-C-only property path in the parent module. The parent module
//! declaration owns this feature gate.

use std::collections::BTreeMap;
use std::path::Path;

use chelis_prove::property_runner::{
    PropertyOutcome, PropertyRunOptions, PropertyRunResult, PropertyTier,
    run_deep_source_properties, run_surf_decls_properties_with_contract_decls,
    run_surf_source_properties,
};
use chelis_surf::ast::Decl;
use serde_json::json;

use super::{ProveOptions, Status, Summary};

fn run_opts(options: &ProveOptions<'_>) -> PropertyRunOptions {
    PropertyRunOptions {
        seed: options.seed.unwrap_or(0),
        samples: options.samples.unwrap_or(100),
        smt_timeout_ms: options.smt_timeout_ms,
        beacon_budget: options.beacon_budget,
        beacon_deadline: options.beacon_deadline,
        tier: options.tier.to_string(),
        only: options.only.map(str::to_string),
        invariant_min_rate: options.invariant_min_rate,
        max_attempts: options.max_attempts,
        runtime: &chelis_std_bundle::EMBEDDED_RUNTIME,
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

pub(super) fn run_surf_linked_properties_shared(
    path: &Path,
    all_decls: &[Decl],
    entry_decls: &[Decl],
    trusted_contract_decls: &[Decl],
    display_names: &BTreeMap<String, String>,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    // The decls handed in here are the reef LINKER's output: every def, the
    // `@property` itself, and the imports carry the linker's internal
    // `pkg__<pkg>__<Module>__<def>` name format. The shared runner re-checks
    // those decls (the Tier-C fuzz path re-serializes each sample and re-checks
    // it through `eval_selected`), so without the linked-program provenance
    // flag the checker's `detect_forged_linker_names` rejects the linker's own
    // names as forged (chelis#580). Install the guard the SAME way the sibling
    // obligation path does in `check_linked_decls`. The flag is thread-local;
    // every check the shared runner performs for this call runs synchronously
    // on this thread (the only worker thread the prove stack spawns is the
    // external SMT subprocess's stdout drain, which never runs the in-process
    // type-check), so holding the guard across the call covers it.
    let _linked = chelis_types::install_linked_program_guard();
    let mut outcomes = match run_surf_decls_properties_with_contract_decls(
        all_decls,
        entry_decls,
        all_decls,
        trusted_contract_decls,
        &run_opts(options),
    ) {
        Ok(PropertyRunResult::Ran(o)) => o,
        Err(message) => return emit_discovery_error(path, &message, options, totals),
    };
    for outcome in &mut outcomes {
        if let Some(display_name) = display_names.get(&outcome.name) {
            outcome.name = display_name.clone();
        }
    }
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
        // `run_deep_source_properties` returns `Err` for two distinct
        // reasons. A parse failure (prefixed `parse:`) already surfaced
        // upstream -- `prove_deep_file` parses the module first and returns
        // that error before this runs -- so it is benign here. Every OTHER
        // `Err` is a malformed-yet-parseable `@property` discovery error (a
        // missing `property_quantifiers`, a quantifier/param mismatch, an
        // absent/invalid `property_source_kind`, or a non-callable fn body).
        // `validate_deep` does not check property-metadata wellformedness, so
        // those reach here un-surfaced; swallowing them as a pass would report
        // the malformed property as PASSED (exit 0), re-introducing the
        // silent skip the shared discoverer's `Err` exists to prevent.
        // Surface it as a prove error (exit 3), the way the obligation path
        // surfaces its discovery/check errors.
        Err(message) if is_parse_error(&message) => return Status::Passed,
        Err(message) => return emit_discovery_error(path, &message, options, totals),
    };
    render_all(path, &outcomes, "user", options, totals)
}

/// Whether a `run_deep_source_properties` error is a parse failure (which
/// already surfaced upstream in `prove_deep_file`) rather than a
/// malformed-property discovery error. The shared runner prefixes parse
/// errors with `parse:` (see `chelis_prove::property_runner`).
fn is_parse_error(message: &str) -> bool {
    message.starts_with("parse:")
}

/// Emit a malformed-`@property` discovery failure as a prove error record and
/// fold an error into `totals`, mirroring the obligation path's
/// type-check-failure surfacing. A malformed property must lower the verdict
/// to Error (never a silent pass), so `chelis prove file.dp` exits non-zero.
fn emit_discovery_error(
    path: &Path,
    message: &str,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    totals.errors += 1;
    if options.json {
        println!(
            "{}",
            json!({
                "kind": "error",
                "stage": "property-discovery",
                "reason": format!("malformed @property; not verified: {message}"),
                "source": json!({ "kind": "user", "file": path.display().to_string() }),
            })
        );
    } else {
        eprintln!(
            "prove error: malformed @property in {}; not verified: {message}",
            path.display()
        );
    }
    Status::Error
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
            "composite_verdict": outcome.composite_verdict.as_str(),
            // chelis#422 (D2): the single `composite_verdict` token is the
            // WEAKEST badge; `qualifiers` carries the full disclosed caveat set
            // (e.g. a proof over reals modulo a fuzz contract is token
            // `proven_modulo_fuzz_validated_contract` with
            // `qualifiers:["fuzz","real_arithmetic"]`).
            "qualifiers": outcome.disclosed_qualifiers(),
            "assumptions": &outcome.assumptions,
            "samples": outcome.samples,
            "seed": outcome.seed,
            "source": json!({ "kind": source_kind, "file": path.display().to_string() }),
        });
        // chelis#436: the discharged proposition travels with the record so a
        // consumer displays exactly what was discharged. Emitted only when the
        // outcome carries a goal (a real verification outcome always does; a
        // bodiless discovery error does not), kept representable-as-absent.
        if let Some(goal) = &outcome.goal {
            value["goal"] = json!(goal);
        }
        // Machine records always carry an explicit tier. Terminal outcomes
        // use `none`, matching Tide, so consumers never reconstruct absence.
        value["proof_tier"] = json!(outcome.proof_tier.as_str());
        if let Some(method) = &outcome.sampling_method {
            value["sampling_method"] = json!(method);
            value["accepted_samples"] = json!(outcome.accepted_samples);
            value["attempted_samples"] = json!(outcome.attempted_samples);
            value["rejected_samples"] = json!(outcome.rejected_samples);
        }
        if matches!(
            outcome.proof_tier,
            PropertyTier::Smt | PropertyTier::Induction | PropertyTier::Beacon
        ) {
            value["arith_model"] = json!("real");
        }
        if let Some(evidence) = &outcome.induction_evidence {
            value["induction"] = json!(evidence);
        }
        if let Some(evidence) = &outcome.engine_evidence {
            value["engine_evidence"] = evidence.clone();
        }
        if let Some(cx) = &outcome.counterexample {
            value["counterexample"] = cx.clone();
            value["shrink_steps"] = json!(outcome.shrink_steps);
        }
        if let Some(r) = &outcome.reason {
            value["reason"] = json!(r);
        }
        // chelis#489: structured failure summary for non-passing properties.
        if status != "passed" {
            value["failure_summary"] = build_failure_summary_json(outcome, status, options);
        }
        println!("{value}");
    } else {
        match status {
            "passed" => {
                let suffix = if outcome.injected { " (injected)" } else { "" };
                if outcome.proof_tier == PropertyTier::Beacon {
                    println!(
                        "property: {} -- {} (real arithmetic on stored weights; floating execution not covered)",
                        outcome.name,
                        outcome.composite_verdict.as_str()
                    );
                } else if outcome.proof_tier == PropertyTier::Induction {
                    println!(
                        "property: {} -- proved (induction: base + step){suffix}",
                        outcome.name
                    )
                } else {
                    println!(
                        "property: {} -- {}/{} passed{suffix}",
                        outcome.name, outcome.samples, outcome.samples
                    )
                }
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

/// Build the `failure_summary` JSON value for a non-passing property (chelis#489).
fn build_failure_summary_json(
    outcome: &PropertyOutcome,
    status: &str,
    options: &ProveOptions<'_>,
) -> serde_json::Value {
    let requested_tier = options.tier;
    let actual_tier = outcome.proof_tier.as_str();
    let degradation = if requested_tier != actual_tier
        && !(requested_tier == "beacon-only" && actual_tier == "beacon")
        && actual_tier != "none"
    {
        Some(json!({
            "degraded": true,
            "reason": format!("requested tier '{}' fell back to '{}'", requested_tier, actual_tier),
            "from_tier": requested_tier,
            "to_tier": actual_tier,
        }))
    } else {
        None
    };
    let mut summary = json!({
        "status": status,
        "actual_tier": actual_tier,
    });
    if requested_tier != actual_tier {
        summary["requested_tier"] = json!(requested_tier);
    }
    if let Some(d) = degradation {
        summary["degradation"] = d;
    }
    if let Some(cx) = &outcome.counterexample {
        summary["counterexample"] = cx.clone();
    }
    summary["seed"] = json!(outcome.seed);
    summary["samples"] = json!(outcome.samples);
    summary["timeout_ms"] = if outcome.proof_tier == PropertyTier::Beacon {
        json!(options.beacon_budget.as_millis())
    } else {
        json!(options.smt_timeout_ms)
    };
    summary
}
