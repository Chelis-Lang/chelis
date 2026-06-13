//! CLI-side execution of derived producer obligations (RFC D-OBLIG,
//! D-TIERB, D-PARITY).
//!
//! The collection + synthesis + Tier B lowering live in `chelis-prove`
//! (shared with tide). This module is the thin CLI driver: it desugars the
//! Surf module to Deep, runs the checker for INFERRED return types, asks
//! chelis-prove for the obligation set, then runs each obligation through
//! Tier B (when the `smt` feature is built) and falls to Tier C otherwise.
//! Each obligation emits the additive `{kind:"obligation", ...}` JSON
//! record (D-OBLIG); the summary's `obligations` count is incremented.
//!
//! Gated on the `chelis-prove` optional dependency (the same gate the
//! Tier B SMT lowering uses).
#![cfg(feature = "chelis-prove")]

use std::collections::BTreeMap;

use chelis_surf::ast::Decl;
use chelis_types::types::Type;
use serde_json::json;

use super::{ProveOptions, Status, Summary, matches_filter};

/// Run every derived producer obligation discovered in `decls` (a flat,
/// module-stripped Surf decl list). Returns the combined obligation
/// status. Obligation errors (covered-or-rejected, signature rejection)
/// emit an error record and produce `Status::Error`.
pub(super) fn run_obligations(
    raw_decls: &[Decl],
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    // Re-parse from the original module-bearing decls: obligations need
    // the lexical module + export wrappers, so desugar the FULL program.
    let exprs = chelis_surf::desugar::desugar_program(raw_decls);

    // Inferred return types: the D-PRODUCER inferred-type path. If the
    // program does not type-check, obligation collection is skipped (the
    // user property path reports the type errors).
    let sigs: BTreeMap<String, Type> = match chelis_types::check_typed_program(&exprs) {
        Ok(checked) => checked
            .signature_inference()
            .functions
            .iter()
            .map(|(n, s)| (n.clone(), s.checked_signature.clone()))
            .collect(),
        Err(_) => return Status::Passed,
    };

    let invariants = chelis_prove::opaque::collect_opaque_invariants(&exprs);
    if invariants.is_empty() {
        return Status::Passed;
    }
    let consts = resolve_module_constants(&exprs, &invariants);
    let collection = chelis_prove::obligations::collect_obligations(&exprs, &invariants, &sigs);

    let mut status = Status::Passed;

    // Covered-or-rejected / signature-rejection errors: emit + Error.
    for err in &collection.errors {
        // Filter: obligation-name selection does not apply to declaration
        // errors; they always surface (a sealed soundness gap).
        totals.errors += 1;
        if options.json {
            println!(
                "{}",
                json!({
                    "kind": "obligation",
                    "obligation_kind": "invariant_producer",
                    "status": "error",
                    "reason": err.to_string(),
                })
            );
        } else {
            println!("obligation error: {err}");
        }
        status = super::combine_status(status, Status::Error);
    }

    for ob in &collection.obligations {
        if !matches_filter(&ob.name, options.only) {
            continue;
        }
        let inv = invariants
            .iter()
            .find(|i| i.type_name == ob.source_type)
            .expect("obligation references a collected invariant");
        let ob_status = run_one_obligation(&exprs, inv, ob, &sigs, &consts, options, totals);
        status = super::combine_status(status, ob_status);
    }
    status
}

fn run_one_obligation(
    exprs: &[chelis_deep::ast::Expr],
    inv: &chelis_prove::opaque::OpaqueInvariant,
    ob: &chelis_prove::obligations::ObligationProperty,
    sigs: &BTreeMap<String, Type>,
    consts: &chelis_prove::opaque::ConstEnv,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    totals.total += 1;
    totals.obligations += 1;
    let seed = options.seed.unwrap_or(0);

    // Tier B: attempt the SMT proof of the obligation (built only under
    // the `smt` feature; without it solve_property returns Timeout and we
    // fall to Tier C).
    if options.tier == "auto" || options.tier == "smt-only" {
        let pparams = producer_param_types(sigs, &ob.producer);
        if let Some(lowered) =
            chelis_prove::tier_b_lower::lower_obligation(exprs, inv, ob, &pparams, consts)
        {
            match chelis_prove::tier_b::solve_property(&lowered.property, options.smt_timeout_ms) {
                chelis_prove::tier_b::TierBResult::Proved => {
                    totals.passed += 1;
                    emit_obligation(options, ob, "passed", "smt", 0, seed, None, None);
                    return Status::Passed;
                }
                chelis_prove::tier_b::TierBResult::Disproved(model) => {
                    totals.failed += 1;
                    emit_obligation(options, ob, "failed", "smt", 0, seed, Some(model), None);
                    return Status::Failed;
                }
                chelis_prove::tier_b::TierBResult::Timeout
                | chelis_prove::tier_b::TierBResult::Unknown => {
                    if options.tier == "smt-only" {
                        totals.unsupported += 1;
                        emit_obligation(
                            options,
                            ob,
                            "unsupported",
                            "smt",
                            0,
                            seed,
                            None,
                            Some("smt timeout/unknown".to_string()),
                        );
                        return Status::Unsupported;
                    }
                    // Fall through to Tier C.
                }
            }
        }
    }

    // Tier C fallback: evaluate the obligation body over sampled producer
    // inputs through the interpreter (reuses the user-property eval path).
    super::tier_c_obligation::run_obligation_tier_c(exprs, inv, ob, sigs, consts, options, totals)
}

/// Build the producer's parameter type list for the Tier B lowering.
fn producer_param_types(
    sigs: &BTreeMap<String, Type>,
    producer: &str,
) -> Vec<(String, chelis_prove::tier_b_lower::ProducerParamType)> {
    use chelis_prove::tier_b_lower::ProducerParamType;
    match sigs.get(producer) {
        Some(Type::Fn(args, _)) => args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let pt = match a {
                    Type::Prim(p) => ProducerParamType::Scalar(prim_name(p)),
                    _ => ProducerParamType::Other,
                };
                // The lowering substitutes by producer param name; here we
                // pass a placeholder name and the lowering replaces it with
                // the producer's own param symbol (it looks the producer
                // body up itself), so the name only needs to be stable.
                (format!("__arg{i}"), pt)
            })
            .collect(),
        _ => vec![],
    }
}

fn prim_name(p: &chelis_types::types::Prim) -> String {
    format!("{p:?}").to_lowercase()
}

/// Resolve in-module zero-argument constant defs referenced by any
/// invariant predicate to concrete f64 values, via the evaluator. Only
/// constants whose bodies evaluate to a scalar are resolved; the rest are
/// left out (the predicate then does not lower and falls to Tier C).
fn resolve_module_constants(
    exprs: &[chelis_deep::ast::Expr],
    invariants: &[chelis_prove::opaque::OpaqueInvariant],
) -> chelis_prove::opaque::ConstEnv {
    use chelis_prove::opaque::ConstEnv;
    let mut env = ConstEnv::new();
    let referenced = referenced_constants(invariants);
    if referenced.is_empty() {
        return env;
    }
    let source = chelis_deep::printer::print_canonical(exprs);
    for name in referenced {
        if let Some(value) = eval_scalar_const(&source, &name) {
            env.insert(name, value);
        }
    }
    env
}

/// The free constant names referenced across all invariant predicates
/// (every predicate free var that is not the binder).
fn referenced_constants(invariants: &[chelis_prove::opaque::OpaqueInvariant]) -> Vec<String> {
    let mut out = Vec::new();
    for inv in invariants {
        for v in chelis_prove::predicate_free_vars(&inv.predicate) {
            if v != inv.binder && !out.contains(&v) {
                out.push(v);
            }
        }
    }
    out
}

/// Evaluate an in-module zero-arg constant def to an f64 by adding a probe
/// root that reads it.
fn eval_scalar_const(source: &str, name: &str) -> Option<f64> {
    use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
    let probe = "__chelis_const_probe";
    let probe_source = format!("{source}\n({probe} = {name})\n");
    let _ = probe_source;
    // The Deep source is canonical s-expressions; append a Deep def probe.
    let deep_probe = format!("{source}\n(def {{}} {probe} (var {{}} {name}))\n");
    let result = chelis_compiler_api::compiler::eval_selected(
        EvalRequest {
            source_kind: SourceKind::Deep,
            source: deep_probe,
            bindings: Default::default(),
        },
        &[probe.to_string()],
    )
    .ok()?;
    match result.roots.as_slice() {
        [root] => match &root.value {
            ExecutionValue::Tensor { value } if value.shape.is_empty() && value.data.len() == 1 => {
                Some(value.data[0])
            }
            _ => None,
        },
        _ => None,
    }
}

/// Emit the additive `{kind:"obligation", ...}` JSON record (D-OBLIG) or
/// the plain-text equivalent.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_obligation(
    options: &ProveOptions<'_>,
    ob: &chelis_prove::obligations::ObligationProperty,
    status: &str,
    proof_tier: &str,
    samples: usize,
    seed: u64,
    counterexample: Option<serde_json::Value>,
    reason: Option<String>,
) {
    if options.json {
        let mut value = json!({
            "kind": "obligation",
            "obligation_kind": ob.meta.obligation_kind,
            "source_type": ob.meta.source_type,
            "producer": ob.meta.producer,
            "name": ob.name,
            "status": status,
            "proof_tier": proof_tier,
            "samples": samples,
            "seed": seed,
        });
        if proof_tier == "smt" {
            value["arith_model"] = json!("real");
        }
        if let Some(cx) = counterexample {
            value["counterexample"] = cx;
        }
        if let Some(r) = reason {
            value["reason"] = json!(r);
        }
        println!("{value}");
    } else {
        match status {
            "passed" => println!("obligation: {} -- proved ({proof_tier})", ob.name),
            "failed" => println!(
                "obligation failure: {} ({proof_tier} counterexample)",
                ob.name
            ),
            "unsupported" => println!(
                "obligation unsupported: {}: {}",
                ob.name,
                reason.unwrap_or_default()
            ),
            _ => {}
        }
    }
}

// Re-export for the Tier C obligation runner so it can emit records too.
pub(super) use emit_obligation as emit_obligation_record;
