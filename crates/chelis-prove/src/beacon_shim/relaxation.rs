//! Exact graph transport for the scalar real-arithmetic relaxation lane.
use super::*;
use chelis_compiler_api::schema::{WireDag, WireRiscOp};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

impl BeaconShim {
    pub(super) fn discharge_scalar_upper(&self, goal: &Goal, timeout_ms: u64) -> Discharge {
        let request = match self.scalar_request(goal, timeout_ms) {
            Ok(request) => request,
            Err(reason) => {
                return untrusted_error(&reason, json!({"engine":"beacon", "error":reason}));
            }
        };
        let bytes = serde_json::to_vec(&request).expect("owned JSON serializes");
        let request_hash = sha256_hex(&bytes);
        let dag_hash =
            sha256_hex(&serde_json::to_vec(&request["dag"]).expect("owned nested DAG serializes"));
        let folded_root = request["dag"]["roots"][1]
            .as_u64()
            .expect("scalar_request supplies a folded root");
        let binary_hash = match std::fs::read(&self.binary) {
            Ok(bytes) => sha256_hex(&bytes),
            Err(error) => {
                return untrusted_error(
                    "cannot hash Beacon executable",
                    json!({"error":error.to_string()}),
                );
            }
        };
        let discharge = match self.run_beacon(&bytes, timeout_ms) {
            SubprocessOutcome::Exited { stdout, stderr } => match self.oracle_mode {
                BeaconOracleMode::ScalarCertified => map_certified_scalar(
                    &stdout,
                    &stderr,
                    &request_hash,
                    &dag_hash,
                    folded_root,
                    &request["inputs"],
                ),
                _ => map_relaxation(&stdout, &stderr, &request_hash),
            },
            SubprocessOutcome::Rejected {
                reason,
                stdout,
                stderr,
            } => untrusted_error(
                &reason,
                json!({"engine":"beacon","error":reason,"beacon_evidence":serde_json::from_str::<Value>(&stdout).ok(),
                    "stdout":stdout,"stderr":stderr,"request_sha256":request_hash}),
            ),
            SubprocessOutcome::TimedOut => untrusted_error(
                "beacon outer timeout",
                json!({"engine":"beacon", "error":"outer_timeout", "request_sha256":request_hash}),
            ),
            SubprocessOutcome::Failed { reason, stderr } => untrusted_error(
                &reason,
                json!({"engine":"beacon", "error":reason, "stderr":stderr, "request_sha256":request_hash}),
            ),
        };
        let mut evidence = discharge.evidence().clone();
        evidence["engine_binary_sha256"] = json!(binary_hash);
        evidence["graph_sha256"] = json!(goal.ir.dag_hash());
        evidence["input_box"] = request["inputs"].clone();
        evidence["search_options"] = request["options"].clone();
        // The selected goal root can differ from the graph's first source
        // output (the lower side selects an appended negation, for example).
        evidence["selected_root"] = json!(goal.ir.root_index());
        evidence["folded_root"] = request["dag"]["roots"][1].clone();
        if let GoalShape::ScalarUpperBound { upper, .. } = goal.shape {
            evidence["upper_bound"] = json!(upper);
        }
        Discharge::new(
            discharge.soundness(),
            discharge.qualifier_set().clone(),
            discharge.result().clone(),
            evidence,
        )
        .unwrap_or_else(internal_error_discharge)
    }

    fn scalar_request(&self, goal: &Goal, timeout_ms: u64) -> Result<Value, String> {
        let GoalShape::ScalarUpperBound { inputs, upper } = &goal.shape else {
            return Err("relaxation requires a scalar upper-bound goal".into());
        };
        // Revalidate public goal fields rather than trusting construction history.
        Goal::scalar_upper_bound(inputs.clone(), *upper).map_err(|e| e.to_string())?;
        let hash = goal.ir.dag_hash().ok_or("missing graph hash")?;
        let root = goal.ir.root_index().ok_or("missing root node ID")?;
        let bytes = self
            .store
            .get(hash)
            .ok_or("graph bytes missing from store")?;
        if sha256_hex(&bytes) != hash {
            return Err("graph hash mismatch".into());
        }
        let wire: WireDag = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        wire.validate_wire_contract().map_err(|e| e.to_string())?;
        let original_root = *wire
            .roots
            .first()
            .ok_or("scalar graph has no original output root")?;
        if !wire.roots.contains(&root) {
            return Err("selected scalar root is not a graph root".into());
        }
        let loads: BTreeSet<_> = wire
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                WireRiscOp::Load { name } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let names: BTreeSet<_> = inputs
            .dims
            .iter()
            .map(|(name, _, _)| name.as_str())
            .collect();
        if loads != names {
            return Err("scalar input box names must exactly match graph Loads".into());
        }
        let mut dag: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let nodes = dag["nodes"].as_array_mut().ok_or("missing nodes")?;
        let root_index = usize::try_from(root).map_err(|_| "root ID exceeds host size")?;
        let node = nodes.get(root_index).ok_or("root ID outside graph")?;
        if node["id"].as_u64() != Some(root)
            || node["output_type"] != json!({"dims":[],"precision":"f64"})
        {
            return Err("scalar upper-bound entry must have scalar f64 output".into());
        }
        // The appended bound nodes derive from the root, so they belong to
        // its owner (declaration and activation).
        let declaration = node["declaration"].clone();
        let activation = node["activation"].clone();
        let constant = nodes.len();
        let folded = constant + 1;
        // Negating a stored f64 flips only its sign bit, with no decimal conversion.
        let negative_bits = upper.as_f64_lossy().to_bits() ^ (1_u64 << 63);
        let node = |id, op, operands| {
            json!({"id":id,"op":op,"inputs":operands,
            "output_type":{"dims":[],"precision":"f64"},"shape_deps":[],"span_id":null,"merged_spans":[],
            "declaration":declaration,"activation":activation})
        };
        nodes.push(node(
            constant,
            json!({"kind":"const", "value":{"dtype":"f64","bits":format!("{negative_bits:016x}")}}),
            vec![],
        ));
        nodes.push(node(
            folded,
            json!({"kind":"add"}),
            vec![root_index, constant],
        ));
        dag["roots"] = json!([original_root, folded]);
        let inputs: BTreeMap<_, _> = inputs
            .dims
            .iter()
            .map(|(name, lo, hi)| (name, json!({"lo":lo.as_f64_lossy(),"hi":hi.as_f64_lossy()})))
            .collect();
        if self.oracle_mode == BeaconOracleMode::ScalarCertified {
            Ok(
                json!({"schema_version":"beacon.scalar_upper_bound.v1", "dag":dag,
                "inputs":inputs,"selected_root_index":1,"strict_upper":0.0,
                "options":{"budget_ms":timeout_ms.saturating_sub(5000),
                    "max_depth":8,"max_boxes":256}}),
            )
        } else {
            Ok(
                json!({"schema_version":"beacon.relu_search.v1", "dag":dag,"inputs":inputs,"clauses":[[1]],
                "options":{"budget_ms":timeout_ms.saturating_sub(5000),"max_leaves":4096,
                "seed":20260912,"depth_cap":usize::MAX,"alpha":"adaptive","branching":"input",
                "input_axis_rule":"absolute","witness_max_candidates":32}}),
            )
        }
    }
}

fn finite_interval(value: &Value) -> Option<(f64, f64)> {
    match (value["lo"].as_f64(), value["hi"].as_f64()) {
        (Some(lo), Some(hi)) if lo.is_finite() && hi.is_finite() && lo <= hi => Some((lo, hi)),
        _ => None,
    }
}

fn map_certified_scalar(
    stdout: &str,
    stderr: &str,
    request_hash: &str,
    dag_hash: &str,
    folded_root: u64,
    request_inputs: &Value,
) -> Discharge {
    let report: Value = match serde_json::from_str(stdout) {
        Ok(report) => report,
        Err(error) => {
            return untrusted_error(
                "invalid certified scalar JSON",
                json!({"engine":"beacon","error":error.to_string(),"stdout":stdout,
                    "stderr":stderr,"request_sha256":request_hash}),
            );
        }
    };
    let evidence = json!({"engine":"beacon","method":"arb_box_cover_v1",
        "request_sha256":request_hash,"semantics":"real_arithmetic_on_stored_weights",
        "float_execution_covered":false,"beacon_evidence":report,"stderr":stderr});
    if report["schema_version"] != "beacon.scalar_upper_bound.v1"
        || report["oracle"] != "arb_box_cover_256"
        || report["semantics_note"]
            != "real-valued semantics; no floating-point roundoff soundness claim"
        || report["selected_root_index"] != 1
        || report["selected_root_node_id"] != folded_root
        || report["strict_upper"] != 0.0
        || report["dag_sha256"] != dag_hash
    {
        return untrusted_error(
            "certified scalar report does not bind the request",
            evidence,
        );
    }
    if report["verdict"] == "unknown" || report["verdict"] == "unsupported" {
        return Discharge::new(
            Soundness::Untrusted,
            QualifierSet::new(),
            TierBResult::Unknown,
            evidence,
        )
        .unwrap_or_else(internal_error_discharge);
    }
    if report["verdict"] == "failed" {
        let point = &report["counterexample"];
        let in_domain = match (request_inputs.as_object(), point["inputs"].as_object()) {
            (Some(domain), Some(values)) if domain.len() == values.len() && !domain.is_empty() => {
                domain.iter().all(|(name, interval)| {
                    let Some((domain_lo, domain_hi)) = finite_interval(interval) else {
                        return false;
                    };
                    values
                        .get(name)
                        .and_then(finite_interval)
                        .is_some_and(|(lo, hi)| lo == hi && domain_lo <= lo && lo <= domain_hi)
                })
            }
            _ => false,
        };
        if point["method"] != "arb_point_256"
            || !in_domain
            || !finite_interval(&point["bound"]).is_some_and(|(lo, _)| lo > 0.0)
        {
            return untrusted_error("invalid certified scalar counterexample", evidence);
        }
        return Discharge::new(
            Soundness::SoundApproximate,
            QualifierSet::from_iter_kinds([Qualifier::RealArith]),
            TierBResult::Disproved(point.clone()),
            evidence,
        )
        .unwrap_or_else(internal_error_discharge);
    }
    let certificate = &report["certificate"];
    let Some(leaves) = certificate["leaves"].as_array() else {
        return untrusted_error("certified scalar report has no leaf certificate", evidence);
    };
    let final_bound = finite_interval(&report["final_bound"]);
    let complete = certificate["method"] == "arb_box_cover_v1"
        && !leaves.is_empty()
        && certificate["leaves_certified"].as_u64() == Some(leaves.len() as u64)
        && certificate["leaves_unresolved"].as_u64() == Some(0)
        && leaves.iter().all(|leaf| {
            finite_interval(&leaf["bound"]).is_some_and(|(_, hi)| hi < 0.0)
                && leaf["inputs"].is_object()
        });
    if report["verdict"] != "certified" || !complete || !final_bound.is_some_and(|(_, hi)| hi < 0.0)
    {
        return untrusted_error("invalid or incomplete certified scalar report", evidence);
    }
    Discharge::new(
        Soundness::SoundApproximate,
        QualifierSet::from_iter_kinds([Qualifier::RealArith]),
        TierBResult::Proved,
        evidence,
    )
    .unwrap_or_else(internal_error_discharge)
}

fn map_relaxation(stdout: &str, stderr: &str, request_hash: &str) -> Discharge {
    let report: Value = match serde_json::from_str(stdout) {
        Ok(report) => report,
        Err(error) => {
            return untrusted_error(
                "invalid relaxation JSON",
                json!({"engine":"beacon","error":error.to_string(),"stdout":stdout,
                    "stderr":stderr,"request_sha256":request_hash}),
            );
        }
    };
    if report["schema_version"] != "beacon.relu_search.v1"
        || report["semantics_note"]
            != "real-valued semantics; no floating-point roundoff soundness claim"
        || !report["tree"].is_array()
    {
        return untrusted_error(
            "unsupported relaxation report",
            json!({"engine":"beacon","report":report,"stderr":stderr,"request_sha256":request_hash}),
        );
    }
    let mut evidence = json!({"engine":"beacon","method":"linear_relaxation_back_substitution",
        "request_sha256":request_hash,"semantics":"real_arithmetic_on_stored_weights",
        "float_execution_covered":false,"beacon_evidence":report});
    let (soundness, qualifiers, result) = match report["verdict"].as_str() {
        Some("certified")
            if report["final_bound"].as_array().is_some_and(|bounds| {
                bounds.len() == 2
                    && bounds.iter().all(|bound| {
                        matches!((bound["lo"].as_f64(), bound["hi"].as_f64()),
                    (Some(lo),Some(hi)) if lo.is_finite() && hi.is_finite() && lo <= hi)
                    })
                    && bounds[1]["hi"].as_f64().is_some_and(|hi| hi < 0.0)
            }) =>
        {
            (
                Soundness::SoundApproximate,
                QualifierSet::from_iter_kinds([Qualifier::RealArith]),
                TierBResult::Proved,
            )
        }
        Some("refuted")
            if report["witness"]["confirmed"] == true
                && matches!((report["witness"]["output_bounds"]["1"]["lo"].as_f64(), report["witness"]["output_bounds"]["1"]["hi"].as_f64()),
                (Some(lo),Some(hi)) if lo.is_finite() && hi.is_finite() && 0.0 < lo && lo <= hi) =>
        {
            (
                Soundness::SoundApproximate,
                QualifierSet::from_iter_kinds([Qualifier::RealArith]),
                TierBResult::Disproved(report["witness"].clone()),
            )
        }
        Some("refuted")
            if report["witness"]["confirmed"] == true
                && report["witness"]["output_bounds"]["1"]["lo"].as_f64() == Some(0.0)
                && report["witness"]["output_bounds"]["1"]["hi"]
                    .as_f64()
                    .is_some_and(|hi| hi.is_finite() && hi >= 0.0) =>
        {
            evidence["semantic_reason"] =
                json!("boundary_witness_does_not_refute_non_strict_bound");
            (
                Soundness::Untrusted,
                QualifierSet::new(),
                TierBResult::Unknown,
            )
        }
        Some("unknown") => (
            Soundness::Untrusted,
            QualifierSet::new(),
            TierBResult::Unknown,
        ),
        _ => return untrusted_error("invalid or unconfirmed relaxation verdict", evidence),
    };
    Discharge::new(soundness, qualifiers, result, evidence).unwrap_or_else(internal_error_discharge)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn certified_report() -> Value {
        json!({"schema_version":"beacon.scalar_upper_bound.v1", "verdict":"certified",
            "oracle":"arb_box_cover_256",
            "semantics_note":"real-valued semantics; no floating-point roundoff soundness claim",
            "selected_root_index":1,"selected_root_node_id":7,"strict_upper":0.0,
            "dag_sha256":"exact-dag","final_bound":{"lo":-2.0,"hi":-0.5},
            "certificate":{"method":"arb_box_cover_v1","leaves_certified":1,
                "leaves_unresolved":0,"leaves":[{"inputs":{"s":{"lo":96.0,"hi":104.0}},
                    "bound":{"lo":-2.0,"hi":-0.5}}]},"reason":"all boxes certified"})
    }

    #[test]
    fn certified_scalar_report_binds_exact_dag_root_and_complete_leaves() {
        let mut report = certified_report();
        let inputs = json!({"s":{"lo":96.0,"hi":104.0}});
        let proved =
            map_certified_scalar(&report.to_string(), "", "request", "exact-dag", 7, &inputs);
        assert_eq!(proved.result(), &TierBResult::Proved);
        assert!(proved.qualifier_set().contains(Qualifier::RealArith));
        assert_eq!(proved.evidence()["request_sha256"], "request");
        for (field, value) in [
            ("dag_sha256", json!("wrong-dag")),
            ("selected_root_node_id", json!(6)),
            ("selected_root_index", json!(0)),
            ("strict_upper", json!(1.0)),
            ("final_bound", json!({"lo":-2.0,"hi":0.0})),
        ] {
            report[field] = value;
            let rejected =
                map_certified_scalar(&report.to_string(), "", "request", "exact-dag", 7, &inputs);
            assert_ne!(rejected.result(), &TierBResult::Proved, "{field}");
            report = certified_report();
        }
        report["certificate"]["leaves"] = json!([]);
        assert_ne!(
            map_certified_scalar(&report.to_string(), "", "request", "exact-dag", 7, &inputs)
                .result(),
            &TierBResult::Proved
        );
        report = certified_report();
        report["certificate"]["leaves_unresolved"] = json!(1);
        assert_ne!(
            map_certified_scalar(&report.to_string(), "", "request", "exact-dag", 7, &inputs)
                .result(),
            &TierBResult::Proved
        );
        report = certified_report();
        report["verdict"] = json!("unknown");
        assert_ne!(
            map_certified_scalar(&report.to_string(), "", "request", "exact-dag", 7, &inputs)
                .result(),
            &TierBResult::Proved
        );
    }

    #[test]
    fn certified_scalar_refutation_requires_strict_in_domain_point_witness() {
        let inputs = json!({"s":{"lo":96.0,"hi":104.0}});
        let mut report = certified_report();
        report["verdict"] = json!("failed");
        report["final_bound"] = Value::Null;
        report["certificate"] = Value::Null;
        report["counterexample"] = json!({"method":"arb_point_256",
            "inputs":{"s":{"lo":100.0,"hi":100.0}},
            "bound":{"lo":10.45,"hi":10.46}});
        let failed =
            map_certified_scalar(&report.to_string(), "", "request", "exact-dag", 7, &inputs);
        assert!(matches!(failed.result(), TierBResult::Disproved(_)));
        assert!(failed.qualifier_set().contains(Qualifier::RealArith));
        for (field, value) in [
            ("bound", json!({"lo":0.0,"hi":1.0})),
            ("inputs", json!({"s":{"lo":105.0,"hi":105.0}})),
            ("method", json!("sampled_f64")),
        ] {
            let mut altered = report.clone();
            altered["counterexample"][field] = value;
            let result =
                map_certified_scalar(&altered.to_string(), "", "request", "exact-dag", 7, &inputs);
            assert!(
                !matches!(result.result(), TierBResult::Disproved(_)),
                "{field}"
            );
        }
    }
    fn report() -> Value {
        json!({"schema_version":"beacon.relu_search.v1",
            "semantics_note":"real-valued semantics; no floating-point roundoff soundness claim",
            "verdict":"certified","reason":"all leaves excluded","tree":[{"status":"certified"}],
            "final_bound":[{"lo":-1.0,"hi":0.5},{"lo":-2.0,"hi":-0.5}]})
    }
    #[test]
    fn malformed_reports_retain_the_sent_request_identity() {
        let mut value = report();
        value["schema_version"] = json!("future");
        for stdout in ["not JSON".to_string(), value.to_string()] {
            let result = map_relaxation(&stdout, "diagnostic", "exact-request");
            assert_eq!(result.soundness(), Soundness::Untrusted);
            assert_eq!(result.evidence()["request_sha256"], "exact-request");
        }
    }
    #[test]
    fn certification_is_qualified_and_unknown_retains_hull_reason_and_tree() {
        let mut value = report();
        let proved = map_relaxation(&value.to_string(), "", "request");
        assert_eq!(proved.result(), &TierBResult::Proved);
        assert!(proved.qualifier_set().contains(Qualifier::RealArith));
        value["verdict"] = json!("unknown");
        value["reason"] = json!("time budget exhausted");
        let unknown = map_relaxation(&value.to_string(), "", "request");
        assert_eq!(unknown.result(), &TierBResult::Unknown);
        assert_eq!(unknown.soundness(), Soundness::Untrusted);
        assert_eq!(unknown.evidence()["beacon_evidence"], value);
    }
    #[test]
    fn malformed_proofs_and_unconfirmed_witnesses_fail_closed() {
        for (field, replacement) in [
            ("schema_version", json!("future")),
            ("semantics_note", json!("float32 sound")),
            ("tree", Value::Null),
            (
                "final_bound",
                json!([{ "lo":-1.0,"hi":1.0 },{"lo":-1.0,"hi":0.0}]),
            ),
        ] {
            let mut value = report();
            value[field] = replacement;
            assert_eq!(
                map_relaxation(&value.to_string(), "", "key").soundness(),
                Soundness::Untrusted
            );
        }
        let mut value = report();
        value["verdict"] = json!("refuted");
        value["witness"] = json!({"confirmed":false,"inputs":{"x":1.0}});
        assert_eq!(
            map_relaxation(&value.to_string(), "", "key").soundness(),
            Soundness::Untrusted
        );
        value["witness"]["confirmed"] = json!(true);
        value["witness"]["output_bounds"] = json!({"1":{"lo":0.0,"hi":0.0}});
        let boundary = map_relaxation(&value.to_string(), "", "key");
        assert_eq!(boundary.soundness(), Soundness::Untrusted);
        assert_eq!(boundary.result(), &TierBResult::Unknown);
        assert_eq!(
            boundary.evidence()["semantic_reason"],
            "boundary_witness_does_not_refute_non_strict_bound"
        );
        value["witness"]["output_bounds"]["1"]["lo"] = json!(0.1);
        value["witness"]["output_bounds"]["1"]["hi"] = json!(0.1);
        assert!(matches!(
            map_relaxation(&value.to_string(), "", "key").result(),
            TierBResult::Disproved(_)
        ));
        value["witness"]["output_bounds"]["1"]["hi"] = json!(-0.1);
        assert_eq!(
            map_relaxation(&value.to_string(), "", "key").soundness(),
            Soundness::Untrusted
        );
    }
}
