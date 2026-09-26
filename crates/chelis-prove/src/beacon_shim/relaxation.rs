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
            SubprocessOutcome::Exited { stdout, stderr } => {
                map_relaxation(&stdout, &stderr, &request_hash)
            }
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
        evidence["output_root"] = request["dag"]["roots"][0].clone();
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
        dag["roots"] = json!([root_index, folded]);
        let inputs: BTreeMap<_, _> = inputs
            .dims
            .iter()
            .map(|(name, lo, hi)| (name, json!({"lo":lo,"hi":hi})))
            .collect();
        Ok(
            json!({"schema_version":"beacon.relu_search.v1", "dag":dag,"inputs":inputs,"clauses":[[1]],
            "options":{"budget_ms":timeout_ms.saturating_sub(5000),"max_leaves":4096,
            "seed":20260912,"depth_cap":usize::MAX,"alpha":"adaptive","branching":"input",
            "input_axis_rule":"absolute","witness_max_candidates":32}}),
        )
    }
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
