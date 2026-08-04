//! Phase-1 entry hard edge for chelis#729's typed wire-schema census.
//!
//! Owning contract: `spec/design/dtype_semantics.md` section C6. This test
//! inventories carrier shape only. Root identity, manifest ordering,
//! dotted-root expansion, `requires_main`, artifact routing, and `HostReason`
//! remain chelis#912's authority.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::Deserialize;
const PERMANENT_WIRE_DISPOSITION: &str = "permanent-disposition(C6 dtype-tagged wire schema complete descriptor set ratified 2026-08-04)";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrozenSurfaceRow {
    kind: &'static str,
    id: &'static str,
    flags: &'static [&'static str],
}

const FROZEN_WIRE_ROWS: &[FrozenSurfaceRow] = &[
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ChangeSignatureResult.rewritten_calls: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::CheckResult.score: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::CheckResult.total_nodes: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::CheckResult.typed_nodes: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::CheckResult.untyped_nodes: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::CompileResult.peak_device_bytes_estimate: Option<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::Diagnostic.severity: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::EvalResult.schema_version: u32",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::EvaluatedRoot.node_id: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ExecutionValue::Bfloat16.value: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ExecutionValue::Float16.value: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ExecutionValue::Float32.value: f32",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ExecutionValue::Float64.value: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ExecutionValue::Int16.value: i16",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ExecutionValue::Int32.value: i32",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ExecutionValue::Int64.value: i64",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::ExecutionValue::Int8.value: i8",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::FitnessComponents.names: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::FitnessComponents.parse: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::FitnessComponents.structure: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::FitnessComponents.types: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::GradResult.forward_nodes_by_name: std::collections::BTreeMap<String, usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::GradResult.grad_nodes_by_name: std::collections::BTreeMap<String, usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::GradResult.output_node: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::LowerResult.named_roots: std::collections::BTreeMap<String, usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::RenameResult.renamed_references: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::Span.len: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::Span.offset: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorElements::Bf16.0: Vec<f64>",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorElements::F16.0: Vec<f64>",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorElements::F32.0: Vec<f32>",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorElements::F64.0: Vec<f64>",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorElements::Int16.0: Vec<i16>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorElements::Int32.0: Vec<i32>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorElements::Int64.0: Vec<i64>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorElements::Int8.0: Vec<i8>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::TensorValue.shape: Vec<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDag.roots: Vec<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDag.schema_version: u32",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDagNode.id: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDagNode.inputs: Vec<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDeepAtom::Float.value: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDeepAtom::Int.value: i64",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDimExpr::Concrete.value: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDimInfo::Lit.size: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireDimInfo::Named.size: Option<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireFusedInput::External.index: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireFusedInput::PreviousStep.index: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireInferredDim::Lit.size: i64",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireInferredDim::Rank.id: u32",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireInferredDim::Var.id: u32",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireInferredPrecision::Var.id: u32",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireInferredType::Var.id: u32",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireLiteral::Float.value: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireLiteral::Int.value: i64",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireLiteral::TypedFloat.value: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireLiteral::TypedInt.value: i64",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Argmax.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Argmin.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Dropout.rate: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Dropout.seed: u64",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Expand.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Gather.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::MaxReduce.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::MinReduce.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::OneHot.vocab: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Pad.fill: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Permute.axes: Vec<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::ProdReduce.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::ReduceWindow.strides: Vec<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::ReduceWindow.window_shape: Vec<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::ReduceWindowGrad.strides: Vec<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::ReduceWindowGrad.window_shape: Vec<usize>",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Scatter.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::ScatterAdd.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::ScatterElements.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Shape.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::Sum.axis: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::UniformLike.high: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::UniformLike.low: f64",
        flags: &["float-carrier"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRiscOp::UniformLike.seed: u64",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRtDim::Lit.value: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireRtDim::Node.input: usize",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireSurfExpr::TupleGet.index: i64",
        flags: &["numeric-field"],
    },
    FrozenSurfaceRow {
        kind: "wire-schema-numeric-field",
        id: "chelis_compiler_api::schema::WireSurfExpr::Vmap.axis: Option<i64>",
        flags: &["numeric-field"],
    },
];

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct SurfaceRow {
    kind: String,
    id: String,
    flags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Baseline {
    version: u32,
    citation: String,
    rows: Vec<SurfaceRow>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("compiler-api crate lives under <workspace>/crates")
        .to_path_buf()
}

fn run_typed_enumerator() -> Output {
    let root = workspace_root();
    Command::new(root.join(".venv/bin/python"))
        .arg(root.join("scripts/capacity_census_typed.py"))
        // No `--target-dir`: the enumerator owns that choice so the wire and
        // PyO3 binding censuses cannot drift onto separate cargo target
        // directories. They previously had one each, so both compiled the
        // common chelis dependency graph from scratch, which is why they were
        // the only tests in the repo over nextest's 60s SLOW threshold.
        .arg("wire")
        .current_dir(&root)
        .output()
        .expect("run typed wire census enumerator")
}

fn baseline_bytes() -> Vec<u8> {
    std::fs::read(workspace_root().join("spec/design/capacity_census_wire.json"))
        .expect("read reviewed wire census baseline")
}

fn frozen_surface_rows() -> Vec<SurfaceRow> {
    FROZEN_WIRE_ROWS
        .iter()
        .map(|row| SurfaceRow {
            kind: row.kind.to_string(),
            id: row.id.to_string(),
            flags: row.flags.iter().map(|flag| (*flag).to_string()).collect(),
        })
        .collect()
}

fn permanent_baseline_problem(bytes: &[u8]) -> Option<String> {
    let baseline: Baseline = match serde_json::from_slice(bytes) {
        Ok(baseline) => baseline,
        Err(error) => return Some(format!("wire baseline is not valid JSON: {error}")),
    };
    if baseline.version != 1 {
        return Some(format!(
            "unknown wire census baseline version {}",
            baseline.version
        ));
    }
    if baseline.citation != PERMANENT_WIRE_DISPOSITION {
        return Some(format!(
            "wire disposition changed: expected {PERMANENT_WIRE_DISPOSITION:?}, got {:?}",
            baseline.citation
        ));
    }
    if baseline.rows != frozen_surface_rows() {
        return Some(
            "wire baseline rows differ from the hand-maintained permanent kind/id/flags manifest"
                .to_string(),
        );
    }
    None
}

#[test]
fn wire_schema_numeric_fields_match_the_reviewed_baseline() {
    let baseline_bytes = baseline_bytes();
    assert!(
        permanent_baseline_problem(&baseline_bytes).is_none(),
        "wire census guard changed: regenerating or editing the baseline is not the fix; \
         review the carrier under dtype_semantics.md section C6 and update the frozen \
         complete hand-maintained descriptor manifest only with that disposition: {:?}",
        permanent_baseline_problem(&baseline_bytes)
    );
    let baseline: Baseline = serde_json::from_slice(&baseline_bytes).expect("wire baseline JSON");
    assert_eq!(baseline.version, 1, "unknown wire census baseline version");
    assert_eq!(baseline.citation, PERMANENT_WIRE_DISPOSITION);

    let output = run_typed_enumerator();
    assert!(
        output.status.success(),
        "typed wire census failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let current: Vec<SurfaceRow> =
        serde_json::from_slice(&output.stdout).expect("typed wire census JSON");
    assert_eq!(
        current, baseline.rows,
        "public serialized numeric wire carrier shape changed. Do not decide root/manifest \
         semantics here. For the carrier itself, use the tagged payload, remove the new \
         numeric channel, or obtain the explicit C6 review disposition; then update the \
         reviewed baseline and frozen descriptor manifest together"
    );
}

#[test]
fn permanent_wire_disposition_is_bound_to_the_complete_descriptor_set() {
    let baseline_bytes = baseline_bytes();
    let mut baseline: serde_json::Value =
        serde_json::from_slice(&baseline_bytes).expect("wire baseline JSON");
    baseline["rows"]
        .as_array_mut()
        .expect("wire rows")
        .push(serde_json::json!({
            "kind": "wire-schema-numeric-field",
            "id": "reviewer::Added.value: f64",
            "flags": ["numeric-field"]
        }));
    let mutated = serde_json::to_vec(&baseline).expect("serialize mutated wire baseline");
    let problem = permanent_baseline_problem(&mutated)
        .expect("copying the permanent top-level disposition onto an added row must fail");
    assert!(
        problem.contains("permanent kind/id/flags manifest"),
        "the production manifest guard must reject the copied disposition: {problem}"
    );
}

#[test]
fn a_typed_permanent_disposition_cannot_move_between_families() {
    let baseline_bytes = baseline_bytes();
    let mut baseline: serde_json::Value =
        serde_json::from_slice(&baseline_bytes).expect("wire baseline JSON");
    baseline["citation"] = serde_json::json!(
        "permanent-disposition(C6 registered PyO3 signature surface complete descriptor set ratified 2026-08-04)"
    );
    let mutated = serde_json::to_vec(&baseline).expect("serialize wrong-family wire baseline");
    let problem = permanent_baseline_problem(&mutated)
        .expect("the binding disposition must not ratify a wire manifest");
    assert!(problem.contains("wire disposition changed"), "{problem}");
}

#[test]
fn adding_or_removing_a_public_serialized_f64_field_changes_the_census() {
    let root = workspace_root();
    let output = Command::new(root.join(".venv/bin/python"))
        .arg(root.join("scripts/test_capacity_census_typed.py"))
        .arg("WireEnumerator")
        .current_dir(&root)
        .output()
        .expect("run typed wire mutation tests");
    assert!(
        output.status.success(),
        "wire add/remove mutation controls failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
