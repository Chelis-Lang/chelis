//! Phase-1 entry hard edge for chelis#729's typed wire-schema census.
//!
//! Owning contract: `spec/design/dtype_semantics.md` section C6. This test
//! inventories carrier shape only. Root identity, manifest ordering,
//! dotted-root expansion, `requires_main`, artifact routing, and `HostReason`
//! remain chelis#912's authority.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::Deserialize;

#[path = "../../../tests/support/capacity_census_authority.rs"]
mod capacity_census_authority;
#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;
use capacity_census_authority::{
    AuthorityRegistries, NumericOperationRegistration, StaticSurfaceDescriptor, SurfaceDescriptor,
};

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

/// Post-ratchet wire fields whose numeric capacity is part of an exact
/// registered operation rather than the frozen 2026-08-04 carrier cohort.
///
/// This is deliberately separate from `FROZEN_WIRE_ROWS` and the JSON
/// baseline: a new field cannot copy the old permanent disposition. The
/// Count is the first real wire descriptor routed through #1288's shared
/// exact final-authority classifier.
const WIRE_CENSUS_FAMILY: &str = "wire-schema";
const COUNT_WIRE_SURFACE: StaticSurfaceDescriptor = StaticSurfaceDescriptor::new(
    WIRE_CENSUS_FAMILY,
    "wire-schema-numeric-field",
    "chelis_compiler_api::schema::WireRiscOp::Count.axes: Vec<usize>",
    &["numeric-field"],
);
const REGISTERED_WIRE_ROWS: &[NumericOperationRegistration] = &[NumericOperationRegistration {
    surface: COUNT_WIRE_SURFACE,
    atom: "[05-OP-29]",
    authority_anchor: "count(x, axes...)",
}];

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
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
    let python = managed_python::managed_python(&root).unwrap_or_else(|error| panic!("{error}"));
    Command::new(python)
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

fn registered_surface_rows() -> Vec<SurfaceRow> {
    REGISTERED_WIRE_ROWS
        .iter()
        .map(|registration| SurfaceRow {
            kind: registration.surface.kind.to_string(),
            id: registration.surface.id.to_string(),
            flags: registration
                .surface
                .flags
                .iter()
                .map(|flag| (*flag).to_string())
                .collect(),
        })
        .collect()
}

fn expected_current_surface_rows() -> Vec<SurfaceRow> {
    let mut rows = frozen_surface_rows();
    rows.extend(registered_surface_rows());
    rows.sort();
    rows
}

fn registered_wire_problem(
    registrations: &[NumericOperationRegistration],
    spec: &str,
) -> Option<String> {
    let surface = SurfaceDescriptor::new(
        COUNT_WIRE_SURFACE.family,
        COUNT_WIRE_SURFACE.kind,
        COUNT_WIRE_SURFACE.id,
        COUNT_WIRE_SURFACE.flags,
    );
    capacity_census_authority::classify_final_authority(
        &surface,
        AuthorityRegistries {
            nonnumeric: &[],
            tagged_transports: &[],
            numeric_operations: registrations,
        },
        spec,
    )
    .map(|_| ())
    .err()
}

fn current_authority_problem(current: &[SurfaceRow], spec: &str) -> Option<String> {
    let legacy = frozen_surface_rows();
    for row in current {
        if legacy.contains(row) {
            continue;
        }
        let surface = SurfaceDescriptor {
            family: WIRE_CENSUS_FAMILY.to_string(),
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.flags.clone(),
        };
        if let Err(problem) = capacity_census_authority::classify_final_authority(
            &surface,
            AuthorityRegistries {
                nonnumeric: &[],
                tagged_transports: &[],
                numeric_operations: REGISTERED_WIRE_ROWS,
            },
            spec,
        ) {
            return Some(problem);
        }
    }
    None
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
    let spec = std::fs::read_to_string(workspace_root().join("spec/05-risc-primitives.md"))
        .expect("read controlling numbered spec");
    assert_eq!(
        registered_wire_problem(REGISTERED_WIRE_ROWS, &spec),
        None,
        "post-ratchet wire numeric fields require exact semantic registration"
    );

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
        current_authority_problem(&current, &spec),
        None,
        "every non-legacy wire row requires exactly one final authority"
    );
    assert_eq!(
        current,
        expected_current_surface_rows(),
        "public serialized numeric wire carrier shape changed. Do not decide root/manifest \
         semantics here. For the carrier itself, use the tagged payload, remove the new \
         numeric channel, or register its exact successor descriptor and governing \
         numbered atom without changing the frozen permanent cohort"
    );
}

#[test]
fn count_wire_axes_are_registered_without_inheriting_the_permanent_disposition() {
    let baseline_bytes = baseline_bytes();
    let mut baseline: serde_json::Value =
        serde_json::from_slice(&baseline_bytes).expect("wire baseline JSON");
    baseline["rows"]
        .as_array_mut()
        .expect("wire rows")
        .push(serde_json::json!({
            "kind": REGISTERED_WIRE_ROWS[0].surface.kind,
            "id": REGISTERED_WIRE_ROWS[0].surface.id,
            "flags": REGISTERED_WIRE_ROWS[0].surface.flags,
        }));
    let copied = serde_json::to_vec(&baseline).expect("serialize copied disposition");
    let problem = permanent_baseline_problem(&copied)
        .expect("Count axes cannot inherit the permanent baseline disposition");
    assert!(
        problem.contains("permanent kind/id/flags manifest"),
        "{problem}"
    );

    let spec = std::fs::read_to_string(workspace_root().join("spec/05-risc-primitives.md"))
        .expect("read controlling numbered spec");
    assert_eq!(registered_wire_problem(REGISTERED_WIRE_ROWS, &spec), None);

    let mut wrong_atom = REGISTERED_WIRE_ROWS.to_vec();
    wrong_atom[0].atom = "[05-OP-28]";
    let problem = registered_wire_problem(&wrong_atom, &spec)
        .expect("a semantically adjacent atom cannot authorize Count axes");
    assert!(problem.contains("unrelated"), "{problem}");
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
    let python = managed_python::managed_python(&root).unwrap_or_else(|error| panic!("{error}"));
    let output = Command::new(python)
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
