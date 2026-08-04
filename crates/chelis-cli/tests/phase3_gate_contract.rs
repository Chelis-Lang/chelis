//! Phase 3 gate-contract regressions for chelis#730.
//!
//! Gates may make an emitter rejection earlier or more specific, but they
//! must not fork target policy by public entry path. These probes lock the
//! two drifted policies reported by chelis#697 and chelis#698.

use std::fs;

use assert_cmd::Command;
use chelis_compiler_api::{
    compiler::compile,
    schema::{CompileRequest, CompileTarget, SourceKind},
};
use tempfile::tempdir;

fn build(source: &str, stem: &str, target: &str) -> assert_cmd::assert::Assert {
    let dir = tempdir().expect("tempdir");
    let source_path = dir.path().join(format!("{stem}.ch"));
    let output_path = dir.path().join("out");
    fs::write(&source_path, source).expect("write source");

    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1").args([
        "build",
        source_path.to_str().expect("utf-8 source path"),
        "--target",
        target,
        "--output",
        output_path.to_str().expect("utf-8 output path"),
    ]);
    command.assert()
}

/// chelis#697: C dtype admission is a property of the lowered operation and
/// backend, never of whether an unrelated declaration happens to need the
/// host lane. Both forms use an active, exactly-supported int64 scalar.
#[test]
fn c_dtype_admission_does_not_depend_on_an_unrelated_host_function() {
    let tensor_only = "def value() -> int64 = cast(1, int64)\n";
    let with_unrelated_host =
        "def label() -> string = \"unrelated\"\ndef value() -> int64 = cast(1, int64)\n";

    build(tensor_only, "int64_tensor_only", "c").success();
    build(with_unrelated_host, "int64_with_host", "c").success();
}

/// Negative parity for the C-gate deletion: removing a stale early gate must
/// not admit a dtype the language still defers. The checker remains the
/// competent rejection boundary for f8e4m3.
#[test]
fn c_build_still_rejects_the_deferred_f8e4m3_dtype() {
    build(
        "def value(x: tensor[1, f8e4m3]) -> tensor[1, f8e4m3] = x\n",
        "deferred_f8e4m3",
        "c",
    )
    .failure()
    .stderr(predicates::str::contains("f8e4m3"));
}

/// chelis#698: the compiler API's stale copy rejected f64 before the HIP
/// emitter, while the CLI copy admitted the backend's typed f64 kernels. One
/// shared policy must admit the same supported program through both entries.
#[test]
fn hip_f64_admission_agrees_across_public_build_paths() {
    let source = "def add64(a: tensor[2, f64], b: tensor[2, f64]) -> tensor[2, f64] = \
                  add(a, b)\n";

    compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::Hip,
        entry_name: Some("add64".to_string()),
    })
    .expect("compiler API must admit the HIP backend's typed f64 add kernel");
    build(source, "hip_f64_add", "hip").success();
}

/// Negative parity for the shared HIP policy: a capability cell that remains
/// absent is rejected by the same typed early gate in the shipped CLI path.
#[test]
fn hip_scatter_elements_rejects_an_unimplemented_f64_payload_cell() {
    build(
        "def apply_scatter(data: tensor[2, 2, f64], indices: tensor[2, 2, int32], \
         updates: tensor[2, 2, f64]) -> tensor[2, 2, f64] = \
         scatter_elements(data, indices, updates, 0)\n",
        "hip_scatter_elements_f64",
        "hip",
    )
    .failure()
    .stderr(predicates::str::contains("scatter_elements"))
    .stderr(predicates::str::contains("early capability gate"));
}
