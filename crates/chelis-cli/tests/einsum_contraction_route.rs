//! chelis#3736 / chelis#3703: a compiled float `einsum` whose [05-OP-33]
//! contraction is spec/05 §4.1's product sum over one reduction label runs
//! the emitted product-sum kernel `matmul` uses, with the bits `chelis eval`
//! computes; every other einsum keeps the runtime kernel.
//!
//! The fixture covers both sides. Its routed definitions span f32, f64, f16
//! and bf16, an explicit f64 accumulator, permuted and swapped operands,
//! batch and runtime row extents (including zero rows), a 257-leaf tree,
//! and inf, NaN and signed-zero products. Its retained definitions are a diagonal, two and
//! zero reduction labels, a matrix-vector product, a runtime contraction
//! extent and an integer contraction.

mod common;

use common::{authored_host_body_definition, gcc_available};
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

use assert_cmd::Command;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/einsum_contraction/route.ch"
);

const ROUTED: &[&str] = &[
    "rows",
    "transposed",
    "swapped",
    "batched",
    "half",
    "brain",
    "wide",
    "runtime_rows",
    "deep",
    "special",
];

const RETAINED: &[&str] = &[
    "diagonal_scale",
    "frobenius",
    "outer_product",
    "matvec",
    "runtime_depth",
    "integer",
];

fn stdout(output: &std::process::Output, lane: &str) -> String {
    assert!(
        output.status.success(),
        "{lane} failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).expect("utf-8 output")
}

/// One definition's owned body: from its definition to the first closing
/// brace in column zero, which ends a top-level C function.
fn body<'a>(emitted: &'a str, name: &str) -> &'a str {
    let rest = authored_host_body_definition(emitted, name);
    &rest[..rest.find("\n}\n").expect("the body closes")]
}

fn build(out_dir: &Path) -> String {
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            FIXTURE,
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build");
    stdout(&built, "build");
    std::fs::read_to_string(out_dir.join("route.c")).expect("generated C")
}

/// Each routed definition's body calls no runtime einsum and the program
/// has the blocked product-sum kernel; each retained one still calls it.
#[test]
fn only_contraction_form_einsums_leave_the_runtime_kernel() {
    let dir = tempdir().expect("tempdir");
    let emitted = build(&dir.path().join("out"));
    for name in ROUTED {
        let body = body(&emitted, name);
        assert!(
            !body.contains("chelis_tensor_einsum("),
            "`{name}` still calls the runtime einsum kernel:\n{body}"
        );
    }
    for name in RETAINED {
        let body = body(&emitted, name);
        assert!(
            body.contains("chelis_tensor_einsum("),
            "`{name}` left the runtime einsum kernel:\n{body}"
        );
    }
    assert_eq!(
        emitted.matches("chelis_tensor_einsum(").count(),
        RETAINED.len(),
        "runtime einsum calls in the generated C"
    );
    assert!(
        emitted.contains("__sum_columns_"),
        "no blocked product-sum kernel in the generated C"
    );
}

/// The compiled program prints exactly what `chelis eval` prints. Printed
/// floats are shortest round-trip, so equal text is equal finite bits, equal
/// infinities and signed zeros, and NaN in the same positions.
#[test]
fn routed_and_retained_einsums_match_eval_bit_for_bit() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("out");
    build(&out_dir);
    let compiled = StdCommand::new(out_dir.join("route"))
        .output()
        .expect("compiled binary should run");
    let compiled = stdout(&compiled, "compiled");
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--timeout", "300", "--file", FIXTURE])
        .output()
        .expect("eval");
    let eval = stdout(&eval, "eval");
    let bindings = eval.lines().filter(|line| line.starts_with("r_")).count();
    assert_eq!(bindings, ROUTED.len() + RETAINED.len() + 1, "{eval}");
    assert_eq!(compiled, eval);
}
