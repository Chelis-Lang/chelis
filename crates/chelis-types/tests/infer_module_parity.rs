//! Behavioral inference baselines over lossless original-production AST fixtures.
//!
//! Accepted checks preserve their annotated Deep, type environment, signature
//! inference, and visit counters. Rejected checks preserve ordered diagnostic
//! kinds, severity, and locations, without freezing diagnostic prose.
//!
//! The recorded baselines live in `tests/fixtures/infer_parity/`. A missing
//! baseline is written and reported as a failure, never silently accepted.

use std::fs;
use std::path::PathBuf;

use chelis_deep::Expr;
use chelis_deep::printer::print_canonical;

use chelis_types::{CheckedProgram, InferResult, check_ir_program};

fn parse_core(source: &str, _label: &str) -> Vec<Expr> {
    serde_json::from_str(source).expect("lossless core fixture")
}

/// The full observable output of an accepted check: the annotated tree (which
/// carries the type metadata inline), the type environment, and the honest
/// visit counters.
fn accepted_snapshot(checked: &CheckedProgram) -> String {
    let mut out = String::new();

    out.push_str("== annotated deep ==\n");
    out.push_str(&print_canonical(checked.annotated_exprs()));
    if !out.ends_with('\n') {
        out.push('\n');
    }

    out.push_str("\n== type env ==\n");
    let mut env: Vec<(&String, String)> = checked
        .type_env()
        .iter()
        .map(|(name, ty)| (name, chelis_deep::printer::print_expr_flat(ty)))
        .collect();
    env.sort();
    for (name, ty) in env {
        out.push_str(&format!("{name} : {ty}\n"));
    }

    out.push_str("\n== signature inference ==\n");
    let mut functions: Vec<_> = checked.signature_inference().functions.iter().collect();
    functions.sort_by_key(|(name, _)| *name);
    for (binding, function) in functions {
        out.push_str(&format!("binding = {binding}\n"));
        out.push_str(&format!("  name = {}\n", function.name));
        out.push_str(&format!(
            "  authored_signature = {}\n",
            function.authored_signature
        ));
        out.push_str(&format!(
            "  authored_signature_type = {:?}\n",
            function.authored_signature_type
        ));
        out.push_str(&format!(
            "  recursive_cycle = {}\n",
            function.recursive_cycle
        ));
        out.push_str(&format!(
            "  checked_signature = {:?}\n",
            function.checked_signature
        ));
        out.push_str(&format!(
            "  display_signature = {:?}\n",
            function.display_signature
        ));
        for param in &function.params {
            out.push_str(&format!(
                "  param[{}] name = {}; written = {}; inferred_read_only = {}; \
                 checked_type = {:?}; display_type = {:?}\n",
                param.index,
                param.name,
                param.written,
                param.inferred_read_only,
                param.checked_type,
                param.display_type,
            ));
        }
    }

    out.push_str("\n== infer stats ==\n");
    let stats = checked.infer_stats();
    out.push_str(&format!(
        "typed_nodes = {}\ntotal_nodes = {}\n",
        stats.typed_nodes, stats.total_nodes
    ));

    out
}

/// Ordered diagnostic kinds, severity, and locations, independent of prose.
fn rejected_snapshot(report: &InferResult) -> String {
    let mut out = format!(
        "typed_nodes = {}\ntotal_nodes = {}\ndiagnostics = {}\n\n",
        report.typed_nodes,
        report.total_nodes,
        report.errors.len()
    );
    for (index, error) in report.errors.iter().enumerate() {
        out.push_str(&format!("[{index}] kind = {:?}\n", error.kind));
        out.push_str(&format!("    severity = {}\n", error.severity));
        out.push_str(&format!("    expected = {:?}\n", error.expected));
        out.push_str(&format!("    got = {:?}\n", error.got));
        out.push_str(&format!("    span_offset = {:?}\n", error.span_offset));
        out.push_str(&format!("    span_id = {:?}\n", error.span_id));
    }
    out
}

fn baseline_path(name: &str) -> PathBuf {
    let (section, test_name) = name
        .split_once('_')
        .expect("the parity fixture name has a section prefix");
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/infer_parity")
        .join(format!("infer_module_parity__{section}__{test_name}.snap"))
}

/// Compare against the recorded baseline. A missing baseline is recorded and
/// then reported as a failure: the point of the fixture is the comparison, so
/// a first run must never be mistaken for a passing run.
fn assert_matches_baseline(name: &str, actual: &str) {
    let path = baseline_path(name);
    if !path.exists() {
        fs::create_dir_all(path.parent().expect("fixture dir")).expect("create fixture dir");
        fs::write(&path, actual).expect("write baseline");
        panic!(
            "{name}: no recorded baseline. Wrote {}. Review it, then run the test again.",
            path.display()
        );
    }
    let expected = fs::read_to_string(&path).expect("read baseline");
    let shown = path.display();
    assert!(
        expected == actual,
        "{name}: inference output differs from the recorded baseline at {shown}.\n\
         The infer module split must not change checked trees, metadata, \
         statistics, or diagnostics.\n\
         --- recorded ---\n{expected}\n--- actual ---\n{actual}"
    );
}

/// A program that must type-check cleanly, compared on its whole output.
fn assert_accepted_parity(name: &str, source: &str) {
    let deep = parse_core(source, name);
    match check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(checked) => assert_matches_baseline(name, &accepted_snapshot(&checked)),
        Err(report) => {
            let msgs: Vec<&str> = report.errors.iter().map(|e| e.message.as_str()).collect();
            panic!("{name}: fixture is supposed to type-check, but it failed: {msgs:#?}");
        }
    }
}

/// A program that must be rejected, compared on its whole ordered diagnostic
/// list.
fn assert_rejected_parity(name: &str, source: &str) {
    let deep = parse_core(source, name);
    match check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => panic!("{name}: fixture is supposed to be rejected, but it type-checked"),
        Err(report) => {
            assert!(
                !report.errors.is_empty(),
                "{name}: a rejection must carry at least one diagnostic"
            );
            assert_matches_baseline(name, &rejected_snapshot(&report));
        }
    }
}

// ── Accepted fixtures ───────────────────────────────────────────────────────

#[test]
fn accepted_generic_calls() {
    assert_accepted_parity(
        "accepted_generic_calls",
        include_str!("fixtures/core/tests/infer_module_parity/accepted_generic_calls.0.json"),
    );
}

#[test]
fn accepted_shape_operations() {
    assert_accepted_parity(
        "accepted_shape_operations",
        include_str!("fixtures/core/tests/infer_module_parity/accepted_shape_operations.0.json"),
    );
}

#[test]
fn accepted_collections() {
    assert_accepted_parity(
        "accepted_collections",
        include_str!("fixtures/core/tests/infer_module_parity/accepted_collections.0.json"),
    );
}

#[test]
fn accepted_records() {
    assert_accepted_parity(
        "accepted_records",
        include_str!("fixtures/core/tests/infer_module_parity/accepted_records.0.json"),
    );
}

#[test]
fn accepted_patterns() {
    assert_accepted_parity(
        "accepted_patterns",
        include_str!("fixtures/core/tests/infer_module_parity/accepted_patterns.0.json"),
    );
}

#[test]
fn accepted_transforms() {
    assert_accepted_parity(
        "accepted_transforms",
        include_str!("fixtures/core/tests/infer_module_parity/accepted_transforms.0.json"),
    );
}

// ── Rejected fixtures ───────────────────────────────────────────────────────

#[test]
fn rejected_numeric_restrictions() {
    assert_rejected_parity(
        "rejected_numeric_restrictions",
        include_str!(
            "fixtures/core/tests/infer_module_parity/rejected_numeric_restrictions.0.json"
        ),
    );
}

#[test]
fn rejected_invalid_shapes() {
    assert_rejected_parity(
        "rejected_invalid_shapes",
        include_str!("fixtures/core/tests/infer_module_parity/rejected_invalid_shapes.0.json"),
    );
}

#[test]
fn rejected_collection_callbacks() {
    assert_rejected_parity(
        "rejected_collection_callbacks",
        include_str!(
            "fixtures/core/tests/infer_module_parity/rejected_collection_callbacks.0.json"
        ),
    );
}

#[test]
fn rejected_records() {
    assert_rejected_parity(
        "rejected_records",
        include_str!("fixtures/core/tests/infer_module_parity/rejected_records.0.json"),
    );
}

#[test]
fn rejected_patterns() {
    assert_rejected_parity(
        "rejected_patterns",
        include_str!("fixtures/core/tests/infer_module_parity/rejected_patterns.0.json"),
    );
}

#[test]
fn rejected_transforms() {
    assert_rejected_parity(
        "rejected_transforms",
        include_str!("fixtures/core/tests/infer_module_parity/rejected_transforms.0.json"),
    );
}
