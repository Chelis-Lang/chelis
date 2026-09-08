//! chelis#1339 / [04-INF-8]: an eager top-level value may not reach a later
//! eager value through a function call or nested lambda during initialization.
//!
//! This is the issue's authoritative acceptance suite. It exercises both
//! public checker ingresses and the `check`, `prove`, `eval`, and C-build commands over
//! the same corpus. The positive control also compiles, links, and runs so a
//! checker-only repair cannot hide a remaining evaluator/backend divergence.
//!
//! Run with:
//! `cargo nextest run -p chelis-cli --test issue_1339_top_level_initialization`

use assert_cmd::Command;
use chelis_deep::parse_and_stamp;
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{check_ir_program, check_typed_program};
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{COMPILER_VERSION, build_and_run, build_and_run_app, make_app, write_file};

type Diagnostics = Vec<(String, String)>;

#[derive(Clone, Copy)]
struct RejectCase {
    name: &'static str,
    source: &'static str,
    root: &'static str,
    later: &'static str,
}

const REJECT_CASES: &[RejectCase] = &[
    RejectCase {
        name: "scalar",
        source: "module Issue1339.Scalar\n\
                 root = read_later(1i32)\n\
                 later = 5i32\n\
                 def read_later(n: int32) -> int32 = add(n, later)\n",
        root: "root",
        later: "later",
    },
    RejectCase {
        name: "list",
        source: "module Issue1339.List\n\
                 root = read_later()\n\
                 later = [1i64, 2i64]\n\
                 def read_later() -> List[int64] = later\n",
        root: "root",
        later: "later",
    },
    RejectCase {
        name: "tensor",
        source: "module Issue1339.Tensor\n\
                 root = read_later()\n\
                 later = to_tensor([1.0f32, 2.0f32])\n\
                 def read_later() -> tensor[2, f32] = later\n",
        root: "root",
        later: "later",
    },
    RejectCase {
        name: "multihop",
        source: "module Issue1339.Multihop\n\
                 root = outer()\n\
                 later = 7i32\n\
                 def outer() -> int32 = middle()\n\
                 def middle() -> int32 = add(later, 1i32)\n",
        root: "root",
        later: "later",
    },
    RejectCase {
        name: "later_external_input",
        source: "module Issue1339.External\n\
                 root = read_input()\n\
                 input: int32 = input\n\
                 def read_input() -> int32 = input\n",
        root: "root",
        later: "input",
    },
    RejectCase {
        name: "nested_lambda",
        // [04-INF-7] deliberately over-approximates this callback as
        // applied during initialization even though `discard` ignores it.
        source: "module Issue1339.Nested\n\
                 root = make_ignored()\n\
                 later = 5i32\n\
                 def make_ignored() -> int32 = discard(fn (n: int32) -> add(n, later))\n\
                 def discard(callback: int32 -> int32) -> int32 = 0i32\n",
        root: "root",
        later: "later",
    },
];

const DIRECT_FORWARD_CONTROL: RejectCase = RejectCase {
    name: "direct_forward_control",
    source: "module Issue1339.Direct\n\
             root = later\n\
             later = 5i32\n",
    root: "root",
    later: "later",
};

const DEEP_INDIRECT_SOURCE: &str = "(defsig {} read_later (t-fn {} (t-prim {} int32) (t-prim {} int32)))\n\n\
     (def {} root (app {} (var {} read_later) (lit {type: (t-prim {} int32)} 1)))\n\n\
     (def {} later (lit {type: (t-prim {} int32)} 5))\n\n\
     (def {} read_later\n\
       (fn {} (params {} (n {type: (t-prim {} int32)}))\n\
         (app {} (var {} add) (var {} n) (var {} later))))\n";

const CYCLE_SOURCE: &str = "module Issue1339.Cycle\n\
                            root = read_cycle()\n\
                            later: int32 = root\n\
                            def read_cycle() -> int32 = later\n";

const CYCLIC_ROOT_WITH_LATER_SOURCE: &str = "module Issue1339.CycleWithLater\n\
                                            root = add(read_root(), later)\n\
                                            later = 5i32\n\
                                            def read_root() -> int32 = root\n";

const CYCLIC_ROOT_WITH_UNKNOWN_SOURCE: &str = "module Issue1339.CycleWithUnknown\n\
                                              root = add(read_root(), missing)\n\
                                              def read_root() -> int32 = root\n";

const POSITIVE_SOURCE: &str = "module Issue1339.Positive\n\
                               base = 5i32\n\
                               backward = read_base(1i32)\n\
                               list_base = [1i64, 2i64]\n\
                               backward_list = read_list(0i32)\n\
                               tensor_base = to_tensor([1.0f32, 2.0f32])\n\
                               backward_tensor = read_tensor(0i32)\n\
                               independent = double(3i32)\n\
                               unrelated_later = 11i32\n\
                               recursive = countdown(3i32)\n\
                               forward_function = plus_one(4i32)\n\
                               def read_base(n: int32) -> int32 = add(n, base)\n\
                               def read_list(_unused: int32) -> List[int64] = list_base\n\
                               def read_tensor(_unused: int32) -> tensor[2, f32] = tensor_base\n\
                               def double(n: int32) -> int32 = mul(n, 2i32)\n\
                               def countdown(n: int32) -> int32 = if n <= 0i32 then base else countdown(n - 1i32) + 1i32\n\
                               def plus_one(n: int32) -> int32 = add(n, 1i32)\n";

const POSITIVE_OUTPUT: &str = "base = 5\n\
                               backward = 6\n\
                               list_base = [1, 2]\n\
                               backward_list = [1, 2]\n\
                               tensor_base = tensor(shape=[2], data=[1.0, 2.0])\n\
                               backward_tensor = tensor(shape=[2], data=[1.0, 2.0])\n\
                               independent = 6\n\
                               unrelated_later = 11\n\
                               recursive = 8\n\
                               forward_function = 5\n";

fn surf_program(source: &str) -> Vec<chelis_deep::Expr> {
    let declarations = parse_surf(source).expect("Surf fixture must parse");
    desugar_program(&declarations)
}

fn ingress_diagnostics_for(program: &[chelis_deep::Expr]) -> (Diagnostics, Diagnostics) {
    let summarize = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|error| {
                (
                    error.kind.diagnostic_name().to_string(),
                    error.message.clone(),
                )
            })
            .collect(),
    };
    (
        summarize(check_ir_program(program)),
        summarize(check_typed_program(program)),
    )
}

fn ingress_diagnostics(source: &str) -> (Diagnostics, Diagnostics) {
    ingress_diagnostics_for(&surf_program(source))
}

fn assert_exact_unbound(diagnostics: &Diagnostics, root: &str, later: &str, label: &str) {
    assert_eq!(
        diagnostics.len(),
        1,
        "{label}: expected one fail-closed diagnostic, got {diagnostics:#?}"
    );
    let (kind, message) = &diagnostics[0];
    assert_eq!(kind, "UnboundVariable", "{label}: {diagnostics:#?}");
    assert!(
        message.contains(root) && message.contains(later),
        "{label}: diagnostic must name initiating root `{root}` and later value `{later}`: \
         {diagnostics:#?}"
    );
}

fn check_output(source: &str, name: &str, extension: &str) -> std::process::Output {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join(format!("{name}.{extension}"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 fixture path")])
        .output()
        .expect("chelis check must run")
}

fn check_report(
    source: &str,
    name: &str,
    extension: &str,
) -> (std::process::ExitStatus, serde_json::Value) {
    let output = check_output(source, name, extension);
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "{name}: chelis check must emit JSON: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status, report)
}

fn assert_cli_unbound(case: RejectCase, must_name_root: bool) {
    let (status, report) = check_report(case.source, case.name, "ch");
    assert!(
        !status.success(),
        "{}: invalid program exited successfully: {report:#}",
        case.name
    );
    assert!(
        report["score"].as_f64().is_some_and(|score| score < 1.0),
        "{}: invalid program reported perfect fitness: {report:#}",
        case.name
    );
    let errors = report["errors"].as_array().expect("errors array");
    assert_eq!(
        errors.len(),
        1,
        "{}: expected one fail-closed error: {report:#}",
        case.name
    );
    assert_eq!(errors[0]["kind"], "UnboundVariable", "{report:#}");
    let message = errors[0]["message"].as_str().expect("error message");
    assert!(
        message.contains(case.later) && (!must_name_root || message.contains(case.root)),
        "{}: check diagnostic must name later binding `{}`{}: {report:#}",
        case.name,
        case.later,
        if must_name_root {
            format!(" and initiating root `{}`", case.root)
        } else {
            String::new()
        }
    );
}

fn assert_eval_and_build_unbound(case: RejectCase, must_name_root: bool) {
    let is_unbound = |diagnostic: &str| {
        diagnostic.contains("UnboundVariable") || diagnostic.contains("unbound variable")
    };
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join(format!("{}.ch", case.name));
    let out_dir = directory.path().join(format!("{}-out", case.name));
    write_file(&path, case.source);

    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 fixture path")])
        .output()
        .expect("chelis eval must run");
    let eval_stderr = String::from_utf8_lossy(&eval.stderr);
    assert!(
        !eval.status.success()
            && is_unbound(&eval_stderr)
            && eval_stderr.contains(case.later)
            && (!must_name_root || eval_stderr.contains(case.root)),
        "{}: eval must reject before execution; stdout={} stderr={eval_stderr}",
        case.name,
        String::from_utf8_lossy(&eval.stdout)
    );

    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().expect("UTF-8 fixture path"),
            "--target",
            "c",
            "--output",
            out_dir.to_str().expect("UTF-8 output path"),
        ])
        .output()
        .expect("chelis build must run");
    let build_stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        !build.status.success()
            && is_unbound(&build_stderr)
            && build_stderr.contains(case.later)
            && (!must_name_root || build_stderr.contains(case.root)),
        "{}: C build must reject before lowering; stdout={} stderr={build_stderr}",
        case.name,
        String::from_utf8_lossy(&build.stdout)
    );
    let emitted = if out_dir.is_dir() {
        std::fs::read_dir(&out_dir)
            .expect("read output directory")
            .map(|entry| entry.expect("output entry").file_name())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    assert!(
        emitted.is_empty(),
        "{}: rejected build emitted artifacts: {emitted:?}",
        case.name
    );
}

fn prove_output(source: &str, name: &str) -> std::process::Output {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join(format!("{name}.ch"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            path.to_str().expect("UTF-8 fixture path"),
            "--json",
        ])
        .output()
        .expect("chelis prove must run")
}

fn prove_records(output: &std::process::Output, name: &str) -> Vec<serde_json::Value> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line).unwrap_or_else(|error| {
                panic!("{name}: prove must emit NDJSON: {error}; line={line:?}")
            })
        })
        .collect()
}

fn assert_prove_unbound(case: RejectCase, must_name_root: bool) {
    let output = prove_output(case.source, case.name);
    let records = prove_records(&output, case.name);
    let errors = records
        .iter()
        .filter(|record| record["kind"] == "error" && record["stage"] == "check")
        .collect::<Vec<_>>();
    assert_eq!(
        output.status.code(),
        Some(3),
        "{}: prove must reject during checking: {records:#?}; stderr={}",
        case.name,
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        errors.len(),
        1,
        "{}: prove must emit one check error: {records:#?}",
        case.name
    );
    let rendered = errors[0].to_string();
    assert!(
        rendered.contains(case.later) && (!must_name_root || rendered.contains(case.root)),
        "{}: prove diagnostic must name later `{}`{}: {records:#?}",
        case.name,
        case.later,
        if must_name_root {
            format!(" and initiating root `{}`", case.root)
        } else {
            String::new()
        }
    );
}

#[test]
fn indirect_forward_values_reject_identically_at_both_checker_ingresses() {
    for &case in REJECT_CASES {
        let (ir, typed) = ingress_diagnostics(case.source);
        assert_eq!(ir, typed, "{}: checker ingresses diverged", case.name);
        assert_exact_unbound(&ir, case.root, case.later, case.name);
    }
}

#[test]
fn check_rejects_the_indirect_forward_reference_corpus() {
    for &case in REJECT_CASES {
        assert_cli_unbound(case, true);
    }
}

#[test]
fn eval_and_build_reject_before_execution_lowering_or_artifact_emission() {
    for &case in REJECT_CASES {
        assert_eval_and_build_unbound(case, true);
    }
}

#[test]
fn prove_rejects_scalar_list_and_tensor_before_proof_work() {
    for &case in &REJECT_CASES[..3] {
        let output = prove_output(case.source, case.name);
        let records = prove_records(&output, case.name);
        assert_eq!(
            output.status.code(),
            Some(3),
            "{}: prove must reject during checking: {records:#?}; stderr={}",
            case.name,
            String::from_utf8_lossy(&output.stderr)
        );
        let check_errors = records
            .iter()
            .filter(|record| record["kind"] == "error" && record["stage"] == "check")
            .collect::<Vec<_>>();
        assert_eq!(
            check_errors.len(),
            1,
            "{}: prove must emit exactly one check-error record: {records:#?}",
            case.name
        );
        let diagnostics = check_errors[0]["diagnostics"]
            .as_array()
            .expect("prove check error carries diagnostics");
        assert_eq!(
            diagnostics.len(),
            1,
            "{}: prove must preserve the one checker diagnostic: {records:#?}",
            case.name
        );
        let diagnostic = diagnostics[0].as_str().expect("diagnostic is text");
        assert!(
            diagnostic.contains(case.root)
                && diagnostic.contains(case.later)
                && diagnostic.contains("[04-INF-8]"),
            "{}: prove must expose the exact initialization failure: {records:#?}",
            case.name
        );
        assert!(
            records.iter().all(|record| {
                !matches!(record["kind"].as_str(), Some("property" | "obligation"))
            }),
            "{}: proof work must not begin after the check failure: {records:#?}",
            case.name
        );
    }
}

#[test]
fn direct_forward_value_remains_an_unbound_variable_control() {
    let case = DIRECT_FORWARD_CONTROL;
    let (ir, typed) = ingress_diagnostics(case.source);
    assert_eq!(ir, typed, "direct control: checker ingresses diverged");
    assert_eq!(ir.len(), 1, "direct control: {ir:#?}");
    assert_eq!(ir[0].0, "UnboundVariable", "direct control: {ir:#?}");
    assert!(
        ir[0].1.contains(case.later),
        "direct control must name later binding: {ir:#?}"
    );
    assert_cli_unbound(case, false);
    assert_prove_unbound(case, false);
    assert_eval_and_build_unbound(case, false);
}

#[test]
fn bare_deep_carrier_rejects_identically_at_both_checker_ingresses_and_check() {
    let program = parse_and_stamp(DEEP_INDIRECT_SOURCE).expect("bare Deep fixture must stamp");
    let (ir, typed) = ingress_diagnostics_for(&program);
    assert_eq!(ir, typed, "bare Deep: checker ingresses diverged");
    assert_exact_unbound(&ir, "root", "later", "bare Deep");

    let (status, report) = check_report(DEEP_INDIRECT_SOURCE, "bare_deep", "dp");
    assert!(!status.success(), "bare Deep check succeeded: {report:#}");
    let errors = report["errors"].as_array().expect("errors array");
    assert_eq!(errors.len(), 1, "bare Deep: {report:#}");
    assert_eq!(errors[0]["kind"], "UnboundVariable", "{report:#}");
    let message = errors[0]["message"].as_str().expect("error message");
    assert!(
        message.contains("root") && message.contains("later"),
        "bare Deep diagnostic must name both bindings: {report:#}"
    );
}

#[test]
fn eager_cycle_keeps_exact_cycle_detected_precedence_on_every_surface() {
    for (source, name) in [
        (CYCLE_SOURCE, "cycle"),
        (CYCLIC_ROOT_WITH_LATER_SOURCE, "cycle_with_later"),
    ] {
        let (ir, typed) = ingress_diagnostics(source);
        assert_eq!(ir, typed, "{name}: checker ingresses diverged");
        assert_eq!(
            ir.iter().map(|(kind, _)| kind.as_str()).collect::<Vec<_>>(),
            ["CycleDetected"],
            "{name}: cycle must suppress forward-reference errors: {ir:#?}"
        );

        let (status, report) = check_report(source, name, "ch");
        assert!(!status.success(), "{name}: check succeeded: {report:#}");
        let errors = report["errors"].as_array().expect("errors array");
        assert_eq!(errors.len(), 1, "{name}: {report:#}");
        assert_eq!(errors[0]["kind"], "CycleDetected", "{name}: {report:#}");

        let directory = tempdir().expect("tempdir");
        let path = directory.path().join(format!("{name}.ch"));
        let out_dir = directory.path().join(format!("{name}-out"));
        write_file(&path, source);
        for (lane, args) in [
            (
                "eval",
                vec!["eval", "--file", path.to_str().expect("UTF-8 path")],
            ),
            (
                "build",
                vec![
                    "build",
                    path.to_str().expect("UTF-8 path"),
                    "--target",
                    "c",
                    "--output",
                    out_dir.to_str().expect("UTF-8 path"),
                ],
            ),
        ] {
            let output = Command::cargo_bin("chelis")
                .expect("chelis binary")
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(args)
                .output()
                .expect("chelis command must run");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success()
                    && stderr.contains("binding cycle")
                    && !stderr.contains("unbound variable"),
                "{name}: {lane} lost CycleDetected precedence; stdout={} stderr={stderr}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
        assert!(
            !out_dir.join(format!("{name}.c")).exists(),
            "{name}: cycle rejection must precede C artifact emission"
        );

        let prove = prove_output(source, name);
        let records = prove_records(&prove, name);
        let check_errors = records
            .iter()
            .filter(|record| record["kind"] == "error" && record["stage"] == "check")
            .collect::<Vec<_>>();
        assert!(
            prove.status.code() == Some(3)
                && check_errors.len() == 1
                && check_errors[0].to_string().contains("binding cycle")
                && !check_errors[0].to_string().contains("unbound variable"),
            "{name}: prove lost CycleDetected precedence; records={records:#?}; stderr={}",
            String::from_utf8_lossy(&prove.stderr)
        );
    }
}

#[test]
fn cycle_precedence_does_not_hide_an_unknown_name() {
    let (ir, typed) = ingress_diagnostics(CYCLIC_ROOT_WITH_UNKNOWN_SOURCE);
    assert_eq!(ir, typed, "cycle-with-unknown: checker ingresses diverged");
    assert_eq!(
        ir.iter().map(|(kind, _)| kind.as_str()).collect::<Vec<_>>(),
        ["UnboundVariable", "CycleDetected"],
        "CycleDetected may suppress only graph-known later values: {ir:#?}"
    );
    assert!(
        ir[0].1.contains("missing"),
        "the unrelated unknown name must remain attributable: {ir:#?}"
    );

    let (status, report) =
        check_report(CYCLIC_ROOT_WITH_UNKNOWN_SOURCE, "cycle_with_unknown", "ch");
    assert!(!status.success(), "cycle-with-unknown passed: {report:#}");
    let kinds = report["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .map(|error| error["kind"].as_str().expect("diagnostic kind"))
        .collect::<Vec<_>>();
    assert_eq!(kinds, ["UnboundVariable", "CycleDetected"], "{report:#}");
}

#[test]
fn backward_independent_and_forward_function_controls_agree_across_lanes() {
    let (ir, typed) = ingress_diagnostics(POSITIVE_SOURCE);
    assert_eq!(ir, typed, "positive controls: checker ingresses diverged");
    assert!(ir.is_empty(), "positive controls must check: {ir:#?}");

    let (status, report) = check_report(POSITIVE_SOURCE, "positive_controls", "ch");
    assert!(
        status.success(),
        "positive controls failed check: {report:#}"
    );
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report:#}");
    assert_eq!(report["errors"].as_array().map(Vec::len), Some(0));

    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("positive_controls.ch");
    write_file(&path, POSITIVE_SOURCE);
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("chelis eval must run");
    assert!(
        eval.status.success(),
        "positive controls failed eval: {}",
        String::from_utf8_lossy(&eval.stderr)
    );
    assert_eq!(
        eval.stdout,
        POSITIVE_OUTPUT.as_bytes(),
        "positive controls must preserve exact source-ordered root observations"
    );

    let compiled = build_and_run(POSITIVE_SOURCE, "issue1339_positive_controls");
    assert_eq!(
        compiled.as_bytes(),
        eval.stdout,
        "positive controls must have byte-identical eval and compiled-C observations"
    );

    let prove = prove_output(POSITIVE_SOURCE, "positive_controls");
    let prove_records = prove_records(&prove, "positive_controls");
    assert_ne!(
        prove.status.code(),
        Some(3),
        "positive controls must not fail prove's check gate: {prove_records:#?}; stderr={}",
        String::from_utf8_lossy(&prove.stderr)
    );
    assert!(
        prove_records
            .iter()
            .all(|record| record["stage"] != "check"),
        "positive controls must emit no prove check error: {prove_records:#?}"
    );
}

#[test]
fn imported_library_values_are_available_before_the_current_unit() {
    let (_directory, reef_home, app_pkg) = make_app("issue-1339-imported-value");
    let library = app_pkg.join("initlib");

    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "issue-1339-imported-value"
version = "0.1.0"
compiler = "={COMPILER_VERSION}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
initlib = {{ path = "./initlib" }}
"#,
        ),
    );
    write_file(
        &library.join("reef.toml"),
        &format!(
            r#"[package]
name = "initlib"
version = "0.1.0"
compiler = "={COMPILER_VERSION}"
module_prefix = "Initlib"
"#,
        ),
    );
    write_file(
        &library.join("src/values.ch"),
        "module Initlib.Values\n\
         export (read_base)\n\n\
         base = 40i32\n\
         def read_base(n: int32) -> int32 = add(base, n)\n",
    );
    let entry = app_pkg.join("src/main.ch");
    write_file(
        &entry,
        "module Demo.Main\n\
         import Initlib.Values (read_base)\n\n\
         imported = read_base(2i32)\n",
    );

    let check = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", entry.to_str().expect("UTF-8 entry path")])
        .output()
        .expect("package-aware chelis check must run");
    let report: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap_or_else(|error| {
        panic!(
            "package-aware check must emit JSON: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&check.stdout),
            String::from_utf8_lossy(&check.stderr)
        )
    });
    assert!(
        check.status.success()
            && report["score"].as_f64() == Some(1.0)
            && report["errors"].as_array().is_some_and(Vec::is_empty),
        "an imported function may read its already-checked library value: {report:#}\nstderr: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["eval", "--file", entry.to_str().expect("UTF-8 entry path")])
        .output()
        .expect("package-aware chelis eval must run");
    assert!(
        eval.status.success(),
        "imported capture failed eval: {}",
        String::from_utf8_lossy(&eval.stderr)
    );
    assert_eq!(eval.stdout, b"imported = 42\n");

    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    // Whether an imported library's own `base` root is manifested is #912's
    // root-boundary question. This control owns only [04-INF-8]'s availability
    // exemption, so require the current unit's exact computed observation
    // without absorbing that separate topology contract.
    assert!(
        compiled.lines().any(|line| line == "imported = 42"),
        "compiled current-unit root did not observe the initialized library value: {compiled}"
    );
}
