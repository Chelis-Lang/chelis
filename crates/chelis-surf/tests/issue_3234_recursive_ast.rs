//! Direct Surf parser callers own and release recursive ASTs on their own stack.

use chelis_surf::parser::parse_str;
use std::process::Command;

const CHILD_ENV: &str = "CHELIS_3234_AST_CHILD";

fn on_small_stack(test_name: &str, probe: impl FnOnce() + Send + 'static) {
    if std::env::var_os(CHILD_ENV).is_some() {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(probe)
            .expect("spawn 2 MiB parser worker")
            .join()
            .expect("parser worker returned");
        return;
    }

    let result = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", test_name, "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("run isolated parser probe");
    assert!(result.status.success(), "{result:?}");
}

fn nested_reference_type_source(depth: usize, valid: bool) -> String {
    let tail = if valid { "" } else { " invalid" };
    format!(
        "module Probe.Ast\ndef probe(x: {}i64) -> i64 = 0i64{tail}\n",
        "& ".repeat(depth)
    )
}

fn nested_invalid_tensor_precision_source(depth: usize) -> String {
    format!(
        "module Probe.Ast\ndef probe(x: tensor[{}i64]) -> i64 = 0i64\n",
        "& ".repeat(depth)
    )
}

fn nested_match_pattern_source(depth: usize, valid: bool) -> String {
    let tail = if valid { "" } else { " invalid" };
    format!(
        "module Probe.Ast\nx = match 1i64 with {{ | {}_{} => 0i64 }}{tail}\n",
        "(".repeat(depth),
        ",)".repeat(depth)
    )
}

fn nested_binding_pattern_source(depth: usize, valid: bool) -> String {
    let tail = if valid { "" } else { " invalid" };
    format!(
        "module Probe.Ast\nx = {{\n{}v{} = 1i64\n0i64\n}}{tail}\n",
        "(".repeat(depth),
        ",)".repeat(depth)
    )
}

#[test]
fn direct_parser_releases_deep_reference_type_on_small_stack() {
    on_small_stack(
        "direct_parser_releases_deep_reference_type_on_small_stack",
        || {
            let source = nested_reference_type_source(20_000, true);
            let parsed = parse_str(&source).expect("nested reference type is Surf syntax");
            drop(parsed);
        },
    );
}

#[test]
fn direct_parser_rejects_invalid_deep_reference_type_on_small_stack() {
    on_small_stack(
        "direct_parser_rejects_invalid_deep_reference_type_on_small_stack",
        || {
            let source = nested_reference_type_source(20_000, false);
            let valid = nested_reference_type_source(20_000, true);
            let parsed = parse_str(&valid).expect("valid control must parse");
            drop(parsed);
            let error = parse_str(&source).expect_err("trailing token must be rejected");
            assert!(error.to_string().contains("byte"), "{error}");
        },
    );
}

#[test]
fn direct_parser_reports_invalid_deep_precision_on_small_stack() {
    on_small_stack(
        "direct_parser_reports_invalid_deep_precision_on_small_stack",
        || {
            let source = nested_invalid_tensor_precision_source(20_000);
            let error = parse_str(&source).expect_err("a reference is not a tensor precision");
            let message = error.to_string();
            assert!(message.contains("precision type name"), "{message}");
            assert!(message.contains("reference type"), "{message}");
            assert!(message.contains("byte"), "{message}");
        },
    );
}

#[test]
fn direct_parser_releases_deep_match_pattern_on_small_stack() {
    on_small_stack(
        "direct_parser_releases_deep_match_pattern_on_small_stack",
        || {
            let source = nested_match_pattern_source(20_000, true);
            let parsed = parse_str(&source).expect("nested match pattern is Surf syntax");
            drop(parsed);
        },
    );
}

#[test]
fn direct_parser_rejects_invalid_deep_match_pattern_on_small_stack() {
    on_small_stack(
        "direct_parser_rejects_invalid_deep_match_pattern_on_small_stack",
        || {
            let source = nested_match_pattern_source(20_000, false);
            let error = parse_str(&source).expect_err("trailing token must be rejected");
            assert!(error.to_string().contains("byte"), "{error}");
        },
    );
}

#[test]
fn direct_parser_releases_deep_binding_pattern_on_small_stack() {
    on_small_stack(
        "direct_parser_releases_deep_binding_pattern_on_small_stack",
        || {
            let source = nested_binding_pattern_source(20_000, true);
            let parsed = parse_str(&source).expect("nested binding pattern is Surf syntax");
            drop(parsed);
        },
    );
}

#[test]
fn direct_parser_rejects_invalid_deep_binding_pattern_on_small_stack() {
    on_small_stack(
        "direct_parser_rejects_invalid_deep_binding_pattern_on_small_stack",
        || {
            let source = nested_binding_pattern_source(20_000, false);
            let error = parse_str(&source).expect_err("trailing token must be rejected");
            assert!(error.to_string().contains("byte"), "{error}");
        },
    );
}
