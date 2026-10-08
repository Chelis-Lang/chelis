//! The auxiliary Surf PEG must not abort an embedding host on rejected input.

use chelis_validate::validate_surf;
use std::process::Command;

const CHILD_ENV: &str = "CHELIS_3234_VALIDATOR_CHILD";

fn nested_comment_source(depth: usize, valid: bool) -> String {
    let comment = format!("{}{}", "{-".repeat(depth), "-}".repeat(depth));
    let tail = if valid { "x = 1i64" } else { "x = 1i64 y" };
    format!("module Probe.Deep\n{comment}\n{tail}\n")
}

fn nested_expression_source(depth: usize, valid: bool) -> String {
    let tail = if valid { "" } else { " y" };
    format!(
        "module Probe.Deep\nx = {}1i64{}{}\n",
        "(".repeat(depth),
        ")".repeat(depth),
        tail
    )
}

fn nested_type_source(depth: usize, valid: bool) -> String {
    let tail = if valid { "" } else { " y" };
    format!(
        "module Probe.Deep\ndef identity(x: {}i64{}) -> i64 = x{tail}\n",
        "(".repeat(depth),
        ")".repeat(depth)
    )
}

fn nested_pattern_source(depth: usize, valid: bool) -> String {
    let tail = if valid { "" } else { " y" };
    format!(
        "module Probe.Deep\nx = match 1i64 with {{ | {}_{} => 1i64 }}{tail}\n",
        "(".repeat(depth),
        ")".repeat(depth)
    )
}

fn run_on_small_stack_in_child(test_name: &str, probe: impl FnOnce() + Send + 'static) {
    if std::env::var_os(CHILD_ENV).is_some() {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(probe)
            .expect("spawn 2 MiB validation worker")
            .join()
            .expect("validation worker returned");
        return;
    }

    let result = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", test_name, "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("run isolated validator probe");
    assert!(result.status.success(), "{result:?}");
}

#[test]
fn accepted_nested_comments_remain_valid_on_a_small_stack() {
    run_on_small_stack_in_child(
        "accepted_nested_comments_remain_valid_on_a_small_stack",
        || {
            let source = nested_comment_source(5_000, true);
            assert!(validate_surf(&source).is_ok(), "valid Surf was rejected");
        },
    );
}

#[test]
fn accepted_comments_above_peg_budget_remain_valid() {
    run_on_small_stack_in_child("accepted_comments_above_peg_budget_remain_valid", || {
        let source = nested_comment_source(100_000, true);
        assert!(validate_surf(&source).is_ok(), "valid Surf was rejected");
    });
}

#[test]
fn rejected_nested_comments_return_a_bounded_error_on_a_small_stack() {
    run_on_small_stack_in_child(
        "rejected_nested_comments_return_a_bounded_error_on_a_small_stack",
        || {
            let source = nested_comment_source(5_000, false);
            assert!(
                chelis_surf::parser::parse_str(&source).is_err(),
                "control parser must reject the input"
            );
            let error = validate_surf(&source).expect_err("rejected Surf must fail validation");
            assert!(
                error.to_string().contains("compiler parse failed"),
                "{error}"
            );
        },
    );
}

#[test]
fn extreme_rejected_comments_report_a_peg_budget_diagnostic() {
    run_on_small_stack_in_child(
        "extreme_rejected_comments_report_a_peg_budget_diagnostic",
        || {
            let source = nested_comment_source(100_000, false);
            let error = validate_surf(&source).expect_err("rejected Surf must fail validation");
            let message = error.to_string();
            assert!(message.contains("PEG classification skipped"), "{message}");
            assert!(message.contains("compiler parse failed"), "{message}");
        },
    );
}

#[test]
fn accepted_nested_expression_survives_a_small_stack() {
    run_on_small_stack_in_child("accepted_nested_expression_survives_a_small_stack", || {
        let source = nested_expression_source(3_000, true);
        assert!(validate_surf(&source).is_ok(), "valid Surf was rejected");
    });
}

#[test]
fn rejected_nested_expression_reports_a_parse_error_on_a_small_stack() {
    run_on_small_stack_in_child(
        "rejected_nested_expression_reports_a_parse_error_on_a_small_stack",
        || {
            let source = nested_expression_source(3_000, false);
            let error = validate_surf(&source).expect_err("invalid Surf must fail");
            assert!(
                error.to_string().contains("compiler parse failed"),
                "{error}"
            );
        },
    );
}

#[test]
fn accepted_extreme_parentheses_survive_a_small_stack() {
    run_on_small_stack_in_child("accepted_extreme_parentheses_survive_a_small_stack", || {
        let source = nested_expression_source(40_000, true);
        assert!(
            chelis_surf::parser::parse_str(&source).is_ok(),
            "valid Surf was rejected by the direct parser"
        );
        assert!(validate_surf(&source).is_ok(), "valid Surf was rejected");
    });
}

#[test]
fn rejected_extreme_parentheses_report_a_parse_error_on_a_small_stack() {
    run_on_small_stack_in_child(
        "rejected_extreme_parentheses_report_a_parse_error_on_a_small_stack",
        || {
            let source = nested_expression_source(40_000, false);
            assert!(
                chelis_surf::parser::parse_str(&source).is_err(),
                "invalid Surf was accepted by the direct parser"
            );
            let error = validate_surf(&source).expect_err("invalid Surf must fail");
            let message = error.to_string();
            assert!(message.contains("compiler parse failed"), "{message}");
            assert!(message.contains("byte"), "{message}");
        },
    );
}

#[test]
fn extreme_type_parentheses_keep_accept_and_reject_verdicts() {
    run_on_small_stack_in_child(
        "extreme_type_parentheses_keep_accept_and_reject_verdicts",
        || {
            let valid = nested_type_source(40_000, true);
            assert!(chelis_surf::parser::parse_str(&valid).is_ok());
            assert!(validate_surf(&valid).is_ok());
            let invalid = nested_type_source(40_000, false);
            assert!(chelis_surf::parser::parse_str(&invalid).is_err());
            let error = validate_surf(&invalid).expect_err("invalid type source must fail");
            let message = error.to_string();
            assert!(message.contains("compiler parse failed"), "{message}");
            assert!(message.contains("byte"), "{message}");
        },
    );
}

#[test]
fn extreme_pattern_parentheses_keep_accept_and_reject_verdicts() {
    run_on_small_stack_in_child(
        "extreme_pattern_parentheses_keep_accept_and_reject_verdicts",
        || {
            let valid = nested_pattern_source(40_000, true);
            assert!(chelis_surf::parser::parse_str(&valid).is_ok());
            assert!(validate_surf(&valid).is_ok());
            let invalid = nested_pattern_source(40_000, false);
            assert!(chelis_surf::parser::parse_str(&invalid).is_err());
            let error = validate_surf(&invalid).expect_err("invalid pattern source must fail");
            let message = error.to_string();
            assert!(message.contains("compiler parse failed"), "{message}");
            assert!(message.contains("byte"), "{message}");
        },
    );
}
