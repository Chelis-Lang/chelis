//! The auxiliary Surf PEG must not abort an embedding host on rejected input.

use chelis_validate::validate_surf;
use std::process::Command;

const CHILD_ENV: &str = "CHELIS_3234_VALIDATOR_CHILD";

fn nested_comment_source(depth: usize, valid: bool) -> String {
    let comment = format!("{}{}", "{-".repeat(depth), "-}".repeat(depth));
    let tail = if valid { "x = 1i64" } else { "x = 1i64 y" };
    format!("module Probe.Deep\n{comment}\n{tail}\n")
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
