//! #3418: a tensor whose element count or byte size does not fit i64 is
//! refused alike in every lane (spec/04-type-system.md section 4.7,
//! [05-MOV-1], [05-OP-33]): a run-time size traps `Overflow` under the
//! owning operation before allocation, in `chelis eval` and in compiled C,
//! and a representable size the machine cannot hold fails as the C runtime's
//! allocation failure in both.

mod common;

use common::{gcc_available, write_file};
use std::process::{Command as StdCommand, Output};
use tempfile::tempdir;

use assert_cmd::Command;

/// 2^61 as a run-time value: the sum of two literal halves, so no stage sees
/// the size as a literal.
const RUNTIME_2_61: &str = "def huge() -> i64 = tensor_to_scalar(sum(expand(to_tensor([1152921504606846976i64]), 0i32, 2i64), 0i32))\n";

/// 2^62 as a run-time value.
const RUNTIME_2_62: &str = "def huge() -> i64 = tensor_to_scalar(sum(expand(to_tensor([2305843009213693952i64]), 0i32, 2i64), 0i32))\n";

fn program(helper: &str, body: &str) -> String {
    format!(
        "{helper}def main() -> i64 ! {{ IO }} = {{\n  _ = print(\"before\")\n  x = {body}\n  shape(x, 0i32)\n}}\nout = main()\n"
    )
}

struct Lanes {
    eval: Output,
    build: Output,
    c: Option<Output>,
}

fn run_lanes(name: &str, source: &str) -> Lanes {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--timeout", "120", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build");
    let c = build.status.success().then(|| {
        StdCommand::new(out_dir.join(name))
            .output()
            .expect("compiled binary should run")
    });
    Lanes { eval, build, c }
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Both lanes build or run, print the effect that precedes the operation,
/// fail, and report exactly `report` and `trap` on their own lines with no
/// panic.
fn assert_both_lanes_fail_with(lanes: &Lanes, report: &str, trap: Option<&str>) {
    assert!(
        lanes.build.status.success(),
        "the C build must accept the program; the failure belongs to run time:\n{}",
        text(&lanes.build)
    );
    let c = lanes.c.as_ref().expect("built binary ran");
    for (lane, output) in [("eval", &lanes.eval), ("c", c)] {
        let all = text(output);
        assert!(!output.status.success(), "{lane} must fail:\n{all}");
        assert!(!all.contains("panicked"), "{lane} must not panic:\n{all}");
        assert!(
            String::from_utf8_lossy(&output.stdout).starts_with("before"),
            "{lane} runs the earlier effect first:\n{all}"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr
                .lines()
                .any(|line| line.trim_start_matches("error: ") == report),
            "{lane} must report `{report}`:\n{all}"
        );
        let traps = all
            .lines()
            .filter(|line| line.contains("numeric trap:"))
            .collect::<Vec<_>>();
        assert_eq!(
            traps,
            trap.into_iter().collect::<Vec<_>>(),
            "{lane}:\n{all}"
        );
    }
}

#[test]
fn runtime_byte_size_overflow_traps_in_expand_in_both_lanes() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let lanes = run_lanes(
        "expand_bytes",
        &program(RUNTIME_2_62, "expand(to_tensor([1.0f32]), 0i32, huge())"),
    );
    assert_both_lanes_fail_with(
        &lanes,
        "Overflow: byte size exceeds i64",
        Some("numeric trap: overflow in expand at i64"),
    );
}

#[test]
fn runtime_element_count_overflow_traps_in_insert_in_both_lanes() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let lanes = run_lanes(
        "insert_count",
        &program(
            RUNTIME_2_62,
            "insert(expand(to_tensor([1.0f32]), 0i32, 4i64), 0i32, huge())",
        ),
    );
    assert_both_lanes_fail_with(
        &lanes,
        "Overflow: extent product exceeds i64",
        Some("numeric trap: overflow in insert at i64"),
    );
}

#[test]
fn runtime_padded_extent_overflow_traps_in_pad_in_both_lanes() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let lanes = run_lanes(
        "pad_extent",
        &program(
            RUNTIME_2_62,
            "pad(to_tensor([1i8]), [[huge(), 9223372036854775807i64]], 0i8)",
        ),
    );
    assert_both_lanes_fail_with(
        &lanes,
        "Overflow: padded extent exceeds i64",
        Some("numeric trap: overflow in pad at i64"),
    );
}

/// 2^61 f32 elements is 2^63 bytes, one past i64: the byte size, not the
/// count, overflows.
#[test]
fn runtime_byte_size_one_past_i64_traps_in_both_lanes() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let lanes = run_lanes(
        "expand_bytes_boundary",
        &program(RUNTIME_2_61, "expand(to_tensor([1.0f32]), 0i32, huge())"),
    );
    assert_both_lanes_fail_with(
        &lanes,
        "Overflow: byte size exceeds i64",
        Some("numeric trap: overflow in expand at i64"),
    );
}

/// Negative parity: 2^61 i8 elements is 2^61 bytes, representable, so no
/// lane reports an overflow; neither machine can hold it, so both fail as
/// the C runtime's allocation failure ([05-OP-33]), without a panic.
#[test]
fn representable_but_unallocatable_size_fails_allocation_in_both_lanes() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let lanes = run_lanes(
        "expand_alloc",
        &program(RUNTIME_2_61, "expand(to_tensor([1i8]), 0i32, huge())"),
    );
    assert_both_lanes_fail_with(
        &lanes,
        "Domain: chelis_alloc tensor allocation failed",
        None,
    );
}

/// Five live tensors of 2^60 f32 elements: each byte size is representable,
/// their sum is not. The C build used to refuse with an internal live-byte
/// bound overflow; the program is valid, and each allocation fails at run
/// time as the machine's capacity dictates.
#[test]
fn live_byte_sum_past_u64_still_builds() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let source = "def main() -> i64 ! { IO } = {\n  _ = print(\"before\")\n  a = expand(to_tensor([1.0f32]), 0i32, 1152921504606846976i64)\n  b = expand(to_tensor([2.0f32]), 0i32, 1152921504606846976i64)\n  c = expand(to_tensor([3.0f32]), 0i32, 1152921504606846976i64)\n  d = expand(to_tensor([4.0f32]), 0i32, 1152921504606846976i64)\n  e = expand(to_tensor([5.0f32]), 0i32, 1152921504606846976i64)\n  shape(a, 0i32) + shape(b, 0i32) + shape(c, 0i32) + shape(d, 0i32) + shape(e, 0i32)\n}\nout = main()\n";
    let lanes = run_lanes("live_sum", source);
    assert_both_lanes_fail_with(
        &lanes,
        "Domain: chelis_alloc tensor allocation failed",
        None,
    );
}

/// Positive control: a small run-time extent still evaluates in both lanes.
#[test]
fn small_runtime_extent_still_evaluates_in_both_lanes() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let helper =
        "def small() -> i64 = tensor_to_scalar(sum(expand(to_tensor([3i64]), 0i32, 2i64), 0i32))\n";
    let source = format!(
        "{helper}def main() -> i64 ! {{ IO }} = {{\n  _ = print(\"before\")\n  x = expand(to_tensor([1.0f32]), 0i32, small())\n  shape(x, 0i32)\n}}\nout = main()\n"
    );
    let lanes = run_lanes("small", &source);
    assert!(lanes.eval.status.success(), "{}", text(&lanes.eval));
    assert!(text(&lanes.eval).contains('6'), "{}", text(&lanes.eval));
    let c = lanes.c.as_ref().expect("built binary ran");
    assert!(c.status.success(), "{}", text(c));
}
