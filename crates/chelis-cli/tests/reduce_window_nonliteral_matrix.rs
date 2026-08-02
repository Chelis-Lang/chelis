//! chelis#725 - `reduce_window_*` with non-literal window/strides lists
//! silently lowers to a no-op (both lists `unwrap_or_default()` to empty,
//! lower.rs:7522-7537), and the compiled binary returns the UNPOOLED input
//! with the wrong shape while `chelis check` scores 1 and eval pools
//! correctly. With only ONE list non-literal, the emitter's length
//! assertion (emit.rs:4509) panics the compiler instead.
//! The site now rejects loudly; chelis#1058 owns implementing compiled
//! runtime-valued window and stride lists.
//!
//! This settled the last open item (item 5) of
//! `docs/investigations/silent_substitution_audit_backlog.md` - and unlike
//! every sibling claim from that sweep, it is REAL: reachable from ordinary
//! Surf (a dynamically parameterized pooling window), silent in the worst
//! case, lane-divergent in all cases.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn eval_first_line(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

/// Build to C; (build_ok, build_stderr, run_stdout_first_line if it ran).
fn c_outcome(program: &str, name: &str) -> (bool, String, Option<String>) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
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
        .expect("chelis build should run");
    let stderr = String::from_utf8_lossy(&built.stderr).into_owned();
    if !built.status.success() {
        return (false, stderr, None);
    }
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link failed for {name}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    (
        true,
        stderr,
        Some(
            String::from_utf8_lossy(&run.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string(),
        ),
    )
}

const POOLED: &str = "tensor(shape=[5], data=[5.0, 5.0, 8.0, 8.0, 9.0])";

/// **`partition` is correct in both lanes** - the control that bounds the
/// `assign_partition` leftover from the audit backlog (its
/// `/* unsupported partition type */` arm needs an internal desync to
/// reach; not reachable from the input surface). Previously probe-only
/// (`part2.ch` in the archived corpus); promoted to a test so the
/// clearance is CI-checked rather than asserted.
#[test]
fn partition_agrees_across_lanes() {
    let program = "xs: List[int64] = [cast(1, int64), cast(3, int64), cast(2, int64), cast(4, int64)]\n\
         buckets = partition(fn (x: int64) -> gt(x, cast(2, int64)), xs)\n\
         out = print(buckets)\n";
    assert_eq!(
        eval_first_line(program).expect("eval"),
        "([3, 4], [1, 2])",
        "partition splits pass/fail in order"
    );
    if c_toolchain_available() {
        let (ok, stderr, stdout) = c_outcome(program, "partition_ctl");
        assert!(ok, "partition must build: {stderr}");
        assert_eq!(stdout.as_deref(), Some("([3, 4], [1, 2])"));
    }
}

/// Literal windows pool correctly in BOTH lanes. The control that isolates
/// the trigger to the non-literal extraction.
#[test]
fn literal_window_pools_correctly_in_both_lanes() {
    let program = "module M.Main\n\
         def f(x: tensor[6, f32]) -> tensor[5, f32] = reduce_window_max(x, [2], [1])\n\
         out = print(f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0])))\n";
    assert_eq!(eval_first_line(program).expect("eval"), POOLED);
    if c_toolchain_available() {
        let (ok, stderr, stdout) = c_outcome(program, "rw_literal");
        assert!(ok, "literal window must build: {stderr}");
        assert_eq!(stdout.as_deref(), Some(POOLED));
    }
}

/// eval pools correctly even with non-literal window AND strides - it does
/// not take the broken lowering path. Any fix must keep eval green.
#[test]
fn eval_pools_correctly_with_nonliteral_window_and_strides() {
    let program = "module M.Main\n\
         def f(x: tensor[6, f32], w: int32, s: int32) -> tensor[5, f32] = \
         reduce_window_max(x, [w], [s])\n\
         out = print(f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0]), 2, 1))\n";
    assert_eq!(eval_first_line(program).expect("eval"), POOLED);
}

/// Observed today: the compiled binary prints
/// `tensor(shape=[6], data=[1.0, 5.0, 2.0, 8.0, 3.0, 9.0])` - the input,
/// unpooled, with shape 6 out of a `-> tensor[5, f32]` function. Both
/// lists defaulted to empty and the ReduceWindow became a no-op.
/// Un-ignored by chelis#730 Phase 1 (census row 8): lowering now raises a
/// fatal branded error on non-literal window/stride lists, which this
/// test accepts as the reject arm. Pooling with runtime windows is
/// op-owner support work.
#[test]
fn c_nonliteral_window_and_strides_pool_or_reject() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[6, f32], w: int32, s: int32) -> tensor[5, f32] = \
         reduce_window_max(x, [w], [s])\n\
         out = print(f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0]), 2, 1))\n";
    let (ok, stderr, stdout) = c_outcome(program, "rw_both_var");
    assert!(
        !stderr.contains("panicked"),
        "the compiler must not panic; stderr: {stderr}"
    );
    if ok {
        assert_eq!(
            stdout.as_deref(),
            Some(POOLED),
            "if it builds, the pool must actually pool"
        );
    } else {
        assert!(
            stderr.contains("error:"),
            "a rejection must be a clean diagnostic; got: {stderr}"
        );
    }
}

/// Observed today: `thread 'main' panicked at crates/chelis-backend-c/src/
/// emit.rs:4509 ... window_shape and strides must have equal length, left: 0`;
/// the half-non-literal case trips the internal assertion instead of a
/// diagnostic.
/// Un-ignored by chelis#730 Phase 1 (census rows 8/12): the half-literal
/// case is rejected at lowering before the emitter arity check (itself
/// now a diagnostic, not an assert).
#[test]
fn c_nonliteral_window_does_not_panic_the_compiler() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[6, f32], w: int32) -> tensor[5, f32] = \
         reduce_window_max(x, [w], [1])\n\
         out = print(f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0]), 2))\n";
    let (_, stderr, _) = c_outcome(program, "rw_one_var");
    assert!(
        !stderr.contains("panicked"),
        "the compiler must not panic on a checkable input; stderr: {stderr}"
    );
}
