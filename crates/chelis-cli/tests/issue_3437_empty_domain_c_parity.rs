//! #3437: a zero-element movement domain does no per-element work in C.
//!
//! An empty tensor whose other extents are huge, such as
//! `[2^32, 2^32, 0]`, holds no element, but its suffix strides are
//! representable, so spec/05's descriptor rules admit it. C copied a
//! movement result through one loop per axis, and the loops outside the
//! zero axis still visited the product of their extents (2^64), so the
//! binary spun instead of finishing. Every run here is bounded: a hang is
//! killed and reported as a failure, never left to the harness.

mod common;

use common::{gcc_available, write_file};
use std::path::Path;
use std::process::{Command as StdCommand, Output, Stdio};
use std::time::{Duration, Instant};
use tempfile::tempdir;

use assert_cmd::cargo::CommandCargoExt;

/// A lane that finishes in well under a second must not need more than this.
const LIMIT: Duration = Duration::from_secs(30);

/// Run `command` to completion, killing it and failing once it exceeds
/// [`LIMIT`].
fn bounded(mut command: StdCommand, what: &str) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("{what}: spawn: {error}"));
    let start = Instant::now();
    loop {
        if child.try_wait().expect("poll").is_some() {
            return child.wait_with_output().expect("collect output");
        }
        if start.elapsed() > LIMIT {
            child.kill().expect("kill");
            child.wait().expect("reap");
            panic!("{what} did not finish within {LIMIT:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// `[2^32, 2^32, 0]` built from run-time extents, so no lane can fold it,
/// followed by `body` binding `y`; the program prints `shape(y, 0i32)`.
fn program(body: &str) -> String {
    format!(
        "def zero() -> i64 = tensor_to_scalar(sum(expand(to_tensor([0i64]), 0i32, 2i64), 0i32))\n\
         def h32() -> i64 = tensor_to_scalar(sum(expand(to_tensor([2147483648i64]), 0i32, 2i64), 0i32))\n\
         def main() -> i64 ! {{ IO }} = {{\n  \
         e = expand(to_tensor([1.0f32]), 0i32, zero())\n  \
         x = insert(insert(e, 0i32, h32()), 0i32, h32())\n  \
         {body}\n  \
         shape(y, 0i32)\n\
         }}\n\
         out = main()\n"
    )
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn eval(path: &Path) -> Output {
    let mut command = StdCommand::cargo_bin("chelis").expect("binary");
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--timeout", "30", "--file"])
        .arg(path);
    bounded(command, "chelis eval")
}

fn build_c(path: &Path, out_dir: &Path, name: &str) -> Output {
    let mut command = StdCommand::cargo_bin("chelis").expect("binary");
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(path)
        .args(["--target", "c", "--output"])
        .arg(out_dir);
    let built = bounded(command, "chelis build");
    assert!(built.status.success(), "{name}: build failed\n{}", text(&built));
    bounded(StdCommand::new(out_dir.join(name)), name)
}

/// Both lanes' outputs for `body`.
fn lanes(body: &str, name: &str) -> (Output, Output) {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, &program(body));
    let eval = eval(&path);
    let c = build_c(&path, &dir.path().join(format!("{name}-out")), name);
    (eval, c)
}

fn out_line(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| line.starts_with("out = "))
        .unwrap_or_else(|| panic!("missing `out` line:\n{}", text(output)))
        .to_string()
}

fn trap_lines(output: &Output) -> Vec<String> {
    text(output)
        .lines()
        .filter(|line| line.starts_with("Overflow:") || line.starts_with("numeric trap:"))
        .map(str::to_string)
        .collect()
}

/// Movement ops whose result keeps the zero extent and representable
/// strides: both lanes finish and agree on the value.
#[test]
fn empty_movement_results_finish_and_agree() {
    for (name, body, expected) in [
        ("insert_empty", "y = x", "out = 4294967296"),
        ("permute_empty", "y = permute(x, 1i32, 0i32, 2i32)", "out = 4294967296"),
        (
            "shrink_empty",
            "y = shrink(x, [[1i64, 3i64], [0i64, h32()], [0i64, 0i64]])",
            "out = 2",
        ),
        (
            "pad_empty",
            "y = pad(x, [[1i64, 0i64], [0i64, 1i64], [0i64, 0i64]], 0.0f32)",
            "out = 4294967297",
        ),
        ("stride_empty", "y = stride(x, 2i64, 1i64, 1i64)", "out = 2147483648"),
        ("concat_empty", "y = concat([x, x], 0i32)", "out = 8589934592"),
    ] {
        let (eval, c) = lanes(body, name);
        assert!(eval.status.success(), "{name}: eval\n{}", text(&eval));
        assert!(c.status.success(), "{name}: C\n{}", text(&c));
        assert_eq!(out_line(&eval), expected, "{name}: eval");
        assert_eq!(out_line(&c), expected, "{name}: C");
    }
}

/// The negative side: an empty result whose own extents are unrepresentable
/// still traps `Overflow` before any iteration, with the same text on both
/// lanes.
#[test]
fn unrepresentable_empty_concat_traps_alike() {
    let body = "w = insert(expand(to_tensor([1.0f32]), 0i32, zero()), 0i32, 4611686018427387904i64)\n  \
                y = concat([w, w], 0i32)";
    let (eval, c) = lanes(body, "concat_overflow");
    assert!(!eval.status.success(), "eval\n{}", text(&eval));
    assert!(!c.status.success(), "C\n{}", text(&c));
    let expected = ["numeric trap: overflow in concat at i64".to_string()];
    assert_eq!(trap_lines(&eval), expected, "eval\n{}", text(&eval));
    assert_eq!(trap_lines(&c), expected, "C\n{}", text(&c));
}

/// An empty permute result whose suffix strides do not fit i64,
/// `[0, 2^32, 2^32]`, traps `Overflow` in C before iterating
/// ([05-OP-33]: an empty result still needs representable strides).
#[test]
fn empty_permute_with_unrepresentable_strides_traps_in_c() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let dir = tempdir().expect("tempdir");
    let name = "permute_stride_overflow";
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, &program("y = permute(x, 2i32, 0i32, 1i32)"));
    let c = build_c(&path, &dir.path().join(format!("{name}-out")), name);
    assert!(!c.status.success(), "C\n{}", text(&c));
    assert_eq!(
        trap_lines(&c),
        [
            "Overflow: stride product exceeds i64".to_string(),
            "numeric trap: overflow in permute at i64".to_string(),
        ],
        "C\n{}",
        text(&c)
    );
}
