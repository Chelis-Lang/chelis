//! A function body's host C reads a top-level value that a tensor helper
//! takes as an input through the global holding it, typed as that global is:
//! a rank-zero tensor as a tensor, a host scalar as a scalar. The helper's
//! `Load` alone cannot tell the two apart, so an input named by no local was
//! typed from its rank, and a rank-zero tensor global emitted `int32_t
//! __tensor_scalar1_1 = y;` over a `chelis_tensor *`, which C rejects.
//! Section 12 of the switch made the dead references reach it, by lowering a
//! trapping value's initializer again inside the function's helper, where it
//! reads the top-level values it names; the live references reached it on
//! main already. Every row's C must compile and end exactly as
//! `chelis eval --file` does.
//!
//! Witnesses: chelis#2586 reviewer 1b's `s12` probes (`b1`, `c1`, `d3`,
//! `d4`), and twins for a total live read, a live shadowing local, and a host
//! scalar global.

use assert_cmd::Command;
use std::path::Path;
use tempfile::tempdir;

const MUL_OVERFLOW: &str = "numeric trap: overflow in mul at i32";

/// A dead reference in a body that also binds a dead local `y`.
const DEAD_UNDER_A_DEAD_LOCAL: &str = "y = scalar_to_tensor(2i32)
v = mul(copy(y), scalar_to_tensor(2000000000i32))
def g(x: tensor[f32]) -> tensor[f32] = {
  y = scalar_to_tensor(1i32)
  dead = copy(v)
  x
}
def main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))
";

/// A dead reference.
const DEAD: &str = "y = scalar_to_tensor(2i32)
v = mul(copy(y), scalar_to_tensor(2000000000i32))
y2 = scalar_to_tensor(1i32)
def g(x: tensor[f32]) -> tensor[f32] = {
  dead = copy(v)
  x
}
def main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))
";

/// A live reference whose initializer traps.
const LIVE_TRAPPING: &str = "y = scalar_to_tensor(2i32)
v = mul(copy(y), scalar_to_tensor(2000000000i32))
def g(x: tensor[f32]) -> tensor[f32] = {
  live = cast(copy(v), f32)
  add(x, live)
}
def main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))
";

/// A live reference to a float value, which is total and not lowered again.
const LIVE_FLOAT: &str = "y = scalar_to_tensor(2.0f32)
v = mul(copy(y), scalar_to_tensor(3.0f32))
def g(x: tensor[f32]) -> tensor[f32] = {
  live = copy(v)
  add(x, live)
}
def main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))
";

/// A live reference to a total integer value.
const LIVE_TOTAL: &str = "y = scalar_to_tensor(2i32)
v = mul(copy(y), scalar_to_tensor(3i32))
def g(x: tensor[f32]) -> tensor[f32] = {
  live = cast(copy(v), f32)
  add(x, live)
}
def main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))
";

/// A live reference beside a live local `y`, which must not answer the
/// initializer's `y`.
const LIVE_BESIDE_A_SHADOWING_LOCAL: &str = "y = scalar_to_tensor(2i32)
v = mul(copy(y), scalar_to_tensor(3i32))
def g(x: tensor[f32]) -> tensor[f32] = {
  y = scalar_to_tensor(1i32)
  live = cast(copy(v), f32)
  add(add(x, live), cast(y, f32))
}
def main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))
";

/// A live reference over a host scalar global `y`.
const LIVE_OVER_A_HOST_SCALAR_GLOBAL: &str = "y = 2i32
v = mul(scalar_to_tensor(y), scalar_to_tensor(3i32))
def g(x: tensor[f32]) -> tensor[f32] = {
  live = cast(copy(v), f32)
  add(x, live)
}
def main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))
";

/// Each row, and how it ends: the trap, or the exact stdout.
const ROWS: [(&str, &str, Result<&str, &str>); 7] = [
    ("dead", DEAD, Err(MUL_OVERFLOW)),
    (
        "dead under a dead local",
        DEAD_UNDER_A_DEAD_LOCAL,
        Err(MUL_OVERFLOW),
    ),
    ("live trapping", LIVE_TRAPPING, Err(MUL_OVERFLOW)),
    (
        "live float",
        LIVE_FLOAT,
        Ok("y = 2.0\nv = 6.0\nmain = 7.0\n"),
    ),
    ("live total", LIVE_TOTAL, Ok("y = 2\nv = 6\nmain = 7.0\n")),
    (
        "live beside a shadowing local",
        LIVE_BESIDE_A_SHADOWING_LOCAL,
        Ok("y = 2\nv = 6\nmain = 8.0\n"),
    ),
    (
        "live over a host scalar global",
        LIVE_OVER_A_HOST_SCALAR_GLOBAL,
        Ok("y = 2\nv = 6\nmain = 7.0\n"),
    ),
];

fn chelis(directory: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Whether `output` ended as `expected` says: aborting with the trap on
/// stderr, or returning with exactly the stdout.
fn ends_as(output: &std::process::Output, expected: Result<&str, &str>) -> bool {
    match expected {
        Err(trap) => {
            !output.status.success() && String::from_utf8_lossy(&output.stderr).contains(trap)
        }
        Ok(stdout) => output.status.success() && String::from_utf8_lossy(&output.stdout) == stdout,
    }
}

/// Evidentiary status: at b47fdd7d3 the C of every row but the host scalar
/// global failed to compile, and the eval rows held. REGRESSION TEST for
/// the two dead rows, which compiled and trapped on main 7807ca4ff; on main
/// every live row's C failed to compile, the host scalar global's included.
#[test]
fn a_function_body_reads_a_top_level_value_as_its_global_holds_it_in_c() {
    let directory = tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (row, source, expected)) in ROWS.into_iter().enumerate() {
        let stem = format!("row_{index}");
        let path = format!("{stem}.ch");
        std::fs::write(directory.path().join(&path), source).unwrap();
        let evaluated = chelis(directory.path(), &["eval", "--file", &path]);
        if !ends_as(&evaluated, expected) {
            failures.push(format!("eval {row}: {}", text(&evaluated).trim()));
        }
        let out_dir = format!("{stem}_out");
        let built = chelis(directory.path(), &["build", &path, "--output", &out_dir]);
        if !built.status.success() {
            failures.push(format!("build {row}: {}", text(&built).trim()));
            continue;
        }
        let out_dir = directory.path().join(&out_dir);
        let ran = std::process::Command::new(out_dir.join(&stem))
            .output()
            .unwrap();
        if !ends_as(&ran, expected) {
            failures.push(format!("C {row}: {}", text(&ran).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
