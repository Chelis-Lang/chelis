//! [05-OP-80]: a dtype binder in `mmap_tensor`'s dtype position resolves in
//! the scope that wrote it, in eval and in compiled C alike.
//!
//! The matrix crosses binder spelling (caller and callee spell their binders
//! alike, or distinctly) with the call shape (a direct call; a generic caller
//! two and three deep; a lambda written in a generic caller; a lambda that
//! reads its writer's binder), with how the call determines the callee's
//! binder (a concrete dtype at the call, or only the caller's own binder),
//! and with the instantiated dtype. Every cell must succeed in both lanes
//! with byte-equal output. The named witnesses below the matrix are the
//! review probes that first showed the defect class.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

fn chelis() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

/// `Ok(stdout)` when the lane ran to success, else the failure text.
fn eval(dir: &TempDir, name: &str, source: &str) -> Result<String, String> {
    let path = dir.path().join(format!("{name}.ch"));
    fs::write(&path, source).expect("write source");
    let output = chelis()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval runs");
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

fn compiled(dir: &TempDir, name: &str, source: &str) -> Result<String, String> {
    let path = dir.path().join(format!("{name}.ch"));
    fs::write(&path, source).expect("write source");
    let out_dir = dir.path().join(format!("{name}-out"));
    let build = chelis()
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(&out_dir)
        .output()
        .expect("chelis build runs");
    if !build.status.success() {
        return Err(format!("build: {}", String::from_utf8_lossy(&build.stderr)));
    }
    let output = StdCommand::new(out_dir.join(name))
        .output()
        .expect("the compiled executable runs");
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

/// The verdict line of one cell: `agree`, or what went wrong.
fn verdict(dir: &TempDir, name: &str, source: &str) -> Result<String, String> {
    match (eval(dir, name, source), compiled(dir, name, source)) {
        (Ok(evaluated), Ok(built)) if evaluated == built => Ok(evaluated),
        (Ok(evaluated), Ok(built)) => {
            Err(format!("lanes disagree: eval {evaluated:?}, C {built:?}"))
        }
        (Err(error), _) => Err(format!("eval failed: {}", error.trim())),
        (_, Err(error)) => Err(format!("C failed: {}", last_line(&error))),
    }
}

fn last_line(text: &str) -> String {
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .chars()
        .take(200)
        .collect()
}

/// f32 [1.5, -2.0] at 0, f64 [0.25, -0.75] at 8, i16 [7, -3] at 24, and
/// bf16 [1.0, 0.5] at 28.
fn payload(dir: &TempDir) -> std::path::PathBuf {
    let mut bytes = Vec::new();
    bytes.extend([1.5f32, -2.0].iter().flat_map(|x| x.to_le_bytes()));
    bytes.extend([0.25f64, -0.75].iter().flat_map(|x| x.to_le_bytes()));
    bytes.extend([7i16, -3].iter().flat_map(|x| x.to_le_bytes()));
    bytes.extend([0x3f80u16, 0x3f00].iter().flat_map(|x| x.to_le_bytes()));
    let path = dir.path().join("payload.bin");
    fs::write(&path, bytes).expect("write payload");
    path
}

fn offset(dtype: &str) -> i64 {
    match dtype {
        "f32" => 0,
        "f64" => 8,
        other => panic!("no payload at {other}"),
    }
}

fn other(dtype: &'static str) -> &'static str {
    if dtype == "f32" { "f64" } else { "f32" }
}

fn main_reading(path: &Path, binding: &str) -> String {
    format!(
        "def main() -> unit ! {{ IO }} = {{\n  m = mmap_file({:?})\n  a: tensor[*, {binding}\n  print(a)\n}}\nrun = main()\n",
        path.to_str().unwrap()
    )
}

#[derive(Clone, Copy, Debug)]
enum Names {
    Same,
    Distinct,
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Direct,
    Nested2,
    Nested3,
    LambdaInCaller,
    LambdaReadsWriterBinder,
}

#[derive(Clone, Copy, Debug)]
enum Determined {
    /// The call site names the callee's dtype concretely.
    Concrete,
    /// The call site names it only through the caller's own binder.
    ThroughCaller,
}

/// The program of one cell, or `None` where the cell has no program: a
/// direct call from `main` has no caller binder to determine through.
fn cell_program(
    path: &Path,
    names: Names,
    shape: Shape,
    determined: Determined,
    dtype: &'static str,
) -> Option<String> {
    let (p, q, r) = match names {
        Names::Same => ("p", "p", "p"),
        Names::Distinct => ("p", "q", "r"),
    };
    let load = format!(
        "def load[{p}: Float](m: MappedFile, off: i64) -> tensor[*, {p}] = mmap_tensor(m, off, 2i64, {p})\n"
    );
    // The dtype a generic caller's inner call is ascribed at: its own binder,
    // or a concrete dtype other than the one `main` instantiates it at, so a
    // binder resolved in the wrong scope reads the wrong payload.
    let inner = |binder: &'static str| match determined {
        Determined::ThroughCaller => (binder, dtype),
        Determined::Concrete => (other(dtype), other(dtype)),
    };
    Some(match shape {
        Shape::Direct => {
            if matches!(determined, Determined::ThroughCaller) {
                return None;
            }
            format!(
                "{load}{}",
                main_reading(path, &format!("{dtype}] = load(m, {}i64)", offset(dtype)))
            )
        }
        Shape::Nested2 => {
            let (ascribed, read) = inner(q);
            format!(
                "{load}def twice[{q}: Float](m: MappedFile, off: i64) -> tensor[*, {q}] = {{\n  x: tensor[*, {ascribed}] = load(m, off)\n  cast(add(x, x), {q})\n}}\n{}",
                main_reading(path, &format!("{dtype}] = twice(m, {}i64)", offset(read)))
            )
        }
        Shape::Nested3 => {
            let (ascribed, read) = inner(r);
            format!(
                "{load}def twice[{q}: Float](m: MappedFile, off: i64) -> tensor[*, {q}] = {{\n  x: tensor[*, {q}] = load(m, off)\n  cast(add(x, x), {q})\n}}\ndef thrice[{r}: Float](m: MappedFile, off: i64) -> tensor[*, {r}] = {{\n  y: tensor[*, {ascribed}] = twice(m, off)\n  cast(add(y, y), {r})\n}}\n{}",
                main_reading(path, &format!("{dtype}] = thrice(m, {}i64)", offset(read)))
            )
        }
        Shape::LambdaInCaller | Shape::LambdaReadsWriterBinder => {
            let (ascribed, read) = inner(q);
            let lambda = if matches!(shape, Shape::LambdaInCaller) {
                format!("fn (z: tensor[*, {ascribed}]) -> add(z, z)")
            } else {
                format!(
                    "fn (z: tensor[*, {ascribed}]) -> {{\n    w: tensor[*, {q}] = mmap_tensor(m, own, 2i64, {q})\n    add(z, cast(w, {ascribed}))\n  }}"
                )
            };
            format!(
                "def apply_it[{p}: Float](m: MappedFile, off: i64, f: (tensor[*, {p}]) -> tensor[*, {p}]) -> tensor[*, {p}] = {{\n  x: tensor[*, {p}] = mmap_tensor(m, off, 2i64, {p})\n  f(x)\n}}\ndef outer[{q}: Float](m: MappedFile, off: i64, own: i64) -> tensor[*, {q}] = {{\n  g: tensor[*, {ascribed}] = apply_it(m, off, {lambda})\n  cast(g, {q})\n}}\n{}",
                main_reading(
                    path,
                    &format!(
                        "{dtype}] = outer(m, {}i64, {}i64)",
                        offset(read),
                        offset(dtype)
                    )
                )
            )
        }
    })
}

fn run_shapes(shapes: &[Shape]) {
    let dir = tempdir().expect("tempdir");
    let path = payload(&dir);
    let mut failures = Vec::new();
    for &shape in shapes {
        for names in [Names::Same, Names::Distinct] {
            for determined in [Determined::Concrete, Determined::ThroughCaller] {
                for dtype in ["f32", "f64"] {
                    let Some(source) = cell_program(&path, names, shape, determined, dtype) else {
                        continue;
                    };
                    let cell = format!("{shape:?}-{names:?}-{determined:?}-{dtype}");
                    let verdict = verdict(&dir, &cell.to_lowercase(), &source);
                    println!(
                        "cell {cell}: {}",
                        verdict
                            .as_ref()
                            .map_or_else(Clone::clone, |_| "agree".to_string())
                    );
                    if let Err(problem) = verdict {
                        failures.push(format!("{cell}: {problem}"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "failing cells:\n{}",
        failures.join("\n")
    );
}

#[test]
fn direct_and_nested_callers_resolve_binders_in_their_own_scope() {
    run_shapes(&[Shape::Direct, Shape::Nested2, Shape::Nested3]);
}

#[test]
fn lambdas_resolve_binders_in_the_scope_that_wrote_them() {
    run_shapes(&[Shape::LambdaInCaller, Shape::LambdaReadsWriterBinder]);
}

/// The review probes, verbatim in shape, with the values eval gives.
#[test]
fn review_witnesses_agree_in_both_lanes() {
    let dir = tempdir().expect("tempdir");
    let path = payload(&dir);
    let lam = |caller: &str| {
        format!(
            "def apply_it[p: Float](m: MappedFile, f: (tensor[*, p]) -> tensor[*, p]) -> tensor[*, p] = {{\n  x: tensor[*, p] = mmap_tensor(m, 0i64, 1i64, p)\n  f(x)\n}}\ndef outer[{caller}: Float](m: MappedFile) -> tensor[*, {caller}] = {{\n  g: tensor[*, f64] = apply_it(m, fn (z: tensor[*, f64]) -> {{\n    w: tensor[*, {caller}] = mmap_tensor(m, 0i64, 1i64, {caller})\n    add(z, cast(w, f64))\n  }})\n  cast(g, {caller})\n}}\n{}",
            main_reading(&path, "f32] = outer(m)")
        )
    };
    let twice = "def twice[q: Float](m: MappedFile) -> tensor[*, q] = {\n  x: tensor[*, q] = load(m, 2i64)\n  add(x, x)\n}\n";
    let witnesses: Vec<(&str, String, Option<&str>)> = vec![
        // The f64 payload's first bytes read as f32 with the f32 payload's
        // first element added: the binder is outer's f32, not apply_it's f64.
        ("lam2", lam("p"), None),
        ("lam3", lam("r"), None),
        (
            "gen5",
            format!(
                "def load[p: Float](m: MappedFile, n: i64) -> tensor[*, p] = mmap_tensor(m, 0i64, n, p)\n{}",
                main_reading(&path, "f32] = load(m, 2i64)")
            ),
            Some("tensor(shape=[2], data=[1.5, -2.0])\nrun = ()\n"),
        ),
        (
            "gen6",
            format!(
                "def load[p: Float](m: MappedFile, n: i64) -> tensor[*, p] = cast(mmap_tensor(m, 0i64, n, f32), p)\n{}",
                main_reading(&path, "f64] = load(m, 2i64)")
            ),
            Some("tensor(shape=[2], data=[1.5, -2.0])\nrun = ()\n"),
        ),
        (
            "gen9_i16",
            format!(
                "def load[p: Numeric](m: MappedFile, off: i64) -> tensor[*, p] = mmap_tensor(m, off, 2i64, p)\n{}",
                main_reading(&path, "i16] = load(m, 24i64)")
            ),
            Some("tensor(shape=[2], data=[7, -3])\nrun = ()\n"),
        ),
        (
            "gen9_bf16",
            format!(
                "def load[p: Numeric](m: MappedFile, off: i64) -> tensor[*, p] = mmap_tensor(m, off, 2i64, p)\n{}",
                main_reading(&path, "bf16] = load(m, 28i64)")
            ),
            Some("tensor(shape=[2], data=[1.0, 0.5])\nrun = ()\n"),
        ),
        (
            "gen10",
            format!(
                "def load[p: Float](m: MappedFile, n: i64) -> tensor[*, p] = cast(mmap_tensor(m, 0i64, n, f32), p)\n{twice}def main() -> unit ! {{ IO }} = {{\n  m = mmap_file({:?})\n  a: tensor[*, f32] = twice(m)\n  b: tensor[*, f64] = twice(m)\n  _ = print(a)\n  print(b)\n}}\nrun = main()\n",
                path.to_str().unwrap()
            ),
            Some(
                "tensor(shape=[2], data=[3.0, -4.0])\ntensor(shape=[2], data=[3.0, -4.0])\nrun = ()\n",
            ),
        ),
        (
            "gen11",
            format!(
                "def load[p: Float](m: MappedFile, n: i64) -> tensor[*, p] = cast(to_tensor([1.0, 2.0], f64), p)\n{twice}{}",
                main_reading(&path, "f32] = twice(m)")
            ),
            Some("tensor(shape=[2], data=[2.0, 4.0])\nrun = ()\n"),
        ),
        (
            "gen12",
            format!(
                "def load[p: Float](m: MappedFile, n: i64) -> tensor[*, p] = mmap_tensor(m, 0i64, n, p)\n{twice}{}",
                main_reading(&path, "f32] = twice(m)")
            ),
            Some("tensor(shape=[2], data=[3.0, -4.0])\nrun = ()\n"),
        ),
        (
            "gen13",
            format!(
                "def twice[q: Float](m: MappedFile) -> tensor[*, q] = {{\n  x: tensor[*, q] = mmap_tensor(m, 0i64, 2i64, q)\n  add(x, x)\n}}\n{}",
                main_reading(&path, "f32] = twice(m)")
            ),
            Some("tensor(shape=[2], data=[3.0, -4.0])\nrun = ()\n"),
        ),
        (
            "gen4",
            format!(
                "def load[p: Float](m: MappedFile, like: tensor[*, p]) -> tensor[*, p] = add(like, mmap_tensor(m, 0i64, 2i64, p))\n{}",
                main_reading(&path, "f32] = load(m, mmap_tensor(m, 0i64, 2i64, f32))")
            ),
            Some("tensor(shape=[2], data=[3.0, -4.0])\nrun = ()\n"),
        ),
    ];
    let mut failures = Vec::new();
    let mut lambda_outputs = Vec::new();
    for (name, source, expected) in witnesses {
        match verdict(&dir, name, &source) {
            Ok(output) => {
                println!("witness {name}: agree");
                if let Some(expected) = expected
                    && output != expected
                {
                    failures.push(format!("{name}: expected {expected:?}, got {output:?}"));
                }
                if name.starts_with("lam") {
                    lambda_outputs.push(output);
                }
            }
            Err(problem) => {
                println!("witness {name}: {problem}");
                failures.push(format!("{name}: {problem}"));
            }
        }
    }
    // Renaming the caller's binder changes nothing.
    if let [same, distinct] = lambda_outputs.as_slice() {
        assert_eq!(same, distinct, "lam2 and lam3 must print the same value");
    }
    assert!(
        failures.is_empty(),
        "failing witnesses:\n{}",
        failures.join("\n")
    );
}
