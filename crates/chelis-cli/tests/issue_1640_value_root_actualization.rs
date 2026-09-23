//! §5.8.1: every reachable tensor precision is concrete at a checked call,
//! including calls whose result is stored directly as a top-level value.
use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn eval(source: &str) -> String {
    let dir = tempdir().unwrap();
    let path = dir.path().join("actualize.ch");
    fs::write(&path, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .assert()
        .success();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{source}\n{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

fn assert_lanes(source: &str, expected: &[(&str, &[&str])]) {
    let interpreted = eval(source);
    let native = common::build_and_run(source, "value_root_actualization");
    for (name, values) in expected {
        let prefix = format!("{name} = ");
        let line = interpreted
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap();
        assert_eq!(
            common::printed_value_tokens(line),
            *values,
            "{source}\n{interpreted}"
        );
    }
    assert_eq!(native.trim(), interpreted.trim(), "{source}");
}

#[test]
fn comparison_value_roots_and_typed_main_twins_execute_at_f32_and_f64() {
    for dtype in ["f32", "f64"] {
        for root in ["def main() -> tensor[3, bool]", "out"] {
            let source = format!(
                "def f[p: Float](xs: tensor[3, p]) -> tensor[3, bool] = gt(xs, insert(scalar_to_tensor(cast(1.5, p)), 0i32, shape(xs, 0i32)))\n{root} = f(to_tensor([1.0{dtype}, 2.0{dtype}, 3.0{dtype}]))\n"
            );
            let name = if root == "out" { "out" } else { "main" };
            assert_lanes(&source, &[(name, &["false", "true", "true"])]);
        }
    }
}

#[test]
fn comparison_value_roots_compile_without_a_typed_main_wrapper() {
    for dtype in ["f32", "f64"] {
        let source = format!(
            "def f[p: Float](xs: tensor[3, p]) -> tensor[3, bool] = gt(xs, insert(scalar_to_tensor(cast(1.5, p)), 0i32, shape(xs, 0i32)))\nout = f(to_tensor([1.0{dtype}, 2.0{dtype}, 3.0{dtype}]))\n"
        );
        let native = common::build_and_run(&source, "value_root_comparison");
        assert_eq!(
            common::printed_value_tokens(native.trim()),
            ["false", "true", "true"]
        );
    }
}

#[test]
fn scalar_witness_constructors_actualize_rank_zero_and_rank_two() {
    for dtype in ["f32", "f64"] {
        for (shape, body, values) in [
            ("", "scalar_to_tensor(cast(1.5, p))", vec!["1.5"]),
            (
                "2, 3, ",
                "insert(insert(scalar_to_tensor(cast(1.5, p)), 0i32, 3i64), 0i32, 2i64)",
                vec!["1.5"; 6],
            ),
        ] {
            for root in [
                format!("def main() -> tensor[{shape}{dtype}]"),
                "out".into(),
            ] {
                let source = format!(
                    "def f[p: Float](witness: p) -> tensor[{shape}p] = {body}\n{root} = f(0.0{dtype})\n"
                );
                let name = if root == "out" { "out" } else { "main" };
                assert_lanes(&source, &[(name, &values)]);
            }
        }
    }
}

#[test]
fn value_roots_keep_distinct_instantiations_and_bool_results_do_not_seed_dtype() {
    // 1.000000001 rounds to 1.0 at f32 and stays distinct at f64. The witness
    // must be finite at every `Float` member, f16 included ([04-LIT-2]).
    let source = "def f[p: Float](xs: tensor[1, p]) -> tensor[1, bool] = gt(xs, insert(scalar_to_tensor(cast(1.000000001, p)), 0i32, shape(xs, 0i32)))\na = f(to_tensor([1.000000001f32]))\nb = f(to_tensor([1.000000001f64]))\nc = f(to_tensor([1.000000002f64]))\n";
    assert_lanes(
        source,
        &[("a", &["false"]), ("b", &["false"]), ("c", &["true"])],
    );
    let source = "def f[p: Float](witness: p) -> tensor[p] = scalar_to_tensor(cast(1.000000001, p))\na = f(0.0f32)\nb = f(0.0f64)\n";
    assert_lanes(source, &[("a", &["1.0"]), ("b", &["1.000000001"])]);
}

#[test]
fn scalar_witnesses_actualize_body_precision_even_when_the_result_is_bool() {
    for dtype in ["f32", "f64"] {
        let source = format!(
            "def f[p: Float](witness: p) -> tensor[bool] = gt(scalar_to_tensor(witness), scalar_to_tensor(cast(1.5, p)))\nout = f(2.0{dtype})\n"
        );
        assert_lanes(&source, &[("out", &["true"])]);
    }
}

#[test]
fn invalid_or_unresolved_dtype_calls_never_reach_execution() {
    for (source, reason) in [
        (
            "def f[p: Float](xs: tensor[1, p]) -> tensor[1, bool] = gt(xs, xs)\nout = f(to_tensor([1i32]))\n",
            "dtype family `Float`",
        ),
        (
            "def f[p](witness: p) -> tensor[p] = scalar_to_tensor(cast(1.5, p))\nout = f(0.0f32)\n",
            "[04-DTYPE-1]",
        ),
        (
            "def f[p: Float](witness: p) -> tensor[p] = scalar_to_tensor(cast(1.5, missing_dtype))\nout = f(0.0f32)\n",
            "not a recognized primitive type",
        ),
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.ch");
        fs::write(&path, source).unwrap();
        for action in ["check", "eval", "build"] {
            let mut command = Command::cargo_bin("chelis").unwrap();
            command.env("CHELIS_STYLE_GATE_DISABLE", "1").arg(action);
            if action == "eval" {
                command.arg("--file");
            }
            command.arg(&path);
            if action == "build" {
                command
                    .args(["--target", "c", "--output"])
                    .arg(dir.path().join("out"));
            }
            let output = command.output().unwrap();
            assert!(
                !output.status.success(),
                "{action} accepted {source}\n{output:?}"
            );
            let diagnostic = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(!diagnostic.contains("BUG:"), "{action}: {diagnostic}");
            assert!(diagnostic.contains(reason), "{action}: {diagnostic}");
        }
    }
}

#[test]
fn an_unconstrained_generic_value_is_not_given_a_fallback_dtype() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("unresolved.ch");
    fs::write(
        &path,
        "def f[p: Float]() -> tensor[p] = scalar_to_tensor(cast(1.5, p))\nout = f()\n",
    )
    .unwrap();
    // [04-INF-9] rejects the unresolved result at its declaration boundary:
    // neither checking nor execution may invent a contract or a dtype.
    Command::cargo_bin("chelis")
        .unwrap()
        .arg("check")
        .arg(&path)
        .assert()
        .failure();
    for action in ["eval", "build"] {
        let mut command = Command::cargo_bin("chelis").unwrap();
        command.arg(action);
        if action == "eval" {
            command.arg("--file");
        }
        command.arg(&path);
        if action == "build" {
            command
                .args(["--target", "c", "--output"])
                .arg(dir.path().join("out"));
        }
        let output = command.output().unwrap();
        assert!(
            !output.status.success(),
            "unresolved value acquired a dtype: {output:?}"
        );
        assert!(output.stdout.is_empty(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("declared contract"),
            "{output:?}"
        );
    }
}
