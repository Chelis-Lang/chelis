//! #1805/#1940/#1941: definition contracts replace callee-body validation.
mod common;

use assert_cmd::Command;
use std::{fs, path::Path, process::Output};
use tempfile::TempDir;

fn cli(command: &str, path: &Path, output: &Path) -> Output {
    let mut cli = Command::cargo_bin("chelis").unwrap();
    cli.env("CHELIS_STYLE_GATE_DISABLE", "1").arg(command);
    if command == "eval" {
        cli.arg("--file");
    }
    cli.arg(path);
    if command == "build" {
        cli.args(["--target", "c", "-o"]).arg(output);
    }
    cli.output().unwrap()
}

fn programs(dtype: &str, values: &str) -> Vec<String> {
    let helper = "def g[p: Float](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n";
    vec![
        format!(
            "{helper}def apply_it(f: tensor[3, {dtype}] -> tensor[{dtype}], x: tensor[3, {dtype}]) -> tensor[{dtype}] = f(x)\nout = apply_it(g, to_tensor({values}))\n"
        ),
        format!(
            "{helper}def mid[p: Float](x: tensor[3, p]) -> tensor[p] = g(x)\nout = mid(to_tensor({values}))\n"
        ),
        format!(
            "{helper}def id2[q](x: tensor[3, q]) -> tensor[3, q] = x\nout = g(id2(to_tensor({values})))\n"
        ),
        format!(
            "def g[p: Float](x: tensor[3, p]) -> tensor[p] = {{\n h = fn(t) -> mean(t, 0i32)\n h(x)\n}}\nout = g(to_tensor({values}))\n"
        ),
        format!(
            "type Box =\n | Box {{ t: tensor[3, {dtype}] }}\n{helper}def pick(b: Box) -> tensor[{dtype}] = g(b.t)\nout = pick(Box {{ t: to_tensor({values}) }})\n"
        ),
    ]
}

#[test]
fn invalid_function_value_chain_lambda_and_field_calls_stop_before_execution() {
    let dir = TempDir::new().unwrap();
    let sources: Vec<_> = programs("i32", "[1i32, 2i32, 4i32]")
        .into_iter()
        .flat_map(|source| [source.replace("[p: Float]", "[p]"), source])
        .collect();
    for (index, source) in sources.iter().enumerate() {
        let path = dir.path().join(format!("rejected_{index}.ch"));
        fs::write(&path, source).unwrap();
        for command in ["check", "eval", "build"] {
            let output = cli(command, &path, &dir.path().join("out"));
            assert!(!output.status.success(), "{command}: {source}");
            let rendered = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                rendered.contains("Float") && rendered.contains("i32"),
                "{command}: {rendered}"
            );
            if command == "check" {
                let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
                assert!(
                    report["errors"]
                        .as_array()
                        .is_some_and(|errors| !errors.is_empty()),
                    "{report}"
                );
            }
        }
    }
}

#[test]
fn bounded_function_value_chain_lambda_and_field_calls_execute_on_eval_and_c() {
    assert!(
        common::gcc_available(),
        "native C execution is required for this receipt"
    );
    let dir = TempDir::new().unwrap();
    for (index, source) in programs("f32", "[1.0f32, 2.0f32, 6.0f32]")
        .iter()
        .enumerate()
    {
        let stem = format!("accepted_{index}");
        let path = dir.path().join(format!("{stem}.ch"));
        let output_dir = dir.path().join(format!("{stem}-out"));
        fs::write(&path, source).unwrap();
        let checked = cli("check", &path, &output_dir);
        assert!(
            checked.status.success(),
            "{}",
            String::from_utf8_lossy(&checked.stdout)
        );
        let evaluated = cli("eval", &path, &output_dir);
        assert!(
            evaluated.status.success(),
            "{}",
            String::from_utf8_lossy(&evaluated.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&evaluated.stdout).trim(),
            "out = 3.0"
        );
        let built = cli("build", &path, &output_dir);
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let run = std::process::Command::new(output_dir.join(&stem))
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let value = String::from_utf8_lossy(&run.stdout);
        assert!(
            value.trim() == "out = 3" || value.trim() == "out = 3.0",
            "{value}"
        );
    }
}

#[test]
fn insufficient_authored_contracts_are_rejected_without_a_call_site() {
    let dir = TempDir::new().unwrap();
    for source in [
        "def g[p](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n",
        "sig g[p: Numeric]: tensor[3, p] -> tensor[p]\ndef g(x) = mean(x, 0i32)\n",
        "def g(x) = sin(x)\n",
        "def g(x) = sin(x)\nout = g(0.0f32)\n",
        "def make() = fn (x) -> sin(x)\n",
        "def source[p: Float]() -> p = cast(0.0f32, p)\ndef make() = source()\n",
        "def g[p: Float](x: p) -> p = sin(x)\ndef wrap(x) = g(x)\n",
    ] {
        let path = dir.path().join("contract.ch");
        fs::write(&path, source).unwrap();
        let checked = cli("check", &path, &dir.path().join("out"));
        assert!(!checked.status.success());
        let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert!(
            report["errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error["kind"] == "PrecisionMismatch"),
            "{report}"
        );
        assert!(report["score"].as_f64().unwrap() < 1.0, "{report}");
        for command in ["eval", "build"] {
            let rejected = cli(command, &path, &dir.path().join("out"));
            assert!(!rejected.status.success(), "{command}: {source}");
            let rendered = format!(
                "{}{}",
                String::from_utf8_lossy(&rejected.stdout),
                String::from_utf8_lossy(&rejected.stderr)
            );
            assert!(rendered.contains("Float"), "{command}: {rendered}");
        }
    }
    for source in [
        "def g(x: f32) -> f32 = sin(x)\n",
        "def make() = fn (x: f32) -> sin(x)\n",
        "primitive = sin\n",
    ] {
        let path = dir.path().join("accepted.ch");
        fs::write(&path, source).unwrap();
        let checked = cli("check", &path, &dir.path().join("out"));
        assert!(
            checked.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&checked.stdout)
        );
        let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert_eq!(report["score"].as_f64(), Some(1.0));
        assert!(report["errors"].as_array().unwrap().is_empty());
    }
    let path = dir.path().join("comparison.ch");
    for (source, accepted) in [
        (
            "def less[p](x: p, y: p) -> bool = lt(x, y)\nout = less(true, false)\n",
            false,
        ),
        (
            "def less[p: Numeric](x: p, y: p) -> bool = lt(x, y)\nout = less(1i32, 2i32)\n",
            true,
        ),
    ] {
        fs::write(&path, source).unwrap();
        let checked = cli("check", &path, &dir.path().join("comparison-out"));
        assert_eq!(
            checked.status.success(),
            accepted,
            "{source}: {}",
            String::from_utf8_lossy(&checked.stdout)
        );
        let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
        if accepted {
            assert_eq!(report["score"].as_f64(), Some(1.0));
            assert!(report["errors"].as_array().unwrap().is_empty());
        } else {
            assert!(
                report["errors"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|error| error["kind"] == "PrecisionMismatch"),
                "{report}"
            );
        }
    }
}

#[test]
fn window_reduction_contracts_reject_invalid_public_calls() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("window-contract.ch");
    for (operation, bad_binder, good_binder, bad_dtype, good_dtype) in [
        ("reduce_window_mean", "p: Numeric", "p: Float", "i32", "f32"),
        ("reduce_window_sum", "p", "p: Numeric", "bool", "i32"),
        ("reduce_window_max", "p", "p: Numeric", "bool", "i32"),
        ("reduce_window_min", "p", "p: Numeric", "bool", "i32"),
    ] {
        for alias in [false, true] {
            let prefix = if alias {
                format!("window_op = {operation}\n")
            } else {
                String::new()
            };
            let callee = if alias { "window_op" } else { operation };
            for (source, accepted) in [
                (
                    format!(
                        "{prefix}def g[{bad_binder}](x: tensor[3, p]) -> tensor[2, p] = \
                         {callee}(x, [2i64], [1i64])\n"
                    ),
                    false,
                ),
                (
                    format!(
                        "{prefix}def g[{good_binder}](x: tensor[3, p]) -> tensor[2, p] = \
                         {callee}(x, [2i64], [1i64])\n"
                    ),
                    true,
                ),
                (
                    format!(
                        "{prefix}def g(x: tensor[3, {bad_dtype}]) -> tensor[2, {bad_dtype}] = \
                         {callee}(x, [2i64], [1i64])\n"
                    ),
                    false,
                ),
                (
                    format!(
                        "{prefix}def g(x: tensor[3, {good_dtype}]) -> tensor[2, {good_dtype}] = \
                         {callee}(x, [2i64], [1i64])\n"
                    ),
                    true,
                ),
            ] {
                fs::write(&path, &source).unwrap();
                let checked = cli("check", &path, &dir.path().join("window-out"));
                assert_eq!(
                    checked.status.success(),
                    accepted,
                    "{source}: {}{}",
                    String::from_utf8_lossy(&checked.stdout),
                    String::from_utf8_lossy(&checked.stderr)
                );
                let report: serde_json::Value =
                    serde_json::from_slice(&checked.stdout).expect("check JSON");
                if accepted {
                    assert_eq!(report["score"].as_f64(), Some(1.0), "{source}: {report}");
                    assert!(report["errors"].as_array().unwrap().is_empty());
                } else {
                    assert!(
                        report["errors"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|error| error["kind"] == "PrecisionMismatch"),
                        "{source}: {report}"
                    );
                }
            }
        }
    }
}

#[test]
fn deferred_window_shape_errors_precede_late_family_rejection() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("deferred-window-contract.ch");
    let source = |dtype: &str, windows: &str| {
        format!(
            "def use(x: tensor[3, {dtype}]) -> tensor[2, {dtype}] = {{\n\
             windowed = fn (value) -> \
             reduce_window_sum(value, {windows}, {windows})\n\
             windowed(x)\n\
             }}\n"
        )
    };

    for dtype in ["bool", "i32"] {
        let invalid_rank = source(dtype, "[1i64, 1i64]");
        fs::write(&path, &invalid_rank).unwrap();
        let checked = cli("check", &path, &dir.path().join("deferred-window-out"));
        assert!(!checked.status.success(), "{invalid_rank}");
        let report: serde_json::Value =
            serde_json::from_slice(&checked.stdout).expect("check JSON");
        let errors = report["errors"].as_array().expect("check errors");
        assert!(
            errors.iter().any(|error| {
                error["kind"] == "DimensionMismatch"
                    && error["expected"] == "window arity at most 1"
                    && error["got"] == "window arity 2"
                    && error["span"]["offset"].as_u64()
                        == invalid_rank
                            .find("reduce_window_sum(")
                            .map(|offset| offset as u64)
            }),
            "{invalid_rank}: {report}"
        );
        assert!(
            errors
                .iter()
                .all(|error| error["kind"] != "PrecisionMismatch"),
            "{invalid_rank}: {report}"
        );
    }

    let valid_rank = source("bool", "[1i64]");
    fs::write(&path, &valid_rank).unwrap();
    let checked = cli("check", &path, &dir.path().join("deferred-window-out"));
    assert!(!checked.status.success(), "{valid_rank}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).expect("check JSON");
    assert!(
        report["errors"]
            .as_array()
            .expect("check errors")
            .iter()
            .any(|error| error["kind"] == "PrecisionMismatch"),
        "{valid_rank}: {report}"
    );
}
