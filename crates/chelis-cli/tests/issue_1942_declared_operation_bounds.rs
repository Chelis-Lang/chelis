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
    let sources: Vec<_> = programs("int32", "[1i32, 2i32, 4i32]")
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
                rendered.contains("Float") && rendered.contains("int32"),
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
        assert!(common::link_generated(&output_dir, &format!("{stem}.c"), &stem).success());
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
    }
}
