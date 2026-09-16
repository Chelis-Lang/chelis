//! spec/06 §§2.2–2.3: actual CLI, C compilation/linking, and native observations.
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn cli(path: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(assert_cmd::cargo_bin!("chelis"));
    command.current_dir(path).args(args);
    command
}

fn run(command: &mut Command) -> Output {
    let output = command.output().unwrap();
    assert!(output.status.success(), "{command:?}\n{output:?}");
    output
}

fn native(path: &Path, source: &str) -> PathBuf {
    std::fs::write(path.join("zero.ch"), source).unwrap();
    run(&mut cli(path, &["fmt", "--check", "zero.ch"]));
    let checked = run(&mut cli(path, &["check", "zero.ch"]));
    let checked: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(checked["score"], 1);
    assert_eq!(checked["errors"], serde_json::json!([]));
    run(&mut cli(
        path,
        &["build", "zero.ch", "--target", "c", "--output", "native"],
    ));
    let dir = path.join("native");
    let binary = dir.join("zero-run");
    let mut cc = Command::new("cc");
    cc.current_dir(&dir)
        .args(["-O2", "zero.c", "libchelis_runtime.a", "-lm", "-o"])
        .arg(&binary);
    if cfg!(target_os = "macos") {
        cc.args(["-framework", "Accelerate"]);
    } else {
        cc.args(["-lpthread", "-ldl"]);
    }
    run(&mut cc);
    binary
}

fn agrees(source: &str, expected: &str) {
    let temp = tempfile::tempdir().unwrap();
    let binary = native(temp.path(), source);
    let evaluated = run(&mut cli(temp.path(), &["eval", "--file", "zero.ch"]));
    let executed = run(&mut Command::new(binary));
    assert_eq!(
        String::from_utf8(evaluated.stdout).unwrap().trim(),
        expected
    );
    assert_eq!(String::from_utf8(executed.stdout).unwrap().trim(), expected);
}

#[test]
fn fused_example_preserves_single_f32_zero_in_native_execution() {
    agrees(
        include_str!("../../../examples/grad_fused_zero.ch"),
        "out = tensor(shape=[2], data=[0.0, 0.0])",
    );
}

#[test]
fn fused_single_f64_zero_finishes_native_execution() {
    agrees(
        "def loss(x: tensor[f64]) -> f64 = 3.0f64\nout = vmap(grad(loss))(to_tensor([2.0f64, 7.0f64]))\n",
        "out = tensor(shape=[2], data=[0.0, 0.0])",
    );
}

#[test]
fn fused_mixed_and_repeated_eval_keeps_recorded_c_rejection_boundary() {
    // Exact sources reject before C emission on immutable c890 and this
    // repair. This records an admission boundary, not semantic invalidity
    // or native duplicate-output cleanup coverage (chelis#1932/#1985).
    for (selection, zero_slots) in [
        ("", vec![false, true]),
        (", wrt=(unused, x)", vec![true, false]),
        (", wrt=(unused, unused, x)", vec![true, true, false]),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let source = format!(
            "def loss(x: tensor[f32], unused: tensor[f32]) -> f32 = tensor_to_scalar(mul(x, x))\nout = vmap(grad(loss{selection}))(to_tensor([2.0f32, 7.0f32]), to_tensor([3.0f32, 11.0f32]))\n"
        );
        std::fs::write(temp.path().join("zero.ch"), source).unwrap();
        run(&mut cli(temp.path(), &["fmt", "--check", "zero.ch"]));
        let checked = run(&mut cli(temp.path(), &["check", "zero.ch"]));
        let checked: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert_eq!(checked["score"], 1);
        assert_eq!(checked["errors"], serde_json::json!([]));
        let output = run(&mut cli(
            temp.path(),
            &["eval", "--json", "--file", "zero.ch"],
        ));
        let output: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let roots = output["roots"].as_array().unwrap();
        assert_eq!(roots.len(), zero_slots.len());
        for (index, (root, zero)) in roots.iter().zip(zero_slots).enumerate() {
            assert_eq!(root["name"], format!("out.{index}"));
            let bits = if zero {
                ["00000000", "00000000"]
            } else {
                ["40800000", "41600000"]
            };
            assert_eq!(
                root["value"],
                serde_json::json!({"type":"tensor", "value":{"shape":[2], "data":{"dtype":"f32", "bits":bits}}})
            );
        }
        let rejected = cli(
            temp.path(),
            &["build", "zero.ch", "--target", "c", "--output", "native"],
        )
        .output()
        .unwrap();
        assert_eq!(rejected.status.code(), Some(1), "{rejected:?}");
        assert!(rejected.stdout.is_empty(), "{rejected:?}");
        let error = String::from_utf8(rejected.stderr).unwrap();
        assert!(
            error.starts_with("error: `chelis build --target c` can't lower these defs."),
            "{error}"
        );
        assert!(error.ends_with("Affected defs: out\n"), "{error}");
    }
}

#[test]
fn fused_constant_primal_trap_survives_successful_native_link() {
    let temp = tempfile::tempdir().unwrap();
    let binary = native(
        temp.path(),
        "def loss(x: tensor[f32]) -> f32 = cast(cast(2147483648.0f64, i32), f32)\nout = vmap(grad(loss))(to_tensor([2.0f32, 7.0f32]))\n",
    );
    for output in [
        cli(temp.path(), &["eval", "--file", "zero.ch"])
            .output()
            .unwrap(),
        Command::new(binary).output().unwrap(),
    ] {
        assert!(!output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("overflow"),
            "{output:?}"
        );
    }
}
