//! Spec/04 §4.7 and runtime_extents.md C6.3: local host-tensor claims
//! retain their initializer, runtime guard, and static rejection.
mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

use assert_cmd::Command;
use result_claims::run;

const ORIGINAL: &str = "module Repro.Ascription\ndef value() -> tensor[1, 2, i64] = {\n ids = (pad_sequences_to([[cast(1, i64), cast(2, i64)]], cast(2, i64), cast(0, i64)) : tensor[1, 2, i64])\n ids\n}\noutput = value()\n";

fn assert_original(native: bool) {
    {
        let (ok, output) = run(ORIGINAL, native);
        assert!(ok, "{output}");
        assert!(
            output.contains("output = tensor(shape=[1, 2], data=[1, 2])"),
            "{output}"
        );
    }
}

fn assert_runtime(native: bool) {
    {
        for width in [2, 3, i64::MAX] {
            let source = format!(
                "def value(width: i64) -> tensor[*, *, i64] ! {{ IO }} = {{\n _ = print(\"before-ascription\")\n ids = (pad_sequences_to([[1i64, 2i64]], width, 0i64) : tensor[1, 2, i64])\n _ = print(\"after-ascription\")\n ids\n}}\nout = value({width}i64)\n"
            );
            let (ok, output) = run(&source, native);
            assert_eq!(ok, width == 2, "{output}");
            assert_eq!(output.matches("before-ascription").count(), 1, "{output}");
            assert_eq!(
                output.matches("after-ascription").count(),
                usize::from(ok),
                "{output}"
            );
            if ok {
                assert!(
                    output.contains("out = tensor(shape=[1, 2], data=[1, 2])"),
                    "{output}"
                );
            } else {
                assert!(
                    output.contains(&format!(
                        "extent `2`: claimed = 2, pad_sequences_to axis 1 = {width}"
                    )),
                    "{output}"
                );
                assert!(
                    output
                        .lines()
                        .any(|line| line == "numeric trap: domain in pad_sequences_to at i64"),
                    "{output}"
                );
                assert!(!output.contains("panicked"), "{output}");
            }
        }
    }
}

#[test]
fn static_host_tensor_ascription_disagreement_rejects_before_execution() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wrong.ch");
    std::fs::write(
        &path,
        ORIGINAL.replace("tensor[1, 2, i64]", "tensor[1, 3, i64]"),
    )
    .unwrap();
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!checked.status.success(), "{checked:?}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert!(report["score"].as_f64().unwrap() < 1.0, "{report}");
    assert!(!report["errors"].as_array().unwrap().is_empty(), "{report}");
    let (ok, output) = run(&std::fs::read_to_string(path).unwrap(), false);
    assert!(
        !ok && !output.contains("panicked") && !output.contains("numeric trap:"),
        "{output}"
    );
}

#[test]
fn eval_original_host_tensor_ascription() {
    assert_original(false);
}
#[test]
fn c_original_host_tensor_ascription() {
    assert_original(true);
}
#[test]
fn eval_runtime_host_tensor_ascription() {
    assert_runtime(false);
}
#[test]
fn c_runtime_host_tensor_ascription() {
    assert_runtime(true);
}

#[test]
fn aliases_retain_the_host_initializer_and_independent_axis_claims() {
    for native in [false, true] {
        for (batch, width, ok, diagnostic) in [
            (1, 2, true, ""),
            (
                2,
                2,
                false,
                "extent `2`: claimed = 2, pad_sequences_to axis 0 = 1",
            ),
            (
                1,
                3,
                false,
                "extent `3`: claimed = 3, pad_sequences_to axis 1 = 2",
            ),
        ] {
            let source = format!(
                "def value(width: i64) -> tensor[*, *, i64] ! {{ IO }} = {{\n _ = print(\"before-producer\")\n raw = pad_sequences_to([[1i64, 2i64]], width, 0i64)\n _ = print(\"between-producer-and-claim\")\n alias = raw\n ids: tensor[{batch}, {width}, i64] = alias\n ids\n}}\nout = value(2i64)\n"
            );
            let (success, output) = run(&source, native);
            assert_eq!(success, ok, "{output}");
            assert_eq!(output.matches("before-producer").count(), 1, "{output}");
            assert_eq!(
                output.matches("between-producer-and-claim").count(),
                usize::from(ok),
                "{output}"
            );
            if ok {
                assert!(
                    output.contains("out = tensor(shape=[1, 2], data=[1, 2])"),
                    "{output}"
                );
            } else {
                assert!(output.contains(diagnostic), "{output}");
            }
        }
    }
}

#[test]
fn named_host_local_witness_is_an_explicit_unsupported_residual() {
    let source = "def value[n](anchor: tensor[n, i64], width: i64) -> tensor[*, *, i64] = {\n ids: tensor[1, n, i64] = pad_sequences_to([[1i64, 2i64]], width, 0i64)\n ids\n}\nout = value(to_tensor([1i64, 2i64]), 2i64)\n";
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("named.ch");
    std::fs::write(&path, source).unwrap();
    for command in ["eval", "build"] {
        let mut cmd = Command::cargo_bin("chelis").unwrap();
        cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([command, "--allow-style-violations"]);
        if command == "eval" {
            cmd.arg("--file");
        }
        cmd.arg(&path);
        if command == "build" {
            cmd.args(["--target", "c", "-o"]).arg(dir.path().join("c"));
        }
        let output = cmd.output().unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.status.success(), "{text}");
        assert!(
            text.contains("named local host tensor witnesses are not implemented"),
            "{text}"
        );
        assert!(text.contains("2374"), "{text}");
        assert!(!text.contains("panicked"), "{text}");
    }
}

#[test]
fn signature_claim_precedes_later_local_claim_at_the_same_host_producer() {
    let source = "def value(width: i64) -> tensor[2, *, i64] = {\n ids: tensor[*, 3, i64] = pad_sequences_to([[1i64, 2i64]], width, 0i64)\n ids\n}\nout = value(4i64)\n";
    for native in [false, true] {
        let (ok, output) = run(source, native);
        assert!(!ok, "{output}");
        assert!(
            output.contains("extent `2`: claimed = 2, pad_sequences_to axis 0 = 1"),
            "{output}"
        );
        assert!(!output.contains("extent `3`"), "{output}");
    }
}

#[test]
fn tensor_native_copy_keeps_its_named_local_witness() {
    let source = "def value[n](anchor: tensor[n, i64], x: tensor[*, i64]) -> tensor[*, i64] = {\n ids: tensor[n, i64] = copy(x)\n ids\n}\nout = value(to_tensor([3i64, 4i64]), to_tensor([1i64, 2i64]))\n";
    for native in [false, true] {
        let (ok, output) = run(source, native);
        assert!(ok, "{output}");
        assert!(
            output.contains("out = tensor(shape=[2], data=[1, 2])"),
            "{output}"
        );
    }
}

#[test]
fn executable_example_retains_its_local_claim_without_a_literal_result_contract() {
    let source = include_str!("../../../examples/checked_host_local_ascription.ch");
    let disagreeing = source.replace("pad_pair(2i64)", "pad_pair(3i64)");
    assert_ne!(
        source, disagreeing,
        "the negative control must change the runtime width"
    );
    for native in [false, true] {
        let (ok, output) = run(source, native);
        assert!(ok, "{output}");
        assert!(
            output.contains("tensor(shape=[1, 2], data=[1, 2])"),
            "{output}"
        );
        let (ok, output) = run(&disagreeing, native);
        assert!(!ok, "{output}");
        assert!(
            output.contains("extent `2`: claimed = 2, pad_sequences_to axis 1 = 3"),
            "{output}"
        );
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in pad_sequences_to at i64"),
            "{output}"
        );
        assert!(!output.contains("panicked"), "{output}");
    }
}
