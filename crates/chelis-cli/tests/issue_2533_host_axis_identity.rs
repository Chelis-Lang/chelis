//! Generic host extraction preserves physical axes (spec/04 §4.7,
//! spec/05 [05-OP-65]); a result permutation cannot retype its input.
mod common;

use common::{gcc_available, link_generated};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn run(binary: &Path, args: &[&str], cwd: &Path) -> Output {
    Command::new(binary)
        .args(args)
        .current_dir(cwd)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .output()
        .unwrap()
}

fn check_case(width: usize, aliased: bool, disagreement: bool) {
    assert!(
        gcc_available(),
        "C compiler required for Eval/C axis parity"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.ch");
    let source = format!(
        r#"def candidate[n](x: tensor[n, f64]) -> tensor[{}, 4, f64] ! {{ IO }} = {{
  _ = print("before-result")
  {}
  result = {}
  _ = print("after-result")
  result
}}
out = candidate(to_tensor([{}]))
"#,
        if disagreement { "5" } else { "n" },
        if aliased {
            if disagreement {
                "a = insert(scalar_to_tensor(16777217.0f64), 0i32, shape(x, 0i32))\n  b = insert(a, 0i32, 4i64)"
            } else {
                "a = x\n  b = insert(a, 0i32, 4i64)"
            }
        } else {
            ""
        },
        if aliased {
            "permute(b, 1i32, 0i32)"
        } else if disagreement {
            "permute(insert(insert(scalar_to_tensor(16777217.0f64), 0i32, shape(x, 0i32)), 0i32, 4i64), 1i32, 0i32)"
        } else {
            "permute(insert(x, 0i32, 4i64), 1i32, 0i32)"
        },
        vec!["16777217.0f64"; width].join(", "),
    );
    fs::write(&path, source).unwrap();
    let cli = Path::new(env!("CARGO_BIN_EXE_chelis"));
    let fmt = run(
        cli,
        &["fmt", "--inplace", path.to_str().unwrap()],
        dir.path(),
    );
    assert!(fmt.status.success(), "{fmt:?}");
    let checked = run(cli, &["check", path.to_str().unwrap()], dir.path());
    assert!(checked.status.success(), "{checked:?}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"], 1.0, "{report}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");
    let eval = run(cli, &["eval", "--file", path.to_str().unwrap()], dir.path());
    let out = dir.path().join("native");
    let built = run(
        cli,
        &[
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            out.to_str().unwrap(),
        ],
        dir.path(),
    );
    assert!(built.status.success(), "{built:?}");
    assert!(link_generated(&out, "fixture.c", "fixture").success());
    let native = run(&out.join("fixture"), &[], dir.path());
    for (lane, result) in [("eval", eval), ("c", native)] {
        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert_eq!(
            result.status.success(),
            !disagreement,
            "{lane}: {stdout}\n{stderr}"
        );
        assert_eq!(
            stdout.matches("before-result").count(),
            1,
            "{lane}: {stdout}\n{stderr}"
        );
        assert_eq!(
            stdout.matches("after-result").count(),
            usize::from(!disagreement),
            "{lane}: {stdout}\n{stderr}"
        );
        if disagreement {
            assert!(
                stderr.contains("numeric trap: domain in permute at i64"),
                "{lane}: {stderr}"
            );
            assert_eq!(
                stderr.matches("numeric trap:").count(),
                1,
                "{lane}: {stderr}"
            );
            assert!(stderr.contains("claimed = 5"), "{lane}: {stderr}");
            assert!(
                stderr.contains(&format!("permute axis 0 = {width}")),
                "{lane}: {stderr}"
            );
        } else {
            let expected = format!(
                "out = tensor(shape=[{width}, 4], data=[{}])",
                vec!["16777217.0"; width * 4].join(", ")
            );
            assert!(stdout.contains(&expected), "{lane}: {stdout}\n{stderr}");
        }
    }
}

#[test]
fn generic_insert_permute_retains_input_axes() {
    for width in [2, 3] {
        for aliased in [false, true] {
            check_case(width, aliased, false);
        }
    }
}

#[test]
fn generic_permuted_result_disagreement_traps_at_operation() {
    for width in [2, 3] {
        for aliased in [false, true] {
            check_case(width, aliased, true);
        }
    }
}
