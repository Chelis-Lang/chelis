use std::fs;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

fn combined(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn cli_emits_i_names_but_preserves_execution_value_interchange_tags() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("canonical.ch");
    fs::write(&path, "value: i64 = 3000000000i64\n").expect("write");

    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep"])
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("(t-prim {} i64)"))
        .stdout(predicate::str::contains("(t-prim {} int64)").not())
        .get_output()
        .stdout
        .clone();
    fs::write(dir.path().join("canonical.dp"), deep).expect("write Deep");

    for input in ["canonical.ch", "canonical.dp"] {
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .current_dir(dir.path())
            .args(["eval", "--json", "--file", input])
            .output()
            .expect("eval");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).expect("JSON");
        assert_eq!(value["roots"][0]["value"]["value"]["dtype"], "int64");
    }
}

#[test]
fn surf_and_deep_migrations_are_explicit_and_value_preserving() {
    let dir = tempdir().expect("tempdir");
    let surf = dir.path().join("legacy.ch");
    let deep = dir.path().join("legacy.dp");
    fs::write(&surf, "value: int64 = 3000000000i64\n").expect("write Surf");
    fs::write(
        &deep,
        "(defsig {} value (t-prim {} int64))\n\
         (def {} value (lit {type: (t-prim {} int64)} 3000000000))\n",
    )
    .expect("write Deep");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "surf", "--from", "0.18"])
        .arg(&surf)
        .assert()
        .success()
        .stdout("value: i64 = 3000000000i64\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["migrate", "deep", "--from", "0.18"])
        .arg(&deep)
        .assert()
        .success()
        .stdout(predicate::str::contains("(t-prim {} i64)"))
        .stdout(predicate::str::contains("(t-prim {} int64)").not());

    assert_eq!(
        fs::read_to_string(&surf).unwrap(),
        "value: int64 = 3000000000i64\n"
    );
    assert!(fs::read_to_string(&deep).unwrap().contains("int64"));
}

#[test]
fn retired_names_reject_at_both_cli_ingresses() {
    let dir = tempdir().expect("tempdir");
    let surf = dir.path().join("legacy.ch");
    let deep = dir.path().join("legacy.dp");
    fs::write(&surf, "value: int64 = 1i64\n").expect("write Surf");
    fs::write(
        &deep,
        "(defsig {} value (t-prim {} int64))\n\
         (def {} value (lit {type: (t-prim {} int64)} 1))\n",
    )
    .expect("write Deep");

    for args in [
        vec!["check", surf.to_str().unwrap()],
        vec!["deep", surf.to_str().unwrap()],
        vec!["fmt", surf.to_str().unwrap()],
        vec!["fmt", "--check", surf.to_str().unwrap()],
        vec!["validate", "--surf", surf.to_str().unwrap()],
    ] {
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(&args)
            .output()
            .expect("run Surf ingress");
        assert!(!output.status.success(), "{args:?}: {}", combined(&output));
        assert!(
            combined(&output).contains("chelis migrate surf --from 0.18"),
            "{args:?}: {}",
            combined(&output)
        );
    }

    for args in [
        vec!["check", deep.to_str().unwrap()],
        vec!["surf", deep.to_str().unwrap()],
        vec!["fmt", deep.to_str().unwrap()],
        vec!["fmt", "--check", deep.to_str().unwrap()],
        vec!["validate", "--deep", deep.to_str().unwrap()],
    ] {
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(&args)
            .output()
            .expect("run Deep ingress");
        assert!(!output.status.success(), "{args:?}: {}", combined(&output));
        assert!(
            combined(&output).contains("chelis migrate deep --from 0.18"),
            "{args:?}: {}",
            combined(&output)
        );
    }
}
