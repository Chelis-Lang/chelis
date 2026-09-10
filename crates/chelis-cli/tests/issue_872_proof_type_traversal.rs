//! LU2 public-surface oracle: ordinary Surf must not lose wrapped producers
//! across either former proof-type recursion limit. No SMT solver is needed.

use assert_cmd::Command;
use serde_json::Value;

fn module(aliases: usize, records: usize, opaque: bool) -> String {
    let mut source = String::from(
        "module M\nexport (make_w)\n@opaque\n@invariant(p) p.value >= 0.0 && p.value <= 1.0\ntype T = | T { value: f32 }\n",
    );
    let mut ty = if opaque { "T" } else { "f32" }.to_string();
    let mut value = if opaque { "T { value: 99.0 }" } else { "99.0" }.to_string();
    for i in 0..aliases {
        source.push_str(&format!("type A{i} = {ty}\n"));
        ty = format!("A{i}");
    }
    for i in 0..records {
        source.push_str(&format!("type W{i} = | W{i} {{ inner: {ty} }}\n"));
        ty = format!("W{i}");
        value = format!("W{i} {{ inner: {value} }}");
    }
    source.push_str(&format!("def make_w(x: f32) -> {ty} = {value}\n"));
    source
}

fn assert_matrix(cases: &[(usize, usize)]) {
    let dir = tempfile::tempdir().unwrap();
    for &(aliases, records) in cases {
        for opaque in [true, false] {
            let path = dir.path().join("module.ch");
            std::fs::write(&path, module(aliases, records, opaque)).unwrap();
            let output = Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["prove", path.to_str().unwrap(), "--json"])
                .output()
                .unwrap();
            let stdout = String::from_utf8(output.stdout).unwrap();
            let rows: Vec<Value> = stdout
                .lines()
                .map(|line| serde_json::from_str(line).expect("JSON record"))
                .collect();
            let obligations: Vec<_> = rows
                .iter()
                .filter(|row| row["kind"] == "obligation")
                .collect();
            assert_eq!(
                output.status.code(),
                Some(if opaque { 3 } else { 0 }),
                "aliases={aliases}, records={records}, opaque={opaque}: {rows:?}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(obligations.len(), usize::from(opaque), "{rows:?}");
            if opaque {
                assert_eq!(obligations[0]["status"], "error", "{rows:?}");
                assert!(
                    obligations[0]["reason"]
                        .as_str()
                        .unwrap()
                        .contains("make_w"),
                    "{rows:?}"
                );
            }
        }
    }
}

#[test]
fn alias_threshold_keeps_the_complete_obligation_set() {
    assert_matrix(&[(31, 1), (32, 1), (33, 1), (40, 1)]);
}

#[test]
fn record_threshold_keeps_the_complete_obligation_set() {
    assert_matrix(&[(0, 15), (0, 16), (0, 17), (0, 24)]);
}
