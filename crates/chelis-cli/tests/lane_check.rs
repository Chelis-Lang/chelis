use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn gate(path: &Path) -> (i32, Vec<Value>) {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lane-check", path.to_str().expect("UTF-8 path"), "--json"])
        .output()
        .expect("run gate");
    let records = String::from_utf8(output.stdout)
        .expect("NDJSON is UTF-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("typed NDJSON record"))
        .collect();
    (output.status.code().expect("normal exit"), records)
}

#[test]
fn exact_outputs_are_compared_in_sorted_order_with_stable_reports() {
    let dir = tempdir().unwrap();
    fs::create_dir(dir.path().join("nested")).unwrap();
    fs::write(dir.path().join("z.ch"), "big = 9007199254740993i64\n").unwrap();
    fs::write(
        dir.path().join("nested/a.ch"),
        "flag = true\nflags = to_tensor([true, false])\n",
    )
    .unwrap();

    let (first_code, first) = gate(dir.path());
    let (second_code, second) = gate(dir.path());
    assert_eq!(first_code, 0, "{first:?}");
    assert_eq!(second_code, 0, "{second:?}");
    assert_eq!(first, second, "report must have no scratch paths or timing");
    assert_eq!(first.len(), 3);
    assert_eq!(first[0]["file"], "nested/a.ch");
    assert_eq!(first[1]["file"], "z.ch");
    assert_eq!(first[0]["status"], "pass");
    assert_eq!(first[1]["status"], "pass");
    assert_eq!(first[2]["status"], "pass");
    assert_eq!(first[2]["compared"], 2);
    assert_eq!(first[2]["proof_scope"]["kind"], "unpinned-host");
}

#[test]
fn empty_and_library_only_corpora_never_claim_agreement() {
    let dir = tempdir().unwrap();
    let (empty_code, empty) = gate(dir.path());
    assert_eq!(empty_code, 2);
    assert_eq!(empty.len(), 1);
    assert_eq!(empty[0]["status"], "error");
    assert_eq!(empty[0]["stage"], "discovery");

    let program = dir.path().join("library.ch");
    fs::write(&program, "def twice(x: i64) -> i64 = (x + 1i64)\n").unwrap();
    let (library_code, library) = gate(&program);
    assert_eq!(library_code, 2, "{library:?}");
    assert_eq!(library[0]["stage"], "discovery");
    assert!(
        library[0]["diagnostic"]
            .as_str()
            .unwrap()
            .contains("observable")
    );
    assert_eq!(library[1]["status"], "error");
}

#[test]
fn evaluation_failure_is_a_typed_error_not_a_skip() {
    let dir = tempdir().unwrap();
    let program = dir.path().join("bad.ch");
    fs::write(&program, "main = does_not_exist\n").unwrap();
    let (code, records) = gate(&program);
    assert_eq!(code, 2, "{records:?}");
    assert_eq!(records[0]["status"], "error");
    assert_eq!(records[0]["stage"], "eval");
    assert!(records[0]["process"]["status"].as_i64().unwrap() != 0);
    assert_eq!(records[1]["status"], "error");
}

#[test]
fn c_only_refusal_is_a_build_error_not_an_inferred_skip() {
    let dir = tempdir().unwrap();
    let program = dir.path().join("eval_only.ch");
    fs::write(&program, "rounded = round_to(1.234f64, cast(2, i32))\n").unwrap();
    let (code, records) = gate(&program);
    assert_eq!(code, 2, "{records:?}");
    assert_eq!(records[0]["stage"], "build", "{records:?}");
    assert_eq!(records[0]["status"], "error");
    assert_ne!(records[0]["process"]["status"], 0);
}

#[test]
fn unavailable_native_compiler_is_a_link_error_with_diagnostics() {
    let dir = tempdir().unwrap();
    let program = dir.path().join("value.ch");
    fs::write(&program, "value = 9007199254740993i64\n").unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_CC", dir.path().join("missing-compiler"))
        .args(["lane-check", program.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let record: Value =
        serde_json::from_slice(output.stdout.split(|byte| *byte == b'\n').next().unwrap()).unwrap();
    assert_eq!(record["stage"], "link", "{record:?}");
    assert!(
        record["process"]["stderr"]
            .as_str()
            .unwrap()
            .contains("compiler")
    );
}

#[test]
fn timed_out_evaluation_reports_lane_and_deadline() {
    let dir = tempdir().unwrap();
    let program = dir.path().join("slow.ch");
    fs::write(&program, "result = process_run(\"/bin/sleep\", [\"5\"])\n").unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args([
            "lane-check",
            program.to_str().unwrap(),
            "--json",
            "--timeout",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let record: Value =
        serde_json::from_slice(output.stdout.split(|byte| *byte == b'\n').next().unwrap()).unwrap();
    assert_eq!(record["stage"], "eval", "{record:?}");
    assert_eq!(record["process"]["timed_out"], true);
    assert_eq!(record["process"]["deadline_seconds"], 1);
}
