//! Wide, guard-free constructor matches should lower as one flat dispatch.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

mod common;
use common::build_and_run;

fn wide_match_source(arms: usize, payload: bool) -> String {
    let mut source = String::from("type Op =\n");
    for index in 0..arms {
        if payload {
            source.push_str(&format!("  | Op{index}(i64)\n"));
        } else {
            source.push_str(&format!("  | Op{index}\n"));
        }
    }
    source.push_str("def code(o: Op) -> i64 =\n  match o with {\n");
    for index in 0..arms {
        if payload {
            source.push_str(&format!("    | Op{index}(v) => add(v, {index}i64)\n"));
        } else {
            source.push_str(&format!("    | Op{index} => {index}i64\n"));
        }
    }
    source.push_str("  }\n");
    if payload {
        source.push_str(&format!(
            "a = code(Op{}(1i64))\nb = code(Op0(1i64))\n",
            arms - 1
        ));
    } else {
        source.push_str(&format!("a = code(Op{})\nb = code(Op0)\n", arms - 1));
    }
    source
}

fn assert_wide_match_builds_flat_c(payload: bool) {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("wide.ch");
    let c_file = dir.path().join("wide.c");
    fs::write(&file, wide_match_source(160, payload)).expect("write Surf source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            file.to_str().expect("utf8 path"),
            "--target",
            "c",
            "--emit-c",
            "--output",
            c_file.to_str().expect("utf8 path"),
        ])
        .output()
        .expect("run build");
    assert!(
        output.status.success(),
        "build failed ({:?}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = fs::metadata(&c_file).expect("emitted C").len();
    assert!(
        bytes < 400_000,
        "160 constructor arms emitted {bytes} C bytes"
    );
}

#[test]
fn wide_nullary_constructor_match_builds_flat_c() {
    assert_wide_match_builds_flat_c(false);
}

#[test]
fn wide_payload_constructor_match_builds_flat_c() {
    assert_wide_match_builds_flat_c(true);
}

#[test]
fn flat_payload_dispatch_reads_the_selected_arm() {
    let printed = build_and_run(&wide_match_source(3, true), "flat_payload_dispatch");
    assert!(printed.contains("a = 3"), "{printed}");
    assert!(printed.contains("b = 1"), "{printed}");
}
