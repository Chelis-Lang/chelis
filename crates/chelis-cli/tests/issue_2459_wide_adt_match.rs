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

fn flat_dispatch_profile(source: &str, arms: usize) -> Option<(usize, usize)> {
    let tests: Vec<_> = source
        .lines()
        .filter_map(|line| {
            let (_, tail) = line.split_once("chelis_host_string_eq_cstr(")?;
            let (tag, label) = tail.split_once(", ")?;
            let label = label.strip_suffix(")) {")?.trim_matches('"');
            label
                .starts_with("Op")
                .then_some((line.len() - line.trim_start().len(), tag, label))
        })
        .collect();
    if tests.len() != arms {
        return None;
    }
    let (indent, tag, _) = tests.first()?;
    if !tests
        .iter()
        .enumerate()
        .all(|(index, (depth, actual, label))| {
            depth == indent && actual == tag && *label == format!("Op{index}")
        })
    {
        return None;
    }
    let mut depth = 0usize;
    let mut maximum = 0;
    for byte in source.bytes() {
        if byte == b'{' {
            depth += 1;
            maximum = maximum.max(depth);
        }
        if byte == b'}' {
            depth = depth.checked_sub(1)?;
        }
    }
    (depth == 0 && maximum <= 8).then_some((source.len(), maximum))
}

fn bounded_linear_increments(samples: &[usize; 3]) -> bool {
    let Some(first) = samples[1].checked_sub(samples[0]) else {
        return false;
    };
    let Some(second) = samples[2].checked_sub(samples[1]) else {
        return false;
    };
    first > 0 && second > 0 && second * 2 <= first * 5
}

#[test]
fn flat_dispatch_oracle_rejects_nested_missing_and_quadratic_emission() {
    let flat = "if (chelis_host_string_eq_cstr(tag, \"Op0\")) {\n}\nelse if (chelis_host_string_eq_cstr(tag, \"Op1\")) {\n}\n";
    assert!(flat_dispatch_profile(flat, 2).is_some());
    assert!(flat_dispatch_profile(&flat.replace("else if", "  else if"), 2).is_none());
    let nested = flat.replace("}\nelse if", "  if") + "}\n";
    assert!(flat_dispatch_profile(&nested, 2).is_none());
    assert!(flat_dispatch_profile(&flat.replace("Op1", "Op0"), 2).is_none());
    assert!(flat_dispatch_profile(&flat.replace("tag, \"Op1\"", "wrong, \"Op1\""), 2).is_none());
    assert!(bounded_linear_increments(&[400, 800, 1600]));
    assert!(!bounded_linear_increments(&[1600, 6400, 25600]));
}

fn assert_wide_match_builds_flat_c(payload: bool) {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("wide.ch");
    let c_file = dir.path().join("wide.c");
    let mut profiles = Vec::new();
    for arms in [40, 80, 160] {
        fs::write(&file, wide_match_source(arms, payload)).expect("write Surf source");
        let output = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                file.to_str().unwrap(),
                "--target",
                "c",
                "--emit-c",
                "--output",
                c_file.to_str().unwrap(),
            ])
            .output()
            .expect("run build");
        assert!(
            output.status.success(),
            "build failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let source = fs::read_to_string(&c_file).expect("emitted C");
        let profile = flat_dispatch_profile(&source, arms)
            .expect("complete sibling dispatch with bounded nesting");
        eprintln!(
            "payload={payload} arms={arms}: {} C bytes, nesting {}",
            profile.0, profile.1
        );
        profiles.push(profile);
    }
    assert!(
        profiles.iter().all(|profile| profile.1 == profiles[0].1),
        "nesting grows with arm count: {profiles:?}"
    );
    let bytes = [profiles[0].0, profiles[1].0, profiles[2].0];
    assert!(
        bounded_linear_increments(&bytes),
        "superlinear incremental C growth: {bytes:?}"
    );
    // The largest dispatch must compile and execute both extreme arms.
    assert!(common::link_generated(dir.path(), "wide.c", "wide").success());
    let run = std::process::Command::new(dir.path().join("wide"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{run:?}");
    let output = String::from_utf8(run.stdout).unwrap();
    assert!(
        output.contains(if payload { "a = 160" } else { "a = 159" }),
        "{output}"
    );
    assert!(
        output.contains(if payload { "b = 1" } else { "b = 0" }),
        "{output}"
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
