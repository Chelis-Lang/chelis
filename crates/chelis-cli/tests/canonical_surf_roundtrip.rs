use std::fs;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn surf_command_resugars_the_expanded_macro_program() {
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("macro.ch");
    fs::write(
        &source,
        concat!(
            "macro bump(x) = add(x, 1)\n",
            "def apply(x: int32) -> int32 = bump(x)\n",
        ),
    )
    .expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf"])
        .arg(&source)
        .assert()
        .success()
        .stdout(predicate::str::contains("def apply(x: int32) -> int32"))
        .stdout(predicate::str::contains("macro").not());
}

#[test]
fn surf_command_reports_macro_expansion_failure() {
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("recursive.ch");
    fs::write(
        &source,
        concat!(
            "macro loop(x) = loop(x)\n",
            "def apply(x: int32) -> int32 = loop(x)\n",
        ),
    )
    .expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf"])
        .arg(&source)
        .assert()
        .failure()
        .stderr(predicate::str::contains("macro expansion limit exceeded"));
}

#[test]
fn macro_program_round_trips_after_expansion_modulo_derived_metadata() {
    let directory = tempdir().expect("tempdir");
    let authored = directory.path().join("authored.ch");
    fs::write(
        &authored,
        concat!(
            "macro bump(x) = add(x, 1)\n",
            "def apply(x: int32) -> int32 = bump(x)\n",
        ),
    )
    .expect("write fixture");

    let original_deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep"])
        .arg(&authored)
        .output()
        .expect("run first desugar");
    assert!(original_deep.status.success());
    let deep_path = directory.path().join("expanded.dp");
    fs::write(&deep_path, &original_deep.stdout).expect("write expanded Deep");

    let resugared = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf"])
        .arg(&deep_path)
        .output()
        .expect("run resugar");
    assert!(resugared.status.success());
    let resugared_path = directory.path().join("resugared.ch");
    fs::write(&resugared_path, &resugared.stdout).expect("write resugared Surf");

    let roundtrip_deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep"])
        .arg(&resugared_path)
        .output()
        .expect("run second desugar");
    assert!(roundtrip_deep.status.success());

    let original = chelis_deep::parser::parse_str(
        std::str::from_utf8(&original_deep.stdout).expect("Deep output is UTF-8"),
    )
    .expect("first Deep output parses");
    let roundtrip = chelis_deep::parser::parse_str(
        std::str::from_utf8(&roundtrip_deep.stdout).expect("Deep output is UTF-8"),
    )
    .expect("roundtrip Deep output parses");
    assert_eq!(
        chelis_deep::printer::print_canonical(
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&original),
        ),
        chelis_deep::printer::print_canonical(
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&roundtrip),
        )
    );
}

#[test]
fn checked_deep_resugars_without_losing_expression_types() {
    let directory = tempdir().expect("tempdir");
    let authored = directory.path().join("typed.ch");
    fs::write(&authored, "def identity(x: int32) -> int32 = x\n").expect("write fixture");

    let checked_deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", "--annotate"])
        .arg(&authored)
        .output()
        .expect("run checked desugar");
    assert!(
        checked_deep.status.success(),
        "checked Deep failed: {}",
        String::from_utf8_lossy(&checked_deep.stderr)
    );
    let deep_path = directory.path().join("checked.dp");
    fs::write(&deep_path, &checked_deep.stdout).expect("write checked Deep");

    let resugared = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf"])
        .arg(&deep_path)
        .output()
        .expect("resugar checked Deep");
    assert!(
        resugared.status.success(),
        "checked Deep resugaring failed: {}",
        String::from_utf8_lossy(&resugared.stderr)
    );
    let resugared_path = directory.path().join("resugared.ch");
    fs::write(&resugared_path, &resugared.stdout).expect("write resugared Surf");

    let roundtrip_deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep"])
        .arg(&resugared_path)
        .output()
        .expect("desugar resugared checked Surf");
    assert!(roundtrip_deep.status.success());

    let checked = chelis_deep::parser::parse_str(
        std::str::from_utf8(&checked_deep.stdout).expect("checked Deep is UTF-8"),
    )
    .expect("checked Deep parses");
    let roundtrip = chelis_deep::parser::parse_str(
        std::str::from_utf8(&roundtrip_deep.stdout).expect("roundtrip Deep is UTF-8"),
    )
    .expect("roundtrip Deep parses");
    assert_eq!(
        chelis_deep::printer::print_canonical(
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&checked),
        ),
        chelis_deep::printer::print_canonical(
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&roundtrip),
        )
    );
}
