use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

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
            "def apply(x: i32) -> i32 = bump(x)\n",
        ),
    )
    .expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf"])
        .arg(&source)
        .assert()
        .success()
        .stdout(predicate::str::contains("def apply(x: i32) -> i32"))
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
            "def apply(x: i32) -> i32 = loop(x)\n",
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

fn assert_macro_roundtrip(source: &str) {
    let directory = tempdir().expect("tempdir");
    let authored = directory.path().join("authored.ch");
    fs::write(&authored, source).expect("write fixture");

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
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&original)
                .expect("valid metadata for round-trip normalization"),
        ),
        chelis_deep::printer::print_canonical(
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&roundtrip)
                .expect("valid metadata for round-trip normalization"),
        )
    );
}

#[test]
fn macro_program_round_trips_after_expansion_modulo_derived_metadata() {
    assert_macro_roundtrip(concat!(
        "macro bump(x) = add(x, 1)\n",
        "def apply(x: i32) -> i32 = bump(x)\n",
    ));
}

#[test]
fn a_hygienized_typed_vocabulary_binder_round_trips_after_expansion() {
    assert_macro_roundtrip(concat!(
        "macro add_one(v) = (fn (record: f32) -> add(record, v))(1.0f32)\n",
        "def run(record: f32) -> f32 = {\n",
        "  result: f32 = add_one(record)\n",
        "  result\n",
        "}\n",
        "out = run(10.0f32)\n",
    ));
}

#[test]
fn checked_deep_resugars_without_losing_expression_types() {
    let directory = tempdir().expect("tempdir");
    let authored = directory.path().join("typed.ch");
    fs::write(&authored, "def identity(x: i32) -> i32 = x\n").expect("write fixture");

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
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&checked)
                .expect("valid metadata for round-trip normalization"),
        ),
        chelis_deep::printer::print_canonical(
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&roundtrip)
                .expect("valid metadata for round-trip normalization"),
        )
    );
}

#[test]
fn surf_command_preserves_checked_standalone_binding_types_and_unit() {
    let directory = tempdir().expect("tempdir");
    let authored = directory.path().join("checked_bindings.ch");
    fs::write(&authored, "answer = 42\nunit_value = ()\n").expect("write fixture");

    let checked_deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", "--annotate"])
        .arg(&authored)
        .output()
        .expect("run checked desugar");
    assert!(checked_deep.status.success());
    let deep_path = directory.path().join("checked.dp");
    fs::write(&deep_path, &checked_deep.stdout).expect("write checked Deep");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf"])
        .arg(&deep_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("answer: i32 = 42"))
        .stdout(predicate::str::contains("unit_value: unit = ()"));
}

#[test]
fn surf_command_recovers_discard_and_tuple_patterns_after_expansion() {
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("patterns.ch");
    fs::write(
        &source,
        concat!(
            "def recover(pair) = {\n",
            "  _ = pair\n",
            "  (left, right) = pair\n",
            "  left\n",
            "}\n",
        ),
    )
    .expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf"])
        .arg(&source)
        .assert()
        .success()
        .stdout(predicate::str::contains("  _ = pair"))
        .stdout(predicate::str::contains("  (left, right) = pair"))
        .stdout(predicate::str::contains("__chelis_tmp").not());
}

#[test]
fn signed_minimum_double_negation_traps_without_aborting_the_compiler() {
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("double_min.ch");
    fs::write(&source, "value = -(-9223372036854775808i64)\n").expect("write fixture");

    let checked = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", "--allow-style-violations"])
        .arg(&source)
        .output()
        .expect("run check");
    assert_ne!(
        checked.status.code(),
        Some(101),
        "check must never abort: {}",
        String::from_utf8_lossy(&checked.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&checked.stderr).contains("panicked at"),
        "check leaked a panic: {}",
        String::from_utf8_lossy(&checked.stderr)
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--allow-style-violations", "--file"])
        .arg(&source)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "numeric trap: overflow in neg at i64",
        ));
}

#[test]
fn repository_surf_corpus_obeys_the_normalized_deep_retraction_law() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("CLI crate lives two levels below the workspace");
    let paths = tracked_surf_files(workspace);
    assert!(!paths.is_empty(), "Surf corpus must not be empty");

    let directory = tempdir().expect("tempdir");
    let resugared_path = directory.path().join("resugared.ch");
    for path in paths {
        assert_retraction_law(&path, &resugared_path);
    }
}

/// Desugar `path`, resugar it, check the resugared text is canonical, and
/// require the two Deep forms to agree after round-trip normalization. Every
/// printed Deep form must also parse back.
fn assert_retraction_law(path: &Path, resugared_path: &Path) {
    let original_deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep"])
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("desugar {}: {error}", path.display()));
    assert!(
        original_deep.status.success(),
        "initial desugar failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&original_deep.stderr)
    );

    let resugared = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf"])
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("resugar {}: {error}", path.display()));
    assert!(
        resugared.status.success(),
        "resugar failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&resugared.stderr)
    );
    fs::write(resugared_path, &resugared.stdout).expect("write resugared fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--check"])
        .arg(resugared_path)
        .assert()
        .success();

    let roundtrip_deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep"])
        .arg(resugared_path)
        .output()
        .expect("desugar resugared source");
    assert!(
        roundtrip_deep.status.success(),
        "roundtrip desugar failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&roundtrip_deep.stderr)
    );

    let original = chelis_deep::parser::parse_str(
        std::str::from_utf8(&original_deep.stdout).expect("original Deep is UTF-8"),
    )
    .expect("original Deep parses");
    let roundtrip = chelis_deep::parser::parse_str(
        std::str::from_utf8(&roundtrip_deep.stdout).expect("roundtrip Deep is UTF-8"),
    )
    .expect("roundtrip Deep parses");
    assert_eq!(
        chelis_deep::printer::print_canonical(
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&original)
                .expect("valid metadata for round-trip normalization"),
        ),
        chelis_deep::printer::print_canonical(
            &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&roundtrip)
                .expect("valid metadata for round-trip normalization"),
        ),
        "normalized Deep retraction law failed for {}",
        path.display()
    );
}

/// A program nested deeply enough that an authored lambda's empty `params`
/// metadata starts past the Deep printer's line width still obeys the
/// retraction law, independently of the repository corpus's shapes.
#[test]
fn deeply_nested_surf_obeys_the_normalized_deep_retraction_law() {
    let depth = 45;
    let mut body = String::from("x |> (fn (v: i64) -> v + 1i64)");
    for level in 0..depth {
        body = format!("if gt(x, {level}i64) then ({body}) else 0i64");
    }
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("deeply_nested.ch");
    fs::write(&source, format!("def nested(x: i64) -> i64 = {body}\n"))
        .expect("write the nested program");
    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep"])
        .arg(&source)
        .output()
        .expect("desugar the nested program");
    assert!(deep.status.success(), "desugar failed");
    let printed = String::from_utf8(deep.stdout).expect("Deep is UTF-8");
    assert!(
        printed.lines().any(|line| {
            let column = line.len() - line.trim_start().len();
            line.trim_start().starts_with("(params") && column + "(params {}".len() > 80
        }),
        "the fixture must nest a `params` node deeply enough that its empty metadata does not fit on the line:\n{printed}"
    );
    assert_retraction_law(&source, &directory.path().join("resugared.ch"));
}

#[test]
fn repository_surf_corpus_ignores_untracked_files() {
    let repository = tempdir().expect("temporary git repository");
    let tracked = repository.path().join("tracked.ch");
    let untracked = repository.path().join("scratch.ch");
    fs::write(&tracked, "tracked = 1\n").expect("write tracked Surf");
    fs::write(&untracked, "scratch = 2\n").expect("write untracked Surf");
    for arguments in [["init", "-q"].as_slice(), ["add", "tracked.ch"].as_slice()] {
        let status = ProcessCommand::new("git")
            .args(arguments)
            .current_dir(repository.path())
            .status()
            .expect("run git fixture command");
        assert!(status.success(), "git {arguments:?} failed");
    }

    assert_eq!(tracked_surf_files(repository.path()), vec![tracked]);
}

fn tracked_surf_files(workspace: &Path) -> Vec<PathBuf> {
    let output = ProcessCommand::new("git")
        .args(["ls-files", "-z", "--", "*.ch"])
        .current_dir(workspace)
        .output()
        .expect("run git ls-files for the Surf corpus");
    assert!(
        output.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let listed = std::str::from_utf8(&output.stdout).expect("git paths are UTF-8");
    listed
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(|path| workspace.join(path))
        .collect()
}

#[test]
fn migration_preserves_empty_property_preconditions() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("property.ch");
    for text in [
        "@property p forall(): true\n",
        "@property p forall(x: i32): x == x\n",
    ] {
        fs::write(&source, text).unwrap();
        Command::cargo_bin("chelis")
            .unwrap()
            .args(["migrate", "surf", "--from", "0.18", "--inplace"])
            .arg(&source)
            .assert()
            .success();
        Command::cargo_bin("chelis")
            .unwrap()
            .args(["migrate", "surf", "--from", "0.18", "--check"])
            .arg(&source)
            .assert()
            .success();
    }
    fs::write(&source, "@property p forall(x: i32):\n").unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["migrate", "surf", "--from", "0.18", "--check"])
        .arg(&source)
        .assert()
        .failure()
        .stderr(predicate::str::contains("panicked").not());
}
