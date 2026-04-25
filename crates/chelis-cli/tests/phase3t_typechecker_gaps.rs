//! Phase 3t.A1 follow-up — captured type-checker gaps (#39).
//!
//! These tests pin the *current observed behavior* of known type-checker
//! soundness gaps so the fix is tracked by a runnable case rather than only
//! prose. They are `#[ignore]`'d so default `cargo test` stays green; flip
//! the inversion in the assertion when the underlying gap is fixed.
//!
//! ## #39 — defsig dim enforcement leaks through wildcard inference
//!
//! When a function declared with concrete tensor dims has a body whose
//! inferred type contains `Dim::Wildcard` (e.g. `pad_sequences_to`,
//! `to_tensor` produce wildcard dims), the unify between the body type
//! and the declared signature succeeds because `unify_dim` treats Wildcard
//! as a matches-anything sentinel. After unification, `infer_top_level`
//! generalizes `body_ty` (which still holds the wildcards) instead of the
//! declared concrete type, so callers that pass the result to a function
//! expecting different concrete dims are silently accepted.
//!
//! Pure-sig case (no body) DOES catch the mismatch — see the second test.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn package_std() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .canonicalize()
        .expect("chelis-std package must exist")
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dir");
    for entry in fs::read_dir(src).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

fn make_app(dir_name: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join(dir_name);
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();
    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "{dir_name}"
version = "0.1.0"
compiler = "=0.2.4"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

#[test]
fn pure_sig_dim_mismatch_is_caught() {
    // Sanity check: when the value being passed is itself a sig (no body),
    // the dim mismatch IS caught. Pin this so a regression doesn't silently
    // weaken the working case alongside a fix for the broken one below.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-tc-pure-sig");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

sig do_thing: tensor[32, 128, f32] -> f32
sig make_3: tensor[1, 3, f32]

result = do_thing(make_3)
"#,
    );
    // `chelis check` exits 0 even with errors in the JSON; assert on the
    // JSON content rather than process exit status.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .stdout(predicate::str::contains("DimensionMismatch"));
}

#[test]
#[ignore = "captured #39 gap: defsig dim enforcement leaks through wildcard inference. Bug, not feature. The make() body inferred type carries Dim::Wildcard from pad_sequences_to; unify(Wildcard, Lit(N)) succeeds; generalize stores wildcards; callers see polymorphic dims. Naive fix (use declared_ty for the scheme) breaks 5 workspace tests via linearity. Re-enable + invert assertion once the ascription-narrowing fix lands."]
fn defsig_dim_enforcement_leaks_through_wildcard_body_in_callers() {
    // The bug: `make` is declared to return `tensor[1, 3, f32]`, but its
    // body inferred type is `tensor[*, *, f32]` (Wildcard dims from
    // pad_sequences_to). `do_thing` expects `tensor[32, 128, f32]`. The
    // call should fail with `DimensionMismatch: Lit(32) vs Lit(1)` but
    // instead `chelis check` reports score 1 with no errors.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-tc-defsig-leak");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

sig do_thing: tensor[32, 128, f32] -> f32

def make() -> tensor[1, 3, f32] =
  pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]], cast(3, int64), cast(0.0, f32))

result = do_thing(make())
"#,
    );
    // When the bug is fixed, this assertion holds and the test passes.
    // While the bug stands, `check` returns score 1 with no errors, so the
    // expected `failure()` here would not match.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("DimensionMismatch"));
}
