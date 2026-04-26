use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn package_std() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .canonicalize()
        .expect("path should exist")
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

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn make_app(dir_name: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join(dir_name);
    copy_dir_recursive(&package_std(), &std_pkg);
    let _ = fs::remove_dir_all(std_pkg.join("dist"));
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
compiler = "=0.2.6"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

#[test]
fn coral_prerequisites() {
    let (_dir, reef_home, app_pkg) = make_app("coral-prereqs");

    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

type Column =
  | FloatCol(tensor[4, f32])
  | IntCol(tensor[4, int64])

def get_float(c: Column) -> tensor[4, f32] = match c with {
  | FloatCol(t) => t
  | IntCol(_) => (to_tensor([0.0, 0.0, 0.0, 0.0]) : tensor[4, f32])
}

value = get_float(FloatCol((to_tensor([1.0, 2.0, 3.0, 4.0]) : tensor[4, f32])))
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));

    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

def make_dict() -> Dict[string, tensor[4, f32]] = {
  d = dict_of([] : List[(string, tensor[4, f32])])
  dict_insert(d, "price", (to_tensor([1.0, 1.0, 1.0, 1.0]) : tensor[4, f32]))
}

value = make_dict()
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));

    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

def filter_bools(mask: tensor[4, bool], indices: tensor[2, int64]) -> tensor[2, bool] =
  gather(mask, indices, 0)
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            app_pkg.join("out-gather-bool").to_str().unwrap(),
        ])
        .assert()
        .success();

    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Construct (arange)

ar_0_4 = arange(cast(0, int32), cast(4, int32))
ar_2_6 = arange(cast(2, int32), cast(6, int32))
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "ar_0_4 = tensor(shape=[4], data=[0.0, 1.0, 2.0, 3.0])",
        ))
        .stdout(predicate::str::contains(
            "ar_2_6 = tensor(shape=[4], data=[2.0, 3.0, 4.0, 5.0])",
        ));
}

#[test]
fn coral_where_indices_builds_and_matches_reference_values() {
    let (_dir, reef_home, app_pkg) = make_app("coral-where-indices");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Mask (where_indices)

mixed_mask = cmplt((to_tensor([0.0, 1.0, 0.0, 1.0, 1.0]) : tensor[5, f32]), (to_tensor([0.5, 0.5, 0.5, 0.5, 0.5]) : tensor[5, f32]))
all_true = cmplt((to_tensor([0.0, 0.0, 0.0]) : tensor[3, f32]), (to_tensor([1.0, 1.0, 1.0]) : tensor[3, f32]))

mixed_idx = where_indices(mixed_mask)
all_true_idx = where_indices(all_true)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "mixed_idx = tensor(shape=[2], data=[0.0, 2.0])",
        ))
        .stdout(predicate::str::contains(
            "all_true_idx = tensor(shape=[3], data=[0.0, 1.0, 2.0])",
        ));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            app_pkg.join("out").to_str().unwrap(),
        ])
        .assert()
        .success();
}

#[test]
fn coral_where_indices_all_false_returns_empty_tensor() {
    // Previously this panicked the evaluator because `numel` floored zero-
    // length tensors to 1, tripping the length assertion in `from_vec`.
    // After the 3t cleanup fix, the evaluator honors the zero dimension and
    // the empty mask path produces a legitimate `tensor[0, int64]`.
    let (_dir, reef_home, app_pkg) = make_app("coral-where-indices-empty");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Mask (where_indices)

all_false = cmplt((to_tensor([1.0, 1.0, 1.0]) : tensor[3, f32]), (to_tensor([0.0, 0.0, 0.0]) : tensor[3, f32]))
value = where_indices(all_false)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("shape=[0]"))
        .stdout(predicate::str::contains("data=[]"));
}
