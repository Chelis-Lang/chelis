use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

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
