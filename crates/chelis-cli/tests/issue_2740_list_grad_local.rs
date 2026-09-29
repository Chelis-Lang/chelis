//! A List gradient must never use a same-spelled top-level value for a local actual.

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, make_app, write_file};

fn assert_local_actual_refused(source: &str, expected_eval: &str, name: &str) {
    let (_dir, reef_home, app_pkg) = make_app(name);
    write_file(&app_pkg.join("src/main.ch"), source);
    let source_path = app_pkg.join("src/main.ch");

    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["eval", "--file", source_path.to_str().unwrap()])
        .assert()
        .success();
    let eval = String::from_utf8(eval.get_output().stdout.clone()).expect("utf-8 eval output");
    assert!(eval.contains(expected_eval), "{eval}");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            source_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            app_pkg.join("out").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "C List gradient cannot reconstruct the local List actual `values`",
        ));
}

#[test]
fn local_list_shadow_is_refused_before_c_emission() {
    assert_local_actual_refused(
        r#"module Demo.Main
import Std.Index (list_index)
def square_selected(xs: List[f32], index: i64) -> f32 = {
  selected = list_index(xs, index)
  mul(selected, selected)
}
values: List[f32] = [2.0f32, 3.0f32, 5.0f32]
from_local = {
  values: List[f32] = [17.0f32, 19.0f32, 23.0f32, 29.0f32]
  grad(square_selected, wrt=xs)(values, 1i64)
}
"#,
        "from_local = [0.0, 38.0, 0.0, 0.0]",
        "issue-2740-local-shadow",
    );
}

#[test]
fn computed_local_list_is_also_refused() {
    assert_local_actual_refused(
        r#"module Demo.Main
import Std.Index (list_index)
def square_selected(xs: List[f32], index: i64) -> f32 = {
  selected = list_index(xs, index)
  mul(selected, selected)
}
values: List[f32] = [2.0f32, 3.0f32]
from_local = {
  values: List[f32] = [add(17.0f32, 0.0f32), 19.0f32]
  grad(square_selected, wrt=xs)(values, 1i64)
}
"#,
        "from_local = [0.0, 38.0]",
        "issue-2740-computed-local",
    );
}

#[test]
fn top_level_list_actual_still_runs_in_c() {
    let (_dir, reef_home, app_pkg) = make_app("issue-2740-top-level-control");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main
import Std.Index (list_index)
def square_selected(xs: List[f32], index: i64) -> f32 = {
  selected = list_index(xs, index)
  mul(selected, selected)
}
values: List[f32] = [2.0f32, 3.0f32, 5.0f32]
from_top = grad(square_selected, wrt=xs)(values, 1i64)
"#,
    );
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    assert!(
        compiled.contains("from_top = [0.0, 6.0, 0.0]"),
        "{compiled}"
    );
}

#[test]
fn top_level_list_aliases_and_wrapper_still_run_in_c() {
    let (_dir, reef_home, app_pkg) = make_app("issue-2740-top-level-aliases");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main
import Std.Index (list_index)
def square_selected(xs: List[f32], index: i64) -> f32 = {
  selected = list_index(xs, index)
  mul(selected, selected)
}
values: List[f32] = [2.0f32, 3.0f32, 5.0f32]
alias: List[f32] = values
alias2: List[f32] = alias
def wrapped_values() -> List[f32] = alias2
from_alias = grad(square_selected, wrt=xs)(alias2, 1i64)
from_wrapper = grad(square_selected, wrt=xs)(wrapped_values(), 1i64)
"#,
    );
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success();
    let eval = String::from_utf8(eval.get_output().stdout.clone()).expect("utf-8 eval output");
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    for (output, lane) in [(&eval, "eval"), (&compiled, "C")] {
        assert!(
            output.contains("from_alias = [0.0, 6.0, 0.0]"),
            "{lane}: {output}"
        );
        assert!(
            output.contains("from_wrapper = [0.0, 6.0, 0.0]"),
            "{lane}: {output}"
        );
    }
}
