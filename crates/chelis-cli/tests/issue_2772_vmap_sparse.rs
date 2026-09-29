//! chelis#2772: sparse axes remain local to each row under vmap.

use assert_cmd::Command;
use std::process::Command as StdCommand;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, link_generated, make_app, write_file};

const GATHER: &str = r#"module Demo.Main

def loss(w: tensor[4, f32]) -> tensor[f32] =
  sum(gather(w, to_tensor([0i64, 2i64]), 0i32), 0i32)

out = vmap(loss)(to_tensor([
  [1.0f32, 2.0f32, 3.0f32, 4.0f32],
  [5.0f32, 6.0f32, 7.0f32, 8.0f32],
  [9.0f32, 10.0f32, 11.0f32, 12.0f32]
]))
"#;

const GRAD: &str = r#"module Demo.Main

def loss(w: tensor[4, f32]) -> f32 =
  tensor_to_scalar(sum(gather(w, to_tensor([0i64, 2i64]), 0i32), 0i32))

out = vmap(grad(loss))(to_tensor([
  [1.0f32, 2.0f32, 3.0f32, 4.0f32],
  [5.0f32, 6.0f32, 7.0f32, 8.0f32]
]))
"#;

const SCATTERS: &str = r#"module Demo.Main

def replace_row(base: tensor[4, f32], indices: tensor[2, i64], updates: tensor[2, f32]) -> tensor[4, f32] =
  scatter_replace(base, indices, updates, 0i32)

def replace_elements_row(base: tensor[4, f32], indices: tensor[2, i64], updates: tensor[2, f32]) -> tensor[4, f32] =
  scatter_elements(base, indices, updates, 0i32)

base = to_tensor([
  [1.0f32, 2.0f32, 3.0f32, 4.0f32],
  [5.0f32, 6.0f32, 7.0f32, 8.0f32]
])
indices = to_tensor([[0i64, 2i64], [0i64, 2i64]])
updates = to_tensor([[9.0f32, 7.0f32], [11.0f32, 13.0f32]])
replaced = vmap(replace_row)(base, indices, updates)
elements = vmap(replace_elements_row)(base, indices, updates)
"#;

const OUT_OF_RANGE: &str = r#"module Demo.Main

def pick(w: tensor[4, f32]) -> tensor[1, f32] =
  gather(w, to_tensor([4i64]), 0i32)

out = vmap(pick)(to_tensor([
  [1.0f32, 2.0f32, 3.0f32, 4.0f32],
  [5.0f32, 6.0f32, 7.0f32, 8.0f32],
  [9.0f32, 10.0f32, 11.0f32, 12.0f32],
  [13.0f32, 14.0f32, 15.0f32, 16.0f32],
  [17.0f32, 18.0f32, 19.0f32, 20.0f32]
]))
"#;

fn run_eval_and_c(source: &str, app_name: &str) -> (String, String) {
    let (_dir, reef_home, app_pkg) = make_app(app_name);
    write_file(&app_pkg.join("src/main.ch"), source);
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .output()
        .expect("run chelis eval");
    assert!(
        eval.status.success(),
        "eval failed: stdout={} stderr={}",
        String::from_utf8_lossy(&eval.stdout),
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval_stdout = String::from_utf8(eval.stdout).unwrap();
    let c_stdout = build_and_run_app(&reef_home, &app_pkg, "main");
    (eval_stdout, c_stdout)
}

#[test]
fn vmapped_gather_has_row_values_and_declared_rank_in_eval_and_c() {
    let (eval_stdout, c_stdout) = run_eval_and_c(GATHER, "issue-2772-vmap-gather");
    let expected = "out = tensor(shape=[3], data=[4.0, 12.0, 20.0])";
    assert!(
        eval_stdout.lines().any(|line| line == expected),
        "{eval_stdout}"
    );
    assert!(c_stdout.lines().any(|line| line == expected), "{c_stdout}");
}

#[test]
fn vmapped_gather_gradient_scatter_adds_within_each_row() {
    let (eval_stdout, c_stdout) = run_eval_and_c(GRAD, "issue-2772-vmap-grad");
    let expected = "out = tensor(shape=[2, 4], data=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])";
    assert!(
        eval_stdout.lines().any(|line| line == expected),
        "{eval_stdout}"
    );
    assert!(c_stdout.lines().any(|line| line == expected), "{c_stdout}");
}

#[test]
fn vmapped_replace_scatters_keep_writes_inside_each_row() {
    let (eval_stdout, c_stdout) = run_eval_and_c(SCATTERS, "issue-2772-vmap-scatters");
    for expected in [
        "replaced = tensor(shape=[2, 4], data=[9.0, 2.0, 7.0, 4.0, 11.0, 6.0, 13.0, 8.0])",
        "elements = tensor(shape=[2, 4], data=[9.0, 2.0, 7.0, 4.0, 11.0, 6.0, 13.0, 8.0])",
    ] {
        assert!(
            eval_stdout.lines().any(|line| line == expected),
            "{eval_stdout}"
        );
        assert!(c_stdout.lines().any(|line| line == expected), "{c_stdout}");
    }
}

#[test]
fn vmapped_gather_rejects_an_index_outside_the_row_even_when_inside_the_batch() {
    let (_dir, reef_home, app_pkg) = make_app("issue-2772-vmap-oob");
    let source = app_pkg.join("src/main.ch");
    write_file(&source, OUT_OF_RANGE);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["eval", "--file", source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!eval.status.success(), "eval accepted an invalid row index");
    assert!(
        String::from_utf8_lossy(&eval.stderr).contains("gather index 4 out of bounds"),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );

    let out_dir = app_pkg.join("c-out");
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(link_generated(&out_dir, "main.c", "main").success());
    let run = StdCommand::new(out_dir.join("main")).output().unwrap();
    assert!(
        !run.status.success(),
        "compiled C accepted an invalid row index"
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("numeric trap: domain in gather"),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
}
