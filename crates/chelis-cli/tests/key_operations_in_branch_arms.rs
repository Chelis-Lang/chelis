//! chelis#2413 B3: key operations inside a where-lowered branch arm carry the
//! arm's activation, like the draws beside them, through `grad` and `vmap`,
//! in `chelis eval` and in compiled C.
//!
//! Every expected bit pattern comes from `key_ref.py` and `slice2_ref.py`
//! (independent transcriptions of [05-RNG-2] and [05-OP-8]) through
//! `briefs/keys-b-b3-probes/b3_ref.py`, never from either lane.

use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, link_generated, make_app, write_file};

fn eval(reef_home: &Path, app_pkg: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .output()
        .expect("chelis eval runs")
}

/// `chelis eval` and the compiled C program over `source` print the same
/// lines; returns them.
fn both_lanes(name: &str, source: &str) -> String {
    let (_dir, reef_home, app_pkg) = make_app(name);
    write_file(&app_pkg.join("src/main.ch"), source);
    let eval = eval(&reef_home, &app_pkg);
    assert!(
        eval.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval = String::from_utf8(eval.stdout).expect("utf-8 stdout");
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    assert_eq!(eval, compiled, "eval and C disagree");
    eval
}

/// The f32 bits of the printed tensor `name`.
fn bits(stdout: &str, name: &str) -> Vec<u32> {
    let prefix = format!("{name} = tensor(");
    let line = stdout
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{name}` in:\n{stdout}"));
    let data = &line[line.find("data=[").expect("data") + "data=[".len()..];
    data[..data.find(']').expect("closing bracket")]
        .split(',')
        .map(|value| value.trim().parse::<f32>().expect("f32").to_bits())
        .collect()
}

/// `grad_arm.ch`: under `grad`, `split_key(k)` in the then arm and a draw of
/// `k` in the else arm. The then arm's gradient is `uniform(a) + uniform(b)`
/// for `(a, b) = split(key(1))`, the else arm's `uniform(key(1))`. Regression
/// test: at b005bb19b both lanes reject the backward graph ("key 0 is
/// consumed twice").
#[test]
fn grad_of_a_split_arm_beside_a_draw_arm_agrees_with_the_reference() {
    let out = both_lanes(
        "b3-grad-arm",
        r#"module Demo.Main
def split_or_draw(k: key, x: tensor[2, f32], gate: i64) -> f32 =
  if eq(gate, 1i64) then {
    (a, b) = split_key(k)
    add(tensor_to_scalar(sum(mul(uniform_like(a, x, 0.0f32, 1.0f32), x), 0i32)), tensor_to_scalar(sum(mul(uniform_like(b, x, 0.0f32, 1.0f32), x), 0i32)))
  } else tensor_to_scalar(sum(mul(uniform_like(k, x, 0.0f32, 1.0f32), x), 0i32))
grad_then = grad(split_or_draw, wrt=x)(key_from_seed(1i64), to_tensor([1.0f32, 2.0f32]), 1i64)
grad_else = grad(split_or_draw, wrt=x)(key_from_seed(1i64), to_tensor([1.0f32, 2.0f32]), 0i64)
"#,
    );
    assert_eq!(bits(&out, "grad_then"), [0x3f73_6d46, 0x3fbd_b202]);
    assert_eq!(bits(&out, "grad_else"), [0x3e35_1a96, 0x3f15_def3]);
}

/// `vmap_arm.ch`: under `vmap`, row 0 takes the else arm with
/// `split_n(key(1), 2)[0]`; row 1 takes the then arm, adding the draws of the
/// two halves of `split_n(key(1), 2)[1]`. Regression test: at b005bb19b the C
/// build rejects the batched graph ("key 3 is consumed twice").
#[test]
fn vmap_of_a_split_arm_beside_a_draw_arm_agrees_with_the_reference() {
    let out = both_lanes(
        "b3-vmap-arm",
        r#"module Demo.Main
def split_if_large(k: key, x: tensor[2, f32]) -> tensor[2, f32] =
  if gt(tensor_to_scalar(sum(copy(x), 0i32)), 3.5f32) then {
    (a, b) = split_key(k)
    add(uniform_like(a, x, 0.0f32, 1.0f32), uniform_like(b, x, 0.0f32, 1.0f32))
  } else uniform_like(k, x, 0.0f32, 1.0f32)
vmapped = vmap(split_if_large)(split_keys(key_from_seed(1i64), 2i64), to_tensor([[1.0f32, 2.0f32], [2.0f32, 2.0f32]]))
"#,
    );
    assert_eq!(
        bits(&out, "vmapped"),
        [0x3df7_8d2f, 0x3f21_f02c, 0x3f0b_21a9, 0x3f4f_9a98]
    );
}

const COUNT_ARM: &str = r#"module Demo.Main
def count_arm(k: key, n: i64, gate: i64, x: tensor[2, f32]) -> f32 =
  if eq(gate, 1i64) then {
    ks = split_keys(k, n)
    tensor_to_scalar(sum(mul(copy(x), x), 0i32))
  } else tensor_to_scalar(sum(copy(x), 0i32))
"#;

/// An unselected arm's `split_keys` reads no count: a negative or impossible
/// count does not trap, in either lane, and a selected one splits. Regression
/// test: at b005bb19b the unselected split traps `domain` (-3) and `overflow`
/// (`i64::MAX`) in both lanes.
#[test]
fn an_unselected_split_keys_does_not_trap_on_its_count() {
    let out = both_lanes(
        "b3-count-arm",
        &format!(
            "{COUNT_ARM}{}",
            r#"unselected_negative_count = grad(count_arm, wrt=x)(key_from_seed(1i64), -3i64, 0i64, to_tensor([1.0f32, 2.0f32]))
unselected_huge_count = grad(count_arm, wrt=x)(key_from_seed(1i64), 9223372036854775807i64, 0i64, to_tensor([1.0f32, 2.0f32]))
selected_count = grad(count_arm, wrt=x)(key_from_seed(1i64), 2i64, 1i64, to_tensor([1.0f32, 2.0f32]))
"#
        ),
    );
    let one = 0x3f80_0000;
    assert_eq!(bits(&out, "unselected_negative_count"), [one, one]);
    assert_eq!(bits(&out, "unselected_huge_count"), [one, one]);
    assert_eq!(bits(&out, "selected_count"), [0x4000_0000, 0x4080_0000]);
}

/// Negative parity: the selected arm's negative count still traps `domain`
/// in both lanes. Disposition lock: it traps at b005bb19b too.
#[test]
fn a_selected_split_keys_still_traps_on_a_negative_count() {
    let trap = "numeric trap: domain in split_keys at i64";
    let (_dir, reef_home, app_pkg) = make_app("b3-count-trap");
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            "{COUNT_ARM}{}",
            "selected_negative_count = grad(count_arm, wrt=x)(key_from_seed(1i64), -3i64, 1i64, to_tensor([1.0f32, 2.0f32]))\n"
        ),
    );
    let eval = eval(&reef_home, &app_pkg);
    assert!(!eval.status.success());
    assert!(
        String::from_utf8_lossy(&eval.stderr).contains(trap),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let out_dir = app_pkg.join("main-out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(link_generated(&out_dir, "main.c", "main").success());
    let run = StdCommand::new(out_dir.join("main"))
        .output()
        .expect("the compiled program runs");
    assert!(!run.status.success());
    assert!(
        String::from_utf8_lossy(&run.stderr).contains(trap),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// A `vmap` body called inside an arm consumes keys the arm splits from a key
/// no other arm uses; the body's draws carry no activation, which V3 admits
/// because nothing shares that key. Row `b` of the selected arm's gradient is
/// `uniform(split_n(key(1), 3)[b])`. Disposition lock: it passes at
/// b005bb19b too. It is why V3's confinement covers only keys derived from a
/// key shared through exclusive activations: this body's draws carry none.
#[test]
fn a_vmap_body_in_an_arm_consumes_the_arms_unshared_keys() {
    let out = both_lanes(
        "b3-vmap-in-arm",
        r#"module Demo.Main
def row(k: key, x: tensor[2, f32]) -> tensor[2, f32] = mul(uniform_like(k, x, 0.0f32, 1.0f32), x)
def mapped_arm(k: key, xs: tensor[3, 2, f32], gate: i64) -> f32 = if eq(gate, 1i64) then tensor_to_scalar(sum(sum(vmap(row)(split_keys(k, 3i64), xs), 0i32), 0i32)) else tensor_to_scalar(sum(sum(copy(xs), 0i32), 0i32))
vmap_in_selected_arm = grad(mapped_arm, wrt=xs)(key_from_seed(1i64), to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]), 1i64)
vmap_in_unselected_arm = grad(mapped_arm, wrt=xs)(key_from_seed(1i64), to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]), 0i64)
"#,
    );
    assert_eq!(
        bits(&out, "vmap_in_selected_arm"),
        [
            0x3df7_8d2f,
            0x3f21_f02c,
            0x3d12_ce0b,
            0x3d6f_f9a8,
            0x3e9f_ed9e,
            0x3f09_a115
        ]
    );
    assert_eq!(bits(&out, "vmap_in_unselected_arm"), [0x3f80_0000; 6]);
}
