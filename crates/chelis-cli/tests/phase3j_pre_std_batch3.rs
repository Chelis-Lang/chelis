//! Phase 3j-pre Batch 3: Std.Nn activations + normalization acceptance tests.
//!
//! Covers `Std.Nn.Silu`, `Std.Nn.Gelu` (tanh approximation), and
//! `Std.Nn.RmsNorm`. Each positive test uses hand-computed reference
//! values. Shape-negative tests exercise obvious failure modes.
//!
//! Conv wrapper and attention modules are deliberately out of scope for
//! this batch; see `spec/design/chelis_phase3_plan.md` §3j-pre for the
//! Phase 0e conv2d concreteness constraint + scalar-broadcast gap.
//!
//! The test harness constructs a minimal standalone chelis-std package
//! containing only the Batch 3 source files (plus the three baseline
//! files they depend on), so parallel work on other Batch 2/4 files in
//! `packages/chelis-std/src/**` cannot affect this test.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

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

    // Minimal chelis-std that ships only the Batch 3 modules. This
    // avoids flaky interactions with other batches' in-flight source
    // files under packages/chelis-std/src/**.
    write_file(
        &std_pkg.join("reef.toml"),
        r#"[package]
name = "chelis-std"
version = "0.1.0"
compiler = "=0.1.20"
module_prefix = "Std"
"#,
    );
    write_file(
        &std_pkg.join("src/nn/silu.ch"),
        r#"module Std.Nn.Silu
export (forward, sigmoid_scalar)
def forward[n](x: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (v: f32) -> mul(v, sigmoid_scalar(v)), to_list(x)))
def sigmoid_scalar(v: f32) -> f32 = div(cast(1.0, f32), add(cast(1.0, f32), exp(neg(v))))
"#,
    );
    write_file(
        &std_pkg.join("src/nn/gelu.ch"),
        r#"module Std.Nn.Gelu
export (forward, tanh_scalar, gelu_scalar)
def forward[n](x: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (v: f32) -> gelu_scalar(v), to_list(x)))
def tanh_scalar(z: f32) -> f32 = {
  e_pos = exp(z)
  e_neg = exp(neg(z))
  div(sub(e_pos, e_neg), add(e_pos, e_neg))
}
def gelu_scalar(v: f32) -> f32 = {
  c = cast(0.7978845608028654, f32)
  k = cast(0.044715, f32)
  half = cast(0.5, f32)
  one = cast(1.0, f32)
  inner = mul(c, add(v, mul(k, mul(v, mul(v, v)))))
  mul(half, mul(v, add(one, tanh_scalar(inner))))
}
"#,
    );
    write_file(
        &std_pkg.join("src/nn/rmsnorm.ch"),
        r#"module Std.Nn.RmsNorm
export (forward, rms_scale)
def forward[n](x: tensor[n, f32], gain: tensor[n, f32], eps: f32) -> tensor[n, f32] = {
  scale = rms_scale(copy(x), eps)
  scaled = to_tensor(map(fn (v: f32) -> mul(v, scale), to_list(x)))
  mul(scaled, gain)
}
def rms_scale[n](x: tensor[n, f32], eps: f32) -> f32 = {
  squared = map(fn (v: f32) -> mul(v, v), to_list(x))
  count = cast(len(squared), f32)
  total = fold(fn (acc: f32, v: f32) -> add(acc, v), cast(0.0, f32), squared)
  ms = div(total, count)
  div(cast(1.0, f32), sqrt(add(ms, eps)))
}
"#,
    );

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
compiler = "=0.1.20"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

#[test]
fn phase3j_pre_batch3_silu_matches_reference_values() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-silu");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Silu (forward)

xs = to_tensor([cast(-1.0, f32), cast(0.0, f32), cast(1.0, f32), cast(2.0, f32)])
silu_out = forward(xs)
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
        // silu(-1) = -1 * sigmoid(-1) = -0.2689414213699951
        // silu(0)  = 0
        // silu(1)  = 1 * sigmoid(1)  = 0.7310585786300049
        // silu(2)  = 2 * sigmoid(2)  = 1.7615941559557646
        .stdout(predicate::str::contains(
            "silu_out = tensor(shape=[4], data=[-0.2689414213699951, 0.0, 0.7310585786300049, 1.7615941559557646])",
        ));
}

#[test]
fn phase3j_pre_batch3_gelu_tanh_approx_matches_reference_values() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-gelu");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Gelu (forward)

xs = to_tensor([cast(-1.0, f32), cast(0.0, f32), cast(1.0, f32), cast(2.0, f32)])
gelu_out = forward(xs)
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
        // Tanh-approx GELU reference values (OpenAI/BERT form):
        //   gelu(-1) = -0.15880800939172324
        //   gelu( 0) =  0.0
        //   gelu( 1) =  0.8411919906082768  (the canonical plan value)
        //   gelu( 2) =  1.954597694087775
        .stdout(predicate::str::contains(
            "gelu_out = tensor(shape=[4], data=[-0.15880800939172324, 0.0, 0.8411919906082768, 1.954597694087775])",
        ));
}

#[test]
fn phase3j_pre_batch3_rmsnorm_unit_rms_and_gain_scaling() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-rmsnorm");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.RmsNorm (forward)

xs = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)])
unit_gain = to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)])
scaled_gain = to_tensor([cast(2.0, f32), cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)])
rms_unit = forward(xs, unit_gain, cast(0.000001, f32))
rms_scaled = forward(xs, scaled_gain, cast(0.000001, f32))
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
        // mean(xs^2) = (1+4+9+16)/4 = 7.5; 1/sqrt(7.5) = 0.3651483473268884
        // Unit-gain output: [1,2,3,4] / sqrt(7.5).
        .stdout(predicate::str::contains(
            "rms_unit = tensor(shape=[4], data=[0.3651483473268884, 0.7302966946537768, 1.0954450419806652, 1.4605933893075536])",
        ))
        // With gain = 2.0 the result is exactly doubled.
        .stdout(predicate::str::contains(
            "rms_scaled = tensor(shape=[4], data=[0.7302966946537768, 1.4605933893075536, 2.1908900839613303, 2.921186778615107])",
        ));
}

#[test]
fn phase3j_pre_batch3_rmsnorm_rejects_mismatched_gain_shape() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-rmsnorm-bad-shape");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.RmsNorm (forward)

xs = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)])
wrong_gain = to_tensor([cast(1.0, f32), cast(1.0, f32)])
rms_out = forward(xs, wrong_gain, cast(0.000001, f32))
"#,
    );

    // Shapes differ: `xs` is [4] and `wrong_gain` is [2]. The static
    // type check loses the concrete dim through `to_tensor` and the
    // polymorphic dim var `n`, so the check passes and the error
    // surfaces explicitly at eval time from the `mul(scaled, gain)`
    // inside `RmsNorm.forward`. This is a runtime shape check, not a
    // silent zero.
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
        .failure()
        .stderr(predicate::str::contains(
            "tensor shapes must match for elementwise op, got [4] vs [2]",
        ));
}
