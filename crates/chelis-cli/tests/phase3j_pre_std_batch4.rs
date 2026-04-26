//! Phase 3j-pre Batch 4 — Std.Loss (KL div, BCE, metrics) and Std.Init
//! (random/kaiming/xavier) acceptance tests.
//!
//! Each numeric assertion uses an exact or tolerance comparison computed
//! inside the Chelis program (booleans emitted as `= true` in the eval
//! output), so the Rust side only matches fully-qualified `name = true`
//! strings — no `contains("true")` ambiguity across lines.
//!
//! Negative parity: KL divergence and BCE with mismatched shapes must fail
//! typechecking with a size/unification error.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn example_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(rel)
        .canonicalize()
        .expect("path should exist")
}

fn package_std() -> PathBuf {
    example_path("../../packages/chelis-std")
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

fn stub_module(path: &Path, module: &str) {
    if path.exists() {
        fs::write(path, format!("module {module}\nexport ()\n")).expect("stub");
    }
}

fn make_app(dir_name: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join(dir_name);
    copy_dir_recursive(&package_std(), &std_pkg);
    // Remove any pre-built shell so publish always re-reads src/.
    let dist = std_pkg.join("dist");
    if dist.exists() {
        fs::remove_dir_all(&dist).expect("remove dist");
    }
    // Stub out sibling-batch modules (3j-pre Batches 2/3) whose source is
    // still WIP in the working tree; Batch 4 does not depend on them. Any
    // file missing from the working tree is left alone, so once those
    // batches land cleanly this stubbing becomes a no-op.
    let src = std_pkg.join("src");
    stub_module(&src.join("nn/gelu.ch"), "Std.Nn.Gelu");
    stub_module(&src.join("nn/silu.ch"), "Std.Nn.Silu");
    stub_module(&src.join("nn/rmsnorm.ch"), "Std.Nn.RmsNorm");
    stub_module(&src.join("tensor/construct.ch"), "Std.Tensor.Construct");
    stub_module(&src.join("tensor/reduce.ch"), "Std.Tensor.Reduce");
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
compiler = "=0.2.7"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

const HELPERS: &str = r#"
def sample_mean[n](t: tensor[n, f32]) -> f32 = {
  xs = to_list(t)
  total = fold(fn (acc: f32, x: f32) -> add(acc, x), cast(0.0, f32), xs)
  div(total, cast(len(xs), f32))
}

def sample_std[n](t: tensor[n, f32]) -> f32 = {
  m = sample_mean(copy(t))
  xs = to_list(t)
  sq = map(fn (x: f32) -> mul(sub(x, m), sub(x, m)), xs)
  total = fold(fn (acc: f32, x: f32) -> add(acc, x), cast(0.0, f32), sq)
  sqrt(div(total, cast(len(sq), f32)))
}

def abs_f32(x: f32) -> f32 = if lt(x, cast(0.0, f32)) then neg(x) else x
def close_to(x: f32, target: f32, tol: f32) -> bool = lt(abs_f32(sub(x, target)), tol)
"#;

#[test]
fn phase3j_pre_batch4_losses_metrics_and_inits_evaluate_correctly() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch4");
    let main_ch = format!(
        r#"module Demo.Main
import Std.Init.Random (normal_like)
import Std.Init.Kaiming (kaiming_uniform, kaiming_normal)
import Std.Init.XavierExt (xavier_uniform, xavier_normal, trunc_normal)
import Std.Loss.KlDiv (kl_divergence)
import Std.Loss.Bce (bce_with_logits)
import Std.Loss.Metrics (accuracy, perplexity)
{HELPERS}

template = to_tensor(map(fn (i: int64) -> cast(0.0, f32), range(cast(0, int64), cast(1000, int64))))

-- KL divergence: uniform distributions give 0; hand case matches reference.
p_uniform = to_tensor([0.5, 0.5])
q_uniform = to_tensor([0.5, 0.5])
kl_uniform = kl_divergence(p_uniform, q_uniform)
kl_uniform_zero = close_to(kl_uniform, cast(0.0, f32), cast(0.000001, f32))

p_case = to_tensor([0.25, 0.75])
q_case = to_tensor([0.5, 0.5])
kl_case = kl_divergence(p_case, q_case)
kl_case_ok = close_to(kl_case, cast(0.13081203, f32), cast(0.0001, f32))

-- BCE with logits: both elements should evaluate to log(1 + e^-0.1) = 0.6443967.
bce_z = to_tensor([0.1, -0.1])
bce_y = to_tensor([1.0, 0.0])
bce_out = bce_with_logits(bce_z, bce_y)
bce_list = to_list(bce_out)
bce0_ok = close_to(index(bce_list, cast(0, int64)), cast(0.6443967, f32), cast(0.0001, f32))
bce1_ok = close_to(index(bce_list, cast(1, int64)), cast(0.6443967, f32), cast(0.0001, f32))

-- Accuracy: argmaxes are [1, 0, 2] vs labels [1, 0, 1] -> 2/3.
acc_logits = pad_sequences_to([[0.1, 0.9, 0.2], [0.6, 0.3, 0.1], [0.0, 0.2, 0.8]], cast(3, int64), 0.0)
acc_labels = to_tensor([cast(1, int64), cast(0, int64), cast(1, int64)])
acc_out = accuracy(acc_logits, acc_labels)
acc_ok = close_to(acc_out, cast(0.6666667, f32), cast(0.0001, f32))

-- Perplexity: exp(ln 3) = 3.
ppx_ok = close_to(perplexity(cast(1.0986123, f32)), cast(3.0, f32), cast(0.0001, f32))

-- normal_like: n = 1000, seed = 42 -> mean ~ 0, std ~ 1 within 0.1.
nl_sample = with seed(42) {{ normal_like(copy(template), cast(0.0, f32), cast(1.0, f32)) }}
nl_mean_ok = close_to(sample_mean(copy(nl_sample)), cast(0.0, f32), cast(0.1, f32))
nl_std_ok = close_to(sample_std(nl_sample), cast(1.0, f32), cast(0.1, f32))

-- Kaiming uniform(fan_in=4): std should be sqrt(6/4)/sqrt(3) = sqrt(1/2) ~ 0.7071.
ku_sample = with seed(43) {{ kaiming_uniform(copy(template), cast(4.0, f32)) }}
ku_std_ok = close_to(sample_std(ku_sample), cast(0.7071, f32), cast(0.1, f32))

-- Kaiming normal(fan_in=4): std should be sqrt(2/4) = 0.7071.
kn_sample = with seed(44) {{ kaiming_normal(copy(template), cast(4.0, f32)) }}
kn_std_ok = close_to(sample_std(kn_sample), cast(0.7071, f32), cast(0.1, f32))

-- Xavier uniform(4, 4): bound = sqrt(6/8) = 0.866, std = bound/sqrt(3) = 0.5.
xu_sample = with seed(45) {{ xavier_uniform(copy(template), cast(4.0, f32), cast(4.0, f32)) }}
xu_std_ok = close_to(sample_std(xu_sample), cast(0.5, f32), cast(0.1, f32))

-- Xavier normal(4, 4): std = sqrt(2/8) = 0.5.
xn_sample = with seed(46) {{ xavier_normal(copy(template), cast(4.0, f32), cast(4.0, f32)) }}
xn_std_ok = close_to(sample_std(xn_sample), cast(0.5, f32), cast(0.1, f32))

-- trunc_normal: all samples should lie inside [-0.5, 0.5].
tn_sample = with seed(47) {{ trunc_normal(copy(template), cast(0.0, f32), cast(1.0, f32), cast(-0.5, f32), cast(0.5, f32)) }}
tn_max = fold(fn (acc: f32, x: f32) -> if gt(x, acc) then x else acc, cast(-10.0, f32), to_list(copy(tn_sample)))
tn_min = fold(fn (acc: f32, x: f32) -> if lt(x, acc) then x else acc, cast(10.0, f32), to_list(tn_sample))
tn_bounds_ok = and(lte(tn_max, cast(0.5, f32)), gte(tn_min, cast(-0.5, f32)))
"#
    );
    write_file(&app_pkg.join("src/main.ch"), &main_ch);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    let output = Command::cargo_bin("chelis")
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
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).expect("utf-8 eval stdout");
    let checks = [
        "kl_uniform_zero = true",
        "kl_case_ok = true",
        "bce0_ok = true",
        "bce1_ok = true",
        "acc_ok = true",
        "ppx_ok = true",
        "nl_mean_ok = true",
        "nl_std_ok = true",
        "ku_std_ok = true",
        "kn_std_ok = true",
        "xu_std_ok = true",
        "xn_std_ok = true",
        "tn_bounds_ok = true",
    ];
    for needle in checks {
        assert!(
            stdout.contains(needle),
            "expected `{needle}` in eval output:\n{stdout}"
        );
    }
}

#[test]
fn phase3j_pre_batch4_kl_divergence_rejects_shape_mismatch() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch4-kl-bad");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main
import Std.Loss.KlDiv (kl_divergence)

-- p has 2 elements, q has 3 — type checker should reject on the dim var
-- once the concrete sizes are pinned via an annotated cast.
def two_elem(xs: List[f32]) -> tensor[2, f32] = to_tensor(xs)
def three_elem(xs: List[f32]) -> tensor[3, f32] = to_tensor(xs)
p = two_elem([0.5, 0.5])
q = three_elem([0.33, 0.33, 0.34])
bad = kl_divergence(p, q)
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
        .failure();
}

#[test]
fn phase3j_pre_batch4_bce_rejects_shape_mismatch() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch4-bce-bad");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main
import Std.Loss.Bce (bce_with_logits)

def two_elem(xs: List[f32]) -> tensor[2, f32] = to_tensor(xs)
def three_elem(xs: List[f32]) -> tensor[3, f32] = to_tensor(xs)
z = two_elem([0.1, -0.1])
y = three_elem([1.0, 0.0, 0.0])
bad = bce_with_logits(z, y)
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
        .failure();
}
