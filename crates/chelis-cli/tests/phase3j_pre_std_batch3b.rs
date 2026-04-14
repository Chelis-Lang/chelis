//! Phase 3j-pre Batch 3b: Std.Nn.Conv and Std.Nn.Attention acceptance.
//!
//! Batch 3 shipped the activations + RmsNorm (see `phase3j_pre_std_batch3.rs`).
//! Batch 3b adds the remaining Batch 3 scope:
//!
//!   - `Std.Nn.Conv`: `conv1d`, `conv2d_small`
//!   - `Std.Nn.Attention`: `scaled_dot_product_attention`,
//!     `multi_head_attention`, `grouped_query_attention`,
//!     `gqa_broadcast_kv`
//!
//! ## Acceptance surface and documented limitations
//!
//! The wrappers ship as **literal concrete-shape** defs sized for a small
//! transformer block (seq=4, d_head=4, n_heads=2, head_dim=4,
//! n_kv_heads=1; conv1d in_c=4/out_c=8/k=3/L=16; conv2d in_c=3/out_c=8/
//! k=3x3/h=w=8). Two underlying compiler limitations force concrete
//! shapes, both tracked as non-silent deferrals in
//! `spec/design/chelis_phase3_plan.md` §3j-pre:
//!
//!   1. `conv2d` requires concrete d-lit output dims at Phase 0e lowering
//!      time (`validate_phase0e_builtin_symbolic_requirements`); a
//!      polymorphic `conv2d_forward[batch, in_c, out_c, ...]` wrapper is
//!      rejected.
//!
//!   2. `tensor_binop` uses a single shared tvar `(T,T)→T`, so the
//!      `/sqrt(d_k)` scaling in SDPA cannot be expressed as a scalar
//!      multiply. Callers must therefore pass `scale` as a pre-built
//!      rank-2 tensor of the same shape as `scores`.
//!
//! The concrete-shape wrappers additionally cannot be exercised with
//! numeric eval through the host runtime for two reasons:
//!
//!   - the host runtime lowering does not implement `matmul`, `softmax`,
//!     `permute`, or `expand`, so consumer-level `chelis eval` of the
//!     attention wrappers is not reachable (only `build`+gcc is; that
//!     path is already validated by the `phase3i_std` oracle pattern and
//!     does not need per-batch duplication here);
//!   - rank-changing `reshape` is rejected by the package-mode
//!     enforce-defsig pass whenever a `def` is present in the same
//!     source unit, so we cannot construct a rank-4 input literal for
//!     the conv wrappers from a rank-1 `to_tensor` result.
//!
//! The acceptance surface this file pins is therefore the same one used
//! by `phase3j_pre_std_batch2` for the rank-changing tensor wrappers
//! (stack/squeeze/unsqueeze): **publish + import-touch + check-level
//! shape negatives + a positive gather eval that exercises the GQA KV
//! broadcast pattern through the host runtime**. Taken together these
//! give:
//!
//!   - positive: the new wrappers type-check inside `chelis-std` and
//!     are importable from a downstream consumer (`score == 1`);
//!   - positive: `gather(src, [0,0,1,1], axis=0)` returns the
//!     broadcasted KV layout SDPA/GQA rely on, with exact numeric
//!     output;
//!   - negative (SDPA): q/k seqlen disagreement errors at check time;
//!   - negative (MHA):  head-dim disagreement errors at check time;
//!   - negative (GQA):  wrong group-map rank errors at check time;
//!   - negative (Conv2d): kernel channel mismatch errors at check time.

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
    // Strip any prebuilt `dist/` entry from the copy so the publish
    // command always rebuilds against current sources.
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
compiler = "=0.1.3"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

/// Publishing the full `chelis-std` package (done by `make_app`) already
/// exercises the type checker + Phase 0e lowering pass on every wrapper
/// in `Std.Nn.Conv` and `Std.Nn.Attention`. If either file failed to
/// type-check or triggered the `conv2d` concrete-output assertion in
/// `tier2.rs`, `reef publish` would fail.
///
/// This test pins the importability of every shipped name from a
/// downstream consumer. The `touch_*` bindings force the name resolver
/// and import surface to walk every exported symbol.
#[test]
fn phase3j_pre_batch3b_conv_and_attention_wrappers_publish_and_import() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch3b-import");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Conv (conv1d, conv2d_small)
import Std.Nn.Attention (
  scaled_dot_product_attention,
  multi_head_attention,
  grouped_query_attention,
  gqa_broadcast_kv,
)

touch_conv1d = conv1d
touch_conv2d = conv2d_small
touch_sdpa = scaled_dot_product_attention
touch_mha = multi_head_attention
touch_gqa = grouped_query_attention
touch_gqa_bcast = gqa_broadcast_kv
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
}

/// Gather is the runtime primitive that `Std.Nn.Attention.gqa_broadcast_kv`
/// lowers to. This test asserts the exact broadcasted layout the GQA
/// wrapper relies on: a 4-entry source tensor and a [0,0,1,1] group
/// map produces the broadcasted sequence [src[0], src[0], src[1],
/// src[1]].
///
/// Writing this as an explicit positive assertion (rather than relying
/// on transitive coverage) is required by the plan; see
/// `feedback_verify_deps_exist`.
#[test]
fn phase3j_pre_batch3b_gather_broadcasts_kv_heads_for_gqa() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch3b-gather");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

kv_src = to_tensor([cast(10.0, f32), cast(20.0, f32), cast(30.0, f32), cast(40.0, f32)])
group_map = to_tensor([cast(0, int64), cast(0, int64), cast(1, int64), cast(1, int64)])
broadcast_kv = gather(kv_src, group_map, 0)
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
            "broadcast_kv = tensor(shape=[4], data=[10.0, 10.0, 20.0, 20.0])",
        ));
}

#[test]
fn phase3j_pre_batch3b_sdpa_rejects_q_k_seqlen_mismatch() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch3b-sdpa-seqlen");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Attention (scaled_dot_product_attention)

def bad_seqlen(q: tensor[3, 4, f32], k: tensor[4, 4, f32], v: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = scaled_dot_product_attention(q, k, v, scale)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1").not())
        .stdout(predicate::str::contains("DimensionMismatch"));
}

#[test]
fn phase3j_pre_batch3b_mha_rejects_head_dim_mismatch() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch3b-mha-headdim");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Attention (multi_head_attention)

def bad_headdim(q: tensor[4, 8, f32], k: tensor[4, 4, f32], v: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = multi_head_attention(q, k, v, scale)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1").not())
        .stdout(predicate::str::contains("DimensionMismatch"));
}

#[test]
fn phase3j_pre_batch3b_gqa_rejects_wrong_group_map_rank() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch3b-gqa-mask");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Attention (gqa_broadcast_kv)

def bad_group_map(pool: tensor[1, 4, 4, f32], gm: tensor[2, 3, int64]) -> tensor[2, 4, 4, f32] = gqa_broadcast_kv(pool, gm)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1").not())
        .stdout(predicate::str::contains("DimensionMismatch"));
}

#[test]
fn phase3j_pre_batch3b_conv2d_rejects_kernel_channel_mismatch() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch3b-conv2d-channels");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Conv (conv2d_small)

def bad_channels(x: tensor[1, 4, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = conv2d_small(x, k)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1").not())
        .stdout(predicate::str::contains("DimensionMismatch"));
}

#[test]
fn phase3j_pre_batch3b_conv1d_rejects_kernel_length_mismatch() {
    let (_dir, reef_home, app_pkg) = make_app("phase3j-pre-batch3b-conv1d-length");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Conv (conv1d)

def bad_kernel_len(x: tensor[1, 4, 1, 16, f32], k: tensor[8, 4, 1, 5, f32]) -> tensor[1, 8, 1, 14, f32] = conv1d(x, k)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1").not())
        .stdout(predicate::str::contains("DimensionMismatch"));
}
