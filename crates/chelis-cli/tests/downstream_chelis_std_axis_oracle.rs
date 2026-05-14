//! Downstream `chelis test` oracle for the negative-axis + rank-0
//! standalone-parameter fixes.
//!
//! Both bugs shared one downstream-visible symptom: `chelis test` on
//! ANY package that links bundled chelis-std panicked during library
//! lowering -- `softmax axis requires a statically known axis in IR
//! lowering`. The panic came from `packages/chelis-std/src/loss/
//! crossentropy.ch`'s `softmax(logits, 1)` (Bug A: `logits` is a
//! borrowed symbolic-dim parameter whose type comes from a separate
//! `sig`, so standalone lowering bound it to a rank-0 type) and would
//! equally come from `packages/chelis-std/src/nn/attention.ch`'s
//! `softmax(scaled, -1)` (Bug B: a negative axis literal mapped to
//! `usize::MAX`). Either unlowerable chelis-std def poisoned the whole
//! linked context, so no downstream `chelis test` could run.
//!
//! This is the acceptance oracle: a minimal downstream package whose
//! test file imports a chelis-std module that transitively uses the
//! previously-unlowerable defs must `chelis test` clean. It is not
//! `#[ignore]`d -- the fix is the thing that makes it green by
//! default, amortized over `SharedReef`'s one-time chelis-std publish.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;
use common::{make_app, write_file};

/// A downstream package whose `tests/` file imports
/// `Std.Loss.CrossEntropy` -- the Bug A trigger module -- and runs a
/// trivial Chelis-native test through `chelis test`. The import forces
/// the linked chelis-std context (including `crossentropy.ch` and,
/// transitively, `attention.ch`) to lower; before the fix that
/// lowering panicked and `chelis test` could not run at all.
#[test]
fn downstream_test_importing_chelis_std_loss_runs_clean() {
    let (_dir, reef_home, app_pkg) = make_app("axis-oracle-loss");

    // Keep `src/main.ch` trivial; the load-bearing part is the test
    // file's `import Std.Loss.CrossEntropy`, which pulls the
    // previously-unlowerable chelis-std defs into the linked context.
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\n\ndef id(x: int32) -> int32 = x\n",
    );
    write_file(
        &app_pkg.join("tests/axis.ch"),
        r#"module Demo.Tests.Axis

import Std.Loss.CrossEntropy (loss)
import Std.Test (assert_true)

def test_chelis_std_loss_links_clean() -> unit =
  assert_true(true, "linking Std.Loss.CrossEntropy must not panic lowering")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        // The specific pre-fix panic string must never appear.
        .stderr(predicate::str::contains("softmax axis requires a statically known axis").not());
}

/// Same shape, but the downstream test imports `Std.Nn.Attention` --
/// the Bug B trigger module, which uses `softmax(scaled, -1)`. Pins
/// that the negative-axis chelis-std def lowers cleanly when linked
/// from a downstream package.
#[test]
fn downstream_test_importing_chelis_std_attention_runs_clean() {
    let (_dir, reef_home, app_pkg) = make_app("axis-oracle-attention");

    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\n\ndef id(x: int32) -> int32 = x\n",
    );
    write_file(
        &app_pkg.join("tests/axis.ch"),
        r#"module Demo.Tests.Axis

import Std.Nn.Attention (scaled_dot_product_attention)
import Std.Test (assert_true)

def test_chelis_std_attention_links_clean() -> unit =
  assert_true(true, "linking Std.Nn.Attention must not panic lowering")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .stderr(predicate::str::contains("softmax axis requires a statically known axis").not());
}
