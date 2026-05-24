//! Issue #212 regression: validator cascade-suppression for conv2d
//! is one-level-deep, so a chain of length 3+ rooted on a
//! non-concrete dim emits a phantom diagnostic for every downstream
//! let-binder past the first cascade-suppressed RHS.
//!
//! Surfaced by the round-4 red team on PR #205. The
//! `failed_let_names` insertion in `validate_ir_expr` (let arm) was
//! guarded by `errors.len() > errs_before`: once cascade suppression
//! activates for level 2 (suppressing `y2`'s RHS diagnostic via
//! `y1 ∈ failed_let_names`), the RHS pushes no errors, so `y2` is
//! never marked failed and the next consumer fires its own cascade
//! error.
//!
//! Fix: insert into `failed_let_names` when the RHS is a recognized
//! shape-sensitive IR builtin (or a passthrough wrapper around one)
//! AND its output type is non-derivable, regardless of whether the
//! RHS pushed errors. The R3 F-A `errs_before` guard's stated intent
//! is "do not suppress when the RHS is not a recognized shape-
//! sensitive call"; substituting an explicit "is this a recognized
//! shape-sensitive shape" check delivers the same narrowness without
//! depending on diagnostic count.
//!
//! Three positive probes (3-level, 4-level, the issue body's named
//! R4 probes) lock the desired ONE-diagnostic shape; the negative
//! parity probe pins that two independent failures still produce two
//! errors so the fix does not over-suppress.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

/// EXPECT (RT-205 round-4, issue #212): a 3-level conv2d let-chain
/// rooted on a non-concrete spatial dim produces exactly ONE
/// `requires concrete tensor argument metadata` diagnostic.
///
/// Before the fix this produced 2 diagnostics (`conv2d(&y1, ...)`
/// and `conv2d(&y2, ...)` both fired) because `y2` was never marked
/// failed: its RHS was cascade-suppressed by `y1`'s failed state, so
/// the RHS pushed no diagnostics, so the `errors.len() > errs_before`
/// guard did not mark `y2`.
#[test]
fn rt205_r4_cascade_propagation_multi_level_pins_bug() {
    let src = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], k3: tensor[32, 16, 3, 3, f32]) -> tensor[1, 32, 2, 2, f32] = {
  y1 = conv2d(&x, &k1, 1, 0)
  y2 = conv2d(&y1, &k2, 1, 0)
  conv2d(&y2, &k3, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete input dim");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "expected exactly 1 metadata-cascade error for 3-level chain, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-4, issue #212): the bug surfaces with bare
/// `conv2d` calls and no wrapper ops between let-bindings. This
/// mirrors the issue body's plain-conv2d reproducer and locks the
/// length-3 minimum boundary.
#[test]
fn rt205_r4_cascade_propagation_bug_plain_conv2d() {
    // The minimal failing case: only 2 let-bindings (`y1`, `y2`) plus
    // a tail `conv2d(&y2, ...)`. Before the fix the validator's
    // suppression covered the `y2` RHS but missed the tail. After
    // the fix the tail is suppressed too.
    let src = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], k3: tensor[32, 16, 3, 3, f32]) -> tensor[1, 32, 2, 2, f32] = {
  y1 = conv2d(&x, &k1, 1, 0)
  y2 = conv2d(&y1, &k2, 1, 0)
  conv2d(&y2, &k3, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "plain-conv2d 3-level chain must produce exactly 1 metadata error, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-4, issue #212): root cause is the
/// non-concrete dim, not the chain length per se. Confirm by varying
/// the chain root while keeping the depth fixed: a non-concrete `in_c`
/// or `h` at the input both produce ONE diagnostic, not N.
#[test]
fn rt205_r4_cascade_propagation_bug_via_nonconcrete_dim() {
    // Non-concrete spatial axis (h).
    let src_h = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], k3: tensor[32, 16, 3, 3, f32]) -> tensor[1, 32, 2, 2, f32] = {
  y1 = conv2d(&x, &k1, 1, 0)
  y2 = conv2d(&y1, &k2, 1, 0)
  conv2d(&y2, &k3, 1, 0)
}
"#;
    let deep = surf_to_deep(src_h);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete h");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "non-concrete h: expected 1 metadata error, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );

    // Non-concrete input-channel axis (in_c). Same depth, different
    // root failure. Per RT-205 round-3 F-C the spatial/channel axes
    // are required concrete (only `batch` may be symbolic), so this
    // also routes through the cascade-dedup path on `y1`.
    let src_in_c = r#"
def f(x: tensor[1, in_c, 8, 16, f32], k1: tensor[8, in_c, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], k3: tensor[32, 16, 3, 3, f32]) -> tensor[1, 32, 2, 2, f32] = {
  y1 = conv2d(&x, &k1, 1, 0)
  y2 = conv2d(&y1, &k2, 1, 0)
  conv2d(&y2, &k3, 1, 0)
}
"#;
    let deep = surf_to_deep(src_in_c);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete in_c");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "non-concrete in_c: expected 1 metadata error, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (issue #212): a 4-level conv2d let-chain produces exactly
/// ONE metadata diagnostic. Before the fix this produced 3
/// diagnostics (for `y2`, `y3`, and the tail) because each
/// downstream let-binder re-emitted its own cascade error. This pins
/// that the fix propagates the suppression unboundedly, not just one
/// extra level.
#[test]
fn rt205_r4_cascade_propagation_four_level_chain() {
    let src = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], k3: tensor[32, 16, 3, 3, f32], k4: tensor[64, 32, 3, 3, f32]) -> tensor[1, 64, 1, 1, f32] = {
  y1 = conv2d(&x, &k1, 1, 0)
  y2 = conv2d(&y1, &k2, 1, 0)
  y3 = conv2d(&y2, &k3, 1, 0)
  conv2d(&y3, &k4, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for 4-level chain");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "4-level chain must produce exactly 1 metadata error, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (issue #212 negative parity): three independent (non-
/// cascading) conv2d failures still produce three diagnostics. The
/// fix's suppression key remains "let-bound name whose RHS is a
/// shape-sensitive call with non-derivable output", not "any
/// duplicate-shaped message". This pins that the new fix does NOT
/// regress the round-2 F3 independent-failure contract when extended
/// from 2 calls to 3.
#[test]
fn rt205_r4_three_independent_failures_not_suppressed() {
    let src = r#"
def f(x1: tensor[1, 3, h, 16, f32], x2: tensor[1, 3, h, 16, f32], x3: tensor[1, 3, h, 16, f32], k: tensor[8, 3, 3, 3, f32]) -> (tensor[1, 8, 6, 6, f32], tensor[1, 8, 6, 6, f32], tensor[1, 8, 6, 6, f32]) = {
  y1 = conv2d(&x1, &k, 1, 0)
  y2 = conv2d(&x2, &k, 1, 0)
  y3 = conv2d(&x3, &k, 1, 0)
  (y1, y2, y3)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete input dims");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        3,
        "expected 3 independent metadata errors (one per call), got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (issue #212): the cascade-suppression depth fix also
/// propagates through R3 F-A passthrough wrappers. A 3-level chain
/// where each intermediate let-RHS wraps the conv2d in `relu(...)`
/// must still produce exactly ONE diagnostic. This pins that the
/// "recognized shape-sensitive RHS" recognition is symmetric with
/// the existing R3 F-A passthrough recognition: if `derive_ir_builtin
/// _output_type` recurses through relu, the failed-marker insertion
/// must too.
#[test]
fn rt205_r4_cascade_propagation_through_relu_wrapper_multi_level() {
    let src = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], k3: tensor[32, 16, 3, 3, f32]) -> tensor[1, 32, 2, 2, f32] = {
  y1 = relu(conv2d(&x, &k1, 1, 0))
  y2 = relu(conv2d(&y1, &k2, 1, 0))
  conv2d(&y2, &k3, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete input dim through relu");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "relu-wrapped 3-level chain must produce exactly 1 metadata error, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (issue #212 systemic): the cascade-suppression depth fix
/// works for an alternating mix of bare conv2d and shape-passthrough
/// wrappers. This pins that the structural recognition predicate
/// does not regress between wrapped and bare segments of the same
/// chain.
#[test]
fn rt205_r4_cascade_propagation_alternating_wrapped_and_bare() {
    let src = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], k3: tensor[32, 16, 3, 3, f32]) -> tensor[1, 32, 2, 2, f32] = {
  y1 = conv2d(&x, &k1, 1, 0)
  y2 = relu(conv2d(&y1, &k2, 1, 0))
  conv2d(&y2, &k3, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for alternating chain");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "alternating wrapped/bare 3-level chain must produce exactly 1 metadata error, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (issue #212 negative parity, design intent lock): a let
/// chain where the intermediate level is a user-defined function
/// call (NOT a recognized shape-sensitive IR builtin or passthrough)
/// must NOT cause cascade-suppression downstream of that user
/// function. This locks the design choice that the recognition
/// predicate stays narrow: structural recognition is limited to the
/// closed set of shape-sensitive IR builtins and the unary/binary
/// passthrough allowlist, not "anything in a let-RHS position".
///
/// Pattern: `helper(t)` is a user fn taking a non-concrete-h tensor;
/// it consumes a let-bound y1 from a failed conv2d. The user-fn
/// call's let-bound name (y2) must NOT be marked failed by the
/// validator's cascade-suppression path (helper is not a recognized
/// shape-sensitive builtin), so any downstream failure on y2 still
/// fires its own diagnostic. We verify this by chaining a downstream
/// conv2d(&y2, ...) and asserting the metadata error count is the
/// validator-level error from y1's RHS plus the downstream
/// validator-level error from conv2d(&y2, ...) -- two errors, NOT
/// one. If the predicate had wrongly recognized helper(...) as a
/// passthrough, only one error would fire.
#[test]
fn rt205_r4_cascade_does_not_suppress_user_fn_chain() {
    // First conv2d fails (non-concrete h); helper is a user fn that
    // returns its input; downstream conv2d uses helper's result.
    // The validator must NOT mark y2 as failed via the cascade path,
    // so the downstream conv2d still emits its own diagnostic.
    let src = r#"
def helper(t: tensor[1, 8, w1, w2, f32]) -> tensor[1, 8, w1, w2, f32] = t

def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y1 = conv2d(&x, &k1, 1, 0)
  y2 = helper(y1)
  conv2d(&y2, &k2, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        2,
        "user-fn-wrapped chain must NOT trigger cascade-suppression: \
         expected 2 metadata errors (y1's RHS + downstream conv2d), got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}
