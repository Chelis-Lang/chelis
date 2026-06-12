//! chelis#258 Tier-3 rank polymorphism acceptance corpus.
//!
//! Name-preserving rank *arithmetic*: a single `def` reduces a named axis in a
//! rank-polymorphic way and the type/shape checker computes the resulting shape
//! symbolically, carrying the surviving named axes through. The headline form
//! is the named-axis reduction:
//!
//! ```text
//! def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32]
//!   = sum(x, seq)
//! ```
//!
//! A tensor shape is now `Rank? (Name Rank?)*` (spreads interleaved with named
//! anchors). Unification locates each named anchor uniquely in the operand and
//! binds the spreads between — unitary because each split is forced by a name.
//!
//! Soundness boundary (`spec/04-type-system.md` §4.5.3, §4.2): a spread preserves
//! the real named dims it covers (not a count), the reduced axis is a retained
//! name, order is preserved, and the Body-Discipline check admits only
//! name-trackable ops (elementwise + named reductions) — a positional
//! `permute`/`reshape` at symbolic rank is rejected.
//!
//! Positive/negative parity per CLAUDE.md: every "checks clean" test has a
//! paired "rejected with the right reason" test.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn check_json(src: &str) -> Value {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    fs::write(&path, src).expect("write file");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn fmt_stdout(src: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    fs::write(&path, src).expect("write file");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap()])
        .output()
        .expect("run chelis fmt");
    assert!(
        output.status.success(),
        "chelis fmt must succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 fmt stdout")
}

fn assert_clean(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        errors.is_empty(),
        "{label}: expected no check errors, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(0.0);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "{label}: expected score 1.0, got {score} ({json})"
    );
}

/// Assert a rejection whose message contains `needle` — pins the *reason*, not
/// just that some error fired, so a wrong-reason regression is caught.
fn assert_rejected_with(json: &Value, needle: &str, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected rejection, got a clean check ({json})"
    );
    let has = errors
        .iter()
        .any(|e| e["message"].as_str().is_some_and(|m| m.contains(needle)));
    assert!(
        has,
        "{label}: expected a rejection mentioning {needle:?}, got {errors:?}"
    );
}

// ── Positives ───────────────────────────────────────────────────────────

/// The headline: ONE rank-poly named-reduce def, callable at ranks 2, 3, and 4
/// from concrete-rank callers, with every surviving named axis carried through.
/// This collapses the `batchnorm1d`/`_2d`/`_3d` verb-name family into one def
/// (chelis#258) for the reduction case.
#[test]
fn named_reduce_callable_at_ranks_2_3_4() {
    let json = check_json(
        "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use2(x: &tensor[seq, hidden, f32]) -> tensor[hidden, f32] = reduce_seq(x)\n\
         def use3(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = reduce_seq(x)\n\
         def use4(x: &tensor[batch, depth, seq, hidden, f32]) -> tensor[batch, depth, hidden, f32] = reduce_seq(x)\n",
    );
    assert_clean(&json, "named reduce callable at ranks 2-4");
}

/// A leading anchor with a trailing spread: reduce the first (named) axis.
#[test]
fn reduce_leading_named_axis() {
    let json = check_json(
        "def drop_batch(x: &tensor[batch, ..rest, f32]) -> tensor[..rest, f32] = sum(x, batch)\n\
         def use(x: &tensor[batch, seq, hidden, f32]) -> tensor[seq, hidden, f32] = drop_batch(x)\n",
    );
    assert_clean(&json, "reduce leading named axis");
}

/// A trailing anchor with a leading spread: reduce the last (named) axis.
#[test]
fn reduce_trailing_named_axis() {
    let json = check_json(
        "def drop_last(x: &tensor[..pre, hidden, f32]) -> tensor[..pre, f32] = sum(x, hidden)\n\
         def use(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, seq, f32] = drop_last(x)\n",
    );
    assert_clean(&json, "reduce trailing named axis");
}

/// Multi-axis (the user's "reduce over 2 axes") via composition: drop two named
/// axes by composing single-axis named reductions. The intermediate type with
/// adjacent spreads is sound — it is only ever the operand of the next reduce
/// (located by name), never split against a ground.
#[test]
fn multi_axis_reduce_via_composition() {
    let json = check_json(
        "def reduce_two(x: &tensor[..a, seq, ..b, head, ..c, f32]) -> tensor[..a, ..b, ..c, f32] = sum(sum(x, head), seq)\n\
         def use(x: &tensor[batch, seq, kv, head, feat, f32]) -> tensor[batch, kv, feat, f32] = reduce_two(x)\n",
    );
    assert_clean(&json, "multi-axis reduce over seq and head via composition");
}

/// `mean` is a named-axis reduction too (not only `sum`) — both lower through
/// the tensor-DAG backend and build end-to-end.
#[test]
fn mean_is_name_tracked() {
    let json = check_json(
        "def m(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = mean(x, seq)\n",
    );
    assert_clean(&json, "mean name-tracked in a ..r body");
}

/// `max_reduce`/`min_reduce`/`prod_reduce` are NOT yet admitted in a `..r` body:
/// they route through the host lane in a rank-poly inline and don't compile
/// (chelis#340), so they are rejected at check time to keep check↔backend in
/// sync (a check-clean program must build). They remain usable at concrete rank.
#[test]
fn max_reduce_in_rank_poly_body_rejected() {
    let json = check_json(
        "def m(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = max_reduce(x, seq)\n",
    );
    assert_rejected_with(&json, "name-trackable", "max_reduce in a ..r body");
}

/// Concrete-rank control: a named reduction on a fully-concrete shape (no
/// spread) drops the named axis and keeps the rest — the feature does not
/// require a spread, and does not regress ordinary defs.
#[test]
fn concrete_rank_named_reduce_clean() {
    let json = check_json(
        "def f(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = sum(x, seq)\n",
    );
    assert_clean(&json, "concrete-rank named reduce");
}

/// An elementwise op composed with a named reduction in the same `..r` body.
#[test]
fn elementwise_then_reduce_in_rank_poly_body() {
    let json = check_json(
        "def f(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(relu(x), seq)\n",
    );
    assert_clean(&json, "relu then named reduce in a ..r body");
}

// ── Negatives ───────────────────────────────────────────────────────────

/// The reduced axis must actually be dropped: declaring a return that keeps the
/// reduced axis is rejected (the body's symbolic output does not unify).
#[test]
fn declared_return_keeps_reduced_axis_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, seq, ..post, f32] = sum(x, seq)\n",
    );
    assert_rejected_with(
        &json,
        "doesn't match declared signature",
        "return keeps the reduced `seq` axis",
    );
}

/// Reducing an axis the operand does not have is a hard error.
#[test]
fn reduce_nonexistent_axis_rejected() {
    let json = check_json(
        "def f(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = sum(x, nope)\n",
    );
    assert_rejected_with(
        &json,
        "neither a compile-time constant nor a named axis",
        "reduce a non-existent axis `nope`",
    );
}

/// Name↔Lit hard-reject: a fully-literal caller carries no name to locate the
/// anchor, so the named reduction cannot resolve — rejected, never a guess.
#[test]
fn fully_literal_operand_cannot_locate_anchor() {
    let json = check_json(
        "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use(x: &tensor[2, 768, 64, f32]) -> tensor[2, 64, f32] = reduce_seq(x)\n",
    );
    assert_rejected_with(&json, "seq", "fully-literal operand has no named seq axis");
}

/// An ambiguous anchor (the name appears more than once in the operand) is
/// rejected rather than silently reducing the first occurrence.
#[test]
fn ambiguous_anchor_rejected() {
    let json = check_json(
        "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use(x: &tensor[seq, mid, seq, f32]) -> tensor[mid, f32] = reduce_seq(x)\n",
    );
    assert_rejected_with(&json, "ambiguous", "anchor `seq` appears twice");
}

/// Two adjacent spreads split against a concrete operand is the undetermined
/// (non-unitary) case — rejected at the call site.
#[test]
fn adjacent_spreads_split_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..a, ..b, f32]) -> tensor[..a, ..b, f32] = relu(x)\n\
         def use(x: &tensor[m, n, f32]) -> tensor[m, n, f32] = bad(x)\n",
    );
    assert_rejected_with(&json, "two adjacent rank spreads", "adjacent-spread split");
}

/// A positional shape-rewriting op (`permute`) in a `..r` body is rejected:
/// not name-trackable at symbolic rank (§4.2 / §4.5.3).
#[test]
fn permute_in_rank_poly_body_rejected() {
    let json = check_json(
        "def evil(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = permute(x, 1, 0)\n",
    );
    assert_rejected_with(&json, "name-trackable", "permute in a ..r body");
}

/// `reshape` (fuses axes, destroys names) in a `..r` body is rejected.
#[test]
fn reshape_in_rank_poly_body_rejected() {
    let json = check_json(
        "def evil(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = reshape(x, [2, 3])\n",
    );
    assert_rejected_with(&json, "shape-rewriting", "reshape in a ..r body");
}

/// A duplicate spread name in one tensor shape is a parse error (it would bind
/// the same run twice).
#[test]
fn duplicate_spread_name_rejected() {
    let json =
        check_json("def f(x: &tensor[..r, seq, ..r, f32]) -> tensor[..r, f32] = sum(x, seq)\n");
    assert_rejected_with(
        &json,
        "distinct rank-spread name",
        "duplicate spread name `..r`",
    );
}

// ── Named-axis expand (R+1, chelis#339) ─────────────────────────────────
// The inverse arithmetic direction (spec/04-type-system.md §4.5.3):
// `expand(x, new, size)` inserts a trailing named axis; the 4-arg
// `expand(x, new, size, anchor)` inserts immediately before an existing
// named anchor. Insertion strictly inside an opaque spread has no anchor
// and stays rejected.

/// Trailing insert: ONE rank-poly def appends a named axis at any rank, and
/// concrete-rank callers at ranks 1/2/3 all monomorphize cleanly.
#[test]
fn named_expand_trailing_callable_at_ranks_1_2_3() {
    let json = check_json(
        "def add_axis(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, one, 1)\n\
         def use1(x: &tensor[seq, f32]) -> tensor[seq, one, f32] = add_axis(x)\n\
         def use2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, one, f32] = add_axis(x)\n\
         def use3(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, seq, hidden, one, f32] = add_axis(x)\n",
    );
    assert_clean(&json, "named trailing expand callable at ranks 1-3");
}

/// Anchored insert: the new axis lands immediately before the named anchor,
/// monomorphized at two distinct concrete shapes with the anchor at
/// DIFFERENT positions (leading and interior).
#[test]
fn named_expand_by_anchor_callable_at_two_anchor_positions() {
    let json = check_json(
        "def widen(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, c, seq, ..post, f32] = expand(x, c, 5, seq)\n\
         def use_lead(x: &tensor[seq, hidden, f32]) -> tensor[c, seq, hidden, f32] = widen(x)\n\
         def use_mid(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, c, seq, hidden, f32] = widen(x)\n",
    );
    assert_clean(&json, "anchored expand at leading and interior anchors");
}

/// Leading-end insert via a leading named anchor: a row that BEGINS with a
/// named anchor admits insertion before it (the spec's leading-end rule).
#[test]
fn named_expand_leading_via_leading_anchor() {
    let json = check_json(
        "def lead(x: &tensor[seq, ..rest, f32]) -> tensor[c, seq, ..rest, f32] = expand(x, c, 2, seq)\n\
         def use(x: &tensor[seq, hidden, f32]) -> tensor[c, seq, hidden, f32] = lead(x)\n",
    );
    assert_clean(&json, "leading insert via leading named anchor");
}

/// Concrete-rank control: the named forms also work without any spread, and
/// compose with the named reduction (reduce the just-inserted axis by name).
#[test]
fn named_expand_concrete_rank_clean() {
    let json = check_json(
        "def f(x: &tensor[batch, seq, f32]) -> tensor[batch, c, seq, f32] = expand(x, c, 4, seq)\n\
         def g(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, one, f32] = expand(x, one, 1)\n",
    );
    assert_clean(&json, "concrete-rank named expand (trailing + anchored)");
}

/// NEGATIVE: insertion strictly inside an opaque spread has no anchor. The
/// computed output row places the new axis only at an end or at an anchor,
/// so a declared result demanding `[..lo, c, ..hi]` from `[..rest]` fails
/// row unification and is rejected (spec §4.5.3).
#[test]
fn named_expand_inside_opaque_spread_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..rest, f32]) -> tensor[..lo, c, ..hi, f32] = expand(x, c, 4)\n",
    );
    assert_rejected_with(
        &json,
        "doesn't match declared signature",
        "insertion strictly inside an opaque spread",
    );
}

/// NEGATIVE: the anchor must be a named axis of the operand. A rank-spread
/// operand with no such axis is a hard error naming the §4.5.3 rule.
#[test]
fn named_expand_absent_anchor_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, c, seq, ..post, f32] = expand(x, c, 5, nope)\n",
    );
    assert_rejected_with(&json, "nope", "anchor `nope` absent from the operand row");
}

/// NEGATIVE: an ambiguous anchor (appears more than once in the operand) is
/// rejected rather than silently picking an occurrence.
#[test]
fn named_expand_ambiguous_anchor_rejected() {
    let json = check_json(
        "def f(x: &tensor[seq, mid, seq, f32]) -> tensor[seq, mid, c, seq, f32] = expand(x, c, 2, seq)\n",
    );
    assert_rejected_with(&json, "ambiguous", "anchor `seq` appears twice");
}

/// NEGATIVE: the inserted name must not collide with an existing axis name —
/// a duplicate dim name would make every later by-name lookup ambiguous.
#[test]
fn named_expand_duplicate_inserted_name_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, seq, ..post, seq, f32] = expand(x, seq, 5)\n",
    );
    assert_rejected_with(
        &json,
        "already names an axis",
        "inserted name collides with existing `seq` axis",
    );
}

/// NEGATIVE: a positional (integer) insert axis on a rank-spread operand is
/// meaningless at symbolic rank — concrete-rank only, like reductions.
#[test]
fn named_expand_positional_axis_on_spread_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, 0, 1)\n",
    );
    assert_rejected_with(
        &json,
        "positional",
        "positional insert axis on a rank-spread operand",
    );
}

/// NEGATIVE (chelis#339 red team): the inserted name collides with an axis
/// the caller's rank spread covers, and the collision is visible in the
/// SIGNATURE (the inserted name survives into the result row). The symbolic
/// collision check inside the def cannot see it; the call-site
/// introduced-name rule must reject it — otherwise the monomorphized result
/// carries `chan` twice with extents 2 and 5 (`[chan, chan, seq]` believed
/// `[2, 2, 3]`, actual `[2, 5, 3]`).
#[test]
fn named_expand_spread_covered_collision_rejected_at_check() {
    let json = check_json(
        "def widen(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, chan, seq, ..post, f32] = expand(x, chan, 5, seq)\n\
         def use_col(x: &tensor[chan, seq, f32]) -> tensor[chan, chan, seq, f32] = widen(x)\n",
    );
    assert_rejected_with(
        &json,
        "collides with an inserted axis name",
        "anchored insert: spread covers the inserted name at the call site",
    );
}

/// NEGATIVE twin for the trailing form: `add_axis` called with an operand
/// whose leading axis is already named `one`.
#[test]
fn named_expand_trailing_spread_covered_collision_rejected_at_check() {
    let json = check_json(
        "def add_axis(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, one, 1)\n\
         def use_col(x: &tensor[one, seq, f32]) -> tensor[one, seq, one, f32] = add_axis(x)\n",
    );
    assert_rejected_with(
        &json,
        "collides with an inserted axis name",
        "trailing insert: spread covers the inserted name at the call site",
    );
}

/// PINNED GAP + loudness lock (chelis#339 red team): when the inserted name
/// is consumed INSIDE the body (insert + reduce-by-name), the signature
/// carries no trace of it, so the check stays clean and the collision only
/// materializes at call-site rank monomorphization. Both lanes must fail
/// LOUDLY — before the lowering guard this silently reduced the WRONG axis
/// (backend printed shape [5, 3] against a declared `[chan, seq]` = [2, 3]).
/// If the check ever learns to reject this at check time, fold this into the
/// check-rejection tests above.
#[test]
fn named_expand_body_internal_collision_fails_loud_not_silent() {
    let source = "def wr(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, seq, ..post, f32] = sum(expand(x, chan, 5, seq), chan)\n\
         def use_col(x: &tensor[chan, seq, f32]) -> tensor[chan, seq, f32] = wr(x)\n\
         y = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         out = use_col(y)\n";
    // The check is clean — the gap this test pins.
    assert_clean(&check_json(source), "body-internal collision checks clean");
    // Build lane: the fatal expand-site lowering guard, never garbage C and
    // never a silent wrong-axis reduce.
    let stderr = build_expecting_failure(source, "body_internal_collision");
    assert!(
        stderr.contains("already carries an axis named `chan`"),
        "expected the expand-site collision diagnostic, got: {stderr}"
    );
    // Eval lane: loud failure (the #338 routing decline or the collision
    // guard, depending on staging), never a silent wrong shape.
    let dir = tempdir().expect("tempdir");
    let eval_stderr = eval_stderr_expecting_failure(dir.path(), source, "body_internal_collision");
    assert!(
        eval_stderr.contains("chan"),
        "expected a loud named-axis failure mentioning the colliding axis, got: {eval_stderr}"
    );
}

/// PINNED GAP + loudness lock (chelis#339 red team): a single-letter dim VAR
/// in the operand signature lowers to `Named` with its source letter, so an
/// inserted axis with the same letter collides at IR level even though the
/// checker (which sees an anonymous `Dim::Var`) stays clean. Must fail loudly
/// in both lanes — the declared result `[c, seq, c]` would otherwise carry
/// extents 2 and 4 under one name.
#[test]
fn named_expand_dvar_letter_collision_fails_loud() {
    let source = "def f(x: &tensor[c, seq, f32]) -> tensor[c, seq, c, f32] = expand(x, c, 4)\n\
         y = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         out = f(y)\n";
    assert_clean(&check_json(source), "d-var letter collision checks clean");
    let stderr = build_expecting_failure(source, "dvar_letter_collision");
    assert!(
        stderr.contains("already carries an axis named `c`"),
        "expected the expand-site collision diagnostic, got: {stderr}"
    );
    let dir = tempdir().expect("tempdir");
    let eval_stderr = eval_stderr_expecting_failure(dir.path(), source, "dvar_letter_collision");
    assert!(
        eval_stderr.contains("already carries an axis named `c`"),
        "expected the expand-site collision diagnostic in eval, got: {eval_stderr}"
    );
}

/// NEGATIVE (chelis#339 red team): the named-insert size must be a
/// compile-time literal. A symbolic-dim size loses the inserted NAME at
/// lowering (the same-body `sum(.., chan)` then cannot locate it and the
/// emitted C referenced the raw symbol), and a runtime int32 size produced a
/// silent shape-0 tensor in the backend. Both are rejected at check time.
#[test]
fn named_expand_size_must_be_compile_time_literal() {
    let symbolic = check_json(
        "def f(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = sum(expand(x, chan, batch), chan)\n",
    );
    assert_rejected_with(
        &symbolic,
        "compile-time literal size",
        "symbolic-dim size in the named insert form",
    );
    let runtime = check_json(
        "def f(x: &tensor[seq, f32], k: int32) -> tensor[seq, chan, f32] = expand(x, chan, k)\n",
    );
    assert_rejected_with(
        &runtime,
        "compile-time literal size",
        "runtime int32 size in the named insert form",
    );
}

/// CONTROL (chelis#339 red team): a visible leading anchor with a
/// trailing-spread-covered axis of the SAME name is legal and must keep
/// reducing the visible (leftmost) anchor — this pins the first-hit
/// resolution a naive multiple-hits ambiguity guard would break.
#[test]
fn named_reduce_visible_anchor_with_spread_covered_duplicate_stays_correct() {
    let source = "def f(x: &tensor[seq, ..rest, f32]) -> tensor[..rest, f32] = sum(x, seq)\n\
         def use_dup(x: &tensor[seq, hidden, seq, f32]) -> tensor[hidden, seq, f32] = f(x)\n\
         y = to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]])\n\
         out = use_dup(y)\n";
    let backend = build_compile_run(source, "visible_anchor_dup");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(out.1, vec![3, 2], "backend shape ({backend})");
    for (i, e) in [11.0, 22.0, 33.0, 44.0, 55.0, 66.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e}",
            out.2[i]
        );
    }
    assert_eval_agrees_with_backend(source, "visible_anchor_dup", &backend);
}

/// The named-expand def builds, compiles, runs, and the backend agrees with
/// the eval oracle — trailing insert at ranks 1 and 2, anchored insert at two
/// anchor positions, all NON-SQUARE so an axis mislabel fails loudly, plus a
/// size>1 broadcast (expand replicates data along the new axis).
#[test]
fn named_expand_builds_runs_and_evals() {
    let source = "def add_axis(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, one, 1)\n\
         def widen(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, c, seq, ..post, f32] = expand(x, c, 3, seq)\n\
         def a1(x: &tensor[seq, f32]) -> tensor[seq, one, f32] = add_axis(x)\n\
         def a2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, one, f32] = add_axis(x)\n\
         def w_lead(x: &tensor[seq, hidden, f32]) -> tensor[c, seq, hidden, f32] = widen(x)\n\
         def w_mid(x: &tensor[batch, seq, f32]) -> tensor[batch, c, seq, f32] = widen(x)\n\
         out1 = a1(to_tensor([1.0, 2.0, 3.0]))\n\
         out2 = a2(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outl = w_lead(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         outm = w_mid(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    let backend = build_compile_run(source, "named_expand_builds_runs");
    let tensors = parse_printed_tensors(&backend);
    // out1: [seq=3] -> [3, 1], data unchanged.
    // out2: [batch=2, seq=3] -> [2, 3, 1], data unchanged.
    // outl: [seq=2, hidden=2] -> [c=3, 2, 2]: 3 copies of the input.
    // outm: [batch=2, seq=3] -> [2, c=3, 3]: per batch row, 3 copies.
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out1", &[3, 1], &[1.0, 2.0, 3.0]),
        ("out2", &[2, 3, 1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        (
            "outl",
            &[3, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        (
            "outm",
            &[2, 3, 3],
            &[
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0,
                5.0, 6.0,
            ],
        ),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        assert_eq!(got.2.len(), data.len(), "{name}: backend len ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "named_expand_builds_runs", &backend);
}

/// Transform lanes over the new capability: `grad` through a named expand
/// (the broadcast adjoint sums over the inserted axis -> gradient = size
/// copies) and `vmap` over a def that calls the rank-poly named-expand def,
/// both eval-vs-backend pinned.
#[test]
fn named_expand_under_grad_and_vmap_evals_and_matches_backend() {
    let source = "def widen(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, c, seq, ..post, f32] = expand(x, c, 3, seq)\n\
         def inner(x: &tensor[seq, f32]) -> tensor[c, seq, f32] = widen(x)\n\
         def total(x: &tensor[seq, f32]) -> f32 = tensor_to_scalar(sum(sum(widen(x), c), seq))\n\
         out = vmap(inner)(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         gr = grad(total)(to_tensor([1.0, 2.0]))\n";
    assert_clean(&check_json(source), "expand grad/vmap matrix checks clean");
    let backend = build_compile_run(source, "named_expand_grad_vmap");
    let tensors = parse_printed_tensors(&backend);
    // out: per batch slice [seq=2] -> [c=3, seq=2] (3 copies).
    // gr: total = 3 * sum(x), so d/dx = 3 everywhere.
    let expected: &[(&str, &[usize], &[f64])] = &[
        (
            "out",
            &[2, 3, 2],
            &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        ),
        ("gr", &[2], &[3.0, 3.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        assert_eq!(got.2.len(), data.len(), "{name}: backend len ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "named_expand_grad_vmap", &backend);
}

/// The anchored 4-arg sig survives `chelis fmt` (round-trip + idempotence +
/// re-checks clean), mirroring the reduction round-trip invariant.
#[test]
fn named_expand_survives_fmt_round_trip() {
    let src = "def widen(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, c, seq, ..post, f32] = expand(x, c, 5, seq)\n";
    let once = fmt_stdout(src);
    assert!(
        once.contains("..pre") && once.contains("expand(x, c, 5, seq)"),
        "fmt must preserve the named-expand call, got:\n{once}"
    );
    let twice = fmt_stdout(&once);
    assert_eq!(once, twice, "chelis fmt must be idempotent on named expand");
    assert_clean(&check_json(&once), "formatted named expand re-checks clean");
}

/// TOP-LEVEL named-axis apps (no def-call boundary): the eval lane's site-A
/// interception must route a bare `expand(y, one, 1)` / `expand(y, c, 3,
/// seq)` / variadic `sum(y, batch, seq)` root through IR lowering — the
/// def-call tests above only exercise site B, so a site-A regression would
/// otherwise be invisible. Both lanes pinned value-for-value.
#[test]
fn top_level_named_expand_and_variadic_sum_eval_match_backend() {
    let source = "def id2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n\
         y = id2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_t = expand(y, one, 1)\n\
         out_a = expand(y, c, 3, seq)\n\
         out_vr = sum(y, batch, seq)\n";
    let backend = build_compile_run(source, "top_level_named_axis_ops");
    let tensors = parse_printed_tensors(&backend);
    // out_t: trailing insert -> [2, 2, 1], data unchanged.
    // out_a: c=3 inserted before seq (axis 1) -> [2, 3, 2], rows tripled.
    // out_vr: all-axes variadic sum -> rank-0 [/* 10 */].
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out_t", &[2, 2, 1], &[1.0, 2.0, 3.0, 4.0]),
        (
            "out_a",
            &[2, 3, 2],
            &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        ),
        ("out_vr", &[], &[10.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        assert_eq!(got.2.len(), data.len(), "{name}: backend len ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "top_level_named_axis_ops", &backend);
}

// ── Variadic named-axis reduction (chelis#339 Part 2) ───────────────────
// `sum(x, seq, head)` reduces several named axes in one call, equivalent
// to the documented composition `sum(sum(x, head), seq)` and
// order-insensitive. Defined for the value reductions
// (sum/mean/max_reduce/min_reduce/prod_reduce); NOT for the
// index-returning argmax/argmin (composition is ill-defined).

/// Variadic `sum`/`mean` check clean at concrete rank AND in a rank-poly
/// body, in both axis orders.
#[test]
fn variadic_reduce_checks_clean() {
    let json = check_json(
        "def two(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = sum(x, seq, head)\n\
         def two_rev(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = sum(x, head, seq)\n\
         def rp(x: &tensor[..a, seq, ..b, head, ..c, f32]) -> tensor[..a, ..b, ..c, f32] = sum(x, seq, head)\n\
         def use_rp(x: &tensor[batch, seq, kv, head, feat, f32]) -> tensor[batch, kv, feat, f32] = rp(x)\n\
         def m2(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = mean(x, seq, head)\n",
    );
    assert_clean(&json, "variadic named reduce checks clean");
}

/// Variadic ≡ composed, numerically, on BOTH lanes: `sum(x, seq, head)`
/// equals `sum(sum(x, head), seq)` and the axis order does not matter.
/// `mean` and `max_reduce` ride along (mean-of-means == joint mean with
/// uniform weights; max is idempotent across orders). Non-square [2,3,4]
/// so an axis mislabel fails loudly.
#[test]
fn variadic_reduce_builds_runs_and_evals() {
    let source = "def direct(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = sum(x, seq, head)\n\
         def swapped(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = sum(x, head, seq)\n\
         def composed(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = sum(sum(x, head), seq)\n\
         def mboth(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = mean(x, seq, head)\n\
         def xboth(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = max_reduce(x, seq, head)\n\
         def rp(x: &tensor[..a, seq, ..b, head, ..c, f32]) -> tensor[..a, ..b, ..c, f32] = sum(x, seq, head)\n\
         def use_rp(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = rp(x)\n\
         def tot(x: &tensor[seq, head, f32]) -> f32 = tensor_to_scalar(sum(x, seq, head))\n\
         def vinner(x: &tensor[seq, head, f32]) -> f32 = tensor_to_scalar(sum(x, seq, head))\n\
         y = to_tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]])\n\
         out_d = direct(y)\n\
         out_s = swapped(y)\n\
         out_c = composed(y)\n\
         out_m = mboth(y)\n\
         out_x = xboth(y)\n\
         out_r = use_rp(y)\n\
         gr = grad(tot)(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_v = vmap(vinner)(to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))\n";
    // NOTE on `out_v`: the vmap probe deliberately uses an INLINE literal
    // operand. `vmap(vinner)(y)` with the shared top-level `y` binding hits
    // a PRE-EXISTING dag.rs symbolic-dim ICE on main (verified at d786744
    // with the explicit composition `sum(sum(x, head), seq)` — vmap +
    // binding-typed Load + a two-stage named reduce; the chelis#346/#351
    // annotation-dims family in a lane those fixes did not cover). The
    // variadic surface desugars to that same composition, so it inherits
    // the gap unchanged; see the chelis#339 PR for the boundary analysis.
    let backend = build_compile_run(source, "variadic_reduce");
    let tensors = parse_printed_tensors(&backend);
    // Per batch slice (3x4): b0 sums 1..=12 = 78; b1 sums 13..=24 = 222.
    // mean = sum/12; max = last element (24 in b1, 12 in b0).
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out_d", &[2], &[78.0, 222.0]),
        ("out_s", &[2], &[78.0, 222.0]),
        ("out_c", &[2], &[78.0, 222.0]),
        ("out_m", &[2], &[6.5, 18.5]),
        ("out_x", &[2], &[12.0, 24.0]),
        ("out_r", &[2], &[78.0, 222.0]),
        // grad of the all-axes sum is ones, and vmap over the variadic
        // scalar reduce yields the per-slice sums (the #351 lesson: pin
        // transform lanes on new rank-poly capability from day one).
        ("gr", &[2, 2], &[1.0, 1.0, 1.0, 1.0]),
        ("out_v", &[2], &[10.0, 26.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "variadic_reduce", &backend);
}

/// NEGATIVE: a duplicate axis name in the variadic list is rejected, never
/// silently deduplicated.
#[test]
fn variadic_reduce_duplicate_axis_rejected() {
    let json = check_json(
        "def bad(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, seq, seq)\n",
    );
    assert_rejected_with(&json, "duplicate", "duplicate axis `seq` in variadic sum");
}

/// NEGATIVE: an unknown axis name in the variadic list is rejected.
#[test]
fn variadic_reduce_unknown_axis_rejected() {
    let json = check_json(
        "def bad(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = sum(x, seq, nope)\n",
    );
    assert_rejected_with(&json, "nope", "unknown axis `nope` in variadic sum");
}

/// NEGATIVE: positional integers are not admitted in the variadic form —
/// each axis must be named.
#[test]
fn variadic_reduce_positional_axes_rejected() {
    let json = check_json(
        "def bad(x: &tensor[batch, seq, head, f32]) -> tensor[batch, f32] = sum(x, 1, 2)\n",
    );
    assert_rejected_with(
        &json,
        "positional",
        "positional integer axes in variadic sum",
    );
}

/// NEGATIVE: index-returning reductions have no variadic form — an index
/// along one axis is not composable with a second reduction.
#[test]
fn variadic_argmax_rejected() {
    let json = check_json(
        "def bad(x: &tensor[batch, seq, head, f32]) -> tensor[batch, int64] = argmax_reduce(x, seq, head)\n",
    );
    assert_rejected_with(
        &json,
        "index-returning",
        "variadic argmax_reduce has no defined semantics",
    );
}

/// NEGATIVE arity pins: the variadic dispatcher must not soften existing
/// wrong-arity rejections — a 1-arg `sum` and an over-applied non-reduction
/// builtin (`relu(x, y)`) still fail.
#[test]
fn variadic_dispatcher_preserves_arity_errors() {
    let json = check_json("def bad(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x)\n");
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        !errors.is_empty(),
        "1-arg sum must still be rejected, got clean: {json}"
    );
    let json2 = check_json(
        "def bad2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x, x)\n",
    );
    let errors2 = json2["errors"].as_array().expect("errors array");
    assert!(
        !errors2.is_empty(),
        "over-applied relu must still be rejected, got clean: {json2}"
    );
}

// ── Formatter round-trip ────────────────────────────────────────────────

/// The anchored multi-spread sig survives `chelis fmt`: `..pre`/`..post` are
/// preserved in every tensor position, fmt is idempotent, and the formatted
/// text re-checks clean (the CLAUDE.md formatter round-trip invariant).
#[test]
fn anchored_spread_survives_fmt_round_trip() {
    let src = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n";
    let once = fmt_stdout(src);
    assert!(
        once.contains("..pre") && once.contains("..post") && once.contains("seq"),
        "fmt must preserve the spreads and the anchor, got:\n{once}"
    );
    let twice = fmt_stdout(&once);
    assert_eq!(
        once, twice,
        "chelis fmt must be idempotent on the anchored sig"
    );
    assert_clean(&check_json(&once), "formatted anchored sig re-checks clean");
}

// ── Backend build + run (call-site rank monomorphization) ───────────────
// A program that calls a rank-poly named-reduce def through a concrete-rank
// caller builds, compiles, and runs: call-site monomorphization substitutes
// the caller's concrete shape for each spread, and the named axis `seq` is
// resolved to a positional index at lowering (where the operand's named dims
// are available). The output value is verified directly against the hand-
// computed reduction AND against the `chelis eval` oracle: since chelis#338,
// the host runtime routes a def call that requires named-axis resolution
// through the same `lower_subexpr_program` + forward-DAG-eval lane the C
// backend uses (see `try_named_axis_def_call` in
// crates/chelis-compiler-api/src/runtime/named_axis.rs), so eval-vs-backend agreement
// is restored for the Tier-3 surface.

fn build_compile_run(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    fs::write(&src, source).expect("write source");

    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        build.status.success(),
        "rank-poly reduce build must succeed; stderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let c_source = format!("{name}.c");
    let needs_blas = fs::read_to_string(out_dir.join(&c_source))
        .map(|t| t.contains("cblas_sgemm(") || t.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );
    let bin = out_dir.join(name);
    let mut cc = StdCommand::new(&toolchain.compiler);
    cc.current_dir(&out_dir)
        .arg("-O2")
        .args(&toolchain.compile_flags)
        .arg(&c_source)
        .args(["-L.", "-lchelis_runtime"])
        .args(&toolchain.link_flags)
        .args(["-o", bin.to_str().unwrap()]);
    let link = cc.status().expect("host compiler runs");
    assert!(
        link.success(),
        "link of rank-poly reduce C must succeed: {link}"
    );

    let run = StdCommand::new(&bin).output().expect("binary runs");
    assert!(
        run.status.success(),
        "rank-poly reduce binary must run: {}\nstderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 stdout")
}

/// Run `chelis build --target c` expecting failure; return stderr so the
/// caller can pin the diagnostic (the loudness lock for monomorphization-time
/// collisions that the checker cannot see).
fn build_expecting_failure(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    fs::write(&src, source).expect("write source");
    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        !build.status.success(),
        "build was expected to fail; stdout: {}",
        String::from_utf8_lossy(&build.stdout)
    );
    String::from_utf8(build.stderr).expect("utf-8 build stderr")
}

/// Run `chelis eval --file` and return its stdout (the evaluator oracle the
/// backend must agree with, per the backend-numerics discipline).
fn eval_stdout(dir: &Path, source: &str, name: &str) -> String {
    let src = dir.join(format!("{name}.ch"));
    fs::write(&src, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            src.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "rank-poly named-reduce eval must succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

/// Run `chelis eval --file` expecting failure; return stderr for message pins.
fn eval_stderr_expecting_failure(dir: &Path, source: &str, name: &str) -> String {
    let src = dir.join(format!("{name}.ch"));
    fs::write(&src, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            src.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("run chelis eval");
    assert!(
        !output.status.success(),
        "eval was expected to fail; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8(output.stderr).expect("utf-8 stderr")
}

/// Assert every backend-printed tensor has an eval twin within 1e-6 (the
/// Tier-2 eval-vs-backend agreement oracle, unblocked for Tier-3 by #338).
fn assert_eval_agrees_with_backend(source: &str, name: &str, backend: &str) {
    assert_eval_agrees_with_backend_tol(source, name, backend, 1e-6);
}

/// Tolerance-parameterized agreement oracle: the backend computes f32,
/// eval computes f64, so transcendental outputs need a looser absolute
/// tolerance than the 1e-6 used for exact-arithmetic corpora.
fn assert_eval_agrees_with_backend_tol(source: &str, name: &str, backend: &str, tol: f64) {
    let backend_tensors = parse_printed_tensors(backend);
    assert!(
        !backend_tensors.is_empty(),
        "{name}: backend printed no tensors: {backend}"
    );
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, name);
    let eval_tensors = parse_printed_tensors(&eval);
    for (tensor_name, shape, data) in &backend_tensors {
        let e = eval_tensors
            .iter()
            .find(|(n, _, _)| n == tensor_name)
            .unwrap_or_else(|| panic!("evaluator output missing `{tensor_name}`: {eval}"));
        assert_eq!(
            shape, &e.1,
            "{tensor_name}: eval-vs-backend shape disagreement"
        );
        assert_eq!(
            data.len(),
            e.2.len(),
            "{tensor_name}: eval-vs-backend length"
        );
        for (i, (b, ev)) in data.iter().zip(e.2.iter()).enumerate() {
            assert!(
                (b - ev).abs() < tol,
                "{tensor_name}[{i}]: eval-vs-backend disagreement: backend {b} vs eval {ev}"
            );
        }
    }
}

fn parse_printed_tensors(stdout: &str) -> Vec<(String, Vec<usize>, Vec<f64>)> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let Some((name, rest)) = line.split_once(" = tensor(") else {
            continue;
        };
        let shape = rest
            .split_once("shape=[")
            .and_then(|(_, s)| s.split_once(']'))
            .map(|(s, _)| {
                s.split(',')
                    .filter_map(|p| p.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let data = rest
            .split_once("data=[")
            .and_then(|(_, s)| s.split_once(']'))
            .map(|(s, _)| {
                s.split(',')
                    .filter_map(|p| p.trim().parse::<f64>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        out.push((name.trim().to_string(), shape, data));
    }
    out
}

/// ONE rank-poly named-reduce def, called at rank 2 and rank 3, builds and
/// runs and the backend reduces over the correct (named) `seq` axis at each
/// rank — the call-site monomorphization + named-axis lowering proof.
///
/// CRITICAL: the operands are deliberately NON-SQUARE (the reduced axis size
/// differs from every surviving axis size). A square operand masks an
/// axis-mislabel bug in call-site monomorphization (the surviving axis was
/// renamed to the reduced axis, which only aborts when the sizes differ —
/// chelis#258 red-team finding). Every distinct size here is load-bearing.
#[test]
fn named_reduce_builds_and_runs_nonsquare_at_ranks_2_3_4() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def avg_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = mean(x, seq)\n\
         def r2(x: &tensor[seq, hidden, f32]) -> tensor[hidden, f32] = reduce_seq(x)\n\
         def r3(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = reduce_seq(x)\n\
         def r4(x: &tensor[batch, depth, seq, hidden, f32]) -> tensor[batch, depth, hidden, f32] = reduce_seq(x)\n\
         def m3(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = avg_seq(x)\n\
         out2 = r2(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         out3 = r3(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n\
         out4 = r4(to_tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]))\n\
         outm = m3(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n";
    let backend = build_compile_run(source, "rank_poly_reduce_nonsquare");
    let tensors = parse_printed_tensors(&backend);

    // out2: seq(=2) reduced from [seq=2, hidden=3] → [hidden=3] = col sums = [5, 7, 9].
    // out3: seq(=2) reduced from [batch=2, seq=2, hidden=3]:
    //   b0 [[1,2,3],[4,5,6]] → [5,7,9];  b1 [[7,8,9],[10,11,12]] → [17,19,21].
    // out4: seq(=2) reduced from [batch=1, depth=1, seq=2, hidden=3] → [1,1,3] = [5,7,9].
    // outm: mean over seq(=2) of out3's input → [2.5,3.5,4.5 ; 8.5,9.5,10.5].
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out2", &[3], &[5.0, 7.0, 9.0]),
        ("out3", &[2, 3], &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]),
        ("out4", &[1, 1, 3], &[5.0, 7.0, 9.0]),
        ("outm", &[2, 3], &[2.5, 3.5, 4.5, 8.5, 9.5, 10.5]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        assert_eq!(got.2.len(), data.len(), "{name}: backend len ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }

    // Backend must agree with the evaluator oracle, value-for-value (#338).
    assert_eval_agrees_with_backend(source, "rank_poly_reduce_nonsquare", &backend);
}

/// The exact chelis#338 repro: a rank-poly named reduce called through a
/// concrete-rank caller evaluates under `chelis eval` and yields the same
/// numerics the C backend produces ([[1,2],[3,4]] summed over `seq` ->
/// [3, 7]). Before #338 this errored with "unknown runtime name `seq`".
#[test]
fn eval_resolves_named_axis_issue_repro() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use2(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = reduce_seq(x)\n\
         out = use2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n";
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "issue_338_repro");
    // A single-root program prints the bare value (no `out = ` prefix);
    // pin the exact line: shape AND the issue's expected numerics.
    assert_eq!(
        eval.trim(),
        "tensor(shape=[2], data=[3.0, 7.0])",
        "issue #338 repro: eval must yield the backend's numerics"
    );
}

/// Named-axis reductions at CONCRETE rank (no spread anywhere) share the same
/// eval gap and the same fix: the def call routes through the lowering lane,
/// which resolves `seq` against the declared (named) param dims. `max_reduce`,
/// `min_reduce`, and `prod_reduce` are included because at concrete rank they
/// are checkable and buildable (the chelis#340 Body-Discipline rejection
/// applies only inside `..r` bodies). `argmax_reduce`/`argmin_reduce` are
/// deliberately absent: the C backend mis-prints their int64 output as a
/// reinterpreted f32 bit pattern (chelis#347, pre-existing and orthogonal;
/// eval is correct), so the agreement oracle cannot include them yet. Fold
/// them in when #347 closes.
/// Operand is non-square (batch=2, seq=3) per the #258 red-team finding.
#[test]
fn concrete_rank_named_reduce_eval_matches_backend() {
    let source = "def sum_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, seq)\n\
         def max_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = max_reduce(x, seq)\n\
         def min_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = min_reduce(x, seq)\n\
         def prod_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = prod_reduce(x, seq)\n\
         outs = sum_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outx = max_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outn = min_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outp = prod_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    let backend = build_compile_run(source, "concrete_named_reduce");
    let tensors = parse_printed_tensors(&backend);
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("outs", &[2], &[6.0, 15.0]),
        ("outx", &[2], &[3.0, 6.0]),
        ("outn", &[2], &[1.0, 4.0]),
        ("outp", &[2], &[6.0, 120.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "concrete_named_reduce", &backend);
}

/// Eval-vs-backend parity on every host/tensor-lane boundary shape the
/// chelis#338 routing has to handle, in one program (one build + one eval):
///
/// - `lp`: a def with an unmarshalable `List` param alongside the tensor
///   (the def-call boundary declines; the body's reduction routes at the
///   reduction site with the frame param's declared type)
/// - `tl`: a named-axis reduction directly at a top-level root (operand
///   typed from the type-env entry of an earlier root)
/// - `blk`: a block body whose reduction operand is a local `let` binding
///   (typed from the checker's annotation on the bound expr)
/// - `al`: a local closure alias of a concrete wrapper (`g = use2`;
///   rank-poly callee routed with the argument's static type)
/// - `pp`: a pipe whose first stage reduces (`x |> sum(seq)`) inside a def
/// - `tp`: a root-level pipe chaining a bare Identity stage before the
///   reduction (`y |> relu |> sum(seq)`; the piped type threads through
///   shape-preserving stages)
/// - `outt`: a routed def declared to return a scalar (`-> f32`)
/// - `gr`: `grad` over a scalar-output def whose body reduces by name
#[test]
fn named_axis_eval_parity_corners() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use2(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = reduce_seq(x)\n\
         def lp(x: &tensor[batch, seq, f32], ys: List[f32]) -> tensor[batch, f32] = sum(x, seq)\n\
         def id2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n\
         def blk(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = {\n\
           y = relu(x)\n\
           sum(y, seq)\n\
         }\n\
         def al(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = {\n\
           g = use2\n\
           g(x)\n\
         }\n\
         def pp(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = x |> sum(seq)\n\
         def total(x: &tensor[seq, f32]) -> f32 = tensor_to_scalar(sum(x, seq))\n\
         y = id2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_lp = lp(to_tensor([[1.0, 2.0], [3.0, 4.0]]), [1.0, 2.0])\n\
         out_tl = sum(y, seq)\n\
         out_blk = blk(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_al = al(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_pp = pp(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_tp = y |> relu |> sum(seq)\n\
         outt = total(to_tensor([1.0, 2.0, 3.0]))\n\
         gr = grad(total)(to_tensor([1.0, 2.0, 3.0]))\n";
    let backend = build_compile_run(source, "named_axis_parity_corners");
    let tensors = parse_printed_tensors(&backend);
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out_lp", &[2], &[3.0, 7.0]),
        ("out_tl", &[2], &[3.0, 7.0]),
        ("out_blk", &[2], &[3.0, 7.0]),
        ("out_al", &[2], &[3.0, 7.0]),
        ("out_pp", &[2], &[3.0, 7.0]),
        ("out_tp", &[2], &[3.0, 7.0]),
        ("gr", &[3], &[1.0, 1.0, 1.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    // The scalar-return root prints as a plain scalar line, outside the
    // tensor parser: pin it on both lanes directly.
    assert!(
        backend.contains("outt = 6"),
        "backend must print the scalar return: {backend}"
    );
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "named_axis_parity_corners");
    assert!(
        eval.contains("outt = 6"),
        "eval must print the scalar return: {eval}"
    );
    assert_eval_agrees_with_backend(source, "named_axis_parity_corners", &backend);
}

/// chelis#351 (the #346 red-team F4 finding): `vmap` over a def that calls a
/// Tier-3 rank-poly named reduce checks clean and runs correctly on the C
/// backend, but `chelis eval` died with the dag.rs symbolic-dim ICE
/// ("symbolic dim `hidden` is referenced by a non-Load node"). Root cause:
/// `apply_transform` marshalled the batched actual as a placeholder with
/// all-Lit dims, while the inlined callee body kept its formal named dims —
/// vmap's rank shift (batched actual = formal rank + 1) defeats the same-rank
/// formal/actual remap (`tensor_dim_substitutions`, chelis#258 filter), so
/// `hidden` stayed unbound and no Load declared it. The fix types the
/// placeholder from the callee's declared formals (the chelis#338/#346
/// pattern for plain def calls): the batch axis stays Lit, the mapped axes
/// carry the formal's names, and the symbolic-dim machinery binds the body's
/// surviving names against the placeholder Load.
///
/// Matrix, all pinned eval-vs-backend:
/// - `out`: the issue reproducer (vmap over a rank-poly named-reduce callee)
/// - `outc`: control — vmap over a CONCRETE named reduce (worked before; must
///   keep working through the formal-typed placeholder path)
/// - `gr`: grad over a rank-poly named-reduce callee (same-rank remap already
///   concretized the dims; pinned so the lanes stay in lockstep)
/// - `gs`: vmap(grad(...)) over the same callee — the vmap rank shift hit the
///   identical guard (`Expand { size: Sym("hidden") }`), fixed by the same
///   placeholder typing
#[test]
fn vmap_over_rank_poly_named_reduce_evals_and_matches_backend() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def inner(x: &tensor[seq, hidden, f32]) -> tensor[hidden, f32] = reduce_seq(x)\n\
         def total(x: &tensor[seq, hidden, f32]) -> f32 = tensor_to_scalar(sum(reduce_seq(x), hidden))\n\
         def sum_seq(x: &tensor[seq, f32]) -> f32 = tensor_to_scalar(sum(x, seq))\n\
         out = vmap(inner)(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n\
         outc = vmap(sum_seq)(to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))\n\
         gr = grad(total)(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         gs = vmap(grad(total))(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n";
    assert_clean(
        &check_json(source),
        "#351 vmap rank-poly matrix checks clean",
    );
    let backend = build_compile_run(source, "vmap_rank_poly_named_reduce");
    let tensors = parse_printed_tensors(&backend);
    // out: per batch slice, sum over seq(=2) of [seq=2, hidden=3]:
    //   b0 [[1,2,3],[4,5,6]] -> [5,7,9];  b1 [[7,8,9],[10,11,12]] -> [17,19,21].
    // outc: row sums of [[1,2],[3,4],[5,6]] = [3,7,11] (the issue's control).
    // gr: d(sum of all elements)/dx = ones, shape [seq=2, hidden=3].
    // gs: gr vmapped over batch(=2) = ones, shape [2,2,3].
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out", &[2, 3], &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]),
        ("outc", &[3], &[3.0, 7.0, 11.0]),
        ("gr", &[2, 3], &[1.0; 6]),
        ("gs", &[2, 2, 3], &[1.0; 12]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        assert_eq!(got.2.len(), data.len(), "{name}: backend len ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "vmap_rank_poly_named_reduce", &backend);
}

/// chelis#351 second flavor (found by the #371 review red team): the SAME
/// vmap-lane ICE reproduced with dim-VAR formals (`tensor[a, seq, f32]` —
/// surf desugars single-letter dims to `d-var`). The original fix staged
/// d-vars as `Lit` (the chelis#346 F5 decision, correct for the same-rank
/// plain-call/grad lanes where `tensor_dim_substitutions` concretizes the
/// body's names), but vmap's rank shift skips that remap, so the body's
/// `Named("a", None)` stayed unbound and the dag.rs guard panicked — check
/// clean, backend correct, eval ICE: exactly the #351 symptom. The vmap lane
/// now stages d-vars as `Named(name, Some(size))` so they bind through the
/// placeholder Load like d-names; the plain-call lane keeps Lit staging
/// (pinned by `dim_var_formal_routes_and_matches_backend`).
///
/// Matrix, all pinned eval-vs-backend:
/// - `outa`: two d-vars surviving the named reduce (`[a, seq, b]` -> `[a, b]`)
/// - `outb`: one LEADING d-var (`[a, seq]` -> `[a]`; position before the
///   reduce axis is what the d-name reproducer did not cover)
/// - `outm`: mixed d-var + d-name formal (`[a, seq, hidden]` -> `[a, hidden]`)
/// - `gv`: vmap(grad(...)) over a d-var formal (gradient 2x, non-constant)
#[test]
fn vmap_over_dim_var_formal_named_reduce_evals_and_matches_backend() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def inner2(x: &tensor[a, seq, b, f32]) -> tensor[a, b, f32] = reduce_seq(x)\n\
         def inner1(x: &tensor[a, seq, f32]) -> tensor[a, f32] = reduce_seq(x)\n\
         def innerm(x: &tensor[a, seq, hidden, f32]) -> tensor[a, hidden, f32] = reduce_seq(x)\n\
         def totalv(x: &tensor[a, seq, f32]) -> f32 = tensor_to_scalar(sum(reduce_seq(x * x), 0))\n\
         outa = vmap(inner2)(to_tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]]))\n\
         outb = vmap(inner1)(to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))\n\
         outm = vmap(innerm)(to_tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]]))\n\
         gv = vmap(grad(totalv))(to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))\n";
    assert_clean(&check_json(source), "#351 vmap dim-var matrix checks clean");
    let backend = build_compile_run(source, "vmap_dim_var_named_reduce");
    let tensors = parse_printed_tensors(&backend);
    // outa/outm: per batch slice [a=2, seq=2, b|hidden=3], sum over seq:
    //   b0 -> [[5,7,9],[17,19,21]];  b1 -> [[29,31,33],[41,43,45]].
    // outb: per batch slice [a=2, seq=2], sum over seq (axis 1):
    //   [[1,2],[3,4]] -> [3,7];  [[5,6],[7,8]] -> [11,15].
    // gv: d(sum of squares)/dx = 2x per slice.
    let expected: &[(&str, &[usize], &[f64])] = &[
        (
            "outa",
            &[2, 2, 3],
            &[
                5.0, 7.0, 9.0, 17.0, 19.0, 21.0, 29.0, 31.0, 33.0, 41.0, 43.0, 45.0,
            ],
        ),
        ("outb", &[2, 2], &[3.0, 7.0, 11.0, 15.0]),
        (
            "outm",
            &[2, 2, 3],
            &[
                5.0, 7.0, 9.0, 17.0, 19.0, 21.0, 29.0, 31.0, 33.0, 41.0, 43.0, 45.0,
            ],
        ),
        (
            "gv",
            &[2, 2, 2],
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        ),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        assert_eq!(got.2.len(), data.len(), "{name}: backend len ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "vmap_dim_var_named_reduce", &backend);
}

/// chelis#351 non-zero-axis arm: `vmap(f, 1)` over the named rank-poly callee
/// must route through the same formal-typed placeholder synthesis (the batch
/// `Lit` is inserted at the vmap axis, between the formal's named dims), with
/// eval-vs-backend agreement. Before the fix this ICEd like the axis-0 form.
#[test]
fn vmap_axis_one_over_rank_poly_named_reduce_evals_and_matches_backend() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def inner(x: &tensor[seq, hidden, f32]) -> tensor[hidden, f32] = reduce_seq(x)\n\
         out = vmap(inner, 1)(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n\
         outz = vmap(inner)(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n";
    assert_clean(&check_json(source), "#351 vmap axis-1 checks clean");
    let backend = build_compile_run(source, "vmap_axis1_rank_poly");
    let tensors = parse_printed_tensors(&backend);
    // out: batch axis 1 (size 2); unbatched slices are x[:, b, :] typed
    // [seq=2, hidden=3], summed over seq:
    //   b0: [1,2,3]+[7,8,9] = [8,10,12];  b1: [4,5,6]+[10,11,12] = [14,16,18].
    // outz: the axis-0 twin on the same data (the issue reproducer's values).
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out", &[3, 2], &[8.0, 14.0, 10.0, 16.0, 12.0, 18.0]),
        ("outz", &[2, 3], &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "vmap_axis1_rank_poly", &backend);
}

/// Negative parity for chelis#351: a vmapped callee whose declared formal
/// CONFLICTS with the marshalled actual (declared `4` vs runtime `3`) must
/// stay a `DimensionMismatch` rejection on both surfaces — check rejects, and
/// eval surfaces the type error — never the dag.rs symbolic-dim ICE and never
/// a silent wrong answer. (The guard itself is untouched by the #351 fix; its
/// own positive/negative pins live in `chelis-ir/src/dag.rs` tests and the
/// #345 matrix in `issue_345_grad_dim_subst.rs`.)
#[test]
fn vmap_callee_dim_conflict_stays_rejected_not_ice() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def inner(x: &tensor[seq, 4, f32]) -> tensor[4, f32] = reduce_seq(x)\n\
         out = vmap(inner)(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n";
    assert_rejected_with(
        &check_json(source),
        "dimension mismatch",
        "#351 negative: conflicting concrete dim through the vmapped callee",
    );
    let dir = tempdir().expect("tempdir");
    let stderr = eval_stderr_expecting_failure(dir.path(), source, "vmap_dim_conflict");
    assert!(
        stderr.contains("DimensionMismatch") || stderr.contains("dimension mismatch"),
        "eval must surface the dimension mismatch, got: {stderr}"
    );
    assert!(
        !stderr.contains("internal compiler error"),
        "the dag.rs symbolic-dim ICE must not resurface: {stderr}"
    );
}

/// chelis#346 red-team blocker (F1/F2/F3): a UNARY elementwise op inside a
/// `..r` body before the named reduce must build, run, and eval with
/// agreeing numerics. Before the elementwise output-type fix, the unary
/// lowering arms took their output type from the body's `{type: ...}`
/// annotation, whose symbolic dims survive rank-poly inlining
/// unsubstituted: under eval `sum(exp(x), seq)` silently reduced the WRONG
/// axis (batch instead of seq), the relu variant aborted on a DAG shape
/// assert, and the C backend emitted garbage rank-0 output (pre-existing
/// since #337, masked because the corpus elementwise test was check-only).
/// Binary elementwise (`mul`) was already correct via
/// `elementwise_out_ty`; this pins the unary arms on the same contract.
/// Shape [2,3,4]: every axis size distinct (square operands mask
/// axis-mislabel bugs). exp inputs stay small so f32-vs-f64 agreement
/// holds at 1e-5 absolute tolerance.
#[test]
fn unary_elementwise_reduce_in_rank_poly_body_builds_runs_and_evals() {
    let source = "def core_exp(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(exp(x), seq)\n\
         def core_relu(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(relu(x), seq)\n\
         def core_neg(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(neg(x), seq)\n\
         def m_exp(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = core_exp(x)\n\
         def m_relu(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = core_relu(x)\n\
         def m_neg(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = core_neg(x)\n\
         out_exp = m_exp(to_tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], [[0.2, 0.4, 0.6, 0.8], [1.0, 0.1, 0.3, 0.5], [0.7, 0.9, 1.1, 0.2]]]))\n\
         out_relu = m_relu(to_tensor([[[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0], [9.0, 10.0, -11.0, 12.0]], [[13.0, -14.0, 15.0, -16.0], [-17.0, 18.0, -19.0, 20.0], [21.0, -22.0, 23.0, -24.0]]]))\n\
         out_neg = m_neg(to_tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]))\n";
    let backend = build_compile_run(source, "unary_elementwise_rank_poly");
    let tensors = parse_printed_tensors(&backend);
    // Hand-computed (exact arithmetic) for relu and neg; exp pinned by
    // shape + cross-lane agreement below.
    // relu: negatives zeroed, then sum over seq (axis 1):
    //   b0: [0+5+9, 2+0+10, 0+7+0, 4+0+12]   = [14, 12, 7, 16]
    //   b1: [13+0+21, 0+18+0, 15+0+23, 0+20+0] = [34, 18, 38, 20]
    // neg: -(sum over seq):
    //   b0: -[15, 18, 21, 24]; b1: -[51, 54, 57, 60]
    let expected: &[(&str, &[usize], &[f64])] = &[
        (
            "out_relu",
            &[2, 4],
            &[14.0, 12.0, 7.0, 16.0, 34.0, 18.0, 38.0, 20.0],
        ),
        (
            "out_neg",
            &[2, 4],
            &[-15.0, -18.0, -21.0, -24.0, -51.0, -54.0, -57.0, -60.0],
        ),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    let exp_out = tensors
        .iter()
        .find(|(n, _, _)| n == "out_exp")
        .unwrap_or_else(|| panic!("backend output missing `out_exp`: {backend}"));
    assert_eq!(
        exp_out.1,
        vec![2, 4],
        "out_exp: backend shape must be [batch=2, hidden=4] ({backend})"
    );
    // e^0.1 + e^0.5 + e^0.9 = 5.21349...: pins that the REDUCED axis is seq.
    assert!(
        (exp_out.2[0] - 5.213_495_24).abs() < 1e-4,
        "out_exp[0]: backend {} != e^0.1+e^0.5+e^0.9 (wrong axis reduced?)",
        exp_out.2[0]
    );
    assert_eval_agrees_with_backend_tol(source, "unary_elementwise_rank_poly", &backend, 1e-5);
}

/// chelis#346 red-team F5: a wrapper whose sig uses a single-letter dim
/// (surf desugars `a` to a dim VARIABLE `d-var`, not a named `d-name`)
/// must still route under eval: the staged placeholder monomorphizes the
/// var from the runtime shape exactly as a build call site binds it.
/// Spec SS4.2's own examples write `tensor[a, b, ...]` sigs, so this is a
/// realistic user shape, and before the fix it declined while build ran.
#[test]
fn dim_var_formal_routes_and_matches_backend() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def w(x: &tensor[a, seq, hidden, f32]) -> tensor[a, hidden, f32] = reduce_seq(x)\n\
         out = w(to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]))\n";
    let backend = build_compile_run(source, "dim_var_formal");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(out.1, vec![2, 2], "backend shape ({backend})");
    for (i, e) in [9.0, 12.0, 27.0, 30.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e}",
            out.2[i]
        );
    }
    // Single-root program: eval prints the bare value (no `out = `
    // prefix), so pin the exact line rather than the named-tensor parser.
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "dim_var_formal");
    assert_eq!(
        eval.trim(),
        "tensor(shape=[2, 2], data=[9.0, 12.0, 27.0, 30.0])",
        "dim-var formal: eval must route and match the backend"
    );
}

/// Known residual gap, pinned (chelis#346 red-team F6): a match-pattern
/// binding feeding a named reduce declines under eval with the targeted
/// chelis#338 diagnostic (pattern bindings carry no declared types), while
/// the backend builds and runs. Decline-not-wrong, like the permute-pipe
/// gap below; fold into the parity corners when pattern bindings learn
/// their checked types.
#[test]
fn match_pattern_operand_is_a_pinned_gap() {
    let source = "type Box =\n\
         \x20\x20| Wrap(tensor[batch, seq, f32])\n\
         def h(b: Box) -> tensor[batch, f32] = {\n\
         \x20\x20match b with {\n\
         \x20\x20\x20\x20| Wrap(v) => sum(v, seq)\n\
         \x20\x20}\n\
         }\n\
         out = h(Wrap(to_tensor([[1.0, 2.0], [3.0, 4.0]])))\n";
    let backend = build_compile_run(source, "match_pattern_gap");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(out.1, vec![2], "backend shape ({backend})");
    for (i, e) in [3.0, 7.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e}",
            out.2[i]
        );
    }
    let dir = tempdir().expect("tempdir");
    let stderr = eval_stderr_expecting_failure(dir.path(), source, "match_pattern_gap");
    assert!(
        stderr.contains("chelis#338") && stderr.contains("statically known tensor type"),
        "expected the targeted named-axis decline diagnostic, got: {stderr}"
    );
    assert!(
        !stderr.contains("unknown runtime name"),
        "the pre-#338 error must not resurface: {stderr}"
    );
}

/// Known residual gap, pinned: a *shape-rewriting* pipe stage (`permute`)
/// between the typed head and the named reduction drops the threaded type
/// (only Identity-class stages and def stages propagate it), so eval declines
/// with the targeted chelis#338 diagnostic while the backend builds and runs.
/// If this test starts failing because eval learned to handle it, delete the
/// decline assertion and fold the case into the parity corners above.
#[test]
fn pipe_rewriting_stage_then_named_reduce_is_a_pinned_gap() {
    let source = "def id2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n\
         y = id2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out = y |> permute(1, 0) |> sum(seq)\n";
    // Backend lane: green (permute carries the names through lowering).
    let backend = build_compile_run(source, "pipe_rewriting_gap");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(out.1, vec![2], "backend shape ({backend})");
    // permute flips to [seq, batch]; reducing `seq` (now axis 0) still
    // sums the same elements per batch: [1+2, 3+4].
    for (i, e) in [3.0, 7.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e}",
            out.2[i]
        );
    }
    // Eval lane: targeted decline, never the bare `unknown runtime name`.
    let dir = tempdir().expect("tempdir");
    let stderr = eval_stderr_expecting_failure(dir.path(), source, "pipe_rewriting_gap");
    assert!(
        stderr.contains("chelis#338") && stderr.contains("statically known tensor type"),
        "expected the targeted named-axis decline diagnostic, got: {stderr}"
    );
    assert!(
        !stderr.contains("unknown runtime name"),
        "the pre-#338 error must not resurface: {stderr}"
    );
}
