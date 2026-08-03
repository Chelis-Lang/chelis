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

/// chelis#340: the whole named-axis reduction family
/// (`max_reduce`/`min_reduce`/`prod_reduce`/`argmax_reduce`/`argmin_reduce`)
/// is now name-tracked in a `..r` body, exactly like `sum`/`mean`. Each
/// routes through the tensor-DAG kernel lane (the host-type inference in
/// `chelis-ir::host` types the call as a tensor so the wrapper extracts a
/// `__tensor_` helper instead of falling through to the host-emit
/// "unsupported builtin" path), so a `chelis check`-clean program builds.
/// This pins the check side; `max_reduce_family_builds_runs_in_rank_poly_body`
/// is the build+run eval-vs-C oracle that the host-lane gap is closed.
#[test]
fn max_reduce_family_name_tracked_in_rank_poly_body() {
    for op in ["max_reduce", "min_reduce", "prod_reduce"] {
        let json = check_json(&format!(
            "def m(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = {op}(x, seq)\n",
        ));
        assert_clean(&json, &format!("{op} name-tracked in a ..r body"));
    }
    // argmax/argmin return an int64 index tensor.
    for op in ["argmax_reduce", "argmin_reduce"] {
        let json = check_json(&format!(
            "def m(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, int64] = {op}(x, seq)\n",
        ));
        assert_clean(&json, &format!("{op} name-tracked in a ..r body"));
    }
}

/// chelis#340 build+run oracle (the converted ex-rejection test): the issue's
/// `reduce_seq`/`r3` program with `max_reduce`/`min_reduce`/`prod_reduce`/
/// `argmax_reduce`/`argmin_reduce` in a rank-poly body builds, compiles, runs,
/// and the C backend agrees with the `chelis eval` oracle value-for-value.
/// Before the fix these emitted non-compiling C via the host scalar lane
/// (`/* unsupported builtin max_reduce */ 0`, with the axis name `seq` leaking
/// as a bare identifier). Operands are NON-SQUARE (batch=2, seq=2, hidden=3 →
/// every distinct axis size) per the #258 red-team finding that a square
/// operand masks an axis-mislabel bug. The argmax/argmin index outputs are
/// pinned too (extreme element is in the second `seq` row, index 1, for the
/// monotone-increasing input; min is index 0).
#[test]
fn max_reduce_family_builds_runs_in_rank_poly_body() {
    let source = "def max_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = max_reduce(x, seq)\n\
         def min_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = min_reduce(x, seq)\n\
         def prod_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = prod_reduce(x, seq)\n\
         def amax_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, int64] = argmax_reduce(x, seq)\n\
         def amin_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, int64] = argmin_reduce(x, seq)\n\
         def rmax(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = max_seq(x)\n\
         def rmin(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = min_seq(x)\n\
         def rprod(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = prod_seq(x)\n\
         def ramax(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, int64] = amax_seq(x)\n\
         def ramin(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, int64] = amin_seq(x)\n\
         ox = rmax(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n\
         on = rmin(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n\
         op = rprod(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n\
         oax = ramax(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n\
         oan = ramin(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n";
    let backend = build_compile_run(source, "rank_poly_reduce_family");
    let tensors = parse_printed_tensors(&backend);

    // seq(=2) reduced from [batch=2, seq=2, hidden=3]:
    //   b0 [[1i64, 2i64, 3i64],[4i64, 5i64, 6i64]];  b1 [[7i64, 8i64, 9i64],[10i64, 11i64, 12i64]].
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("ox", &[2, 3], &[4.0, 5.0, 6.0, 10.0, 11.0, 12.0]),
        ("on", &[2, 3], &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]),
        ("op", &[2, 3], &[4.0, 10.0, 18.0, 70.0, 88.0, 108.0]),
        ("oax", &[2, 3], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ("oan", &[2, 3], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
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
    assert_eval_agrees_with_backend(source, "rank_poly_reduce_family", &backend);
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
        "def evil(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = reshape(x, [2i64, 3i64])\n",
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
        "def add_axis(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, one, 1i64)\n\
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
         def g(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, one, f32] = expand(x, one, 1i64)\n",
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
        "def bad(x: &tensor[..rest, f32]) -> tensor[..lo, c, ..hi, f32] = expand(x, c, 4i64)\n",
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
        "def bad(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, seq, ..post, seq, f32] = expand(x, seq, 5i64)\n",
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
        "def bad(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, 0, 1i64)\n",
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
/// `[2i64, 2i64, 3i64]`, actual `[2i64, 5i64, 3i64]`).
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
        "def add_axis(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, one, 1i64)\n\
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
/// (backend printed shape [5i64, 3i64] against a declared `[chan, seq]` = [2i64, 3i64]).
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
    let source = "def f(x: &tensor[c, seq, f32]) -> tensor[c, seq, c, f32] = expand(x, c, 4i64)\n\
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
    let source = "def add_axis(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, one, 1i64)\n\
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
    // out1: [seq=3] -> [3i64, 1i64], data unchanged.
    // out2: [batch=2, seq=3] -> [2i64, 3i64, 1i64], data unchanged.
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
/// interception must route a bare `expand(y, one, 1i64)` / `expand(y, c, 3,
/// seq)` / variadic `sum(y, batch, seq)` root through IR lowering — the
/// def-call tests above only exercise site B, so a site-A regression would
/// otherwise be invisible. Both lanes pinned value-for-value.
#[test]
fn top_level_named_expand_and_variadic_sum_eval_match_backend() {
    let source = "def id2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n\
         y = id2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_t = expand(y, one, 1i64)\n\
         out_a = expand(y, c, 3, seq)\n\
         out_vr = sum(y, batch, seq)\n";
    let backend = build_compile_run(source, "top_level_named_axis_ops");
    let tensors = parse_printed_tensors(&backend);
    // out_t: trailing insert -> [2i64, 2i64, 1i64], data unchanged.
    // out_a: c=3 inserted before seq (axis 1) -> [2i64, 3i64, 2i64], rows tripled.
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
/// uniform weights; max is idempotent across orders). Non-square [2i64, 3i64, 4i64]
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
         out_v = vmap(vinner)(y)\n";
    // chelis#383 (CLOSED): `out_v` now uses the shared top-level `y`
    // BINDING, not an inline literal. `vmap(vinner)(y)` with a binding-typed
    // Load + a two-stage named reduce to a scalar (the variadic surface
    // desugars to the `sum(sum(x, head), seq)` composition) previously ICE'd
    // the dag.rs symbolic-dim guard in the C-build lane ("symbolic dim `seq`
    // referenced by a non-Load node") — verified at d786744. Wave-1's
    // by-position named-axis recovery re-validation (chelis#549) closed it;
    // this line is the regression lock (build + run + eval agreement over the
    // once-ICEing binding-operand form). The single-stage and inline-literal
    // forms are locked separately in
    // `issue_383_vmap_two_stage_named_reduce_regression_matrix`.
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
        // chelis#383: over the shared top-level `y` binding ([2, 3, 4]), the
        // per-batch sums are 1..=12=78 and 13..=24=222 (was [10i64, 26i64] on the
        // old inline [2, 2, 2] literal).
        ("out_v", &[2], &[78.0, 222.0]),
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

/// chelis#383 regression matrix. `vmap(f)(y)` where `y` is a top-level
/// BINDING (not an inline literal) and `f` performs a two-stage named reduce
/// to a scalar previously ICE'd the dag.rs symbolic-dim guard in the C-build
/// lane ("symbolic dim `seq` referenced by a non-Load node") — a binding-typed
/// Load + a two-stage named reduce, verified at d786744. Wave-1's by-position
/// named-axis recovery re-validation (chelis#549) closed the forward case; this
/// pins it green across the build+run+eval oracle AND locks the single-stage
/// and inline-literal forms (the two that were "unaffected" and must stay so).
///
/// FD note: the forward reduce cases carry no gradient, so their oracle is
/// build-vs-eval agreement on exact per-slice sums. The grad+vmap case below
/// (`vmap(grad(vsq))`) is the finite-difference twin: grad of `sum(x^2)` is
/// `2 x`, exactly the central-difference gradient. Its EVAL result matches
/// `2 y`. Its C-BUILD lane still ICEs on the same symbolic-dim guard via the
/// grad-backward `Expand { size: Sym("seq") }` over a monomorphized (concrete)
/// Load — a distinct grad-build symbolic-dim gap (chelis#513 family, the
/// grad+vmap interaction), NOT the forward #383 case this locks. Recorded as a
/// residual; the eval lane is correct today.
#[test]
fn issue_383_vmap_two_stage_named_reduce_regression_matrix() {
    // Forward lane: all three build + run + eval-agree. `y` is a top-level
    // binding ([2i64, 2i64, 2i64]); per-slice sum of [1i64, 2i64, 3i64, 4i64]=10 and [5i64, 6i64, 7i64, 8i64]=26.
    let forward = "def vsingle(x: &tensor[seq, head, f32]) -> f32 = tensor_to_scalar(sum(x, seq, head))\n\
         def vtwostage(x: &tensor[seq, head, f32]) -> f32 = tensor_to_scalar(sum(sum(x, head), seq))\n\
         y = to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])\n\
         out_two_binding = vmap(vtwostage)(y)\n\
         out_single_binding = vmap(vsingle)(y)\n\
         out_inline = vmap(vtwostage)(to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))\n";
    let backend = build_compile_run(forward, "issue_383_forward");
    let tensors = parse_printed_tensors(&backend);
    let expected: &[(&str, &[usize], &[f64])] = &[
        // chelis#383 headline: the once-ICEing two-stage + top-level-binding form.
        ("out_two_binding", &[2], &[10.0, 26.0]),
        // Regression locks: single-stage and inline-literal must stay green.
        ("out_single_binding", &[2], &[10.0, 26.0]),
        ("out_inline", &[2], &[10.0, 26.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(forward, "issue_383_forward", &backend);

    // FD twin (eval lane): grad of a two-stage `sum(x^2)` reduce, vmapped over
    // the top-level binding, must equal `2 y` (== the central-difference
    // gradient). Eval only — the C-build lane of this grad+vmap form has a
    // separate symbolic-dim-grad residual (see doc comment).
    let grad_src = "def vsq(x: &tensor[seq, head, f32]) -> f32 = tensor_to_scalar(sum(sum(mul(x, x), head), seq))\n\
         y = to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])\n\
         out_grad_two = vmap(grad(vsq))(y)\n";
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), grad_src, "issue_383_grad");
    let eval_tensors = parse_printed_tensors(&eval);
    let (_, shape, data) = eval_tensors
        .iter()
        .find(|(n, _, _)| n == "out_grad_two")
        .unwrap_or_else(|| panic!("eval missing `out_grad_two`: {eval}"));
    assert_eq!(shape, &vec![2, 2, 2], "grad+vmap shape ({eval})");
    // 2 * y (central-difference gradient of sum(x^2)).
    let want = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    for (i, (g, w)) in data.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "grad+vmap elem {i}: eval {g} != 2*y {w} (finite-difference twin)"
        );
    }
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

/// Build `source` to C and return the generated `<name>.c` file contents
/// (chelis#469 codegen-determinism oracle). Each call is an independent
/// `chelis build` subprocess, so two calls exercise two fresh HashMap seeds —
/// the condition under which the pre-fix non-deterministic tensor-kernel input
/// ordering surfaced.
fn build_c_source(source: &str, name: &str) -> String {
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
        "build must succeed; stderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    fs::read_to_string(out_dir.join(format!("{name}.c"))).expect("read generated C")
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
            // [05-OBS-4] (chelis#732 P1): eval renders a rank-0 root as its
            // bare element, so a labeled `name = <number>` line is a rank-0
            // tensor twin for the agreement oracle. The compiled lane keeps
            // the wrapper until Phase 2.
            if let Some((name, payload)) = line.split_once(" = ")
                && let Ok(value) = payload.trim().parse::<f64>()
            {
                out.push((name.trim().to_string(), Vec::new(), vec![value]));
            }
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

    // out2: seq(=2) reduced from [seq=2, hidden=3] → [hidden=3] = col sums = [5i64, 7i64, 9i64].
    // out3: seq(=2) reduced from [batch=2, seq=2, hidden=3]:
    //   b0 [[1i64, 2i64, 3i64],[4i64, 5i64, 6i64]] → [5i64, 7i64, 9i64];  b1 [[7i64, 8i64, 9i64],[10i64, 11i64, 12i64]] → [17i64, 19i64, 21i64].
    // out4: seq(=2) reduced from [batch=1, depth=1, seq=2, hidden=3] → [1i64, 1i64, 3i64] = [5i64, 7i64, 9i64].
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
/// numerics the C backend produces ([[1i64, 2i64],[3i64, 4i64]] summed over `seq` ->
/// [3i64, 7i64]). Before #338 this errored with "unknown runtime name `seq`".
#[test]
fn eval_resolves_named_axis_issue_repro() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use2(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = reduce_seq(x)\n\
         out = use2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n";
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "issue_338_repro");
    // [05-OBS-6] labels every root, including a single root. Pin the exact
    // line: root identity, shape, and the issue's expected numerics.
    assert_eq!(
        eval.trim(),
        "out = tensor(shape=[2i64], data=[3.0, 7.0])",
        "issue #338 repro: eval must yield the backend's numerics"
    );
}

/// Named-axis reductions at CONCRETE rank (no spread anywhere) share the same
/// eval gap and the same fix: the def call routes through the lowering lane,
/// which resolves `seq` against the declared (named) param dims. `max_reduce`,
/// `min_reduce`, and `prod_reduce` are included because at concrete rank they
/// are checkable and buildable (the chelis#340 Body-Discipline rejection
/// applies only inside `..r` bodies). `argmax_reduce`/`argmin_reduce` are now
/// folded in too: chelis#347 closed (the C backend prints their int64 output
/// correctly instead of as a reinterpreted f32 bit pattern), so the agreement
/// oracle covers them — `argmax`/`argmin` over a row return the index of the
/// extreme element (eval and backend now agree).
/// Operand is non-square (batch=2, seq=3) per the #258 red-team finding.
#[test]
fn concrete_rank_named_reduce_eval_matches_backend() {
    let source = "def sum_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, seq)\n\
         def max_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = max_reduce(x, seq)\n\
         def min_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = min_reduce(x, seq)\n\
         def prod_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = prod_reduce(x, seq)\n\
         def amax_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, int64] = argmax_reduce(x, seq)\n\
         def amin_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, int64] = argmin_reduce(x, seq)\n\
         outs = sum_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outx = max_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outn = min_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outp = prod_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outax = amax_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outan = amin_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    let backend = build_compile_run(source, "concrete_named_reduce");
    let tensors = parse_printed_tensors(&backend);
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("outs", &[2], &[6.0, 15.0]),
        ("outx", &[2], &[3.0, 6.0]),
        ("outn", &[2], &[1.0, 4.0]),
        ("outp", &[2], &[6.0, 120.0]),
        // argmax/argmin over each row: both rows are ascending, so the max
        // is at index 2 and the min at index 0 (chelis#347 closed — int64
        // indices now print correctly in the C backend).
        ("outax", &[2], &[2.0, 2.0]),
        ("outan", &[2], &[0.0, 0.0]),
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
    //   b0 [[1i64, 2i64, 3i64],[4i64, 5i64, 6i64]] -> [5i64, 7i64, 9i64];  b1 [[7i64, 8i64, 9i64],[10i64, 11i64, 12i64]] -> [17i64, 19i64, 21i64].
    // outc: row sums of [[1i64, 2i64],[3i64, 4i64],[5i64, 6i64]] = [3i64, 7i64, 11i64] (the issue's control).
    // gr: d(sum of all elements)/dx = ones, shape [seq=2, hidden=3].
    // gs: gr vmapped over batch(=2) = ones, shape [2i64, 2i64, 3i64].
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
    //   b0 -> [[5i64, 7i64, 9i64],[17i64, 19i64, 21i64]];  b1 -> [[29i64, 31i64, 33i64],[41i64, 43i64, 45i64]].
    // outb: per batch slice [a=2, seq=2], sum over seq (axis 1):
    //   [[1i64, 2i64],[3i64, 4i64]] -> [3i64, 7i64];  [[5i64, 6i64],[7i64, 8i64]] -> [11i64, 15i64].
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
    //   b0: [1i64, 2i64, 3i64]+[7i64, 8i64, 9i64] = [8i64, 10i64, 12i64];  b1: [4i64, 5i64, 6i64]+[10i64, 11i64, 12i64] = [14i64, 16i64, 18i64].
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
/// Shape [2i64, 3i64, 4i64]: every axis size distinct (square operands mask
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
    //   b0: [0+5+9, 2+0+10, 0+7+0, 4+0+12]   = [14i64, 12i64, 7i64, 16i64]
    //   b1: [13+0+21, 0+18+0, 15+0+23, 0+20+0] = [34i64, 18i64, 38i64, 20i64]
    // neg: -(sum over seq):
    //   b0: -[15i64, 18i64, 21i64, 24i64]; b1: -[51i64, 54i64, 57i64, 60i64]
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
    // [05-OBS-6] labels every root, including a single root, so pin the exact
    // line rather than weakening this to a value-only comparison.
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "dim_var_formal");
    assert_eq!(
        eval.trim(),
        "out = tensor(shape=[2i64, 2i64], data=[9.0, 12.0, 27.0, 30.0])",
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

/// PR #800 closes the former chelis#338 pinned gap: authoritative owner
/// inference stamps the shape-rewriting `permute` stage with its named output
/// dims, so the following named reduction resolves `seq` in eval just as the
/// backend does. Keep this in the executable parity corpus; a future loss of
/// the stage stamp must fail as an eval-vs-backend divergence, not be accepted
/// as a targeted decline.
#[test]
fn pipe_rewriting_stage_then_named_reduce_eval_matches_backend() {
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
    assert_eval_agrees_with_backend(source, "pipe_rewriting_gap", &backend);
}

/// chelis#388: a named-axis reduction over a *literal-shaped* operand
/// (`to_tensor([[...]])`) must build and eval, not die in rank
/// monomorphization. The call site binds the formal parameter
/// `x: tensor[batch, seq]` to a literal node whose dims are concrete
/// `[Lit(2), Lit(3)]` — the named axis `seq` is erased — so the by-name
/// reduce-axis lookup against the operand's dims fails; the fix recovers
/// the axis from the formal-parameter position recorded at the inline site.
///
/// `reduce_seq` reduces `seq` (axis 1, size 3) of a non-square `[2i64, 3i64]`
/// operand, so a position mislabel (reducing axis 0 instead) would yield a
/// `[3i64]` shape of column sums `[5i64, 7i64, 9i64]` rather than the correct `[2i64]`
/// row sums `[6i64, 15i64]`. The shapes differ, so the error cannot hide.
#[test]
fn named_reduce_over_literal_operand_builds_runs_evals() {
    // Two bindings so both lanes print the `out = ` prefix the
    // parse_printed_tensors / agreement oracle keys on (a single-root
    // program prints the bare value with no name prefix).
    let source = "def reduce_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, seq)\n\
         src = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         out = reduce_seq(src)\n";
    let backend = build_compile_run(source, "issue_388_named_reduce_literal");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(out.1, vec![2], "backend reduced the wrong axis ({backend})");
    for (i, e) in [6.0, 15.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
    // The locked invariant: a check-clean program must build AND the
    // backend must agree with the evaluator, value-for-value.
    assert_eval_agrees_with_backend(source, "issue_388_named_reduce_literal", &backend);
}

/// chelis#388 (eval lane): the exact-line oracle for the named reduction
/// over a literal operand, including [05-OBS-6]'s required root label.
#[test]
fn named_reduce_over_literal_operand_eval_exact() {
    let source = "def reduce_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, seq)\n\
         out = reduce_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "issue_388_eval_exact");
    assert_eq!(
        eval.trim(),
        "out = tensor(shape=[2i64], data=[6.0, 15.0])",
        "issue #388: named reduce over a literal operand must eval the seq-axis sums"
    );
}

/// chelis#388 NEGATIVE: a named axis the operand genuinely lacks is still a
/// loud failure, never a silent default. `reduce_seq` declares `seq` but the
/// body reduces an undeclared `chan`; the checker rejects it, so the
/// position-recovery fix never masks a real missing-axis bug.
#[test]
fn named_reduce_unknown_axis_still_rejected() {
    let json = check_json(
        "def reduce_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, chan)\n",
    );
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        !errors.is_empty(),
        "reducing an undeclared named axis must be rejected: {json}"
    );
}

/// chelis#364: a reduction whose axis is `cast(N, int32)` — a form the
/// checker admits as a compile-time-constant axis — must lower to axis N in
/// the DAG/backend lane, not silently to axis 0. Pre-fix, `extract_axis_raw`
/// returned 0 for any non-literal axis expr, so `sum(x, cast(1, int32))`
/// reduced axis 0 (wrong numerics, a shape contradicting the checked type,
/// and — under grad — eval and backend agreeing on the SAME wrong gradient).
///
/// Non-square `[2i64, 3i64]`: axis 1 (`cast(1, int32)`) sums to `[6i64, 15i64]`; a
/// silent axis-0 default would produce a `[3i64]` shape of `[5i64, 7i64, 9i64]`.
#[test]
fn cast_axis_reduction_lowers_to_named_axis_not_zero() {
    let source = "def f(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, cast(1, int32))\n\
         src = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         out = f(src)\n";
    let backend = build_compile_run(source, "issue_364_cast_axis");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(
        out.1,
        vec![2],
        "cast-axis reduction lowered the wrong axis ({backend})"
    );
    for (i, e) in [6.0, 15.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
    // Eval is the reference; the backend must match it byte-for-byte on this
    // integer-indexed, exact-arithmetic reduction.
    assert_eval_agrees_with_backend(source, "issue_364_cast_axis", &backend);
}

/// chelis#364 (axis-2 cross-rank control): a `cast(2, int32)` axis on a
/// rank-3 operand reduces the *third* axis, proving the fix carries the
/// constant through rather than clamping to a fixed axis.
#[test]
fn cast_axis_reduction_axis_two_rank_three() {
    let source = "def f(x: &tensor[a, b, c, f32]) -> tensor[a, b, f32] = sum(x, cast(2, int32))\n\
         src = to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])\n\
         out = f(src)\n";
    let backend = build_compile_run(source, "issue_364_cast_axis_two");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    // [1, 3, 2] reduced over axis 2 -> [1, 3] = [[3, 7, 11]].
    assert_eq!(
        out.1,
        vec![1, 3],
        "cast(2) reduced the wrong axis ({backend})"
    );
    for (i, e) in [3.0, 7.0, 11.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
    assert_eval_agrees_with_backend(source, "issue_364_cast_axis_two", &backend);
}

/// chelis#383: `vmap` over a two-stage named reduce-to-scalar with a
/// *top-level-binding* operand previously tripped the `dag.rs`
/// symbolic-dim guard at build/eval-lowering time (`symbolic dim `seq` is
/// referenced by a non-Load node ... but no Load input declares it`). The
/// root cause was the vmap dim-symbol remap zipping the rank-N formal
/// parameter type against the rank-(N+1) vmap argument: the `same-rank`
/// filter dropped the substitution, so the inlined body's named dims
/// (`seq`, `head`) never resolved to the operand's concrete dims and a
/// `Named("seq")` dim survived in the vmapped reduce with no declaring
/// Load. The fix batch-prepends the formal parameter types before the
/// remap so the ranks align.
///
/// This is the EXACT form the prior `variadic_reduce_builds_runs_and_evals`
/// fixture documented as a workaround (it used an inline-literal operand to
/// avoid the ICE); the workaround is now obsolete for the eval lane. Pins
/// the evaluator oracle (the reference lane per the backend-numerics
/// discipline): `sum(sum(slice, head), seq)` per 2x2 slice = the slice sum,
/// so `[[[1i64, 2i64],[3i64, 4i64]], [[5i64, 6i64],[7i64, 8i64]]]` -> `[10i64, 26i64]`.
#[test]
fn vmap_two_stage_named_reduce_top_level_binding_evals() {
    let source = "def vinner(x: &tensor[seq, head, f32]) -> f32 = tensor_to_scalar(sum(sum(x, head), seq))\n\
         y = to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])\n\
         out = vmap(vinner)(y)\n";
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "issue_383_vmap_two_stage");
    let tensors = parse_printed_tensors(&eval);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("eval output missing `out`: {eval}"));
    assert_eq!(out.1, vec![2], "vmap two-stage reduce shape ({eval})");
    for (i, e) in [10.0, 26.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: eval {} != {e} ({eval})",
            out.2[i]
        );
    }
}

/// chelis#383 (build lane): the vmap-over-two-stage-named-reduce that
/// previously tripped the `dag.rs` symbolic-dim ICE at build time now builds,
/// runs, and the C backend agrees with the evaluator. The IR-level fix (the
/// vmap dim-symbol remap batch-prepend) removed the ICE; once the downstream
/// C-identifier hygiene landed on main the build lane completes end to end.
/// Pins the full compile-run-eval agreement so the IR fix cannot silently
/// regress the build lane.
#[test]
fn vmap_two_stage_named_reduce_top_level_binding_builds_and_matches_backend() {
    let source = "def vinner(x: &tensor[seq, head, f32]) -> f32 = tensor_to_scalar(sum(sum(x, head), seq))\n\
         y = to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])\n\
         out = vmap(vinner)(y)\n";
    let backend = build_compile_run(source, "issue_383_vmap_build");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(
        out.1,
        vec![2],
        "vmap two-stage reduce build shape ({backend})"
    );
    for (i, e) in [10.0, 26.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
    assert_eval_agrees_with_backend(source, "issue_383_vmap_build", &backend);
}

/// chelis#383 cross-lane (lowering x C-identifier hygiene, #379/#467): the
/// vmap-over-two-stage-named-reduce build lane must stay correct when the
/// captured top-level binding's name is a C KEYWORD (`static`). This
/// exercises BOTH the #383 IR fix (the vmap dim-symbol remap that removed
/// the dag.rs symbolic-dim ICE) AND `c_ident`'s `chelis_user__` mangle: the
/// keyword binding is declared, referenced into the vmap helper's arg slot,
/// and printed under the mangled name consistently, while the `__tensor_arg*`
/// temps stay unmangled (no `chelis_user____tensor_` double-prefix). A
/// regression in either lane breaks compilation or the run, so the
/// compile-run-eval agreement is the cross-lane guard. (Repro contributed by
/// WS-2B; verified `chelis_user__static` appears at decl/ref/print with no
/// double-prefix.)
#[test]
fn vmap_two_stage_named_reduce_keyword_binding_builds_and_matches_backend() {
    let source = "def vinner(x: &tensor[seq, head, f32]) -> f32 = tensor_to_scalar(sum(sum(x, head), seq))\n\
         static = to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])\n\
         out = vmap(vinner)(static)\n";
    let backend = build_compile_run(source, "issue_383_vmap_keyword_binding");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(
        out.1,
        vec![2],
        "vmap two-stage reduce over a keyword-named binding ({backend})"
    );
    for (i, e) in [10.0, 26.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
    assert_eval_agrees_with_backend(source, "issue_383_vmap_keyword_binding", &backend);
}

/// chelis#384/#397 (A): a §4.7.2 Form-3 runtime `expand` size sourced from a
/// `shape(tensor, axis)` read must produce C that AGREES with the evaluator.
/// This is the spec's own canonical example (`bias_broadcast` from
/// `examples/illustrative/runtime_shape_semantics.ch`), which the C backend
/// previously mis-compiled: it read the new axis extent from the expand
/// OPERAND (`b`, size 4) instead of the `shape(x, 0)` source (`x`, size 2),
/// emitting `[4i64, 4i64]` against the evaluator's `[2i64, 4i64]`. The fix keeps the
/// shape-source operand `x` live (a `shape_dep`) so the symbolic dim `n`
/// binds from `x`'s shape. Non-square `[2i64, 4i64]` so an operand/source mixup
/// changes the shape and cannot hide.
#[test]
fn form3_shape_sourced_expand_matches_backend() {
    let source = "def bias_broadcast(x: &tensor[n, 4, f32], b: &tensor[4, f32]) -> tensor[n, 4, f32] = expand(b, 0, shape(x, cast(0, int64)))\n\
         xs = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])\n\
         bs = to_tensor([10.0, 20.0, 30.0, 40.0])\n\
         out = bias_broadcast(xs, bs)\n";
    let backend = build_compile_run(source, "issue_397_shape_sourced_expand");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    // The broadcast extent `n` comes from `x`'s axis 0 (=2), NOT from `b`
    // (=4); a source mixup would print shape `[4, 4]`.
    assert_eq!(
        out.1,
        vec![2, 4],
        "Form-3 shape-sourced expand bound the extent to the wrong tensor ({backend})"
    );
    // The backend must agree with the evaluator, value-for-value AND
    // shape-for-shape (the #338 oracle): the original divergence was a
    // shape disagreement, so assert_eval_agrees_with_backend pins it.
    assert_eval_agrees_with_backend(source, "issue_397_shape_sourced_expand", &backend);
}

/// chelis#384 (B): a §4.7.2 Form-3 runtime `expand` size that is a bare
/// runtime scalar parameter (`k: int32`) with NO tensor source is rejected
/// loudly at lowering, not silently mis-compiled. Pre-fix the C backend read
/// the extent from an out-of-range operand axis (`x` is rank 1; the codegen
/// read `inputs[0i64]->shape[1i64]`), emitting a garbage shape that disagreed with
/// the evaluator's `[2i64, 3i64]`. There is no tensor whose shape carries the
/// extent, so the form has no backend representation and must reject.
#[test]
fn form3_scalar_param_expand_size_rejected() {
    let source = "def f(x: &tensor[seq, f32], k: int32) -> tensor[seq, chan, f32] = expand(x, 1, k)\n\
         out = f(to_tensor([1.0, 2.0]), 3)\n";
    let stderr = build_expecting_failure(source, "issue_384_scalar_param_expand");
    assert!(
        stderr.contains("expand")
            && stderr.contains("no tensor in scope carries it")
            && stderr.contains("chelis#469"),
        "expected the Form-3 sourceless-size reject diagnostic citing #469, got: {stderr}"
    );
    // The reject must be a clean diagnostic, never the internal-compiler-error
    // ICE the sourceless symbol previously triggered downstream.
    assert!(
        !stderr.contains("internal compiler error"),
        "sourceless Form-3 expand size must reject cleanly, not ICE: {stderr}"
    );
}

/// chelis#384/#397 (B): the eval lane rejects the sourceless Form-3 expand
/// size identically to the backend — no eval-vs-backend divergence. Pre-fix
/// eval computed a (correct) result while the backend silently diverged;
/// both lanes now reject the unsupported form with the same diagnostic.
#[test]
fn form3_scalar_param_expand_size_rejected_in_eval() {
    let source = "def bcast[a, n](g: tensor[n, f32], a_dim: int64) -> tensor[a, n, f32] = expand(g, 0, a_dim)\n\
         out = bcast(to_tensor([1.0, 2.0]), cast(3, int64))\n";
    let dir = tempdir().expect("tempdir");
    let stderr = eval_stderr_expecting_failure(dir.path(), source, "issue_397_eval_reject");
    assert!(
        stderr.contains("expand")
            && stderr.contains("no tensor in scope carries it")
            && stderr.contains("chelis#469"),
        "eval must reject the sourceless Form-3 expand size with the same \
         #469 diagnostic as the backend, got: {stderr}"
    );
}

/// chelis#384/#397 (A) liveness lock: a `shape(x, axis)`-sourced Form-3
/// expand inside a `vmap`ped def must STILL bind the extent to the correct
/// tensor after `vmap`'s `vectorize_axis0` rebuild. The fix records the
/// shape source as a `DagNode::shape_deps` liveness edge; every DAG-rebuild
/// pass (DCE, copy/drop insertion, BLAS specialization, fusion, grad, vmap,
/// splice, CSE) must preserve it, or the source `Load` is dead-code-
/// eliminated and the wrong-shape regression returns SILENTLY. This test
/// drives the dep through the vmap rebuild specifically; `bias_broadcast`
/// already drives it through specialization. Per-slice: `x` row is rank-1
/// `[3i64]` (n=3), `b` is a scalar broadcast to `[3i64]`, so the extent is read
/// from `x`'s shape — out `[2i64, 3i64]`, C == eval.
#[test]
fn form3_shape_dep_survives_vmap_rebuild() {
    let source = "def bcast(x: &tensor[n, f32], b: &tensor[f32]) -> tensor[n, f32] = expand(b, 0, shape(x, cast(0, int64)))\n\
         xs = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         bs = to_tensor([10.0, 20.0])\n\
         out = vmap(bcast)(xs, bs)\n";
    let backend = build_compile_run(source, "shape_dep_vmap");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    // n (=3) comes from each x-slice's shape; a dropped shape_dep would
    // mis-bind it (the original silent wrong-shape bug) or fail to build.
    assert_eq!(
        out.1,
        vec![2, 3],
        "shape_dep was dropped across the vmap rebuild ({backend})"
    );
    let expected = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0];
    for (i, e) in expected.iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
    assert_eval_agrees_with_backend(source, "shape_dep_vmap", &backend);
}

/// chelis#397 (check↔build symmetry): the EXACT 0.7.26-regression repro —
/// `bcast_1d_to_2d`, a positional-axis Form-3 `expand` whose size is a bare
/// runtime scalar parameter (`a_dim: int64`) with no tensor source — must be
/// rejected at CHECK time, not just at build/eval. Pre-fix, `chelis check`
/// returned score 1.0 / empty errors while build and eval rejected it with the
/// #469 sourceless-size diagnostic: a check-clean program that does not build,
/// the precise invariant violation #397 reports (`rank_poly_tier3.rs:158-161`).
/// The fix moves the sourceless-runtime-scalar rejection into the type checker
/// so all three lanes agree by construction. The paired build/eval rejections
/// are pinned by `form3_scalar_param_expand_size_rejected{,_in_eval}` above.
#[test]
fn form3_scalar_param_expand_size_rejected_at_check() {
    let json = check_json(
        "def bcast_1d_to_2d[a, n](g: tensor[n, f32], a_dim: int64) -> tensor[a, n, f32] = expand(g, 0, a_dim)\n",
    );
    assert_rejected_with(
        &json,
        "no tensor in scope carries it",
        "Form-3 sourceless runtime-scalar expand size at check (rank-1 -> rank-2)",
    );
    assert_rejected_with(
        &json,
        "chelis#469",
        "check rejection cites the #469 sourceless-size rule",
    );
}

/// chelis#397 (check↔build symmetry, chained rank-1 -> rank-4): the full
/// `broadcast_to_achw` repro from the issue — three chained positional-axis
/// Form-3 `expand`s, each sized by a bare runtime `int64` parameter
/// (`h_dim`/`w_dim`/`a_dim`) with no tensor source, with the explicit
/// `step1`/`step2`/`step3` type ascriptions. This is the rank-1 -> rank-4 NN
/// broadcast pattern (school's `broadcast_to_achw`) that previously
/// type-checked clean but died in build/eval. It must now be rejected at check
/// time with the same #469 diagnostic — the first chained-expand member of the
/// "check-clean must build" invariant family.
#[test]
fn form3_chained_rank4_expand_rejected_at_check() {
    let json = check_json(
        "def broadcast_to_achw[c, h, w, a](v: &tensor[c, f32], h_dim: int64, w_dim: int64, a_dim: int64) -> tensor[a, c, h, w, f32] = {\n\
        \x20 step1: tensor[c, h, f32] = expand(v, 1, h_dim)\n\
        \x20 step2: tensor[c, h, w, f32] = expand(step1, 2, w_dim)\n\
        \x20 step3: tensor[a, c, h, w, f32] = expand(step2, 0, a_dim)\n\
        \x20 step3\n\
        }\n",
    );
    assert_rejected_with(
        &json,
        "no tensor in scope carries it",
        "chained Form-3 sourceless expand at check (rank-1 -> rank-4)",
    );
    assert_rejected_with(
        &json,
        "chelis#469",
        "chained-expand check rejection cites the #469 sourceless-size rule",
    );
}

/// chelis#397 (build/eval consistency, chained rank-1 -> rank-4): the chained
/// `broadcast_to_achw` repro is rejected IDENTICALLY at build and at eval with
/// the #469 sourceless-size diagnostic — no eval-vs-backend divergence and no
/// rank-monomorphization ICE (the original 0.7.26 failure mode was the internal
/// `tensor rank mismatch: 1 dims vs 4 dims`). A call site instantiates the def
/// so the chained expands are actually lowered. With the check-time rejection
/// landed, build/eval reject before lowering; this test pins that the loud
/// #469 reason survives across both lanes regardless of where it fires.
#[test]
fn form3_chained_rank4_expand_rejected_in_build_and_eval() {
    let source = "def broadcast_to_achw[c, h, w, a](v: &tensor[c, f32], h_dim: int64, w_dim: int64, a_dim: int64) -> tensor[a, c, h, w, f32] = {\n\
        \x20 step1: tensor[c, h, f32] = expand(v, 1, h_dim)\n\
        \x20 step2: tensor[c, h, w, f32] = expand(step1, 2, w_dim)\n\
        \x20 step3: tensor[a, c, h, w, f32] = expand(step2, 0, a_dim)\n\
        \x20 step3\n\
        }\n\
        out = broadcast_to_achw(to_tensor([1.0, 2.0]), cast(3, int64), cast(4, int64), cast(5, int64))\n";
    let build_stderr = build_expecting_failure(source, "issue_397_chained_achw_build");
    assert!(
        build_stderr.contains("no tensor in scope carries it")
            && build_stderr.contains("chelis#469"),
        "chained rank-1 -> rank-4 expand must reject at build with the #469 \
         sourceless-size diagnostic, got: {build_stderr}"
    );
    assert!(
        !build_stderr.contains("rank mismatch")
            && !build_stderr.contains("internal compiler error"),
        "the chained expand must reject cleanly, never the original \
         rank-monomorphization ICE: {build_stderr}"
    );
    let dir = tempdir().expect("tempdir");
    let eval_stderr =
        eval_stderr_expecting_failure(dir.path(), source, "issue_397_chained_achw_eval");
    assert!(
        eval_stderr.contains("no tensor in scope carries it") && eval_stderr.contains("chelis#469"),
        "chained rank-1 -> rank-4 expand must reject at eval with the same #469 \
         diagnostic as the build lane, got: {eval_stderr}"
    );
}

// ── chelis#397 source-tracking: positive (materializable) acceptance ──────
//
// The reworked §4.7.2 expand-size predicate discriminates by PROVENANCE, not
// spelling: a runtime size is accepted at check iff it provably folds to a
// constant (`Static`) or derives from an in-scope tensor's `shape(t, axis)`
// read (`ShapeSourced`) — followed transitively through `let`, `cast`, and
// integer arithmetic — and rejected otherwise. These pin the materializable
// cases the prior spelling-based predicate over-rejected (#397 BLOCKER 2/3).

/// chelis#397 (BLOCKER 3): a `let`-bound shape-derived expand size is
/// semantically identical to the inline `shape(x, ...)` form
/// (`form3_shape_sourced_expand_matches_backend`) and must be ACCEPTED at
/// check — the prior spelling-based predicate wrongly rejected it ("no tensor
/// in scope carries it" while `x` plainly does). Source-tracking follows the
/// `let` binding `a_dim = shape(x, cast(0, int32))` back to `x`'s shape, so
/// the size is `ShapeSourced`. The evaluator materializes the extent from `x`
/// (axis 0 = 2), producing `[2i64, 4i64]`.
#[test]
fn form3_let_bound_shape_sourced_expand_accepted_at_check() {
    let source = "def f(x: &tensor[n, 4, f32], b: &tensor[4, f32]) -> tensor[n, 4, f32] = {\n\
        \x20 a_dim: int32 = shape(x, cast(0, int32))\n\
        \x20 expand(b, 0, a_dim)\n\
        }\n\
        xs = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])\n\
        out = f(xs, to_tensor([10.0, 20.0, 30.0, 40.0]))\n";
    let json = check_json(source);
    assert_clean(
        &json,
        "let-bound shape-derived expand size accepted at check (#397 BLOCKER 3)",
    );
    // The evaluator materializes the extent from `x`'s axis 0 (=2), NOT `b`.
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "issue_397_let_bound_shape_eval");
    let tensors = parse_printed_tensors(&eval);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("eval output missing `out`: {eval}"));
    assert_eq!(
        out.1,
        vec![2, 4],
        "let-bound shape-derived expand must materialize the extent from `x` (=2), got {eval}"
    );
}

/// chelis#397 (BLOCKER 2): an arithmetic-built positive runtime size whose
/// operands are all compile-time constants (`some_count = sub(cast(4,
/// int32), cast(1, int32))` = 3) folds to a constant and is therefore a
/// materializable `Static` extent — it must be ACCEPTED at check, not
/// rejected as "sourceless". The evaluator computes the arithmetic and
/// produces shape `[3i64, 2i64]`. This is the sibling of the host-runtime defense
/// path exercised by `chelis-compiler-api`'s
/// `host_runtime_expand_negative_count_errors` (which builds a non-positive
/// arithmetic count the SAME way): both rely on static-arithmetic sizes
/// passing check so the runtime/eval lane can compute them. The C backend now
/// const-folds inline / def-scoped static arithmetic to a concrete extent
/// (chelis#469; the build+agreement oracle is
/// `form3_static_arithmetic_expand_size_matches_backend`). This test's
/// TOP-LEVEL-NAMED spelling (`some_count = sub(...)`) is a residual: top-level
/// binding provenance is not threaded into IR lowering, so it still rejects at
/// build (fail-closed, never a silent miscompile). The check layer's job here
/// is to stop over-rejecting a materializable extent.
#[test]
fn form3_static_arithmetic_expand_size_accepted_at_check() {
    let source = "b = to_tensor([1.0, 2.0])\n\
        some_count = sub(cast(4, int32), cast(1, int32))\n\
        out = expand(b, cast(0, int32), some_count)\n";
    let json = check_json(source);
    assert_clean(
        &json,
        "static-arithmetic expand size accepted at check (#397 BLOCKER 2)",
    );
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "issue_397_static_arith_eval");
    let tensors = parse_printed_tensors(&eval);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("eval output missing `out`: {eval}"));
    assert_eq!(
        out.1,
        vec![3, 2],
        "static-arithmetic expand size (4-1=3) must evaluate to shape [3i64, 2i64], got {eval}"
    );
}

// ── chelis#469: Form-3 runtime expand-size RESOLUTION (build lane) ──
//
// The check-layer tests above pin that these materializable sizes are
// ACCEPTED at check and EVALUATE correctly. #469 closes the C-codegen
// materialization gap so each also BUILDS and the compiled C AGREES with the
// evaluator (the #338 eval-vs-backend oracle). Fail-closed parity: a size the
// backend cannot yet materialize (arithmetic that COMBINES a shape source with
// another term) must still reject loudly, never silently emit a wrong extent.

/// chelis#469 (Case 1, let-bound shape source): the `let`-bound
/// `shape(x, axis)` form — `a_dim = shape(x, 0); expand(b, 0, a_dim)` —
/// now BUILDS and the compiled C agrees with the evaluator. Pre-#469 it
/// type-checked and evaluated (`form3_let_bound_shape_sourced_expand_accepted_at_check`)
/// but the C `build` rejected the residual `Sym("a_dim")` at lowering (a
/// check-accept / build-reject asymmetry). The fix resolves the size through
/// the `let` indirection to `x`'s shape source (recording the `shape_dep` that
/// keeps `x` live), producing the same DAG as the inline
/// `form3_shape_sourced_expand_matches_backend`. Extent `n` (=2) comes from
/// `x` axis 0, NOT `b`; a mixup would print `[4i64, 4i64]`.
#[test]
fn form3_let_bound_shape_sourced_expand_matches_backend() {
    let source = "def f(x: &tensor[n, 4, f32], b: &tensor[4, f32]) -> tensor[n, 4, f32] = {\n\
        \x20 a_dim: int32 = shape(x, cast(0, int32))\n\
        \x20 expand(b, 0, a_dim)\n\
        }\n\
        xs = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])\n\
        out = f(xs, to_tensor([10.0, 20.0, 30.0, 40.0]))\n";
    let backend = build_compile_run(source, "issue_469_let_bound_shape");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(
        out.1,
        vec![2, 4],
        "let-bound shape-sourced expand must bind the extent to `x` (=2), got ({backend})"
    );
    assert_eval_agrees_with_backend(source, "issue_469_let_bound_shape", &backend);
}

/// chelis#469 RT-3 (let-to-let alias of a shape name): an intermediate `let`
/// alias of a shape-bound name — `a = shape(x, 0); c = a; expand(b, 0, c)`,
/// and the `cast`-wrapped `c = cast(a, int32); expand(b, 0, cast(c, int64))` —
/// is check-clean and eval-correct and must now also BUILD with C agreeing
/// with the evaluator. Pre-RT-3 the shape recovery followed a name bound
/// DIRECTLY to `shape(...)` (plus a use-site `cast`) but not through a
/// `let`-to-`let` alias, so build rejected #469 (a check-accept/build-reject
/// asymmetry the static path did not have). `resolve_shape_binding_source`
/// threads the alias, recording the underlying `shape(x, 0)` app so the extent
/// binds to `x` (=2), NOT `b`.
#[test]
fn form3_shape_alias_expand_matches_backend() {
    for (variant, body) in [
        (
            "bare_alias",
            "def f(x: &tensor[n, 4, f32], b: &tensor[4, f32]) -> tensor[n, 4, f32] = {\n\
            \x20 a: int32 = shape(x, cast(0, int32))\n\
            \x20 c: int32 = a\n\
            \x20 expand(b, 0, c)\n\
            }\n",
        ),
        (
            "cast_alias",
            "def f(x: &tensor[n, 4, f32], b: &tensor[4, f32]) -> tensor[n, 4, f32] = {\n\
            \x20 a: int32 = shape(x, cast(0, int32))\n\
            \x20 c: int32 = cast(a, int32)\n\
            \x20 expand(b, 0, cast(c, int64))\n\
            }\n",
        ),
    ] {
        let source = format!(
            "{body}\
             xs = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])\n\
             out = f(xs, to_tensor([10.0, 20.0, 30.0, 40.0]))\n"
        );
        let name = format!("issue_469_shape_alias_{variant}");
        let backend = build_compile_run(&source, &name);
        let tensors = parse_printed_tensors(&backend);
        let out = tensors
            .iter()
            .find(|(n, _, _)| n == "out")
            .unwrap_or_else(|| panic!("[{variant}] backend output missing `out`: {backend}"));
        assert_eq!(
            out.1,
            vec![2, 4],
            "[{variant}] aliased shape-sourced expand must bind the extent to `x` (=2) ({backend})"
        );
        assert_eval_agrees_with_backend(&source, &name, &backend);
    }
}

/// chelis#469 RT-3 axis-discriminator (correct-source SAFETY): the alias
/// recovery must bind the `shape_dep` to the ACTUAL source tensor AND axis,
/// never a neighbouring axis or the expand operand. A NON-SQUARE source
/// `x: tensor[2i64, 3i64]` (axis 0 = 2, axis 1 = 3) with the size aliased from
/// `shape(x, 1)` through a `let` must produce `[3i64, 4i64]` — a wrong-axis
/// resolution would give `[2i64, 4i64]` and a wrong-source (operand `b`) resolution
/// `[4i64, 4i64]`. Build and eval must agree, so any mis-resolution surfaces as
/// C != eval rather than a silently plausible-but-wrong shape.
#[test]
fn form3_shape_alias_axis_discriminator_matches_backend() {
    let source = "def f(x: &tensor[2, 3, f32], b: &tensor[4, f32]) -> tensor[three, 4, f32] = {\n\
        \x20 a: int32 = shape(x, cast(1, int32))\n\
        \x20 c: int32 = a\n\
        \x20 expand(b, 0, c)\n\
        }\n\
        xs = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
        out = f(&xs, to_tensor([10.0, 20.0, 30.0, 40.0]))\n";
    let backend = build_compile_run(source, "issue_469_shape_alias_axis");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(
        out.1,
        vec![3, 4],
        "aliased `shape(x, 1)` (=3) must bind axis 1 of `x`, not axis 0 (=2) or `b` (=4) ({backend})"
    );
    assert_eval_agrees_with_backend(source, "issue_469_shape_alias_axis", &backend);
}

/// chelis#469 RT-3 shadowing safety: an alias of a shape name that is then
/// RE-BOUND to a sourceless scalar (`a = shape(x, 0); c = a; c = k`) must drop
/// the stale shape provenance — the later `expand(b, 0, c)` is sourceless and
/// must be REJECTED (at check, and fail-closed at build), never recover the
/// stale extent. Pins that `resolve_shape_binding_source` clears the alias
/// entry on re-bind, mirroring the direct-name rebind
/// (`form3_shape_to_sourceless_rebind_expand_rejected_at_check`).
#[test]
fn form3_shape_alias_rebound_to_sourceless_rejected_at_check() {
    let json = check_json(
        "def f(x: &tensor[n, 4, f32], b: &tensor[4, f32], k: int32) -> tensor[n, 4, f32] = {\n\
        \x20 a: int32 = shape(x, cast(0, int32))\n\
        \x20 c: int32 = a\n\
        \x20 c: int32 = k\n\
        \x20 expand(b, 0, c)\n\
        }\n",
    );
    assert_rejected_with(
        &json,
        "no tensor in scope carries it",
        "alias re-bound to a sourceless scalar must drop shape provenance (chelis#469 RT-3)",
    );
    assert_rejected_with(&json, "chelis#469", "RT-3 alias rebind cites #469");
}

/// chelis#469 (Case 1, static arithmetic): a size built from all-constant
/// integer arithmetic (`sub(cast(4, int32), cast(1, int32))` = 3) now
/// const-folds to a concrete extent and BUILDS, with C agreeing with the
/// evaluator. Pre-#469 this SILENTLY miscompiled: check + eval produced
/// `[3i64, 2i64]` while the C backend defaulted the unresolved size to extent 1 and
/// emitted `[1i64, 2i64]` — a live eval-vs-backend divergence the fail-closed
/// invariant forbids. (Inline form; the top-level-NAMED spelling
/// `some_count = sub(...)` still rejects at build — top-level binding
/// tracking is a documented residual, fail-closed.)
#[test]
fn form3_static_arithmetic_expand_size_matches_backend() {
    let source = "b = to_tensor([1.0, 2.0])\n\
        out = expand(b, cast(0, int32), sub(cast(4, int32), cast(1, int64)))\n";
    let backend = build_compile_run(source, "issue_469_static_arith");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(
        out.1,
        vec![3, 2],
        "static-arithmetic expand size (4-1=3) must build to shape [3i64, 2i64], not the \
         pre-#469 silent extent-1 [1i64, 2i64] ({backend})"
    );
    assert_eval_agrees_with_backend(source, "issue_469_static_arith", &backend);
}

/// chelis#469 (Case 1, let-bound static): a `let`-bound STATIC size
/// (`k = cast(3, int32); expand(b, 0, cast(k, int64))`) folds to its value
/// through the `let` and `cast` in IR lowering, matching the §4.7.2
/// `SizeClass::Static` "followed transitively through `let` bindings" contract.
/// Pre-#469 the cast-wrapped form silently defaulted to extent 1 (eval
/// `[3i64, 3i64]` vs C `[1i64, 3i64]`); now the C `build` lane resolves it and emits the
/// correct extent + values.
///
/// BUILD-ONLY oracle: unlike the shape-sourced forms, `chelis eval` on this
/// exact form is blocked by a SEPARATE, PRE-EXISTING checker gap — the checker
/// annotates a shape-sensitive `expand` app's `type:` metadata for
/// ShapeSourced sizes but not for `let`-bound Static sizes, so the eval
/// lowering's `assert_ir_typed` rejects it ("shape-sensitive IR app nodes must
/// carry explicit type metadata"). That is a chelis-types annotation issue,
/// independent of this #469 IR/backend lowering work; the DAG-level fold is
/// additionally pinned by `issue_369_expand_let_bound_non_shape_does_not_recover`.
#[test]
fn form3_let_bound_static_expand_size_builds_correct_extent() {
    let source = "def f(b: &tensor[3, f32]) -> tensor[3, 3, f32] = {\n\
        \x20 k: int32 = cast(3, int32)\n\
        \x20 expand(b, 0, cast(k, int64))\n\
        }\n\
        b = to_tensor([7.0, 8.0, 9.0])\n\
        out = f(&b)\n";
    let backend = build_compile_run(source, "issue_469_let_bound_static");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(
        out.1,
        vec![3, 3],
        "let-bound static expand size (k=3) must build to [3i64, 3i64], not [1i64, 3i64] ({backend})"
    );
    // Each row is the broadcast of b = [7i64, 8i64, 9i64]; a wrong extent would change
    // the row count or the value pattern.
    let expected = [7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0];
    for (i, e) in expected.iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
}

/// chelis#469 fail-closed parity: an expand size that is integer arithmetic
/// COMBINING a `shape(tensor, axis)` read with another term
/// (`mul(shape(x, 0), cast(2, int32))`) is admitted at check as `ShapeSourced`
/// but has no single tensor axis the backend can read the extent from and no
/// `DimExpr` representation for the arithmetic. It must REJECT loudly at build
/// with the #469 diagnostic — NEVER the pre-fix silent extent-1 default
/// (eval `[4i64, 4i64]` vs C `[1i64, 4i64]`). This is the negative twin of
/// `form3_static_arithmetic_expand_size_matches_backend`: all-constant
/// arithmetic folds and builds; arithmetic that touches a runtime shape does
/// not (yet) and rejects rather than miscompiles.
#[test]
fn form3_arith_over_shape_expand_size_rejected_at_build() {
    let source = "def f(x: &tensor[n, 4, f32], b: &tensor[4, f32]) -> tensor[m, 4, f32] = expand(b, 0, mul(shape(x, cast(0, int32)), cast(2, int64)))\n\
        xs = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])\n\
        out = f(xs, to_tensor([10.0, 20.0, 30.0, 40.0]))\n";
    let stderr = build_expecting_failure(source, "issue_469_arith_over_shape");
    assert!(
        stderr.contains("cannot")
            && stderr.contains("materialize")
            && stderr.contains("chelis#469"),
        "arithmetic-over-shape expand size must reject at build with the #469 \
         materialization diagnostic, got: {stderr}"
    );
    assert!(
        !stderr.contains("internal compiler error"),
        "arith-over-shape expand size must reject cleanly, not ICE: {stderr}"
    );
}

/// chelis#469 codegen determinism: the spec's canonical Form-3 example
/// (`bias_broadcast` from `examples/illustrative/runtime_shape_semantics.ch`)
/// must build to BYTE-IDENTICAL C across independent `chelis build`
/// invocations. The shape-source operand `x` is referenced ONLY via
/// `shape(x, …)`, so its `Load` is pre-created alongside `b`'s in
/// `lower_subexpr_program_inner`; that pre-creation iterated a `HashMap`
/// (per-process-random order), flipping the tensor-kernel input slots
/// (`inputs[0i64]`/`inputs[1i64]`) build-to-build — a codegen-determinism-invariant
/// violation the `bias_broadcast` oracle could otherwise never assert
/// "byte-identical". Sorting the pre-creation by name makes the kernel ABI
/// stable. Two independent subprocess builds (each a fresh HashMap seed) must
/// emit identical `.c`.
#[test]
fn form3_bias_broadcast_c_is_byte_deterministic() {
    let source = "def bias_broadcast(x: &tensor[n, 4, f32], b: &tensor[4, f32]) -> tensor[n, 4, f32] = expand(b, 0, shape(x, cast(0, int64)))\n\
        xs = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])\n\
        out = bias_broadcast(xs, to_tensor([10.0, 20.0, 30.0, 40.0]))\n";
    let first = build_c_source(source, "issue_469_determinism_a");
    let second = build_c_source(source, "issue_469_determinism_b");
    assert_eq!(
        first, second,
        "Form-3 shape-sourced expand C must be byte-identical across builds"
    );
}

/// chelis#469 axis-side separation (rlronan's shared-walker constraint, the
/// #364 sibling): the extent-side static-arithmetic fold must NOT leak into
/// the reduction/softmax/gather AXIS path. `extract_int_for_dim` (the shared
/// axis walker) is deliberately left un-widened; the fold lives only in
/// `fold_static_size` on the extent path. A static-arithmetic reduction axis
/// (`sum(x, sub(cast(2, int32), cast(1, int32)))`) therefore stays REJECTED at
/// check ("axis must be a compile-time constant or a named axis") — proving the
/// extent fold did not silently widen the axis contract.
#[test]
fn static_arith_reduction_axis_still_rejected_at_check() {
    let json = check_json(
        "def f(x: &tensor[2, 3, f32]) -> tensor[2, f32] = sum(x, sub(cast(2, int32), cast(1, int32)))\n",
    );
    assert_rejected_with(
        &json,
        "axis must be a compile-time constant or a named axis",
        "static-arithmetic reduction axis must stay rejected at check (chelis#469/#364 \
         extent-vs-axis separation)",
    );
}

/// chelis#397 (chained rank-1 -> rank-4, shape-sourced): the chained
/// `broadcast_to_achw`-shaped pattern is ACCEPTED at check when every
/// inserted-axis size is a `shape(src, axis)` read of an in-scope tensor —
/// the materializable counterpart to the sourceless chained form rejected by
/// `form3_chained_rank4_expand_rejected_at_check`. Source-tracking applies
/// uniformly to each of the three chained expands.
#[test]
fn form3_chained_rank4_shape_sourced_expand_accepted_at_check() {
    let json = check_json(
        "def bcast4[c, h, w, a](v: &tensor[c, f32], src: &tensor[a, c, h, w, f32]) -> tensor[a, c, h, w, f32] = {\n\
        \x20 step1: tensor[c, h, f32] = expand(v, 1, shape(src, cast(2, int64)))\n\
        \x20 step2: tensor[c, h, w, f32] = expand(step1, 2, shape(src, cast(3, int64)))\n\
        \x20 step3: tensor[a, c, h, w, f32] = expand(step2, 0, shape(src, cast(0, int64)))\n\
        \x20 step3\n\
        }\n",
    );
    assert_clean(
        &json,
        "chained rank-1 -> rank-4 shape-sourced expand accepted at check (#397)",
    );
}

// ── chelis#397 source-tracking: negative (sourceless) rejection, uniform ──
//
// A truly-sourceless runtime size — a bare `int32`/`int64` parameter, a
// `cast`/arithmetic over one, or a `let` bound to such — has no backend
// representation and must be REJECTED at check with the #469 diagnostic,
// UNIFORMLY across spellings. The bare-`var` spelling is already pinned by
// `form3_scalar_param_expand_size_rejected_at_check`; these pin the spellings
// the prior predicate (which keyed on the bare-`(var)` form only) let escape.

/// chelis#397 (MAJOR 4): the `cast`-wrapped sourceless form
/// `expand(g, 0, cast(a_dim, int64))` — which the prior spelling-based
/// predicate let ESCAPE check entirely and then silently miscompile in C —
/// must now be REJECTED at check with the #469 diagnostic. Source-tracking
/// strips the `cast` and finds the bare runtime scalar `a_dim` underneath.
#[test]
fn form3_cast_wrapped_sourceless_expand_rejected_at_check() {
    let json = check_json(
        "def g[a, n](b: tensor[n, f32], a_dim: int32) -> tensor[a, n, f32] = expand(b, 0, cast(a_dim, int64))\n",
    );
    assert_rejected_with(
        &json,
        "no tensor in scope carries it",
        "cast-wrapped sourceless expand size rejected at check (#397 MAJOR 4)",
    );
    assert_rejected_with(
        &json,
        "chelis#469",
        "cast-wrapped sourceless rejection cites the #469 sourceless-size rule",
    );
}

/// chelis#397: a `let`-bound sourceless size — `d = a_dim` where `a_dim` is a
/// bare runtime scalar parameter — must be REJECTED at check, identically to
/// the bare-`var` form. Source-tracking follows the `let` binding to its
/// sourceless RHS; binding it to a name does not give it a tensor source.
#[test]
fn form3_let_bound_sourceless_expand_rejected_at_check() {
    let json = check_json(
        "def g[a, n](b: tensor[n, f32], a_dim: int32) -> tensor[a, n, f32] = {\n\
        \x20 d: int32 = a_dim\n\
        \x20 expand(b, 0, d)\n\
        }\n",
    );
    assert_rejected_with(
        &json,
        "no tensor in scope carries it",
        "let-bound sourceless expand size rejected at check (#397)",
    );
    assert_rejected_with(
        &json,
        "chelis#469",
        "let-bound sourceless rejection cites the #469 sourceless-size rule",
    );
}

/// chelis#397: integer arithmetic that TOUCHES a sourceless runtime scalar is
/// itself sourceless (`Sourceless` is absorbing) — `add(a_dim, cast(1,
/// int32))` cannot be materialized because `a_dim` has no shape source. It
/// must be REJECTED at check, distinguishing it from the all-constant
/// arithmetic accepted by `form3_static_arithmetic_expand_size_accepted_at_check`.
#[test]
fn form3_arithmetic_over_sourceless_expand_rejected_at_check() {
    let json = check_json(
        "def g[a, n](b: tensor[n, f32], a_dim: int32) -> tensor[a, n, f32] = expand(b, 0, add(a_dim, cast(1, int64)))\n",
    );
    assert_rejected_with(
        &json,
        "no tensor in scope carries it",
        "arithmetic over a sourceless scalar rejected at check (#397)",
    );
    assert_rejected_with(
        &json,
        "chelis#469",
        "arithmetic-over-sourceless rejection cites the #469 sourceless-size rule",
    );
}

// ── chelis#397 source-tracking: re-review soundness blockers (A/B/C) ──────
//
// A fresh issues+soundness re-review found three execution-confirmed holes
// where the predicate accepted a sourceless size it should reject. Each is a
// silent-miscompile or check↔build/eval divergence — the exact class #469 /
// #397 exist to prevent — so each must reject at CHECK with the #469 reason.

/// chelis#397 (BLOCKER A): a function-call-derived inline size
/// `expand(b, 0, ident(a_dim))` (with `def ident(x: int32) -> int32 = x`)
/// must be REJECTED at check. Pre-fix it classified as `Unknown` and reached
/// `check_expand_signature`'s non-rejecting `_` arm: check-clean AND
/// build-clean, with the C backend emitting a hardcoded extent-1 axis (eval
/// `[3i64, 2i64]` vs compiled-C `[1i64, 2i64]`) — the same silent-miscompile class as
/// MAJOR 4. A non-int-arithmetic `app` is a runtime value with no shape
/// source, so it is `Sourceless`. The `cast`- and `add`-wrapped forms reduce
/// to the same call and must reject identically.
#[test]
fn form3_function_call_inline_expand_size_rejected_at_check() {
    let bare = check_json(
        "def ident(x: int32) -> int32 = x\n\
        def g[a, n](b: tensor[n, f32], a_dim: int32) -> tensor[a, n, f32] = expand(b, 0, ident(a_dim))\n",
    );
    assert_rejected_with(
        &bare,
        "no tensor in scope carries it",
        "function-call inline expand size rejected at check (#397 BLOCKER A)",
    );
    assert_rejected_with(&bare, "chelis#469", "BLOCKER A bare form cites #469");

    let cast_wrapped = check_json(
        "def ident(x: int32) -> int32 = x\n\
        def g[a, n](b: tensor[n, f32], a_dim: int32) -> tensor[a, n, f32] = expand(b, 0, cast(ident(a_dim), int32))\n",
    );
    assert_rejected_with(
        &cast_wrapped,
        "no tensor in scope carries it",
        "cast-wrapped function-call inline expand size rejected at check (#397 BLOCKER A)",
    );

    let arith_wrapped = check_json(
        "def ident(x: int32) -> int32 = x\n\
        def g[a, n](b: tensor[n, f32], a_dim: int32) -> tensor[a, n, f32] = expand(b, 0, add(ident(a_dim), cast(0, int64)))\n",
    );
    assert_rejected_with(
        &arith_wrapped,
        "no tensor in scope carries it",
        "arith-wrapped function-call inline expand size rejected at check (#397 BLOCKER A)",
    );
}

/// chelis#397 (BLOCKER B): a name that re-binds from a shape source to a
/// sourceless RHS — `len = shape(x, 0); len = k; expand(b, 0, len)` — must be
/// REJECTED at check. Pre-fix the provenance map was add-only (no clear on
/// the sourceless arm), so the stale `ShapeSourced` entry survived the
/// re-bind: check ACCEPTED, eval materialized a runtime extent from `k` that
/// contradicts the checked type (a check↔eval divergence and type-soundness
/// violation). The clear-on-rebind restores add/clear symmetry.
#[test]
fn form3_shape_to_sourceless_rebind_expand_rejected_at_check() {
    let json = check_json(
        "def f(x: &tensor[n, 4, f32], b: &tensor[4, f32], k: int32) -> tensor[n, 4, f32] = {\n\
        \x20 len: int32 = shape(x, cast(0, int32))\n\
        \x20 len: int32 = k\n\
        \x20 expand(b, 0, len)\n\
        }\n",
    );
    assert_rejected_with(
        &json,
        "no tensor in scope carries it",
        "shape-sourced -> sourceless rebind rejected at check (#397 BLOCKER B)",
    );
    assert_rejected_with(&json, "chelis#469", "BLOCKER B cites #469");
}

/// chelis#397 (BLOCKER C): a sourceless value parameter that SHADOWS an outer
/// shape-sourced name — `d = shape(&xs, 0)` at top level, then `def f(..., d:
/// int32) = expand(g, 0, d)` — must be REJECTED at check. Pre-fix `Env`'s
/// derived `Clone` copied the outer `d`'s `ShapeSourced` provenance into the
/// function scope and the value parameter did not clear it: check ACCEPTED
/// while build/eval rejected #469 — the exact check-clean-fails-build #397
/// class this PR set out to kill. Clearing provenance at every param bind
/// fixes it.
#[test]
fn form3_sourceless_param_shadowing_shape_name_rejected_at_check() {
    let json = check_json(
        "xs = to_tensor([[1.0, 2.0], [3.0, 4.0]])\n\
        d = shape(&xs, cast(0, int32))\n\
        def f[m, q](g: tensor[q, f32], d: int32) -> tensor[m, q, f32] = expand(g, 0, d)\n",
    );
    assert_rejected_with(
        &json,
        "no tensor in scope carries it",
        "sourceless param shadowing an outer shape name rejected at check (#397 BLOCKER C)",
    );
    assert_rejected_with(&json, "chelis#469", "BLOCKER C cites #469");
}

/// chelis#397 (BLOCKER B/C positive parity): the clear-on-rebind /
/// clear-on-param-bind must NOT over-reject the legitimate inverse. A name
/// that re-binds the OTHER way — sourceless then shape-sourced — is
/// materializable, so `expand(b, 0, len)` after `len = shape(x, 0)` (re-bound
/// over an earlier sourceless `len`) stays ACCEPTED. Pins that the provenance
/// clear is scoped to the offending arm and does not poison a later valid
/// shape-source rebind.
#[test]
fn form3_sourceless_to_shape_rebind_expand_accepted_at_check() {
    let json = check_json(
        "def f(x: &tensor[n, 4, f32], b: &tensor[4, f32], k: int32) -> tensor[n, 4, f32] = {\n\
        \x20 len: int32 = k\n\
        \x20 len: int32 = shape(x, cast(0, int32))\n\
        \x20 expand(b, 0, len)\n\
        }\n",
    );
    assert_clean(
        &json,
        "sourceless -> shape-sourced rebind accepted at check (#397 BLOCKER B inverse)",
    );
}

/// chelis#364/#319 regression: a NEGATIVE axis literal (`softmax(x, -1)`,
/// `sum(x, -1)`) must lower to the last axis. The #364 axis-extraction fix
/// (FATAL on a non-constant axis) initially over-rejected the `-1` form,
/// which Surf desugars to `(app (var neg) (lit 1))` — `extract_int_for_dim`
/// returns `None` for that shape — and broke the cross-module SDPA grad
/// control (`softmax(mul(scores, scale), -1)` in
/// `issue_319_grad_crossmodule_precision_poly_attn`). `extract_int_axis` now
/// resolves the negative-axis desugar, so `-1` normalizes to the last axis.
///
/// `softmax(x, -1)` over a `[2i64, 3i64]` operand must equal `softmax(x, 1)` (the
/// explicit last axis); a uniform row keeps the result exact across the
/// eval-f64 / backend-f32 lanes.
#[test]
fn negative_axis_softmax_resolves_to_last_axis() {
    let source = "def sm(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = softmax(x, -1)\n\
         src = to_tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])\n\
         out = sm(src)\n";
    let backend = build_compile_run(source, "neg_axis_softmax");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(out.1, vec![2, 3], "softmax(-1) shape ({backend})");
    // Last-axis softmax of a uniform row is uniform 1/3; an axis mislabel
    // (e.g. -1 -> 0, the pre-#364 silent behavior) would normalize down the
    // batch axis instead and give a different distribution.
    for (i, e) in [1.0 / 3.0; 6].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
    assert_eval_agrees_with_backend(source, "neg_axis_softmax", &backend);
}

/// chelis#364/#319 regression (reduction lane): `sum(x, -1)` lowers to the
/// last axis in the C backend. (The host evaluator separately rejects a
/// negative reduction axis — a pre-existing host-interpreter limitation, not
/// part of this fix — so this pins the BUILD lane only, where the negative
/// axis must normalize correctly rather than FATAL-error or mis-reduce.)
#[test]
fn negative_axis_reduce_lowers_to_last_axis_in_backend() {
    let source = "def red(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, -1)\n\
         src = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         out = red(src)\n";
    let backend = build_compile_run(source, "neg_axis_reduce");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    // Last-axis sum: row sums [1+2+3, 4+5+6] = [6i64, 15i64]. A -1 -> 0 mislabel
    // would sum down the batch axis to [5, 7, 9].
    assert_eq!(out.1, vec![2], "sum(-1) reduced the wrong axis ({backend})");
    for (i, e) in [6.0, 15.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e} ({backend})",
            out.2[i]
        );
    }
}

/// Parse the [05-OBS-6] labelled `name = tensor(shape=[...], data=[...])`
/// line `eval` prints for a top-level root.
fn parse_eval_tensor(stdout: &str) -> (Vec<usize>, Vec<f64>) {
    let line = stdout
        .lines()
        .find(|l| l.contains(" = tensor(shape="))
        .unwrap_or_else(|| panic!("eval printed no labelled tensor line:\n{stdout}"));
    let parse_usize_list = |s: &str| {
        s.split(',')
            .filter_map(|p| p.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    };
    let shape = line
        .split_once("shape=[")
        .and_then(|(_, s)| s.split_once(']'))
        .map(|(s, _)| parse_usize_list(s))
        .unwrap_or_default();
    let data = line
        .split_once("data=[")
        .and_then(|(_, s)| s.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .filter_map(|p| p.trim().parse::<f64>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    (shape, data)
}

// ── chelis#373: grad through a Tier-3 named-axis reduction via a concrete-rank
//    intermediate def ───────────────────────────────────────────────────────

/// chelis#373: `grad` through a TWO-LEVEL call chain into a Tier-3 rank-spread
/// named reduce. `loss_t3`'s concrete-rank parameter (`tensor[2i64, 3i64]`) flows
/// into `sum_rows` (concrete-rank formal `[b, seq]`) which calls `sum_seq`
/// (rank-spread formal `[..pre, seq, ..post]`).
///
/// In the FORWARD lane each nested def call routes through
/// `try_named_axis_def_call`, which re-stamps the callee's declared formal
/// named dims onto the staged placeholder, so the operand reaching `sum_seq`'s
/// `extract_rank_var_bindings` still carries `Named("seq")`. In the GRAD lane
/// the whole body is inlined into one DAG, the operand monomorphizes to
/// `[Lit(2), Lit(3)]`, and the by-name anchor split in
/// `extract_rank_var_bindings` could not find `seq` — it tripped the
/// `rank-spread anchor \`seq\` absent from monomorphized actual` lowering error.
///
/// `loss_t3` reduces ALL elements (`sum_rows` over `seq`, then `sum` over the
/// surviving axis), so the gradient w.r.t. every input is exactly 1. A shape of
/// [2i64, 3i64] with all-ones is the analytic gradient of sum-of-all-elements; an
/// axis-mislabel in the recovered split would surface here as a wrong shape.
#[test]
fn grad_through_concrete_then_spread_named_reduce() {
    let source = "def sum_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def sum_rows(x: &tensor[b, seq, f32]) -> tensor[b, f32] = sum_seq(x)\n\
         def loss_t3(x: tensor[2, 3, f32]) -> f32 = tensor_to_scalar(sum(sum_rows(&x), 0))\n\
         out = grad(loss_t3)(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    assert_clean(&check_json(source), "#373 grad chain checks clean");
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "grad_concrete_then_spread");
    let (shape, data) = parse_eval_tensor(&eval);
    assert_eq!(
        shape,
        vec![2, 3],
        "#373: grad of sum-of-all must keep the [2i64, 3i64] input shape ({eval})"
    );
    assert_eq!(data.len(), 6, "#373: grad has 6 elements ({eval})");
    for (i, g) in data.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-9,
            "#373: grad[{i}] = {g}, expected 1.0 (sum-of-all gradient is ones) ({eval})"
        );
    }
}

/// chelis#373 forward control (negative parity): the SAME two-level call chain
/// evaluated WITHOUT `grad` must still reduce over the correct (named `seq`)
/// axis. Row sums of [[1i64, 2i64, 3i64],[4i64, 5i64, 6i64]] over `seq`(=axis 1) are [6i64, 15i64]; an
/// axis-mislabel in the spread-aware anchor recovery would sum the batch axis
/// to [5i64, 7i64, 9i64] (rank 3) instead. Pins that the fix does not regress forward.
#[test]
fn forward_through_concrete_then_spread_named_reduce_control() {
    let source = "def sum_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def sum_rows(x: &tensor[b, seq, f32]) -> tensor[b, f32] = sum_seq(x)\n\
         def fwd_ok(x: tensor[2, 3, f32]) -> tensor[2, f32] = sum_rows(&x)\n\
         out = fwd_ok(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    assert_clean(&check_json(source), "#373 forward control checks clean");
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "forward_concrete_then_spread");
    let (shape, data) = parse_eval_tensor(&eval);
    assert_eq!(
        shape,
        vec![2],
        "#373 control: forward must reduce seq, yielding shape [2i64] ({eval})"
    );
    assert_eq!(data, vec![6.0, 15.0], "#373 control: row sums ({eval})");
}

/// chelis#373 grad asymmetry guard: a NON-square chain where reducing the
/// wrong (recovered) axis would change the GRADIENT SHAPE, so an unsound
/// positional guess is caught even though sum-of-all gradients are all ones.
/// `sum_cols` reduces the LEADING named axis (`row`) via a spread formal
/// `[row, ..rest]`; the concrete intermediate `pick(x: tensor[4i64, 2i64])` pins
/// `row` at position 0. The gradient w.r.t. the [4i64, 2i64] input is all ones; a
/// mislabel recovering `row` at the wrong index would mis-shape the reduce and
/// surface as a non-[4i64, 2i64] gradient.
#[test]
fn grad_through_leading_spread_named_reduce() {
    let source = "def sum_first(x: &tensor[row, ..rest, f32]) -> tensor[..rest, f32] = sum(x, row)\n\
         def pick(x: &tensor[row, col, f32]) -> tensor[col, f32] = sum_first(x)\n\
         def loss(x: tensor[4, 2, f32]) -> f32 = tensor_to_scalar(sum(pick(&x), 0))\n\
         out = grad(loss)(to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))\n";
    assert_clean(&check_json(source), "#373 leading-spread grad checks clean");
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "grad_leading_spread");
    let (shape, data) = parse_eval_tensor(&eval);
    assert_eq!(
        shape,
        vec![4, 2],
        "#373: grad must keep the [4i64, 2i64] input shape ({eval})"
    );
    for (i, g) in data.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-9,
            "#373: grad[{i}] = {g}, expected 1.0 ({eval})"
        );
    }
}

/// chelis#373 negative parity: a Tier-3 named-reduce whose reduced axis name is
/// NOT a real axis of the operand stays rejected. `sum(x, ghost)` names an axis
/// that the signature never declares, so the checker must reject it — the
/// spread-aware anchor recovery must not paper over a genuinely-absent axis
/// into a silent (wrong) positional guess.
#[test]
fn reduce_unknown_named_axis_still_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, ghost)\n",
    );
    assert_rejected_with(&json, "ghost", "#373: unknown reduced axis rejected");
}

/// chelis#373 grad-VALUE guard: the sibling sum-of-all grad tests above all
/// produce an all-ones gradient, which is axis-INSENSITIVE in value — a reduce
/// over the wrong (recovered) axis would still pass as long as the gradient
/// keeps the input shape. This test pins the recovered axis by an axis-sensitive
/// gradient: `loss` is `sum((sum_rows x)^2)`, so the gradient w.r.t. each input
/// is `2 * rowsum[row]`, NOT a constant. Row sums of [[1i64, 2i64, 3i64],[4i64, 5i64, 6i64]] over the
/// named `seq` axis are [6i64, 15i64], so the analytic (and finite-difference) grad is
/// [[12i64, 12i64, 12i64], [30i64, 30i64, 30i64]]. If the spread-aware anchor recovery reduced the
/// batch axis instead, the column sums [5i64, 7i64, 9i64] would yield different values
/// (and a different intermediate shape), so a wrong-axis-but-right-shape
/// regression surfaces here even though the shape stays [2i64, 3i64].
#[test]
fn grad_through_concrete_then_spread_named_reduce_values_are_axis_sensitive() {
    let source = "def sum_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def sum_rows(x: &tensor[b, seq, f32]) -> tensor[b, f32] = sum_seq(x)\n\
         def loss(x: tensor[2, 3, f32]) -> f32 = {\n\
           r = sum_rows(&x)\n\
           sq = r * r\n\
           tensor_to_scalar(sum(sq, 0))\n\
         }\n\
         out = grad(loss)(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    assert_clean(&check_json(source), "#373 axis-sensitive grad checks clean");
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "grad_axis_sensitive_values");
    let (shape, data) = parse_eval_tensor(&eval);
    assert_eq!(
        shape,
        vec![2, 3],
        "#373: grad must keep the [2i64, 3i64] input shape ({eval})"
    );
    // 2 * rowsum: row 0 sum = 6 -> 12; row 1 sum = 15 -> 30.
    let expected = [12.0, 12.0, 12.0, 30.0, 30.0, 30.0];
    assert_eq!(
        data.len(),
        expected.len(),
        "#373: grad has 6 elements ({eval})"
    );
    for (i, (g, e)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "#373: grad[{i}] = {g}, expected {e} (2*rowsum, reduced over named `seq`) ({eval})"
        );
    }
}
