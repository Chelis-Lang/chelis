//! Phase E: tests for `check_linearity_with_context`.
//!
//! Acceptance per Phase E plan:
//! 1. New API reachable from `chelis_types`.
//! 2. Existing `check_linearity` continues to work (covered by tests in
//!    `linearity.rs`).
//! 3. New unit test: a library function that consumes a tensor parameter
//!    doesn't "consume" the new-code's free variables. A new-code function
//!    that calls library `matmul` on its own input tensor must succeed.
//! 4. New unit test: parity vs monolithic for at least 3 library + snippet
//!    pairs.
//!
//! The library leg is fed through the same IR check + linearity pipeline
//! as the monolithic flow, so library-internal linearity is validated
//! before any new-code check runs.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::{
    build_type_env_from_library, check_ir_program, check_ir_with_context, check_linearity,
    check_linearity_with_context, check_typed_program,
};

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    desugar_program(&decls).expect("Surf fixture must desugar")
}

fn check_library_with_linearity(library_src: &str) -> chelis_types::CheckedProgram {
    // Library leg: full IR check check, then linearity. Mirrors what the
    // production pipeline does before stashing a CompiledContext.
    let library_deep = surf_to_deep(library_src);
    let checked = check_ir_program(&library_deep)
        .unwrap_or_else(|e| panic!("library IR check check failed: {:?}", e.errors));
    check_linearity(&checked).expect("library linearity must be clean")
}

fn check_new_with_context(
    library_src: &str,
    new_src: &str,
) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let library_program = check_library_with_linearity(library_src);
    let ctx =
        build_type_env_from_library(&surf_to_deep(library_src)).expect("library context build OK");
    let new_deep = surf_to_deep(new_src);
    let new_checked = check_ir_with_context(&ctx, &new_deep)
        .unwrap_or_else(|e| panic!("with-context IR check failed: {:?}", e.errors));
    check_linearity_with_context(&library_program, &new_checked).map(|_| ())
}

fn check_monolithic_combined(
    library_src: &str,
    new_src: &str,
) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let combined_src = format!("{library_src}\n{new_src}");
    let combined_deep = surf_to_deep(&combined_src);
    let checked = check_typed_program(&combined_deep)
        .unwrap_or_else(|e| panic!("monolithic IR check failed: {:?}", e.errors));
    check_linearity(&checked).map(|_| ())
}

// ── Subtle requirement #1: library tensor params don't consume new-code ──

#[test]
fn library_consume_does_not_steal_new_code_input_tensor() {
    // Library defines `lib_consume` which takes a tensor and returns a
    // tensor — its parameter is consumed inside the library body. New
    // code calls it with its OWN input tensor. The new-code call should
    // consume only the new-code's `my_x`, not anything carried over from
    // the library body.
    let library_src = "def lib_consume(t: tensor[4, f32]) -> tensor[4, f32] = realize(t)";
    let new_src = r#"
def caller(my_x: tensor[4, f32]) -> tensor[4, f32] =
  lib_consume(my_x)
"#;

    check_new_with_context(library_src, new_src)
        .expect("calling library tensor-consuming def on new-code input must be linearity-clean");
}

#[test]
fn library_matmul_call_on_new_code_inputs_succeeds() {
    // Mirror the canonical example in the Phase E plan: a library
    // matmul-like def, called by new code on its own input tensors.
    // Both new-code inputs must be consumed exactly once at the call;
    // no leak of library param names.
    let library_src = r#"
def lib_matmul(a: &tensor[4, f32], b: &tensor[4, f32]) -> tensor[4, f32] =
  add(a, b)
"#;
    let new_src = r#"
def caller(my_x: tensor[4, f32], my_w: tensor[4, f32]) -> tensor[4, f32] =
  {
    out: tensor[4, f32] = lib_matmul(my_x, my_w)
    _ = drop(my_x)
    _ = drop(my_w)
    out
  }
"#;

    check_new_with_context(library_src, new_src).expect("library matmul call on new-code tensors");
}

#[test]
fn new_code_consuming_fanout_is_accepted_with_context() {
    // Same library, but new code consumes `my_x` twice — once by realize,
    // once by the library call. The implicit linearity model treats this as
    // source fan-out and inserts an IR Copy during lowering, so the
    // with-context checker must agree with the monolithic checker.
    let library_src = r#"
def lib_id(t: tensor[4, f32]) -> tensor[4, f32] = t
"#;
    let new_src = r#"
def bad(my_x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = realize(my_x)
    lib_id(my_x)
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(with_ctx.is_ok(), mono.is_ok());
    assert!(
        with_ctx.is_ok(),
        "source-level consuming fan-out is auto-copied: {:?}",
        with_ctx.err()
    );
}

#[test]
fn library_def_reference_is_non_linear_no_consumption() {
    // Just referencing `lib_id` (not calling it) must not trigger any
    // linearity error. Library def types are functions, which are
    // non-linear — they carry no tensor content directly.
    let library_src = r#"
def lib_id(t: tensor[4, f32]) -> tensor[4, f32] = t
"#;
    let new_src = r#"
def use_lib(my_x: tensor[4, f32]) -> tensor[4, f32] =
  lib_id(my_x)
"#;
    check_new_with_context(library_src, new_src).expect("library def reference is non-consuming");
}

// ── Subtle requirement #3: state does not leak between snippets ──

#[test]
fn no_leak_between_snippets_against_same_library() {
    // Snippet A uses consuming fan-out, which is valid in the implicit
    // linearity model. After running it, snippet B must still succeed. This
    // exercises function-purity: no shared mutable checker state.
    let library_program =
        check_library_with_linearity("def lib_id(t: tensor[4, f32]) -> tensor[4, f32] = t");

    let lib_deep = surf_to_deep("def lib_id(t: tensor[4, f32]) -> tensor[4, f32] = t");
    let ctx = build_type_env_from_library(&lib_deep).expect("ctx OK");

    // Snippet A: consuming fan-out — accepted and handled later by Copy
    // insertion during lowering.
    let bad_deep = surf_to_deep(
        r#"
def bad(my_x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = realize(my_x)
    lib_id(my_x)
  }
"#,
    );
    let bad_checked = check_ir_with_context(&ctx, &bad_deep).expect("IR check clean");
    let fanout_res = check_linearity_with_context(&library_program, &bad_checked);
    assert!(
        fanout_res.is_ok(),
        "source fan-out snippet must be accepted: {:?}",
        fanout_res.err()
    );

    // Snippet B: clean — must still succeed despite snippet A's failure.
    let good_deep = surf_to_deep("def good(my_x: tensor[4, f32]) -> tensor[4, f32] = lib_id(my_x)");
    let good_checked = check_ir_with_context(&ctx, &good_deep).expect("IR check clean");
    let good_res = check_linearity_with_context(&library_program, &good_checked);
    assert!(
        good_res.is_ok(),
        "clean snippet must succeed; checker state must not leak between calls: {:?}",
        good_res.err()
    );
}

// ── Subtle requirement #4: parity vs monolithic ──

#[test]
fn parity_pair_one_clean_library_call() {
    let library_src = "def lib_relu(t: &tensor[4, f32]) -> tensor[4, f32] = relu(t)";
    let new_src = r#"
def caller(my_x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = lib_relu(my_x)
    _ = drop(my_x)
    y
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(
        with_ctx.is_ok(),
        mono.is_ok(),
        "with-context Result must match monolithic Result"
    );
}

#[test]
fn parity_pair_two_consuming_fanout_accepted() {
    let library_src = "def lib_id(t: tensor[4, f32]) -> tensor[4, f32] = t";
    let new_src = r#"
def bad(my_x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = realize(my_x)
    lib_id(my_x)
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(with_ctx.is_ok(), mono.is_ok());
    assert!(
        with_ctx.is_ok(),
        "consuming fan-out is legal before lowering inserts Copy: {:?}",
        with_ctx.err()
    );
}

#[test]
fn parity_pair_three_borrow_keeps_caller_live() {
    let library_src = r#"
def lib_rank(t: &tensor[4, f32]) -> i32 = rank(t)
"#;
    let new_src = r#"
def caller(my_x: tensor[4, f32]) -> tensor[4, f32] =
  {
    n: i32 = lib_rank(my_x)
    y: tensor[4, f32] = relu(my_x)
    _ = drop(my_x)
    y
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(with_ctx.is_ok(), mono.is_ok());
    assert!(
        with_ctx.is_ok(),
        "borrow keeps my_x live across library call"
    );
}

#[test]
fn parity_pair_four_copy_allows_reuse() {
    // Explicit `copy` remains valid under the implicit model. Same
    // semantics with or without context.
    let library_src = "def lib_consume(t: tensor[4, f32]) -> tensor[4, f32] = realize(t)";
    let new_src = r#"
def caller(my_x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = lib_consume(copy(my_x))
    out: tensor[4, f32] = add(my_x, y)
    _ = drop(my_x)
    _ = drop(y)
    out
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(with_ctx.is_ok(), mono.is_ok());
    assert!(with_ctx.is_ok());
}

#[test]
fn parity_pair_five_observational_query_does_not_consume() {
    // Library def whose new-code call passes a tensor through an
    // observational builtin (`shape`) — must not consume `my_x`.
    let library_src = r#"
def lib_shape0(t: &tensor[2, 3, f32]) -> i64 = shape(t, 0)
"#;
    let new_src = r#"
def caller(my_x: tensor[2, 3, f32]) -> tensor[2, 3, f32] =
  {
    n: i64 = lib_shape0(my_x)
    y: tensor[2, 3, f32] = relu(my_x)
    _ = drop(my_x)
    y
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(with_ctx.is_ok(), mono.is_ok());
}

// ── Phase G' regression: aliased library calls in a let-block ──

/// Regression: `linspace(...) -> assert_shape(actual, ...) ->
/// assert_close_tensor(actual, ...)` is the exact shape that broke
/// `chelis test` on chelis-std's `test_linspace_endpoints`. Monolithic
/// `check_linearity` accepts because library function bodies don't
/// produce new-code consume sites — `assert_shape` consumes its OWN
/// param `t` inside its body, which has nothing to do with the
/// caller's `actual`. The Phase E `_with_context` variant must agree.
///
/// Phase G' fix: pre-compute a per-library callable consumption
/// signature at context build, then look it up at call sites instead
/// of treating every library-call argument as consuming.
#[test]
fn library_assert_then_assert_does_not_double_consume_caller() {
    // Two library defs that each consume their tensor parameter
    // internally — but the new-code caller's `actual` is its OWN
    // separate tensor. The call sites pass `actual` to two consecutive
    // library calls, which monolithic linearity accepts.
    let library_src = r#"
def assert_shape_lib(t: &tensor[4, f32]) -> i32 = rank(t)
def assert_close_lib(a: &tensor[4, f32], b: &tensor[4, f32]) -> i32 = rank(a)
"#;
    let new_src = r#"
def caller(actual: tensor[4, f32], expected: tensor[4, f32]) -> i32 =
  {
    _shape: i32 = assert_shape_lib(&actual)
    close: i32 = assert_close_lib(actual, expected)
    _ = drop(actual)
    _ = drop(expected)
    close
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(
        with_ctx.is_ok(),
        mono.is_ok(),
        "with-context Result must match monolithic Result on borrowed-then-consume pattern"
    );
    assert!(
        with_ctx.is_ok(),
        "linspace-style aliased library calls must be linearity-clean: {:?}",
        with_ctx.err()
    );
}

/// Phase G' regression — `_ = expr` discard-binding form. This is the
/// exact pattern in chelis-std's `test_linspace_endpoints`:
///   `_ = assert_shape(actual, ...)`
///   `assert_close_tensor(actual, ...)`
/// The wildcard `_` discards the call's return value but the call's
/// argument-consume semantics still apply at the call site. Empirical
/// observation: monolithic check ACCEPTS this pattern on the chelis-std
/// production corpus. With-context must agree.
#[test]
fn library_underscore_discard_then_call_with_context_matches_monolithic() {
    let library_src = r#"
def assert_shape_lib(t: &tensor[4, f32], n: i64) -> i32 = rank(t)
def assert_close_lib(a: &tensor[4, f32], b: &tensor[4, f32]) -> i32 = rank(a)
"#;
    // Use `_ =` discard form, mirroring the chelis-std pattern.
    let new_src = r#"
def caller(actual: tensor[4, f32], expected: tensor[4, f32]) -> i32 =
  {
    _ = assert_shape_lib(actual, cast(4, i64))
    close: i32 = assert_close_lib(actual, expected)
    _ = drop(actual)
    _ = drop(expected)
    close
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(
        with_ctx.is_ok(),
        mono.is_ok(),
        "with-context must match monolithic on `_ = consume_call(actual); consume_call(actual)` \
         pattern: with_ctx={:?}, mono={:?}",
        with_ctx,
        mono,
    );
}

/// Phase G' regression — exact `linspace -> assert_shape -> assert_close_tensor`
/// shape WITHOUT explicit borrows. Both library calls take `actual`
/// directly. If monolithic accepts this pattern (as chelis-std's
/// `test_linspace_endpoints` empirically does on the production
/// pipeline), the with-context variant must accept too.
#[test]
fn library_calls_aliased_without_explicit_borrow_match_monolithic() {
    let library_src = r#"
def assert_shape_lib(t: &tensor[4, f32], n: i64) -> i32 = rank(t)
def assert_close_lib(a: &tensor[4, f32], b: &tensor[4, f32]) -> i32 = rank(a)
"#;
    let new_src = r#"
def caller(actual: tensor[4, f32], expected: tensor[4, f32]) -> i32 =
  {
    _shape: i32 = assert_shape_lib(actual, cast(4, i64))
    close: i32 = assert_close_lib(actual, expected)
    _ = drop(actual)
    _ = drop(expected)
    close
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    // Whatever monolithic decides, with-context must match — the cache
    // path cannot reject a snippet the production pipeline accepts.
    assert_eq!(
        with_ctx.is_ok(),
        mono.is_ok(),
        "with-context Result must match monolithic Result on bare-arg double-call: \
         with_ctx={:?}, mono={:?}",
        with_ctx,
        mono,
    );

    // Borrow-typed library signatures auto-borrow bare tensor arguments,
    // so the previously aliased double-call pattern is clean.
    assert!(with_ctx.is_ok());
    assert!(mono.is_ok());
}

#[test]
fn library_two_consecutive_borrowing_calls_do_not_consume() {
    // When both library calls borrow (`&actual`), the caller's tensor
    // stays live past both calls and can still be returned from the
    // function. Mirrors the actual chelis-std pattern where
    // `assert_shape(&t)` and `assert_close_tensor(&a, &b)` are
    // observational.
    let library_src = r#"
def lib_a(t: &tensor[4, f32]) -> i32 = rank(t)
def lib_b(t: &tensor[4, f32]) -> i32 = rank(t)
"#;
    let new_src = r#"
def caller(actual: tensor[4, f32]) -> tensor[4, f32] =
  {
    _x: i32 = lib_a(actual)
    _y: i32 = lib_b(actual)
    y: tensor[4, f32] = relu(actual)
    _ = drop(actual)
    y
  }
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(
        with_ctx.is_ok(),
        mono.is_ok(),
        "with-context must agree with monolithic on observation patterns"
    );
    assert!(with_ctx.is_ok());
}

// ── Tensor-carrying ADTs ──

#[test]
fn library_tensor_carrying_adt_borrow_is_accepted_with_context() {
    // The library declares a record-style ADT (`BatchNormParams`) whose
    // fields contain `tensor`. New code calls a library function that
    // takes `&BatchNormParams[n]`. The borrow must succeed without
    // `InvalidBorrow`, which requires that `check_linearity_with_context`
    // populates `tensor_carrying_adts` from the LIBRARY half too — not
    // only the new-code half. Pre-fix, the library declaration was
    // invisible to the carrier set and the borrow was rejected.
    let library_src = r#"
type BatchNormParams[n] =
  | BatchNormParams { gamma: tensor[n, f32], beta: tensor[n, f32] }

sig lib_borrow_params[n]: &BatchNormParams[n] -> bool
def lib_borrow_params(p) = true
"#;
    let new_src = r#"
def use_params[n](p: BatchNormParams[n]) -> bool = lib_borrow_params(&p)
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(
        with_ctx.is_ok(),
        mono.is_ok(),
        "library tensor-carrying ADT must be visible to with-context borrow check"
    );
    assert!(
        with_ctx.is_ok(),
        "borrow of library-declared tensor-carrying ADT must succeed: {:?}",
        with_ctx.err()
    );
}

#[test]
fn new_code_adt_wrapping_library_tensor_adt_borrow_is_accepted_with_context() {
    // Cross-package transitive carry: the library declares a
    // tensor-carrying ADT (`Inner`); new code declares an ADT
    // (`Outer`) whose only field is `Inner`. `&Outer` must be
    // recognized as borrow-eligible, which requires that the
    // fixed-point pass that decides `tensor_carrying_adts` sees BOTH
    // the library's `Inner` and the new-code's `Outer` in the same
    // run — otherwise the new-code pass cannot resolve `Inner` as
    // already-known carrier when deciding `Outer`'s status. Pre-fix
    // (`compute_tensor_carrying_adts` called twice with independent
    // local sets, then unioned) this regressed silently: `Outer` was
    // not in the carrier set and `&o` was rejected with
    // `InvalidBorrow`.
    let library_src = r#"
type Inner[n] =
  | Inner { values: tensor[n, f32] }
"#;
    let new_src = r#"
type Outer[n] =
  | Outer { inner: Inner[n] }

sig borrow_outer[n]: &Outer[n] -> bool
def borrow_outer(o) = true
def consume_outer[n](o: Outer[n]) -> bool = borrow_outer(&o)
"#;

    let with_ctx = check_new_with_context(library_src, new_src);
    let mono = check_monolithic_combined(library_src, new_src);

    assert_eq!(
        with_ctx.is_ok(),
        mono.is_ok(),
        "with-context must agree with monolithic on cross-package ADT composition"
    );
    assert!(
        with_ctx.is_ok(),
        "new-code ADT wrapping a library tensor-carrying ADT must borrow cleanly: {:?}",
        with_ctx.err()
    );
}

// ── Carrier-set always-recompute invariant ──
//
// `check_linearity_with_context` chains the library and new-code
// `annotated_exprs()` into a single `compute_tensor_carrying_adts`
// call so the fixed-point pass resolves cross-package transitive
// carriers in one go. The function holds no per-call state and
// `CheckedProgram` deliberately does NOT cache the carrier set:
// any future change that introduces caching MUST preserve the
// always-recompute-from-union contract, or a stale cache will
// silently desync borrow semantics. These tests lock that in.

#[test]
fn check_linearity_with_context_is_pure_across_repeated_calls() {
    // Call the function repeatedly, alternating between a library
    // that makes the borrow legal and one that doesn't. Each call's
    // outcome must depend ONLY on the library passed in — no
    // in-process state may leak between calls.
    //
    // Library A declares a tensor-carrying ADT named `Carrier`; new
    // code wraps it in `Outer` and borrows `&o`. Against library A
    // the borrow is accepted (Outer is transitively tensor-carrying).
    // Library B declares an unrelated, non-carrying ADT and provides
    // its own `Carrier` shaped WITHOUT a tensor field — under
    // library B, `Outer.inner: Carrier` is no longer tensor-carrying,
    // so `&o` would be rejected with `InvalidBorrow` if linearity
    // saw it correctly. A stale cached carrier set from a prior
    // call to library A would silently mis-accept it.
    let lib_a = r#"
type Carrier[n] =
  | Carrier { values: tensor[n, f32] }
"#;
    let lib_b = r#"
type Carrier[n] =
  | Carrier { tag: i32 }
"#;
    let new_src = r#"
type Outer[n] =
  | Outer { inner: Carrier[n] }

sig borrow_outer[n]: &Outer[n] -> bool
def borrow_outer(o) = true
def consume_outer[n](o: Outer[n]) -> bool = borrow_outer(&o)
"#;

    // Interleave the calls to make any leaked state fail loudly.
    for _ in 0..3 {
        let r_a = check_new_with_context(lib_a, new_src);
        assert!(
            r_a.is_ok(),
            "library A (tensor-carrying Carrier) must accept &Outer borrow on every call; \
             got: {:?}",
            r_a.err(),
        );

        let r_b = check_new_with_context(lib_b, new_src);
        assert!(
            r_b.is_err(),
            "library B (Carrier carries no tensor) must reject &Outer borrow on every call; \
             a stale carrier set from a prior library-A call would silently mis-accept it",
        );
        let errors = r_b.expect_err("library B must error");
        let cites_invalid_borrow = errors
            .iter()
            .any(|e| matches!(e.kind, chelis_types::errors::CheckErrorKind::InvalidBorrow));
        assert!(
            cites_invalid_borrow,
            "library B rejection must be `InvalidBorrow` (the carrier-set-driven diagnostic), \
             got: {:?}",
            errors,
        );
    }
}

// ── Empty / degenerate cases ──

#[test]
fn empty_library_matches_monolithic() {
    // Library is empty (no defs). New code is a standalone def that
    // would pass `check_linearity` on its own. The two paths must
    // agree.
    let empty_lib_program = check_library_with_linearity("");
    let new_deep = surf_to_deep(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = relu(x)
    _ = drop(x)
    y
  }
"#,
    );
    let ctx_empty = chelis_types::TypeEnv::empty();
    let new_checked = check_ir_with_context(&ctx_empty, &new_deep).expect("IR check clean");
    let res = check_linearity_with_context(&empty_lib_program, &new_checked);
    assert!(
        res.is_ok(),
        "empty-library context must accept a clean snippet"
    );
}
