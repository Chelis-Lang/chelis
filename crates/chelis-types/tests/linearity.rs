use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

fn check_core(source: &str) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let deep: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    let checked = check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("type check should succeed");
    check_linearity(&checked, &mut chelis_types::NativeServices::new(None)).map(|_| ())
}

fn typecheck_core(source: &str) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let deep: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .map(|_| ())
    .map_err(|result| result.errors)
}

#[test]
fn detects_use_after_consume() {
    let errors = check_core(include_str!(
        "fixtures/core/tests/linearity/detects_use_after_consume.0.deep"
    ))
    .expect_err("linearity should reject reusing a consumed tensor");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `x`")
            && error.message.contains("realize")
    }));
}

#[test]
fn copy_allows_reuse() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/copy_allows_reuse.0.deep"
    ))
    .expect("copy should preserve a later consuming use");
}

#[test]
fn auto_borrowed_read_only_primitives_allow_fanout() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/auto_borrowed_read_only_primitives_allow_fanout.0.deep"
    ))
    .expect("read-only tensor primitives should borrow their tensor inputs");
}

#[test]
fn pipe_auto_borrows_read_only_stage_input() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/pipe_auto_borrows_read_only_stage_input.0.deep"
    ))
    .expect("pipe stages should use the same call-site auto-borrow rule");
}

#[test]
fn explicit_borrow_does_not_satisfy_owned_parameter() {
    // Fixture renamed from `take` for chelis#353: `take` is a builtin
    // name and bare shadowing defs are now rejected at declaration time.
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/linearity/explicit_borrow_does_not_satisfy_owned_parameter.0.deep"
    ))
    .expect_err("borrow-to-owned must require an explicit copy");

    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch))
    );
}

#[test]
fn copy_accepts_explicit_borrow_and_returns_owned_tensor() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/copy_accepts_explicit_borrow_and_returns_owned_tensor.0.deep"
    ))
    .expect("copy(&x) should fork ownership from a borrowed tensor");
}

#[test]
fn borrowed_but_never_consumed_is_auto_dropped() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/borrowed_but_never_consumed_is_auto_dropped.0.deep"
    ))
    .expect("implicit drop handles borrowed local owners at scope end");
}

#[test]
fn borrow_cannot_escape_as_function_result() {
    let errors = check_core(include_str!(
        "fixtures/core/tests/linearity/borrow_cannot_escape_as_function_result.0.deep"
    ))
    .expect_err("borrows cannot be returned from functions");

    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::InvalidBorrow))
    );
}

#[test]
fn closure_capture_consumes_outer_tensor() {
    let errors = check_core(include_str!(
        "fixtures/core/tests/linearity/closure_capture_consumes_outer_tensor.0.deep"
    ))
    .expect_err("capturing a tensor should consume it");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("closure capture")
    }));
}

#[test]
fn closure_capture_consumes_outer_tensor_carrying_adt() {
    // Mirrors `closure_capture_consumes_outer_tensor` but for a
    // record-style ADT whose tensor lives in a variant field, not in
    // a type-argument position. Locks the parameterized
    // `type_expr_contains_tensor` call inside `check_fn`'s capture
    // pass at `linearity.rs:737-740`: if that predicate ever stops
    // consulting `tensor_carrying_adts`, this case would silently
    // regress — the outer `p` would survive the closure capture and
    // the second `use_params(p)` would type-check, hiding a use-after-
    // consume.
    let errors = check_core(include_str!(
        "fixtures/core/tests/linearity/closure_capture_consumes_outer_tensor_carrying_adt.0.deep"
    ))
    .expect_err("capturing a tensor-carrying ADT must consume it");

    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains("closure capture")
        }),
        "expected UseAfterConsume with closure-capture site; got {errors:?}"
    );
}

#[test]
fn match_consumes_tuple_scrutinee() {
    let errors = check_core(include_str!(
        "fixtures/core/tests/linearity/match_consumes_tuple_scrutinee.0.deep"
    ))
    .expect_err("reusing a tuple carrying a tensor after match should fail");

    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume))
    );
}

#[test]
fn tensor_shape_queries_do_not_consume_tensor_inputs() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/tensor_shape_queries_do_not_consume_tensor_inputs.0.deep"
    ))
    .expect("shape queries should be observational, not consuming");
}

#[test]
fn to_list_does_not_consume_tensor_input() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/to_list_does_not_consume_tensor_input.0.deep"
    ))
    .expect("to_list reads the tensor without freeing it, so x must remain live");
}

#[test]
fn to_list_still_flags_use_after_genuine_consume() {
    let errors = check_core(include_str!(
        "fixtures/core/tests/linearity/to_list_still_flags_use_after_genuine_consume.0.deep"
    ))
    .expect_err("to_list after a consuming use of x should still be rejected");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `x`")
    }));
}

#[test]
fn tensor_to_scalar_does_not_consume_tensor_input() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/tensor_to_scalar_does_not_consume_tensor_input.0.deep"
    ))
    .expect("tensor_to_scalar reads the tensor without freeing it, so x must remain live");
}

#[test]
fn tensor_to_scalar_still_flags_use_after_genuine_consume() {
    let errors = check_core(
        include_str!("fixtures/core/tests/linearity/tensor_to_scalar_still_flags_use_after_genuine_consume.0.deep"),
    )
    .expect_err("tensor_to_scalar after a consuming use of x should still be rejected");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `x`")
    }));
}

#[test]
fn len_does_not_consume_list_argument() {
    // chelis#527: `len` reads a `List`'s length via `chelis_list_len`
    // (a `const *` that never frees), so the caller still owns the
    // list afterwards.  The parameter is named `params` — a
    // Deep-tag-colliding identifier that desugars through the MetaExpr
    // param form #343 began consume-tracking — so this is a faithful
    // regression for the School `_step_list` read-then-reuse shape.
    check_core(include_str!(
        "fixtures/core/tests/linearity/len_does_not_consume_list_argument.0.deep"
    ))
    .expect("len reads the list without freeing it, so params must remain live (chelis#527)");
}

#[test]
fn index_does_not_consume_list_argument() {
    // chelis#527: `index` retains the element it returns and reads the
    // list via a `const *` (`chelis_list_index`), never freeing it.
    check_core(include_str!(
        "fixtures/core/tests/linearity/index_does_not_consume_list_argument.0.deep"
    ))
    .expect(
        "index reads an element without freeing the list, so params must remain live (chelis#527)",
    );
}

#[test]
fn list_len_then_index_then_reuse_compiles() {
    // The full School `_step_list` read-then-reuse shape: read the
    // length and an element, then hand the list onward.  Compiled
    // clean at 0.10.0, regressed at 0.10.1 (chelis#527).
    check_core(include_str!(
        "fixtures/core/tests/linearity/list_len_then_index_then_reuse_compiles.0.deep"
    ))
    .expect("read length + element then reuse the list must compile (chelis#527)");
}

#[test]
fn len_still_flags_use_after_genuine_consume() {
    // Negative parity: borrowing on `len` must not blind the checker
    // to a real prior consume.  An explicit `drop` frees `params`, so
    // the later `len(params)` borrow-read is a use-after-consume.
    let errors = check_core(include_str!(
        "fixtures/core/tests/linearity/len_still_flags_use_after_genuine_consume.0.deep"
    ))
    .expect_err("len after dropping params should still be rejected");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `params`")
    }));
}

#[test]
fn index_still_flags_use_after_genuine_consume() {
    // Negative parity for `index`, mirroring the `len` case above.
    let errors = check_core(include_str!(
        "fixtures/core/tests/linearity/index_still_flags_use_after_genuine_consume.0.deep"
    ))
    .expect_err("index after dropping params should still be rejected");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::UseAfterConsume)
            && error.message.contains("variable `params`")
    }));
}

#[test]
fn len_explicit_container_borrow_is_a_type_error() {
    // chelis#527: auto-borrow is the idiom for the read-only container
    // queries; the explicit `len(&xs)` surface form is intentionally
    // *not* supported and is rejected at check time (spec
    // `05-risc-primitives.md` §1.3.1). An explicit `&List` would
    // type-check past the front end but the host-value lane does not
    // erase the borrow wrapper for container values, so it would fail C
    // codegen — a worse footgun than a clear front-end error. This locks
    // the rejection so a future signature change cannot silently admit
    // `len(&xs)`.
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/linearity/len_explicit_container_borrow_is_a_type_error.0.deep"
    ))
    .expect_err("explicit &List is not a supported surface form for len");

    // The diagnostic must point at the real problem — `len` auto-borrows, so
    // the explicit `&` is redundant/unsupported — not the misleading bare
    // "expects List or Dict input, got &List …" (the input *is* a List).
    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::TypeMismatch)
            && error
                .message
                .contains("len auto-borrows its List/Dict argument")
            && error.message.contains("write `len(xs)`, not `len(&xs)`")
    }));
}

#[test]
fn index_explicit_container_borrow_is_a_type_error() {
    // Negative parity for `index`, mirroring the `len` case above:
    // `index(&xs, i)` is rejected at check time (chelis#527).
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/linearity/index_explicit_container_borrow_is_a_type_error.0.deep"
    ))
    .expect_err("explicit &List is not a supported surface form for index");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::TypeMismatch)
            && error
                .message
                .contains("index auto-borrows its List argument")
            && error
                .message
                .contains("write `index(xs, i)`, not `index(&xs, i)`")
    }));
}

#[test]
fn len_of_non_container_does_not_mention_auto_borrow() {
    // The auto-borrow guidance must fire only for a genuine `&List`/`&Dict`
    // (a valid container with a redundant `&`). A non-container input is a
    // different mistake, so it must keep the plain "expects List or Dict
    // input" diagnostic and must NOT misleadingly claim `len` auto-borrows
    // an argument that is not even a container.
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/linearity/len_of_non_container_does_not_mention_auto_borrow.0.deep"
    ))
    .expect_err("len of a scalar is a type error");

    assert!(errors.iter().any(|error| {
        matches!(error.kind, CheckErrorKind::TypeMismatch)
            && error.message.contains("len expects List or Dict input")
            && !error.message.contains("auto-borrows")
    }));
}

#[test]
fn grad_accepts_function_with_borrowed_tensor_parameter() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/grad_accepts_function_with_borrowed_tensor_parameter.0.deep"
    ))
    .expect("grad(f)(x) should auto-borrow when f takes a borrowed tensor");
}

#[test]
fn vmap_lifts_function_with_borrowed_tensor_parameter() {
    check_core(include_str!(
        "fixtures/core/tests/linearity/vmap_lifts_function_with_borrowed_tensor_parameter.0.deep"
    ))
    .expect("vmap should preserve borrowed tensor parameters while adding the batch axis");
}
