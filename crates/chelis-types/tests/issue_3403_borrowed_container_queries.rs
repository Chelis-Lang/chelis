//! chelis#3403: the read-only container queries `len` and `index` read a
//! container that is already borrowed.
//!
//! `spec/04-type-system.md` §8.2 makes `&T` the borrow type and `&x` an
//! optional borrow expression, and `spec/05-risc-primitives.md` §1.3.1 makes
//! `len` and `index` auto-borrow their container. A `&List[T]` or
//! `&Dict[K, V]` value, such as a borrowed parameter, is therefore read
//! unchanged. Only the explicit expression `len(&xs)` / `index(&xs, i)` is
//! the unsupported surface form, so that refusal reads the source operand,
//! not the operand's type. Before the fix both queries refused every
//! `&List`/`&Dict` operand as if the source had written `&`.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_linearity, check_typed_program};

/// Type errors, then linearity errors, of a Surf program.
fn check_errors(source: &str) -> Vec<CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    match check_typed_program(&deep) {
        Ok(checked) => match check_linearity(&checked) {
            Ok(_) => Vec::new(),
            Err(errors) => errors,
        },
        Err(result) => result.errors,
    }
}

#[track_caller]
fn assert_checks(source: &str, what: &str) {
    let errors = check_errors(source);
    assert!(
        errors.is_empty(),
        "{what}: expected no errors; got {errors:?}"
    );
}

/// The one error a program reports, which must have exactly `kind` and
/// `message`.
#[track_caller]
fn assert_single_error(source: &str, kind: CheckErrorKind, message: &str, what: &str) {
    let errors = check_errors(source);
    assert!(
        matches!(
            errors.as_slice(),
            [error] if std::mem::discriminant(&error.kind) == std::mem::discriminant(&kind)
                && error.message == message
        ),
        "{what}: expected exactly one {kind:?} `{message}`; got {errors:?}"
    );
}

#[track_caller]
fn assert_single_mismatch(source: &str, message: &str, what: &str) {
    assert_single_error(source, CheckErrorKind::TypeMismatch, message, what);
}

const LEN_EXPLICIT_BORROW: &str = "len auto-borrows its List/Dict argument, so an explicit `&` is \
     not a supported surface form: write `len(xs)`, not `len(&xs)` (got &List tensor[2, f32])";

const INDEX_EXPLICIT_BORROW: &str = "index auto-borrows its List argument, so an explicit `&` is \
     not a supported surface form: write `index(xs, i)`, not `index(&xs, i)` (got &List \
     tensor[2, f32])";

#[test]
fn len_reads_a_borrowed_list_parameter() {
    assert_checks(
        "def count_l(xs: &List[tensor[2, f32]]) -> i64 = len(xs)\n",
        "len of a &List parameter",
    );
}

#[test]
fn len_of_a_borrowed_list_is_i64() {
    // The borrowed operand is decided on its referent, so the result is the
    // owned call's `i64`, not a fresh variable that any declared result
    // would satisfy.
    assert_single_error(
        "def count_l(xs: &List[tensor[2, f32]]) -> f32 = len(xs)\n",
        CheckErrorKind::TypeMismatch,
        "def 'count_l' body doesn't match declared signature: expected \
         `(&List tensor[2, f32]) -> f32`, got `(&List tensor[2, f32]) -> i64`",
        "len of a &List declared f32",
    );
}

#[test]
fn len_reads_a_borrowed_dict_parameter() {
    assert_checks(
        "def count_d(d: &Dict[string, tensor[2, f32]]) -> i64 = len(d)\n",
        "len of a &Dict parameter",
    );
}

#[test]
fn index_reads_a_borrowed_list_parameter_at_its_element_type() {
    assert_checks(
        "def first_l(xs: &List[tensor[2, f32]]) -> tensor[2, f32] = index(xs, 0i64)\n",
        "index of a &List parameter",
    );
    // The issue's reproducer: the element feeds a borrowing tensor reduction.
    assert_checks(
        "def first_l(xs: &List[tensor[2, f32]]) -> f32 = \
         tensor_to_scalar(sum(index(xs, 0i64), 0i32))\n",
        "index of a &List parameter, reduced",
    );
}

#[test]
fn index_of_a_borrowed_list_yields_the_element_type() {
    assert_single_error(
        "def first_l(xs: &List[tensor[2, f32]]) -> tensor[3, f32] = index(xs, 0i64)\n",
        CheckErrorKind::DimensionMismatch,
        "def 'first_l' body doesn't match declared signature: expected \
         `(&List tensor[2, f32]) -> tensor[3, f32]`, got \
         `(&List tensor[2, f32]) -> tensor[2, f32]`",
        "index of a &List[tensor[2, f32]] declared tensor[3, f32]",
    );
}

#[test]
fn index_of_a_borrowed_list_still_requires_an_i64_index() {
    assert_single_mismatch(
        "def first_l(xs: &List[tensor[2, f32]]) -> tensor[2, f32] = index(xs, 0i32)\n",
        "index expects i64 index, got i32",
        "index of a &List with an i32 index",
    );
}

#[test]
fn generic_borrowed_list_parameter_is_queryable() {
    assert_checks(
        "def count_any[T](xs: &List[T]) -> i64 = len(xs)\n",
        "len of a generic &List parameter",
    );
}

#[test]
fn a_query_decided_after_its_operand_settles_reads_the_borrow() {
    // `ys` is unresolved when `len(ys)` is inferred, so the call suspends and
    // replays once the application binds `ys` to `&List`.
    assert_checks(
        r#"
def count_l(xs: &List[tensor[2, f32]]) -> i64 = {
  f = fn (ys) -> len(ys)
  f(xs)
}
"#,
        "deferred len of a borrowed list",
    );
}

#[test]
fn a_query_used_as_a_function_value_reads_the_borrow() {
    // The transported collection contract decides the same rule.
    assert_checks(
        r#"
def count_l(xs: &List[tensor[2, f32]]) -> i64 = {
  f = len
  f(xs)
}
"#,
        "len as a function value over a borrowed list",
    );
    assert_checks(
        r#"
def first_l(xs: &List[tensor[2, f32]]) -> tensor[2, f32] = {
  g = index
  g(xs, 0i64)
}
"#,
        "index as a function value over a borrowed list",
    );
}

#[test]
fn caller_keeps_a_list_lent_to_borrowed_queries() {
    assert_checks(
        r#"
def count_l(xs: &List[tensor[2, f32]]) -> i64 = len(xs)
def last_l(xs: &List[tensor[2, f32]]) -> f32 = {
  n = len(xs)
  tensor_to_scalar(sum(index(xs, sub(n, 1i64)), 0i32))
}
def main() -> f32 = {
  xs = [to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])]
  n = count_l(xs)
  tail = last_l(xs)
  head = tensor_to_scalar(sum(index(xs, 0i64), 0i32))
  add(add(cast(n, f32), tail), add(head, cast(len(xs), f32)))
}
"#,
        "caller reuses a list after lending it",
    );
}

#[test]
fn explicit_borrow_of_a_borrowed_list_is_still_refused() {
    // The refusal is the source spelling, so it holds whether the operand
    // `xs` is owned or already borrowed.
    assert_single_mismatch(
        "def count_l(xs: &List[tensor[2, f32]]) -> i64 = len(&xs)\n",
        LEN_EXPLICIT_BORROW,
        "len(&xs) of a &List parameter",
    );
    assert_single_mismatch(
        "def first_l(xs: &List[tensor[2, f32]]) -> tensor[2, f32] = index(&xs, 0i64)\n",
        INDEX_EXPLICIT_BORROW,
        "index(&xs, i) of a &List parameter",
    );
}

#[test]
fn explicit_borrow_of_an_owned_list_is_still_refused() {
    assert_single_mismatch(
        "def count_l(xs: List[tensor[2, f32]]) -> i64 = len(&xs)\n",
        LEN_EXPLICIT_BORROW,
        "len(&xs) of an owned List",
    );
    assert_single_mismatch(
        "def first_l(xs: List[tensor[2, f32]]) -> tensor[2, f32] = index(&xs, 0i64)\n",
        INDEX_EXPLICIT_BORROW,
        "index(&xs, i) of an owned List",
    );
}

#[test]
fn explicit_borrow_is_refused_whatever_its_operand() {
    // The refusal is the spelling, so a borrowed tensor gets it too; the
    // operand's type only names what was borrowed.
    assert_single_mismatch(
        "def count_t(t: tensor[2, f32]) -> i64 = len(&t)\n",
        "len auto-borrows its List/Dict argument, so an explicit `&` is not a supported surface \
         form: write `len(xs)`, not `len(&xs)` (got &tensor[2, f32])",
        "len(&t) of a tensor",
    );
    assert_single_mismatch(
        "def count_d(d: Dict[string, tensor[2, f32]]) -> i64 = len(&d)\n",
        "len auto-borrows its List/Dict argument, so an explicit `&` is not a supported surface \
         form: write `len(xs)`, not `len(&xs)` (got &Dict string tensor[2, f32])",
        "len(&d) of an owned Dict",
    );
}

#[test]
fn explicit_borrow_refusal_survives_a_deferred_operand() {
    // `ys` is unresolved when `len(&ys)` is inferred; the replay that decides
    // the call once `ys` binds still reads the source spelling.
    assert_single_mismatch(
        r#"
def count_l(xs: List[tensor[2, f32]]) -> i64 = {
  f = fn (ys) -> len(&ys)
  f(xs)
}
"#,
        LEN_EXPLICIT_BORROW,
        "deferred len(&ys)",
    );
}

#[test]
fn a_borrowed_list_is_not_consumable() {
    // Reading through the borrow is all the queries gain: a consuming
    // collection builtin still refuses a borrowed list.
    assert_single_mismatch(
        "def grow(xs: &List[tensor[2, f32]], t: tensor[2, f32]) -> List[tensor[2, f32]] = \
         append(xs, t)\n",
        "append expects List input, got &List tensor[2, f32]",
        "append to a &List parameter",
    );
}

#[test]
fn len_of_a_borrowed_tensor_is_not_a_container_query() {
    assert_single_mismatch(
        "def count_t(t: &tensor[2, f32]) -> i64 = len(t)\n",
        "len expects List or Dict input, got &tensor[2, f32]",
        "len of a &tensor parameter",
    );
    assert_single_mismatch(
        "def first_t(t: &tensor[2, f32]) -> f32 = index(t, 0i64)\n",
        "index expects List input, got &tensor[2, f32]",
        "index of a &tensor parameter",
    );
}
