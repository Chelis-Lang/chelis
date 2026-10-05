//! chelis#3178 and chelis#3181: what a `drop` may follow, what it forbids
//! afterwards, and what dropping a projection does to its parent.
//!
//! - A `drop` after an ordinary consume is consuming fan-out, repaired by a
//!   copy at the earlier use ([05-OP-67], spec/04 section 8.3).
//! - A `drop` of an owner a closure borrows is refused ([04-LIN-2]).
//! - A projection moves its component out of the parent; a `drop` of it leaves
//!   the parent and that component unusable, and disjoint components usable
//!   ([04-LIN-11]).

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_linearity, check_typed_program};

fn linearity(source: &str) -> Result<(), Vec<CheckError>> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).map(|_| ())
}

#[track_caller]
fn assert_accepted(source: &str, what: &str) {
    if let Err(errors) = linearity(source) {
        panic!("{what}: expected the program to be accepted; got {errors:?}");
    }
}

#[track_caller]
fn assert_refused(source: &str, kind: CheckErrorKind, needles: &[&str], what: &str) {
    let errors = linearity(source).expect_err(&format!("{what}: expected a refusal"));
    assert!(
        errors.iter().any(|error| {
            std::mem::discriminant(&error.kind) == std::mem::discriminant(&kind)
                && needles.iter().all(|needle| error.message.contains(needle))
        }),
        "{what}: expected {kind:?} containing {needles:?}; got {errors:?}"
    );
}

// chelis#3178(a): consume, then drop.

#[test]
fn a_drop_after_an_ordinary_consume_is_accepted() {
    assert_accepted(
        r#"
def eat(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    a = eat(x)
    c = drop(x)
    a
  }
"#,
        "eat(x) then drop(x)",
    );
}

#[test]
fn a_drop_after_consuming_a_destructured_component_is_refused() {
    assert_refused(
        r#"
def eat(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (a, b) = p
    e = eat(a)
    c = drop(a)
    add(e, b)
  }
"#,
        CheckErrorKind::UseAfterConsume,
        &["variable `a`"],
        "drop of a consumed destructured component",
    );
}

// chelis#3178(b): a drop while a closure borrows the owner.

#[track_caller]
fn assert_drop_while_closure_borrows(source: &str, name: &str, what: &str) {
    assert_refused(
        source,
        CheckErrorKind::InvalidBorrow,
        &[&format!("variable `{name}` is dropped"), "still borrows it"],
        what,
    );
}

#[test]
fn calling_a_borrowing_closure_after_drop_is_refused() {
    assert_drop_while_closure_borrows(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    g = fn (k: i32) -> exp(x)
    c = drop(x)
    g(1)
  }
"#,
        "x",
        "closure called after the drop",
    );
}

#[test]
fn a_borrowing_closure_never_called_again_still_blocks_the_drop() {
    assert_drop_while_closure_borrows(
        r#"
def f(x: tensor[4, f32]) -> i32 =
  {
    g = fn (k: i32) -> exp(x)
    c = drop(x)
    1
  }
"#,
        "x",
        "closure created before the drop and never called",
    );
}

#[test]
fn a_borrowing_closure_returned_after_the_drop_is_refused() {
    assert_drop_while_closure_borrows(
        r#"
def f(x: tensor[4, f32]) -> ((i32) -> tensor[4, f32], i32) =
  {
    g = fn (k: i32) -> exp(x)
    c = drop(x)
    (g, 1)
  }
"#,
        "x",
        "closure returned after the drop",
    );
}

#[test]
fn a_closure_borrowing_an_alias_blocks_dropping_the_source() {
    assert_drop_while_closure_borrows(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y = x
    g = fn (k: i32) -> exp(y)
    c = drop(x)
    g(1)
  }
"#,
        "x",
        "closure borrowing an alias",
    );
}

#[test]
fn a_closure_created_on_one_branch_blocks_a_later_drop() {
    assert_drop_while_closure_borrows(
        r#"
def f(x: tensor[4, f32], z: tensor[4, f32], b: bool) -> tensor[4, f32] =
  {
    g = if b then fn (k: i32) -> exp(x) else fn (k: i32) -> exp(z)
    c = drop(x)
    g(1)
  }
"#,
        "x",
        "closure created on one branch",
    );
}

#[test]
fn a_closure_borrowing_a_copy_does_not_block_the_drop() {
    assert_accepted(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    xc = copy(x)
    g = fn (k: i32) -> exp(xc)
    c = drop(x)
    g(1)
  }
"#,
        "closure borrowing an independent copy",
    );
}

#[test]
fn a_closure_borrowing_another_variable_does_not_block_the_drop() {
    assert_accepted(
        r#"
def f(x: tensor[4, f32], z: tensor[4, f32]) -> tensor[4, f32] =
  {
    g = fn (k: i32) -> exp(z)
    c = drop(x)
    g(1)
  }
"#,
        "closure borrowing a different variable",
    );
}

#[test]
fn a_borrowing_closure_without_a_drop_is_accepted() {
    assert_accepted(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    g = fn (k: i32) -> exp(x)
    g(1)
  }
"#,
        "closure borrow with no drop",
    );
}

#[test]
fn a_drop_before_the_borrowing_closure_is_refused_as_a_use_after_drop() {
    assert_refused(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    c = drop(x)
    g = fn (k: i32) -> exp(x)
    g(1)
  }
"#,
        CheckErrorKind::UseAfterConsume,
        &["variable `x`", "call to `drop`"],
        "closure created after the drop",
    );
}

// chelis#3181: a projection moves its component out of the parent.

#[track_caller]
fn assert_use_after_component_drop(source: &str, name: &str, what: &str) {
    assert_refused(
        source,
        CheckErrorKind::UseAfterConsume,
        &[&format!("`{name}`"), "call to `drop`"],
        what,
    );
}

#[test]
fn using_the_tuple_after_dropping_a_component_is_refused() {
    assert_use_after_component_drop(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> (tensor[4, f32], tensor[4, f32]) =
  {
    c = drop(p.0)
    p
  }
"#,
        "p",
        "whole tuple after drop(p.0)",
    );
}

#[test]
fn projecting_the_dropped_component_again_is_refused() {
    assert_use_after_component_drop(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    c = drop(p.0)
    p.0
  }
"#,
        "p",
        "p.0 after drop(p.0)",
    );
}

#[test]
fn matching_the_tuple_after_dropping_a_component_is_refused() {
    assert_use_after_component_drop(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    c = drop(p.0)
    match p with {
      | (a, b) => b
    }
  }
"#,
        "p",
        "match on p after drop(p.0)",
    );
}

#[test]
fn using_the_tuple_through_an_alias_after_a_component_drop_is_refused() {
    assert_use_after_component_drop(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> (tensor[4, f32], tensor[4, f32]) =
  {
    q = p
    c = drop(q.0)
    p
  }
"#,
        "p",
        "p after drop(q.0) with q = p",
    );
}

#[test]
fn using_the_tuple_after_a_component_drop_on_one_branch_is_refused() {
    assert_use_after_component_drop(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32]), b: bool) -> (tensor[4, f32], tensor[4, f32]) =
  {
    c = if b then drop(p.0) else ()
    p
  }
"#,
        "p",
        "p after a branch drop(p.0)",
    );
}

#[test]
fn a_disjoint_component_stays_usable_after_a_component_drop() {
    assert_accepted(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    c = drop(p.0)
    p.1
  }
"#,
        "p.1 after drop(p.0)",
    );
}

#[test]
fn an_ordinary_consume_of_a_component_leaves_the_tuple_usable() {
    assert_accepted(
        r#"
def eat(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(p: (tensor[4, f32], tensor[4, f32])) -> ((tensor[4, f32], tensor[4, f32]), tensor[4, f32]) =
  {
    a = eat(p.0)
    (p, a)
  }
"#,
        "eat(p.0) then p",
    );
}

#[test]
fn a_nested_component_drop_leaves_its_sibling_usable() {
    assert_accepted(
        r#"
def f(p: ((tensor[4, f32], tensor[4, f32]), tensor[4, f32])) -> tensor[4, f32] =
  {
    c = drop((p.0).1)
    (p.0).0
  }
"#,
        "p.0.0 after drop((p.0).1)",
    );
}

#[test]
fn a_nested_component_drop_leaves_its_enclosing_component_unusable() {
    assert_use_after_component_drop(
        r#"
def f(p: ((tensor[4, f32], tensor[4, f32]), tensor[4, f32])) -> (tensor[4, f32], tensor[4, f32]) =
  {
    c = drop((p.0).1)
    p.0
  }
"#,
        "p",
        "p.0 after drop((p.0).1)",
    );
}

#[test]
fn using_a_record_after_dropping_a_field_is_refused() {
    assert_use_after_component_drop(
        r#"
type Holder =
  | Holder { items: tensor[4, f32], other: tensor[4, f32] }
def f(h: Holder) -> Holder =
  {
    c = drop(h.items)
    h
  }
"#,
        "h",
        "whole record after drop(h.items)",
    );
}

#[test]
fn a_disjoint_field_stays_usable_after_a_field_drop() {
    assert_accepted(
        r#"
type Holder =
  | Holder { items: tensor[4, f32], other: tensor[4, f32] }
def f(h: Holder) -> tensor[4, f32] =
  {
    c = drop(h.items)
    h.other
  }
"#,
        "h.other after drop(h.items)",
    );
}

#[test]
fn an_ordinary_consume_of_a_field_leaves_the_record_usable() {
    assert_accepted(
        r#"
type Holder =
  | Holder { items: tensor[4, f32], other: tensor[4, f32] }
def eat(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(h: Holder) -> (Holder, tensor[4, f32]) =
  {
    a = eat(h.items)
    (h, a)
  }
"#,
        "eat(h.items) then h",
    );
}

// Round 1 of #3205: which earlier consumes a `drop` may follow, and naming
// the source binding after a destructure.

#[test]
fn a_drop_after_realize_is_accepted() {
    assert_accepted(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    a = realize(x)
    c = drop(x)
    a
  }
"#,
        "realize(x) then drop(x)",
    );
}

#[test]
fn a_drop_after_a_match_on_the_value_is_refused() {
    assert_refused(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    r = match p with {
      | (a, b) => b
    }
    c = drop(p)
    r
  }
"#,
        CheckErrorKind::UseAfterConsume,
        &["variable `p`", "match scrutinee"],
        "drop after a match scrutinee",
    );
}

#[test]
fn a_drop_after_a_consuming_capture_is_refused() {
    assert_refused(
        r#"
def eat(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    g = fn (k: i32) -> eat(x)
    c = drop(x)
    g(1)
  }
"#,
        CheckErrorKind::UseAfterConsume,
        &["variable `x`", "closure capture"],
        "drop after a consuming capture",
    );
}

#[test]
fn destructuring_after_a_component_drop_names_the_source_binding() {
    let errors = linearity(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    c = drop(p.0)
    (a, b) = p
    b
  }
"#,
    )
    .expect_err("destructuring p after drop(p.0) must be refused");
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains("variable `p`")
                && error.message.contains("call to `drop`")
        }),
        "expected UseAfterConsume naming `p`; got {errors:?}"
    );
    assert!(
        errors
            .iter()
            .all(|error| !error.message.contains("__chelis_tmp")),
        "no diagnostic may name a desugarer binding; got {errors:?}"
    );
}

// Round 1 addendum of #3205: a source-spelled destructured component is an
// owner like any other, and a destructuring `let` is an ordinary consume.

#[test]
fn using_a_component_whole_after_dropping_its_projection_is_refused() {
    assert_use_after_component_drop(
        r#"
def f(p: ((tensor[4, f32], tensor[4, f32]), tensor[4, f32])) -> ((tensor[4, f32], tensor[4, f32]), tensor[4, f32]) =
  {
    (a, b) = p
    c = drop(a.0)
    (a, b)
  }
"#,
        "a",
        "component a whole after drop(a.0)",
    );
}

#[test]
fn projecting_a_dropped_projection_of_a_component_again_is_refused() {
    assert_use_after_component_drop(
        r#"
def f(p: ((tensor[4, f32], tensor[4, f32]), tensor[4, f32])) -> tensor[4, f32] =
  {
    (a, b) = p
    c = drop(a.0)
    a.0
  }
"#,
        "a",
        "a.0 after drop(a.0)",
    );
}

#[test]
fn using_a_component_after_dropping_a_projection_through_its_alias_is_refused() {
    assert_use_after_component_drop(
        r#"
def f(p: ((tensor[4, f32], tensor[4, f32]), tensor[4, f32])) -> (tensor[4, f32], tensor[4, f32]) =
  {
    (a, b) = p
    q = a
    c = drop(q.0)
    a
  }
"#,
        "a",
        "a after drop(q.0) with q = a",
    );
}

#[test]
fn a_drop_after_a_destructuring_let_is_accepted() {
    // A destructuring `let` consumes its value ordinarily (spec/04 section
    // 8.3), so the later `drop` is fan-out repaired by copy insertion.
    assert_accepted(
        r#"
def f(p: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (a, b) = p
    c = drop(p)
    b
  }
"#,
        "(a, b) = p then drop(p)",
    );
}
