//! chelis#3177: `drop(x)` ends the owner's lifetime at the call ([05-OP-67]),
//! so no inserted copy can keep `x` usable afterward ([04-LIN-3]). Every later
//! use of `x` is `UseAfterConsume`, whether it borrows `x`, passes it to a
//! consuming call, or moves it by bare name into a result, an aggregate, or a
//! new binding. Before the fix only a borrowing call argument was refused.

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
fn assert_use_after_drop(source: &str, name: &str, what: &str) {
    let errors = linearity(source).expect_err(&format!(
        "{what}: a use of `{name}` after drop({name}) must be refused"
    ));
    let variable = format!("variable `{name}`");
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains(&variable)
                && error.message.contains("call to `drop`")
        }),
        "{what}: expected UseAfterConsume naming `{name}` and the drop; got {errors:?}"
    );
}

#[track_caller]
fn assert_accepted(source: &str, what: &str) {
    if let Err(errors) = linearity(source) {
        panic!("{what}: expected the program to be accepted; got {errors:?}");
    }
}

#[test]
fn tail_return_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    c = drop(x)
    x
  }
"#,
        "x",
        "tail return",
    );
}

#[test]
fn tuple_element_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> (tensor[4, f32], i32) =
  {
    c = drop(x)
    (x, 1)
  }
"#,
        "x",
        "tuple element",
    );
}

#[test]
fn record_field_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
type Holder =
  | Holder { items: tensor[4, f32] }
def f(x: tensor[4, f32]) -> Holder =
  {
    c = drop(x)
    Holder { items: x }
  }
"#,
        "x",
        "record field",
    );
}

#[test]
fn list_element_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> List[tensor[4, f32]] =
  {
    c = drop(x)
    [x]
  }
"#,
        "x",
        "list element",
    );
}

#[test]
fn let_binding_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    c = drop(x)
    y = x
    y
  }
"#,
        "x",
        "let right-hand side",
    );
}

#[test]
fn top_level_binding_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
x = to_tensor([1.5f32, 2.5f32, 3.5f32])
d = drop(x)
y = x
"#,
        "x",
        "top-level binding",
    );
}

#[test]
fn declaration_reading_a_dropped_top_level_value_is_refused() {
    assert_use_after_drop(
        r#"
x = to_tensor([1.5f32, 2.5f32, 3.5f32])
d = drop(x)
def g() -> tensor[3, f32] = x
"#,
        "x",
        "top-level declaration body",
    );
}

#[test]
fn if_arms_after_drop_are_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32], z: tensor[4, f32], b: bool) -> tensor[4, f32] =
  {
    c = drop(x)
    if b then x else z
  }
"#,
        "x",
        "if arm",
    );
}

#[test]
fn match_arm_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32], n: i32) -> tensor[4, f32] =
  {
    c = drop(x)
    match n with {
      | 0 => x
      | _ => x
    }
  }
"#,
        "x",
        "match arm",
    );
}

#[test]
fn match_scrutinee_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(p: (tensor[4, f32], i32)) -> i32 =
  {
    c = drop(p)
    match p with {
      | (t, n) => n
    }
  }
"#,
        "p",
        "match scrutinee",
    );
}

#[test]
fn consuming_closure_capture_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    c = drop(x)
    g = fn () -> x
    g()
  }
"#,
        "x",
        "consuming closure capture",
    );
}

#[test]
fn consuming_call_argument_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def eat(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    c = drop(x)
    eat(x)
  }
"#,
        "x",
        "consuming call argument",
    );
}

#[test]
fn moving_an_alias_of_a_dropped_value_is_refused() {
    // [04-LIN-3]: `y = x` names the same owner, so dropping one name ends both.
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y = x
    c = drop(y)
    x
  }
"#,
        "x",
        "source of a dropped alias",
    );
}

#[test]
fn move_after_a_drop_in_one_branch_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32], b: bool) -> tensor[4, f32] =
  {
    c = if b then drop(x) else ()
    x
  }
"#,
        "x",
        "move after a branch drop",
    );
}

#[test]
fn second_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> i32 =
  {
    c = drop(x)
    d = drop(x)
    1
  }
"#,
        "x",
        "second drop",
    );
}

// Negative twins.

#[test]
fn borrow_after_drop_is_refused() {
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    c = drop(x)
    exp(x)
  }
"#,
        "x",
        "borrowing call argument",
    );
}

#[test]
fn moving_a_different_variable_after_drop_is_accepted() {
    assert_accepted(
        r#"
def f(x: tensor[4, f32], z: tensor[4, f32]) -> (tensor[4, f32], i32) =
  {
    c = drop(x)
    (z, 1)
  }
"#,
        "moving `z` after dropping `x`",
    );
}

#[test]
fn moving_before_drop_through_a_copy_is_accepted() {
    assert_accepted(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y = copy(x)
    c = drop(x)
    y
  }
"#,
        "an explicit copy taken before the drop",
    );
}

#[test]
fn drop_after_a_borrow_is_accepted() {
    assert_accepted(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y = exp(x)
    c = drop(x)
    y
  }
"#,
        "borrowing `x` before dropping it",
    );
}

#[test]
fn ordinary_consuming_fanout_is_still_accepted() {
    // A second consume after an ordinary consume is repaired by copy insertion
    // (spec/04 section 8.3); only `drop` ends the lifetime outright.
    assert_accepted(
        r#"
def eat(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) =
  {
    a = eat(x)
    (a, x)
  }
"#,
        "ordinary consuming fan-out",
    );
}

#[test]
fn a_user_function_named_drop_does_not_end_the_lifetime() {
    // Only the builtin `drop` ends a lifetime; a local binding that shadows the
    // name is an ordinary consuming call.
    assert_accepted(
        r#"
def f(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) =
  {
    drop = fn (t: tensor[4, f32]) -> t
    a = drop(x)
    (a, x)
  }
"#,
        "a shadowing local named drop",
    );
}

#[test]
fn binding_an_alias_of_a_dropped_value_is_refused() {
    // `z = y` is an aliasing bind whose own target `y` is live, but `y` names
    // the owner `drop(x)` ended.
    assert_use_after_drop(
        r#"
def f(x: tensor[4, f32]) -> tensor[4, f32] =
  {
    y = x
    c = drop(x)
    z = y
    z
  }
"#,
        "y",
        "alias bound after the drop",
    );
}

#[test]
fn move_after_a_drop_that_follows_an_ordinary_consume_is_refused() {
    // Copy insertion repairs `eat(x)` by copying into it, so the `drop` still
    // ends the original owner and the tail `x` is a use after it.
    assert_use_after_drop(
        r#"
def eat(t: tensor[4, f32]) -> tensor[4, f32] = t
def f(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) =
  {
    a = eat(x)
    c = drop(x)
    (a, x)
  }
"#,
        "x",
        "move after consume then drop",
    );
}
