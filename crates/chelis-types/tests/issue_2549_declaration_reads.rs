//! chelis#2549: a top-level function declaration reads the top-level values
//! its body names; it does not capture them.
//!
//! `spec/02-surf-syntax.md` scopes a named declaration's free references to
//! the declaration scope, and only an anonymous `fn` captures lexical
//! bindings at its creation site. [04-INF-7] makes a top-level `def` whose
//! initializer is a lambda a function declaration rather than an eager
//! value, and [04-LIN-4] makes each call's result a new owner. The checker
//! used to walk every `(def f (fn ...))` as a closure created in the
//! top-level scope, so a declaration whose body returned a top-level tensor
//! value recorded a structural closure-capture consume on that value, and a
//! second reader (another declaration, a top-level initializer, or a second
//! importing module of a package) was rejected with "already consumed by
//! closure capture".
//!
//! The negative controls pin what stays rejected: consumes inside one
//! declaration body, an anonymous closure created by an eager value
//! initializer ([04-LIN-2]), and a declaration that reads a value an
//! earlier initializer already consumed.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{
    build_type_env_from_library, check_ir_program, check_ir_with_context, check_linearity,
    check_linearity_with_context, check_typed_program,
};

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    desugar_program(&decls).expect("Surf fixture must desugar")
}

fn linearity(source: &str) -> Result<(), Vec<CheckError>> {
    let checked = check_typed_program(&surf_to_deep(source))
        .unwrap_or_else(|e| panic!("type check must succeed: {:?}", e.errors));
    check_linearity(&checked).map(|_| ())
}

fn linearity_with_library(library_src: &str, new_src: &str) -> Result<(), Vec<CheckError>> {
    let library_deep = surf_to_deep(library_src);
    let library_checked = check_ir_program(&library_deep)
        .unwrap_or_else(|e| panic!("library IR check failed: {:?}", e.errors));
    let library_program =
        check_linearity(&library_checked).expect("library linearity must be clean");
    let ctx = build_type_env_from_library(&library_deep).expect("library context build OK");
    let new_checked = check_ir_with_context(&ctx, &surf_to_deep(new_src))
        .unwrap_or_else(|e| panic!("with-context IR check failed: {:?}", e.errors));
    check_linearity_with_context(&library_program, &new_checked).map(|_| ())
}

fn assert_clean(result: Result<(), Vec<CheckError>>, what: &str) {
    if let Err(errors) = result {
        panic!("{what} must be linearity-clean; got {errors:?}");
    }
}

fn assert_use_after_consume(result: Result<(), Vec<CheckError>>, fragments: &[&str], what: &str) {
    let errors = result.expect_err(what);
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && fragments
                    .iter()
                    .all(|fragment| error.message.contains(fragment))
        }),
        "{what}: expected UseAfterConsume naming {fragments:?}; got {errors:?}"
    );
}

const TWO_DECLARATIONS: &str = r#"
sampled = to_tensor([1.0f32, 1.0f32])
def first() -> tensor[2, f32] = sampled
def second() -> tensor[2, f32] = sampled
def main() -> tensor[2, f32] = add(first(), second())
"#;

#[test]
fn two_declarations_returning_one_top_level_value_are_accepted() {
    assert_clean(
        linearity(TWO_DECLARATIONS),
        "two declarations reading `sampled`",
    );
}

#[test]
fn two_module_wrapped_declarations_are_accepted() {
    assert_clean(
        linearity(&format!("module Test\n{TWO_DECLARATIONS}")),
        "module-wrapped declarations reading `sampled`",
    );
}

#[test]
fn three_declarations_and_a_lambda_spelled_declaration_are_accepted() {
    // `third = fn () -> sampled` is the same Deep `(def third (fn ...))` as
    // the `def` spelling, so [04-INF-7] makes it a declaration too.
    assert_clean(
        linearity(
            r#"
sampled = to_tensor([1.0f32, 1.0f32])
def first() -> tensor[2, f32] = sampled
def second() -> tensor[2, f32] = sampled
third = fn () -> sampled
def main() -> tensor[2, f32] = add(add(first(), second()), third())
"#,
        ),
        "three declarations reading `sampled`",
    );
}

#[test]
fn declaration_then_top_level_initializer_reading_the_value_is_accepted() {
    assert_clean(
        linearity(
            r#"
sampled = to_tensor([1.0f32, 1.0f32])
def first() -> tensor[2, f32] = sampled
twice = add(first(), sampled)
"#,
        ),
        "a declaration and a top-level borrow of `sampled`",
    );
}

#[test]
fn declarations_reading_a_tensor_carrying_adt_value_are_accepted() {
    assert_clean(
        linearity(
            r#"
type Holder =
  | Holder { items: tensor[2, f32] }
held = Holder { items: to_tensor([1.0f32, 1.0f32]) }
def first() -> Holder = held
def second() -> Holder = held
"#,
        ),
        "two declarations reading the ADT value `held`",
    );
}

#[test]
fn two_new_code_declarations_reading_one_library_value_are_accepted() {
    // The package shape of chelis#2549: the library value is pre-declared in
    // the with-context scope and two importing modules each return it.
    assert_clean(
        linearity_with_library(
            "sampled = to_tensor([1.0f32, 1.0f32])",
            r#"
def main_a() -> tensor[2, f32] = sampled
def main_b() -> tensor[2, f32] = sampled
"#,
        ),
        "two new-code declarations reading library `sampled`",
    );
}

#[test]
fn consume_then_borrow_inside_one_declaration_is_still_rejected() {
    assert_use_after_consume(
        linearity(
            r#"
sampled = to_tensor([1.0f32, 1.0f32])
def first() -> tensor[2, f32] = {
  y = realize(sampled)
  add(sampled, y)
}
"#,
        ),
        &["variable `sampled`", "realize"],
        "a consume then borrow inside one declaration body",
    );
}

#[test]
fn local_closure_capture_inside_a_declaration_is_still_rejected_on_reuse() {
    assert_use_after_consume(
        linearity(
            r#"
sampled = to_tensor([1.0f32, 1.0f32])
def first() -> tensor[2, f32] = {
  g = fn () -> sampled
  add(g(), sampled)
}
"#,
        ),
        &["variable `sampled`", "closure capture"],
        "a local closure capture then reuse inside a declaration body",
    );
}

#[test]
fn eager_initializer_closure_capture_is_still_rejected_on_reuse() {
    // A lambda nested inside an eager value's initializer is a closure
    // created at that initializer ([04-LIN-2]), not a declaration.
    assert_use_after_consume(
        linearity(
            r#"
sampled = to_tensor([1.0f32, 1.0f32])
pair = (fn () -> sampled, 1)
twice = add(sampled, sampled)
"#,
        ),
        &["variable `sampled`", "closure capture"],
        "an eager initializer's closure capture then reuse",
    );
}

#[test]
fn declaration_reading_an_already_consumed_value_is_still_rejected() {
    assert_use_after_consume(
        linearity(
            r#"
sampled = to_tensor([1.0f32, 1.0f32])
y = realize(sampled)
def first() -> tensor[2, f32] = sampled
"#,
        ),
        &["variable `sampled`", "realize"],
        "a declaration reading a value an earlier initializer consumed",
    );
}
