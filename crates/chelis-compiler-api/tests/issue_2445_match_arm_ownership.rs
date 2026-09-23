//! chelis#2445, round-2 review of its C lowering: an arm that a `bool` test
//! decides must not leak the owned ADT value it matched.
//!
//! When an arm's pattern and guard have more than one failure exit, host
//! lowering decides the arm with a `bool` test and the selected body
//! destructures the scrutinee again ([04-PAT-2]). Each pass carries only what
//! it needs: the test carries the pattern's tests and the names its guard
//! reads, and the selected pass carries the pattern's constructor and its
//! bindings. Before that, the test read owned fields that nothing used, and
//! the selected pass re-tested literals and nested constructors that bound
//! nothing, which put a branch inside the arm. An owned ADT scrutinee is not
//! released on a branching arm (chelis#2458), so programs that were balanced
//! before the planner leaked on every call.
//!
//! Oracle: the compiled program runs against the `ownership-ledger` runtime,
//! and every allocation must be finalized with no live owner left. A nested
//! constructor sub-pattern that binds a name, such as `Box(Some(n))`, still
//! branches inside the selected arm and is chelis#2458's to fix; it is not
//! claimed here.

mod ownership_support;

fn assert_balanced(name: &str, source: &str, expected: &str) {
    let generated = ownership_support::emit(source, name);
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    assert_eq!(stdout, expected, "{name}");
}

/// REGRESSION TEST. On `4e905b45b` the ledger ended with 2000 live owners,
/// and `leaks --atExit` counted 5000 leaked blocks.
#[test]
fn a_constructor_arm_with_a_literal_field_releases_its_scrutinee_on_every_call() {
    assert_balanced(
        "boxed_literal",
        "type Box =\n  | Box(i64)\n  | NoBox\n\
         def f(b: Box) -> i32 =\n  match b with {\n    | Box(3) => 1\n    | _ => 0\n  }\n\
         def repeat(k: i64, acc: i32) -> i32 =\n  \
         if eq(k, 0i64) then acc else repeat(sub(k, 1i64), add(acc, f(Box(3i64))))\n\
         a = repeat(1000i64, 0)\n",
        "a = 1000\n",
    );
}

/// REGRESSION TEST. On `4e905b45b` the ledger ended with 3 live owners, and
/// `leaks --atExit` counted 5 leaked blocks.
#[test]
fn a_literal_after_a_wildcard_field_releases_its_scrutinee() {
    assert_balanced(
        "second_field_literal",
        "type Box =\n  | Box(string, i64)\n  | NoBox\n\
         def f(b: Box) -> i32 =\n  match b with {\n    | Box(_, 3) => 1\n    | _ => 0\n  }\n\
         a = f(Box(\"s\", 3i64))\n\
         b = f(Box(\"s\", 4i64))\n",
        "a = 1\nb = 0\n",
    );
}

/// REGRESSION TEST. On `4e905b45b` the ledger ended with 3 live owners, and
/// `leaks --atExit` counted 5 leaked blocks.
#[test]
fn a_record_arm_with_a_literal_field_releases_its_scrutinee() {
    assert_balanced(
        "record_literal",
        "type Shape =\n  | Circle { r: f32 }\n  | Rect { w: f32, h: f32, tag: string }\n\
         def cls(s: Shape) -> i32 =\n  match s with {\n    | Rect { h: 2.0 } => 2\n    | _ => 0\n  }\n\
         b = cls(Rect { w: 5.0, h: 2.0, tag: \"t\" })\n\
         c = cls(Circle { r: 1.0 })\n",
        "b = 2\nc = 0\n",
    );
}

/// `f` called 100 times on `arg`, summed, for a program whose `f` holds the
/// arm under test.
fn repeated(declarations: &str, arm: &str, result: &str, arg: &str) -> String {
    format!(
        "{declarations}\
         def f(v: Wrapped) -> {result} =\n  match v with {{\n    | {arm} => 1{result}\n    | _ => 0{result}\n  }}\n\
         def repeat(k: i64, acc: {result}) -> {result} =\n  \
         if eq(k, 0i64) then acc else repeat(sub(k, 1i64), add(acc, f({arg})))\n\
         a = repeat(100i64, 0{result})\n"
    )
}

/// REGRESSION TEST. On `d23960ff5` the arm's test read the string field
/// bound to `s`, which nothing uses, and released it on neither branch: the
/// ledger ended with 100 live owners, and `leaks --atExit` counted 300 leaked
/// blocks.
#[test]
fn a_test_does_not_read_a_field_only_the_body_binds() {
    assert_balanced(
        "unused_owned_field",
        &repeated(
            "type Wrapped =\n  | Box(i64, string)\n  | NoBox\n",
            "Box(3, s)",
            "i64",
            "Box(3i64, \"abc\")",
        ),
        "a = 100\n",
    );
}

/// REGRESSION TEST. On `d23960ff5` the selected pass re-tested `ModeA`, a
/// nested constructor that binds nothing: the ledger ended with 400 live
/// owners, and `leaks --atExit` counted 900 leaked blocks.
#[test]
fn a_nested_nullary_constructor_is_not_tested_again() {
    assert_balanced(
        "nested_nullary",
        &repeated(
            "type Mode =\n  | ModeA\n  | ModeB\ntype Wrapped =\n  | W(Mode)\n  | X\n",
            "W(ModeA)",
            "i32",
            "W(ModeA)",
        ),
        "a = 100\n",
    );
}

/// REGRESSION TEST. On `d23960ff5` the selected pass re-tested `Some(_)`:
/// the ledger ended with 300 live owners, and `leaks --atExit` counted 600
/// leaked blocks.
#[test]
fn a_nested_option_that_binds_nothing_is_not_tested_again() {
    assert_balanced(
        "nested_option",
        &repeated(
            "type Wrapped =\n  | Box(Option[i64])\n  | NoBox\n",
            "Box(Some(_))",
            "i32",
            "Box(Some(3i64))",
        ),
        "a = 100\n",
    );
}

/// REGRESSION TEST. On `d23960ff5` the selected pass re-tested `Nil`: the
/// ledger ended with 400 live owners, and `leaks --atExit` counted 900 leaked
/// blocks.
#[test]
fn a_nested_empty_list_is_not_tested_again() {
    assert_balanced(
        "nested_nil",
        &repeated(
            "type Wrapped =\n  | Items(List[i64], string)\n  | Empty\n",
            "Items(Nil, _)",
            "i32",
            "Items([], \"x\")",
        ),
        "a = 100\n",
    );
}
