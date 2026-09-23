//! chelis#2445, round-2 review of its C lowering: an arm that a `bool` test
//! decides must not leak the owned ADT value it matched.
//!
//! When an arm's pattern and guard have more than one failure exit, host
//! lowering decides the arm with a `bool` test and the selected body
//! destructures the scrutinee again ([04-PAT-2]). Testing a literal
//! sub-pattern again in that second destructuring put a branch inside the arm,
//! and an owned ADT scrutinee is not released on a branching arm
//! (chelis#2458), so programs that were balanced before the planner leaked
//! their scrutinee on every call. The second destructuring now skips the
//! literals the test already matched.
//!
//! Oracle: the compiled program runs against the `ownership-ledger` runtime,
//! and every allocation must be finalized with no live owner left. Nested
//! constructor sub-patterns such as `Box(Some(n))` still branch inside the
//! arm and are chelis#2458's to fix; they are not claimed here.

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
