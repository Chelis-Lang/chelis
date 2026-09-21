//! Issue #2292: `List` is the builtin `Cons`/`Nil` ADT in both execution lanes.

mod common;

use assert_cmd::Command;
use common::{build_and_run, write_file};

fn evaluate(source: &str, name: &str) -> String {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .expect("eval should run");
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8 stdout")
}

#[test]
fn direct_cons_nil_patterns_match_in_eval_and_c() {
    let source = r#"
def head_or(xs: List[i64], fallback: i64) -> i64 =
  match xs with {
    | Cons(head, tail) => add(head, len(tail))
    | Nil => fallback
  }
def nil_first(xs: List[i64]) -> i64 =
  match xs with {
    | Nil => 10i64
    | _ => 20i64
  }
def cons_first(xs: List[i64]) -> i64 =
  match xs with {
    | Cons(head, tail) => add(head, len(tail))
    | _ => -1i64
  }
cons_value = head_or([7i64, 2i64], 99i64)
nil_value = head_or(Nil, 99i64)
nil_fallback = nil_first([1i64])
cons_fallback = cons_first(Nil)
"#;

    let interpreted = evaluate(source, "direct_list_patterns");
    let compiled = build_and_run(source, "direct_list_patterns");
    assert_eq!(
        compiled, interpreted,
        "eval/C output must be byte-identical"
    );
    for expected in [
        "cons_value = 8",
        "nil_value = 99",
        "nil_fallback = 20",
        "cons_fallback = -1",
    ] {
        assert!(
            interpreted.lines().any(|line| line == expected),
            "{interpreted}"
        );
    }
}

#[test]
fn nested_list_patterns_preserve_pattern_semantics_in_eval_and_c() {
    let source = r#"
type Boxed = | Boxed(i64)
type Named = | Named { value: i64 }
def exact_pair(xs: List[i64]) -> i64 =
  match xs with {
    | Cons(7, Cons(second, Nil)) => second
    | _ => -1i64
  }
def whole_length(xs: List[i64]) -> i64 =
  match xs with {
    | whole @ Cons(_, _) => len(whole)
    | Nil => 0i64
  }
def tuple_head(xs: List[(i64, i64)]) -> i64 =
  match xs with {
    | Cons((left, right), Nil) => add(left, right)
    | _ => -1i64
  }
def boxed_head(xs: List[Boxed]) -> i64 =
  match xs with {
    | Cons(Boxed(value), Nil) => value
    | _ => -1i64
  }
def named_head(xs: List[Named]) -> i64 =
  match xs with {
    | Cons(Named { value }, Nil) => value
    | _ => -1i64
  }
def optional_head(xs: List[Option[i64]]) -> i64 =
  match xs with {
    | Cons(Some(value), Nil) => value
    | _ => -1i64
  }
def nested_head(xs: List[List[i64]]) -> i64 =
  match xs with {
    | Cons(Cons(value, Nil), Nil) => value
    | _ => -1i64
  }
def narrow_literal(xs: List[i32]) -> i64 =
  match xs with {
    | Cons(19, Nil) => 1i64
    | _ => 0i64
  }
exact = exact_pair([7i64, 4i64])
literal_fallback = exact_pair([8i64, 4i64])
whole = whole_length([1i64, 2i64, 3i64])
tupled = tuple_head([(3i64, 4i64)])
boxed = boxed_head([Boxed(9i64)])
named = named_head([Named { value: 11i64 }])
optional = optional_head([Some(13i64)])
nested = nested_head([[17i64]])
narrow = narrow_literal([19i32])
"#;

    let interpreted = evaluate(source, "nested_list_patterns");
    let compiled = build_and_run(source, "nested_list_patterns");
    assert_eq!(
        compiled, interpreted,
        "eval/C output must be byte-identical"
    );
    for expected in [
        "exact = 4",
        "literal_fallback = -1",
        "whole = 3",
        "tupled = 7",
        "boxed = 9",
        "named = 11",
        "optional = 13",
        "nested = 17",
        "narrow = 1",
    ] {
        assert!(
            interpreted.lines().any(|line| line == expected),
            "{interpreted}"
        );
    }
}

#[test]
fn cons_without_nil_remains_a_checker_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("non_exhaustive.ch");
    write_file(
        &path,
        r#"
def head(xs: List[i64]) -> i64 =
  match xs with {
    | Cons(value, tail) => value
  }
result = head([1i64])
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check"])
        .arg(path)
        .output()
        .expect("check should run");
    assert!(
        !output.status.success(),
        "a checker rejection must produce a failing process status"
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("check JSON");
    assert!(
        report["score"].as_f64().is_some_and(|score| score < 1.0),
        "{report}"
    );
    assert!(
        report["errors"]
            .as_array()
            .is_some_and(|errors| errors.iter().any(|error| {
                error["kind"] == "NonExhaustiveMatch"
                    && error["message"]
                        .as_str()
                        .is_some_and(|message| message.contains("Nil"))
            })),
        "{report}"
    );
}
