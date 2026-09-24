//! Linearity-F3 PR 2 fixtures: top-level statements wrapped in a
//! `module` declaration must reject linearity violations as errors,
//! matching the bare-top-level path.
//!
//! Background: PR 1 (#65) extended the pre-declare loop and main walk
//! in `check_linearity` to recurse through `(module {} name ...)`
//! wrappers, and emitted the surfaced violations as warnings on
//! `LinearityInfo::warnings()` for a deprecation window. PR 1's corpus
//! sweep across `examples/`, `examples/illustrative/`, `packages/`,
//! and `crates/*/tests/` found zero surfaced warnings, so PR 2's
//! "fix surfaced violations" step was a no-op. PR 2 flips the severity:
//! module-wrapped violations now route through `Checker::errors`, the
//! warning plumbing on `LinearityInfo` is removed, and the CLI's
//! `warning: linearity: ...` stderr emission is gone.
//!
//! Each fixture asserts that the SAME violation that fires for a bare
//! top-level program ALSO surfaces as a hard error when the program
//! is wrapped in `module Test`. This is the post-PR-2 contract.

use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

/// Run the linearity check on a Deep fixture. Returns the errors so
/// fixtures that assert a violation can match on
/// `CheckErrorKind::UseAfterConsume` and message content.
fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let deep = chelis_deep::parser::parse_str(source).expect("Deep fixture");
    let checked = check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("type check should succeed");
    check_linearity(&checked, &mut chelis_types::NativeServices::new(None))
        .expect_err("linearity check must error on use-after-consume")
}

#[test]
fn bare_top_level_realize_then_borrow_errors_today() {
    let errors = linearity_errors(
        r#"(def {}
  x
  (app {span: "surf:5..31"}
    (var {span: "surf:5..14"} to_tensor)
    (app {span: "surf:15..30"}
      (var {} Cons)
      (lit {span: "surf:16..19", type: (t-prim {} f32)} 1.0)
      (app {}
        (var {} Cons)
        (lit {span: "surf:21..24", type: (t-prim {} f32)} 2.0)
        (app {}
          (var {} Cons)
          (lit {span: "surf:26..29", type: (t-prim {} f32)} 3.0)
          (var {} Nil))))))

(def {} y (realize {span: "surf:36..46"} (var {span: "surf:44..45"} x)))

(def {}
  b
  (app {span: "surf:51..60"}
    (var {span: "surf:51..54"} add)
    (var {span: "surf:55..56"} x)
    (var {span: "surf:58..59"} y)))
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("realize")
        }),
        "bare top-level realize-then-borrow must still error (this is the control); got {errors:?}"
    );
}

#[test]
fn bare_top_level_consuming_call_then_borrow_errors_today() {
    let errors = linearity_errors(
        r#"(defsig {}
  consume_it
  (t-fn {}
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))))

(def {}
  consume_it
  (fn {}
    (params {} (t {type: (t-var {} _)}))
    (realize {span: "surf:55..65"} (var {span: "surf:63..64"} t))))

(def {}
  x
  (app {span: "surf:71..97"}
    (var {span: "surf:71..80"} to_tensor)
    (app {span: "surf:81..96"}
      (var {} Cons)
      (lit {span: "surf:82..85", type: (t-prim {} f32)} 1.0)
      (app {}
        (var {} Cons)
        (lit {span: "surf:87..90", type: (t-prim {} f32)} 2.0)
        (app {}
          (var {} Cons)
          (lit {span: "surf:92..95", type: (t-prim {} f32)} 3.0)
          (var {} Nil))))))

(def {}
  y
  (app {span: "surf:102..115"}
    (var {span: "surf:102..112"} consume_it)
    (var {span: "surf:113..114"} x)))

(def {}
  b
  (app {span: "surf:120..129"}
    (var {span: "surf:120..123"} mul)
    (var {span: "surf:124..125"} x)
    (var {span: "surf:127..128"} y)))
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("consume_it")
        }),
        "bare top-level consuming-call-then-borrow must still error (control); got {errors:?}"
    );
}

#[test]
fn bare_top_level_multi_realize_then_borrow_errors_today() {
    let errors = linearity_errors(
        r#"(def {}
  x
  (app {span: "surf:5..31"}
    (var {span: "surf:5..14"} to_tensor)
    (app {span: "surf:15..30"}
      (var {} Cons)
      (lit {span: "surf:16..19", type: (t-prim {} f32)} 1.0)
      (app {}
        (var {} Cons)
        (lit {span: "surf:21..24", type: (t-prim {} f32)} 2.0)
        (app {}
          (var {} Cons)
          (lit {span: "surf:26..29", type: (t-prim {} f32)} 3.0)
          (var {} Nil))))))

(def {} y (realize {span: "surf:36..46"} (var {span: "surf:44..45"} x)))

(def {} b (realize {span: "surf:51..61"} (var {span: "surf:59..60"} x)))

(def {} c (realize {span: "surf:66..76"} (var {span: "surf:74..75"} x)))

(def {}
  d
  (app {span: "surf:81..90"}
    (var {span: "surf:81..84"} add)
    (var {span: "surf:85..86"} x)
    (var {span: "surf:88..89"} y)))
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("realize")
        }),
        "bare top-level multi-realize-then-borrow must still error (control); got {errors:?}"
    );
}

#[test]
fn module_wrapped_realize_then_borrow_errors() {
    // The same statements as `bare_top_level_realize_then_borrow_errors_today`,
    // but wrapped in `module Test`. After Linearity-F3 PR 2 the
    // module-recursive walk must surface the violation as a hard error
    // — the warning-mode plumbing from PR 1 is gone.
    let errors = linearity_errors(
        r#"(module {surf_path: "Test"}
  test
  (def {}
    x
    (app {span: "surf:18..44"}
      (var {span: "surf:18..27"} to_tensor)
      (app {span: "surf:28..43"}
        (var {} Cons)
        (lit {span: "surf:29..32", type: (t-prim {} f32)} 1.0)
        (app {}
          (var {} Cons)
          (lit {span: "surf:34..37", type: (t-prim {} f32)} 2.0)
          (app {}
            (var {} Cons)
            (lit {span: "surf:39..42", type: (t-prim {} f32)} 3.0)
            (var {} Nil))))))
  (def {} y (realize {span: "surf:49..59"} (var {span: "surf:57..58"} x)))
  (def {}
    b
    (app {span: "surf:64..73"}
      (var {span: "surf:64..67"} add)
      (var {span: "surf:68..69"} x)
      (var {span: "surf:71..72"} y))))
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("realize")
        }),
        "expected a UseAfterConsume error for `x` consumed by `realize`; got {errors:?}"
    );
}

#[test]
fn module_wrapped_consuming_call_then_borrow_errors() {
    // Mirrors `bare_top_level_consuming_call_then_borrow_errors_today`
    // but inside a module. The consume site is the call to
    // `consume_it`, so the error message must call that out.
    let errors = linearity_errors(
        r#"(module {surf_path: "Test"}
  test
  (defsig {}
    consume_it
    (t-fn {}
      (t-tensor {} (d-lit {} 3) (t-prim {} f32))
      (t-tensor {} (d-lit {} 3) (t-prim {} f32))))
  (def {}
    consume_it
    (fn {}
      (params {} (t {type: (t-var {} _)}))
      (realize {span: "surf:68..78"} (var {span: "surf:76..77"} t))))
  (def {}
    x
    (app {span: "surf:84..110"}
      (var {span: "surf:84..93"} to_tensor)
      (app {span: "surf:94..109"}
        (var {} Cons)
        (lit {span: "surf:95..98", type: (t-prim {} f32)} 1.0)
        (app {}
          (var {} Cons)
          (lit {span: "surf:100..103", type: (t-prim {} f32)} 2.0)
          (app {}
            (var {} Cons)
            (lit {span: "surf:105..108", type: (t-prim {} f32)} 3.0)
            (var {} Nil))))))
  (def {}
    y
    (app {span: "surf:115..128"}
      (var {span: "surf:115..125"} consume_it)
      (var {span: "surf:126..127"} x)))
  (def {}
    b
    (app {span: "surf:133..142"}
      (var {span: "surf:133..136"} mul)
      (var {span: "surf:137..138"} x)
      (var {span: "surf:140..141"} y))))
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("consume_it")
        }),
        "expected a UseAfterConsume error for `x` consumed by `consume_it`; got {errors:?}"
    );
}

#[test]
fn module_wrapped_multi_realize_then_borrow_errors() {
    // PR #60 (V2-F4) tightened the binding-aliasing path for bare
    // top-level statements (`y = x` chains). After Linearity-F3 PR 2
    // the same chain wrapped in `module Test` must surface the use-
    // after-consume as a hard error, not a warning.
    let errors = linearity_errors(
        r#"(module {surf_path: "Test"}
  test
  (def {}
    x
    (app {span: "surf:18..44"}
      (var {span: "surf:18..27"} to_tensor)
      (app {span: "surf:28..43"}
        (var {} Cons)
        (lit {span: "surf:29..32", type: (t-prim {} f32)} 1.0)
        (app {}
          (var {} Cons)
          (lit {span: "surf:34..37", type: (t-prim {} f32)} 2.0)
          (app {}
            (var {} Cons)
            (lit {span: "surf:39..42", type: (t-prim {} f32)} 3.0)
            (var {} Nil))))))
  (def {} y (realize {span: "surf:49..59"} (var {span: "surf:57..58"} x)))
  (def {} b (realize {span: "surf:64..74"} (var {span: "surf:72..73"} x)))
  (def {} c (realize {span: "surf:79..89"} (var {span: "surf:87..88"} x)))
  (def {}
    d
    (app {span: "surf:94..103"}
      (var {span: "surf:94..97"} add)
      (var {span: "surf:98..99"} x)
      (var {span: "surf:101..102"} y))))
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("realize")
        }),
        "expected a UseAfterConsume error for `x` consumed by `realize`; got {errors:?}"
    );
}
