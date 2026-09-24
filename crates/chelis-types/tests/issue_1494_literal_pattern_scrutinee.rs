//! Issue #1494: a literal pattern is a typing constraint on the scrutinee.
//!
//! `spec/04-type-system.md` [04-PAT-1] states the rule this suite exercises.
//! A literal pattern's value atom must agree with the scrutinee's primitive
//! under [04-LIT-1]'s closed pairing, a non-primitive scrutinee admits no
//! literal pattern at all, and an integer pattern outside the scrutinee
//! width's range is rejected under the same range rule §5.3 and §5.6 apply to
//! a literal bound at that type. Before the fix `pattern_bindings` did nothing
//! at `pat-lit`, so an `f32` pattern against an `i32` scrutinee scored a
//! clean 1.0 and produced a silently dead arm.
//!
//! Two properties are asserted for every fixture. Each negative rejects with a
//! `TypeMismatch` naming [04-PAT-1], and the two checker ingresses return the
//! same diagnostics for the same program (chelis#1107): the check lives in the
//! one `pattern_bindings` walk both ingresses share, and these tests are what
//! prove that rather than assume it.
//!
//! What is claimed is exactly the forms named in the tests below: the four
//! atom families against a disagreeing primitive, the non-primitive scrutinee
//! shapes named in `literal_pattern_against_a_non_primitive_scrutinee_rejects`,
//! the integer range boundaries at `i8`, and the tuple, record, and
//! constructor nestings. It is not a claim about every dtype pairing or every
//! pattern shape.

use chelis_deep::Expr;

use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn parse_core(source: &str) -> Vec<Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("valid core fixture")
}

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|e| format!("[{:?}] {}", e.kind, e.message))
        .collect();
    out.sort();
    out
}

/// Every diagnostic for `source`, having first asserted that the stamped
/// ingress (`check_typed_program`) and the normalizing ingress
/// (`check_ir_program`) agree on it.
fn agreed_diagnostics(source: &str, label: &str) -> Vec<String> {
    let typed = match check_typed_program(
        &parse_core(source),
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    let ir = match check_ir_program(
        &parse_core(source),
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    assert_eq!(
        typed,
        ir,
        "{label}: both checker ingresses must return the same diagnostics \
         (chelis#1107).\ntyped-only: {:?}\nir-only: {:?}",
        typed.iter().filter(|m| !ir.contains(m)).collect::<Vec<_>>(),
        ir.iter().filter(|m| !typed.contains(m)).collect::<Vec<_>>(),
    );
    typed
}

/// Assert that `source` is rejected by both ingresses with exactly one
/// [04-PAT-1] `TypeMismatch`, and return its message.
fn sole_pattern_rejection(source: &str, label: &str) -> String {
    let diagnostics = agreed_diagnostics(source, label);
    let [only] = diagnostics.as_slice() else {
        panic!("{label}: expected exactly one diagnostic, got {diagnostics:?}");
    };
    assert!(
        only.starts_with("[TypeMismatch] "),
        "{label}: [04-PAT-1] rejections are TypeMismatch, got {only}"
    );
    only.clone()
}

/// Assert that `source` type-checks on both ingresses.
fn accepts(source: &str, label: &str) {
    let diagnostics = agreed_diagnostics(source, label);
    assert!(
        diagnostics.is_empty(),
        "{label}: must type-check, got {diagnostics:?}"
    );
}

// ---------------------------------------------------------------------------
// Regression tests: the seven forms measured as score-1 dead arms on the
// pre-fix tree, plus the two further nestings. Every one is red before the fix.
// ---------------------------------------------------------------------------

/// REGRESSION TEST. The exact program from chelis#1494: an `f32` literal
/// pattern against an `i32` scrutinee. `chelis check` scored 1.0 with an
/// empty error list and `chelis eval` printed `r = 2.5`, because the `1.5` arm
/// can never match.
#[test]
fn float_pattern_against_an_integer_scrutinee_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/float_pattern_against_an_integer_scrutinee_rejects.0.deep"
        ),
        "f32 pattern vs i32 scrutinee",
    );
    assert!(
        message.contains("floating-point literal pattern `1.5`") && message.contains("`i32`"),
        "the diagnostic must name the pattern and the scrutinee dtype, got {message}"
    );
}

/// REGRESSION TEST. The issue's second spelling: the scrutinee is a call to a
/// function with no declared return type, so the scrutinee dtype comes from
/// inference rather than from an annotation. The rejection must not depend on
/// the callee being fully annotated.
#[test]
fn float_pattern_against_an_inferred_integer_scrutinee_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/float_pattern_against_an_inferred_integer_scrutinee_rejects.0.deep"
        ),
        "f32 pattern vs inferred i32 scrutinee",
    );
    assert!(
        message.contains("`i32`"),
        "the diagnostic must name the inferred i32 scrutinee, got {message}"
    );
}

/// REGRESSION TEST. The mirror direction: an integer literal pattern against
/// an `f32` scrutinee. [04-LIT-1]'s one cross-family form needs `lit` metadata
/// a `pat-lit` cannot carry, so an integer pattern does not reach a float
/// primitive, and the evaluator agrees: `| 1 =>` never matches `1.0f32`.
#[test]
fn integer_pattern_against_a_float_scrutinee_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/integer_pattern_against_a_float_scrutinee_rejects.0.deep"
        ),
        "integer pattern vs f32 scrutinee",
    );
    assert!(
        message.contains("integer literal pattern `1`") && message.contains("`f32`"),
        "the diagnostic must name the pattern and the scrutinee dtype, got {message}"
    );
}

/// REGRESSION TEST. A boolean pattern against an `i32` scrutinee.
#[test]
fn bool_pattern_against_an_integer_scrutinee_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/bool_pattern_against_an_integer_scrutinee_rejects.0.deep"
        ),
        "bool pattern vs i32 scrutinee",
    );
    assert!(
        message.contains("boolean literal pattern `true`") && message.contains("`i32`"),
        "the diagnostic must name the pattern and the scrutinee dtype, got {message}"
    );
}

/// REGRESSION TEST. A string pattern against an `i32` scrutinee.
#[test]
fn string_pattern_against_an_integer_scrutinee_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/string_pattern_against_an_integer_scrutinee_rejects.0.deep"
        ),
        "string pattern vs i32 scrutinee",
    );
    assert!(
        message.contains("string literal pattern") && message.contains("`i32`"),
        "the diagnostic must name the pattern and the scrutinee dtype, got {message}"
    );
}

/// REGRESSION TEST. An integer pattern outside the scrutinee width's range:
/// `300` can never equal an `i8` value, whose range is [-128, 127]. The
/// families agree, so only the range rule catches this one.
#[test]
fn out_of_range_integer_pattern_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/out_of_range_integer_pattern_rejects.0.deep"
        ),
        "300 pattern vs i8 scrutinee",
    );
    assert!(
        message.contains("`300`") && message.contains("[-128, 127]"),
        "the diagnostic must name the value and the i8 range, got {message}"
    );
}

/// REGRESSION TEST. Three non-primitive scrutinees, none of which admits a
/// literal pattern: a tensor, a nominal type, and a tuple. Each is a separate
/// concrete shape, so this claims those three and nothing wider.
#[test]
fn literal_pattern_against_a_non_primitive_scrutinee_rejects() {
    for (source, label, expected) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/literal_pattern_against_a_non_primitive_scrutinee_rejects.0.deep"
            ),
            "integer pattern vs tensor scrutinee",
            "tensor[3, i32]",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/literal_pattern_against_a_non_primitive_scrutinee_rejects.1.deep"
            ),
            "integer pattern vs ADT scrutinee",
            "Shape",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/literal_pattern_against_a_non_primitive_scrutinee_rejects.2.deep"
            ),
            "integer pattern vs tuple scrutinee",
            "(i32, f32)",
        ),
    ] {
        let message = sole_pattern_rejection(source, label);
        assert!(
            message.contains("only against a primitive scrutinee") && message.contains(expected),
            "{label}: the diagnostic must name the non-primitive scrutinee type \
             `{expected}`, got {message}"
        );
    }
}

/// REGRESSION TEST. A literal pattern nested inside a tuple pattern. The check
/// sits in the recursive `pattern_bindings` walk, so a nested violation is the
/// same finding at depth; this is the fixture that proves it rather than
/// assuming the recursion carries it.
#[test]
fn float_pattern_nested_in_a_tuple_pattern_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/float_pattern_nested_in_a_tuple_pattern_rejects.0.deep"
        ),
        "f32 pattern nested in a tuple pattern vs i32 component",
    );
    assert!(
        message.contains("floating-point literal pattern `1.5`") && message.contains("`i32`"),
        "the nested diagnostic must name the component dtype, got {message}"
    );
}

/// REGRESSION TEST. The same violation nested inside a record pattern, against
/// the declared field type rather than a tuple component.
#[test]
fn float_pattern_nested_in_a_record_pattern_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/float_pattern_nested_in_a_record_pattern_rejects.0.deep"
        ),
        "f32 pattern nested in a record pattern vs i32 field",
    );
    assert!(
        message.contains("floating-point literal pattern `1.5`") && message.contains("`i32`"),
        "the nested diagnostic must name the field dtype, got {message}"
    );
}

/// REGRESSION TEST. The same violation nested inside a positional constructor
/// pattern, against the declared variant argument type.
#[test]
fn float_pattern_nested_in_a_constructor_pattern_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/float_pattern_nested_in_a_constructor_pattern_rejects.0.deep"
        ),
        "f32 pattern nested in a constructor pattern vs i32 argument",
    );
    assert!(
        message.contains("floating-point literal pattern `1.5`") && message.contains("`i32`"),
        "the nested diagnostic must name the argument dtype, got {message}"
    );
}

/// REGRESSION TEST. A transparent alias resolves before classification, so an
/// alias to `i32` rejects the same float pattern its target does. Without
/// alias expansion the scrutinee would read as a nominal type and the
/// non-primitive arm would fire with the wrong reason.
#[test]
fn float_pattern_against_an_alias_of_an_integer_rejects() {
    let message = sole_pattern_rejection(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/float_pattern_against_an_alias_of_an_integer_rejects.0.deep"
        ),
        "f32 pattern vs an alias of i32",
    );
    assert!(
        message.contains("`i32`") && !message.contains("primitive scrutinee"),
        "the alias must resolve to i32 and take the family arm, got {message}"
    );
}

// ---------------------------------------------------------------------------
// Positive controls. These are the over-rejection guard: a literal pattern
// selects no width, so an unsuffixed integer pattern must keep matching every
// integer scrutinee. Unifying with §5.3's i32 default instead would reject
// most of this section.
// ---------------------------------------------------------------------------

/// DISPOSITION LOCK. An unsuffixed integer pattern is admissible against every
/// integer primitive, and an unsuffixed float pattern against every float
/// primitive. Green before the fix and green after; the fix must not narrow
/// these, because a `pat-lit` admits no suffix and so has no way to say which
/// width it meant.
#[test]
fn same_family_patterns_accept_at_every_width() {
    for (dtype, pattern, label) in [
        ("i8", "1", "i8 scrutinee"),
        ("i16", "1", "i16 scrutinee"),
        ("i32", "1", "i32 scrutinee"),
        ("i64", "1", "i64 scrutinee"),
    ] {
        accepts(
            &format!(
                r#"(defsig {{}} g (t-fn {{}} (t-prim {{}} {dtype}) (t-prim {{}} {dtype})))

(def {{}}
  g
  (fn {{}}
    (params {{}} (n {{type: (t-var {{}} _)}}))
    (app {{}}
      (var {{}} add)
      (lit {{type: (t-prim {{}} {dtype})}} 0)
      (var {{}} n))))

(defsig {{}} r (t-prim {{}} i32))

(def {{}}
  r
  (match {{}}
    (app {{}}
      (var {{}} g)
      (lit {{type: (t-prim {{}} {dtype})}} 1))
    (arm {{}}
      (pat-lit {{}} {pattern})
      ()
      (lit {{type: (t-prim {{}} i32)}} 10))
    (arm {{}}
      (pat-wild {{}})
      ()
      (lit {{type: (t-prim {{}} i32)}} 20))))
"#
            ),
            label,
        );
    }
    for (dtype, label) in [
        ("f32", "f32 scrutinee"),
        ("f64", "f64 scrutinee"),
        ("bf16", "bf16 scrutinee"),
        ("f16", "f16 scrutinee"),
    ] {
        accepts(
            &format!(
                r#"(defsig {{}} g (t-fn {{}} (t-prim {{}} {dtype}) (t-prim {{}} {dtype})))

(def {{}}
  g
  (fn {{}}
    (params {{}} (n {{type: (t-var {{}} _)}}))
    (app {{}}
      (var {{}} add)
      (lit {{type: (t-prim {{}} {dtype})}} 0.0)
      (var {{}} n))))

(defsig {{}} r (t-prim {{}} i32))

(def {{}}
  r
  (match {{}}
    (app {{}}
      (var {{}} g)
      (lit {{type: (t-prim {{}} {dtype})}} 1.0))
    (arm {{}}
      (pat-lit {{}} 1.5)
      ()
      (lit {{type: (t-prim {{}} i32)}} 10))
    (arm {{}}
      (pat-wild {{}})
      ()
      (lit {{type: (t-prim {{}} i32)}} 20))))
"#
            ),
            label,
        );
    }
}

/// DISPOSITION LOCK. The failure twin of the range rejection: both `i8`
/// boundaries are in range and must keep type-checking, and one step past
/// each boundary must reject. This is what pins the comparison to the exact
/// width rather than to some wider default.
#[test]
fn integer_range_boundaries_are_exact() {
    for value in ["127", "-128", "0"] {
        accepts(
            &format!(
                r#"(defsig {{}} g (t-fn {{}} (t-prim {{}} i8) (t-prim {{}} i8)))

(def {{}}
  g
  (fn {{}}
    (params {{}} (n {{type: (t-var {{}} _)}}))
    (app {{}}
      (var {{}} add)
      (lit {{type: (t-prim {{}} i8)}} 0)
      (var {{}} n))))

(defsig {{}} r (t-prim {{}} i32))

(def {{}}
  r
  (match {{}}
    (app {{}}
      (var {{}} g)
      (lit {{type: (t-prim {{}} i8)}} 1))
    (arm {{}}
      (pat-lit {{}} {value})
      ()
      (lit {{type: (t-prim {{}} i32)}} 10))
    (arm {{}}
      (pat-wild {{}})
      ()
      (lit {{type: (t-prim {{}} i32)}} 20))))
"#
            ),
            &format!("in-range i8 pattern {value}"),
        );
    }
    for value in ["128", "-129"] {
        let message = sole_pattern_rejection(
            &format!(
                r#"(defsig {{}} g (t-fn {{}} (t-prim {{}} i8) (t-prim {{}} i8)))

(def {{}}
  g
  (fn {{}}
    (params {{}} (n {{type: (t-var {{}} _)}}))
    (app {{}}
      (var {{}} add)
      (lit {{type: (t-prim {{}} i8)}} 0)
      (var {{}} n))))

(defsig {{}} r (t-prim {{}} i32))

(def {{}}
  r
  (match {{}}
    (app {{}}
      (var {{}} g)
      (lit {{type: (t-prim {{}} i8)}} 1))
    (arm {{}}
      (pat-lit {{}} {value})
      ()
      (lit {{type: (t-prim {{}} i32)}} 10))
    (arm {{}}
      (pat-wild {{}})
      ()
      (lit {{type: (t-prim {{}} i32)}} 20))))
"#
            ),
            &format!("out-of-range i8 pattern {value}"),
        );
        assert!(
            message.contains("[-128, 127]"),
            "the range rejection must name the i8 range, got {message}"
        );
    }
}

/// DISPOSITION LOCK. `bool` and `string` scrutinees keep matching their own
/// families. The failure twins are the bool-vs-int and string-vs-int
/// regressions above.
#[test]
fn bool_and_string_patterns_accept_their_own_scrutinee() {
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/bool_and_string_patterns_accept_their_own_scrutinee.0.deep"
        ),
        "bool pattern vs bool scrutinee",
    );
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/bool_and_string_patterns_accept_their_own_scrutinee.1.deep"
        ),
        "string pattern vs string scrutinee",
    );
}

/// DISPOSITION LOCK. The positive twins of the three nesting regressions: a
/// same-family literal nested in a tuple, record, and constructor pattern
/// still type-checks, so the recursion did not become a blanket rejection of
/// nested literals.
#[test]
fn same_family_patterns_accept_when_nested() {
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/same_family_patterns_accept_when_nested.0.deep"
        ),
        "int pattern nested in a tuple pattern vs i32 component",
    );
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/same_family_patterns_accept_when_nested.1.deep"
        ),
        "int pattern nested in a record pattern vs i32 field",
    );
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/same_family_patterns_accept_when_nested.2.deep"
        ),
        "int pattern nested in a constructor pattern vs i32 argument",
    );
}

/// DISPOSITION LOCK. A transparent alias of `i32` accepts an integer
/// pattern, the positive twin of the alias rejection above.
#[test]
fn same_family_pattern_accepts_against_an_alias() {
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/same_family_pattern_accepts_against_an_alias.0.deep"
        ),
        "int pattern vs an alias of i32",
    );
}

/// DISPOSITION LOCK. The non-literal pattern forms are untouched: a variable
/// pattern, a wildcard, an as-pattern, and a constructor pattern over a
/// non-primitive scrutinee all still bind and still type-check. The
/// non-primitive rejection is scoped to `pat-lit`, not to matching a
/// non-primitive scrutinee at all.
#[test]
fn non_literal_patterns_are_untouched() {
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/non_literal_patterns_are_untouched.0.deep"
        ),
        "constructor patterns over an ADT scrutinee",
    );
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/non_literal_patterns_are_untouched.1.deep"
        ),
        "variable pattern over a tensor scrutinee",
    );
    accepts(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/non_literal_patterns_are_untouched.2.deep"
        ),
        "as-pattern over an i32 scrutinee",
    );
}

/// Mismatched literal families reject across scalar scrutinee dtypes.
#[test]
fn mismatched_literal_families_reject_across_scrutinee_dtypes() {
    for (scrutinee, sample, pattern) in [
        ("bool", r#"(lit {type: (t-prim {} bool)} true)"#, "1"),
        ("string", r#"(lit {type: (t-prim {} string)} "a")"#, "1"),
        ("i32", r#"(lit {type: (t-prim {} i32)} 2)"#, "1.5"),
        ("i8", r#"(lit {type: (t-prim {} i8)} 1)"#, "1.5"),
        ("f32", r#"(lit {type: (t-prim {} f32)} 2.0)"#, "1"),
        ("f64", r#"(lit {type: (t-prim {} f64)} 2.0)"#, "1"),
        ("bf16", r#"(lit {type: (t-prim {} bf16)} 2.0)"#, "1"),
        ("f16", r#"(lit {type: (t-prim {} f16)} 2.0)"#, "1"),
    ] {
        sole_pattern_rejection(
            &format!(
                r#"(defsig {{}} g (t-fn {{}} (t-prim {{}} {scrutinee}) (t-prim {{}} {scrutinee})))

(def {{}}
  g
  (fn {{}}
    (params {{}} (v {{type: (t-var {{}} _)}}))
    (var {{}} v)))

(defsig {{}} r (t-prim {{}} i32))

(def {{}}
  r
  (match {{}}
    (app {{}}
      (var {{}} g)
      {sample})
    (arm {{}}
      (pat-lit {{}} {pattern})
      ()
      (lit {{type: (t-prim {{}} i32)}} 10))
    (arm {{}}
      (pat-wild {{}})
      ()
      (lit {{type: (t-prim {{}} i32)}} 20))))
"#
            ),
            &format!("{pattern} pattern vs {scrutinee} scrutinee"),
        );
    }
}

/// DISPOSITION LOCK. A `TypeMismatch` upstream of the match must not gain a
/// second, invented [04-PAT-1] rejection off the failed scrutinee type. The
/// check declines on an unresolved or already-failed scrutinee, so the root
/// cause stays single-sourced (chelis#731 §C3 cascade suppression).
#[test]
fn a_failed_scrutinee_does_not_cascade_a_pattern_rejection() {
    let diagnostics = agreed_diagnostics(
        include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/a_failed_scrutinee_does_not_cascade_a_pattern_rejection.0.deep"
        ),
        "ill-typed callee body under a well-formed match",
    );
    assert!(
        !diagnostics.is_empty(),
        "the ill-typed callee body must still be reported"
    );
    assert!(
        diagnostics.iter().all(|d| !d.contains("[04-PAT-1]")),
        "a well-formed integer pattern must not gain a [04-PAT-1] rejection \
         from an unrelated upstream failure, got {diagnostics:?}"
    );
}

/// DISPOSITION LOCK. `CheckErrorKind::TypeMismatch` is the kind [04-PAT-1]
/// names, and the rejection carries a repair suggestion rather than a bare
/// message. Pinned so the diagnostic contract cannot quietly degrade.
#[test]
fn the_rejection_carries_its_kind_and_a_suggestion() {
    let Err(result) = check_ir_program(
        &parse_core(include_str!(
            "fixtures/core/tests/issue_1494_literal_pattern_scrutinee/the_rejection_carries_its_kind_and_a_suggestion.0.deep"
        )),
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) else {
        panic!("the issue program must be rejected");
    };
    let [error] = result.errors.as_slice() else {
        panic!("expected exactly one error, got {:?}", result.errors);
    };
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
    assert!(
        !error.suggestions.is_empty(),
        "the rejection must carry a repair suggestion, got {error:?}"
    );
}
