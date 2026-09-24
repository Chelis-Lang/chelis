//! chelis#1417: authored dtype-family bounds are enforced by the checker.
//!
//! `spec/04-type-system.md` §5.9 [04-DTYPE-2]. Each family is exercised at
//! every dtype §1.1 admits into it and at every dtype it excludes, and the
//! bound is checked through aliases, wrappers, higher-order values, and
//! recursion rather than at a named callee.

use chelis_deep::parser::parse_str as parse_core;
use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

const FLOATS: &[&str] = &["f16", "bf16", "f32", "f64"];
const INTS: &[&str] = &["i8", "i16", "i32", "i64"];
const NON_NUMERIC: &[&str] = &["bool"];

fn diagnostics(source: &str) -> Vec<CheckError> {
    let decls = if source.trim_start().starts_with('[') {
        serde_json::from_str(source).expect("lossless AST fixture")
    } else {
        parse_core(source).expect("Deep fixture must parse")
    };
    match check_typed_program(&decls, chelis_types::ProgramOrigin::Unlinked) {
        Ok(_) => Vec::new(),
        Err(report) => report.errors,
    }
}

fn rendered(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

fn assert_accepted(source: &str, context: &str) {
    let errors = diagnostics(source);
    assert!(
        errors.is_empty(),
        "{context} must type-check; got:\n{}",
        rendered(&errors)
    );
}

fn assert_family_rejection(source: &str, context: &str, family: &str, dtype: &str) {
    let errors = diagnostics(source);
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                && error.message.contains(family)
                && error.message.contains(dtype)
        }),
        "{context} must be rejected with a `{family}` diagnostic naming `{dtype}`; got:\n{}",
        rendered(&errors)
    );
}

// === The issue's two reproducers ===

fn arange_shaped(dtype: &str) -> String {
    format!(
        r###"(defsig {{dtype_bounds: {{p: int}}}}
  arange_like
  (p)
  (t-fn {{}} (t-var {{}} p) (t-var {{}} p) (t-var {{}} p)))

(def {{}}
  arange_like
  (fn {{}}
    (params {{}} start stop)
    (var {{span: "surf:69..74"}} start)))

(defsig {{}} probe (t-fn {{}} (t-prim {{}} {dtype}) (t-prim {{}} {dtype}) (t-prim {{}} {dtype})))

(def {{}}
  probe
  (fn {{}}
    (params {{}} (a {{type: (t-var {{}} _)}}) (b {{type: (t-var {{}} _)}}))
    (app {{span: "surf:110..127"}}
      (var {{span: "surf:110..121"}} arange_like)
      (var {{span: "surf:122..123"}} a)
      (var {{span: "surf:125..126"}} b))))
"###
    )
}

fn linspace_shaped(dtype: &str) -> String {
    format!(
        r###"(defsig {{dtype_bounds: {{p: float}}}}
  linspace_like
  (p)
  (t-fn {{}} (t-var {{}} p) (t-var {{}} p) (t-prim {{}} i64) (t-var {{}} p)))

(def {{}}
  linspace_like
  (fn {{}}
    (params {{}} start stop count)
    (var {{span: "surf:89..94"}} start)))

(defsig {{}} probe (t-fn {{}} (t-prim {{}} {dtype}) (t-prim {{}} {dtype}) (t-prim {{}} {dtype})))

(def {{}}
  probe
  (fn {{}}
    (params {{}} (a {{type: (t-var {{}} _)}}) (b {{type: (t-var {{}} _)}}))
    (app {{span: "surf:130..163"}}
      (var {{span: "surf:130..143"}} linspace_like)
      (var {{span: "surf:144..145"}} a)
      (var {{span: "surf:147..148"}} b)
      (cast {{span: "surf:150..162"}}
        (lit {{span: "surf:155..156", surf_literal_style: "unsuffixed", type: (t-prim {{}} i64)}} 3)
        (t-prim {{}} i64)))))
"###
    )
}

#[test]
fn an_int_bounded_signature_rejects_every_float() {
    for dtype in FLOATS {
        assert_family_rejection(
            &arange_shaped(dtype),
            "an `Int`-bounded arange-shaped signature at a float",
            "Int",
            dtype,
        );
    }
}

#[test]
fn an_int_bounded_signature_accepts_every_active_signed_integer() {
    for dtype in INTS {
        assert_accepted(
            &arange_shaped(dtype),
            &format!("an `Int`-bounded signature at {dtype}"),
        );
    }
}

#[test]
fn a_float_bounded_signature_rejects_every_integer() {
    for dtype in INTS {
        assert_family_rejection(
            &linspace_shaped(dtype),
            "a `Float`-bounded linspace-shaped signature at an integer",
            "Float",
            dtype,
        );
    }
}

#[test]
fn a_float_bounded_signature_accepts_every_active_float() {
    for dtype in FLOATS {
        assert_accepted(
            &linspace_shaped(dtype),
            &format!("a `Float`-bounded signature at {dtype}"),
        );
    }
}

#[test]
fn no_family_admits_bool() {
    for dtype in NON_NUMERIC {
        assert_family_rejection(
            &arange_shaped(dtype),
            "an `Int` bound at bool",
            "Int",
            dtype,
        );
        assert_family_rejection(
            &linspace_shaped(dtype),
            "a `Float` bound at bool",
            "Float",
            dtype,
        );
        assert_family_rejection(
            &format!(
                r###"(defsig {{dtype_bounds: {{p: numeric}}}}
  total
  (p)
  (t-fn {{}} (t-var {{}} p) (t-var {{}} p) (t-var {{}} p)))

(def {{}} total (fn {{}} (params {{}} a b) (var {{span: "surf:54..55"}} a)))

(defsig {{}} probe (t-fn {{}} (t-prim {{}} {dtype}) (t-prim {{}} {dtype}) (t-prim {{}} {dtype})))

(def {{}}
  probe
  (fn {{}}
    (params {{}} (x {{type: (t-var {{}} _)}}) (y {{type: (t-var {{}} _)}}))
    (app {{span: "surf:91..102"}}
      (var {{span: "surf:91..96"}} total)
      (var {{span: "surf:97..98"}} x)
      (var {{span: "surf:100..101"}} y))))
"###
            ),
            "a `Numeric` bound at bool",
            "Numeric",
            dtype,
        );
    }
}

#[test]
fn a_numeric_bound_admits_both_numeric_families() {
    for dtype in FLOATS.iter().chain(INTS) {
        assert_accepted(
            &format!(
                r###"(defsig {{dtype_bounds: {{p: numeric}}}}
  total
  (p)
  (t-fn {{}} (t-var {{}} p) (t-var {{}} p) (t-var {{}} p)))

(def {{}} total (fn {{}} (params {{}} a b) (var {{span: "surf:54..55"}} a)))

(defsig {{}} probe (t-fn {{}} (t-prim {{}} {dtype}) (t-prim {{}} {dtype}) (t-prim {{}} {dtype})))

(def {{}}
  probe
  (fn {{}}
    (params {{}} (x {{type: (t-var {{}} _)}}) (y {{type: (t-var {{}} _)}}))
    (app {{span: "surf:91..102"}}
      (var {{span: "surf:91..96"}} total)
      (var {{span: "surf:97..98"}} x)
      (var {{span: "surf:100..101"}} y))))
"###
            ),
            &format!("a `Numeric`-bounded signature at {dtype}"),
        );
    }
}

// === The bound rides on the scheme, not on a callee name ===

#[test]
fn a_bound_survives_a_top_level_alias() {
    assert_family_rejection(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_bound_survives_a_top_level_alias_172.deep"
        ),
        "an alias of an `Int`-bounded function",
        "Int",
        "f32",
    );
}

#[test]
fn a_bound_survives_a_generic_wrapper() {
    assert_family_rejection(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_bound_survives_a_generic_wrapper_187.deep"
        ),
        "a generic wrapper over an `Int`-bounded function",
        "Int",
        "f64",
    );
}

#[test]
fn a_bound_survives_a_higher_order_call() {
    assert_family_rejection(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_bound_survives_a_higher_order_call_202.deep"
        ),
        "an `Float`-bounded function passed as a value",
        "Float",
        "i32",
    );
}

#[test]
fn a_bound_survives_a_recursive_helper() {
    assert_family_rejection(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_bound_survives_a_recursive_helper_217.deep"
        ),
        "a recursive `Int`-bounded helper",
        "Int",
        "bf16",
    );
}

#[test]
fn a_wrapper_of_a_bounded_function_still_accepts_its_own_family() {
    // The negative tests above must fail for the bound, not because the
    // wrapper shapes are themselves broken.
    assert_accepted(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_wrapper_of_a_bounded_function_still_accepts_its_own_family_233.deep"
        ),
        "a generic wrapper at an admitted dtype",
    );
}

// === Family intersection ===

#[test]
fn a_numeric_binder_narrows_to_float_through_a_float_bounded_callee() {
    assert_family_rejection(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_numeric_binder_narrows_to_float_through_a_float_bounded_callee_248.deep"
        ),
        "a `Numeric` binder identified with a `Float` one",
        "Float",
        "i32",
    );
}

#[test]
fn a_narrowed_numeric_binder_still_accepts_its_intersection() {
    // Intersection is valid for inference variables, not a license to narrow
    // an authored Numeric contract. The declaration fails even with an f32 call.
    let source = include_str!(
        "fixtures/core/tests/issue_1417_dtype_family_bounds/a_narrowed_numeric_binder_still_accepts_its_intersection_265.json"
    );
    let errors = diagnostics(source);
    assert!(
        errors.iter().any(
            |error| matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                && error.message.contains("any_numeric")
                && error.message.contains("Numeric")
                && error.message.contains("Float")
        ),
        "{}",
        rendered(&errors)
    );
    assert_accepted(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_narrowed_numeric_binder_still_accepts_its_intersection_float.json"
        ),
        "the explicitly sufficient contract admits f32",
    );
}

#[test]
fn float_and_int_bounds_cannot_name_the_same_variable() {
    let errors = diagnostics(include_str!(
        "fixtures/core/tests/issue_1417_dtype_family_bounds/float_and_int_bounds_cannot_name_the_same_variable_292.deep"
    ));
    assert!(
        errors.iter().any(
            |error| matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                && error.message.contains("Float")
                && error.message.contains("Int")
        ),
        "an empty family intersection must name both families; got:\n{}",
        rendered(&errors)
    );
}

// === A bound in a tensor precision slot ===

#[test]
fn a_bound_reaches_a_tensor_precision_slot() {
    assert_family_rejection(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_bound_reaches_a_tensor_precision_slot_316.deep"
        ),
        "a `Float` bound occupying a tensor precision slot",
        "Float",
        "i32",
    );
}

#[test]
fn a_precision_slot_bound_accepts_its_own_family() {
    assert_accepted(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_precision_slot_bound_accepts_its_own_family_330.deep"
        ),
        "a `Float` precision-slot bound at f32",
    );
}

// === Declaration well-formedness ===

#[test]
fn a_bounded_binder_cannot_name_a_dimension() {
    let errors = diagnostics(include_str!(
        "fixtures/core/tests/issue_1417_dtype_family_bounds/a_bounded_binder_cannot_name_a_dimension_343.deep"
    ));
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("cannot be used as a dimension slot")),
        "a dtype family cannot name an extent; got:\n{}",
        rendered(&errors)
    );
}

#[test]
fn a_bounded_binder_cannot_name_a_rank_spread() {
    let errors = diagnostics(include_str!(
        "fixtures/core/tests/issue_1417_dtype_family_bounds/a_bounded_binder_cannot_name_a_rank_spread_355.deep"
    ));
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("cannot be used as a rank spread")),
        "a dtype family cannot name a run of extents; got:\n{}",
        rendered(&errors)
    );
}

#[test]
fn a_bound_must_name_a_binder_the_signature_uses() {
    let errors = diagnostics(include_str!(
        "fixtures/core/tests/issue_1417_dtype_family_bounds/a_bound_must_name_a_binder_the_signature_uses_367.deep"
    ));
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("does not occur")),
        "a bound naming an absent binder is a typo, not a no-op; got:\n{}",
        rendered(&errors)
    );
}

#[test]
fn a_used_bound_is_accepted_where_the_absent_one_is_not() {
    assert_accepted(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_used_bound_is_accepted_where_the_absent_one_is_not_380.deep"
        ),
        "a bound naming a binder the signature uses",
    );
}

#[test]
fn a_def_without_a_sig_carries_its_own_bound() {
    // The negative control for the previous test: the restriction is about
    // two binder lists, not about bounds on a `def`.
    assert_family_rejection(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_def_without_a_sig_carries_its_own_bound_407.deep"
        ),
        "a `def`-declared `Int` bound",
        "Int",
        "f32",
    );
}

#[test]
fn a_def_declared_bound_accepts_its_own_family() {
    assert_accepted(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/a_def_declared_bound_accepts_its_own_family_420.deep"
        ),
        "a `def`-declared `Int` bound at i64",
    );
}

#[test]
fn an_unbounded_binder_still_admits_a_non_dtype_type() {
    // [04-DTYPE-2] leaves an unbounded binder a general type variable; this
    // is what a narrowing implementation would silently break.
    assert_accepted(
        include_str!(
            "fixtures/core/tests/issue_1417_dtype_family_bounds/an_unbounded_binder_still_admits_a_non_dtype_type_433.deep"
        ),
        "an unbounded binder at a non-dtype type",
    );
}
