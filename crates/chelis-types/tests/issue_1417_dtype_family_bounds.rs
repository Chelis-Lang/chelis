//! chelis#1417: authored dtype-family bounds are enforced by the checker.
//!
//! `spec/04-type-system.md` §5.9 [04-DTYPE-2]. Each family is exercised at
//! every dtype §1.1 admits into it and at every dtype it excludes, and the
//! bound is checked through aliases, wrappers, higher-order values, and
//! recursion rather than at a named callee.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

const FLOATS: &[&str] = &["f16", "bf16", "f32", "f64"];
const INTS: &[&str] = &["int8", "int16", "int32", "int64"];
const NON_NUMERIC: &[&str] = &["bool"];

fn diagnostics(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    match check_typed_program(&desugar_program(&decls)) {
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
        r#"
sig arange_like[p: Int]: p -> p -> p
def arange_like(start, stop) = start
def probe(a: {dtype}, b: {dtype}) -> {dtype} = arange_like(a, b)
"#
    )
}

fn linspace_shaped(dtype: &str) -> String {
    format!(
        r#"
sig linspace_like[p: Float]: p -> p -> int64 -> p
def linspace_like(start, stop, count) = start
def probe(a: {dtype}, b: {dtype}) -> {dtype} = linspace_like(a, b, cast(3, int64))
"#
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
                r#"
sig total[p: Numeric]: p -> p -> p
def total(a, b) = a
def probe(x: {dtype}, y: {dtype}) -> {dtype} = total(x, y)
"#
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
                r#"
sig total[p: Numeric]: p -> p -> p
def total(a, b) = a
def probe(x: {dtype}, y: {dtype}) -> {dtype} = total(x, y)
"#
            ),
            &format!("a `Numeric`-bounded signature at {dtype}"),
        );
    }
}

// === The bound rides on the scheme, not on a callee name ===

#[test]
fn a_bound_survives_a_top_level_alias() {
    assert_family_rejection(
        r#"
sig only_ints[p: Int]: p -> p
def only_ints(x) = x
alias = only_ints
def probe(x: f32) -> f32 = alias(x)
"#,
        "an alias of an `Int`-bounded function",
        "Int",
        "f32",
    );
}

#[test]
fn a_bound_survives_a_generic_wrapper() {
    assert_family_rejection(
        r#"
sig only_ints[p: Int]: p -> p
def only_ints(x) = x
def wrap[q: Int](x: q) -> q = only_ints(x)
def probe(x: f64) -> f64 = wrap(x)
"#,
        "a generic wrapper over an `Int`-bounded function",
        "Int",
        "f64",
    );
}

#[test]
fn a_bound_survives_a_higher_order_call() {
    assert_family_rejection(
        r#"
sig only_floats[p: Float]: p -> p
def only_floats(x) = x
def apply_to[q](f: q -> q, x: q) -> q = f(x)
def probe(x: int32) -> int32 = apply_to(only_floats, x)
"#,
        "an `Float`-bounded function passed as a value",
        "Float",
        "int32",
    );
}

#[test]
fn a_bound_survives_a_recursive_helper() {
    assert_family_rejection(
        r#"
sig countdown[p: Int]: p -> p -> p
def countdown(current, stop) = if gte(current, stop) then current else countdown(current, stop)
def probe(x: bf16) -> bf16 = countdown(x, x)
"#,
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
        r#"
sig only_ints[p: Int]: p -> p
def only_ints(x) = x
def wrap[q: Int](x: q) -> q = only_ints(x)
def probe(x: int16) -> int16 = wrap(x)
"#,
        "a generic wrapper at an admitted dtype",
    );
}

// === Family intersection ===

#[test]
fn a_numeric_binder_narrows_to_float_through_a_float_bounded_callee() {
    assert_family_rejection(
        r#"
sig only_floats[p: Float]: p -> p
def only_floats(x) = x
sig any_numeric[q: Numeric]: q -> q
def any_numeric(x) = only_floats(x)
def probe(x: int32) -> int32 = any_numeric(x)
"#,
        "a `Numeric` binder identified with a `Float` one",
        "Float",
        "int32",
    );
}

#[test]
fn a_narrowed_numeric_binder_still_accepts_its_intersection() {
    // Intersection is valid for inference variables, not a license to narrow
    // an authored Numeric contract. The declaration fails even with an f32 call.
    let source = r#"
sig only_floats[p: Float]: p -> p
def only_floats(x) = x
sig any_numeric[q: Numeric]: q -> q
def any_numeric(x) = only_floats(x)
def probe(x: f32) -> f32 = any_numeric(x)
"#;
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
        &source.replace("q: Numeric", "q: Float"),
        "the explicitly sufficient contract admits f32",
    );
}

#[test]
fn float_and_int_bounds_cannot_name_the_same_variable() {
    let errors = diagnostics(
        r#"
sig only_floats[p: Float]: p -> p
def only_floats(x) = x
sig only_ints[q: Int]: q -> q
def only_ints(x) = x
def probe[r](x: r) -> r = only_ints(only_floats(x))
"#,
    );
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
        r#"
sig scale[p: Float]: tensor[n, p] -> tensor[n, p]
def scale(x) = x
def probe(x: tensor[4, int32]) -> tensor[4, int32] = scale(x)
"#,
        "a `Float` bound occupying a tensor precision slot",
        "Float",
        "int32",
    );
}

#[test]
fn a_precision_slot_bound_accepts_its_own_family() {
    assert_accepted(
        r#"
sig scale[p: Float]: tensor[n, p] -> tensor[n, p]
def scale(x) = x
def probe(x: tensor[4, f32]) -> tensor[4, f32] = scale(x)
"#,
        "a `Float` precision-slot bound at f32",
    );
}

// === Declaration well-formedness ===

#[test]
fn a_bounded_binder_cannot_name_a_dimension() {
    let errors = diagnostics("sig f[n: Float]: tensor[n, f32] -> tensor[n, f32]");
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
    let errors = diagnostics("sig f[r: Float]: tensor[..r, f32] -> tensor[..r, f32]");
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
    let errors = diagnostics("sig f[q: Float]: int32 -> int32");
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
        "sig f[p: Float]: p -> p\ndef f(x) = x",
        "a bound naming a binder the signature uses",
    );
}

#[test]
fn a_def_may_not_bound_a_binder_its_sig_owns() {
    let error = parse_surf(
        r#"
sig f[p: Float]: p -> p
def f[p: Float](x: p) -> p = x
"#,
    )
    .expect_err("duplicate bound ownership must reject before desugaring");
    assert!(
        error
            .to_string()
            .contains("a declaration's `defsig` owns its binders"),
        "a bound belongs to one binder list; got: {error}"
    );
}

#[test]
fn a_def_without_a_sig_carries_its_own_bound() {
    // The negative control for the previous test: the restriction is about
    // two binder lists, not about bounds on a `def`.
    assert_family_rejection(
        r#"
def only_ints[p: Int](x: p) -> p = x
def probe(x: f32) -> f32 = only_ints(x)
"#,
        "a `def`-declared `Int` bound",
        "Int",
        "f32",
    );
}

#[test]
fn a_def_declared_bound_accepts_its_own_family() {
    assert_accepted(
        r#"
def only_ints[p: Int](x: p) -> p = x
def probe(x: int64) -> int64 = only_ints(x)
"#,
        "a `def`-declared `Int` bound at int64",
    );
}

#[test]
fn an_unbounded_binder_still_admits_a_non_dtype_type() {
    // [04-DTYPE-2] leaves an unbounded binder a general type variable; this
    // is what a narrowing implementation would silently break.
    assert_accepted(
        r#"
def pick[a](x: a, y: a) -> a = x
def probe(x: List[int32], y: List[int32]) -> List[int32] = pick(x, y)
"#,
        "an unbounded binder at a non-dtype type",
    );
}
