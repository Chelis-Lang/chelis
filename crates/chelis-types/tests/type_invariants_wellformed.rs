//! Declaration-time well-formedness for opaque-type invariants
//! (RFC `opaque_invariants_rfc.md` D-WF). This is the W2 acceptance
//! oracle for the checker pass `validate_type_invariants_in_program`.
//!
//! Spec-first: every rejection here pins a D-WF requirement. Positive
//! and negative parity throughout (AGENTS.md). Covers BOTH the `.ch`
//! (parse + desugar) path and the raw `.dp` path, since the checker pass
//! runs in both.
//!
//! The single CRITICAL positive (D-WF + feature statement): an invariant
//! VIOLATED by an in-module constructor still type-checks. The invariant
//! is recorded, never evaluated; this is not refinement typing.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckError;
use chelis_types::infer_ir_program;

// ── Harness ──────────────────────────────────────────────────────

fn deep_of_surf(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse should succeed");
    desugar_program(&decls)
}

fn deep_of_dp(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str_strict(source).expect("deep parse should succeed")
}

fn errors(exprs: &[chelis_deep::Expr]) -> Vec<CheckError> {
    infer_ir_program(exprs).errors
}

fn kind_name(err: &CheckError) -> String {
    format!("{:?}", err.kind)
}

/// The invariant-well-formedness violations (an `OpaqueTypeViolation`
/// whose message mentions an invariant).
fn invariant_violations(errors: &[CheckError]) -> Vec<&CheckError> {
    errors
        .iter()
        .filter(|e| kind_name(e) == "OpaqueTypeViolation" && e.message.contains("invariant"))
        .collect()
}

fn assert_has_violation(errors: &[CheckError], needle: &str) {
    let viols = invariant_violations(errors);
    assert!(
        viols.iter().any(|e| e.message.contains(needle)),
        "expected an invariant violation containing `{needle}`, got: {:?}",
        errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

fn assert_no_invariant_violation(errors: &[CheckError]) {
    let viols = invariant_violations(errors);
    assert!(
        viols.is_empty(),
        "expected no invariant violation, got: {:?}",
        viols.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// Stronger than `assert_no_invariant_violation`: NO check error of any
/// kind. Catches spurious deep-validate structural errors leaking from
/// the embedded predicate fn metadata into the check path.
fn assert_check_clean(errors: &[CheckError]) {
    assert!(
        errors.is_empty(),
        "expected a clean check, got: {:?}",
        errors
            .iter()
            .map(|e| format!("{:?}: {}", e.kind, e.message))
            .collect::<Vec<_>>()
    );
}

// ===========================================================================
// Positive: well-formed invariants pass (one per amenability class)
// ===========================================================================

#[test]
fn linear_invariant_is_well_formed() {
    let deep = deep_of_surf(
        "module Stats.Prob\n@opaque\n\
         @invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
         type Probability = | Probability { value: f32 }",
    );
    // The strongest form: a well-formed invariant produces a fully clean
    // check (no spurious deep-validate structural error from the embedded
    // predicate fn metadata).
    assert_check_clean(&errors(&deep));
}

#[test]
fn polynomial_invariant_is_well_formed() {
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) (p.value * p.value) <= 1.0\n\
         type Probability = | Probability { value: f32 }",
    );
    assert_no_invariant_violation(&errors(&deep));
}

#[test]
fn transcendental_invariant_is_well_formed() {
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) exp(p.value) <= 3.0\n\
         type Probability = | Probability { value: f32 }",
    );
    assert_no_invariant_violation(&errors(&deep));
}

#[test]
fn simplex_tolerance_band_is_well_formed() {
    // W6 flagship declaration: sum over a fixed-shape tensor field with a
    // module-constant tolerance band MUST be legal now.
    let deep = deep_of_surf(
        "module Stats.Simplex\neps = 0.001\n@opaque\n\
         @invariant(p) (sum(p.weights) >= (1.0 - eps)) && (sum(p.weights) <= (1.0 + eps))\n\
         type Simplex = | Simplex { weights: tensor[3, f32] }",
    );
    assert_no_invariant_violation(&errors(&deep));
}

// ===========================================================================
// Check 1: invariant requires opaque (raw .dp; Surf rejects at parse)
// ===========================================================================

#[test]
fn invariant_without_opaque_is_rejected() {
    let error = chelis_deep::parser::parse_str(
        "(deftype {invariant: (fn {} (params {} p) \
            (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0))), \
            invariant_amenability: \"linear\"} \
            Probability () (variant {} Probability (field {} value (t-prim {} f32))))",
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("invariant") && error.contains("opaque"),
        "{error}"
    );
}

#[test]
fn opaque_with_invariant_does_not_trip_requires_opaque() {
    // Negative parity: the opaque-present case never raises check 1.
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) p.value >= 0.0\n\
         type Probability = | Probability { value: f32 }",
    );
    let errs = errors(&deep);
    let viols = invariant_violations(&errs);
    assert!(
        !viols
            .iter()
            .any(|e| e.message.contains("requires `@opaque`")),
        "opaque-present must not trip the requires-opaque check"
    );
}

// ===========================================================================
// Check 2: representation value class
// ===========================================================================

#[test]
fn list_field_is_rejected() {
    // A `List[f32]` field is outside the V1 value class.
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) p.count >= 0.0\n\
         type Bag = | Bag { items: List[f32], count: f32 }",
    );
    assert_has_violation(&errors(&deep), "value class");
}

#[test]
fn function_field_is_rejected() {
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) p.value >= 0.0\n\
         type Fancy = | Fancy { value: f32, f: f32 -> f32 }",
    );
    assert_has_violation(&errors(&deep), "value class");
}

#[test]
fn non_literal_tensor_dim_is_rejected() {
    // A symbolic tensor dimension (`n`) is not fixed-shape.
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) sum(p.weights) >= 0.0\n\
         type Vec = | Vec { weights: tensor[n, f32] }",
    );
    assert_has_violation(&errors(&deep), "value class");
}

#[test]
fn fixed_shape_tensor_field_is_accepted() {
    // Negative parity for the dim check: a fully literal shape passes.
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) sum(p.weights) >= 0.0\n\
         type Vec = | Vec { weights: tensor[4, f32] }",
    );
    assert_no_invariant_violation(&errors(&deep));
}

#[test]
fn nested_single_variant_record_field_is_accepted() {
    // A field typed as an in-program single-variant record of value-class
    // types is in the value class (RFC nested-record allowance).
    let deep = deep_of_surf(
        "module M\ntype Inner = | Inner { a: f32 }\n@opaque\n\
         @invariant(p) p.inner.a >= 0.0\n\
         type Outer = | Outer { inner: Inner }",
    );
    assert_no_invariant_violation(&errors(&deep));
}

#[test]
fn multi_variant_adt_is_rejected() {
    // Raw .dp: an invariant-carrying type with two variants.
    let deep = deep_of_dp(
        "(module {} m \
            (deftype {opaque: true, \
                invariant: (fn {} (params {} p) \
                    (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0))), \
                invariant_amenability: \"linear\"} \
                T () \
                (variant {} A (field {} value (t-prim {} f32))) \
                (variant {} B (field {} value (t-prim {} f32)))))",
    );
    assert_has_violation(&errors(&deep), "exactly one");
}

// ===========================================================================
// Check 3: predicate grammar + free vars + boolean shape
// ===========================================================================

#[test]
fn match_in_predicate_is_rejected() {
    // A `match` is out of grammar. Built as raw .dp (Surf would also
    // reject, but the .dp path is the forge surface).
    let deep = deep_of_dp(
        "(module {} m \
            (deftype {opaque: true, \
                invariant: (fn {} (params {} p) \
                    (match {} (access {} (var {} p) value) \
                        (arm {} (pat-wild {}) () (lit {type: (t-prim {} bool)} true)))), \
                invariant_amenability: \"opaque\"} \
                T () (variant {} T (field {} value (t-prim {} f32)))))",
    );
    assert_has_violation(&errors(&deep), "outside the predicate grammar");
}

#[test]
fn general_call_in_predicate_is_rejected() {
    let deep = deep_of_surf(
        "module M\ndef helper(x: f32) -> f32 = x\n@opaque\n\
         @invariant(p) helper(p.value) >= 0.0\n\
         type Probability = | Probability { value: f32 }",
    );
    assert_has_violation(&errors(&deep), "outside the predicate grammar");
}

#[test]
fn out_of_scope_free_var_is_rejected() {
    // `eps` is referenced but not declared as an in-module constant.
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) p.value >= eps\n\
         type Probability = | Probability { value: f32 }",
    );
    assert_has_violation(&errors(&deep), "not the binder or an in-module");
}

#[test]
fn in_module_constant_free_var_is_accepted() {
    // Negative parity: a declared in-module constant is a legal free var.
    let deep = deep_of_surf(
        "module M\neps = 0.01\n@opaque\n@invariant(p) p.value >= eps\n\
         type Probability = | Probability { value: f32 }",
    );
    assert_no_invariant_violation(&errors(&deep));
}

#[test]
fn non_boolean_predicate_is_rejected() {
    // A predicate whose top is arithmetic, not boolean. Raw .dp so the
    // amenability is set consistently.
    let deep = deep_of_dp(
        "(module {} m \
            (deftype {opaque: true, \
                invariant: (fn {} (params {} p) \
                    (app {} (var {} add) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 1.0))), \
                invariant_amenability: \"linear\"} \
                T () (variant {} T (field {} value (t-prim {} f32)))))",
    );
    assert_has_violation(&errors(&deep), "boolean predicate at the top");
}

// ===========================================================================
// Check 4: recorded amenability matches recomputation (protects .dp)
// ===========================================================================

#[test]
fn dp_amenability_mismatch_is_rejected() {
    // Hand-written .dp records "linear" for a polynomial predicate.
    let deep = deep_of_dp(
        "(module {} m \
            (deftype {opaque: true, \
                invariant: (fn {} (params {} p) \
                    (app {} (var {} lte) \
                        (app {} (var {} mul) (access {} (var {} p) value) (access {} (var {} p) value)) \
                        (lit {type: (t-prim {} f32)} 1.0))), \
                invariant_amenability: \"linear\"} \
                T () (variant {} T (field {} value (t-prim {} f32)))))",
    );
    assert_has_violation(&errors(&deep), "records amenability `linear`");
}

#[test]
fn dp_correct_amenability_is_accepted() {
    // Negative parity: a .dp recording the correct class passes.
    let deep = deep_of_dp(
        "(module {} m \
            (deftype {opaque: true, \
                invariant: (fn {} (params {} p) \
                    (app {} (var {} lte) \
                        (app {} (var {} mul) (access {} (var {} p) value) (access {} (var {} p) value)) \
                        (lit {type: (t-prim {} f32)} 1.0))), \
                invariant_amenability: \"polynomial\"} \
                T () (variant {} T (field {} value (t-prim {} f32)))))",
    );
    assert_no_invariant_violation(&errors(&deep));
}

#[test]
fn dp_missing_amenability_is_rejected() {
    let deep = deep_of_dp(
        "(module {} m \
            (deftype {opaque: true, \
                invariant: (fn {} (params {} p) \
                    (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0)))} \
                T () (variant {} T (field {} value (t-prim {} f32)))))",
    );
    assert_has_violation(&errors(&deep), "missing the `invariant_amenability`");
}

// ===========================================================================
// CRITICAL: invariant is invisible to type checking (D-WF + feature
// statement). A program whose invariant is VIOLATED by an in-module
// constructor still type-checks (no invariant violation, the program is
// otherwise well-typed).
// ===========================================================================

#[test]
fn violated_invariant_still_type_checks() {
    // The constructor produces `value: 5.0`, violating `value <= 1.0`.
    // The checker records the invariant; it never evaluates it. The
    // program must have NO invariant well-formedness violation: the
    // declaration itself is well-formed; the construction is in-module
    // (legal); the invariant is not refinement typing.
    let deep = deep_of_surf(
        "module Stats.Prob\n@opaque\n\
         @invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
         type Probability = | Probability { value: f32 }\n\
         def bad() -> Probability = Probability { value: 5.0 }",
    );
    assert_no_invariant_violation(&errors(&deep));
}

#[test]
fn non_opaque_type_with_no_invariant_is_untouched() {
    // Negative parity: an ordinary ADT with no invariant raises nothing.
    let deep = deep_of_surf("module M\ntype Point = | Point { x: f32, y: f32 }");
    assert_no_invariant_violation(&errors(&deep));
}

// ── Review follow-up: the D-WF value class must match the prover's model, so
//    a representation `chelis check` admits is one `chelis prove` can verify
//    (covered-or-rejected). A wider check class silently dropped obligations.

#[test]
fn string_scalar_field_is_outside_the_value_class() {
    // `string` is a prim but no arithmetic invariant can be verified over it;
    // the prover does not model it, so the checker must reject it rather than
    // admit a representation that yields zero obligations.
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) p.value >= 0.0\n\
         type Tagged = | Tagged { tag: string, value: f32 }",
    );
    assert_has_violation(&errors(&deep), "value class");
}

#[test]
fn integer_element_tensor_field_is_outside_the_value_class() {
    // The prover only models f32/f64 tensors; an int-element tensor field is
    // not in the value class (it would otherwise be silently dropped).
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) p.ok >= 0.0\n\
         type T = | T { ok: f32, bad: tensor[4, int32] }",
    );
    assert_has_violation(&errors(&deep), "value class");
}

#[test]
fn nested_record_of_value_class_fields_is_admitted() {
    // Positive parity: a nested single-variant record of numeric fields IS in
    // the value class (the prover models it as a flattened record), so it must
    // NOT be rejected.
    let deep = deep_of_surf(
        "module M\ntype Inner = | Inner { value: f32 }\n\
         @opaque\n@invariant(p) p.inner.value >= 0.0\n\
         type Boxed = | Boxed { inner: Inner }",
    );
    assert_no_invariant_violation(&errors(&deep));
}

#[test]
fn if_predicate_with_non_boolean_condition_is_rejected() {
    // The `if` CONDITION must be boolean, not just the branches. Checking only
    // the branches let an f32 condition `if p.value then true else false` pass
    // D-WF even though the type rule requires a `bool` condition.
    let deep = deep_of_surf(
        "module M\n@opaque\n@invariant(p) if p.value then true else false\n\
         type T = | T { value: f32 }",
    );
    assert_has_violation(&errors(&deep), "boolean predicate");
}

#[test]
fn if_predicate_with_comparison_condition_is_admitted() {
    // Positive parity: a boolean-shaped (comparison) condition is fine.
    let deep = deep_of_surf(
        "module M\n@opaque\n\
         @invariant(p) if p.value >= 0.0 then p.value <= 1.0 else p.value >= -1.0\n\
         type T = | T { value: f32 }",
    );
    assert_no_invariant_violation(&errors(&deep));
}
