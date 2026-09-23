//! W1 checker-enforced opacity: the spec-first acceptance suite for
//! RFC D-CHECK (spec/design/opaque_invariants_rfc.md).
//!
//! Written RED-FIRST per AGENTS.md spec-first development: every
//! rejection test in this file was committed before the enforcement
//! existed and verified to fail (the checker passed the violation
//! clean). Baseline-behavior tests (inside-module passes, the
//! DuplicateDefinition lock, the wildcard-arm pass, and the nested
//! pat-var negative-parity pin) pass before and after.
//!
//! Rejections preserve opacity ownership and do not cascade into unrelated errors.

use chelis_types::errors::CheckError;
use chelis_types::{infer_ir_program, infer_program};

// ── Harness ──────────────────────────────────────────────────────

/// Compose independently generated Deep module fixtures.
fn core_modules(sources: &[&str]) -> Vec<chelis_deep::Expr> {
    sources
        .iter()
        .flat_map(|source| chelis_deep::parser::parse_str(source).expect("Deep module fixture"))
        .collect()
}

fn deep_of_dp(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str_strict(source).expect("deep parse should succeed")
}

/// Errors from the executable check pipeline (the path `chelis check`
/// uses for `.dp` ingestion and the fitness scorer).
fn errors_ir(exprs: &[chelis_deep::Expr]) -> Vec<CheckError> {
    infer_ir_program(exprs).errors
}

/// Errors from the plain program driver. D-CHECK installs the
/// `OpacityContext` in BOTH drivers; the parity tests below exercise
/// this one explicitly.
fn errors_plain(exprs: &[chelis_deep::Expr]) -> Vec<CheckError> {
    infer_program(exprs).errors
}

fn kind_name(err: &CheckError) -> String {
    format!("{:?}", err.kind)
}

fn opaque_violations(errors: &[CheckError]) -> Vec<&CheckError> {
    errors
        .iter()
        .filter(|e| kind_name(e) == "OpaqueTypeViolation")
        .collect()
}

/// Exactly one error total, it is an OpaqueTypeViolation, and its
/// no unrelated diagnostics follow. "Exactly one error total" also locks the
/// D-CHECK no-cascade rule (the hook returns the true type).
fn assert_single_violation(errors: &[CheckError]) {
    let violations = opaque_violations(errors);
    assert_eq!(
        violations.len(),
        1,
        "expected exactly one OpaqueTypeViolation, got: {errors:?}"
    );
    assert_eq!(
        errors.len(),
        1,
        "no cascade errors expected alongside the violation: {errors:?}"
    );
}

fn assert_clean(errors: &[CheckError]) {
    assert!(
        errors.is_empty(),
        "expected a clean check, got errors: {errors:?}"
    );
}

// ── Shared fixtures (canonical `chelis fmt` form) ────────────────

/// Defining module: record-shaped opaque type, exported producer and
/// exported reader.
const PROB_MODULE: &str = include_str!("fixtures/core/tests/opaque_types/shared_104.deep");

/// Defining module for the sixth rejection (D-CHECK, RT-0 C2):
/// exported producer plus unexported bindings whose signatures
/// mention the opaque type in every polarity the RFC enumerates.
const PROB_SIXTH_MODULE: &str = include_str!("fixtures/core/tests/opaque_types/shared_116.deep");

/// Positional-variant opaque type.
const METERS_MODULE: &str = include_str!("fixtures/core/tests/opaque_types/shared_133.deep");

/// Multi-variant opaque type with a nullary constructor.
const FLAG_MODULE: &str = include_str!("fixtures/core/tests/opaque_types/shared_142.deep");

// ── Inside the defining module: everything passes clean ──────────

#[test]
fn inside_module_record_literal_and_producer_pass() {
    // Covers: record literal, producer fn returning the opaque type,
    // and field access inside the defining module.
    let exprs = core_modules(&[PROB_MODULE]);
    assert_clean(&errors_ir(&exprs));
}

#[test]
fn inside_module_positional_ctor_app_passes() {
    let exprs = core_modules(&[METERS_MODULE]);
    assert_clean(&errors_ir(&exprs));
}

#[test]
fn inside_module_nullary_ctor_passes() {
    let exprs = core_modules(&[FLAG_MODULE]);
    assert_clean(&errors_ir(&exprs));
}

#[test]
fn inside_module_pattern_matches_pass() {
    let src = include_str!(
        "fixtures/core/tests/opaque_types/inside_module_pattern_matches_pass_178.deep"
    );
    let exprs = core_modules(&[src]);
    assert_clean(&errors_ir(&exprs));

    let src_pos = include_str!(
        "fixtures/core/tests/opaque_types/inside_module_pattern_matches_pass_191.deep"
    );
    let exprs = core_modules(&[src_pos]);
    assert_clean(&errors_ir(&exprs));
}

#[test]
fn inside_module_record_update_passes() {
    let src = r#"(module {}
  stats.prob
  (export {} probability)
  (deftype {opaque: true}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32))))
  (defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability)))
  (def {}
    probability
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x)))))
  (def {}
    bump
    (fn {}
      (params {} (p {type: (t-adt {} Probability)}))
      (record-update {} (var {} p) (kv {} value (lit {type: (t-prim {} f32)} 0.9))))))
"#;
    let exprs = deep_of_dp(src);
    assert_clean(&errors_ir(&exprs));
}

// ── Outside-module rejections (D-CHECK rejection set) ────────────

#[test]
fn outside_module_record_literal_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_record_literal_rejected_234.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_record_literal_rejected_in_plain_driver_too() {
    // D-CHECK driver parity: the OpacityContext is installed in both
    // program-level inference drivers.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_record_literal_rejected_in_plain_driver_too_251.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_plain(&exprs));
}

#[test]
fn outside_module_violation_severity_is_pinned() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_violation_severity_is_pinned_266.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    let errors = errors_ir(&exprs);
    let violations = opaque_violations(&errors);
    assert_eq!(violations.len(), 1, "expected one violation: {errors:?}");
    assert_eq!(
        violations[0].severity, 0.8,
        "OpaqueTypeViolation default severity is pinned at 0.8"
    );
}

#[test]
fn outside_module_positional_ctor_app_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_positional_ctor_app_rejected_281.deep"
    );
    let exprs = core_modules(&[METERS_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_bare_ctor_reference_rejected() {
    // The constructor binding itself is hidden: `grab = Probability`
    // takes the constructor as a first-class function value.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_bare_ctor_reference_rejected_298.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_nullary_ctor_reference_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_nullary_ctor_reference_rejected_313.deep"
    );
    let exprs = core_modules(&[FLAG_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_pat_record_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_pat_record_rejected_326.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_pat_ctor_rejected() {
    let outside =
        include_str!("fixtures/core/tests/opaque_types/outside_module_pat_ctor_rejected_343.deep");
    let exprs = core_modules(&[METERS_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_tuple_nested_pat_record_rejected() {
    // Negative-parity twin of `outside_module_pat_record_rejected`:
    // wrapping the out-of-module opaque scrutinee in a tuple
    // (`(p, 0)`) desugars the destructure into a `pat-tuple` whose
    // first child is the opaque `pat-record`. Before the `pat-tuple`
    // arm in `pattern_bindings`, that child never reached
    // `check_opaque_use` (the catch-all silently dropped tuple-nested
    // sub-patterns), so the RFC D-CHECK opacity gate was bypassed for
    // tuple-nested opaque destructure. The fix recurses each
    // `pat-tuple` child, so this match is now rejected with the same
    // `record pattern match on opaque type` violation as the direct
    // form.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_tuple_nested_pat_record_rejected_371.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_field_access_on_annotated_param_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_field_access_on_annotated_param_rejected_388.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_field_access_on_call_result_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_field_access_on_call_result_rejected_403.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_field_access_on_deferred_lambda_param_rejected() {
    // The access target is an unannotated lambda parameter whose type
    // is still a Var when the access is inferred; the pipe stage
    // application (`p |> fn (q) -> q.value` means the lambda applied
    // to `p`) pins it to the opaque type later in the def. D-CHECK
    // requires a deferred-access ledger (mirroring the
    // deferred-borrow ledger) re-checked at def-level resolution.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_field_access_on_deferred_lambda_param_rejected_427.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_let_generalized_accessor_rejected_fail_closed() {
    // Let-polymorphism generalizes `f = fn (q) -> q.value` BEFORE the
    // call `f(p)` runs, so the call instantiates fresh type variables
    // and the recorded access target is never pinned -- a laundering
    // channel for opaque values through a polymorphic accessor. The
    // ledger mirrors the deferred-borrow precedent and rejects
    // never-pinned targets fail-closed (scoped to check units that
    // declare an opaque type).
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_let_generalized_accessor_rejected_fail_closed_449.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    let errors = errors_ir(&exprs);
    let violations = opaque_violations(&errors);
    assert_eq!(
        violations.len(),
        1,
        "expected exactly one fail-closed OpaqueTypeViolation, got: {errors:?}"
    );
}

const RECORD_UPDATE_DEFINING_DP: &str = r#"(module {}
  stats.prob
  (export {} probability prob_value)
  (deftype {opaque: true}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32))))
  (defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability)))
  (def {}
    probability
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x)))))
  (defsig {} prob_value (t-fn {} (t-adt {} Probability) (t-prim {} f32)))
  (def {}
    prob_value
    (fn {}
      (params {} (p {type: (t-adt {} Probability)}))
      (access {} (var {} p) value))))
"#;

#[test]
fn outside_module_record_update_typed_target_rejected() {
    let agent = r#"(module {}
  agent.strategy
  (def {}
    tweak
    (fn {}
      (params {} (p {type: (t-adt {} Probability)}))
      (record-update {} (var {} p) (kv {} value (lit {type: (t-prim {} f32)} 0.5))))))
"#;
    let exprs = deep_of_dp(&format!("{RECORD_UPDATE_DEFINING_DP}{agent}"));
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_record_update_deferred_target_rejected() {
    // The update target is an unannotated lambda parameter pinned to
    // the opaque type only by the later `prob_value` call: the
    // deferred ledger must catch it at def-level resolution.
    let agent = r#"(module {}
  agent.strategy
  (def {}
    tweak
    (fn {}
      (params {} p)
      (let {}
        (bind {} q (record-update {} (var {} p) (kv {} value (lit {type: (t-prim {} f32)} 0.5))))
        (app {} (var {} prob_value) (var {} p))))))
"#;
    let exprs = deep_of_dp(&format!("{RECORD_UPDATE_DEFINING_DP}{agent}"));
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_cast_into_t_prim_shape_rejected() {
    let agent = r#"(module {}
  agent.strategy
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (cast {} (var {} x) (t-prim {} Probability)))))
"#;
    let exprs = deep_of_dp(&format!("{RECORD_UPDATE_DEFINING_DP}{agent}"));
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_cast_into_t_adt_shape_rejected() {
    let agent = r#"(module {}
  agent.strategy
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (cast {} (var {} x) (t-adt {} Probability)))))
"#;
    let exprs = deep_of_dp(&format!("{RECORD_UPDATE_DEFINING_DP}{agent}"));
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_cast_out_rejected() {
    let agent = r#"(module {}
  agent.strategy
  (def {}
    extract
    (fn {}
      (params {} (p {type: (t-adt {} Probability)}))
      (cast {} (var {} p) (t-prim {} f32)))))
"#;
    let exprs = deep_of_dp(&format!("{RECORD_UPDATE_DEFINING_DP}{agent}"));
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_lit_forge_surf_expression_ascription_rejected() {
    // RT-0: `0.5 : Probability` desugars to
    // `(lit {type: (t-adt {} Probability)} 0.5)`; the forge gate must
    // cover the Surf ascription surface, not only `.dp` ingestion.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_lit_forge_surf_expression_ascription_rejected_602.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_lit_forge_block_binding_ascription_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_module_lit_forge_block_binding_ascription_rejected_617.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn outside_module_lit_forge_dp_metadata_rejected() {
    let agent = r#"(module {}
  agent.strategy
  (def {} forged (lit {type: (t-adt {} Probability)} 0.5)))
"#;
    let exprs = deep_of_dp(&format!("{RECORD_UPDATE_DEFINING_DP}{agent}"));
    assert_single_violation(&errors_ir(&exprs));
}

// ── Sixth rejection (D-CHECK, RT-0 C2, broadened in RFC v3) ──────

#[test]
fn sixth_rejection_result_position() {
    let outside =
        include_str!("fixtures/core/tests/opaque_types/sixth_rejection_result_position_661.deep");
    let exprs = core_modules(&[PROB_SIXTH_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn sixth_rejection_param_position() {
    let outside =
        include_str!("fixtures/core/tests/opaque_types/sixth_rejection_param_position_673.deep");
    let exprs = core_modules(&[PROB_SIXTH_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn sixth_rejection_function_typed_param_domain() {
    // RFC v3: an unexported `with_t(f: Probability -> f32) -> f32`
    // must not be callable from outside; it hands caller-supplied
    // code unobligated values.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/sixth_rejection_function_typed_param_domain_688.deep"
    );
    let exprs = core_modules(&[PROB_SIXTH_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn sixth_rejection_non_function_binding() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/sixth_rejection_non_function_binding_700.deep"
    );
    let exprs = core_modules(&[PROB_SIXTH_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn sixth_rejection_transitive_containment() {
    // "Mentions" is containment chased through named type
    // definitions: `helper -> WrapRec` mentions Probability because
    // the non-opaque WrapRec carries a Probability field.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/sixth_rejection_transitive_containment_715.deep"
    );
    let exprs = core_modules(&[PROB_SIXTH_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn exported_producer_reference_stays_legal() {
    // Negative parity for the sixth rejection: the exported producer
    // and reader remain callable from outside.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/exported_producer_reference_stays_legal_729.deep"
    );
    let exprs = core_modules(&[PROB_SIXTH_MODULE, outside]);
    assert_clean(&errors_ir(&exprs));
}

// ── Alias laundering ─────────────────────────────────────────────

#[test]
fn alias_laundering_ascription_rejected() {
    // `type P2 = Probability` in another module must not launder the
    // forge gate: aliases resolve to the nominal registry entry.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/alias_laundering_ascription_rejected_742.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn alias_laundering_cast_rejected() {
    let agent = r#"(module {}
  agent.alias
  (typealias {} P2 () (t-adt {} Probability))
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (cast {} (var {} x) (t-adt {} P2)))))
"#;
    let exprs = deep_of_dp(&format!("{RECORD_UPDATE_DEFINING_DP}{agent}"));
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn alias_laundering_record_construction_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/alias_laundering_record_construction_rejected_779.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

// ── Macro call-site attribution ──────────────────────────────────

#[test]
fn expanded_record_is_attributed_to_call_site_module() {
    let exprs = deep_of_dp(include_str!(
        "fixtures/core/tests/opaque_types/expanded_call_site.deep"
    ));
    assert_single_violation(&errors_ir(&exprs));
}

#[test]
fn lexical_cross_module_macro_call_fails_closed_as_unbound() {
    // Baseline pin (passes before and after W1): in the lexical
    // encoding a macro defined inside the defining module is NOT
    // visible to sibling modules; the call neither expands nor
    // constructs, and the checker reports the unbound name. If macro
    // visibility ever widens, this pin trips and the call-site
    // attribution tests above must take over the construction case.
    let defining = include_str!(
        "fixtures/core/tests/opaque_types/lexical_cross_module_macro_call_fails_closed_as_unbound_852.deep"
    );
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/lexical_cross_module_macro_call_fails_closed_as_unbound_860.deep"
    );
    let exprs = core_modules(&[defining, outside]);
    let errors = errors_ir(&exprs);
    assert!(
        errors.iter().any(|e| {
            kind_name(e) == "UnboundVariable" && e.message.contains("unbound variable: forge_prob")
        }),
        "cross-module macro call must fail closed as unbound, never silently construct: {errors:?}"
    );
}

// ── Nested containment ───────────────────────────────────────────

#[test]
fn nested_record_with_legally_obtained_opaque_value_passes() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/nested_record_with_legally_obtained_opaque_value_passes_877.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_clean(&errors_ir(&exprs));
}

#[test]
fn nested_record_with_inline_opaque_literal_rejected() {
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/nested_record_with_inline_opaque_literal_rejected_888.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_single_violation(&errors_ir(&exprs));
}

// ── Declaration rules ────────────────────────────────────────────

#[test]
fn opaque_outside_named_module_is_declaration_error() {
    // D-CHECK / RT-0 M6: `@opaque` requires a named enclosing module.
    let bare = include_str!(
        "fixtures/core/tests/opaque_types/opaque_outside_named_module_is_declaration_error_908.deep"
    );
    let exprs = core_modules(&[bare]);
    let errors = errors_ir(&exprs);
    let violations = opaque_violations(&errors);
    assert_eq!(
        violations.len(),
        1,
        "expected exactly one declaration error, got: {errors:?}"
    );
}

#[test]
fn duplicate_same_name_deftype_across_modules_rejected() {
    // Baseline lock (passes before and after W1): nominal opacity
    // keying relies on the existing DuplicateDefinition rejection;
    // the registry insert is otherwise last-write-wins.
    let second = include_str!(
        "fixtures/core/tests/opaque_types/duplicate_same_name_deftype_across_modules_rejected_931.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, second]);
    let errors = errors_ir(&exprs);
    assert!(
        errors.iter().any(|e| {
            kind_name(e) == "DuplicateDefinition"
                && e.message
                    == "duplicate type definition: `Probability` was already declared as a deftype"
        }),
        "expected the DuplicateDefinition rejection, got: {errors:?}"
    );
}

// ── Module re-open forge (RT-1 F2, RFC v4b) ──────────────────────

#[test]
fn module_reopen_same_name_rejected() {
    // RT-1 F2: a named module opened by a SECOND `(module ...)`
    // wrapper in the same check unit forges module identity -- the
    // second wrapper constructs and accesses the opaque type as if it
    // were inside the defining module. A named module may be opened
    // at most once per check unit.
    let program = r#"(module {}
  stats.prob
  (deftype {opaque: true}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32))))
  (defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability)))
  (def {}
    probability
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x))))))
(module {}
  stats.prob
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x))))))
"#;
    let exprs = deep_of_dp(program);
    let errors = errors_ir(&exprs);
    let dup: Vec<&CheckError> = errors
        .iter()
        .filter(|e| kind_name(e) == "DuplicateModule")
        .collect();
    assert_eq!(
        dup.len(),
        1,
        "expected exactly one DuplicateModule error, got: {errors:?}"
    );
}

#[test]
fn distinct_module_names_in_one_check_unit_pass() {
    // Negative parity: two DIFFERENT module wrappers in one check
    // unit is the ordinary out-of-module setup the whole suite relies
    // on; it must NOT trip the re-open rule.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/distinct_module_names_in_one_check_unit_pass_1000.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    let errors = errors_ir(&exprs);
    assert!(
        !errors.iter().any(|e| kind_name(e) == "DuplicateModule"),
        "distinct module names must not trip the re-open rule: {errors:?}"
    );
    assert_clean(&errors);
}

#[test]
fn module_reopen_reported_once_per_name() {
    // Three wrappers of the same name yield ONE DuplicateModule error
    // for that name (reported once, not once per extra wrapper).
    let program = r#"(module {} a.b (def {} f (lit {type: (t-prim {} f32)} 1.0)))
(module {} a.b (def {} g (lit {type: (t-prim {} f32)} 2.0)))
(module {} a.b (def {} h (lit {type: (t-prim {} f32)} 3.0)))
"#;
    let exprs = deep_of_dp(program);
    let errors = errors_ir(&exprs);
    let dup: Vec<&CheckError> = errors
        .iter()
        .filter(|e| kind_name(e) == "DuplicateModule")
        .collect();
    assert_eq!(
        dup.len(),
        1,
        "a re-opened name is reported once, got: {errors:?}"
    );
    assert!(dup[0].message.contains("module `a.b`"));
}

// ── Reef-stem module-identity forge (RT-1 F2 bypass, RFC v5) ──────

fn reserved_name_violations(errors: &[CheckError]) -> Vec<&CheckError> {
    errors
        .iter()
        .filter(|e| kind_name(e) == "ReservedLinkerName")
        .collect()
}

#[test]
fn stem_only_mangled_names_rejected() {
    // RT-1 F2 bypass core case: a pure-flat program of reef-internal
    // mangled names (NO `(module ...)` wrappers). The mangled
    // `deftype Pkg__foo__Secret` and `def pkg__foo__forge` both
    // stem-key to module `foo`, so without the fix the forge is
    // treated as in-module and constructs/inspects the opaque type
    // clean. The reserved-name format is the reef linker's private
    // output; a raw program using it is a declaration error.
    let program = r#"(deftype {opaque: true}
  Pkg__foo__Secret
  ()
  (variant {} Pkg__foo__Secret (field {} value (t-prim {} f32))))
(defsig {} pkg__foo__forge (t-fn {} (t-prim {} f32) (t-adt {} Pkg__foo__Secret)))
(def {}
  pkg__foo__forge
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (record {} Pkg__foo__Secret (kv {} value (var {} x)))))
"#;
    let exprs = deep_of_dp(program);
    let errors = errors_ir(&exprs);
    let reserved = reserved_name_violations(&errors);
    assert!(
        !reserved.is_empty(),
        "mangled deftype/def names must be rejected as ReservedLinkerName, got: {errors:?}"
    );
}

#[test]
fn stem_with_lexical_wrapper_collision_rejected() {
    // RT-1 F2 bypass + belt-and-suspenders: a stem-derived module
    // identity (mangled flat `deftype Pkg__foo__Secret` keys to `foo`)
    // colliding with a lexical `(module {} foo ...)` wrapper is BOTH a
    // ReservedLinkerName (the mangled name) AND a DuplicateModule (the
    // stem-vs-wrapper collision).
    let program = r#"(deftype {opaque: true}
  Pkg__foo__Secret
  ()
  (variant {} Pkg__foo__Secret (field {} value (t-prim {} f32))))
(module {}
  foo
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Pkg__foo__Secret (kv {} value (var {} x))))))
"#;
    let exprs = deep_of_dp(program);
    let errors = errors_ir(&exprs);
    assert!(
        !reserved_name_violations(&errors).is_empty(),
        "the mangled deftype must be a ReservedLinkerName: {errors:?}"
    );
    let dup: Vec<&CheckError> = errors
        .iter()
        .filter(|e| kind_name(e) == "DuplicateModule")
        .collect();
    assert_eq!(
        dup.len(),
        1,
        "stem-vs-lexical collision must be one DuplicateModule: {errors:?}"
    );
}

#[test]
fn stem_collision_belt_fires_even_when_linker_name_check_is_off() {
    // Belt-and-suspenders independence (RFC v5): with the linked
    // provenance flag forced TRUE (simulating linker output, where the
    // reserved-name check is off), the stem-vs-lexical collision still
    // fires as DuplicateModule. Genuine linker output never has lexical
    // wrappers, so this never false-fires on it; here a lexical
    // wrapper is present, exposing the forge.
    let program = r#"(deftype {opaque: true}
  Pkg__foo__bar__Secret
  ()
  (variant {} Pkg__foo__bar__Secret (field {} value (t-prim {} f32))))
(module {}
  foo.bar
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Pkg__foo__bar__Secret (kv {} value (var {} x))))))
"#;
    let exprs = deep_of_dp(program);
    let _linked = chelis_types::install_linked_program_guard();
    let errors = errors_ir(&exprs);
    assert!(
        reserved_name_violations(&errors).is_empty(),
        "with the linked flag on, the reserved-name check is off: {errors:?}"
    );
    assert!(
        errors.iter().any(|e| kind_name(e) == "DuplicateModule"),
        "the stem-vs-lexical collision must still fire as DuplicateModule: {errors:?}"
    );
}

#[test]
fn mangled_names_accepted_when_linked_flag_is_set() {
    // No-regression unit: the SAME flat mangled program is accepted
    // (no ReservedLinkerName) when the linked-program provenance flag
    // is set -- this is what keeps the reef linker's own output
    // checking clean.
    let program = r#"(deftype {opaque: true}
  Pkg__foo__Secret
  ()
  (variant {} Pkg__foo__Secret (field {} value (t-prim {} f32))))
(defsig {} pkg__foo__forge (t-fn {} (t-prim {} f32) (t-adt {} Pkg__foo__Secret)))
(def {}
  pkg__foo__forge
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (record {} Pkg__foo__Secret (kv {} value (var {} x)))))
"#;
    let exprs = deep_of_dp(program);
    let _linked = chelis_types::install_linked_program_guard();
    let errors = errors_ir(&exprs);
    assert!(
        reserved_name_violations(&errors).is_empty(),
        "linker output must check clean under the linked flag: {errors:?}"
    );
}

// ── Exhaustiveness over opaque scrutinees ────────────────────────

#[test]
fn outside_match_with_irrefutable_var_arm_passes() {
    // RT-0 verified false positive: a bare `| x =>` arm currently
    // reports NonExhaustiveMatch. Outside code may match an opaque
    // scrutinee only with irrefutable patterns, so the arm-level fix
    // is part of W1 (red until Unit 5).
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_match_with_irrefutable_var_arm_passes_1185.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_clean(&errors_ir(&exprs));
}

#[test]
fn outside_match_with_irrefutable_as_arm_passes() {
    // `| q @ x =>` is the pat-as-wrapped irrefutable shape; RFC v2
    // widened the fix to cover it.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_match_with_irrefutable_as_arm_passes_1198.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_clean(&errors_ir(&exprs));
}

#[test]
fn outside_match_with_wildcard_arm_passes() {
    // Baseline (passes before and after): `| _ =>` already covers.
    let outside = include_str!(
        "fixtures/core/tests/opaque_types/outside_match_with_wildcard_arm_passes_1210.deep"
    );
    let exprs = core_modules(&[PROB_MODULE, outside]);
    assert_clean(&errors_ir(&exprs));
}

#[test]
fn nested_pat_var_on_non_opaque_adt_still_does_not_cover() {
    // Negative parity for the exhaustiveness fix: a NESTED pat-var
    // must keep not-covering, so exhaustiveness is not weakened on
    // ordinary ADTs. Baseline behavior, pinned before the fix.
    let src = include_str!(
        "fixtures/core/tests/opaque_types/nested_pat_var_on_non_opaque_adt_still_does_not_cover_1224.deep"
    );
    let exprs = core_modules(&[src]);
    let errors = errors_ir(&exprs);
    assert!(
        errors.iter().any(|e| {
            kind_name(e) == "NonExhaustiveMatch"
                && e.message
                    .contains("non-exhaustive match: missing variants [\"Empty\"]")
        }),
        "nested pat-var must not cover the match: {errors:?}"
    );
}
