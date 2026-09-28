//! chelis#486: actionable diagnostic hints for common agent-repair failures.
//!
//! Tests that type mismatch errors involving opaque types and Option[T]
//! produce targeted suggestions.

use chelis_types::errors::enrich_type_mismatch_suggestions;

// ── Item 1: Opaque accessor hints ──────────────────────────────────

#[test]
fn opaque_type_vs_primitive_suggests_accessor() {
    let mut suggestions = Vec::new();
    enrich_type_mismatch_suggestions("type mismatch: Probability vs f32", &mut suggestions);
    assert!(
        suggestions
            .iter()
            .any(|s| s.contains("probability_value(...)") && s.contains("extract the inner f32")),
        "expected opaque accessor hint; got: {suggestions:?}"
    );
}

#[test]
fn primitive_vs_opaque_type_suggests_accessor() {
    let mut suggestions = Vec::new();
    enrich_type_mismatch_suggestions("type mismatch: f32 vs Price", &mut suggestions);
    assert!(
        suggestions
            .iter()
            .any(|s| s.contains("price_value(...)") && s.contains("extract the inner f32")),
        "expected opaque accessor hint; got: {suggestions:?}"
    );
}

#[test]
fn non_opaque_adt_no_accessor_hint() {
    let mut suggestions = Vec::new();
    enrich_type_mismatch_suggestions("type mismatch: Option[f32] vs f32", &mut suggestions);
    // Should NOT suggest accessor for Option (it's not opaque)
    assert!(
        !suggestions.iter().any(|s| s.contains("_value(...)")),
        "should not suggest accessor for Option; got: {suggestions:?}"
    );
}

// ── Item 2: Option constructor hints ────────────────────────────────

#[test]
fn option_t_vs_t_suggests_match_unwrap() {
    let mut suggestions = Vec::new();
    enrich_type_mismatch_suggestions("type mismatch: Option[f32] vs f32", &mut suggestions);
    assert!(
        suggestions
            .iter()
            .any(|s| s.contains("Option[f32]") && s.contains("match ... with")),
        "expected Option unwrap hint; got: {suggestions:?}"
    );
}

#[test]
fn t_vs_option_t_suggests_match_unwrap() {
    let mut suggestions = Vec::new();
    enrich_type_mismatch_suggestions("type mismatch: i32 vs Option[i32]", &mut suggestions);
    assert!(
        suggestions
            .iter()
            .any(|s| s.contains("Option[i32]") && s.contains("match ... with")),
        "expected Option unwrap hint; got: {suggestions:?}"
    );
}

#[test]
fn option_mismatch_with_different_inner_type_no_hint() {
    let mut suggestions = Vec::new();
    enrich_type_mismatch_suggestions("type mismatch: Option[f32] vs i32", &mut suggestions);
    // inner type f32 != i32, so no option hint
    assert!(
        !suggestions.iter().any(|s| s.contains("match ... with")),
        "should not suggest unwrap when inner types differ; got: {suggestions:?}"
    );
}
