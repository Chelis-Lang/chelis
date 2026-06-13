//! Unit tests for the invariant-predicate grammar and amenability
//! classifier (RFC D-PRED / D-WF). Predicate fn nodes are parsed from
//! Deep source so the tests are pinned to the real encoding the
//! desugarer produces (verified against the surf desugar probe in W2
//! pre-flight).

use super::*;
use chelis_deep::parser::parse_str;

/// Parse a single Deep expression (the predicate fn node).
fn fnnode(src: &str) -> Expr {
    let exprs = parse_str(src).expect("predicate fn parses");
    assert_eq!(exprs.len(), 1, "expected one top-level expr");
    exprs.into_iter().next().unwrap()
}

// ---------------------------------------------------------------------------
// Representative predicates (one per amenability class) in the canonical
// `(fn {} (params {} p) <body>)` schema, matching the desugar probe.
// ---------------------------------------------------------------------------

/// Linear: `p.value >= 0.0 and p.value <= 1.0`.
const LINEAR: &str = "(fn {} (params {} p) \
    (app {} (var {} and) \
        (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0)) \
        (app {} (var {} lte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 1.0))))";

/// Polynomial: `p.value * p.value <= 1.0`.
const POLYNOMIAL: &str = "(fn {} (params {} p) \
    (app {} (var {} lte) \
        (app {} (var {} mul) (access {} (var {} p) value) (access {} (var {} p) value)) \
        (lit {type: (t-prim {} f32)} 1.0)))";

/// Transcendental: `exp(p.value) <= 3.0`.
const TRANSCENDENTAL: &str = "(fn {} (params {} p) \
    (app {} (var {} lte) \
        (app {} (var {} exp) (access {} (var {} p) value)) \
        (lit {type: (t-prim {} f32)} 3.0)))";

/// Simplex tolerance band (W6 flagship):
/// `sum(p.weights) >= 1.0 - eps and sum(p.weights) <= 1.0 + eps`.
const SIMPLEX: &str = "(fn {} (params {} p) \
    (app {} (var {} and) \
        (app {} (var {} gte) (app {} (var {} sum) (access {} (var {} p) weights)) \
            (app {} (var {} sub) (lit {type: (t-prim {} f32)} 1.0) (var {} eps))) \
        (app {} (var {} lte) (app {} (var {} sum) (access {} (var {} p) weights)) \
            (app {} (var {} add) (lit {type: (t-prim {} f32)} 1.0) (var {} eps)))))";

// ===========================================================================
// PredAmenability as_str / from_str round-trip
// ===========================================================================

#[test]
fn amenability_as_str_canonical() {
    assert_eq!(PredAmenability::Linear.as_str(), "linear");
    assert_eq!(PredAmenability::Polynomial.as_str(), "polynomial");
    assert_eq!(PredAmenability::Transcendental.as_str(), "transcendental");
    assert_eq!(PredAmenability::Opaque.as_str(), "opaque");
}

#[test]
fn amenability_from_str_round_trip() {
    for a in [
        PredAmenability::Linear,
        PredAmenability::Polynomial,
        PredAmenability::Transcendental,
        PredAmenability::Opaque,
    ] {
        assert_eq!(PredAmenability::from_str(a.as_str()), Some(a));
    }
}

#[test]
fn amenability_from_str_rejects_unknown() {
    assert_eq!(PredAmenability::from_str("Linear"), None);
    assert_eq!(PredAmenability::from_str(""), None);
    assert_eq!(PredAmenability::from_str("nonlinear"), None);
}

// ===========================================================================
// classify_predicate: one per class
// ===========================================================================

#[test]
fn classify_linear() {
    assert_eq!(classify_predicate(&fnnode(LINEAR)), PredAmenability::Linear);
}

#[test]
fn classify_polynomial() {
    assert_eq!(
        classify_predicate(&fnnode(POLYNOMIAL)),
        PredAmenability::Polynomial
    );
}

#[test]
fn classify_transcendental() {
    assert_eq!(
        classify_predicate(&fnnode(TRANSCENDENTAL)),
        PredAmenability::Transcendental
    );
}

#[test]
fn classify_simplex_is_linear() {
    // The simplex tolerance band is affine: `sum` of a field is a linear
    // combination, the band bounds are affine in the module constant eps.
    assert_eq!(
        classify_predicate(&fnnode(SIMPLEX)),
        PredAmenability::Linear
    );
}

// ===========================================================================
// Misclassification probes (negative parity for the classifier)
// ===========================================================================

#[test]
fn polynomial_disguised_as_linear() {
    // `p.value * p.value` LOOKS like two field accesses; it must NOT be
    // classified Linear. A product of two non-constant subterms is
    // Polynomial.
    assert_eq!(
        classify_predicate(&fnnode(POLYNOMIAL)),
        PredAmenability::Polynomial
    );
}

#[test]
fn scaling_by_constant_stays_linear() {
    // `p.value * 2.0 <= 1.0`: a product where one operand is a literal
    // is affine, NOT polynomial.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} mul) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 2.0)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Linear);
}

#[test]
fn scaling_by_module_constant_stays_linear() {
    // `p.value * scale <= 1.0` where `scale` is an in-module constant
    // (a bare var that is not the binder) is affine.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} mul) (access {} (var {} p) value) (var {} scale)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Linear);
}

#[test]
fn transcendental_dominates_polynomial() {
    // `exp(p.value * p.value) <= 3.0`: a polynomial argument under a
    // transcendental call classifies Transcendental.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} exp) \
                (app {} (var {} mul) (access {} (var {} p) value) (access {} (var {} p) value))) \
            (lit {type: (t-prim {} f32)} 3.0)))";
    assert_eq!(
        classify_predicate(&fnnode(src)),
        PredAmenability::Transcendental
    );
}

#[test]
fn nested_polynomial_under_if_lifts() {
    // Polynomial buried inside an `if` branch still lifts the class.
    let src = "(fn {} (params {} p) \
        (if {} (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0)) \
            (app {} (var {} lte) \
                (app {} (var {} mul) (access {} (var {} p) value) (access {} (var {} p) value)) \
                (lit {type: (t-prim {} f32)} 1.0)) \
            (lit {type: (t-prim {} bool)} false)))";
    assert_eq!(
        classify_predicate(&fnnode(src)),
        PredAmenability::Polynomial
    );
}

#[test]
fn out_of_grammar_classifies_opaque() {
    // A general function call (`my_helper`) is out of grammar.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} my_helper) (access {} (var {} p) value)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Opaque);
}

#[test]
fn malformed_fn_classifies_opaque() {
    // Not a fn node at all.
    let src = "(app {} (var {} and) (lit {type: (t-prim {} bool)} true))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Opaque);
}

// ===========================================================================
// predicate_in_grammar: accept / reject table
// ===========================================================================

#[test]
fn grammar_accepts_representative_predicates() {
    assert!(predicate_in_grammar(&fnnode(LINEAR)).is_ok());
    assert!(predicate_in_grammar(&fnnode(POLYNOMIAL)).is_ok());
    assert!(predicate_in_grammar(&fnnode(TRANSCENDENTAL)).is_ok());
    assert!(predicate_in_grammar(&fnnode(SIMPLEX)).is_ok());
}

#[test]
fn grammar_accepts_all_intrinsics() {
    for f in INTRINSIC_WHITELIST {
        let src = format!(
            "(fn {{}} (params {{}} p) \
                (app {{}} (var {{}} lte) \
                    (app {{}} (var {{}} {f}) (access {{}} (var {{}} p) value)) \
                    (lit {{type: (t-prim {{}} f32)}} 1.0)))"
        );
        assert!(
            predicate_in_grammar(&fnnode(&src)).is_ok(),
            "intrinsic `{f}` should be in grammar"
        );
    }
}

#[test]
fn grammar_accepts_if_and_division() {
    // `if` and `/` are admitted (division is partial — totality not
    // guaranteed, per D-WF).
    let src = "(fn {} (params {} p) \
        (if {} (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0)) \
            (app {} (var {} lte) \
                (app {} (var {} div) (lit {type: (t-prim {} f32)} 1.0) (access {} (var {} p) value)) \
                (lit {type: (t-prim {} f32)} 1.0)) \
            (lit {type: (t-prim {} bool)} false)))";
    assert!(predicate_in_grammar(&fnnode(src)).is_ok());
}

#[test]
fn grammar_rejects_general_call() {
    let src = "(fn {} (params {} p) \
        (app {} (var {} my_helper) (access {} (var {} p) value)))";
    assert_eq!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedCall("my_helper".to_string()))
    );
}

#[test]
fn grammar_rejects_match() {
    let src = "(fn {} (params {} p) \
        (match {} (access {} (var {} p) value) \
            (arm {} (pat-wild {}) () (lit {type: (t-prim {} bool)} true))))";
    assert!(matches!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedNode(_))
    ));
}

#[test]
fn grammar_rejects_lambda() {
    let src = "(fn {} (params {} p) \
        (fn {} (params {} q) (access {} (var {} q) value)))";
    assert!(matches!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedNode(_))
    ));
}

#[test]
fn grammar_rejects_record_construction() {
    let src = "(fn {} (params {} p) \
        (record {} Foo (kv {} x (lit {type: (t-prim {} f32)} 1.0))))";
    assert!(matches!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::DisallowedNode(_))
    ));
}

#[test]
fn grammar_rejects_sum_over_non_field() {
    // `sum` applied to a literal (not a binder field projection).
    let src = "(fn {} (params {} p) \
        (app {} (var {} gte) \
            (app {} (var {} sum) (lit {type: (t-prim {} f32)} 1.0)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::BadSum)
    );
}

#[test]
fn grammar_rejects_non_fn_top() {
    let src = "(app {} (var {} and) (lit {type: (t-prim {} bool)} true))";
    assert!(matches!(
        predicate_in_grammar(&fnnode(src)),
        Err(PredGrammarError::NotAPredicateFn(_))
    ));
}

#[test]
fn grammar_accepts_sum_over_field() {
    let src = "(fn {} (params {} p) \
        (app {} (var {} gte) \
            (app {} (var {} sum) (access {} (var {} p) weights)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert!(predicate_in_grammar(&fnnode(src)).is_ok());
}

// ===========================================================================
// predicate_free_vars
// ===========================================================================

#[test]
fn free_vars_binder_only_is_empty() {
    // LINEAR references only the binder `p` and its field; no free vars.
    assert!(predicate_free_vars(&fnnode(LINEAR)).is_empty());
}

#[test]
fn free_vars_picks_up_module_constant() {
    // SIMPLEX references `eps`, an in-module constant, plus the binder.
    assert_eq!(
        predicate_free_vars(&fnnode(SIMPLEX)),
        vec!["eps".to_string()]
    );
}

#[test]
fn free_vars_excludes_field_selectors() {
    // The field names `value` are selectors, never variables.
    let vars = predicate_free_vars(&fnnode(POLYNOMIAL));
    assert!(!vars.iter().any(|v| v == "value"));
    assert!(vars.is_empty());
}

#[test]
fn free_vars_dedup_in_order() {
    // `p.value * a + b * a` references `a` (twice) and `b`; dedup keeps
    // first-seen order.
    let src = "(fn {} (params {} p) \
        (app {} (var {} gte) \
            (app {} (var {} add) \
                (app {} (var {} mul) (access {} (var {} p) value) (var {} a)) \
                (app {} (var {} mul) (var {} b) (var {} a))) \
            (lit {type: (t-prim {} f32)} 0.0)))";
    assert_eq!(
        predicate_free_vars(&fnnode(src)),
        vec!["a".to_string(), "b".to_string()]
    );
}

// ===========================================================================
// sum-expansion classification (D-PRED / D-WF)
// ===========================================================================

#[test]
fn sum_over_field_classifies_linear() {
    // `sum(p.weights) >= 1.0`: a sum of scalar terms is affine.
    let src = "(fn {} (params {} p) \
        (app {} (var {} gte) \
            (app {} (var {} sum) (access {} (var {} p) weights)) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(classify_predicate(&fnnode(src)), PredAmenability::Linear);
}

#[test]
fn sum_squared_is_polynomial() {
    // `sum(p.weights) * sum(p.weights) <= 1.0`: a product of two
    // non-constant sums is polynomial.
    let src = "(fn {} (params {} p) \
        (app {} (var {} lte) \
            (app {} (var {} mul) \
                (app {} (var {} sum) (access {} (var {} p) weights)) \
                (app {} (var {} sum) (access {} (var {} p) weights))) \
            (lit {type: (t-prim {} f32)} 1.0)))";
    assert_eq!(
        classify_predicate(&fnnode(src)),
        PredAmenability::Polynomial
    );
}

#[test]
fn intrinsic_whitelist_is_eight() {
    assert_eq!(INTRINSIC_WHITELIST.len(), 8);
    for f in ["abs", "min", "max", "sqrt", "exp", "log", "sin", "cos"] {
        assert!(INTRINSIC_WHITELIST.contains(&f), "missing {f}");
    }
}
