#![cfg(test)]

use super::*;

fn deep(source: &str) -> Expr {
    chelis_deep::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("Deep fixture parses")
        .into_iter()
        .next()
        .expect("one expression")
}

fn substituted(source: &str, substitutions: &[(&str, &str)]) -> String {
    let substitutions = substitutions
        .iter()
        .map(|(name, replacement)| (name.to_string(), deep(replacement)))
        .collect::<UnordMap<_, _>>();
    crate::test_support::with_operation(|operation| {
        chelis_deep::printer::print_expr_flat(&substitute_expr(
            &deep(source),
            &substitutions,
            &UnordSet::new(),
            operation,
        ))
    })
}

fn flat(source: &str) -> String {
    chelis_deep::printer::print_expr_flat(&deep(source))
}

const ADD_K: &str = "(fn {} (params {} z) (app {} (var {} add) (var {} z) (var {} k)))";

#[test]
fn a_pattern_binder_shadows_the_formal_in_its_guard_and_body_only() {
    // The scrutinee and the binder-free arm are outside `Some(x)`'s scope,
    // so only they read the replacement.
    let source = "(match {} (var {} x) \
             (arm {} (pat-ctor {} Some (pat-var {} x)) \
               (app {} (var {} gt) (var {} x) (lit {} 5)) (var {} x)) \
             (arm {} (pat-wild {}) () (var {} x)))";
    assert_eq!(
        substituted(source, &[("x", "(var {} y)")]),
        flat(
            "(match {} (var {} y) \
                 (arm {} (pat-ctor {} Some (pat-var {} x)) \
                   (app {} (var {} gt) (var {} x) (lit {} 5)) (var {} x)) \
                 (arm {} (pat-wild {}) () (var {} y)))"
        )
    );
}

#[test]
fn a_capturing_arm_binder_is_renamed_in_pattern_guard_and_body() {
    // `g`'s replacement reads a free `k`; `Some(k)` would capture it.
    // The other arm binds nothing, so its `k` is the outer one and stays.
    let source = "(match {} (var {} o) \
             (arm {} (pat-ctor {} Some (pat-var {} k)) \
               (app {} (var {} gt) (app {} (var {} g) (var {} k)) (lit {} 10)) \
               (app {} (var {} g) (var {} k))) \
             (arm {} (pat-wild {}) () (var {} k)))";
    let expected = format!(
        "(match {{}} (var {{}} o) \
             (arm {{}} (pat-ctor {{}} Some (pat-var {{}} k__inl1)) \
               (app {{}} (var {{}} gt) (app {{}} {ADD_K} (var {{}} k__inl1)) (lit {{}} 10)) \
               (app {{}} {ADD_K} (var {{}} k__inl1))) \
             (arm {{}} (pat-wild {{}}) () (var {{}} k)))"
    );
    assert_eq!(substituted(source, &[("g", ADD_K)]), flat(&expected));
}

#[test]
fn binders_nested_in_as_tuple_record_and_constructor_patterns_are_renamed() {
    // Constructor names and the record field label are not binders and
    // must survive the rename; every binder that would capture is renamed.
    let lambda = "(fn {} (params {} z) \
             (app {} (var {} add) (var {} k) (var {} j) (var {} m)))";
    let source = "(match {} (var {} o) \
             (arm {} (pat-tuple {} (pat-as {} k (pat-wild {})) \
               (pat-record {} Pt (kv {} j (pat-var {} j))) \
               (pat-ctor {} Some (pat-var {} m))) () \
               (app {} (var {} g) (var {} k) (var {} j) (var {} m))))";
    let expected = format!(
        "(match {{}} (var {{}} o) \
             (arm {{}} (pat-tuple {{}} (pat-as {{}} k__inl1 (pat-wild {{}})) \
               (pat-record {{}} Pt (kv {{}} j (pat-var {{}} j__inl1))) \
               (pat-ctor {{}} Some (pat-var {{}} m__inl1))) () \
               (app {{}} {lambda} (var {{}} k__inl1) (var {{}} j__inl1) (var {{}} m__inl1))))"
    );
    assert_eq!(substituted(source, &[("g", lambda)]), flat(&expected));
}

#[test]
fn a_nested_match_arm_shadows_only_inside_itself() {
    let source = "(match {} (var {} o) \
             (arm {} (pat-ctor {} Some (pat-var {} a)) () \
               (match {} (var {} p) \
                 (arm {} (pat-ctor {} Some (pat-var {} x)) () (var {} x)) \
                 (arm {} (pat-wild {}) () (var {} x)))))";
    assert_eq!(
        substituted(source, &[("x", "(var {} y)")]),
        flat(
            "(match {} (var {} o) \
                 (arm {} (pat-ctor {} Some (pat-var {} a)) () \
                   (match {} (var {} p) \
                     (arm {} (pat-ctor {} Some (pat-var {} x)) () (var {} x)) \
                     (arm {} (pat-wild {}) () (var {} y)))))"
        )
    );
}

#[test]
fn a_binder_beside_a_shadowed_formal_is_not_renamed() {
    // `x`'s replacement reads `k`, but the arm also binds `x`, so that
    // replacement never reaches this arm and `k` cannot be captured: the
    // arm must come back unchanged rather than with a gratuitous rename.
    let source = "(match {} (var {} o) \
             (arm {} (pat-tuple {} (pat-var {} x) (pat-var {} k)) () \
               (app {} (var {} add) (var {} x) (var {} k))))";
    assert_eq!(substituted(source, &[("x", "(var {} k)")]), flat(source));
}
