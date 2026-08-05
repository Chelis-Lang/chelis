//! A Deep literal's atom kind and declared primitive family must agree.

use chelis_deep::parser::parse_str;
use chelis_types::{check_ir_program, check_typed_program};

fn diagnostics(source: &str) -> Vec<String> {
    let exprs = parse_str(source).expect("Deep fixture must parse");
    match check_ir_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|error| error.message.clone())
            .collect(),
    }
}

#[test]
fn int_atom_under_f32_metadata_is_rejected() {
    let errors = diagnostics("(lit {type: (t-prim {} f32)} 18014399583223809)");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("integer atom") && error.contains("f32")),
        "contradictory literal metadata checked clean: {errors:#?}"
    );
}

#[test]
fn all_canonical_atom_prim_pairs_are_accepted() {
    for source in [
        "(lit {type: (t-prim {} int32)} 7)",
        "(lit {type: (t-prim {} f32)} 7.0)",
        "(lit {type: (t-prim {} f32), literal_source: integer} 18014399583223809)",
        "(lit {type: (t-prim {} bool)} true)",
        "(lit {type: (t-prim {} string)} \"ok\")",
    ] {
        let errors = diagnostics(source);
        assert!(
            errors.is_empty(),
            "canonical literal rejected: {source}: {errors:#?}"
        );
    }
}

#[test]
fn every_cross_family_atom_prim_pair_is_rejected() {
    for source in [
        "(lit {type: (t-prim {} int32)} 1.0)",
        "(lit {type: (t-prim {} int32)} true)",
        "(lit {type: (t-prim {} int32)} \"x\")",
        "(lit {type: (t-prim {} f64)} 1)",
        "(lit {type: (t-prim {} f64)} true)",
        "(lit {type: (t-prim {} bool)} 1)",
        "(lit {type: (t-prim {} bool)} \"x\")",
        "(lit {type: (t-prim {} string)} 1)",
        "(lit {type: (t-prim {} string)} false)",
        "(lit {type: (t-prim {} int32), literal_source: integer} 1)",
        "(lit {type: (t-prim {} f32), literal_source: float} 1)",
        "(lit {type: (t-prim {} f32), literal_source: integer} 1.0)",
        "(lit {literal_source: integer} 1)",
    ] {
        let errors = diagnostics(source);
        assert!(
            !errors.is_empty(),
            "cross-family literal checked clean: {source}"
        );
    }
}

#[test]
fn integer_spelled_float_suffix_is_marked_by_both_producers() {
    let deep_errors = diagnostics("(def {} value (lit {} 7f32))");
    assert!(
        deep_errors.is_empty(),
        "Deep's typed-token producer emitted a contradictory atom: {deep_errors:#?}"
    );

    let declarations = chelis_surf::parser::parse_str(
        "def scalar() -> f32 = 7f32\n\
         def vector() -> tensor[2, f32] = [1, 2]",
    )
    .expect("Surf producer fixture must parse");
    let program = chelis_surf::desugar::desugar_program(&declarations);
    let errors = match check_typed_program(&program) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    };
    assert!(
        errors.is_empty(),
        "Surf's suffix/context producers emitted contradictory atoms: {errors:#?}"
    );
}
