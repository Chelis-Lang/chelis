//! Ordinary inline function types have one owner under Surf §P4 and §5.2.

use chelis_deep::{Expr, parse_and_stamp_file, printer::print_canonical_flat};
use chelis_surf::{
    desugar::desugar_program,
    parser::parse_str,
    resugar::{normalize_deep_for_surface_roundtrip, resugar_program},
};

fn lower(source: &str) -> Vec<Expr> {
    desugar_program(&parse_str(source).expect("parse source fixture"))
}

#[test]
fn an_inline_parameter_has_one_type_owner() {
    let program = lower("def classify(x: f8e4m3) -> int32 = 0i32");
    let text = print_canonical_flat(&program);
    assert_eq!(text.matches("f8e4m3").count(), 1, "{text}");
    assert!(
        text.contains("(params {} (x {type: (t-var {} _)}))"),
        "{text}"
    );
    assert!(
        text.contains("(t-fn {} (t-prim {} f8e4m3) (t-prim {} int32))"),
        "{text}"
    );
}

#[test]
fn a_standalone_signature_keeps_the_independent_parameter_annotation() {
    let text = print_canonical_flat(&lower(
        "sig classify: f8e4m3 -> int32\ndef classify(x: f8e4m3) -> int32 = 0i32",
    ));
    assert_eq!(text.matches("f8e4m3").count(), 2, "{text}");
    assert!(
        text.contains("(params {} (x {type: (t-prim {} f8e4m3)}))"),
        "{text}"
    );
}

#[test]
fn a_lambda_keeps_its_own_type_annotation() {
    let text = print_canonical_flat(&lower("classify = fn (x: f8e4m3) -> 0i32"));
    assert_eq!(text.matches("f8e4m3").count(), 1, "{text}");
    assert!(
        text.contains("(params {} (x {type: (t-prim {} f8e4m3)}))"),
        "{text}"
    );
    assert!(!text.contains("defsig"), "{text}");
}

#[test]
fn explicit_holes_and_omitted_annotations_remain_distinct() {
    let text = print_canonical_flat(&lower("def choose(x: _, y, z: int32) -> int32 = z"));
    assert!(
        text.contains("(params {} (x {type: (t-var {} _)}) y (z {type: (t-var {} _)}))"),
        "{text}"
    );
}

#[test]
fn an_inline_parameter_keeps_its_original_precision_scope() {
    let text = print_canonical_flat(&lower("def inspect(x: tensor[3, p]) -> int32 = 0i32"));
    assert!(
        text.contains("(t-tensor {} (d-lit {} 3) (t-prim {} p))"),
        "{text}"
    );
    assert!(!text.contains("(t-var {} p)"), "{text}");
    let text = print_canonical_flat(&lower("def inspect[p](x: tensor[3, p]) -> int32 = 0i32"));
    assert!(
        text.contains("(t-tensor {} (d-lit {} 3) (t-var {} p))"),
        "{text}"
    );
    assert!(!text.contains("(t-prim {} p)"), "{text}");
}

#[test]
fn the_single_owner_form_satisfies_the_round_trip_law() {
    let program = parse_and_stamp_file(
        "(defsig {} identity (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n\
         (def {} identity (fn {} (params {} (x {type: (t-var {} _)})) (var {} x)))",
    )
    .expect("parse canonical owner fixture");
    let restored = desugar_program(&resugar_program(&program).expect("resugar a declared slot"));
    assert_eq!(
        print_canonical_flat(
            &normalize_deep_for_surface_roundtrip(&restored).expect("normalize restored program")
        ),
        print_canonical_flat(
            &normalize_deep_for_surface_roundtrip(&program).expect("normalize original program")
        ),
    );
}

#[test]
fn a_real_parameter_disagreement_cannot_normalize_to_a_hole() {
    let program = parse_and_stamp_file(
        "(defsig {} identity (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n\
         (def {} identity (fn {} (params {} (x {type: (t-prim {} bool)})) (var {} x)))",
    )
    .expect("parse independent constraint fixture");
    assert!(resugar_program(&program).is_err());
    let normalized =
        normalize_deep_for_surface_roundtrip(&program).expect("normalize valid syntax");
    assert!(print_canonical_flat(&normalized).contains("type: (t-prim {} bool)"));
}
