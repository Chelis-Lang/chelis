//! Ascription metadata must not replace a literal's independently bound dtype.
use chelis_deep::printer::print_canonical_flat;
use chelis_surf::{
    desugar::desugar_program,
    format::{format_program, format_source},
    parser::parse_str,
    resugar::{normalize_deep_for_surface_roundtrip, resugar_program},
};

#[test]
fn matching_and_mismatched_literal_ascriptions_preserve_both_types_and_roundtrip() {
    for source in [
        "out = (1.5f64 : f32)\n",
        "out = (1.5f64 : f64)\n",
        "out = ((1.5f64 : f32) : f64)\n",
        "out = {\n  y: f32 = 1.5f64\n  y\n}\n",
        "out = {\n  y: f64 = 1.5f64\n  y\n}\n",
        "out = (1 : f64)\n",
    ] {
        let deep = desugar_program(&parse_str(source).unwrap()).unwrap();
        let printed = print_canonical_flat(&normalize_deep_for_surface_roundtrip(&deep).unwrap());
        let surface = resugar_program(&deep).expect("ascription remains representable");
        let rendered = format_program(&surface);
        let recovered = desugar_program(&parse_str(&rendered).unwrap()).unwrap();
        assert_eq!(
            print_canonical_flat(&normalize_deep_for_surface_roundtrip(&deep).unwrap()),
            print_canonical_flat(&normalize_deep_for_surface_roundtrip(&recovered).unwrap()),
            "{source} -> {rendered}"
        );
        assert!(printed.contains("(block {type:"), "{source}: {printed}");
        if source.contains("1.5f64") {
            assert!(
                printed.contains("(lit {type: (t-prim {} f64)"),
                "{source}: {printed}"
            );
        } else {
            assert!(
                printed.contains("(lit {type: (t-prim {} i32)"),
                "{source}: {printed}"
            );
        }
    }
}

#[test]
fn checking_ascriptions_resugar_without_synthetic_do_blocks() {
    for source in [
        "out = (1.5f64 : f32)\n",
        "out = (1.5f64 : f64)\n",
        "out = ((1.5f64 : f32) : f64)\n",
        "out = {\n  y: f32 = 1.5f64\n  y\n}\n",
        "out = {\n  y: f64 = 1.5f64\n  y\n}\n",
        "out = {\n  y: i32 = 1\n  y\n}\n",
        "out = {\n  y: f64 = (1.5f64 : f32)\n  y\n}\n",
    ] {
        let deep = desugar_program(&parse_str(source).unwrap()).unwrap();
        let rendered = format_program(&resugar_program(&deep).unwrap());
        assert_eq!(rendered, format_source(source).unwrap(), "{source}");
        let recovered = desugar_program(&parse_str(&rendered).unwrap()).unwrap();
        assert_eq!(
            print_canonical_flat(&normalize_deep_for_surface_roundtrip(&deep).unwrap()),
            print_canonical_flat(&normalize_deep_for_surface_roundtrip(&recovered).unwrap()),
            "{source} -> {rendered}"
        );
    }
}

#[test]
fn untyped_and_multiple_expression_do_blocks_keep_their_structure() {
    for source in [
        "out = do { 1 }\n",
        "out = (do { x } : i32)\n",
        "out = (do { {\n  y = 1\n  y\n} } : i32)\n",
        "out = (do { 1; 2 } : i32)\n",
        "out = {\n  y: i32 = do { 1; 2 }\n  y\n}\n",
    ] {
        let deep = desugar_program(&parse_str(source).unwrap()).unwrap();
        let rendered = format_program(&resugar_program(&deep).unwrap());
        assert_eq!(rendered, format_source(source).unwrap(), "{source}");
        let recovered = desugar_program(&parse_str(&rendered).unwrap()).unwrap();
        assert_eq!(
            print_canonical_flat(&normalize_deep_for_surface_roundtrip(&deep).unwrap()),
            print_canonical_flat(&normalize_deep_for_surface_roundtrip(&recovered).unwrap()),
            "{source} -> {rendered}"
        );
    }
}
