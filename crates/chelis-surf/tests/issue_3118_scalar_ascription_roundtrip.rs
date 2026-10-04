//! Ascription metadata must not replace a literal's independently bound dtype.
use chelis_deep::printer::print_canonical_flat;
use chelis_surf::{
    desugar::desugar_program,
    format::format_program,
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
