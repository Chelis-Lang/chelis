//! Scalar ascriptions check the suffix/default binding; [04] §5.3/§5.6.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::check_typed_program;

fn check(source: &str) -> Result<(), String> {
    let deep = desugar_program(&parse_str(source).expect("Surf parse")).expect("desugar");
    check_typed_program(&deep)
        .map(|_| ())
        .map_err(|e| format!("{:?}", e.errors))
}

#[test]
fn scalar_expression_and_local_binding_ascriptions_reject_dtype_changes() {
    for (literal, target) in [
        ("1.5f64", "f32"),
        ("16777217.0f64", "f32"),
        ("1.1f32", "f64"),
        ("1.1", "f64"),
        ("1", "f64"),
        ("1i64", "i32"),
        ("1i32", "i64"),
    ] {
        // A declaration states the dtype of an unsuffixed literal initializer
        // (spec/04 §5.6), so only a suffixed one can mismatch there; an
        // ascription states nothing and always checks.
        let suffixed = ["f32", "f64", "i32", "i64"]
            .iter()
            .any(|suffix| literal.ends_with(suffix));
        let mut sources = vec![format!("out = ({literal} : {target})\n")];
        if suffixed {
            sources.push(format!("out = {{\n  y: {target} = {literal}\n  y\n}}\n"));
            sources.push(format!("out: {target} = {literal}\n"));
        }
        for source in sources {
            let error = check(&source).expect_err(&source);
            assert!(
                error.contains("PrecisionMismatch") || error.contains("TypeMismatch"),
                "{source}: {error}"
            );
            assert!(
                !error.contains("integer atom cannot carry"),
                "{source}: {error}"
            );
        }
    }
}

#[test]
fn matching_ascriptions_and_explicit_casts_keep_their_literal_binding() {
    for (literal, target) in [
        ("1.5f64", "f64"),
        ("1.1f32", "f32"),
        ("1.1", "f32"),
        ("1", "i32"),
        ("1i64", "i64"),
    ] {
        for source in [
            format!("out = ({literal} : {target})\n"),
            format!("out = {{\n  y: {target} = {literal}\n  y\n}}\n"),
        ] {
            check(&source).unwrap_or_else(|e| panic!("{source}: {e}"));
        }
    }
    for (literal, target) in [("1.1", "f64"), ("1", "f64"), ("3000000000", "i64")] {
        for source in [
            format!("out = {{\n  y: {target} = {literal}\n  y\n}}\n"),
            format!("out: {target} = {literal}\n"),
        ] {
            check(&source).unwrap_or_else(|e| panic!("{source}: {e}"));
        }
    }
    for source in [
        "out = (cast(1.1f32, f64) : f64)\n",
        "out = (cast(1.1, f64) : f64)\n",
        "out = {\n  y: f64 = cast(1.1, f64)\n  y\n}\n",
    ] {
        check(source).unwrap_or_else(|e| panic!("{source}: {e}"));
    }
}

#[test]
fn every_numeric_width_checks_the_same_ascription_contract() {
    for (value, widths) in [
        ("1.5", &["f16", "bf16", "f32", "f64"][..]),
        ("1", &["i8", "i16", "i32", "i64"][..]),
    ] {
        for source_width in widths {
            for target_width in widths {
                for source in [
                    format!("out = ({value}{source_width} : {target_width})\n"),
                    format!("out = {{\n  y: {target_width} = {value}{source_width}\n  y\n}}\n"),
                ] {
                    if source_width == target_width {
                        check(&source).unwrap_or_else(|e| panic!("{source}: {e}"));
                    } else {
                        let error = check(&source).expect_err(&source);
                        assert!(
                            error.contains("PrecisionMismatch") || error.contains("TypeMismatch"),
                            "{source}: {error}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn transparent_aliases_and_nested_ascriptions_cannot_erase_a_mismatch() {
    for source in [
        "type Wide = f64\nout = (1.1 : Wide)\n",
        "out = ((1.1f64 : f32) : f64)\n",
        "out = {\n  y: f64 = (1.1f64 : f32)\n  y\n}\n",
    ] {
        let error = check(source).expect_err(source);
        assert!(
            error.contains("PrecisionMismatch") || error.contains("TypeMismatch"),
            "{source}: {error}"
        );
    }
}
