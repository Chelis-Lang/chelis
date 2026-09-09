//! Spec/05 section 3.4: normalization is an authored composition.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{builtin_decl, check_typed_program};

#[test]
fn undeclared_normalize_is_an_unknown_binding() {
    let source = "def f(x: tensor[4, f32]) -> tensor[4, f32] = normalize(x)";
    let parsed = parse_str(source).expect("valid Surf");
    let report = check_typed_program(&desugar_program(&parsed)).expect_err("no normalize builtin");
    assert!(
        report.errors.iter().any(|e| e.message.contains("normalize")
            && matches!(
                e.kind,
                chelis_types::errors::CheckErrorKind::UnboundVariable { .. }
            )),
        "{:#?}",
        report.errors
    );
    assert!(builtin_decl("normalize").is_none());
}

#[test]
fn authored_normalize_is_an_ordinary_function() {
    let source = "def normalize(x: f32) -> f32 = x / 2.0\ny = normalize(4.0)";
    let parsed = parse_str(source).expect("valid Surf");
    assert!(check_typed_program(&desugar_program(&parsed)).is_ok());
}
