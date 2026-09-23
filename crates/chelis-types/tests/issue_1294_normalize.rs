//! Spec/05 section 3.4: normalization is an authored composition.
use chelis_deep::parser::parse_str;
use chelis_types::{builtin_decl, check_typed_program};

#[test]
fn undeclared_normalize_is_an_unknown_binding() {
    let source = include_str!(
        "fixtures/core/tests/issue_1294_normalize/undeclared_normalize_is_an_unknown_binding_7.deep"
    );
    let parsed = parse_str(source).expect("valid Deep fixture");
    let report = check_typed_program(&parsed).expect_err("no normalize builtin");
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
    let source = include_str!(
        "fixtures/core/tests/issue_1294_normalize/authored_normalize_is_an_ordinary_function_24.deep"
    );
    let parsed = parse_str(source).expect("valid Deep fixture");
    assert!(check_typed_program(&parsed).is_ok());
}
