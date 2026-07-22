use std::collections::HashMap;

use chelis_types::CheckedProgram;

#[test]
fn public_reconstruction_rejects_a_missing_runtime_stamp_exactly_once() {
    let exprs = chelis_deep::parser::parse_str("(app {})").expect("fixture parses");
    let error = CheckedProgram::try_from_parts(exprs, HashMap::new())
        .expect_err("unchecked Deep must not become a CheckedProgram");

    assert_eq!(error.errors.len(), 1);
    assert!(error.errors[0].message.contains("missing its type stamp"));
}
