//! A `defsig` describes a same-unit `def`; it is not a runtime declaration.

use chelis_deep::parser::parse_str;
use chelis_types::{
    build_type_env_from_library, check_ir_program, check_ir_with_context,
    install_linked_program_guard,
};

fn errors(source: &str) -> Vec<String> {
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
fn bare_orphan_defsig_is_rejected() {
    let got = errors("(defsig {} missing (t-fn {} (t-prim {} string)))");
    assert!(
        got.iter()
            .any(|error| error.contains("missing") && error.contains("def")),
        "orphan defsig checked clean: {got:#?}"
    );
}

#[test]
fn module_orphan_defsig_is_rejected() {
    let got = errors(
        "(module {} M (export {} missing) (defsig {} missing (t-fn {} (t-prim {} string))))",
    );
    assert!(
        got.iter()
            .any(|error| error.contains("missing") && error.contains("def")),
        "module orphan defsig checked clean: {got:#?}"
    );
}

#[test]
fn paired_defsig_and_def_are_accepted() {
    let got = errors(
        "(defsig {} present (t-fn {} (t-prim {} string)))\n\n\
         (def {} present (fn {} (params {}) (lit {type: (t-prim {} string)} \"ok\")))",
    );
    assert!(got.is_empty(), "paired declaration rejected: {got:#?}");
}

#[test]
fn def_without_defsig_remains_valid() {
    let got = errors("(def {} inferred (lit {type: (t-prim {} i32)} 1))");
    assert!(got.is_empty(), "defsig-less def rejected: {got:#?}");
}

#[test]
fn a_previous_library_def_does_not_back_a_new_units_orphan_defsig() {
    let library =
        parse_str("(def {} existing (lit {type: (t-prim {} i32)} 1))").expect("library fixture");
    let context = build_type_env_from_library(&library).expect("library must check");
    let next = parse_str("(defsig {} existing (t-prim {} i32))").expect("next fixture");
    let errors = check_ir_with_context(&context, &next)
        .expect_err("a defsig must be backed in its own check unit")
        .errors;
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("same check unit")),
        "persistent context incorrectly backed an orphan defsig: {errors:#?}"
    );
}

#[test]
fn trusted_linked_dependency_interface_may_carry_signature_only_rows() {
    let _linked = install_linked_program_guard();
    let got = errors("(defsig {} pkg__dep__Api__external (t-fn {} (t-prim {} string)))");
    assert!(
        got.is_empty(),
        "trusted linked dependency interfaces are not authored check units: {got:#?}"
    );
}

#[test]
fn linked_provenance_does_not_exempt_an_ordinary_source_name() {
    let _linked = install_linked_program_guard();
    let got = errors("(defsig {} authored_missing (t-fn {} (t-prim {} string)))");
    assert!(
        got.iter()
            .any(|error| error.contains("authored_missing") && error.contains("def")),
        "linked provenance alone must not turn an authored orphan into an interface row: {got:#?}"
    );
}
