//! A `defsig` describes a same-unit `def`; it is not a runtime declaration.

use chelis_deep::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{
    ProgramOrigin, TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context,
};

fn errors(source: &str, origin: ProgramOrigin) -> Vec<CheckError> {
    let exprs = parse_str(source).expect("Deep fixture must parse");
    match check_ir_program(&exprs, origin) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

#[test]
fn bare_orphan_defsig_is_rejected() {
    let got = errors(
        "(defsig {} missing (t-fn {} (t-prim {} string)))",
        ProgramOrigin::Unlinked,
    );
    assert!(
        got.iter().any(|error| matches!(
            &error.kind,
            CheckErrorKind::UnboundVariable { identifier } if identifier == "missing"
        )),
        "orphan defsig checked clean: {got:#?}"
    );
}

#[test]
fn module_orphan_defsig_is_rejected() {
    let got = errors(
        "(module {} M (export {} missing) (defsig {} missing (t-fn {} (t-prim {} string))))",
        ProgramOrigin::Unlinked,
    );
    assert!(
        got.iter().any(|error| matches!(
            &error.kind,
            CheckErrorKind::UnboundVariable { identifier } if identifier == "M.missing"
        )),
        "module orphan defsig checked clean: {got:#?}"
    );
}

#[test]
fn paired_defsig_and_def_are_accepted() {
    let got = errors(
        "(defsig {} present (t-fn {} (t-prim {} string)))\n\n\
         (def {} present (fn {} (params {}) (lit {type: (t-prim {} string)} \"ok\")))",
        ProgramOrigin::Unlinked,
    );
    assert!(got.is_empty(), "paired declaration rejected: {got:#?}");
}

#[test]
fn def_without_defsig_remains_valid() {
    let got = errors(
        "(def {} inferred (lit {type: (t-prim {} i32)} 1))",
        ProgramOrigin::Unlinked,
    );
    assert!(got.is_empty(), "defsig-less def rejected: {got:#?}");
}

#[test]
fn a_previous_library_def_does_not_back_a_new_units_orphan_defsig() {
    let library =
        parse_str("(def {} existing (lit {type: (t-prim {} i32)} 1))").expect("library fixture");
    let context =
        build_type_env_from_library(&library, ProgramOrigin::Unlinked).expect("library must check");
    let next = parse_str("(defsig {} existing (t-prim {} i32))").expect("next fixture");
    let errors = check_ir_with_context(&context, &next, ProgramOrigin::Unlinked)
        .expect_err("a defsig must be backed in its own check unit")
        .errors;
    assert!(
        errors.iter().any(|error| matches!(
            &error.kind,
            CheckErrorKind::UnboundVariable { identifier } if identifier == "existing"
        )),
        "persistent context incorrectly backed an orphan defsig: {errors:#?}"
    );
}

#[test]
fn trusted_linked_dependency_interface_may_carry_signature_only_rows() {
    let source = "(defsig {} pkg__dep__Api__external (t-fn {} (t-prim {} string)))";
    let got = errors(source, ProgramOrigin::Linked);
    assert!(
        got.is_empty(),
        "trusted linked dependency interfaces are not authored check units: {got:#?}"
    );

    let raw = errors(source, ProgramOrigin::Unlinked);
    assert!(
        raw.iter().any(|error| matches!(
            &error.kind,
            CheckErrorKind::UnboundVariable { identifier } if identifier == "pkg__dep__Api__external"
        )),
        "linked acceptance must not back a later raw signature: {raw:#?}"
    );
    assert!(
        raw.iter()
            .any(|error| matches!(error.kind, CheckErrorKind::ReservedLinkerName)),
        "raw linker-reserved names must still reject: {raw:#?}"
    );
}

#[test]
fn linked_provenance_does_not_exempt_an_ordinary_source_name() {
    let got = errors(
        "(defsig {} authored_missing (t-fn {} (t-prim {} string)))",
        ProgramOrigin::Linked,
    );
    assert!(
        got.iter().any(|error| matches!(
            &error.kind,
            CheckErrorKind::UnboundVariable { identifier } if identifier == "authored_missing"
        )),
        "linked provenance alone must not turn an authored orphan into an interface row: {got:#?}"
    );
}

#[test]
fn cached_linked_context_does_not_authorize_a_new_unlinked_unit() {
    let library = parse_str(
        "(defsig {} pkg__dep__Api__external (t-fn {} (t-prim {} string)))\n\
         (def {} existing (lit {type: (t-prim {} i32)} 1))",
    )
    .expect("linked library fixture");
    let context = build_type_env_from_library(&library, ProgramOrigin::Linked)
        .expect("trusted linked dependency interface checks");
    let encoded = bincode::serialize(&context).expect("library context serializes");
    let cached: TypeEnv = bincode::deserialize(&encoded).expect("library context deserializes");
    let next = parse_str(
        "(defsig {} pkg__dep__Api__external (t-fn {} (t-prim {} string)))\n\
         (defsig {} existing (t-prim {} i32))",
    )
    .expect("new raw unit");
    let errors = check_ir_with_context(&cached, &next, ProgramOrigin::Unlinked)
        .expect_err("cached linked declarations grant neither origin nor same-unit definitions")
        .errors;
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::ReservedLinkerName)),
        "a linked context must not authorize new reserved declarations: {errors:#?}"
    );
    for expected in ["pkg__dep__Api__external", "existing"] {
        assert!(
            errors.iter().any(|error| matches!(
                &error.kind,
                CheckErrorKind::UnboundVariable { identifier } if identifier == expected
            )),
            "a cached declaration must not back the new unit's orphan {expected}: {errors:#?}"
        );
    }
}
