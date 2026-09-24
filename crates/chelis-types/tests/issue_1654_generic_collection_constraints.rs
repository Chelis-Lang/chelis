//! chelis#1654: collection admission is explicit at authored boundaries and
//! durable on already-checked function values.

use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{
    TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context,
    check_typed_program,
};

fn fixture(name: &str) -> Vec<chelis_deep::Expr> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/core/tests/issue_1654_generic_collection_constraints")
        .join(format!("{name}.deep"));
    let source = std::fs::read_to_string(path).expect("core fixture");
    chelis_deep::parser::parse_str(&source).expect("valid Deep fixture")
}

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut messages = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>();
    messages.sort();
    messages
}

fn whole_program(source: &str) -> [(&'static str, Result<(), Vec<CheckError>>); 2] {
    let desugared = fixture(source);
    let expanded = fixture(&format!("{source}_expanded"));
    [
        (
            "typed",
            check_typed_program(
                &desugared,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            )
            .map(|_| ())
            .map_err(|report| report.errors),
        ),
        (
            "IR",
            check_ir_program(
                &expanded,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            )
            .map(|_| ())
            .map_err(|report| report.errors),
        ),
    ]
}

fn serialized_context(source: &str) -> Result<(), Vec<CheckError>> {
    use chelis_deep::{DeepTag, ExprCarrier};
    let declarations = fixture(source);
    let boundary = declarations
        .iter()
        .take_while(|expr| match expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Defsig, _, _) => true,
            ExprCarrier::DecodedNode(DeepTag::Def, _, children) => children
                .get(1)
                .is_some_and(|body| body.tag() == Some(DeepTag::Fn)),
            _ => false,
        })
        .count();
    let (library, caller) = declarations.split_at(boundary);
    let context = build_type_env_from_library(
        library,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .map_err(|report| report.errors)?;
    let encoded = bincode::serialize(&context).expect("checked context serializes");
    let restored: TypeEnv = bincode::deserialize(&encoded).expect("checked context restores");
    check_ir_with_context(
        &restored,
        caller,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .map(|_| ())
    .map_err(|report| report.errors)
}

fn accepts(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        if let Err(errors) = result {
            panic!(
                "{label} [{entry}] over-rejected:\n{}\nsource:\n{source}",
                rendered(&errors).join("\n")
            );
        }
    }
}

fn rejects_type_mismatch(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
            "{label} [{entry}] expected TypeMismatch:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

fn rejects_type_or_dimension(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        let errors = match result {
            Err(errors) => errors,
            Ok(()) => panic!("{label} [{entry}] unexpectedly accepted"),
        };
        assert!(
            errors.iter().any(|error| {
                matches!(
                    error.kind,
                    CheckErrorKind::TypeMismatch | CheckErrorKind::DimensionMismatch
                )
            }),
            "{label} [{entry}] expected a type/dimension rejection:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

fn rejects_once(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        let errors = result.unwrap_err();
        assert_eq!(
            errors.len(),
            1,
            "{label} [{entry}] expected one isolated rejection:\n{}",
            rendered(&errors).join("\n")
        );
        assert!(
            matches!(
                errors[0].kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::DimensionMismatch
            ),
            "{label} [{entry}] did not retain the expected rejection:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

#[test]
fn authored_generic_wrappers_must_declare_collection_constructors() {
    for source in [
        "case_000", "case_001", "case_002", "case_003", "case_004", "case_005", "case_006",
        "case_007", "case_008",
    ] {
        rejects_type_mismatch("insufficient collection contract", source);
    }
}

#[test]
fn explicit_list_and_dict_contracts_are_sufficient() {
    accepts("explicit List length", "case_009");
    accepts("explicit Dict length", "case_010");
    accepts("explicit collection relations", "case_011");
    rejects_type_mismatch("checked user function type survives aliasing", "case_012");
}

#[test]
fn local_first_application_remains_monomorphic() {
    accepts("local list binding", "case_013");
    rejects_type_mismatch("local scalar binding", "case_014");
}

#[test]
fn builtin_contracts_survive_alias_return_aggregate_and_higher_order_passage() {
    for source in [
        "case_015", "case_016", "case_017", "case_018", "case_019", "case_020",
    ] {
        rejects_type_mismatch("transported len contract", source);
    }
    rejects_type_mismatch("transported len result", "case_021");
    for source in [
        "case_022", "case_023", "case_024", "case_025", "case_026", "case_027", "case_028",
        "case_029",
    ] {
        accepts("valid transported len contract", source);
    }
}

#[test]
fn each_collection_relation_survives_indirect_calls() {
    for source in [
        "case_030", "case_031", "case_032", "case_033", "case_034", "case_035",
    ] {
        rejects_type_mismatch("transported relation", source);
    }
    for source in ["case_036", "case_037", "case_038", "case_039"] {
        accepts("valid transported relation", source);
    }
    for source in ["case_040", "case_041", "case_042"] {
        rejects_type_mismatch("transported result relation", source);
    }
}

#[test]
fn tensor_concat_transport_preserves_axis_and_exact_result_equations() {
    rejects_type_or_dimension("direct tensor concat result", "case_043");
    for (route, source) in [
        ("alias", "case_044".to_string()),
        ("higher-order return", "case_045".to_string()),
        ("higher-order passage", "case_046".to_string()),
        ("aggregate", "case_047".to_string()),
        ("recursive return", "case_048".to_string()),
        ("monomorphic annotated alias", "case_049".to_string()),
        ("nested monomorphic function value", "case_050".to_string()),
    ] {
        rejects_type_or_dimension(&format!("{route} tensor concat result"), &source);
    }

    for (route, source) in [
        ("direct", "case_051".to_string()),
        ("alias", "case_052".to_string()),
        ("monomorphic annotated alias", "case_053".to_string()),
    ] {
        rejects_type_mismatch(&format!("{route} tensor concat axis"), &source);
    }

    for source in [
        "case_054", "case_055", "case_056", "case_057", "case_058", "case_059", "case_060",
    ] {
        accepts("valid transported tensor concat", source);
    }
}

#[test]
fn tensor_concat_call_evidence_is_owned_and_cleaned_up_per_application() {
    accepts("one alias supports independent axis equations", "case_061");
    accepts(
        "independently specialized aliases do not share evidence",
        "case_062",
    );
    rejects_once(
        "a failed call does not poison the following valid call",
        "case_063",
    );
}

#[test]
fn checked_contracts_survive_serialized_library_contexts() {
    for source in ["case_064", "case_065"] {
        let errors = serialized_context(source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
            "{}",
            rendered(&errors).join("\n")
        );
    }
    for source in ["case_066", "case_067"] {
        serialized_context(source)
            .unwrap_or_else(|errors| panic!("{}", rendered(&errors).join("\n")));
    }

    let invalid_tensor_concat = "case_068";
    let errors = serialized_context(invalid_tensor_concat).unwrap_err();
    assert!(
        errors.iter().any(|error| {
            matches!(
                error.kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::DimensionMismatch
            )
        }),
        "{}",
        rendered(&errors).join("\n")
    );
    serialized_context("case_069")
        .unwrap_or_else(|errors| panic!("{}", rendered(&errors).join("\n")));

    let invalid_monomorphic_tensor_concat = "case_070";
    let errors = serialized_context(invalid_monomorphic_tensor_concat).unwrap_err();
    assert!(
        errors.iter().any(|error| {
            matches!(
                error.kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::DimensionMismatch
            )
        }),
        "{}",
        rendered(&errors).join("\n")
    );
    serialized_context("case_071")
        .unwrap_or_else(|errors| panic!("{}", rendered(&errors).join("\n")));
}

#[test]
fn recursive_indirect_calls_keep_the_checked_contract() {
    accepts("recursive indirect list call", "case_072");
    rejects_type_mismatch("recursive indirect scalar call", "case_073");
}

#[test]
fn same_scc_returns_propagate_checked_collection_contracts() {
    let mutual_scalar = "case_074";
    let mutual_list = "case_075";
    rejects_type_mismatch(
        "mutual recursion retains returned len contract",
        mutual_scalar,
    );
    accepts("mutual recursion accepts matching List call", mutual_list);
    let errors = serialized_context(mutual_scalar).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| { matches!(error.kind, CheckErrorKind::TypeMismatch) }),
        "{}",
        rendered(&errors).join("\n")
    );
    serialized_context(mutual_list)
        .unwrap_or_else(|errors| panic!("{}", rendered(&errors).join("\n")));

    rejects_type_mismatch(
        "four-member recursion propagates returned len contract",
        "case_076",
    );
    accepts(
        "four-member recursion accepts matching List call",
        "case_077",
    );
}
