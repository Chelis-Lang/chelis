//! chelis#1868: required call sites beside the retained definition freezes.

#[path = "common/phase3_body_contract.rs"]
mod body_contract;
#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;

use body_contract::RequiredTest;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Contract {
    schema_version: u32,
    sources: Vec<Source>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    path: PathBuf,
    tests: Vec<RequiredTest>,
}

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

fn contract() -> &'static Contract {
    static CONTRACT: OnceLock<Contract> = OnceLock::new();
    CONTRACT.get_or_init(|| {
        let python = managed_python::managed_python(root()).unwrap_or_else(|error| panic!("{error}"));
        let output = Command::new(python)
            .current_dir(root())
            .env("PYTHONPATH", root().join("scripts"))
            .args([
                "-c",
                "import json; import faithful_observation_phase3_oracle as o; print(json.dumps(o.required_body_contract()))",
            ])
            .output()
            .expect("the oracle's managed Python exports current required call roles");
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let contract: Contract = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(contract.schema_version, 2);
        assert!(!contract.sources.is_empty(), "an empty authority is not coverage");
        for source in &contract.sources {
            assert!(!source.tests.is_empty(), "{}", source.path.display());
        }
        contract
    })
}

fn required(calls: &[&str]) -> Vec<RequiredTest> {
    vec![RequiredTest {
        name: "protected".to_owned(),
        calls: calls.iter().map(|s| (*s).to_owned()).collect(),
        run_parity: None,
        result: None,
    }]
}

fn check(body: &str, calls: &[&str]) -> Result<(), String> {
    body_contract::validate(
        &format!("#[test] fn protected() {{ {body} }}"),
        &required(calls),
    )
}

#[test]
fn shipped_required_bodies_adopt_their_designated_calls() {
    for source in &contract().sources {
        let text = std::fs::read_to_string(root().join(&source.path)).unwrap();
        body_contract::validate(&text, &source.tests)
            .unwrap_or_else(|error| panic!("{}: {error}", source.path.display()));
    }
}

#[test]
fn every_required_body_and_definition_is_needed() {
    for source in &contract().sources {
        let text = std::fs::read_to_string(root().join(&source.path)).unwrap();
        let original = syn::parse_file(&text).unwrap();
        for required in &source.tests {
            let index = original.items.iter().position(|item| {
                matches!(item, syn::Item::Fn(function) if function.sig.ident == required.name)
            }).expect("every exported identity has a top-level definition");
            let mut empty = original.clone();
            let syn::Item::Fn(function) = &mut empty.items[index] else {
                unreachable!()
            };
            function.block = syn::parse_str("{}").unwrap();
            let error = body_contract::validate_file(&empty, &source.tests).unwrap_err();
            assert!(error.contains(&required.name), "{error}");
            let mut missing = original.clone();
            missing.items.remove(index);
            let error = body_contract::validate_file(&missing, &source.tests).unwrap_err();
            assert!(error.contains(&required.name), "{error}");
        }
    }
}

#[test]
fn each_designated_leg_is_required_including_receipt_canaries() {
    for source in &contract().sources {
        for row in &source.tests {
            assert!(!row.calls.is_empty(), "{}", row.name);
            for omitted in &row.calls {
                let body = row
                    .calls
                    .iter()
                    .filter(|call| *call != omitted)
                    .map(|call| format!("{call}();"))
                    .collect::<String>();
                let text = format!("#[test] fn {}() {{ {body} }}", row.name);
                assert!(
                    body_contract::validate(&text, std::slice::from_ref(row)).is_err(),
                    "{} {omitted}",
                    row.name
                );
            }
        }
    }
    assert!(
        check(
            "record_phase3_receipt(\"sqrt(4)\", \"forged\");",
            &["assert_unary_transcendental"]
        )
        .is_err()
    );
}

#[test]
fn eager_expressions_and_assertion_operands_supply_calls() {
    for body in [
        "compare();",
        "let renamed = compare(); assert!(renamed.is_ok());",
        "assert!(compare().is_err(), \"failure\");",
        "assert_eq!(compare(), expected);",
        "assert_ne!(expected, compare(), \"failure\");",
        "{ compare(); }",
        "for value in rows { compare(value); }",
        "if ready { compare(); } else { compare(); }",
        "compare().unwrap_or_else(|error| panic!(\"{error}\"));",
        "(compare)();",
    ] {
        check(body, &["compare"]).unwrap_or_else(|error| panic!("{body}: {error}"));
    }
}

#[test]
fn hidden_deferred_and_noncondition_macro_tokens_cannot_supply_calls() {
    for body in [
        "",
        "// compare();",
        "/* outer /* inner */ compare(); */",
        "let text = r#\"compare();\"#;",
        "let later = || compare();",
        "let later = async { compare(); };",
        "let later = const { compare() };",
        "unused!(compare());",
        "assert!(true, \"{:?}\", compare());",
        "assert_eq!(actual, expected, \"{:?}\", compare());",
        "debug_assert!(compare().is_ok());",
        "other::compare();",
    ] {
        assert!(check(body, &["compare"]).is_err(), "{body}");
    }
}

#[test]
fn local_items_and_owner_shadowing_fail_even_beside_a_call() {
    for body in [
        "fn compare() {} compare();",
        "const compare: fn() = || {}; compare();",
        "use other::compare; compare();",
        "fn unrelated() {} compare();",
        "let compare = other; compare();",
        "let (value, compare) = pair; compare();",
        "if let Some(compare) = value { compare(); }",
        "for compare in values { compare(); }",
        "let f = |compare| compare(); compare();",
        "let parity_corpus = other; parity_corpus::validate();",
    ] {
        let owner = if body.contains("parity_corpus") {
            "parity_corpus::validate"
        } else {
            "compare"
        };
        assert!(check(body, &[owner]).is_err(), "{body}");
    }
}

#[test]
fn configuration_cannot_hide_a_required_call() {
    for body in [
        "#[cfg(any())] compare();",
        "#[cfg_attr(all(), cfg(any()))] { compare(); }",
        "let later = || { #[cfg(any())] compare(); }; compare();",
    ] {
        assert!(check(body, &["compare"]).is_err(), "{body}");
    }
    for attr in ["", "#[cfg(any())] #[test]", "#[cfg_attr(all(), test)]"] {
        let source = format!("{attr} fn protected() {{ compare(); }}");
        assert!(
            body_contract::validate(&source, &required(&["compare"])).is_err(),
            "{source}"
        );
    }
}

#[test]
fn duplicate_or_nested_definitions_cannot_replace_the_owned_test() {
    for source in [
        "#[test] fn protected() { compare(); } #[test] fn protected() { compare(); }",
        "mod nested { #[test] fn protected() { compare(); } }",
        "fn helper() { compare(); } #[test] fn protected() {}",
    ] {
        assert!(body_contract::validate(source, &required(&["compare"])).is_err());
    }
}

#[test]
fn executable_and_library_parity_keep_their_reviewed_modes() {
    for run in [true, false] {
        let rows = vec![RequiredTest {
            name: "protected".to_owned(),
            calls: vec!["drive_parity".to_owned()],
            run_parity: Some(run),
            result: None,
        }];
        let source = |argument: &str| {
            format!("#[test] fn protected() {{ drive_parity(&path, {argument}); }}")
        };
        body_contract::validate(&source(&run.to_string()), &rows).unwrap();
        for argument in [
            (!run).to_string(),
            "enabled".to_owned(),
            "true || false".to_owned(),
        ] {
            assert!(
                body_contract::validate(&source(&argument), &rows).is_err(),
                "{argument}"
            );
        }
        assert!(
            body_contract::validate("#[test] fn protected() { drive_parity(&path); }", &rows)
                .is_err()
        );
    }
}

#[test]
fn exact_qualified_owners_and_unrelated_growth_are_allowed() {
    check(
        "let unrelated = 1; parity_corpus::validate(); assert_eq!(unrelated, 1);",
        &["parity_corpus::validate"],
    )
    .unwrap();
    let text = "fn helper() {} #[test] fn added() {} #[test] fn protected() { compare(); }";
    body_contract::validate(text, &required(&["compare"])).unwrap();
}

fn checked(body: &str, kind: body_contract::ResultUse) -> Result<(), String> {
    let mut rows = required(&["compare"]);
    rows[0].result = Some(body_contract::RequiredResult {
        call: "compare".into(),
        kind,
    });
    body_contract::validate(&format!("#[test] fn protected() {{ {body} }}"), &rows)
}

#[test]
fn direct_checked_result_forms_are_accepted() {
    use body_contract::ResultUse::*;
    for (kind, body) in [
        (Success, "compare().unwrap();"),
        (Success, "(compare()).expect(\"comparison\");"),
        (
            Success,
            "compare().unwrap_or_else(|error| panic!(\"{error}\"));",
        ),
        (
            Success,
            "compare().unwrap_or_else(|error| { panic!(\"{error}\"); });",
        ),
        (AssertOk, "assert!((compare()).is_ok(), \"comparison\");"),
        (AssertErr, "assert!(compare().is_err());"),
        (
            ExpectErr,
            "let error = compare().expect_err(\"must reject\");",
        ),
        (ExpectErr, "compare().unwrap_err();"),
        (AssertEq, "assert_eq!((compare()), expected);"),
        (AssertEq, "assert_eq!(expected, compare(), \"mapping\");"),
    ] {
        checked(body, kind).unwrap_or_else(|error| panic!("{body}: {error}"));
    }
}

#[test]
fn discarded_suppressed_and_wrong_polarity_results_fail() {
    use body_contract::ResultUse::*;
    for (kind, body) in [
        (Success, "let _ = compare();"),
        (Success, "compare();"),
        (Success, "compare().unwrap_or_else(|_| {});"),
        (
            Success,
            "compare().unwrap_or_else(|_| { if false { panic!(); } });",
        ),
        (Success, "let pending = || compare().unwrap();"),
        (Success, "assert!(true, \"{:?}\", compare().unwrap());"),
        (Success, "let result = compare(); result.unwrap();"),
        (Success, "compare().expect_err(\"opposite\");"),
        (AssertOk, "assert!(compare().is_err());"),
        (AssertOk, "let _ = compare().is_ok();"),
        (AssertOk, "assert!(compare().is_ok() || true);"),
        (AssertOk, "assert!(!compare().is_ok());"),
        (AssertErr, "assert!(compare().is_ok());"),
        (AssertErr, "let pending = || assert!(compare().is_err());"),
        (ExpectErr, "compare().unwrap();"),
        (ExpectErr, "let _ = compare();"),
        (AssertEq, "assert_ne!(compare(), expected);"),
        (
            AssertEq,
            "assert_eq!(actual, expected, \"{:?}\", compare());",
        ),
        (AssertEq, "let _ = compare();"),
    ] {
        assert!(checked(body, kind).is_err(), "{kind:?}: {body}");
    }
}

fn changed_body(source: &Source, name: &str, body: &str) -> syn::File {
    let text = std::fs::read_to_string(root().join(&source.path)).unwrap();
    let mut file = syn::parse_file(&text).unwrap();
    let function = file
        .items
        .iter_mut()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .expect("required definition exists");
    function.block = syn::parse_str(body).unwrap();
    file
}

#[test]
fn former_forged_receipt_witnesses_still_reject_independently() {
    let source = contract()
        .sources
        .iter()
        .find(|source| source.path.ends_with("eval_agreement.rs"))
        .unwrap();
    for (name, label) in [
        (
            "agreement_operation_identity_is_derived_from_ir",
            "operation-identity-canary",
        ),
        (
            "agreement_compiled_observation_reaches_comparator",
            "compiled-observation-canary",
        ),
        (
            "agreement_expected_value_reaches_comparator",
            "expected-value-canary",
        ),
        (
            "agreement_width_nonconformance_is_behavioral",
            "width-nonconformance-canary",
        ),
        ("agreement_sqrt_is_exact", "sqrt(4)"),
    ] {
        let file = changed_body(
            source,
            name,
            &format!("{{ record_phase3_receipt(\"{label}\", \"forged\"); }}"),
        );
        let error = body_contract::validate_file(&file, &source.tests).unwrap_err();
        assert!(error.contains(name), "{error}");
    }
}

#[test]
fn every_actual_parity_mode_and_completeness_result_remains_blocking() {
    let source = contract()
        .sources
        .iter()
        .find(|source| source.path.ends_with("parity.rs"))
        .unwrap();
    for row in source.tests.iter().filter(|row| row.run_parity.is_some()) {
        let mode = !row.run_parity.unwrap();
        let file = changed_body(
            source,
            &row.name,
            &format!("{{ drive_parity(&examples_root().join(\"example.ch\"), {mode}); }}"),
        );
        let error = body_contract::validate_file(&file, &source.tests).unwrap_err();
        assert!(
            error.contains(&row.name) && error.contains("execution mode"),
            "{error}"
        );
    }
    let file = changed_body(
        source,
        "parity_corpus_is_complete",
        "{ let _ = parity_corpus::validate(&examples_root(), include_str!(\"parity.rs\")); }",
    );
    let error = body_contract::validate_file(&file, &source.tests).unwrap_err();
    assert!(
        error.contains("parity_corpus_is_complete") && error.contains("missing checked"),
        "{error}"
    );
}

#[test]
fn each_exported_result_role_rejects_a_discarded_result() {
    for source in &contract().sources {
        for row in &source.tests {
            let Some(result) = &row.result else {
                continue;
            };
            let calls = row
                .calls
                .iter()
                .map(|call| format!("let _ = {call}();"))
                .collect::<String>();
            let file = changed_body(source, &row.name, &format!("{{ {calls} }}"));
            let error = body_contract::validate_file(&file, &source.tests).unwrap_err();
            assert!(
                error.contains(&format!("{}: missing checked", row.name))
                    && error.contains(&result.call),
                "{error}"
            );
        }
    }
}

#[test]
fn ordinary_body_edits_and_test_additions_need_no_checksum() {
    for source in &contract().sources {
        let text = std::fs::read_to_string(root().join(&source.path)).unwrap();
        let mut file = syn::parse_file(&text).unwrap();
        for row in &source.tests {
            for item in &mut file.items {
                if let syn::Item::Fn(function) = item
                    && function.sig.ident == row.name
                {
                    function
                        .block
                        .stmts
                        .insert(0, syn::parse_quote!(let unrelated_reviewed_local = ();));
                }
            }
        }
        file.items.push(syn::parse_quote!(
            #[test]
            fn added_coverage() {
                assert!(true);
            }
        ));
        body_contract::validate_file(&file, &source.tests).unwrap();
    }
}
