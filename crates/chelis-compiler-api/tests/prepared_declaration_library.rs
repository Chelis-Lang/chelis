//! Repeated property probes compose with one checked and lowered library.
use std::collections::BTreeMap;

use chelis_compiler_api::{build_stdlib_context, compiler};

#[test]
fn prepared_library_probes_agree_with_monolithic_evaluation() {
    let library = chelis_surf::parser::parse_str("def twice(x: i64) -> i64 = x + x\n").unwrap();
    let context = build_stdlib_context(&library).unwrap();
    for value in [0, 7, -3, 9_007_199_254_740_993i64] {
        let probe =
            chelis_surf::parser::parse_str(&format!("answer = twice({value}i64)\n")).unwrap();
        let selected = ["answer".to_string()];
        let result = compiler::eval_decls_selected_with_library(
            &context,
            &probe,
            BTreeMap::new(),
            &selected,
        )
        .unwrap();
        let mut whole = library.clone();
        whole.extend(probe);
        let baseline = compiler::eval_decls_selected(&whole, BTreeMap::new(), &selected).unwrap();
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            serde_json::to_value(baseline).unwrap()
        );
    }
}

#[test]
fn prepared_library_probes_reject_wrong_arguments_and_recover() {
    let library = chelis_surf::parser::parse_str("def twice(x: i64) -> i64 = x + x\n").unwrap();
    let context = build_stdlib_context(&library).unwrap();
    let selected = ["answer".to_string()];
    let bad = chelis_surf::parser::parse_str("answer = twice(true)\n").unwrap();
    assert!(
        compiler::eval_decls_selected_with_library(&context, &bad, BTreeMap::new(), &selected)
            .is_err()
    );
    let good = chelis_surf::parser::parse_str("answer = twice(7i64)\n").unwrap();
    let result =
        compiler::eval_decls_selected_with_library(&context, &good, BTreeMap::new(), &selected)
            .unwrap();
    assert_eq!(result.roots[0].display.as_deref(), Some("14"));
}

#[test]
fn prepared_library_keeps_globals_strings_and_tensor_bindings() {
    let library = chelis_surf::parser::parse_str(
        "offset = 3i64\ndef label(x: string) -> string = x\ndef shift(x: i64) -> i64 = x + offset\n",
    ).unwrap();
    let context = build_stdlib_context(&library).unwrap();
    for source in [
        "answer = (label(\"sample\"), shift(7i64))\n",
        "def answer(input: tensor[2, f32]) -> tensor[2, f32] = add(input, input)\n",
    ] {
        let probe = chelis_surf::parser::parse_str(source).unwrap();
        let selected = ["answer".to_string()];
        let bindings = if source.starts_with("def answer") {
            BTreeMap::from([(
                "input".to_string(),
                chelis_compiler_api::schema::TensorValue {
                    shape: vec![2],
                    data: chelis_types::finalize_tensor(
                        "probe",
                        chelis_types::types::Prim::F32,
                        chelis_types::RawTensor::Float(vec![1.25, -2.5]),
                    )
                    .unwrap(),
                },
            )])
        } else {
            BTreeMap::new()
        };
        let actual = compiler::eval_decls_selected_with_library(
            &context,
            &probe,
            bindings.clone(),
            &selected,
        )
        .unwrap();
        let mut whole = library.clone();
        whole.extend(probe);
        let baseline = compiler::eval_decls_selected(&whole, bindings, &selected).unwrap();
        // Composition manifests describe the new declarations; the monolith
        // additionally lists library globals. DAG node IDs also depend on
        // that composition. Selected names and exact values must agree.
        assert_eq!(actual.roots.len(), baseline.roots.len());
        for (actual, baseline) in actual.roots.into_iter().zip(baseline.roots) {
            assert_eq!(actual.name, baseline.name);
            assert_eq!(actual.display, baseline.display);
            assert_eq!(
                serde_json::to_value(actual.value).unwrap(),
                serde_json::to_value(baseline.value).unwrap()
            );
        }
    }
}
