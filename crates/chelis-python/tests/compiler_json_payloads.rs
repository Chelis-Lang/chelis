//! spec/11 §1.1 and spec/10 §§3.2–3.5: typed compiler JSON at real PyO3 edges.
//! The public value remains a Python string; its numeric payload does not.

use chelis_compiler_api::{compiler, schema::*};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyModule};
use serde_json::{Value, json};

#[path = "../src/compiler_json.rs"]
mod compiler_json;

// The included production adapter resolves its existing parent exception.
use chelis_python::ChelisError;

fn native(py: Python<'_>) -> Bound<'_, PyModule> {
    let module = PyModule::new(py, "_compiler_json_controls").unwrap();
    chelis_python::register_module(&module).unwrap();
    module
}

fn decode(value: Bound<'_, pyo3::types::PyAny>) -> Value {
    let text = value
        .extract::<String>()
        .expect("public result is a Python str");
    serde_json::from_str(&text).expect("the string contains the typed JSON result")
}

#[test]
fn native_check_json_preserves_success_and_failure_reports() {
    Python::with_gil(|py| {
        let module = native(py);
        for source in ["x = 9007199254740993i64\n", "x = missing_name\n"] {
            let expected = compiler::check(CheckRequest {
                source_kind: SourceKind::Surf,
                source: source.into(),
            })
            .unwrap();
            let actual = decode(
                module
                    .getattr("check_json")
                    .unwrap()
                    .call1((source,))
                    .unwrap(),
            );
            assert_eq!(actual, serde_json::to_value(expected).unwrap());
            let checked: WireCheckResult = serde_json::from_value(actual.clone()).unwrap();
            assert_eq!(
                checked.errors.is_empty(),
                source.contains("9007199254740993")
            );
            assert_eq!(actual["score"] == json!(1.0), checked.errors.is_empty());
        }
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("source_kind", "unknown").unwrap();
        let error = module
            .getattr("check_json")
            .unwrap()
            .call(("x = 1i64\n",), Some(&kwargs))
            .unwrap_err();
        assert!(error.is_instance_of::<PyValueError>(py));
    });
}

#[test]
fn native_compile_json_preserves_its_typed_result_contract() {
    let source = "def sample(x: tensor[2, f32]) -> tensor[2, f32] = mul(copy(x), x)\n";
    Python::with_gil(|py| {
        let module = native(py);
        let expected = compiler::compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            target: CompileTarget::C,
            entry_name: None,
        })
        .unwrap();
        let actual = decode(
            module
                .getattr("compile_json")
                .unwrap()
                .call1((source,))
                .unwrap(),
        );
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
        let compiled: CompileResult = serde_json::from_value(actual).unwrap();
        assert!(!compiled.files.is_empty());
        assert_eq!(compiled.target, CompileTarget::C);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("target", "unknown").unwrap();
        assert!(
            module
                .getattr("compile_json")
                .unwrap()
                .call((source,), Some(&kwargs))
                .unwrap_err()
                .is_instance_of::<PyValueError>(py)
        );
        assert!(
            module
                .getattr("compile_json")
                .unwrap()
                .call1(("def broken(",))
                .unwrap_err()
                .is_instance_of::<ChelisError>(py)
        );
    });
}

#[test]
fn native_desugar_json_preserves_source_numbers_and_rejects_invalid_source() {
    let source = "wide = 9007199254740993i64\nzero = -0.0f64\n";
    Python::with_gil(|py| {
        let module = native(py);
        let expected = compiler::desugar(DesugarRequest {
            source: source.into(),
        })
        .unwrap();
        let actual = decode(
            module
                .getattr("desugar_json")
                .unwrap()
                .call1((source,))
                .unwrap(),
        );
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
        let desugared: DesugarResult = serde_json::from_value(actual).unwrap();
        assert!(
            serde_json::to_string(&desugared.deep_ast)
                .unwrap()
                .contains("9007199254740993")
        );
        for source in ["def broken(", "x = 1e309f64\n"] {
            assert!(
                module
                    .getattr("desugar_json")
                    .unwrap()
                    .call1((source,))
                    .unwrap_err()
                    .is_instance_of::<ChelisError>(py)
            );
        }
    });
}

#[test]
fn native_eval_json_preserves_exact_execution_values() {
    Python::with_gil(|py| {
        let module = native(py);
        let signature = PyModule::import(py, "inspect")
            .unwrap()
            .getattr("signature")
            .unwrap()
            .call1((module.getattr("eval_json").unwrap(),))
            .unwrap();
        let default = signature
            .getattr("parameters")
            .unwrap()
            .get_item("bindings_json")
            .unwrap()
            .getattr("default")
            .unwrap()
            .extract::<String>()
            .unwrap();
        assert_eq!(default, "{}");
        let source = "wide = 9007199254740993i64\nzero = -0.0f64\n";
        let value = decode(
            module
                .getattr("eval_json")
                .unwrap()
                .call1((source,))
                .unwrap(),
        );
        assert_eq!(
            value["schema_version"],
            json!(EXECUTION_VALUE_SCHEMA_VERSION)
        );
        let roots = value["roots"].as_array().unwrap();
        let root = |name| &roots.iter().find(|root| root["name"] == name).unwrap()["value"];
        assert_eq!(
            root("wide"),
            &json!({"type":"scalar","value":{"dtype":"int64","value":9007199254740993_i64}})
        );
        assert_eq!(
            root("zero"),
            &json!({"type":"scalar","value":{"dtype":"f64","bits":"8000000000000000"}})
        );
        let _: EvalResult = serde_json::from_value(value).unwrap();
        assert!(
            module
                .getattr("eval_json")
                .unwrap()
                .call1(("def broken(",))
                .unwrap_err()
                .is_instance_of::<ChelisError>(py)
        );
    });
}

#[test]
fn native_eval_bindings_reject_invalid_payloads_before_either_route() {
    Python::with_gil(|py| {
        let module = native(py);
        for (language_dtype, data) in [
            (
                "i64",
                json!({"dtype":"int64","values":[9007199254740993_i64]}),
            ),
            ("f64", json!({"dtype":"f64","bits":["8000000000000000"]})),
            ("f64", json!({"dtype":"f64","bits":["7ff8000000000001"]})),
        ] {
            let tensor = json!({"shape":[1],"data":data});
            let source = format!("x = (x : tensor[1, {language_dtype}])\n");
            let actual = decode(
                module
                    .getattr("eval_json")
                    .unwrap()
                    .call1((source, json!({"x":tensor.clone()}).to_string()))
                    .unwrap(),
            );
            let root = actual["roots"]
                .as_array()
                .unwrap()
                .iter()
                .find(|root| root["name"] == "x")
                .unwrap();
            assert_eq!(root["value"], json!({"type":"tensor","value":tensor}));
        }
        let source = "x = (x : tensor[1, i64])\n";
        let invalid = [
            "not json".to_string(),
            json!({"x":{"shape":[1],"data":{"dtype":"int8","values":[128]}}}).to_string(),
            json!({"x":{"shape":[2],"data":{"dtype":"f32","bits":["00000000"]}}}).to_string(),
            json!({"x":{"shape":[-1],"data":{"dtype":"f32","bits":[]}}}).to_string(),
            json!({"x":{"shape":[1],"data":{"dtype":"f32","bits":["7FC00001"]}}}).to_string(),
            json!({"x":{"shape":[1],"data":{"dtype":"f32","values":[0.0]}}}).to_string(),
        ];
        for project_root in [None, Some("/a/compiler-json-control/nonexistent-root")] {
            let kwargs = pyo3::types::PyDict::new(py);
            if let Some(root) = project_root {
                kwargs.set_item("project_root", root).unwrap();
            }
            for wire in &invalid {
                let error = module
                    .getattr("eval_json")
                    .unwrap()
                    .call((source, wire.as_str()), Some(&kwargs))
                    .unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py));
                assert!(
                    error.to_string().contains("invalid bindings json"),
                    "{error}"
                );
            }
            let error = module
                .getattr("eval_json")
                .unwrap()
                .call((source, 7), Some(&kwargs))
                .unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        }
    });
}

#[test]
fn compiler_json_conversion_rejects_invalid_typed_execution_envelopes() {
    Python::with_gil(|py| {
        for version in [
            EXECUTION_VALUE_SCHEMA_VERSION,
            0,
            EXECUTION_VALUE_SCHEMA_VERSION + 1,
        ] {
            let result = EvalResult {
                schema_version: version,
                roots: vec![],
                manifest: RootManifestResult::default(),
                transcript: vec![],
            };
            let converted = compiler_json::EvalJson::new(result).into_pyobject(py);
            if version == EXECUTION_VALUE_SCHEMA_VERSION {
                assert_eq!(
                    decode(converted.unwrap().into_any())["schema_version"],
                    json!(version)
                );
            } else {
                assert!(converted.unwrap_err().is_instance_of::<ChelisError>(py));
            }
        }
    });
}

#[test]
fn native_context_eval_json_uses_the_same_execution_codec() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    std::fs::create_dir(root.join("src")).unwrap();
    std::fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"json-control\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"JsonControl\"\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    )
    .unwrap();
    std::fs::write(
        root.join("reef.lock"),
        "dependencies = []\n\n[package]\nname = \"json-control\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/lib.ch"),
        "module JsonControl.Lib\nexport (keep, keep_float)\ndef keep(x: tensor[1, i64]) -> tensor[1, i64] = neg(neg(x))\ndef keep_float(x: tensor[1, f64]) -> tensor[1, f64] = neg(neg(x))\n",
    )
    .unwrap();
    Python::with_gil(|py| {
        let module = native(py);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs
            .set_item("project_root", root.to_str().unwrap())
            .unwrap();
        for (language_dtype, function, data) in [
            (
                "i64",
                "keep",
                json!({"dtype":"int64","values":[9007199254740993_i64]}),
            ),
            (
                "f64",
                "keep_float",
                json!({"dtype":"f64","bits":["8000000000000000"]}),
            ),
        ] {
            let tensor = json!({"shape":[1],"data":data});
            let bindings = json!({"x":tensor.clone()});
            let source = format!(
                "module JsonControl.Entry\nimport JsonControl.Lib ({function})\nx = (x : tensor[1, {language_dtype}])\ny = {function}(x)\n"
            );
            let value = decode(
                module
                    .getattr("eval_json")
                    .unwrap()
                    .call((source, bindings.to_string()), Some(&kwargs))
                    .unwrap(),
            );
            assert_eq!(
                value["schema_version"],
                json!(EXECUTION_VALUE_SCHEMA_VERSION)
            );
            let roots = value["roots"].as_array().unwrap();
            let root = roots.iter().find(|root| root["name"] == "y").unwrap();
            assert_eq!(root["value"], json!({"type":"tensor","value":tensor}));
            let _: EvalResult = serde_json::from_value(value).unwrap();
        }
        assert!(
            module
                .getattr("eval_json")
                .unwrap()
                .call(("def broken(",), Some(&kwargs))
                .unwrap_err()
                .is_instance_of::<ChelisError>(py)
        );
        kwargs.set_item("source_kind", "deep").unwrap();
        let error = module
            .getattr("eval_json")
            .unwrap()
            .call(("(int {} 1)",), Some(&kwargs))
            .unwrap_err();
        assert!(error.is_instance_of::<PyValueError>(py));
        assert!(error.to_string().contains("Surf source only"));
        kwargs.set_item("source_kind", "surf").unwrap();
        kwargs.set_item("project_root", " ").unwrap();
        let error = module
            .getattr("eval_json")
            .unwrap()
            .call(("x = 1i64",), Some(&kwargs))
            .unwrap_err();
        assert!(error.is_instance_of::<ChelisError>(py));
        assert!(error.to_string().contains("project_root= is empty"));
    });
}

#[test]
fn compiler_json_construction_accepts_only_its_exact_result_type() {
    // These calls compile and execute each production constructor/converter.
    // The same acceptance runner also compiles the separate String/Value/
    // wrong-root/private-field controls against this production module.
    Python::with_gil(|py| {
        let check = compiler::check(CheckRequest {
            source_kind: SourceKind::Surf,
            source: "x = 1i64".into(),
        })
        .unwrap();
        assert_eq!(
            decode(
                compiler_json::CheckJson::new(check)
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
            )["score"],
            json!(1.0)
        );
        let compile = CompileResult {
            target: CompileTarget::C,
            entry_name: "fixture".into(),
            files: vec![],
            compile_flags: vec![],
            link_flags: vec![],
            peak_device_bytes_estimate: Some(
                numbers::NonnegativeCount::new(9007199254740993).unwrap(),
            ),
            manifest: RootManifestResult::default(),
        };
        assert_eq!(
            decode(
                compiler_json::CompileJson::new(compile)
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
            )["peak_device_bytes_estimate"],
            json!(9007199254740993_i64)
        );
        let desugar = compiler::desugar(DesugarRequest {
            source: "x = 9007199254740993i64".into(),
        })
        .unwrap();
        assert!(
            decode(
                compiler_json::DesugarJson::new(desugar)
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
            )
            .to_string()
            .contains("9007199254740993")
        );
        let eval = EvalResult {
            schema_version: EXECUTION_VALUE_SCHEMA_VERSION,
            roots: vec![],
            manifest: RootManifestResult::default(),
            transcript: vec![],
        };
        assert_eq!(
            decode(
                compiler_json::EvalJson::new(eval)
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
            )["roots"],
            json!([])
        );
        assert!(
            compiler_json::EvalBindingsJson::empty()
                .into_bindings()
                .is_empty()
        );
        let text = pyo3::types::PyString::new(py, "{}");
        assert!(
            text.extract::<compiler_json::EvalBindingsJson>()
                .unwrap()
                .into_bindings()
                .is_empty()
        );
        assert!(
            pyo3::types::PyString::new(py, "not json")
                .extract::<compiler_json::EvalBindingsJson>()
                .is_err()
        );
    });
}
