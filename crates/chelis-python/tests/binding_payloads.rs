//! Executed Python payload boundaries for the independent nonnumeric C6 rows.

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[test]
fn source_json_results_have_exact_nonnumeric_shapes() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "binding_payloads").unwrap();
        chelis_python::register_module(&module).unwrap();
        let validate: String = module
            .getattr("validate_json")
            .unwrap()
            .call1(("x = 1",))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&validate).unwrap(),
            serde_json::json!({"mode": "surf", "valid": true})
        );
        let decompile: String = module
            .getattr("decompile_json")
            .unwrap()
            .call1(("(def {} x (lit {} 42))",))
            .unwrap()
            .extract()
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&decompile).unwrap();
        assert_eq!(result.as_object().unwrap().len(), 1);
        assert!(result["surf_text"].as_str().unwrap().contains("42"));
    });
}

#[test]
fn source_json_errors_raise_string_exceptions_without_result_payloads() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "binding_payloads").unwrap();
        chelis_python::register_module(&module).unwrap();
        for (name, input) in [("validate_json", "("), ("decompile_json", "(")] {
            let error = module.getattr(name).unwrap().call1((input,)).unwrap_err();
            assert!(
                error
                    .matches(py, module.getattr("ChelisError").unwrap())
                    .unwrap()
            );
            let args: (String,) = error.value(py).getattr("args").unwrap().extract().unwrap();
            assert!(!args.0.is_empty());
        }
        for name in [
            "validate_json",
            "decompile_json",
            "load",
            "compile_and_load",
        ] {
            assert!(
                module
                    .getattr(name)
                    .unwrap()
                    .call1((0.5,))
                    .unwrap_err()
                    .is_instance_of::<pyo3::exceptions::PyTypeError>(py)
            );
        }
    });
}

#[test]
fn constructors_fail_with_string_errors_instead_of_fabricated_handles() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "binding_payloads").unwrap();
        chelis_python::register_module(&module).unwrap();
        let absent = tempfile::tempdir().unwrap().path().join("absent.ch");
        for name in ["load", "compile_and_load"] {
            let error = module
                .getattr(name)
                .unwrap()
                .call1((absent.to_str().unwrap(),))
                .unwrap_err();
            let args: (String,) = error.value(py).getattr("args").unwrap().extract().unwrap();
            assert!(!args.0.is_empty());
        }
    });
}

#[test]
fn model_constructors_and_getters_expose_handles_names_and_paths() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("model.ch");
    std::fs::write(
        &source,
        "def main(x: tensor[1, f32]) -> tensor[1, f32] = copy(x)\n",
    )
    .unwrap();
    Python::with_gil(|py| {
        let module = PyModule::new(py, "binding_payloads").unwrap();
        chelis_python::register_module(&module).unwrap();
        let model = module
            .getattr("compile_and_load")
            .unwrap()
            .call1((source.to_str().unwrap(),))
            .unwrap();
        assert!(
            model
                .is_instance(&module.getattr("CompiledModel").unwrap())
                .unwrap()
        );
        let path: String = model.getattr("path").unwrap().extract().unwrap();
        assert!(std::path::Path::new(&path).is_file());
        let loaded = module.getattr("load").unwrap().call1((&path,)).unwrap();
        for handle in [&model, &loaded] {
            assert_eq!(
                handle
                    .getattr("target")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "c"
            );
            assert_eq!(
                handle
                    .getattr("input_names")
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["x"]
            );
            let names: Vec<String> = handle.getattr("output_names").unwrap().extract().unwrap();
            assert_eq!(names.len(), 1);
            assert!(!names[0].is_empty());
            assert!(
                handle.getattr("loaded").is_err(),
                "private Rust ownership is not a Python payload"
            );
            for getter in ["path", "target", "input_names", "output_names"] {
                assert!(
                    handle.setattr(getter, 0.5).is_err(),
                    "{getter} must be read-only"
                );
            }
        }
    });
}
