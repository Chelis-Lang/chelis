//! Actual registered CompilerJson exposures, built by the private verifier.
use pyo3::{
    prelude::*,
    types::{PyCFunction, PyModule},
};
use serde_json::json;
use sha2::{Digest, Sha256};

#[path = "../../../tests/support/pyo3_registration.rs"]
mod pyo3_registration;

fn main() {
    const SOURCE: &str = include_str!("../src/lib.rs");
    let registrations = pyo3_registration::registrations(SOURCE)
        .expect("actual compiled PyO3 registration provenance");
    let selected = ["check_json", "compile_json", "desugar_json", "eval_json"];
    let registrations: Vec<_> = registrations
        .into_iter()
        .filter(|row| row.owner.is_none() && selected.contains(&row.python_name.as_str()))
        .collect();
    assert_eq!(registrations.len(), selected.len());
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_compiler_json_provenance").unwrap();
        chelis_python::register_module(&module).unwrap();
        for name in selected {
            let callable = module.getattr(name).unwrap();
            assert!(callable.is_instance_of::<PyCFunction>());
            assert!(callable.is_callable());
            assert_eq!(
                registrations
                    .iter()
                    .filter(|row| row.python_name == name && row.kind == "function")
                    .count(),
                1
            );
        }
    });
    println!(
        "{}",
        json!({
            "source_path":"crates/chelis-python/src/lib.rs",
            "source_sha256": format!("{:x}",Sha256::digest(SOURCE.as_bytes())),
            "registrations": registrations,
        })
    );
}
