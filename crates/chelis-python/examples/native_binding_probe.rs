//! Executed only from the current-source native binding verifier.
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[path = "../../../tests/support/pyo3_native_registration.rs"]
mod native_registration;

fn main() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_native_binding_provenance").unwrap();
        chelis_python::register_module(&module).unwrap();
        let report =
            native_registration::probe(&module).expect("actual compiled native registration");
        println!("{}", serde_json::to_string(&report).unwrap());
    });
}
