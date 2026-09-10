//! Direct Cargo/nextest native tests must use the PyO3-selected interpreter.

mod support;

use pyo3::prelude::*;
use std::process::Command;

#[test]
fn selected_python_packages_load_with_clean_ambient_environment() {
    const WORKER: &str = "CHELIS_NATIVE_ENV_TEST_WORKER";
    if std::env::var_os(WORKER).is_none() {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "selected_python_packages_load_with_clean_ambient_environment",
                "--nocapture",
            ])
            .env(WORKER, "1")
            .env_remove("PYTHONPATH")
            .env_remove("PYTHONHOME")
            .env_remove("PYTHONUSERBASE")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let environment = support::initialize();
    Python::with_gil(|py| {
        let globals = pyo3::types::PyDict::new(py);
        support::run_case(py, c"pass", &globals);
        let numpy = py.import("numpy").expect("declared NumPy prerequisite");
        let file: String = numpy.getattr("__file__").unwrap().extract().unwrap();
        assert!(
            environment
                .package_paths
                .iter()
                .any(|path| std::path::Path::new(&file).starts_with(path)),
            "NumPy did not come from the selected interpreter: {file}"
        );
    });
}

#[test]
fn missing_selected_interpreter_is_a_prerequisite_error_without_fallback() {
    let missing = tempfile::tempdir().unwrap().path().join("missing-python");
    assert!(support::query_environment(&missing).is_err());
}
