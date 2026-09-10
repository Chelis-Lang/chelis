//! Test-only embedded Python setup, shared by directly discovered native suites.
//! No installation, skip, system-interpreter fallback, or process-global env writes.

use pyo3::{prelude::*, types::PyList};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

#[derive(serde::Deserialize)]
pub struct Environment {
    paths: Vec<String>,
    pub package_paths: Vec<PathBuf>,
    version: (u8, u8),
}

pub fn query_environment(executable: &Path) -> Result<Environment, String> {
    let output = Command::new(executable)
        .args([
            "-I",
            "-c",
            r#"
import json, sys, sysconfig
packages = list(dict.fromkeys(sysconfig.get_path(name) for name in ('purelib', 'platlib')))
paths = list(dict.fromkeys([*(path for path in sys.path if path), *packages]))
print(json.dumps(dict(paths=paths, package_paths=packages, version=sys.version_info[:2])))
"#,
        ])
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONUSERBASE")
        .env_remove("PYTHONPATH")
        .output()
        .map_err(|error| format!("selected interpreter {}: {error}", executable.display()))?;
    if !output.status.success() {
        return Err(format!(
            "selected interpreter {} failed: {}",
            executable.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let environment: Environment =
        serde_json::from_slice(&output.stdout).map_err(|error| error.to_string())?;
    if environment.package_paths.is_empty()
        || environment
            .package_paths
            .iter()
            .any(|path| !path.is_absolute() || !path.is_dir())
    {
        return Err("selected interpreter has no usable absolute sysconfig package paths".into());
    }
    Ok(environment)
}

pub fn initialize() -> &'static Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| {
        let executable = pyo3_build_config::get()
            .executable
            .as_ref()
            .expect("prerequisite: PyO3 must record its selected native interpreter");
        let environment = query_environment(Path::new(executable))
            .expect("prerequisite: query the PyO3-selected interpreter's actual package paths");
        Python::with_gil(|py| {
            let sys = py.import("sys").expect("embedded Python sys module");
            let version = sys.getattr("version_info").unwrap();
            let actual = (
                version.get_item(0).unwrap().extract::<u8>().unwrap(),
                version.get_item(1).unwrap().extract::<u8>().unwrap(),
            );
            assert_eq!(
                actual, environment.version,
                "embedded Python differs from the selected package interpreter"
            );
            sys.setattr("path", PyList::new(py, &environment.paths).unwrap())
                .unwrap();
        });
        environment
    })
}

/// Destroy fixture globals and Python cycles on the thread that created the
/// unsendable registered model, even when the Python assertion failed.
pub fn run_case<'py>(
    py: Python<'py>,
    source: &std::ffi::CStr,
    globals: &Bound<'py, pyo3::types::PyDict>,
) {
    let outcome = py.run(source, Some(globals), None);
    globals.clear();
    py.import("gc")
        .expect("Python gc")
        .call_method0("collect")
        .expect("collect fixture cycles on their owner thread");
    outcome.expect("registered native boundary case");
}
