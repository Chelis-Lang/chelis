//! Phase-1 entry hard edge for chelis#729's registered-PyO3 signature census.
//!
//! Owning contract: `spec/design/dtype_semantics.md` section C6. Private
//! runtime-dtype decoding helpers are deliberately out of scope; only a live
//! registered callable signature can enter this inventory.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyModule, PyType};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const FROZEN_BASELINE_SHA256: &str =
    "f5a8f16eb2cd7abcd6845f21fc76fc0827f54974ff43ab6ad35b60f77b1b41ec";

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct SurfaceRow {
    kind: String,
    id: String,
    flags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Baseline {
    version: u32,
    citation: String,
    rows: Vec<SurfaceRow>,
}

#[pyfunction]
fn reviewer_raw_dtype_probe(dtype: i32) -> i32 {
    dtype
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("chelis-python crate lives under <workspace>/crates")
        .to_path_buf()
}

#[derive(Debug)]
struct RegisteredSurface {
    functions: Vec<String>,
    methods: Vec<String>,
}

fn registered_surface(include_reviewer_probe: bool) -> RegisteredSurface {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_capacity_census").expect("create Python module");
        chelis_python::register_module(&module).expect("register chelis-python module");
        if include_reviewer_probe {
            module
                .add_function(wrap_pyfunction!(reviewer_raw_dtype_probe, &module).expect("wrap"))
                .expect("register reviewer raw-dtype probe");
        }
        let mut functions = Vec::new();
        let mut methods = Vec::new();
        for (key, value) in module.dict().iter() {
            let name = key.extract::<String>().expect("registered item name");
            if value.is_instance_of::<PyCFunction>() {
                functions.push(name);
                continue;
            }
            if !value.is_instance_of::<PyType>() || name == "ChelisError" {
                continue;
            }
            let keys = value
                .getattr("__dict__")
                .and_then(|dictionary| dictionary.call_method0("keys"))
                .expect("registered class dictionary keys");
            for method_name in keys.try_iter().expect("iterate registered class keys") {
                let method_name = method_name
                    .expect("registered class key")
                    .extract::<String>()
                    .expect("registered method name");
                let method = value
                    .getattr(method_name.as_str())
                    .expect("registered class attribute");
                if !method_name.starts_with("__") || method.is_callable() {
                    methods.push(format!("{name}::{method_name}"));
                }
            }
        }
        functions.sort();
        methods.sort();
        RegisteredSurface { functions, methods }
    })
}

fn run_typed_enumerator(surface: &RegisteredSurface) -> Output {
    let root = workspace_root();
    let mut command = Command::new(root.join(".venv/bin/python"));
    command
        .arg(root.join("scripts/capacity_census_typed.py"))
        .args(["bindings", "--target-dir"])
        .arg(root.join("target/agents/729-capacity-bindings-rustdoc"))
        .current_dir(&root);
    for name in &surface.functions {
        command.args(["--registered", name]);
    }
    for method in &surface.methods {
        command.args(["--registered-method", method]);
    }
    command.output().expect("run typed PyO3 census enumerator")
}

fn baseline_bytes() -> Vec<u8> {
    std::fs::read(workspace_root().join("spec/design/capacity_census_bindings.json"))
        .expect("read reviewed binding census baseline")
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[test]
fn registered_pyfunctions_match_the_reviewed_rustdoc_signatures() {
    let baseline_bytes = baseline_bytes();
    assert_eq!(
        sha256(&baseline_bytes),
        FROZEN_BASELINE_SHA256,
        "binding census guard changed: editing the baseline is not the fix; dispose the \
         registered signature under dtype_semantics.md section C6 and update the frozen \
         fingerprint only with that review"
    );
    let baseline: Baseline =
        serde_json::from_slice(&baseline_bytes).expect("binding baseline JSON");
    assert_eq!(
        baseline.version, 1,
        "unknown binding census baseline version"
    );
    assert!(
        baseline.citation.contains("chelis#729"),
        "the frozen binding baseline must remain liveness-bound to chelis#729"
    );

    let output = run_typed_enumerator(&registered_surface(false));
    assert!(
        output.status.success(),
        "typed binding census failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let current: Vec<SurfaceRow> =
        serde_json::from_slice(&output.stdout).expect("typed binding census JSON");
    assert_eq!(
        current, baseline.rows,
        "registered PyO3 signature shape changed. A raw dtype id or bare numeric carrier \
         has no ordinary issue-citation path: redesign onto the typed carrier, remove the \
         ingress, or obtain the explicit C6 review disposition"
    );
}

#[test]
fn a_registered_pyfunction_with_a_raw_dtype_parameter_is_rejected() {
    let output = run_typed_enumerator(&registered_surface(true));
    assert!(
        !output.status.success(),
        "a newly registered callable absent from the rustdoc-signature census passed"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("reviewer_raw_dtype_probe")
            && stderr.contains("no top-level rustdoc JSON signature"),
        "unexpected fail-closed diagnostic: {stderr}"
    );

    let root = workspace_root();
    let mutation = Command::new(root.join(".venv/bin/python"))
        .arg(root.join("scripts/test_capacity_census_typed.py"))
        .arg("BindingEnumerator")
        .current_dir(&root)
        .output()
        .expect("run typed binding mutation tests");
    assert!(
        mutation.status.success(),
        "binding raw-dtype mutation controls failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&mutation.stdout),
        String::from_utf8_lossy(&mutation.stderr)
    );
}
