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

#[path = "../../../tests/support/capacity_census_authority.rs"]
mod capacity_census_authority;
#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;
use capacity_census_authority::{AuthorityRegistries, SurfaceDescriptor};

const PERMANENT_BINDING_DISPOSITION: &str = "permanent-disposition(C6 registered PyO3 signature surface complete descriptor set ratified 2026-08-04)";
const BINDING_CENSUS_FAMILY: &str = "pyo3-binding";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrozenSurfaceRow {
    kind: &'static str,
    id: &'static str,
    flags: &'static [&'static str],
}

const FROZEN_BINDING_ROWS: &[FrozenSurfaceRow] = &[
    FrozenSurfaceRow {
        kind: "binding-pyfunction",
        id: "chelis_python::check_json(py: Python<'_>, source: &str, source_kind: &str) -> PyResult<String>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pyfunction",
        id: "chelis_python::compile_and_load(py: Python<'_>, source_path: &str, target: &str, source_kind: &str, entry_name: Option<String>, artifact_dir: Option<&str>, project_root: Option<&str>, force_bare: bool) -> PyResult<NativeCompiledModel>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pyfunction",
        id: "chelis_python::compile_json(py: Python<'_>, source: &str, target: &str, source_kind: &str, entry_name: Option<String>) -> PyResult<String>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pyfunction",
        id: "chelis_python::decompile_json(py: Python<'_>, source: &str) -> PyResult<String>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pyfunction",
        id: "chelis_python::desugar_json(py: Python<'_>, source: &str) -> PyResult<String>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pyfunction",
        id: "chelis_python::eval_json(py: Python<'_>, source: &str, bindings_json: &str, source_kind: &str, project_root: Option<&str>) -> PyResult<String>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pyfunction",
        id: "chelis_python::load(py: Python<'_>, path: &str) -> PyResult<NativeCompiledModel>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pyfunction",
        id: "chelis_python::validate_json(py: Python<'_>, source: &str, mode: &str) -> PyResult<String>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::CompiledModel::__call__(self: &Self, py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyObject>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::CompiledModel::input_names(self: &Self) -> Vec<String>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::CompiledModel::output_names(self: &Self) -> Vec<String>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::CompiledModel::path(self: &Self) -> String",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::CompiledModel::target(self: &Self) -> String",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::NativeTensor::__dlpack__(self: &Self, py: Python<'_>, stream: Option<usize>, max_version: Option<&Bound<'_, PyAny>>, dl_device: Option<&Bound<'_, PyAny>>, copy: Option<bool>) -> PyResult<PyObject>",
        flags: &["numeric-param"],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::NativeTensor::__dlpack_device__(self: &Self) -> (i32, i32)",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::NativeTensor::dtype(self: &Self) -> PyResult<&'static str>",
        flags: &[],
    },
    FrozenSurfaceRow {
        kind: "binding-pymethod",
        id: "chelis_python::NativeTensor::shape(self: &Self) -> Vec<usize>",
        flags: &[],
    },
];

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
    let python = managed_python::managed_python(&root).unwrap_or_else(|error| panic!("{error}"));
    let mut command = Command::new(python);
    command
        .arg(root.join("scripts/capacity_census_typed.py"))
        // No `--target-dir`: see the wire census for why the enumerator owns
        // that choice. Both censuses share one cargo target directory so the
        // second one to run reuses the first's compiled dependency graph.
        .arg("bindings")
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

fn frozen_surface_rows() -> Vec<SurfaceRow> {
    FROZEN_BINDING_ROWS
        .iter()
        .map(|row| SurfaceRow {
            kind: row.kind.to_string(),
            id: row.id.to_string(),
            flags: row.flags.iter().map(|flag| (*flag).to_string()).collect(),
        })
        .collect()
}

fn permanent_baseline_problem(bytes: &[u8]) -> Option<String> {
    let baseline: Baseline = match serde_json::from_slice(bytes) {
        Ok(baseline) => baseline,
        Err(error) => return Some(format!("binding baseline is not valid JSON: {error}")),
    };
    if baseline.version != 1 {
        return Some(format!(
            "unknown binding census baseline version {}",
            baseline.version
        ));
    }
    if baseline.citation != PERMANENT_BINDING_DISPOSITION {
        return Some(format!(
            "binding disposition changed: expected {PERMANENT_BINDING_DISPOSITION:?}, got {:?}",
            baseline.citation
        ));
    }
    if baseline.rows != frozen_surface_rows() {
        return Some(
            "binding baseline rows differ from the hand-maintained permanent kind/id/flags manifest"
                .to_string(),
        );
    }
    None
}

fn current_authority_problem(current: &[SurfaceRow]) -> Option<String> {
    let legacy = frozen_surface_rows();
    for row in current {
        if legacy.contains(row) {
            continue;
        }
        let surface = SurfaceDescriptor {
            family: BINDING_CENSUS_FAMILY.to_string(),
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.flags.clone(),
        };
        if let Err(problem) = capacity_census_authority::classify_final_authority(
            &surface,
            AuthorityRegistries {
                nonnumeric: &[],
                tagged_transports: &[],
                numeric_operations: &[],
            },
            "",
        ) {
            return Some(problem);
        }
    }
    None
}

#[test]
fn registered_pyfunctions_match_the_reviewed_rustdoc_signatures() {
    let baseline_bytes = baseline_bytes();
    assert!(
        permanent_baseline_problem(&baseline_bytes).is_none(),
        "binding census guard changed: editing the baseline is not the fix; dispose the \
         registered signature under dtype_semantics.md section C6 and update the frozen \
         complete hand-maintained descriptor manifest only with that review: {:?}",
        permanent_baseline_problem(&baseline_bytes)
    );
    let baseline: Baseline =
        serde_json::from_slice(&baseline_bytes).expect("binding baseline JSON");
    assert_eq!(
        baseline.version, 1,
        "unknown binding census baseline version"
    );
    assert_eq!(baseline.citation, PERMANENT_BINDING_DISPOSITION);

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
        current_authority_problem(&current),
        None,
        "every non-legacy binding row requires exactly one final authority"
    );
    assert_eq!(
        current, baseline.rows,
        "registered PyO3 signature shape changed. A raw dtype id or bare numeric carrier \
         has no ordinary issue-citation path: redesign onto the typed carrier, remove the \
         ingress, or add the exact final authority registration without changing the \
         sealed legacy cohort"
    );
}

#[test]
fn permanent_binding_disposition_is_bound_to_the_complete_descriptor_set() {
    let baseline_bytes = baseline_bytes();
    let mut baseline: serde_json::Value =
        serde_json::from_slice(&baseline_bytes).expect("binding baseline JSON");
    baseline["rows"]
        .as_array_mut()
        .expect("binding rows")
        .push(serde_json::json!({
            "kind": "binding-pyfunction",
            "id": "chelis_python::reviewer_probe(dtype: i32) -> i32",
            "flags": ["raw-dtype-int"]
        }));
    let mutated = serde_json::to_vec(&baseline).expect("serialize mutated binding baseline");
    let problem = permanent_baseline_problem(&mutated)
        .expect("copying the permanent top-level disposition onto an added row must fail");
    assert!(
        problem.contains("permanent kind/id/flags manifest"),
        "the production manifest guard must reject the copied disposition: {problem}"
    );
}

#[test]
fn a_typed_permanent_disposition_cannot_move_between_families() {
    let baseline_bytes = baseline_bytes();
    let mut baseline: serde_json::Value =
        serde_json::from_slice(&baseline_bytes).expect("binding baseline JSON");
    baseline["citation"] = serde_json::json!(
        "permanent-disposition(C6 dtype-tagged wire schema complete descriptor set ratified 2026-08-04)"
    );
    let mutated = serde_json::to_vec(&baseline).expect("serialize wrong-family binding baseline");
    let problem = permanent_baseline_problem(&mutated)
        .expect("the wire disposition must not ratify a binding manifest");
    assert!(problem.contains("binding disposition changed"), "{problem}");
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
    let python = managed_python::managed_python(&root).unwrap_or_else(|error| panic!("{error}"));
    let mutation = Command::new(python)
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
