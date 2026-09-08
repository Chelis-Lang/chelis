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
use capacity_census_authority::{AuthorityRegistries, StaticSurfaceDescriptor, SurfaceDescriptor};

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

// Only these unchanged foundation rows retain the temporary admission path.
const ACTIVE_LEGACY_IDS: &[&str] = &[
    "chelis_python::check_json(py: Python<'_>, source: &str, source_kind: &str) -> PyResult<String>",
    "chelis_python::compile_json(py: Python<'_>, source: &str, target: &str, source_kind: &str, entry_name: Option<String>) -> PyResult<String>",
    "chelis_python::desugar_json(py: Python<'_>, source: &str) -> PyResult<String>",
    "chelis_python::eval_json(py: Python<'_>, source: &str, bindings_json: &str, source_kind: &str, project_root: Option<&str>) -> PyResult<String>",
    "chelis_python::CompiledModel::__call__(self: &Self, py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyObject>",
    "chelis_python::NativeTensor::__dlpack__(self: &Self, py: Python<'_>, stream: Option<usize>, max_version: Option<&Bound<'_, PyAny>>, dl_device: Option<&Bound<'_, PyAny>>, copy: Option<bool>) -> PyResult<PyObject>",
    "chelis_python::NativeTensor::__dlpack_device__(self: &Self) -> (i32, i32)",
    "chelis_python::NativeTensor::shape(self: &Self) -> Vec<usize>",
];

const NONNUMERIC_BINDINGS: &[StaticSurfaceDescriptor] = &[
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pyfunction",
        "chelis_python::decompile_json(py: Python<'_>, source: &str) -> PyResult<source_json::SourceJson<chelis_compiler_api::schema::DecompileResult>>",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pyfunction",
        "chelis_python::compile_and_load(py: Python<'_>, source_path: &str, target: &str, source_kind: &str, entry_name: Option<String>, artifact_dir: Option<&str>, project_root: Option<&str>, force_bare: bool) -> PyResult<NativeCompiledModel>",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pyfunction",
        "chelis_python::load(py: Python<'_>, path: &str) -> PyResult<NativeCompiledModel>",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pyfunction",
        "chelis_python::validate_json(py: Python<'_>, source: &str, mode: &str) -> PyResult<source_json::SourceJson<chelis_compiler_api::schema::ValidateResult>>",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pymethod",
        "chelis_python::CompiledModel::input_names(self: &Self) -> Vec<String>",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pymethod",
        "chelis_python::CompiledModel::output_names(self: &Self) -> Vec<String>",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pymethod",
        "chelis_python::CompiledModel::path(self: &Self) -> String",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pymethod",
        "chelis_python::CompiledModel::target(self: &Self) -> String",
        &[],
    ),
    StaticSurfaceDescriptor::new(
        BINDING_CENSUS_FAMILY,
        "binding-pymethod",
        "chelis_python::NativeTensor::dtype(self: &Self) -> PyResult<&'static str>",
        &[],
    ),
];

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct SurfaceRow {
    kind: String,
    id: String,
    flags: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    version: u32,
    rows: Vec<BaselineRow>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineRow {
    kind: String,
    id: String,
    flags: Vec<String>,
    #[serde(default)]
    citation: Option<String>,
    #[serde(default)]
    authority: Option<String>,
    #[serde(default)]
    graph_identity: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DiscoveredRow {
    kind: String,
    id: String,
    flags: Vec<String>,
    legacy_flags: Vec<String>,
    identity: Option<String>,
    problem: Option<String>,
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
    classes: Vec<(String, String)>,
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
        let mut classes = Vec::new();
        let mut declared = chelis_python::capacity_census_classes().to_vec();
        declared.sort();
        for (key, value) in module.dict().iter() {
            let name = key.extract::<String>().expect("registered item name");
            if value.is_instance_of::<PyCFunction>() {
                functions.push(name);
                continue;
            }
            if !value.is_instance_of::<PyType>() || name == "ChelisError" {
                continue;
            }
            classes.push(name.clone());
            let keys = value
                .getattr("__dict__")
                .and_then(|dictionary| dictionary.call_method0("keys"))
                .expect("registered class dictionary keys");
            for method_name in keys.try_iter().expect("iterate registered class keys") {
                let method_name = method_name
                    .expect("registered class key")
                    .extract::<String>()
                    .expect("registered method name");
                if method_name == "__new__"
                    && declared
                        .iter()
                        .any(|(class, _, has_constructor)| *class == name && !has_constructor)
                {
                    continue;
                }
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
        classes.sort();
        assert_eq!(
            classes,
            declared
                .iter()
                .map(|(name, _, _)| name.to_string())
                .collect::<Vec<_>>(),
            "every live registered class requires an exact Rust PyClass identity"
        );
        RegisteredSurface {
            functions,
            methods,
            classes: declared
                .into_iter()
                .map(|(name, identity, _)| (name.to_string(), identity.to_string()))
                .collect(),
        }
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
        .arg("bindings-discovery")
        .current_dir(&root);
    for name in &surface.functions {
        command.args(["--registered", name]);
    }
    for method in &surface.methods {
        command.args(["--registered-method", method]);
    }
    for (name, identity) in &surface.classes {
        command.args(["--registered-class", &format!("{name}={identity}")]);
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

fn active_legacy_rows() -> Vec<SurfaceRow> {
    frozen_surface_rows()
        .into_iter()
        .filter(|row| ACTIVE_LEGACY_IDS.contains(&row.id.as_str()))
        .collect()
}

fn current_authority_problem(current: &[SurfaceRow]) -> Option<String> {
    let legacy = active_legacy_rows();
    let mut seen = std::collections::BTreeSet::new();
    for row in current {
        if !seen.insert((&row.kind, &row.id)) {
            return Some(format!("duplicate binding identity {}", row.id));
        }
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
                nonnumeric: NONNUMERIC_BINDINGS,
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

fn baseline_problem(bytes: &[u8]) -> Option<String> {
    let baseline: Baseline = match serde_json::from_slice(bytes) {
        Ok(value) => value,
        Err(error) => return Some(format!("invalid binding baseline: {error}")),
    };
    if baseline.version != 2 {
        return Some("unknown binding baseline version".into());
    }
    let legacy = active_legacy_rows();
    let mut rows = Vec::new();
    for row in &baseline.rows {
        let surface = SurfaceRow {
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.flags.clone(),
        };
        if legacy.contains(&surface) {
            if row.citation.as_deref() != Some(PERMANENT_BINDING_DISPOSITION)
                || row.authority.is_some()
                || row.graph_identity.is_some()
            {
                return Some("changed legacy binding disposition".into());
            }
        } else if row.citation.is_some()
            || row.authority.as_deref() != Some("nonnumeric")
            || !row
                .graph_identity
                .as_ref()
                .is_some_and(|id| id.len() == 64 && id.bytes().all(|c| c.is_ascii_hexdigit()))
        {
            return Some(
                "final binding requires only its authority and current graph identity".into(),
            );
        }
        rows.push(surface);
    }
    if let Some(problem) = current_authority_problem(&rows) {
        return Some(problem);
    }
    if rows.len() != legacy.len() + NONNUMERIC_BINDINGS.len()
        || legacy.iter().any(|row| !rows.contains(row))
        || NONNUMERIC_BINDINGS
            .iter()
            .any(|row| !rows.iter().any(|r| r.kind == row.kind && r.id == row.id))
    {
        return Some("binding census and authority registries are not bijective".into());
    }
    None
}

fn discovered_problem(discovered: &[DiscoveredRow], baseline: &Baseline) -> Option<String> {
    let legacy = active_legacy_rows();
    let mut current = Vec::new();
    for row in discovered {
        let old = SurfaceRow {
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.legacy_flags.clone(),
        };
        let surface = if legacy.contains(&old) {
            old
        } else {
            if let Some(problem) = &row.problem {
                return Some(format!("{}: {problem}", row.id));
            }
            if row.identity.is_none() {
                return Some(format!("{}: missing exposure identity", row.id));
            }
            SurfaceRow {
                kind: row.kind.clone(),
                id: row.id.clone(),
                flags: row.flags.clone(),
            }
        };
        let Some(expected) = baseline
            .rows
            .iter()
            .find(|expected| expected.kind == surface.kind && expected.id == surface.id)
        else {
            return Some(format!("unregistered binding {}", row.id));
        };
        if surface.flags != expected.flags {
            return Some(format!("changed binding capacity {}", row.id));
        }
        if !legacy.contains(&surface) && row.identity != expected.graph_identity {
            return Some(format!("changed reachable binding graph {}", row.id));
        }
        current.push(surface);
    }
    if current.len() != baseline.rows.len() {
        return Some("binding inventory is not bijective".into());
    }
    current_authority_problem(&current)
}

#[test]
fn registered_pyfunctions_match_the_reviewed_rustdoc_signatures() {
    let bytes = baseline_bytes();
    assert_eq!(baseline_problem(&bytes), None);
    let baseline: Baseline = serde_json::from_slice(&bytes).unwrap();
    let output = run_typed_enumerator(&registered_surface(false));
    assert!(
        output.status.success(),
        "typed binding discovery failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let current: Vec<DiscoveredRow> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        discovered_problem(&current, &baseline),
        None,
        "only exact unchanged legacy rows may defer exposure proof; every final/new row requires complete discovery and authority"
    );
}

#[test]
fn retired_binding_rows_cannot_regain_legacy_admission() {
    let retired: Vec<_> = frozen_surface_rows()
        .into_iter()
        .filter(|row| !active_legacy_rows().contains(row))
        .collect();
    assert_eq!(retired.len(), 9);
    for mut row in retired {
        row.flags = vec!["float-carrier".into()];
        assert!(current_authority_problem(&[row]).is_some());
    }
    let mut baseline: serde_json::Value = serde_json::from_slice(&baseline_bytes()).unwrap();
    let row = baseline["rows"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row.get("authority").is_some())
        .unwrap();
    row["citation"] = serde_json::json!(PERMANENT_BINDING_DISPOSITION);
    assert!(baseline_problem(&serde_json::to_vec(&baseline).unwrap()).is_some());
}

#[test]
fn copied_missing_and_duplicate_binding_registrations_fail() {
    for mutation in ["copy", "missing", "duplicate", "numeric"] {
        let mut baseline: serde_json::Value = serde_json::from_slice(&baseline_bytes()).unwrap();
        let rows = baseline["rows"].as_array_mut().unwrap();
        match mutation {
            "copy" => {
                let mut row = rows[0].clone();
                row["id"] = serde_json::json!("new_binding");
                rows.push(row);
            }
            "missing" => {
                rows.pop();
            }
            "duplicate" => rows.push(rows[0].clone()),
            "numeric" => {
                let row = rows
                    .iter_mut()
                    .find(|row| row.get("authority").is_some())
                    .unwrap();
                row["flags"] = serde_json::json!(["numeric-return"]);
            }
            _ => unreachable!(),
        }
        assert!(
            baseline_problem(&serde_json::to_vec(&baseline).unwrap()).is_some(),
            "{mutation}"
        );
    }
}

#[test]
fn final_bindings_require_successful_current_exposure() {
    let baseline: Baseline = serde_json::from_slice(&baseline_bytes()).unwrap();
    let make_current = || {
        baseline
            .rows
            .iter()
            .map(|row| DiscoveredRow {
                kind: row.kind.clone(),
                id: row.id.clone(),
                flags: row.flags.clone(),
                legacy_flags: row.flags.clone(),
                identity: row.graph_identity.clone(),
                problem: None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(discovered_problem(&make_current(), &baseline), None);
    for mutation in ["unresolved", "numeric", "stale", "missing", "duplicate"] {
        let mut current = make_current();
        let row = current
            .iter_mut()
            .find(|row| row.identity.is_some())
            .unwrap();
        match mutation {
            "unresolved" => row.problem = Some("missing defining artifact".into()),
            "numeric" => row.flags = vec!["float-carrier".into(), "numeric-return".into()],
            "stale" => row.identity = Some("f".repeat(64)),
            "missing" => row.identity = None,
            "duplicate" => {
                let duplicate = current.remove(0);
                current[0] = duplicate;
            }
            _ => unreachable!(),
        }
        assert!(
            discovered_problem(&current, &baseline).is_some(),
            "{mutation}"
        );
    }
    let mut value: serde_json::Value = serde_json::from_slice(&baseline_bytes()).unwrap();
    value["citation"] = serde_json::json!(PERMANENT_BINDING_DISPOSITION);
    assert!(baseline_problem(&serde_json::to_vec(&value).unwrap()).is_some());
}

#[test]
fn a_registered_pyfunction_with_a_raw_dtype_parameter_is_rejected() {
    let output = run_typed_enumerator(&registered_surface(true));
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("reviewer_raw_dtype_probe")
            && stderr.contains("missing registered function"),
        "{stderr}"
    );
    let root = workspace_root();
    let mutation = Command::new(managed_python::managed_python(&root).unwrap())
        .arg(root.join("scripts/test_capacity_census_bindings.py"))
        .current_dir(&root)
        .output()
        .unwrap();
    assert!(
        mutation.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&mutation.stdout),
        String::from_utf8_lossy(&mutation.stderr)
    );
}
