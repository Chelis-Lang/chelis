//! Phase-1 entry hard edge for chelis#729's registered-PyO3 signature census.
//!
//! Owning contract: `spec/design/dtype_semantics.md` section C6. Private
//! runtime-dtype decoding helpers are deliberately out of scope; only a live
//! registered callable signature can enter this inventory.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, io::Write};

use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyModule, PyType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[path = "../../../tests/support/capacity_census_authority.rs"]
mod capacity_census_authority;
#[path = "../../../tests/support/capacity_census_compiler_json.rs"]
mod capacity_census_compiler_json;
#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;
#[path = "../../../tests/support/pyo3_registration.rs"]
mod pyo3_registration;
#[path = "../../../tests/support/pyo3_registration_fixture.rs"]
mod pyo3_registration_fixture;
use capacity_census_authority::{AuthorityRegistries, StaticSurfaceDescriptor, SurfaceDescriptor};
use capacity_census_compiler_json::{DiscoveredRow, VerifiedBindingCensus};

const BINDING_CENSUS_FAMILY: &str = "pyo3-binding";
const BINDING_CENSUS_WRITE_ENV: &str = "CHELIS_CAPACITY_CENSUS_BINDINGS_WRITE";

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

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    version: u32,
    rows: Vec<BaselineRow>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineRow {
    flags: Vec<String>,
    id: String,
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    citation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    authority: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    graph_identity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    contract: Option<String>,
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
    provenance: Provenance,
}

#[derive(Debug, Serialize)]
struct Provenance {
    source_path: &'static str,
    source_sha256: String,
    registrations: Vec<pyo3_registration::Registration>,
}

fn registered_surface(include_reviewer_probe: bool) -> RegisteredSurface {
    const SOURCE: &str = include_str!("../src/lib.rs");
    let provenance = Provenance {
        source_path: "crates/chelis-python/src/lib.rs",
        source_sha256: format!("{:x}", Sha256::digest(SOURCE.as_bytes())),
        registrations: pyo3_registration::registrations(SOURCE)
            .expect("derive exact PyO3 registration provenance from compiled source"),
    };
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
                    let raw = value
                        .getattr("__dict__")
                        .and_then(|dictionary| dictionary.get_item(&method_name))
                        .expect("actual class descriptor");
                    let descriptor = raw.get_type().name().unwrap().to_string();
                    let rust_class = declared
                        .iter()
                        .find(|(class, _, _)| *class == name)
                        .expect("registered class provenance")
                        .1;
                    let registrations: Vec<_> = provenance
                        .registrations
                        .iter()
                        .filter(|row| {
                            row.owner.as_ref().is_some_and(|owner| {
                                format!("chelis_python::{owner}") == rust_class
                            }) && row.python_name == method_name
                        })
                        .collect();
                    assert!(
                        !registrations.is_empty(),
                        "missing live descriptor provenance: {name}::{method_name}"
                    );
                    for registration in registrations {
                        let expected = match registration.kind.as_str() {
                            "getter" | "setter" => "getset_descriptor",
                            "constructor" => "builtin_function_or_method",
                            "classmethod" => "classmethod_descriptor",
                            "staticmethod" => "staticmethod",
                            "method" if method_name == "__call__" => "wrapper_descriptor",
                            "method" => "method_descriptor",
                            _ => panic!("unsupported live descriptor provenance"),
                        };
                        assert_eq!(
                            descriptor, expected,
                            "live descriptor provenance for {name}::{method_name}"
                        );
                    }
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
            provenance,
        }
    })
}

#[test]
fn source_registration_matches_actual_pyo3_descriptor_kinds() {
    let surface = registered_surface(false);
    assert_eq!(surface.provenance.registrations.len(), 17);
    assert_eq!(surface.functions.len(), 8);
    assert_eq!(surface.methods.len(), 9);
}

#[test]
fn renamed_callables_join_compiled_execution_to_actual_rustdoc() {
    pyo3_registration_fixture::verify_execution();
    const PATH: &str = "tests/support/pyo3_registration_fixture.rs";
    const SOURCE: &str = include_str!("../../../tests/support/pyo3_registration_fixture.rs");
    let provenance = Provenance {
        source_path: PATH,
        source_sha256: format!("{:x}", Sha256::digest(SOURCE.as_bytes())),
        registrations: pyo3_registration::registrations(SOURCE).unwrap(),
    };
    let root = workspace_root();
    let temporary = tempfile::tempdir().unwrap();
    let metadata = temporary.path().join("provenance.json");
    std::fs::write(&metadata, serde_json::to_vec(&provenance).unwrap()).unwrap();
    let deps = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    // Cargo may retain artifacts from older dependency feature combinations.
    // The fixture uses the current build's newest PyO3 artifact, with the
    // same lockfile and literal fixture source as its compiled execution.
    let pyo3 = std::fs::read_dir(&deps)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_name().to_string_lossy().starts_with("libpyo3-")
                && entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "rlib")
        })
        .max_by_key(|entry| entry.metadata().unwrap().modified().unwrap())
        .unwrap()
        .path();
    let output = Command::new("rustdoc")
        .args([
            PATH,
            "--edition=2024",
            "--crate-name",
            "chelis_python",
            "--crate-type",
            "lib",
            "--extern",
            &format!("pyo3={}", pyo3.display()),
            "-L",
            &format!("dependency={}", deps.display()),
            "--output-format",
            "json",
            "-Z",
            "unstable-options",
            "--document-private-items",
            "-o",
        ])
        .arg(temporary.path())
        .env("RUSTC_BOOTSTRAP", "1")
        .current_dir(&root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let python = managed_python::managed_python(&root).unwrap();
    let output = Command::new(python).arg("-c").arg(r#"
import copy, json, sys
sys.path.insert(0, 'scripts')
from capacity_census_bindings import discover_bindings
document = json.load(open(sys.argv[1]))
proof = json.load(open(sys.argv[2]))
functions = ['public_function']
methods = ['NativeTensor::' + name for name in ['__new__', 'dtype', 'run', 'from_code', 'accept', '__call__']]
classes = {'NativeTensor': 'chelis_python::NativeTensor'}
rows = discover_bindings([document], functions, methods, classes, provenance=proof)
assert len(rows) == 8, rows
direct = [row for row in rows if not row['implementation'].endswith('#classmethod')]
assert len(direct) == 7 and all(row['problem'] is None and row['flags'] for row in direct), rows
class_row = next(row for row in rows if row['implementation'].endswith('#classmethod'))
assert class_row['implementation'] == 'chelis_python::NativeTensor::build#classmethod'
assert 'dynamic Python payload requires' in class_row['problem'], class_row
assert any(row['implementation'] == 'chelis_python::NativeTensor::dtype_code#getter' and row['flags'] == ['numeric-return'] for row in rows)
assert any(row['implementation'] == 'chelis_python::NativeTensor::assign_dtype#setter' and row['flags'] == ['raw-dtype-int'] for row in rows)
assert any(row['implementation'] == 'chelis_python::NativeTensor::create#constructor' and row['flags'] == ['raw-dtype-int'] for row in rows)
for key in ['source_sha256', 'registrations']:
    bad = copy.deepcopy(proof)
    bad[key] = '0' * 64 if key == 'source_sha256' else bad[key][1:]
    try: discover_bindings([document], functions, methods, classes, provenance=bad)
    except Exception as error: assert 'provenance' in str(error), str(error)
    else: raise AssertionError('stale/missing provenance accepted')
print('Eight compiled renamed implementations bound; seven direct numeric payloads classified; dynamic class receiver rejected; decoy helpers excluded; stale/missing receipts rejected.')
"#).arg(temporary.path().join("chelis_python.json")).arg(metadata).current_dir(root).output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_typed_enumerator(surface: &RegisteredSurface) -> Result<VerifiedBindingCensus, String> {
    capacity_census_compiler_json::discover(
        &surface.functions,
        &surface.methods,
        &surface.classes,
        &surface.provenance,
    )
}

fn baseline_bytes() -> Vec<u8> {
    std::fs::read(workspace_root().join("spec/design/capacity_census_bindings.json"))
        .expect("read reviewed binding census baseline")
}

fn valid_graph_identity(identity: &str) -> bool {
    identity.len() == 64
        && identity
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn refresh_binding_baseline_graphs(
    bytes: &[u8],
    current: &[DiscoveredRow],
) -> Result<Vec<u8>, String> {
    if let Some(problem) = baseline_problem(bytes) {
        return Err(format!(
            "refusing to rewrite an invalid binding baseline: {problem}"
        ));
    }
    let mut baseline: Baseline = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid binding baseline: {error}"))?;
    if current.len() != baseline.rows.len() {
        return Err("binding writer cannot add or remove baseline rows".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for discovered in current {
        let identity = discovered
            .identity
            .as_deref()
            .filter(|identity| valid_graph_identity(identity))
            .ok_or_else(|| format!("{} lacks a current graph identity", discovered.id))?;
        if discovered.problem.is_some() {
            return Err(format!("{} has an unresolved current graph", discovered.id));
        }
        let Some(expected) = baseline
            .rows
            .iter_mut()
            .find(|row| row.kind == discovered.kind && row.id == discovered.id)
        else {
            return Err(format!(
                "binding writer cannot classify a changed surface: {}",
                discovered.id
            ));
        };
        if !seen.insert((discovered.kind.as_str(), discovered.id.as_str())) {
            return Err(format!("duplicate discovered binding {}", discovered.id));
        }
        let expected_authority = match expected.authority.as_deref() {
            Some("nonnumeric") => None,
            _ => expected.authority.as_deref(),
        };
        if discovered.flags != expected.flags
            || discovered.authority.as_deref() != expected_authority
            || discovered.contract != expected.contract
        {
            return Err(format!(
                "binding writer cannot change flags or authority: {}",
                discovered.id
            ));
        }
        expected.graph_identity = Some(identity.to_string());
    }
    if seen.len() != baseline.rows.len() {
        return Err("binding writer did not cover every reviewed row".into());
    }
    let mut rendered = serde_json::to_vec_pretty(&baseline).map_err(|error| error.to_string())?;
    rendered.push(b'\n');
    if let Some(problem) = baseline_problem(&rendered) {
        return Err(format!("rewritten binding baseline is invalid: {problem}"));
    }
    Ok(rendered)
}

fn write_binding_baseline(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "binding baseline has no parent directory".to_string())?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    temporary
        .write_all(bytes)
        .and_then(|()| temporary.as_file_mut().sync_all())
        .map_err(|error| error.to_string())?;
    temporary
        .persist(path)
        .map_err(|error| error.error.to_string())?;
    Ok(())
}

fn current_authority_problem(
    current: &[SurfaceRow],
    execution: Option<&VerifiedBindingCensus>,
) -> Option<String> {
    let mut seen = std::collections::BTreeSet::new();
    for row in current {
        if !seen.insert((&row.kind, &row.id)) {
            return Some(format!("duplicate binding identity {}", row.id));
        }
        let surface = SurfaceDescriptor {
            family: BINDING_CENSUS_FAMILY.to_string(),
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.flags.clone(),
        };
        if execution.is_some_and(|witness| witness.permits(&surface)) {
            continue;
        }
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
    let mut rows = Vec::new();
    for row in &baseline.rows {
        let surface = SurfaceRow {
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.flags.clone(),
        };
        if row.citation.is_some()
            || !matches!(
                row.authority.as_deref(),
                Some("nonnumeric" | "TaggedTransport" | "NumericOperation")
            )
            || !row
                .graph_identity
                .as_ref()
                .is_some_and(|id| id.len() == 64 && id.bytes().all(|c| c.is_ascii_hexdigit()))
        {
            return Some(
                "final binding requires only its authority and current graph identity".into(),
            );
        }
        match row.authority.as_deref() {
            Some("TaggedTransport") => {
                let compiler = ["check_json", "compile_json", "desugar_json", "eval_json"]
                    .iter()
                    .any(|name| {
                        row.contract.as_deref()
                            == Some(format!("compiler-json/chelis_python::{name}").as_str())
                            && row.id.starts_with(&format!("chelis_python::{name}("))
                            && row.kind == "binding-pyfunction"
                    });
                let native = [
                    (
                        "native/compiled-tensor-call",
                        "chelis_python::CompiledModel::__call__(",
                    ),
                    (
                        "native/dlpack-capsule",
                        "chelis_python::NativeTensor::__dlpack__(",
                    ),
                    (
                        "native/dlpack-device",
                        "chelis_python::NativeTensor::__dlpack_device__(",
                    ),
                ]
                .iter()
                .any(|(contract, prefix)| {
                    row.contract.as_deref() == Some(*contract)
                        && row.id.starts_with(prefix)
                        && row.kind == "binding-pymethod"
                });
                if (!compiler && !native) || row.flags.is_empty() {
                    return Some(
                        "invalid binding transport baseline; execution still required".into(),
                    );
                }
            }
            Some("NumericOperation") => {
                if row.contract.as_deref() != Some("[05-OP-45]")
                    || !row.id.starts_with("chelis_python::NativeTensor::shape(")
                    || row.kind != "binding-pymethod"
                    || row.flags != vec!["numeric-return".to_string()]
                {
                    return Some("invalid native shape operation registration".into());
                }
            }
            Some("nonnumeric") if row.contract.is_some() => {
                return Some("unexpected binding authority contract".into());
            }
            _ => {}
        }
        rows.push(surface);
    }
    // Persisted numeric rows carry comparison data only. They cannot be
    // handed to current_authority_problem as an admission registry.
    let structural: Vec<_> = rows
        .iter()
        .filter(|surface| {
            !baseline.rows.iter().any(|row| {
                row.kind == surface.kind
                    && row.id == surface.id
                    && matches!(
                        row.authority.as_deref(),
                        Some("TaggedTransport" | "NumericOperation")
                    )
            })
        })
        .map(|row| SurfaceRow {
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.flags.clone(),
        })
        .collect();
    if let Some(problem) = current_authority_problem(&structural, None) {
        return Some(problem);
    }
    let transports = baseline
        .rows
        .iter()
        .filter(|row| row.authority.as_deref() == Some("TaggedTransport"))
        .count();
    let numeric_operations = baseline
        .rows
        .iter()
        .filter(|row| row.authority.as_deref() == Some("NumericOperation"))
        .count();
    let transport_contracts: std::collections::BTreeSet<_> = baseline
        .rows
        .iter()
        .filter(|row| row.authority.as_deref() == Some("TaggedTransport"))
        .map(|row| row.contract.as_ref())
        .collect();
    let unique: std::collections::BTreeSet<_> =
        rows.iter().map(|row| (&row.kind, &row.id)).collect();
    if unique.len() != rows.len()
        || transport_contracts.len() != transports
        || transports != 7
        || numeric_operations != 1
        || rows.len() != NONNUMERIC_BINDINGS.len() + transports + numeric_operations
        || NONNUMERIC_BINDINGS
            .iter()
            .any(|row| !rows.iter().any(|r| r.kind == row.kind && r.id == row.id))
    {
        return Some("binding census and authority registries are not bijective".into());
    }
    None
}

fn discovered_problem(
    discovered: &[DiscoveredRow],
    baseline: &Baseline,
    execution: Option<&VerifiedBindingCensus>,
) -> Option<String> {
    let mut current = Vec::new();
    for row in discovered {
        if let Some(problem) = &row.problem {
            return Some(format!("{}: {problem}", row.id));
        }
        if row.identity.is_none() {
            return Some(format!("{}: missing exposure identity", row.id));
        }
        let surface = SurfaceRow {
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.flags.clone(),
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
        if row.identity != expected.graph_identity {
            return Some(format!("changed reachable binding graph {}", row.id));
        }
        if matches!(
            expected.authority.as_deref(),
            Some("TaggedTransport" | "NumericOperation")
        ) && (row.authority != expected.authority || row.contract != expected.contract)
        {
            return Some(format!("changed binding transport contract {}", row.id));
        }
        current.push(surface);
    }
    if current.len() != baseline.rows.len() {
        return Some("binding inventory is not bijective".into());
    }
    current_authority_problem(&current, execution)
}

#[test]
fn registered_pyfunctions_match_the_reviewed_rustdoc_signatures() {
    // The baseline cannot issue authority, so collect the sealed current
    // execution witness before comparing any persisted identity.
    let execution =
        run_typed_enumerator(&registered_surface(false)).unwrap_or_else(|error| panic!("{error}"));
    let path = workspace_root().join("spec/design/capacity_census_bindings.json");
    let mut bytes = baseline_bytes();
    assert_eq!(baseline_problem(&bytes), None);
    let current = execution.rows();
    match env::var(BINDING_CENSUS_WRITE_ENV) {
        Ok(value) => {
            assert_eq!(
                value, "1",
                "{BINDING_CENSUS_WRITE_ENV} accepts only the exact value 1"
            );
            bytes = refresh_binding_baseline_graphs(&bytes, current)
                .unwrap_or_else(|problem| panic!("{problem}"));
            write_binding_baseline(&path, &bytes)
                .unwrap_or_else(|problem| panic!("cannot write {}: {problem}", path.display()));
            println!("refreshed executed graph identities in {}", path.display());
        }
        Err(env::VarError::NotPresent) => {}
        Err(error) => panic!("{BINDING_CENSUS_WRITE_ENV} is invalid: {error}"),
    }
    let baseline: Baseline = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        discovered_problem(current, &baseline, Some(&execution)),
        None,
        "every numeric binding requires complete current discovery and authority\nactual discovery: {}\nexecution summary: {}",
        serde_json::to_string(current).unwrap(),
        serde_json::to_string(&execution.evidence()["native"]["execution"]).unwrap()
    );
    let surfaces = current
        .iter()
        .filter(|row| {
            matches!(
                row.authority.as_deref(),
                Some("TaggedTransport" | "NumericOperation")
            )
        })
        .map(|row| SurfaceRow {
            kind: row.kind.clone(),
            id: row.id.clone(),
            flags: row.flags.clone(),
        })
        .collect::<Vec<_>>();
    assert_eq!(surfaces.len(), 8);
    assert!(
        current_authority_problem(&surfaces, None).is_some(),
        "executed rows are comparison data; copying them supplies no witness"
    );
}

#[test]
fn verified_binding_authority_cannot_be_replaced_by_descriptor_or_baseline() {
    let allowed = &NONNUMERIC_BINDINGS[0];
    assert_eq!(
        current_authority_problem(
            &[SurfaceRow {
                kind: allowed.kind.into(),
                id: allowed.id.into(),
                flags: vec![]
            }],
            None
        ),
        None
    );
    let descriptors = [
        (
            "binding-pyfunction",
            "chelis_python::check_json(typed) -> PyResult<compiler_json::Adapter>",
            vec!["float-carrier".into(), "numeric-return".into()],
        ),
        (
            "binding-pyfunction",
            "chelis_python::compile_json(typed) -> PyResult<compiler_json::Adapter>",
            vec!["numeric-return".into()],
        ),
        (
            "binding-pyfunction",
            "chelis_python::desugar_json(typed) -> PyResult<compiler_json::Adapter>",
            vec!["float-carrier".into(), "numeric-return".into()],
        ),
        (
            "binding-pyfunction",
            "chelis_python::eval_json(typed) -> PyResult<compiler_json::Adapter>",
            vec![
                "float-carrier".into(),
                "numeric-param".into(),
                "numeric-return".into(),
            ],
        ),
        (
            "binding-pymethod",
            "chelis_python::CompiledModel::__call__(typed)",
            vec![
                "float-carrier".into(),
                "numeric-param".into(),
                "numeric-return".into(),
            ],
        ),
        (
            "binding-pymethod",
            "chelis_python::NativeTensor::__dlpack__(typed)",
            vec![
                "float-carrier".into(),
                "numeric-param".into(),
                "numeric-return".into(),
            ],
        ),
        (
            "binding-pymethod",
            "chelis_python::NativeTensor::__dlpack_device__(typed)",
            vec!["numeric-return".into()],
        ),
        (
            "binding-pymethod",
            "chelis_python::NativeTensor::shape(typed)",
            vec!["numeric-return".into()],
        ),
    ];
    for (kind, id, flags) in descriptors {
        let descriptor = SurfaceRow {
            kind: kind.into(),
            id: id.into(),
            flags,
        };
        assert!(current_authority_problem(&[descriptor], None).is_some());
    }
}

#[test]
fn final_binding_rows_cannot_regain_legacy_admission() {
    let baseline: serde_json::Value = serde_json::from_slice(&baseline_bytes()).unwrap();
    let final_ids = baseline["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert_eq!(final_ids.len(), 17);
    for id in final_ids {
        let mut mutation = baseline.clone();
        let row = mutation["rows"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|row| row["id"].as_str() == Some(id.as_str()))
            .unwrap();
        row["citation"] = serde_json::json!("retired legacy admission");
        assert!(
            baseline_problem(&serde_json::to_vec(&mutation).unwrap()).is_some(),
            "final binding regained citation admission: {id}"
        );
    }
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
                    .find(|row| {
                        row.get("authority").and_then(|value| value.as_str()) == Some("nonnumeric")
                    })
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
                authority: row.authority.clone(),
                contract: row.contract.clone(),
                problem: None,
                implementation: String::new(),
            })
            .collect::<Vec<_>>()
    };
    let structural_baseline = Baseline {
        version: baseline.version,
        rows: baseline
            .rows
            .iter()
            .filter(|row| row.authority.as_deref() == Some("nonnumeric"))
            .cloned()
            .collect(),
    };
    let structural_current: Vec<_> = make_current()
        .into_iter()
        .filter(|row| row.authority.as_deref() == Some("nonnumeric"))
        .collect();
    assert_eq!(
        discovered_problem(&structural_current, &structural_baseline, None),
        None
    );
    assert!(
        discovered_problem(&make_current(), &baseline, None).is_some(),
        "baseline rows cannot issue numeric binding authority"
    );
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
            discovered_problem(&current, &baseline, None).is_some(),
            "{mutation}"
        );
    }
    let mut value: serde_json::Value = serde_json::from_slice(&baseline_bytes()).unwrap();
    value["citation"] = serde_json::json!("retired legacy admission");
    assert!(baseline_problem(&serde_json::to_vec(&value).unwrap()).is_some());
}

#[test]
fn a_registered_pyfunction_with_a_raw_dtype_parameter_is_rejected() {
    let stderr = run_typed_enumerator(&registered_surface(true)).unwrap_err();
    assert!(
        stderr.contains("reviewer_raw_dtype_probe")
            && stderr.contains("missing registration provenance"),
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

fn writer_fixture_rows() -> Vec<DiscoveredRow> {
    let baseline: Baseline = serde_json::from_slice(&baseline_bytes()).unwrap();
    baseline
        .rows
        .into_iter()
        .map(|row| DiscoveredRow {
            kind: row.kind,
            id: row.id,
            flags: row.flags.clone(),
            legacy_flags: row.flags,
            identity: row.graph_identity,
            problem: None,
            implementation: "executed fixture".into(),
            authority: match row.authority.as_deref() {
                Some("nonnumeric") => None,
                _ => row.authority,
            },
            contract: row.contract,
        })
        .collect()
}

#[test]
fn binding_baseline_writer_changes_only_executed_graph_identities() {
    let before = baseline_bytes();
    let mut rows = writer_fixture_rows();
    rows[0].identity = Some("b".repeat(64));
    let after = refresh_binding_baseline_graphs(&before, &rows).unwrap();
    assert_eq!(baseline_problem(&after), None);

    let mut before_value: serde_json::Value = serde_json::from_slice(&before).unwrap();
    let mut after_value: serde_json::Value = serde_json::from_slice(&after).unwrap();
    let before_rows = before_value["rows"].as_array_mut().unwrap();
    let after_rows = after_value["rows"].as_array_mut().unwrap();
    for (before_row, after_row) in before_rows.iter_mut().zip(after_rows.iter_mut()) {
        before_row.as_object_mut().unwrap().remove("graph_identity");
        after_row.as_object_mut().unwrap().remove("graph_identity");
    }
    assert_eq!(before_value, after_value);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&after).unwrap()["rows"][0]["graph_identity"],
        "b".repeat(64)
    );
    assert_eq!(
        refresh_binding_baseline_graphs(&after, &rows).unwrap(),
        after,
        "a current executed graph must be byte-idempotent"
    );
}

#[test]
fn binding_baseline_writer_rejects_surface_and_authority_drift() {
    let before = baseline_bytes();
    for mutation in [
        "missing",
        "identity",
        "flags",
        "authority",
        "contract",
        "problem",
    ] {
        let mut rows = writer_fixture_rows();
        match mutation {
            "missing" => {
                rows.pop();
            }
            "identity" => rows[0].id.push_str(" changed"),
            "flags" => rows[0].flags.push("numeric-param".into()),
            "authority" => rows[0].authority = Some("NumericOperation".into()),
            "contract" => rows[0].contract = Some("[05-OP-999]".into()),
            "problem" => rows[0].problem = Some("unresolved current graph".into()),
            _ => unreachable!(),
        }
        assert!(
            refresh_binding_baseline_graphs(&before, &rows).is_err(),
            "{mutation}"
        );
    }
}
