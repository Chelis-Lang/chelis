//! Test-owned native execution artifacts. This confers no production authority.
#![allow(dead_code)] // The fixed integration suites use different capture operations.
use pyo3::{prelude::*, types::PyDict};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output},
};

const CAPTURE: &str = "CHELIS_NATIVE_EXECUTION_CAPTURE";
const SUITE: &str = "CHELIS_NATIVE_EXECUTION_SUITE";

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn safe_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_:".contains(&c))
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .and_then(|mut file| file.write_all(bytes))
        .map_err(|error| error.to_string())
}
fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    write_new(
        path,
        &serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?,
    )
}

pub struct ArtifactDirectory {
    path: PathBuf,
    temporary: Option<tempfile::TempDir>,
    identity: Option<(String, String)>,
}

impl ArtifactDirectory {
    pub fn new() -> Result<Self, String> {
        match (std::env::var_os(CAPTURE), std::env::var_os(SUITE)) {
            (None, None) => Self::temporary(),
            (Some(root), Some(suite)) => {
                let suite = suite.to_str().ok_or("capture suite is not UTF-8")?;
                let thread = std::thread::current();
                let name = thread
                    .name()
                    .ok_or("capture requires the actual libtest case name")?;
                Self::for_capture(Path::new(&root), suite, name)
            }
            _ => Err("incomplete native capture environment".into()),
        }
    }

    pub fn temporary() -> Result<Self, String> {
        let temporary = tempfile::tempdir().map_err(|error| error.to_string())?;
        Ok(Self {
            path: temporary.path().to_owned(),
            temporary: Some(temporary),
            identity: None,
        })
    }

    pub fn for_capture(root: &Path, suite: &str, test: &str) -> Result<Self, String> {
        if !root.is_absolute() || !safe_name(suite) || !safe_name(test) {
            return Err("capture requires absolute root and exact case identities".into());
        }
        let root = root.canonicalize().map_err(|error| error.to_string())?;
        let cases = root.join("fixtures").join(test);
        fs::create_dir_all(&cases).map_err(|error| error.to_string())?;
        let temporary = tempfile::Builder::new()
            .prefix("instance-")
            .tempdir_in(cases)
            .map_err(|error| error.to_string())?;
        // Preserve incomplete attempts too. Only finish() writes completion.
        let path = temporary.keep();
        Ok(Self {
            path,
            temporary: None,
            identity: Some((suite.into(), test.into())),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn install<'py>(&self, py: Python<'py>, globals: &Bound<'py, PyDict>) -> PyResult<()> {
        globals.set_item(
            "artifact_root",
            self.path.to_str().expect("UTF-8 artifact path"),
        )?;
        globals.set_item("native_capture_enabled", self.identity.is_some())?;
        py.run(c"
import hashlib as _capture_hashlib
import json as _capture_json
from pathlib import Path as _CapturePath

def _capture_native_model(model):
    if not native_capture_enabled:
        return model
    library = _CapturePath(model.path)
    assert library.is_absolute() and library.is_file() and not library.is_symlink()
    original = library.parent
    destination = _CapturePath(artifact_root) / 'loaded-artifacts'
    assert not destination.exists()
    # Enumerate before creating destination: foreign fixtures live in artifact_root.
    sources = sorted(original.rglob('*'))
    assert all(not source.is_symlink() for source in sources)
    sources = [source for source in sources if source.is_file()]
    destination.mkdir()
    records = []
    for source in sources:
        relative = source.relative_to(original)
        captured = destination / relative
        captured.parent.mkdir(parents=True, exist_ok=True)
        before = source.read_bytes()
        with captured.open('xb') as output:
            output.write(before)
        digest = _capture_hashlib.sha256(before).hexdigest()
        assert _capture_hashlib.sha256(captured.read_bytes()).hexdigest() == digest
        assert _capture_hashlib.sha256(source.read_bytes()).hexdigest() == digest
        records.append(dict(original=str(source), captured=str(captured.relative_to(artifact_root)), sha256=digest))
    packet = dict(path=str(library), root=str(original), snapshot='loaded-artifacts',
                  library=str((destination / library.name).relative_to(artifact_root)), files=records)
    with (_CapturePath(artifact_root) / 'loaded-model.json').open('x') as output:
        _capture_json.dump(packet, output, sort_keys=True)
    return model
", Some(globals), None)
    }

    pub fn command_output(&self, name: &str, command: &mut Command) -> Result<Output, String> {
        let output = command.output().map_err(|error| error.to_string())?;
        if self.identity.is_some() {
            record_process(&self.path, name, command, &output)?;
        }
        Ok(output)
    }

    pub fn finish(&self) -> Result<(), String> {
        let Some((suite, test)) = &self.identity else {
            return Ok(());
        };
        let model: Value = serde_json::from_slice(
            &fs::read(self.path.join("loaded-model.json")).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let relative = model["library"]
            .as_str()
            .ok_or("missing captured model library")?;
        let library = self.path.join(relative);
        let canonical = library.canonicalize().map_err(|error| error.to_string())?;
        if !canonical.starts_with(&self.path) || library.is_symlink() || !library.is_file() {
            return Err("captured library is foreign or missing".into());
        }
        let files = inventory(&self.path)?;
        write_json(
            &self.path.join("completion.json"),
            &json!({
                "schema":1, "suite":suite, "test":test, "completed":true,
                "library":library, "files":files,
            }),
        )
    }
}

fn inventory(root: &Path) -> Result<Vec<Value>, String> {
    fn visit(root: &Path, directory: &Path, result: &mut Vec<Value>) -> Result<(), String> {
        for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
            let path = entry.map_err(|error| error.to_string())?.path();
            let kind = fs::symlink_metadata(&path)
                .map_err(|error| error.to_string())?
                .file_type();
            if kind.is_symlink() {
                return Err("capture cannot retain symlinks".into());
            }
            if kind.is_dir() {
                visit(root, &path, result)?;
            } else if kind.is_file() {
                let bytes = fs::read(&path).map_err(|error| error.to_string())?;
                result
                    .push(json!({"path":path.strip_prefix(root).unwrap(), "sha256":hash(&bytes)}));
            } else {
                return Err("capture requires regular files".into());
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, root, &mut files)?;
    files.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    Ok(files)
}

fn record_process(
    root: &Path,
    name: &str,
    command: &Command,
    output: &Output,
) -> Result<(), String> {
    if !safe_name(name) {
        return Err("invalid process record name".into());
    }
    let arguments = std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|value| value.to_str().ok_or("non-UTF-8 process argument"))
        .collect::<Result<Vec<_>, _>>()?;
    let cwd = command
        .get_current_dir()
        .map(Path::to_owned)
        .unwrap_or(std::env::current_dir().map_err(|error| error.to_string())?);
    let mut record = json!({"command":arguments,"cwd":cwd,"returncode":output.status.code()});
    for (stream, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        let path = root.join(format!("{name}.{stream}.log"));
        write_new(&path, bytes)?;
        record[stream] = json!({"path":path,"sha256":hash(bytes)});
    }
    write_json(&root.join(format!("{name}.json")), &record)
}

pub fn worker_output(test: &str, command: &mut Command) -> Result<Output, String> {
    let output = command.output().map_err(|error| error.to_string())?;
    if let Some(root) = std::env::var_os(CAPTURE) {
        if !safe_name(test) {
            return Err("invalid worker case identity".into());
        }
        let path = PathBuf::from(root).join("children").join(test);
        fs::create_dir_all(&path).map_err(|error| error.to_string())?;
        record_process(&path, "worker", command, &output)?;
    }
    Ok(output)
}
