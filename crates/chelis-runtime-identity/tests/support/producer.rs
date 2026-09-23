use chelis_runtime_identity::{
    BuildProvenance, Descriptor, RecordKind, decode_archive, decode_archive_provenance,
    decode_image, decode_image_provenance, hash_bytes,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub struct Fixture {
    pub root: PathBuf,
    pub source: PathBuf,
    pub cargo: OsString,
    _temporary: tempfile::TempDir,
}
#[derive(Clone, Debug)]
pub struct Product {
    pub profile: Value,
    pub descriptor: Descriptor,
    pub bytes: Vec<u8>,
    pub provenance: BuildProvenance,
    pub runtime_archives: Vec<PathBuf>,
}

pub fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if [
            ".git",
            "target",
            ".work",
            ".worktrees",
            ".venv",
            ".devenv",
            "node_modules",
            "__pycache__",
        ]
        .iter()
        .any(|excluded| name == *excluded)
        {
            continue;
        }
        let source = entry.path();
        let destination = to.join(name);
        if entry.file_type().unwrap().is_symlink() {
            #[cfg(unix)]
            std::os::unix::fs::symlink(fs::read_link(&source).unwrap(), &destination).unwrap();
            #[cfg(not(unix))]
            panic!("native fixture source copy requires Unix symlink semantics");
        } else if source.is_dir() {
            copy_tree(&source, &destination);
        } else if source.is_file() {
            fs::copy(&source, &destination).unwrap();
        }
    }
}

fn require(output: Output, command: &str) -> Output {
    assert!(
        output.status.success(),
        "mandatory real-build prerequisite or producer failed: {command}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

impl Fixture {
    pub fn new() -> Self {
        assert!(
            cfg!(any(target_os = "linux", target_os = "macos")),
            "native producer suite requires Linux or macOS"
        );
        let temporary = tempfile::Builder::new()
            .prefix("chelis-identity-contract-")
            .tempdir()
            .unwrap();
        let root = temporary.path().to_owned();
        let source = root.join("source");
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        copy_tree(repository, &source);
        let cargo = std::env::var_os("CHELIS_IDENTITY_REAL_CARGO")
            .or_else(|| std::env::var_os("CARGO"))
            .unwrap_or_else(|| "cargo".into());
        let version = require(
            Command::new(&cargo).arg("--version").output().unwrap(),
            "cargo --version",
        );
        eprintln!(
            "producer fixture: {}",
            json!({"source":source,"cargo":String::from_utf8_lossy(&version.stdout),"host":std::env::consts::OS,"arch":std::env::consts::ARCH})
        );
        Self {
            root,
            source,
            cargo,
            _temporary: temporary,
        }
    }

    pub fn assert_no_runtime_static_archive(&self) {
        let mut directories = vec![self.root.clone()];
        while let Some(directory) = directories.pop() {
            for entry in fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                assert!(
                    !(name == "libchelis_runtime.a"
                        || name.starts_with("libchelis_runtime-") && name.ends_with(".a")),
                    "archive-free producer witness contains {}",
                    entry.path().display()
                );
                if entry.file_type().unwrap().is_dir() {
                    directories.push(entry.path());
                }
            }
        }
    }

    pub fn edit_manifest(&self, relative: &str, edit: impl FnOnce(&mut toml::Value)) {
        let path = self.source.join(relative);
        let mut value: toml::Value = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        edit(&mut value);
        fs::write(path, toml::to_string(&value).unwrap()).unwrap();
    }

    pub fn install_input_fixture(&self) {
        let templates = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/producer");
        copy_tree(&templates, &self.source.join("identity-fixtures"));
        self.edit_manifest("Cargo.toml", |manifest| {
            let members = manifest["workspace"]["members"].as_array_mut().unwrap();
            for name in ["input", "leaf", "bridge"] {
                members.push(format!("identity-fixtures/{name}").into());
            }
        });
        self.edit_manifest("crates/chelis-runtime/Cargo.toml", |manifest| {
            manifest["dependencies"].as_table_mut().unwrap().insert(
                "fixture-runtime-input".into(),
                toml::Value::Table(toml::map::Map::from_iter([(
                    "path".into(),
                    "../../identity-fixtures/input".into(),
                )])),
            );
        });
        self.edit_manifest("crates/chelis-cli/Cargo.toml", |manifest| {
            manifest["dependencies"].as_table_mut().unwrap().insert(
                "fixture-consumer-bridge".into(),
                toml::Value::Table(toml::map::Map::from_iter([
                    ("path".into(), "../../identity-fixtures/bridge".into()),
                    ("optional".into(), true.into()),
                ])),
            );
            manifest["features"]
                .as_table_mut()
                .unwrap()
                .insert("identity-consumer-only".into(), toml::Value::Array(vec![]));
            manifest["features"].as_table_mut().unwrap().insert(
                "identity-unify".into(),
                toml::Value::Array(vec!["dep:fixture-consumer-bridge".into()]),
            );
        });
        let runtime = self.source.join("crates/chelis-runtime/src/lib.rs");
        let mut text = fs::read_to_string(&runtime).unwrap();
        text.push_str("\nconst _: u32 = fixture_runtime_input::VALUE;\n");
        fs::write(runtime, text).unwrap();
    }

    pub fn invoke(
        &self,
        target: &str,
        package: &str,
        extra: &[&str],
        variables: &[(&str, &str)],
        managed: bool,
    ) -> Output {
        let mut command = if managed {
            let mut command = Command::new(
                std::env::var_os("PYO3_PYTHON")
                    .expect("producer fixtures require the configured Python 3.11 interpreter"),
            );
            command
                .arg(self.source.join("scripts/runtime_identity_build.py"))
                .arg("--cargo")
                .arg(&self.cargo)
                .arg("--");
            command
        } else {
            Command::new(&self.cargo)
        };
        command.current_dir(&self.source).args([
            "build",
            "--message-format=json-render-diagnostics",
            "-p",
            package,
        ]);
        if package == "chelis-cli" {
            if !extra.contains(&"--bin") {
                command.args(["--bin", "chelis"]);
            }
            if !extra.contains(&"--no-default-features") {
                command.arg("--no-default-features");
            }
        } else if !extra.contains(&"--lib") {
            command.arg("--lib");
        }
        command
            .args(extra)
            .arg("--target-dir")
            .arg(self.root.join(target))
            .env("CARGO_TARGET_DIR", self.root.join(target))
            .env("CHELIS_IDENTITY_PROVENANCE", "source-worktree")
            .env("CHELIS_IDENTITY_WORKSPACE", &self.source)
            .env(
                "CHELIS_IDENTITY_STATE",
                self.root.join(format!("{target}-observations")),
            );
        if !managed {
            command
                .env_remove("RUSTC_WRAPPER")
                .env_remove("RUSTC_WORKSPACE_WRAPPER")
                .env_remove("CHELIS_IDENTITY_PROTOCOL")
                .env_remove("CHELIS_IDENTITY_BACKEND");
        }
        for (name, value) in variables {
            command.env(name, value);
        }
        eprintln!("producer command: {command:?}");
        command.output().unwrap()
    }

    pub fn build_standalone(
        &self,
        target: &str,
        package: &str,
        extra: &[&str],
        variables: &[(&str, &str)],
    ) -> Product {
        let output = require(
            self.invoke(target, package, extra, variables, true),
            package,
        );
        let messages: Vec<Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| {
                serde_json::from_str(line)
                    .unwrap_or_else(|error| panic!("driver corrupted Cargo JSON: {error}: {line}"))
            })
            .collect();
        let runtime_archives = messages
            .iter()
            .filter(|message| {
                message["reason"] == "compiler-artifact"
                    && message["target"]["name"] == "chelis_runtime"
            })
            .flat_map(|message| message["filenames"].as_array().unwrap())
            .map(|filename| PathBuf::from(filename.as_str().unwrap()))
            .filter(|path| path.extension().is_some_and(|extension| extension == "a"))
            .collect();
        let kind = match package {
            "chelis-runtime" => RecordKind::Runtime,
            "chelis-cli" => RecordKind::Cli,
            "chelis-python" => RecordKind::Python,
            _ => panic!("unknown producer"),
        };
        let target_name = match kind {
            RecordKind::Runtime => "chelis_runtime",
            RecordKind::Cli => "chelis",
            RecordKind::Python => "chelis_python",
        };
        let mut artifacts = Vec::new();
        let mut profile = None;
        for message in &messages {
            if message["reason"] != "compiler-artifact" || message["target"]["name"] != target_name
            {
                continue;
            }
            for filename in message["filenames"].as_array().unwrap() {
                let path = PathBuf::from(filename.as_str().unwrap());
                let matches = match kind {
                    RecordKind::Runtime => {
                        path.extension().is_some_and(|extension| extension == "a")
                    }
                    RecordKind::Cli => message["executable"].as_str() == path.to_str(),
                    RecordKind::Python => path
                        .extension()
                        .is_some_and(|extension| extension == "so" || extension == "dylib"),
                };
                if matches {
                    artifacts.push(path);
                    profile = Some(message["profile"].clone());
                }
            }
        }

        artifacts.sort();
        artifacts.dedup();
        assert_eq!(
            artifacts.len(),
            1,
            "expected one exact producer artifact in Cargo events, got {artifacts:?}"
        );
        let artifact = artifacts.remove(0);
        let bytes = fs::read(&artifact).unwrap();
        let descriptor = match kind {
            RecordKind::Runtime => decode_archive(&bytes),
            _ => decode_image(&bytes, kind),
        }
        .unwrap_or_else(|error| panic!("{}: {error:?}", artifact.display()));
        let provenance = provenance(&bytes, kind);
        eprintln!(
            "producer evidence: {}",
            json!({"package":package,"artifact":artifact,"artifact_sha256":hash_bytes(&bytes),"descriptor":descriptor,"provenance":provenance})
        );
        Product {
            profile: profile.unwrap(),
            descriptor,
            bytes,
            provenance,
            runtime_archives,
        }
    }
    pub fn build(
        &self,
        target: &str,
        package: &str,
        extra: &[&str],
        variables: &[(&str, &str)],
    ) -> Product {
        // Build equivalent package roots in independent targets: Cargo feature
        // unification is part of the observed recipe, not a fixture assumption.
        let mut roots = vec![
            "-p",
            "chelis-runtime",
            "-p",
            "chelis-cli",
            "-p",
            "chelis-python",
            "--lib",
            "--bin",
            "chelis",
            "--no-default-features",
        ];
        roots.extend_from_slice(extra);
        self.build_standalone(target, package, &roots, variables)
    }

    pub fn build_unified(
        &self,
        target: &str,
        package: &str,
        extra: &[&str],
        variables: &[(&str, &str)],
    ) -> Product {
        let mut features = vec!["--features", "chelis-cli/identity-unify"];
        features.extend_from_slice(extra);
        self.build(target, package, &features, variables)
    }

    pub fn append(&self, relative: &str, text: &str) {
        let path = self.source.join(relative);
        let mut source = fs::read_to_string(&path).unwrap();
        source.push_str(text);
        fs::write(path, source).unwrap();
    }
}

fn provenance(bytes: &[u8], kind: RecordKind) -> BuildProvenance {
    match kind {
        RecordKind::Runtime => decode_archive_provenance(bytes),
        _ => decode_image_provenance(bytes, kind),
    }
    .expect("one retained role-specific provenance record bound to the native identity")
}

pub fn dimensions(descriptor: &Descriptor) -> BTreeMap<&'static str, String> {
    BTreeMap::from([
        ("source", descriptor.source.to_string()),
        ("interface", descriptor.interface.to_string()),
        ("target", descriptor.target.to_string()),
        ("features", descriptor.features.to_string()),
        ("compile", descriptor.compile.to_string()),
        ("toolchain", descriptor.toolchain.to_string()),
    ])
}
