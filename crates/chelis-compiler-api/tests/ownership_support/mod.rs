//! Compile the real selected C artifact against an exactly identified ledger runtime.
#![allow(dead_code)]

use chelis_compiler_api::compiler::{compile, compile_for_execution};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
use serde_json::Value;
use std::fmt;
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[derive(Clone)]
pub struct GeneratedProgram {
    source: String,
    header: String,
    declarations: chelis_backend_c::GeneratedHeader,
    validate_before_compile: bool,
}

impl GeneratedProgram {
    pub fn new(source: String, header: String) -> Self {
        let declarations = chelis_backend_c::GeneratedHeader::parse(&header)
            .expect("compiler API must return generated declaration metadata");
        Self {
            source,
            header,
            declarations,
            validate_before_compile: true,
        }
    }

    pub fn from_codegen(artifact: &chelis_backend_c::CodegenResult) -> Self {
        Self::new(artifact.c_source.clone(), artifact.h_header.clone())
    }

    pub fn symbol(&self, source_name: &str) -> &str {
        self.declarations
            .declaration(source_name)
            .unwrap_or_else(|| panic!("generated header has no declaration for `{source_name}`"))
            .symbol()
    }

    #[allow(dead_code)]
    pub fn declaration(&self, source_name: &str) -> &str {
        self.declarations
            .declaration(source_name)
            .unwrap_or_else(|| panic!("generated header has no declaration for `{source_name}`"))
            .declaration()
    }

    #[allow(dead_code)]
    pub fn definition_digest(&self, source_name: &str) -> &str {
        self.declarations
            .declaration(source_name)
            .unwrap_or_else(|| panic!("generated header has no declaration for `{source_name}`"))
            .definition_digest()
    }

    pub fn header(&self) -> &str {
        &self.header
    }

    pub fn with_source(&self, source: String) -> Self {
        Self {
            source,
            header: self.header.clone(),
            declarations: self.declarations.clone(),
            validate_before_compile: false,
        }
    }

    #[allow(dead_code)]
    pub fn with_header(
        &self,
        header: String,
    ) -> Result<Self, chelis_backend_c::GeneratedHeaderError> {
        let declarations = chelis_backend_c::GeneratedHeader::parse(&header)?;
        Ok(Self {
            source: self.source.clone(),
            header,
            declarations,
            validate_before_compile: true,
        })
    }

    #[allow(dead_code)]
    pub fn validate(&self) -> Result<(), chelis_backend_c::GeneratedHeaderError> {
        self.declarations.validate_source(&self.source)
    }
}

impl Deref for GeneratedProgram {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl fmt::Display for GeneratedProgram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.source)
    }
}

fn runtime() -> &'static Path {
    static ARCHIVE: OnceLock<PathBuf> = OnceLock::new();
    ARCHIVE.get_or_init(|| {
        // A managed build puts its observed Cargo launcher first on PATH;
        // env!("CARGO") names the real binary, which cannot build a producer.
        let output = Command::new("cargo")
            .current_dir(root())
            .env(
                "CARGO_TARGET_DIR",
                root().join("target/ownership-ledger-runtime"),
            )
            .args([
                "build",
                "--locked",
                "-p",
                "chelis-runtime",
                "--lib",
                "--features",
                "ownership-ledger",
                "--message-format=json",
            ])
            .output()
            .expect("build ledger runtime");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let rows: Vec<Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|row| {
                row["reason"] == "compiler-artifact" && row["target"]["name"] == "chelis_runtime"
            })
            .collect();
        assert_eq!(rows.len(), 1, "exact runtime artifact required");
        assert!(
            rows[0]["features"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == "ownership-ledger")
        );
        let paths: Vec<_> = rows[0]["filenames"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .filter(|p| p.ends_with(".a"))
            .collect();
        assert_eq!(paths.len(), 1);
        PathBuf::from(paths[0])
    })
}

pub fn emit(source: &str, entry: &str) -> GeneratedProgram {
    // Keep otherwise DAG-only constant functions in an authored host module.
    // This additional function is emitted, never invoked by the C driver.
    let host_anchor = "\ndef host_loss(t: tensor[1, f32]) -> f32 = tensor_to_scalar(sum(t, 0))\ndef host_grad(t: tensor[1, f32]) -> tensor[1, f32] = grad(host_loss)(t)\n";
    let artifact = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: format!("{source}{host_anchor}"),
        target: CompileTarget::C,
        entry_name: Some("fixture".into()),
    })
    .unwrap_or_else(|e| panic!("{entry}: {e:?}"));
    let source = artifact
        .files
        .iter()
        .find(|f| f.path == "fixture.c")
        .expect("generated C file")
        .contents
        .clone();
    let header = artifact
        .files
        .iter()
        .find(|f| f.path == "fixture.h")
        .expect("generated C header")
        .contents
        .clone();
    GeneratedProgram::new(source, header)
}

pub fn emit_selected(source: &str, entry: &str) -> GeneratedProgram {
    let artifact = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some(entry.into()),
    })
    .unwrap_or_else(|e| panic!("{entry}: {e:?}"));
    let generated_path = format!("{}.c", artifact.compile_result.entry_name);
    let source = artifact
        .compile_result
        .files
        .iter()
        .find(|f| f.path == generated_path)
        .expect("selected C file")
        .contents
        .clone();
    let generated_path = format!("{}.h", artifact.compile_result.entry_name);
    let header = artifact
        .compile_result
        .files
        .iter()
        .find(|f| f.path == generated_path)
        .expect("selected C header")
        .contents
        .clone();
    GeneratedProgram::new(source, header)
}

pub fn run(source: &GeneratedProgram, driver: &str) -> Value {
    run_with_peers(source, &[], driver)
}

#[allow(dead_code)]
pub fn run_program(source: &GeneratedProgram) -> (Value, String) {
    execute_program(source, &[], "")
}

pub fn run_with_peers(source: &GeneratedProgram, peers: &[String], driver: &str) -> Value {
    execute_program(source, peers, driver).0
}

#[allow(dead_code)]
pub fn run_with_stdout(source: &GeneratedProgram, driver: &str) -> (Value, String) {
    execute_program(source, &[], driver)
}

#[allow(dead_code)]
pub fn run_expect_failure(source: &GeneratedProgram, driver: &str) {
    let (_dir, binary) = compile_program(source, &[], driver);
    let output = Command::new(binary).output().expect("execute failing C");
    assert!(
        !output.status.success(),
        "observer sink failure was silent: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn execute_program(source: &GeneratedProgram, peers: &[String], driver: &str) -> (Value, String) {
    let (_dir, binary) = compile_program(source, peers, driver);
    let ledger = binary.with_file_name("ledger.jsonl");
    let output = Command::new(binary)
        .env("CHELIS_OWNERSHIP_LEDGER_PATH", &ledger)
        .output()
        .expect("execute C");
    assert!(
        output.status.success(),
        "C status {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<Value> = fs::read_to_string(ledger)
        .expect("ledger required")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows[0]["schema"], "compiled-value-ownership-ledger-v1");
    assert_eq!(rows.iter().filter(|r| r["event"] == "summary").count(), 1);
    let summary = rows.last().expect("summary row").clone();
    assert_eq!(summary["event"], "summary");
    assert_eq!(summary["invalid_operations"], 0, "{summary}");
    (
        summary,
        String::from_utf8(output.stdout).expect("utf-8 stdout"),
    )
}

fn compile_program(
    source: &GeneratedProgram,
    peers: &[String],
    driver: &str,
) -> (tempfile::TempDir, PathBuf) {
    if source.validate_before_compile {
        source
            .validate()
            .expect("generated header and source must agree before native compilation");
    }
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("generated.h"), source.header()).unwrap();
    let c = dir.path().join("probe.c");
    fs::write(
        &c,
        format!(
            "#include \"chelis_runtime.h\"\n#include \"generated.h\"\n{}\n{PRELUDE}\n{driver}",
            &**source
        ),
    )
    .unwrap();
    let binary = dir.path().join("probe");
    let mut cc = Command::new("cc");
    cc.args(["-std=c11", "-O0"])
        .arg(&c)
        .arg("-I")
        .arg(root().join("crates/chelis-runtime/include"))
        .arg(runtime())
        .args(["-lm", "-lpthread"]);
    for (index, source) in peers.iter().enumerate() {
        let peer = dir.path().join(format!("peer_{index}.c"));
        fs::write(&peer, source).unwrap();
        cc.arg(peer);
    }
    if cfg!(target_os = "macos") {
        cc.args([
            "-framework",
            "Accelerate",
            "-framework",
            "Security",
            "-framework",
            "CoreFoundation",
        ]);
    } else {
        cc.arg("-ldl");
    }
    let output = cc.arg("-o").arg(&binary).output().expect("compile C");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (dir, binary)
}

pub fn balanced(summary: &Value) {
    assert_eq!(summary["live_owners"], 0, "{summary}");
    assert_eq!(summary["live_bytes"], 0, "{summary}");
    assert_eq!(summary["allocations"], summary["finalized"], "{summary}");
    assert!(summary["allocations"].as_u64().unwrap() > 0);
}

const PRELUDE: &str = r#"
#include <assert.h>
#include <inttypes.h>
#include <string.h>
static chelis_tensor *input(int64_t n) {
    chelis_tensor *x = chelis_alloc(1, &n, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    chelis_write_view view = chelis_tensor_write_view(guard);
    for (int64_t i = 0; i < n; ++i) ((float *)view.data)[i] = (float)(2*i-3);
    chelis_tensor_end_write(guard);
    return x;
}
static void tensor_bits(const chelis_tensor *x, int64_t n, const float *expected) {
    assert(chelis_tensor_rank(x) == 1);
    assert(chelis_tensor_shape(x, 0) == n);
    assert(chelis_tensor_numel(x) == n);
    chelis_read_view view = chelis_tensor_read_view(x);
    assert(view.dtype == CHELIS_DTYPE_F32 && view.count == n);
    for (int64_t i = 0; i < n; ++i) {
        assert(memcmp((const float *)view.data+i, expected+i, sizeof(float)) == 0);
    }
}
"#;
