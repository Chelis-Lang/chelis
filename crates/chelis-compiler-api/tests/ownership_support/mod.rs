//! Compile the real selected C artifact against an exactly identified ledger runtime.

use chelis_compiler_api::compiler::{compile, compile_for_execution};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn runtime() -> &'static Path {
    static ARCHIVE: OnceLock<PathBuf> = OnceLock::new();
    ARCHIVE.get_or_init(|| {
        let output = Command::new(env!("CARGO"))
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

pub fn emit(source: &str, entry: &str) -> String {
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
    artifact
        .files
        .into_iter()
        .find(|f| f.path == "fixture.c")
        .expect("generated C file")
        .contents
}

pub fn emit_selected(source: &str, entry: &str) -> String {
    let artifact = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some(entry.into()),
    })
    .unwrap_or_else(|e| panic!("{entry}: {e:?}"));
    artifact
        .compile_result
        .files
        .into_iter()
        .find(|f| f.path == format!("{entry}.c"))
        .expect("selected C file")
        .contents
}

pub fn run(source: &str, driver: &str) -> Value {
    run_with_peers(source, &[], driver)
}

pub fn run_with_peers(source: &str, peers: &[String], driver: &str) -> Value {
    let dir = tempfile::tempdir().unwrap();
    let c = dir.path().join("probe.c");
    fs::write(&c, format!("{source}\n{PRELUDE}\n{driver}")).unwrap();
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
    let ledger = dir.path().join("ledger.jsonl");
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
    let summary = rows.last().unwrap().clone();
    assert_eq!(summary["event"], "summary");
    assert_eq!(summary["invalid_operations"], 0, "{summary}");
    summary
}

pub fn balanced(summary: &Value) {
    assert_eq!(summary["live_owners"], 0, "{summary}");
    assert_eq!(summary["live_bytes"], 0, "{summary}");
    assert_eq!(summary["allocations"], summary["finalized"], "{summary}");
    assert!(summary["allocations"].as_u64().unwrap() > 0);
}

const PRELUDE: &str = r#"
#include <assert.h>
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
