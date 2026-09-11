//! Execution of the separate device owner with the real checked Rust metadata
//! archive and an explicit CPU HIP allocation/copy fixture. This is not a GPU test.
use std::{env, fs, path::PathBuf, process::Command, sync::OnceLock};

struct Fixture {
    _directory: tempfile::TempDir,
    binary: PathBuf,
}
fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let runtime = env::var_os("CHELIS_RUNTIME_LIB").map(PathBuf::from)
            .expect("prerequisite: CHELIS_RUNTIME_LIB must name the exact-head runtime archive in the owned target; this test never starts a nested Cargo build");
        assert!(runtime.is_absolute() && runtime.is_file(), "missing exact-head runtime archive: {}", runtime.display());
        let directory = tempfile::tempdir().unwrap();
        let binary = directory.path().join("device-owner-contract");
        let object = directory.path().join("device-owner.o");
        let compiled = Command::new(env::var_os("CXX").unwrap_or_else(|| "c++".into()))
            .args(["-std=c++17", "-O1", "-g", "-Dchelis_metadata_plan_release=fixture_metadata_plan_release", "-I"])
            .arg(root.join("tests/fixtures/device_owner_sdk"))
            .arg("-I").arg(root.join("runtime"))
            .arg("-I").arg(root.join("../chelis-runtime/include"))
            .arg("-c").arg(root.join("runtime/chelis_device_owner.cpp"))
            .arg("-o").arg(&object).output().expect("start companion compiler");
        assert!(compiled.status.success(), "companion compile failed: {}", String::from_utf8_lossy(&compiled.stderr));
        let mut command = Command::new(env::var_os("CXX").unwrap_or_else(|| "c++".into()));
        command.args(["-std=c++17", "-O1", "-g"])
            .arg("-I").arg(root.join("tests/fixtures/device_owner_sdk"))
            .arg("-I").arg(root.join("runtime"))
            .arg("-I").arg(root.join("../chelis-runtime/include"))
            .arg(&object)
            .arg(root.join("tests/fixtures/device_owner_contract.cpp"))
            .arg(runtime).args(["-lpthread", "-lm"]);
        if cfg!(target_os = "linux") { command.arg("-ldl"); }
        if cfg!(target_os = "macos") { command.arg("-liconv"); }
        let output = command.arg("-o").arg(&binary).output().expect("start C++ fixture compiler");
        assert!(output.status.success(), "device owner fixture compile/link failed:\n{}", String::from_utf8_lossy(&output.stderr));
        Fixture { _directory: directory, binary }
    })
}

#[test]
fn checked_owner_materializes_exact_strided_bits_and_releases_only_owned_storage() {
    for case in [
        "logical-order",
        "dynamic-rank",
        "explicit-borrow",
        "device-context",
    ] {
        let output = Command::new(&fixture().binary).arg(case).output().unwrap();
        assert!(
            output.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn malformed_packet_capacity_and_transfer_requests_trap_before_copy() {
    for case in [
        "reserved",
        "dtype",
        "owned-import",
        "count",
        "capacity",
        "negative-stride",
        "rank",
        "null-shape",
        "null-data",
        "borrow-write",
        "dtype-transfer",
        "count-transfer",
        "borrow-capacity",
        "gapped-allocation",
        "empty-clone-overflow",
        "pointer-device",
        "context-view",
        "context-clone",
        "context-transfer",
    ] {
        let output = Command::new(&fixture().binary).arg(case).output().unwrap();
        assert!(!output.status.success(), "{case} returned success");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let class = if case == "empty-clone-overflow" {
            "overflow"
        } else {
            "domain"
        };
        assert!(
            stderr.contains(&format!("numeric trap: {class} in metadata_plan at int64")),
            "{case}: {stderr}"
        );
        assert!(
            !stderr.contains("Assertion"),
            "{case} reached fixture memory access before rejection: {stderr}"
        );
    }
}

#[test]
fn published_owner_is_opaque_and_packet_observation_cannot_be_mutated() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = tempfile::tempdir().unwrap();
    for (name, body, success) in [
        (
            "opaque-use",
            "void probe(chelis_device_tensor_owner *owner) { const chelis_gpu_tensor *view = chelis_device_tensor_view(owner); (void)view; }",
            true,
        ),
        (
            "owner-construction",
            "void probe() { chelis_device_tensor_owner owner{}; }",
            false,
        ),
        (
            "owner-field",
            "void probe(chelis_device_tensor_owner *owner) { (void)owner->plan; }",
            false,
        ),
        (
            "packet-write",
            "void probe(chelis_device_tensor_owner *owner) { chelis_device_tensor_view(owner)->count = 7; }",
            false,
        ),
        (
            "packet-finalizer",
            "void probe(chelis_device_tensor_owner *owner) { chelis_device_tensor_release(chelis_device_tensor_view(owner)); }",
            false,
        ),
    ] {
        let source = directory.path().join(format!("{name}.cpp"));
        fs::write(
            &source,
            format!("#include \"chelis_device_owner.h\"\n{body}\n"),
        )
        .unwrap();
        let output = Command::new(env::var_os("CXX").unwrap_or_else(|| "c++".into()))
            .args(["-std=c++17", "-fsyntax-only", "-I"])
            .arg(root.join("runtime"))
            .arg("-I")
            .arg(root.join("../chelis-runtime/include"))
            .arg(source)
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            success,
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
