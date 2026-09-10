//! Consumer compile controls over the actual std+vocab ABI owner.

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Probe(PathBuf);

impl Drop for Probe {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove own compiler probe");
    }
}

impl Probe {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "chelis-abi-privacy-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let probe = Self(path);
        let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for (name, source) in [
            (
                "chelis_vocab",
                crate_root.join("../chelis-vocab/src/lib.rs"),
            ),
            ("chelis_abi", crate_root.join("src/lib.rs")),
        ] {
            let mut command = Command::new("rustc");
            command.args(["--edition=2024", "--crate-type=rlib", "--crate-name", name]);
            command.arg(source).arg("--out-dir").arg(&probe.0);
            if name == "chelis_abi" {
                command.arg("--extern").arg(format!(
                    "chelis_vocab={}",
                    probe.0.join("libchelis_vocab.rlib").display()
                ));
            }
            let result = command
                .output()
                .expect("compile actual dependency-bottom owner");
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
        probe
    }

    fn compile(&self, source: &str) -> Output {
        let input = self.0.join("consumer.rs");
        std::fs::write(&input, source).unwrap();
        Command::new("rustc")
            .args([
                "--edition=2024",
                "--crate-type=rlib",
                "--crate-name",
                "consumer",
            ])
            .arg(input)
            .arg("--out-dir")
            .arg(&self.0)
            .arg("-L")
            .arg(format!("dependency={}", self.0.display()))
            .arg("--extern")
            .arg(format!(
                "chelis_abi={}",
                self.0.join("libchelis_abi.rlib").display()
            ))
            .arg("--extern")
            .arg(format!(
                "chelis_vocab={}",
                self.0.join("libchelis_vocab.rlib").display()
            ))
            .output()
            .expect("compile actual consumer")
    }

    fn reject_after_positive(&self, negative: &str, code: &str, witness: &str) {
        let imports = "use chelis_abi::metadata::{ShapeMetadata,ElementCount,ByteCount,AllocationBytes}; use chelis_vocab::RuntimeDType;";
        let positive = self.compile(&format!(
            "{imports} fn valid() {{ let m=ShapeMetadata::contiguous(&[2,3],RuntimeDType::F64).unwrap(); let _:i64=m.elements().get(); let _:usize=m.bytes().allocation().unwrap().get(); }}"
        ));
        assert!(
            positive.status.success(),
            "positive prerequisite: {}",
            String::from_utf8_lossy(&positive.stderr)
        );
        let result = self.compile(&format!("{imports}\n{negative}"));
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success() && stderr.contains(code) && stderr.contains(witness),
            "wrong privacy rejection: {stderr}"
        );
    }
}

#[test]
fn consumers_cannot_construct_unchecked_count_or_capacity() {
    let probe = Probe::new();
    for ty in ["ElementCount", "ByteCount", "AllocationBytes"] {
        probe.reject_after_positive(&format!("fn invalid() {{ let _={ty}(1); }}"), "E0423", ty);
    }
}

#[test]
fn consumers_cannot_replace_checked_metadata_fields() {
    let probe = Probe::new();
    for field in ["shape", "strides", "rank", "elements", "bytes", "dtype"] {
        probe.reject_after_positive(
            &format!("fn invalid(m:&mut ShapeMetadata) {{ let _=&mut m.{field}; }}"),
            "E0616",
            field,
        );
    }
}

#[test]
fn observed_shape_is_not_a_mutable_metadata_escape() {
    Probe::new().reject_after_positive(
        "fn invalid(m:&mut ShapeMetadata) { m.shape()[0]=99; }",
        "E0594",
        "&",
    );
}
