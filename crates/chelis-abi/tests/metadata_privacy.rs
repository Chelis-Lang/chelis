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

#[test]
fn strided_projection_cannot_mutate_metadata_or_enter_contiguous_indexing() {
    let probe = Probe::new();
    let imports = "use chelis_abi::metadata::{ByteCount,ShapeMetadata,StridedMetadata}; use chelis_vocab::RuntimeDType;";
    let positive = probe.compile(&format!(
        "{imports} fn valid() {{ let v=StridedMetadata::new(&[2,3],&[0,1],RuntimeDType::F64,ByteCount::from_declared(24).unwrap()).unwrap(); assert_eq!(v.strides(),&[0,1]); }}"
    ));
    assert!(
        positive.status.success(),
        "{}",
        String::from_utf8_lossy(&positive.stderr)
    );
    for (body, code, witness) in [
        (
            "fn invalid(v:&mut StridedMetadata) { v.strides()[0]=1; }",
            "E0594",
            "&",
        ),
        (
            "fn invalid(v:&mut StridedMetadata) { v.shape()[0]=1; }",
            "E0594",
            "&",
        ),
        (
            "fn host(_: &ShapeMetadata) {} fn invalid(v:&StridedMetadata) { host(v); }",
            "E0308",
            "ShapeMetadata",
        ),
    ] {
        let result = probe.compile(&format!("{imports} {body}"));
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success() && stderr.contains(code) && stderr.contains(witness),
            "{stderr}"
        );
    }
    for field in [
        "shape",
        "strides",
        "rank",
        "elements",
        "bytes",
        "dtype",
        "required_span",
    ] {
        let result = probe.compile(&format!(
            "{imports} fn invalid(v:&mut StridedMetadata) {{ let _=&mut v.{field}; }}"
        ));
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success() && stderr.contains("E0616") && stderr.contains(field),
            "{stderr}"
        );
    }
}

#[test]
fn generated_public_views_preserve_exact_transport_fields() {
    let probe = Probe::new();
    let result = probe.compile(
        r#"
mod owner {
    chelis_abi::define_read_view!(pub ReadView, public_fields);
    chelis_abi::define_write_view!(pub WriteView, public_fields);
}
fn transport() {
    let read = owner::ReadView {
        data: std::ptr::null(), count: 1_i64, dtype: 0_u8, reserved: [0_u8; 7],
    };
    let write = owner::WriteView {
        data: std::ptr::null_mut(), count: 1_i64, dtype: 0_u8, reserved: [0_u8; 7],
    };
    let _: *const std::ffi::c_void = read.data;
    let _: *mut std::ffi::c_void = write.data;
    let _: (i64, u8, [u8; 7]) = (read.count, read.dtype, read.reserved);
    let _: (i64, u8, [u8; 7]) = (write.count, write.dtype, write.reserved);
}
"#,
    );
    assert!(
        result.status.success(),
        "public OP31 projection prerequisite: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let result = probe.compile(
        r#"
chelis_abi::define_read_view!(ReadView, public_fields);
fn requires_owner(_: &chelis_abi::metadata::ShapeMetadata) {}
fn invalid(view: &ReadView) { requires_owner(view); }
"#,
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !result.status.success() && stderr.contains("E0308") && stderr.contains("ShapeMetadata"),
        "view must not mint metadata authority: {stderr}"
    );
}

#[test]
fn raw_device_packet_fields_are_private_to_the_invoking_module() {
    let probe = Probe::new();
    let owner = r#"
mod owner {
    chelis_abi::define_device_descriptor!(pub DeviceDescriptor);
    pub fn raw_packet() -> DeviceDescriptor {
        DeviceDescriptor {
            data: std::ptr::null_mut(), shape: std::ptr::null(),
            strides: std::ptr::null(), count: 1_i64, byte_capacity: 8_i64,
            rank: 0_i32, dtype: 0_u8, ownership: 0_u8, reserved: [0_u8; 2],
        }
    }
}
"#;
    let result = probe.compile(&format!(
        "{owner}\nfn valid() {{ let _ = owner::raw_packet(); }}"
    ));
    assert!(
        result.status.success(),
        "owning-module packet prerequisite: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    for field in [
        "data",
        "shape",
        "strides",
        "count",
        "byte_capacity",
        "rank",
        "dtype",
        "ownership",
        "reserved",
    ] {
        let result = probe.compile(&format!(
            "{owner}\nfn invalid(packet:&mut owner::DeviceDescriptor) {{ let _=&mut packet.{field}; }}"
        ));
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success() && stderr.contains("E0616") && stderr.contains(field),
            "packet field must remain owner-private: {stderr}"
        );
    }
}
