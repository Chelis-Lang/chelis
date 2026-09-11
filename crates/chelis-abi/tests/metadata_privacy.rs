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
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace.join("target"));
        let target = if target.is_absolute() {
            target
        } else {
            workspace.join(target)
        };
        let parent = target.join("metadata-privacy-probes");
        std::fs::create_dir_all(&parent).unwrap();
        let path = parent.join(format!(
            "{}-{}",
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

    fn compile_plan_contract(&self, owner: &str) -> PathBuf {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tests = std::fs::read_to_string(root.join("tests/metadata_plans.rs")).unwrap();
        let marker = "use chelis_abi::metadata::{";
        assert_eq!(tests.matches(marker).count(), 1);
        // The contract is unchanged; only its import selects the actual
        // source under mutation instead of the already-built external crate.
        let source = tests.replacen(marker, "mod metadata;\nuse metadata::{", 1);
        std::fs::write(self.0.join("metadata.rs"), owner).unwrap();
        let input = self.0.join("plan_contract.rs");
        std::fs::write(&input, source).unwrap();
        let binary = self.0.join("plan_contract");
        let result = Command::new("rustc")
            .args([
                "--edition=2024",
                "--test",
                "-C",
                "opt-level=2",
                "-C",
                "overflow-checks=off",
                "-C",
                "debug-assertions=off",
            ])
            .arg(input)
            .arg("-o")
            .arg(&binary)
            .arg("--extern")
            .arg(format!(
                "chelis_vocab={}",
                self.0.join("libchelis_vocab.rlib").display()
            ))
            .output()
            .expect("compile actual plan owner and contract");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        binary
    }
}

#[test]
fn transferred_plan_fields_remain_private_to_the_checked_owner() {
    let probe = Probe::new();
    let imports = "use chelis_abi::metadata::{ShapeMetadata,MovementMetadata,WindowMetadata,MatmulMetadata}; use chelis_vocab::RuntimeDType;";
    let positive = probe.compile(&format!(
        "{imports} fn valid() {{ let m=ShapeMetadata::contiguous(&[2,2],RuntimeDType::F32).unwrap(); let _=MovementMetadata::permuted(&m,&[1,0]).unwrap().index(0).unwrap(); let _=WindowMetadata::new(&m,&[1],&[1]).unwrap().index(0,0).unwrap(); let _=MatmulMetadata::new(&m,&m,RuntimeDType::F32).unwrap().batches().get(); }}"
    ));
    assert!(
        positive.status.success(),
        "positive prerequisite: {}",
        String::from_utf8_lossy(&positive.stderr)
    );
    for (owner, fields) in [
        (
            "MovementMetadata",
            &["input", "result", "projection", "source_domain"][..],
        ),
        (
            "WindowMetadata",
            &["input", "result", "window", "steps", "leading", "count"][..],
        ),
        (
            "MatmulMetadata",
            &["result", "dimensions", "matrices", "totals", "batches"][..],
        ),
    ] {
        for field in fields {
            let result = probe.compile(&format!(
                "{imports} fn bad(m:&mut {owner}) {{ let _=&mut m.{field}; }}"
            ));
            let stderr = String::from_utf8_lossy(&result.stderr);
            assert!(
                !result.status.success() && stderr.contains("E0616") && stderr.contains(field),
                "wrong {owner}.{field} rejection: {stderr}"
            );
        }
    }
    let result = probe.compile("use chelis_abi::metadata::AxisProjection;");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !result.status.success() && stderr.contains("E0603") && stderr.contains("AxisProjection"),
        "projection owner must remain private: {stderr}"
    );
}

#[test]
fn weakened_transferred_plan_validation_fails_the_optimized_contract() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let owner = std::fs::read_to_string(root.join("src/metadata.rs")).unwrap();
    let probe = Probe::new();
    let controls = [
        (
            ".checked_mul(axis.step)",
            ".checked_mul(0)",
            "movement_plans_project_checked_domains_without_coordinate_scratch",
        ),
        (
            ".checked_add(axis.offset)",
            ".checked_add(0)",
            "movement_plans_project_checked_domains_without_coordinate_scratch",
        ),
        (
            "input.require_permutation(&result, axes)?;",
            "",
            "movement_plans_reject_bad_geometry_and_preserve_rank_zero_empty_and_int64",
        ),
        (
            ".checked_mul(self.steps[window_axis])",
            ".checked_mul(0)",
            "window_metadata_binds_valid_padding_and_row_major_source_indices",
        ),
        (
            "w > input.shape()[axis]",
            "false",
            "window_metadata_binds_valid_padding_and_row_major_source_indices",
        ),
        (
            ".checked_mul(matrix)",
            ".checked_mul(0)",
            "matmul_metadata_binds_matrix_spans_and_vendor_projection_without_storage",
        ),
        (
            "self.dimensions.iter().any(|&extent| extent > limit)",
            "false",
            "matmul_metadata_binds_matrix_spans_and_vendor_projection_without_storage",
        ),
    ];
    let execute = |binary: &PathBuf, witness: &str, success: bool| {
        let result = Command::new(binary)
            .args(["--exact", witness, "--nocapture"])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            stdout.contains("running 1 test"),
            "missing exact witness: {stdout}"
        );
        assert_eq!(
            result.status.success(),
            success,
            "{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            stdout.contains(if success { "1 passed" } else { "1 failed" }),
            "missing expected outcome: {stdout}"
        );
    };
    let binary = probe.compile_plan_contract(&owner);
    for (_, _, witness) in controls {
        execute(&binary, witness, true);
    }
    for (from, to, witness) in controls {
        assert_eq!(
            owner.matches(from).count(),
            1,
            "unique production mutation: {from}"
        );
        let binary = probe.compile_plan_contract(&owner.replacen(from, to, 1));
        execute(&binary, witness, false);
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
    for field in ["domain", "strides"] {
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
    let imports = "use chelis_abi::metadata::{ByteCount, ShapeMetadata, StridedMetadata}; use chelis_vocab::RuntimeDType;";
    let positive = probe.compile(&format!(
        "{imports} fn valid() {{ let v=StridedMetadata::new(&[2,3], &[0,1], RuntimeDType::F64, ByteCount::from_declared(24).unwrap()).unwrap(); assert_eq!(v.strides(), &[0,1]); }}"
    ));
    assert!(
        positive.status.success(),
        "{}",
        String::from_utf8_lossy(&positive.stderr)
    );
    for (body, code) in [
        (
            "fn invalid(v:&mut StridedMetadata) { v.strides()[0]=1; }",
            "E0594",
        ),
        (
            "fn invalid(v:&mut StridedMetadata) { v.shape()[0]=1; }",
            "E0594",
        ),
        (
            "fn host(_: &ShapeMetadata) {} fn invalid(v:&StridedMetadata) { host(v); }",
            "E0308",
        ),
        (
            "fn invalid(v:&mut StridedMetadata) { let _=&mut v.domain; }",
            "E0616",
        ),
        (
            "fn invalid(v:&mut StridedMetadata) { let _=&mut v.strides; }",
            "E0616",
        ),
        (
            "fn invalid(v:&mut StridedMetadata) { let _=&mut v.required_span; }",
            "E0616",
        ),
    ] {
        let result = probe.compile(&format!("{imports} {body}"));
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success() && stderr.contains(code),
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
