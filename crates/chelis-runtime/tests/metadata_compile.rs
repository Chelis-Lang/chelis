//! C2.2: compiler-internal callers cannot forge or independently mutate metadata.
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_PROBE: AtomicUsize = AtomicUsize::new(0);

struct Probe(PathBuf);

impl Probe {
    fn new() -> Self {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/metadata-compile-probes")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_PROBE.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(directory.parent().unwrap()).unwrap();
        fs::create_dir(&directory).unwrap();
        Self(directory)
    }

    fn compile(&self, name: &str, source: &str, external: bool) -> Output {
        let input = self.0.join(format!("{name}.rs"));
        fs::write(&input, source).unwrap();
        let mut command = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));
        command.args(["--edition=2024", "--crate-type=rlib", "--crate-name", name]);
        command.arg(input).arg("--out-dir").arg(&self.0);
        if external {
            command.arg("--extern").arg(format!(
                "chelis_vocab={}",
                self.0.join("libchelis_vocab.rlib").display()
            ));
        }
        command.output().expect("execute metadata compile control")
    }

    fn execute_contract(&self, owner: &str, tests: &str, name: &str) -> Output {
        fs::write(self.0.join("metadata.rs"), owner).unwrap();
        let input = self.0.join("contract.rs");
        fs::write(&input, tests).unwrap();
        let binary = self.0.join("contract");
        // Always execute mutations with release overflow behavior. A debug
        // arithmetic panic must not substitute for an explicit checked path.
        compiled(
            Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
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
                .arg(&input)
                .arg("-o")
                .arg(&binary)
                .arg("--extern")
                .arg(format!(
                    "chelis_vocab={}",
                    self.0.join("libchelis_vocab.rlib").display()
                ))
                .output()
                .expect("compile real metadata contract"),
        );
        let output = Command::new(binary)
            .args(["--exact", name, "--nocapture"])
            .output()
            .expect("execute exact metadata contract witness");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
            "missing execution receipt for {name}"
        );
        output
    }
}

#[test]
fn weakened_metadata_construction_fails_the_executable_contract() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let owner = fs::read_to_string(root.join("src/metadata.rs")).unwrap();
    let tests = fs::read_to_string(root.join("tests/checked_metadata.rs")).unwrap();
    let path = "#[path = \"../src/metadata.rs\"]";
    assert_eq!(tests.matches(path).count(), 1);
    let tests = tests.replacen(path, "#[path = \"metadata.rs\"]", 1);
    let probe = Probe::new();
    compiled(probe.compile(
        "chelis_vocab",
        &fs::read_to_string(root.join("../chelis-vocab/src/lib.rs")).unwrap(),
        false,
    ));
    for (from, to, witness) in [
        (
            ".checked_mul(self.steps[window_axis])",
            ".checked_mul(0)",
            "window_metadata_binds_valid_padding_and_row_major_source_indices",
        ),
        (
            "w > input.shape[axis]",
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
        (
            "linear / self.inner.get() % self.indices.elements().get()",
            "0",
            "sparse_metadata_binds_indices_to_exact_hyperplane_and_elementwise_domains",
        ),
        (
            "let coordinate = if axis == self.axis {",
            "let coordinate = if false {",
            "sparse_metadata_binds_indices_to_exact_hyperplane_and_elementwise_domains",
        ),
        (
            "*slot = true;",
            "*slot = false;",
            "reduction_metadata_binds_grouping_to_checked_input_and_result_domains",
        ),
        (
            "            index = coordinate\n",
            "            index = 0_i64\n",
            "reduction_metadata_binds_grouping_to_checked_input_and_result_domains",
        ),
        (
            ".checked_add(before[axis])",
            ".wrapping_add(before[axis]).checked_add(0)",
            "affine_metadata_checks_exact_extents_and_offsets_without_storage",
        ),
        (
            ".checked_mul(step)",
            ".wrapping_mul(step).checked_add(0)",
            "affine_metadata_checks_exact_extents_and_offsets_without_storage",
        ),
        (
            "end[axis] > input",
            "false",
            "affine_metadata_checks_exact_extents_and_offsets_without_storage",
        ),
        (
            "self.shape.as_ref() == domain",
            "true",
            "checked_iteration_steps_preserve_exact_large_domains_without_storage",
        ),
        (
            "if self.rank == 0 {\n            Ok(0)",
            "if self.rank == 0 {\n            Ok(1)",
            "checked_iteration_steps_preserve_exact_large_domains_without_storage",
        ),
        (
            "self.shape.as_ref() == domain {\n            Ok(1)",
            "self.shape.as_ref() == domain {\n            Ok(0)",
            "checked_iteration_steps_preserve_exact_large_domains_without_storage",
        ),
        (
            "self.normalize_axis(previous)? == axis",
            "false",
            "checked_movement_relations_reject_invalid_bijections_and_bystanders",
        ),
        (
            "target.shape[out_axis] != self.shape[axis]",
            "false",
            "checked_movement_relations_reject_invalid_bijections_and_bystanders",
        ),
        (
            "!inserted && self.shape[axis] != 1",
            "false",
            "checked_movement_relations_reject_invalid_bijections_and_bystanders",
        ),
        (
            "extent != self.shape[input_axis]",
            "false",
            "checked_movement_relations_reject_invalid_bijections_and_bystanders",
        ),
        (
            "write(axis, linear % extent)",
            "write(axis, 0)",
            "checked_movement_coordinates_preserve_exact_large_indices_without_storage",
        ),
        (
            "**extent < 0",
            "false",
            "zeros_do_not_hide_negative_extents_or_canonical_stride_overflow",
        ),
        (
            "product\n                .0\n                .checked_mul(extent)",
            "product\n                .0\n                .wrapping_mul(extent).checked_add(0)",
            "exact_int64_counts_reject_product_and_representation_byte_overflow",
        ),
        (
            ".checked_mul(width)",
            ".wrapping_mul(width).checked_add(0)",
            "exact_int64_counts_reject_product_and_representation_byte_overflow",
        ),
        (
            ".checked_mul(shape[axis])",
            ".wrapping_mul(shape[axis]).checked_add(0)",
            "zeros_do_not_hide_negative_extents_or_canonical_stride_overflow",
        ),
        (
            "value > limit",
            "false",
            "target_projection_is_checked_without_requesting_a_large_allocation",
        ),
        (
            "self.bytes.0 > capacity.0",
            "false",
            "checked_indexing_and_byte_ranges_reject_out_of_bounds_before_access",
        ),
        (
            ".checked_add(extra)",
            ".wrapping_add(extra).checked_add(0)",
            "scratch_lengths_and_rank_projections_are_checked_before_allocation",
        ),
    ] {
        assert_eq!(
            owner.matches(from).count(),
            1,
            "unique mutation anchor: {from}"
        );
        compiled(probe.execute_contract(&owner, &tests, witness));
        let output = probe.execute_contract(&owner.replacen(from, to, 1), &tests, witness);
        assert_eq!(
            output.status.code(),
            Some(101),
            "weakened contract was accepted: {from}"
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 failed"),
            "missing failing test receipt: {from}"
        );
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove owned metadata compile probe");
    }
}

fn compiled(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn rejected(output: Output, code: &str, witness: &str) {
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        !output.status.success(),
        "negative metadata control compiled: {witness}"
    );
    assert!(
        stderr.contains(code) && stderr.contains(witness),
        "wrong rejection: {stderr}"
    );
}

fn declaration(source: &str, name: &str) -> String {
    let marker = format!("struct {name} {{");
    assert_eq!(
        source.matches(&marker).count(),
        1,
        "unique production declaration"
    );
    let start = source.find(&marker).unwrap();
    let end = start + source[start..].find("\n}\n").unwrap() + 3;
    source[start..end].to_owned()
}

#[test]
fn internal_callers_cannot_forge_counts_or_restore_independent_tensor_fields() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let probe = Probe::new();
    compiled(probe.compile(
        "chelis_vocab",
        &fs::read_to_string(root.join("../chelis-vocab/src/lib.rs")).unwrap(),
        false,
    ));
    let owner = fs::read_to_string(root.join("src/metadata.rs")).unwrap();
    let library = fs::read_to_string(root.join("src/lib.rs")).unwrap();
    fs::write(probe.0.join("metadata.rs"), &owner).unwrap();
    let declarations = format!(
        "{}\n{}",
        declaration(&library, "chelis_tensor"),
        declaration(&library, "TensorStorage")
    );
    let context = format!(
        "#![allow(dead_code, non_camel_case_types)]\nmod metadata;\n\
         use metadata::{{ShapeMetadata, ElementCount, ByteCount, AllocationBytes, ReductionMetadata, SparseMetadata, MatmulMetadata, WindowMetadata}};\n\
         use chelis_vocab::RuntimeDType;\nuse std::sync::atomic::AtomicU8;\n\
         struct HeapHeader;\nstruct TensorStorageProvenance;\nstruct chelis_tensor_write;\n\
         {declarations}\n"
    );
    let positive = r#"
        fn valid(t: &chelis_tensor, storage: &TensorStorage) {
            let count = ElementCount::from_extents(&[2, 3]).unwrap();
            let bytes: ByteCount = count.bytes(RuntimeDType::I32).unwrap();
            let allocation: AllocationBytes = bytes.allocation().unwrap();
            assert_eq!(allocation.get(), 24);
            let metadata = ShapeMetadata::contiguous(&[2, 3], RuntimeDType::I32).unwrap();
            metadata.require_capacity(bytes).unwrap();
            let _: i64 = t.metadata.elements().get();
            let _: i64 = storage.byte_capacity.get();
            let plan = ReductionMetadata::new(&[2, 3], &[1], RuntimeDType::I64).unwrap();
            let _: i64 = plan.index(1, 2).unwrap();
            let indices = ShapeMetadata::contiguous(&[2], RuntimeDType::I64).unwrap();
            let sparse = SparseMetadata::new(&metadata, &indices, 1, false).unwrap();
            let _: i64 = sparse.data_index(0, 1).unwrap();
        }
    "#;
    let mut negatives = vec![
        ("fn bad(m: &mut WindowMetadata) { m.result = ShapeMetadata::contiguous(&[99], RuntimeDType::I64).unwrap(); }".to_owned(), "E0616", "result"),
        ("fn bad(m: &mut WindowMetadata) { m.count = ElementCount::from_extents(&[99]).unwrap(); }".to_owned(), "E0616", "count"),
        (
            "fn bad(m: &mut MatmulMetadata) { m.result = ShapeMetadata::contiguous(&[99], RuntimeDType::I64).unwrap(); }".to_owned(),
            "E0616",
            "result",
        ),
        (
            "fn bad(m: &mut MatmulMetadata) { m.batches = ElementCount::from_extents(&[99]).unwrap(); }".to_owned(),
            "E0616",
            "batches",
        ),
        (
            "fn bad(m: &mut SparseMetadata) { m.domain = ShapeMetadata::contiguous(&[99], RuntimeDType::I64).unwrap(); }".to_owned(),
            "E0616",
            "domain",
        ),
        (
            "fn bad(m: &mut SparseMetadata) { m.axis = 99; }".to_owned(),
            "E0616",
            "axis",
        ),
        (
            "fn bad(m: &mut ReductionMetadata) { m.leaves = ElementCount::from_extents(&[99]).unwrap(); }".to_owned(),
            "E0616",
            "leaves",
        ),
        (
            "fn bad(m: &mut ReductionMetadata) { m.result = ShapeMetadata::contiguous(&[99], RuntimeDType::I64).unwrap(); }".to_owned(),
            "E0616",
            "result",
        ),
        (
            "fn bad() { let _ = ElementCount(1); }".to_owned(),
            "E0423",
            "ElementCount",
        ),
        (
            "fn bad() { let _ = ByteCount(1); }".to_owned(),
            "E0423",
            "ByteCount",
        ),
        (
            "fn bad() { let _ = AllocationBytes(1); }".to_owned(),
            "E0423",
            "AllocationBytes",
        ),
        (
            "fn bad(s: &mut TensorStorage) { s.byte_capacity = 1; }".to_owned(),
            "E0308",
            "ByteCount",
        ),
        (
            "fn bad(m: &mut ShapeMetadata) { m.shape()[0] = 1; }".to_owned(),
            "E0594",
            "&",
        ),
        (
            "fn bad(c: ByteCount) { let _: AllocationBytes = c; }".to_owned(),
            "E0308",
            "AllocationBytes",
        ),
        (
            "fn bad(m: &ShapeMetadata) { let _ = m.index_step_for_checked_shape(&[-1]); }"
                .to_owned(),
            "E0624",
            "index_step_for_checked_shape",
        ),
    ];
    for field in ["shape", "strides", "rank", "elements", "bytes", "dtype"] {
        negatives.push((
            format!("fn bad(m: &mut ShapeMetadata) {{ let _ = &mut m.{field}; }}"),
            "E0616",
            field,
        ));
    }
    for field in ["shape", "strides", "size", "rank", "dtype"] {
        negatives.push((
            format!("fn bad(t: &mut chelis_tensor) {{ let _ = &mut t.{field}; }}"),
            "E0609",
            field,
        ));
    }
    for (negative, code, witness) in negatives {
        // Every negative gets a successful compile of this same real owner,
        // declarations and dependency; unrelated build failures cannot count.
        compiled(probe.compile("positive", &format!("{context}\n{positive}"), true));
        rejected(
            probe.compile("negative", &format!("{context}\n{negative}"), true),
            code,
            witness,
        );
    }

    // Positive mutation control: exposing a field must make the forbidden
    // caller compile, proving the negative above detects this weakened owner.
    let anchor = "    elements: ElementCount,";
    let carrier = declaration(&owner, "ShapeMetadata");
    assert_eq!(
        carrier.matches(anchor).count(),
        1,
        "exact ShapeMetadata count field"
    );
    fs::write(
        probe.0.join("metadata.rs"),
        owner.replacen(
            &carrier,
            &carrier.replacen(anchor, "    pub(crate) elements: ElementCount,", 1),
            1,
        ),
    )
    .unwrap();
    compiled(probe.compile(
        "exposed",
        &format!("{context}\nfn bad(m: &mut ShapeMetadata) {{ let _ = &mut m.elements; }}"),
        true,
    ));
}
