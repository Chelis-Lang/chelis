//! Tests for `spec/upstream-bugs/producer-string-sanitization.md` —
//! every producer-supplied string flowing into a HIP-backend `printf`-
//! style format string is routed through
//! `chelis_ir::span_sanitize::sanitize_for_format_string`.
//!
//! HIP emits TWO entrypoints:
//! - host entrypoint (`{func_name}` taking `chelis_tensor**`)
//! - device entrypoint (`{func_name}_device` taking `chelis_gpu_tensor**`)
//!
//! Each has its own input-shape preamble with fprintf reports. Both
//! must apply the format-string sanitizer.

use chelis_backend_hip::codegen_hip;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::load_store_name::LoadStoreName;
use chelis_types::types::Prim;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn dirty_name(s: &str) -> LoadStoreName {
    let json = serde_json::to_string(s).expect("string serializes");
    serde_json::from_str(&json)
        .expect("LoadStoreName deserialize is transparent (no re-validation)")
}

#[test]
fn hip_fprintf_format_string_escapes_percent_in_func_name() {
    // The host entrypoint emits `fprintf(stderr, "{func_name}: ...")`
    // and the device entrypoint emits `fprintf(stderr, "{func_name}_device:
    // ...")`. Both must escape `%` in the producer-supplied func_name.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_hip(&dag, "f%spct").unwrap();
    let src = &result.c_source;

    // Host entrypoint format string carries `%%`.
    assert!(
        src.contains("\"f%%spct: expected"),
        "host entrypoint must escape `%` in func_name; source:\n{src}"
    );
    // Device entrypoint format string carries `%%_device`.
    assert!(
        src.contains("\"f%%spct_device: expected"),
        "device entrypoint must escape `%` in func_name; source:\n{src}"
    );
    // No raw `%s` reaches either format string.
    assert!(
        !src.contains("\"f%spct"),
        "raw `%s` leaked into HIP fprintf format; source:\n{src}"
    );
}

#[test]
fn hip_fprintf_format_string_escapes_percent_in_load_name() {
    let mut dag = Dag::new();
    let bad = dirty_name("inp%s");
    let a = dag.add_node(RiscOp::Load { name: bad }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_hip(&dag, "test_load_pct").unwrap();
    let src = &result.c_source;

    // Both host and device input-shape preambles emit "input `<label>`".
    assert!(
        src.contains("input `inp%%s`"),
        "expected `%%`-escaped Load label in HIP fprintf format; source:\n{src}"
    );
    assert!(
        !src.contains("input `inp%s`"),
        "raw `%s` in Load name leaked into HIP fprintf format; source:\n{src}"
    );
}

#[test]
fn hip_fprintf_format_string_escapes_newline_in_func_name() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_hip(&dag, "f\nINJECT").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("\"f\\nINJECT: expected"),
        "host entrypoint must escape newline in func_name; source:\n{src}"
    );
    assert!(
        src.contains("\"f\\nINJECT_device: expected"),
        "device entrypoint must escape newline in func_name; source:\n{src}"
    );
}

#[test]
fn hip_clean_func_name_emitted_verbatim() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_hip(&dag, "my_func").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("\"my_func: expected"),
        "clean host func_name must be emitted verbatim; source:\n{src}"
    );
    assert!(
        src.contains("\"my_func_device: expected"),
        "clean device func_name must be emitted verbatim; source:\n{src}"
    );
}
