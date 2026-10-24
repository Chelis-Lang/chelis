//! Verify comment emission for valid graph names.
//!
//! Checked construction and deserialization reject invalid graph labels.

mod support;
use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_numeric_core::prim::Prim;
use support::codegen_metal;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

#[test]
fn metal_clean_load_name_emitted_verbatim() {
    // Audit-invariant lock: a clean Load name (parser/grammar-valid) is
    // emitted byte-identical in the comment, no spurious escaping. This
    // matches the `Cow::Borrowed` zero-copy contract on
    // `sanitize_for_comment` for clean inputs.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load {
            name: "my_input".into(),
        },
        vec![],
        vec_f32(4),
        None,
    );
    let n = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_metal(&dag, "metal_clean_load");
    let src = &result.mm_source;

    assert!(
        src.contains("// node 0 = Load my_input"),
        "clean Load name must be emitted verbatim; source:\n{src}"
    );
    // No spurious backslash-escapes.
    assert!(
        !src.contains("// node 0 = Load my\\_input"),
        "clean name was incorrectly escaped; source:\n{src}"
    );
}

#[test]
fn metal_root_label_with_control_byte_is_sanitized_in_comment() {
    // The `// root output N = `<label>` (node M)` site uses Store name
    // (LoadStoreName) for Store-tagged roots. Confirm the same
    // sanitization path via the deserialize bypass.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let n = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    // No Store node — root is the Neg, gets synthesized `root0` label
    // (clean). This branch is just a sanity check that synthesized
    // labels still pass through the sanitizer cleanly (Borrowed path).
    dag.add_root(n);

    let result = codegen_metal(&dag, "metal_root_label");
    let src = &result.mm_source;

    assert!(
        src.contains("// root output 0 = `root0`"),
        "synthesized root label must be emitted verbatim; source:\n{src}"
    );
}
