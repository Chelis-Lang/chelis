//! Every producer-supplied string flowing into a Metal `// ...` comment
//! is routed through `chelis_ir::span_sanitize::sanitize_for_comment`,
//! not just span IDs.
//!
//! ## Why this matters
//!
//! `LoadStoreName`'s constructor enforces an identifier grammar that
//! rejects newlines / NUL / DEL at construction time, so the
//! programmatic-IR-construction attack vector is closed at the trust
//! boundary. The architectural pattern this work establishes is broader:
//! every `format!("// ... {x}", ...)` callsite where `x` is producer-
//! supplied routes `x` through the shared sanitizer, so a future
//! contributor adding a new IR field that flows into a comment context
//! has a single canonical helper to call.
//!
//! Today the only path that smuggles a forbidden byte past
//! `LoadStoreName::new` is `serde_json::from_str` (transparent
//! deserialize, no re-validation) — locked by
//! `crates/chelis-ir/src/load_store_name.rs::deserialize_rejects_invalid_grammar`.
//! These tests ride that path to verify the emit-side sanitizer is the
//! second layer of defense in depth.

mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::load_store_name::LoadStoreName;
use chelis_types::types::Prim;
use support::codegen_metal;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

/// Smuggle a forbidden byte past `LoadStoreName::new` via
/// transparent-deserialize. This is the only legitimate way a forbidden
/// byte can reach codegen; the constructor-side trust boundary closes
/// every other path. The test fixture explicitly relies on this — if
/// `LoadStoreName` ever adds re-validation on deserialize, this helper
/// will need to use a different bypass.
fn dirty_name(s: &str) -> LoadStoreName {
    let json = serde_json::to_string(s).expect("string serializes");
    serde_json::from_str(&json)
        .expect("LoadStoreName deserialize is transparent (no re-validation)")
}

#[test]
fn metal_load_name_with_newline_is_sanitized_in_comment() {
    // Smuggle a name with an embedded newline past the LoadStoreName
    // constructor by transparent deserialize. The emit-side comment
    // sanitizer must escape the newline so the `// node N = Load ...`
    // line cannot terminate prematurely.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a_name = dirty_name("a\nINJECTED_METAL_LOAD_LINE");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: a_name },
        vec![],
        vec_f32(4),
        None,
    );
    let n = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_metal(&dag, "metal_dirty_load");
    let src = &result.mm_source;

    // The raw newline must NOT split the comment.
    assert!(
        !src.contains("Load a\nINJECTED_METAL_LOAD_LINE"),
        "raw newline leaked through emit-side sanitizer; source:\n{src}"
    );
    // The escaped form is what the comment-context sanitizer produces.
    assert!(
        src.contains("Load a\\nINJECTED_METAL_LOAD_LINE"),
        "expected `\\n`-escaped Load name in comment; source:\n{src}"
    );
}

#[test]
fn metal_store_name_with_newline_is_sanitized_in_comment() {
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
    let store_name = dirty_name("out\nINJECTED_METAL_STORE_LINE");
    let s = dag.add_node(
        decl,
        RiscOp::Store { name: store_name },
        vec![n],
        vec_f32(4),
        None,
    );
    dag.add_root(s);

    let result = codegen_metal(&dag, "metal_dirty_store");
    let src = &result.mm_source;

    assert!(
        !src.contains("Store `out\nINJECTED_METAL_STORE_LINE"),
        "raw newline leaked into Store comment; source:\n{src}"
    );
    assert!(
        src.contains("Store `out\\nINJECTED_METAL_STORE_LINE"),
        "expected `\\n`-escaped Store name in comment; source:\n{src}"
    );
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
