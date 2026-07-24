//! The no-sixth-layer payload census (chelis#729 rework; chelis#856).
//!
//! The rework sealed the IR constant payloads (`RiscOp::Const` /
//! `ConstTensor` carry `chelis_types::ScalarValue` / `TensorStorage`,
//! the FIFTH storage layer of dtype_semantics.md section C3). This
//! census makes the closure mechanical: it sweeps the two declaration
//! surfaces that can carry numeric VALUES into execution - the IR op
//! vocabulary (`chelis-ir/src/dag.rs`) and its wire mirror
//! (`chelis-compiler-api/src/schema.rs`, the `WireRiscOp` block) - and
//! pins the exact set of remaining raw `f64`/`Vec<f64>` payload fields.
//! Growing a new numeric payload field without a census entry (a sealed
//! type, or a citation here) is a RED TEST, not a review catch.
//!
//! The cited allowlist:
//!
//! * `UniformLike { low, high }` and `Dropout { rate }` - float-only op
//!   PARAMETERS (RNG bounds and a probability); f64 is their exact
//!   domain, no integer capacity exists to lose.
//! * `Pad { fill }` - a VALUE at the padded tensor's dtype with f64
//!   capacity only; the user-reachable int64 collapse above 2^53 is
//!   filed as chelis#878 (the chelis#856 sibling) and owns this row
//!   until the fill is sealed.
//!
//! Prove's own `f64` fields (`ad_rail`/`arb_oracle` box and range
//! bounds) are real-valued ENVELOPE mathematics, not dtype-carrying
//! value storage: prove reads wire constants through the named-lossy
//! accessors by design until the chelis#688 exact-env swap (Phase 2),
//! recorded in `graph_extract.rs`. They are out of this census's class.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/chelis-cli has a workspace root two levels up")
        .to_path_buf();
    assert!(
        root.join("crates").is_dir() && root.join("spec").is_dir(),
        "workspace root discovery failed: {root:?}"
    );
    root
}

/// Slice the body of `pub enum <name> {` from `source` (to the closing
/// brace at column zero).
fn enum_block<'a>(source: &'a str, name: &str) -> &'a str {
    let needle = format!("pub enum {name} {{");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("`{needle}` not found"));
    let rest = &source[start..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`{name}` block does not close"));
    &rest[..end]
}

/// Every `field: f64` / `field: Vec<f64>` declaration inside `block`,
/// as sorted `field: type` strings.
fn raw_float_payload_fields(block: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in block.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        for ty in ["f64", "Vec<f64>", "f32", "Vec<f32>"] {
            let suffix_comma = format!(": {ty},");
            let suffix_bare = format!(": {ty}");
            if let Some(name) = trimmed
                .strip_suffix(&suffix_comma)
                .or_else(|| trimmed.strip_suffix(&suffix_bare))
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !name.is_empty()
            {
                out.push(format!("{name}: {ty}"));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn risc_op_has_no_unsealed_numeric_payload_beyond_the_census() {
    let source = std::fs::read_to_string(repo_root().join("crates/chelis-ir/src/dag.rs"))
        .expect("dag.rs readable");
    let block = enum_block(&source, "RiscOp");

    let fields = raw_float_payload_fields(block);
    assert_eq!(
        fields,
        vec![
            // Pad's fill: chelis#878 owns this row (see the file header).
            "fill: f64".to_string(),
            // UniformLike bounds + Dropout rate: float-only op
            // parameters, exact in f64 by domain.
            "high: f64".to_string(),
            "low: f64".to_string(),
            "rate: f64".to_string(),
        ],
        "a raw float payload field entered or left `RiscOp` without a \
         census entry. Sealed constants go through the dtype_semantics \
         module (chelis#856); anything else needs a citation HERE and, \
         if it carries a user value, a filed issue (the chelis#878 \
         precedent)."
    );

    // The fifth layer itself: the constant payloads are the SEALED
    // module types, constructed only through dtype_semantics.
    assert!(
        block.contains("value: chelis_types::ScalarValue"),
        "RiscOp::Const must carry the sealed ScalarValue payload (chelis#856)"
    );
    assert!(
        block.contains("data: chelis_types::TensorStorage"),
        "RiscOp::ConstTensor must carry the sealed TensorStorage payload (chelis#856)"
    );
}

#[test]
fn wire_risc_op_mirror_has_no_unsealed_numeric_payload_beyond_the_census() {
    let source =
        std::fs::read_to_string(repo_root().join("crates/chelis-compiler-api/src/schema.rs"))
            .expect("schema.rs readable");
    let block = enum_block(&source, "WireRiscOp");

    let fields = raw_float_payload_fields(block);
    assert_eq!(
        fields,
        vec![
            // The wire mirrors of the cited RiscOp rows; Pad's fill is
            // chelis#878's, the rest are float-only op parameters.
            "fill: f64".to_string(),
            "high: f64".to_string(),
            "low: f64".to_string(),
            "rate: f64".to_string(),
        ],
        "a raw float payload field entered or left `WireRiscOp` without \
         a census entry (wire v4 carries constants as sealed dtype-tagged \
         payloads; see the WIRE_DAG_SCHEMA_VERSION history)."
    );

    assert!(
        block.contains("value: chelis_types::ScalarValue"),
        "WireRiscOp::Const must carry the sealed dtype-tagged payload (wire v4)"
    );
    assert!(
        block.contains("data: chelis_types::TensorStorage"),
        "WireRiscOp::ConstTensor must carry the sealed dtype-tagged payload (wire v4)"
    );
}
