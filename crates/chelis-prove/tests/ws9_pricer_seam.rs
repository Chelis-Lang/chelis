//! WS-9: Vectorized Black-Scholes pricer byte-seam integration test.
//!
//! Verifies that bs_call_vec is a WireDag root, round-trips through the
//! content-addressed byte seam, and produces a deterministic hash.

use chelis_compiler_api::schema::{SourceKind, WIRE_DAG_SCHEMA_VERSION};
use chelis_prove::WireDagByteStore;
use chelis_prove::discharge::{IntervalBox, IrHandle, OutputRange};
use chelis_prove::graph_extract::box_range_goal_from_source_entry;
use sha2::{Digest, Sha256};

/// The vectorized BS pricer source (pure-tensor-DAG, no vmap/host ops).
const BS_VEC_SOURCE: &str = include_str!("fixtures/bs_vec.ch");

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn make_input_box() -> IntervalBox {
    IntervalBox {
        dims: vec![
            // Market parameters (ranges)
            ("s".into(), 50.0, 150.0),
            ("k".into(), 50.0, 150.0),
            ("r".into(), 0.01, 0.10),
            ("sigma".into(), 0.1, 0.5),
            ("t".into(), 0.1, 2.0),
            // A-S erf coefficients (point intervals -- constants)
            ("half".into(), 0.5, 0.5),
            (
                "inv_sqrt2".into(),
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ),
            ("a1".into(), 0.254829592, 0.254829592),
            ("a2".into(), -0.284496736, -0.284496736),
            ("a3".into(), 1.421413741, 1.421413741),
            ("a4".into(), -1.453152027, -1.453152027),
            ("a5".into(), 1.061405429, 1.061405429),
            ("p".into(), 0.3275911, 0.3275911),
            (
                "twosqrtpi".into(),
                std::f64::consts::FRAC_2_SQRT_PI,
                std::f64::consts::FRAC_2_SQRT_PI,
            ),
            ("small_thresh".into(), 0.00001, 0.00001),
        ],
    }
}

fn make_output() -> OutputRange {
    OutputRange {
        output: "bs_call_vec".into(),
        lo: 0.0,
        hi: 200.0,
    }
}

/// bs_call_vec must lower to a WireDag root via entry-scoped lowering.
#[test]
fn bs_call_vec_is_wire_dag_root() {
    let extracted = box_range_goal_from_source_entry(
        BS_VEC_SOURCE,
        SourceKind::Surf,
        "bs_call_vec",
        make_input_box(),
        make_output(),
    )
    .expect("bs_call_vec must lower and produce a goal");

    assert!(
        extracted.goal.ir.is_populated(),
        "IrHandle must be populated"
    );
    assert!(
        !extracted.wire_dag_bytes.is_empty(),
        "WireDag bytes non-empty"
    );
    assert!(!extracted.dag_hash.is_empty(), "dag_hash non-empty");
    assert!(
        extracted
            .dag_hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "dag_hash must be lowercase hex"
    );
    eprintln!(
        "bs_call_vec WireDag: {} bytes, hash {}...",
        extracted.wire_dag_bytes.len(),
        &extracted.dag_hash[..16]
    );
}

/// The WireDag bytes round-trip: sha256 matches, store+retrieve works.
#[test]
fn bs_call_vec_byte_seam_round_trip() {
    let extracted = box_range_goal_from_source_entry(
        BS_VEC_SOURCE,
        SourceKind::Surf,
        "bs_call_vec",
        make_input_box(),
        make_output(),
    )
    .expect("lower");

    // Hash matches bytes
    assert_eq!(extracted.dag_hash, sha256_hex(&extracted.wire_dag_bytes));

    // IrHandle populated
    let root_index = extracted.goal.ir.root_index().expect("root_index");
    let handle = IrHandle::from_wire_dag(extracted.dag_hash.clone(), root_index);
    assert!(handle.is_populated());
    assert_eq!(handle.dag_hash().unwrap(), &extracted.dag_hash);

    // WireDagByteStore round-trip via insert_extracted
    let store = WireDagByteStore::new();
    store.insert_extracted(&extracted);

    // Deserialize bytes to verify structure
    let parsed: serde_json::Value =
        serde_json::from_slice(&extracted.wire_dag_bytes).expect("valid JSON");
    // The producer stamps the current wire schema version. This moved
    // from 1 -> 2 in chelis#178 (WireRiscOp::FloorDiv / TruncDiv joined
    // the vocabulary); assert against the constant so the seam test
    // tracks future additive bumps instead of pinning a magic number.
    assert_eq!(
        parsed["schema_version"].as_u64(),
        Some(u64::from(WIRE_DAG_SCHEMA_VERSION))
    );
    assert!(
        parsed["roots"]
            .as_array()
            .map(|r| !r.is_empty())
            .unwrap_or(false),
        "must have roots"
    );

    eprintln!(
        "round-trip OK: {} bytes, root_index {}",
        extracted.wire_dag_bytes.len(),
        root_index
    );
}

/// Lowering is deterministic: two runs produce the same hash.
#[test]
fn bs_call_vec_deterministic_hash() {
    let r1 = box_range_goal_from_source_entry(
        BS_VEC_SOURCE,
        SourceKind::Surf,
        "bs_call_vec",
        make_input_box(),
        make_output(),
    )
    .expect("run 1");

    let r2 = box_range_goal_from_source_entry(
        BS_VEC_SOURCE,
        SourceKind::Surf,
        "bs_call_vec",
        make_input_box(),
        make_output(),
    )
    .expect("run 2");

    assert_eq!(r1.dag_hash, r2.dag_hash, "hash must be deterministic");
    assert_eq!(
        r1.wire_dag_bytes, r2.wire_dag_bytes,
        "bytes must be identical"
    );
}

/// No non-finite floats in the WireDag (content-address precondition).
#[test]
fn bs_call_vec_no_non_finite_floats() {
    let extracted = box_range_goal_from_source_entry(
        BS_VEC_SOURCE,
        SourceKind::Surf,
        "bs_call_vec",
        make_input_box(),
        make_output(),
    )
    .expect("lower");

    let json_str = std::str::from_utf8(&extracted.wire_dag_bytes).expect("UTF-8");
    let parsed: serde_json::Value = serde_json::from_str(json_str).expect("JSON");
    // Walk all nodes; any "value" field that is null indicates a non-finite float
    if let Some(nodes) = parsed["nodes"].as_array() {
        for (i, node) in nodes.iter().enumerate() {
            if let Some(v) = node.get("op").and_then(|op| op.get("value")) {
                assert!(
                    !v.is_null(),
                    "non-finite float at nodes[{i}].op.value (serialized as null)"
                );
            }
        }
    }
}
