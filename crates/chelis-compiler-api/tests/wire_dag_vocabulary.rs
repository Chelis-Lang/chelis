//! #1269: a pinned reader must see every operation-vocabulary change.
use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagNode, WireRiscOp, WireTensorType,
};
use chelis_types::{scalar_from_i64, types::Prim};
use serde::Deserialize;

// Capture the actual serde decoder's closed vocabulary, rather than maintaining
// another production enum or parsing Rust source. An unexpected decoder path is
// a test failure, never an empty vocabulary that can pass the pin.
#[derive(Debug)]
struct VocabularyError(Vec<String>);

impl std::fmt::Display for VocabularyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "operation vocabulary: {:?}", self.0)
    }
}

impl std::error::Error for VocabularyError {}

impl serde::de::Error for VocabularyError {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        panic!("unexpected vocabulary decoder error: {message}")
    }

    fn unknown_variant(_: &str, variants: &'static [&'static str]) -> Self {
        Self(variants.iter().map(|name| (*name).to_owned()).collect())
    }
}

#[test]
fn wire_dag_operation_vocabulary_is_pinned_to_its_schema_version() {
    let decoder = serde::de::value::MapDeserializer::<_, VocabularyError>::new(
        [("kind", "__unknown_operation__")].into_iter(),
    );
    let VocabularyError(mut actual) = WireRiscOp::deserialize(decoder).unwrap_err();
    let mut expected = vec![
        "add",
        "sub",
        "mul",
        "div",
        "floor_div",
        "trunc_div",
        "compare",
        "logical",
        "where",
        "max_elem",
        "min_elem",
        "extrema_adjoint",
        "relu",
        "relu_adjoint",
        "neg",
        "recip",
        "exp",
        "log",
        "sin",
        "sqrt",
        "cos",
        "tan",
        "atan",
        "abs",
        "floor",
        "ceil",
        "round",
        "uniform_like",
        "dropout",
        "sum",
        "count",
        "max_reduce",
        "min_reduce",
        "prod_reduce",
        "reduce_window",
        "reduce_window_grad",
        "argmax",
        "argmin",
        "reshape",
        "permute",
        "expand",
        "one_hot",
        "pad",
        "shrink",
        "stride",
        "const",
        "const_tensor",
        "shape",
        "extent_witness",
        "load",
        "store",
        "copy",
        "drop",
        "realize",
        "cast",
        "cast_trunc",
        "fused_elem",
        "blas_matmul",
        "gather",
        "scatter_add",
        "scatter",
        "scatter_elements",
        // Version 10 (chelis#1747): checked reshape scalars, checked unit-axis
        // refinements, and the explicit integer remainder operation.
        "checked_reshape_extent",
        "checked_unit_axis",
        "mod",
        // Version 16 (chelis#1464): the guarded abort that keeps a
        // transformed `fail(...)` branch from becoming a placeholder value.
        "guarded_fail",
    ];
    actual.sort();
    expected.sort();
    assert_eq!(
        WIRE_DAG_SCHEMA_VERSION, 16,
        "review vocabulary and migration history with every version change"
    );
    assert_eq!(actual.len(), 66);
    assert_eq!(
        actual, expected,
        "operation changes require a schema-version and migration-history review"
    );
}

#[test]
fn cast_trunc_has_its_exact_wire_spelling_and_round_trips() {
    let op = WireRiscOp::CastTrunc {
        new_precision: "int32".into(),
    };
    let encoded = serde_json::to_value(&op).unwrap();
    assert_eq!(
        encoded,
        serde_json::json!({"kind": "cast_trunc", "new_precision": "int32"})
    );
    assert!(
        matches!(serde_json::from_value::<WireRiscOp>(encoded).unwrap(),
        WireRiscOp::CastTrunc { new_precision } if new_precision == "int32")
    );
}

#[test]
fn cast_trunc_rejects_unknown_spelling_and_missing_target() {
    for encoded in [
        serde_json::json!({"kind": "cast_truncate", "new_precision": "int32"}),
        serde_json::json!({"kind": "cast_trunc"}),
    ] {
        assert!(serde_json::from_value::<WireRiscOp>(encoded).is_err());
    }
}

#[test]
fn wire_dag_accepts_current_version_and_rejects_missing_old_and_future_versions() {
    let mut encoded =
        serde_json::json!({"schema_version": WIRE_DAG_SCHEMA_VERSION, "nodes": [], "roots": []});
    assert!(WireDag::from_validated_json(&encoded.to_string()).is_ok());
    for version in [WIRE_DAG_SCHEMA_VERSION - 1, WIRE_DAG_SCHEMA_VERSION + 1] {
        encoded["schema_version"] = version.into();
        assert!(WireDag::from_validated_json(&encoded.to_string()).is_err());
    }
    encoded.as_object_mut().unwrap().remove("schema_version");
    assert!(WireDag::from_validated_json(&encoded.to_string()).is_err());
}

#[test]
fn wire_dag_integer_dtype_vocabulary_stays_ecosystem_spelled() {
    let dag = WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![WireDagNode {
            shape_deps: vec![],
            span_id: None,
            merged_spans: vec![],
            id: 0,
            op: WireRiscOp::Pad {
                padding: vec![],
                fill: scalar_from_i64("wire vocabulary test", Prim::Int64, 1)
                    .expect("int64 scalar"),
            },
            inputs: vec![],
            output_type: WireTensorType {
                dims: vec![],
                precision: "int64".to_string(),
            },
        }],
        roots: vec![0],
    };
    let canonical = serde_json::to_value(&dag).expect("encode WireDag");
    WireDag::from_validated_json(&canonical.to_string()).expect("canonical WireDag");

    for language_name in ["i8", "i16", "i32", "i64"] {
        let mut encoded = canonical.clone();
        encoded["nodes"][0]["output_type"]["precision"] = language_name.into();
        assert!(
            WireDag::from_validated_json(&encoded.to_string()).is_err(),
            "WireDag v{WIRE_DAG_SCHEMA_VERSION} must reject language dtype {language_name}"
        );
    }
}
