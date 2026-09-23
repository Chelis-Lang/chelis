//! spec/10 §§3.2–3.4 (wire v17) and [05-OP-8/37]: random nodes carry no
//! fields; their controls and key are operands. Every key is produced by a
//! `DrawKey` and consumed at most once; adjoint replays read a key without
//! consuming it. These tests exercise codecs and object admission, not random
//! kernel output.
use chelis_compiler_api::schema::{
    CheckRequest, LowerRequest, SourceKind, WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagDecodeError,
    WireDagNode,
};
use serde_json::{Value, json};

fn construct(value: &Value) -> WireDag {
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: serde_json::from_value::<Vec<WireDagNode>>(value["nodes"].clone()).unwrap(),
        roots: serde_json::from_value(value["roots"].clone()).unwrap(),
    }
}

fn accepts(value: &Value) {
    let text = value.to_string();
    let direct: WireDag = serde_json::from_str(&text).unwrap();
    let admitted = WireDag::from_validated_json(&text).unwrap();
    for wire in [direct, admitted, construct(value)] {
        assert_eq!(serde_json::to_value(wire).unwrap(), *value);
    }
}

fn rejects_domain(value: &Value, reason: &str) {
    let text = value.to_string();
    let direct = serde_json::from_str::<WireDag>(&text).unwrap_err();
    assert!(direct.to_string().contains(reason), "{direct}: {text}");
    let admitted = WireDag::from_validated_json(&text).unwrap_err();
    assert!(matches!(admitted, WireDagDecodeError::Contract(_)));
    assert!(admitted.to_string().contains(reason), "{admitted}: {text}");
    let serialization = serde_json::to_value(construct(value)).unwrap_err();
    assert!(
        serialization.to_string().contains(reason),
        "{serialization}: {text}"
    );
}

fn lower(source: &str, entry: &str) -> Value {
    let checked = chelis_compiler_api::compiler::check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
    })
    .unwrap();
    assert!(checked.errors.is_empty(), "{:?}", checked.errors);
    let lowered = chelis_compiler_api::compiler::lower(LowerRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        entry: Some(entry.into()),
    })
    .unwrap();
    let dag = serde_json::to_value(lowered).unwrap()["dag"].clone();
    assert_eq!(dag["schema_version"], json!(WIRE_DAG_SCHEMA_VERSION));
    dag
}

fn first(dag: &Value, kind: &str) -> usize {
    dag["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .position(|node| node["op"]["kind"] == kind)
        .unwrap_or_else(|| panic!("the lowered graph has no `{kind}` node"))
}

fn input(dag: &Value, node: usize, slot: usize) -> usize {
    usize::try_from(dag["nodes"][node]["inputs"][slot].as_u64().unwrap()).unwrap()
}

/// A handled runtime-rate dropout and a uniform draw: two scoped draw keys
/// of one `with seed` region, each consumed by its primitive.
const HANDLED: &str = concat!(
    "def sample(x: tensor[2, f32], rate: f32) -> (tensor[2, f32], tensor[2, f32]) = with seed(7i64) {\n",
    "  (dropout(copy(x), rate), uniform_like(x, 0.0f32, 1.0f32))\n",
    "}\n"
);

#[test]
fn lowered_key_operand_draws_round_trip_with_operand_controls() {
    let dag = lower(HANDLED, "sample");
    accepts(&dag);
    let dropout = first(&dag, "dropout");
    let uniform = first(&dag, "uniform_like");
    assert_eq!(dag["nodes"][dropout]["op"], json!({"kind":"dropout"}));
    assert_eq!(dag["nodes"][uniform]["op"], json!({"kind":"uniform_like"}));
    assert_eq!(dag["nodes"][dropout]["inputs"].as_array().unwrap().len(), 3);
    assert_eq!(dag["nodes"][uniform]["inputs"].as_array().unwrap().len(), 4);
    let dropout_key = &dag["nodes"][input(&dag, dropout, 2)];
    let uniform_key = &dag["nodes"][input(&dag, uniform, 3)];
    for (key, draw) in [(dropout_key, "dropout"), (uniform_key, "uniform_like")] {
        assert_eq!(key["op"]["kind"], "draw_key");
        assert_eq!(key["op"]["draw"], draw);
        assert_eq!(key["op"]["dtype"], "f32");
        assert_eq!(key["op"]["handler"]["kind"], "scoped");
        assert_eq!(key["output_type"], json!({"dims":[],"precision":"key"}));
        let seed = usize::try_from(key["inputs"][0].as_u64().unwrap()).unwrap();
        assert_eq!(dag["nodes"][seed]["op"]["kind"], "const");
        assert_eq!(
            dag["nodes"][seed]["output_type"],
            json!({"dims":[],"precision":"int64"})
        );
    }
    // One region, one scoped instance.
    assert_eq!(
        dropout_key["op"]["handler"]["instance"],
        uniform_key["op"]["handler"]["instance"]
    );
    // The runtime rate reaches the primitive as its operand.
    assert_eq!(
        dag["nodes"][input(&dag, dropout, 1)]["op"]["kind"],
        json!("load")
    );
}

#[test]
fn version_16_random_payloads_have_no_version_17_spelling() {
    let dag = lower(HANDLED, "sample");
    let dropout = first(&dag, "dropout");
    for payload in [
        json!({"kind":"dropout","rate":{"dtype":"f32","bits":"3f000000"},"seed":7}),
        json!({"kind":"uniform_like","low":{"dtype":"f32","bits":"00000000"},"high":{"dtype":"f32","bits":"3f800000"},"seed":7}),
    ] {
        let mut stale = dag.clone();
        stale["nodes"][dropout]["op"] = payload;
        let text = stale.to_string();
        assert!(serde_json::from_str::<WireDag>(&text).is_err(), "{text}");
        assert!(
            matches!(
                WireDag::from_validated_json(&text),
                Err(WireDagDecodeError::Parse(_))
            ),
            "{text}"
        );
    }
}

#[test]
fn a_key_is_consumed_once_and_only_by_a_random_primitive() {
    let dag = lower(HANDLED, "sample");
    let dropout = first(&dag, "dropout");
    let uniform = first(&dag, "uniform_like");
    let dropout_key = input(&dag, dropout, 2);

    // A second primitive consuming the same key.
    let mut twice = dag.clone();
    twice["nodes"][uniform]["inputs"][3] = json!(dropout_key);
    rejects_domain(&twice, "consumed by more than one");

    // A key read by an operation that takes none.
    let mut foreign = dag.clone();
    let nodes = foreign["nodes"].as_array_mut().unwrap();
    let id = nodes.len();
    nodes.push(
        json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":id,
        "op":{"kind":"neg"},"inputs":[dropout_key],
        "output_type":{"dims":[],"precision":"key"}}),
    );
    rejects_domain(&foreign, "does not take one");

    // A key slot fed by something other than a draw key.
    let mut not_drawn = dag.clone();
    let rate = input(&dag, dropout, 1);
    not_drawn["nodes"][dropout]["inputs"][2] = json!(rate);
    rejects_domain(&not_drawn, "produced by a draw key");
}

#[test]
fn random_controls_seeds_and_keys_keep_their_structural_types() {
    let dag = lower(HANDLED, "sample");
    let dropout = first(&dag, "dropout");
    let key = input(&dag, dropout, 2);
    let seed = input(&dag, key, 0);
    let data = input(&dag, dropout, 0);

    // The rate is a rank-zero value of the data dtype, never an integer.
    let mut integer_rate = dag.clone();
    integer_rate["nodes"][dropout]["inputs"][1] = json!(seed);
    rejects_domain(&integer_rate, "random control must be a rank-zero value");

    // A scoped draw key's first input is its literal int64 seed.
    let mut runtime_seed = dag.clone();
    runtime_seed["nodes"][key]["inputs"][0] = json!(data);
    rejects_domain(&runtime_seed, "literal seed");

    // A draw key produces one rank-zero key.
    let mut numeric_key = dag.clone();
    numeric_key["nodes"][key]["output_type"]["precision"] = json!("f32");
    rejects_domain(&numeric_key, "draw key produces one rank-zero key");

    // A missing key operand is a malformed primitive.
    let mut keyless = dag.clone();
    keyless["nodes"][dropout]["inputs"]
        .as_array_mut()
        .unwrap()
        .truncate(2);
    rejects_domain(&keyless, "missing an input");
}

#[test]
fn gradient_random_lowering_preserves_scalar_bool_activation() {
    // spec/06 §§2.10.1/2.11 and [05-OP-8]: differentiation keeps the
    // handled forward stream, including the IR's Bool path activation on
    // both the draw key and its primitive.
    let source = concat!(
        "def loss(x: tensor[2, f32]) -> f32 ! { Random } = tensor_to_scalar(sum(mul(copy(x), uniform_like(x, 0.0f32, 1.0f32)), 0))\n",
        "def derivative(x: tensor[2, f32]) -> tensor[2, f32] = with seed(7i64) { grad(loss)(x) }\n"
    );
    let dag = lower(source, "derivative");
    // `loss`'s own region draws without an activation; the differentiated
    // copy inside `derivative` carries one.
    let uniform = dag["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .position(|node| {
            node["op"]["kind"] == "uniform_like"
                && node["inputs"]
                    .as_array()
                    .is_some_and(|inputs| inputs.len() == 5)
        })
        .expect("gradient lowering must emit a UniformLike with a path activation");
    let activation = input(&dag, uniform, 4);
    let key = input(&dag, uniform, 3);
    assert_eq!(
        dag["nodes"][activation]["output_type"],
        json!({"dims":[],"precision":"bool"})
    );
    let key_inputs = dag["nodes"][key]["inputs"].as_array().unwrap();
    assert_eq!(key_inputs.last(), Some(&json!(activation)));
    accepts(&dag);

    // The actual producer's activation still owes wire admission on every
    // encode/decode path; a consumer cannot replace it with a numeric scalar.
    let mut malformed = dag.clone();
    malformed["nodes"][activation]["op"] = json!({"kind":"load","name":"bad_activation"});
    malformed["nodes"][activation]["output_type"]["precision"] = json!("f32");
    rejects_domain(&malformed, "rank-zero Bool activation");
}
