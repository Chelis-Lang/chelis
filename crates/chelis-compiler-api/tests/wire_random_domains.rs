//! spec/10 §§3.2–3.4 (wire v17) and [05-OP-8/37]: random nodes carry no
//! fields; their controls and key are operands. Every key is produced by a
//! `DrawKey` and consumed at most once; adjoint replays read a key without
//! consuming it, and no key is a graph root or a shape dependency. These tests
//! exercise codecs and object admission, not random kernel output.
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
fn a_key_is_consumed_once_and_only_by_a_key_consumer() {
    let dag = lower(HANDLED, "sample");
    let dropout = first(&dag, "dropout");
    let uniform = first(&dag, "uniform_like");
    let dropout_key = input(&dag, dropout, 2);

    // A second primitive consuming the same key.
    let mut twice = dag.clone();
    twice["nodes"][uniform]["inputs"][3] = json!(dropout_key);
    rejects_domain(&twice, "is consumed twice");

    // A key read by an operation that takes none.
    let mut foreign = dag.clone();
    let nodes = foreign["nodes"].as_array_mut().unwrap();
    let id = nodes.len();
    nodes.push(
        json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":id,
        "op":{"kind":"neg"},"inputs":[dropout_key],
        "output_type":{"dims":[],"precision":"key"}}),
    );
    rejects_domain(
        &foreign,
        "only a key operation or a random primitive consumes a key",
    );

    // A key slot fed by something that is not a key.
    let mut not_a_key = dag.clone();
    let rate = input(&dag, dropout, 1);
    not_a_key["nodes"][dropout]["inputs"][2] = json!(rate);
    rejects_domain(
        &not_a_key,
        "requires a key batch matching its data's leading axes",
    );

    // A draw key's key derived from: only its draw consumes it.
    let mut derived = dag.clone();
    let id = derived["nodes"].as_array().unwrap().len();
    derived["nodes"].as_array_mut().unwrap().push(
        json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":id,
        "op":{"kind":"split","branch":"left"},"inputs":[dropout_key],
        "output_type":{"dims":[],"precision":"key"}}),
    );
    derived["roots"].as_array_mut().unwrap().push(json!(id));
    rejects_domain(&derived, "a draw key's key feeds only its draw");
}

/// spec/10 §3.2 (v18): a key enters a graph as a key operation's or draw
/// key's output or as a key-precision `Load`, and may be a root; it is never
/// a shape dependency, and no constant carries one.
#[test]
fn a_key_may_be_loaded_or_rooted_and_is_never_a_dependency_or_a_constant() {
    let dag = lower(HANDLED, "sample");
    let dropout = first(&dag, "dropout");
    let key = input(&dag, dropout, 2);

    // A key-precision Load, unconsumed: dropping a key is allowed.
    let mut loaded = dag.clone();
    let id = loaded["nodes"].as_array().unwrap().len();
    loaded["nodes"].as_array_mut().unwrap().push(
        json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":id,
        "op":{"kind":"load","name":"k"},"inputs":[],
        "output_type":{"dims":[],"precision":"key"}}),
    );
    accepts(&loaded);

    // The loaded key as a root. A draw key's key is its draw's alone, so a
    // consumed draw key is no root.
    let mut loaded_root = loaded.clone();
    loaded_root["roots"].as_array_mut().unwrap().push(json!(id));
    accepts(&loaded_root);
    let mut key_root = dag.clone();
    key_root["roots"].as_array_mut().unwrap().push(json!(key));
    rejects_domain(&key_root, "a draw key's key feeds only its draw");

    // A key as a shape dependency of its own consumer.
    let mut key_dependency = dag.clone();
    key_dependency["nodes"][dropout]["shape_deps"] = json!([key]);
    rejects_domain(&key_dependency, "as a dependency");

    // A constant typed as a key: every key is a node's output, never bits.
    let mut constant = dag.clone();
    let id = constant["nodes"].as_array().unwrap().len();
    constant["nodes"].as_array_mut().unwrap().push(
        json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":id,
        "op":{"kind":"const","value":{"dtype":"int64","value":7}},"inputs":[],
        "output_type":{"dims":[],"precision":"key"}}),
    );
    let text = constant.to_string();
    assert!(WireDag::from_validated_json(&text).is_err(), "{text}");
}

fn wire_node(id: usize, op: Value, inputs: &[usize], dims: &[u64], precision: &str) -> Value {
    json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":id,"op":op,
        "inputs":inputs,
        "output_type":{"dims":dims.iter().map(|size| json!({"kind":"lit","size":size})).collect::<Vec<_>>(),
        "precision":precision}})
}

/// The explicit key chain of spec/10 §3.2 (v18): key_from_seed, both split
/// branches, fold_in, split_n, a batched dropout, and a key root.
fn key_chain() -> Value {
    let nodes = vec![
        wire_node(
            0,
            json!({"kind":"const","value":{"dtype":"int64","value":-3}}),
            &[],
            &[],
            "int64",
        ),
        wire_node(1, json!({"kind":"key_from_seed"}), &[0], &[], "key"),
        wire_node(2, json!({"kind":"split","branch":"left"}), &[1], &[], "key"),
        wire_node(
            3,
            json!({"kind":"split","branch":"right"}),
            &[1],
            &[],
            "key",
        ),
        wire_node(
            4,
            json!({"kind":"const","value":{"dtype":"int64","value":-5}}),
            &[],
            &[],
            "int64",
        ),
        wire_node(5, json!({"kind":"fold_in"}), &[2, 4], &[], "key"),
        wire_node(
            6,
            json!({"kind":"split_n","count":{"bound":"lit","value":3}}),
            &[5],
            &[3],
            "key",
        ),
        wire_node(7, json!({"kind":"load","name":"x"}), &[], &[3, 4], "f32"),
        wire_node(
            8,
            json!({"kind":"const","value":{"dtype":"f32","bits":"3f000000"}}),
            &[],
            &[],
            "f32",
        ),
        wire_node(9, json!({"kind":"dropout"}), &[7, 8, 6], &[3, 4], "f32"),
        wire_node(
            10,
            json!({"kind":"const","value":{"dtype":"int64","value":9}}),
            &[],
            &[],
            "int64",
        ),
        wire_node(11, json!({"kind":"fold_in"}), &[3, 10], &[], "key"),
    ];
    json!({"schema_version": WIRE_DAG_SCHEMA_VERSION, "nodes": nodes, "roots": [9, 11]})
}

fn push(graph: &mut Value, op: Value, inputs: &[usize], dims: &[u64], precision: &str) -> usize {
    let nodes = graph["nodes"].as_array_mut().unwrap();
    let id = nodes.len();
    nodes.push(wire_node(id, op, inputs, dims, precision));
    id
}

/// Oracle (d) on the wire: the codec applies the IR verifier's key rules
/// and its own operand rules to the explicit key operations.
#[test]
fn the_codec_admits_the_key_chain_and_rejects_every_malformed_key_form() {
    accepts(&key_chain());

    let mut two_lefts = key_chain();
    two_lefts["nodes"][3]["op"]["branch"] = json!("left");
    rejects_domain(&two_lefts, "split twice for the Left branch");

    let mut split_and_fold = key_chain();
    split_and_fold["nodes"][5]["inputs"][0] = json!(1);
    rejects_domain(&split_and_fold, "is consumed twice");

    let mut added = key_chain();
    let sum = push(&mut added, json!({"kind":"add"}), &[11, 11], &[], "key");
    added["roots"].as_array_mut().unwrap().push(json!(sum));
    rejects_domain(
        &added,
        "only a key operation or a random primitive consumes a key",
    );

    let mut selected = key_chain();
    let condition = push(
        &mut selected,
        json!({"kind":"load","name":"c"}),
        &[],
        &[],
        "bool",
    );
    let chosen = push(
        &mut selected,
        json!({"kind":"where"}),
        &[condition, 11, 11],
        &[],
        "key",
    );
    selected["roots"]
        .as_array_mut()
        .unwrap()
        .push(json!(chosen));
    // Keys select through activations, never through `where` (V4); a
    // key-precision `where` is also no key operation (V1).
    rejects_domain(
        &selected,
        &format!("key 11 reaches node {chosen} input 1; only a key operation"),
    );

    let mut dependency = key_chain();
    dependency["nodes"][9]["shape_deps"] = json!([6]);
    rejects_domain(&dependency, "as a dependency");

    // V5: a key batch that does not match the data's leading axis.
    let mut batch = key_chain();
    batch["nodes"][6]["op"]["count"] = json!({"bound":"lit","value":2});
    batch["nodes"][6]["output_type"]["dims"] = json!([{"kind":"lit","size":2}]);
    rejects_domain(&batch, "key batch matching its data's leading axes");

    // V5 at rank 2: G split into [3] and then [3, 2] keys batch a draw over
    // [3, 2, 4] data, with a rate shaped like the key's leading axis.
    let rank_two = |data: &[u64], rate: &[u64]| {
        let mut graph = key_chain();
        let rows = push(
            &mut graph,
            json!({"kind":"split_n","count":{"bound":"lit","value":3}}),
            &[11],
            &[3],
            "key",
        );
        let keys = push(
            &mut graph,
            json!({"kind":"split_n","count":{"bound":"lit","value":2}}),
            &[rows],
            &[3, 2],
            "key",
        );
        let x = push(
            &mut graph,
            json!({"kind":"load","name":"x2"}),
            &[],
            data,
            "f32",
        );
        let rates = push(
            &mut graph,
            json!({"kind":"load","name":"r"}),
            &[],
            rate,
            "f32",
        );
        let drawn = push(
            &mut graph,
            json!({"kind":"dropout"}),
            &[x, rates, keys],
            data,
            "f32",
        );
        graph["roots"] = json!([9, drawn]);
        graph
    };
    accepts(&rank_two(&[3, 2, 4], &[3]));
    accepts(&rank_two(&[3, 2, 4], &[3, 2]));
    rejects_domain(
        &rank_two(&[3, 2, 4], &[2]),
        "shaped like a leading part of its key's shape",
    );
    rejects_domain(
        &rank_two(&[2, 3, 4], &[]),
        "key batch matching its data's leading axes",
    );

    // A declared count axis that disagrees with its literal count.
    let mut declared = key_chain();
    declared["nodes"][6]["output_type"]["dims"] = json!([{"kind":"lit","size":4}]);
    assert!(WireDag::from_validated_json(&declared.to_string()).is_err());

    // V2 counts uses of a key: a root is one, and every Load of one
    // parameter is the same key.
    let mut rooted = key_chain();
    rooted["roots"].as_array_mut().unwrap().push(json!(6));
    rejects_domain(&rooted, "is a graph root and is also consumed");
    let mut reloaded = key_chain();
    let y = push(
        &mut reloaded,
        json!({"kind":"load","name":"y"}),
        &[],
        &[4],
        "f32",
    );
    let mut roots = vec![json!(9), json!(11)];
    for _ in 0..2 {
        let k = push(
            &mut reloaded,
            json!({"kind":"load","name":"k"}),
            &[],
            &[],
            "key",
        );
        let drawn = push(
            &mut reloaded,
            json!({"kind":"dropout"}),
            &[y, 8, k],
            &[4],
            "f32",
        );
        roots.push(json!(drawn));
    }
    reloaded["roots"] = json!(roots);
    rejects_domain(&reloaded, "is consumed twice");

    // The same disagreement against a named count axis whose extent is
    // known; the batched draw's data declares that axis too.
    let mut named = key_chain();
    let n4 = json!({"kind":"named","name":"n","size":4});
    named["nodes"][6]["output_type"]["dims"] = json!([n4]);
    for data in [7, 9] {
        named["nodes"][data]["output_type"]["dims"] = json!([n4, {"kind":"lit","size":4}]);
    }
    rejects_domain(&named, "with a split's count axis appended last");
    named["nodes"][6]["op"]["count"] = json!({"bound":"lit","value":4});
    accepts(&named);

    // A count carrier outside `lit` and a node at slot 1.
    let mut axis_count = key_chain();
    axis_count["nodes"][6]["op"]["count"] =
        json!({"bound":"input_axis","tensor":1,"axis":{"axis":"lit","value":0}});
    assert!(WireDag::from_validated_json(&axis_count.to_string()).is_err());

    // A seed of the wrong integer width, and a fold of unequal shapes.
    let mut narrow = key_chain();
    narrow["nodes"][0] = wire_node(
        0,
        json!({"kind":"const","value":{"dtype":"int32","value":-3}}),
        &[],
        &[],
        "int32",
    );
    rejects_domain(&narrow, "wrong operand dtype");
    let mut unequal = key_chain();
    let indices = push(
        &mut unequal,
        json!({"kind":"load","name":"n"}),
        &[],
        &[2],
        "int64",
    );
    unequal["nodes"][11]["inputs"][1] = json!(indices);
    // Node 11 now reads a later node, which the reference rules reject too;
    // rebuild it at the end so only the shape rule applies.
    let fold = push(
        &mut unequal,
        json!({"kind":"fold_in"}),
        &[3, indices],
        &[],
        "key",
    );
    unequal["nodes"][11]["inputs"][1] = json!(10);
    unequal["nodes"][11]["op"] = json!({"kind":"neg"});
    unequal["nodes"][11]["inputs"] = json!([10]);
    unequal["nodes"][11]["output_type"]["precision"] = json!("int64");
    unequal["roots"] = json!([9, fold]);
    rejects_domain(&unequal, "element-wise over one exact shape");

    // A replay of a key no draw consumes.
    let mut replay = key_chain();
    let g = push(
        &mut replay,
        json!({"kind":"load","name":"g"}),
        &[],
        &[3, 4],
        "f32",
    );
    let replayed = push(
        &mut replay,
        json!({"kind":"dropout_replay"}),
        &[g, 8, 11],
        &[3, 4],
        "f32",
    );
    replay["roots"] = json!([9, replayed]);
    rejects_domain(&replay, "that no forward random primitive consumes");
}

/// Rule V3 on the wire: two draws share a key only under exclusive arms.
#[test]
fn the_codec_admits_exclusive_arms_and_rejects_overlapping_ones() {
    let arms = |exclusive: bool| {
        let mut graph = key_chain();
        let condition = push(
            &mut graph,
            json!({"kind":"load","name":"c"}),
            &[],
            &[],
            "bool",
        );
        let other = if exclusive {
            push(
                &mut graph,
                json!({"kind":"logical","logical":"not"}),
                &[condition],
                &[],
                "bool",
            )
        } else {
            condition
        };
        let x = push(
            &mut graph,
            json!({"kind":"load","name":"y"}),
            &[],
            &[4],
            "f32",
        );
        let first = push(
            &mut graph,
            json!({"kind":"dropout"}),
            &[x, 8, 11, condition],
            &[4],
            "f32",
        );
        let second = push(
            &mut graph,
            json!({"kind":"dropout"}),
            &[x, 8, 11, other],
            &[4],
            "f32",
        );
        graph["roots"] = json!([9, first, second]);
        graph
    };
    accepts(&arms(true));
    rejects_domain(&arms(false), "whose activations are not exclusive");
}

/// Rule V3 on the wire after constant folding: an activation that is the
/// `bool` constant `false` never draws, so it is exclusive with any other
/// activation; two `true` constants are not.
#[test]
fn the_codec_admits_a_constant_false_arm_and_rejects_two_true_ones() {
    let arms = |first: bool, second: bool| {
        let mut graph = key_chain();
        let mut constant = |value: bool| {
            push(
                &mut graph,
                json!({"kind":"const","value":{"dtype":"bool","value":value}}),
                &[],
                &[],
                "bool",
            )
        };
        let (first, second) = (constant(first), constant(second));
        let x = push(
            &mut graph,
            json!({"kind":"load","name":"y"}),
            &[],
            &[4],
            "f32",
        );
        let a = push(
            &mut graph,
            json!({"kind":"dropout"}),
            &[x, 8, 11, first],
            &[4],
            "f32",
        );
        let b = push(
            &mut graph,
            json!({"kind":"dropout"}),
            &[x, 8, 11, second],
            &[4],
            "f32",
        );
        graph["roots"] = json!([9, a, b]);
        graph
    };
    accepts(&arms(true, false));
    accepts(&arms(false, true));
    accepts(&arms(false, false));
    rejects_domain(&arms(true, true), "whose activations are not exclusive");
}

/// spec/10 §3.2: a key is read only by the draw that consumes it and that
/// draw's replays, and a `DrawKey` carries its draw and that draw's controls.
/// The codec applies the IR verifier's key rules, so it rejects each payload
/// below exactly when `verify` would.
///
/// Evidentiary status: REGRESSION TEST. At 7d0b996ca `from_validated_json`
/// accepted all five payloads.
#[test]
fn a_replay_reads_only_its_forward_draws_key_under_that_draws_controls() {
    let source = concat!(
        "def loss(x: tensor[4, f32]) -> f32 ! { Random } = tensor_to_scalar(sum(dropout(x, 0.5f32), 0i32))\n",
        "def sample(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = with seed(7i64) {\n",
        "  dead = dropout(copy(x), 0.25f32)\n",
        "  u = uniform_like(copy(x), 0.0f32, 1.0f32)\n",
        "  g = grad(loss)(x)\n",
        "  (g, u)\n",
        "}\n"
    );
    let dag = lower(source, "sample");
    accepts(&dag);
    let replay = first(&dag, "dropout_replay");
    let uniform = first(&dag, "uniform_like");
    let forward_key = input(&dag, replay, 2);
    let uniform_key = input(&dag, uniform, 3);
    // The discarded dropout's key: scoped, and neither the replay's nor the
    // uniform draw's.
    let dead_key = dag["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .position(|(id, node)| {
            node["op"]["kind"] == "draw_key"
                && node["op"]["handler"]["kind"] == "scoped"
                && id != forward_key
                && id != uniform_key
        })
        .expect("the discarded dropout keeps its scoped draw key");
    let dead_rate = input(&dag, dead_key, 1);

    let mut replay_reads_uniform_key = dag.clone();
    replay_reads_uniform_key["nodes"][replay]["inputs"][2] = json!(uniform_key);
    rejects_domain(&replay_reads_uniform_key, "changes its forward node");

    let mut replay_changes_rate = dag.clone();
    replay_changes_rate["nodes"][replay]["inputs"][1] = json!(dead_rate);
    rejects_domain(&replay_changes_rate, "changes its forward node");

    let mut key_validates_other_rate = dag.clone();
    key_validates_other_rate["nodes"][forward_key]["inputs"][1] = json!(dead_rate);
    rejects_domain(&key_validates_other_rate, "does not validate the controls");

    let mut replay_reads_unconsumed_key = dag.clone();
    replay_reads_unconsumed_key["nodes"][replay]["inputs"][2] = json!(dead_key);
    rejects_domain(
        &replay_reads_unconsumed_key,
        "that no forward random primitive consumes",
    );

    let mut uniform_consumes_dropout_key = dag.clone();
    uniform_consumes_dropout_key["nodes"][uniform]["inputs"][3] = json!(dead_key);
    rejects_domain(
        &uniform_consumes_dropout_key,
        "does not validate the controls",
    );
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
    rejects_domain(
        &integer_rate,
        "random control must be a value of the draw's dtype",
    );

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
    rejects_domain(&malformed, "exactly one Bool activation");
}
