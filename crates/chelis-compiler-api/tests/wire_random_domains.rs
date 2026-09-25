//! spec/10 §§3.2–3.4 and [05-OP-8/37]: random nodes carry no fields; their
//! controls and key are operands. A key is produced by a key operation or
//! enters as a key-precision `Load`, and is consumed at most once; adjoint
//! replays read a key without consuming it, and no key is a shape dependency
//! or a constant. These tests exercise codecs and object admission, not
//! random kernel output.
use chelis_compiler_api::schema::{
    CheckRequest, GradRequest, LowerRequest, SourceKind, WIRE_DAG_SCHEMA_VERSION, WireDag,
    WireDagDecodeError, WireDagNode,
};
use chelis_ir::dag::{Dag, DimInfo, KeyBranch, NodeId, RiscOp, RtDim, TensorType, UniformBound};
use chelis_types::scalar_from_i64;
use chelis_types::types::Prim;
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

/// A runtime-rate dropout and a uniform draw, each keyed by its own key
/// parameter: two key-precision `Load`s, each consumed by its primitive.
const KEYED: &str = "def sample(k: key, j: key, x: tensor[2, f32], rate: f32) -> (tensor[2, f32], tensor[2, f32]) = (dropout(k, copy(x), rate), uniform_like(j, x, 0.0f32, 1.0f32))\n";

#[test]
fn lowered_key_operand_draws_round_trip_with_operand_controls() {
    let dag = lower(KEYED, "sample");
    accepts(&dag);
    let dropout = first(&dag, "dropout");
    let uniform = first(&dag, "uniform_like");
    assert_eq!(dag["nodes"][dropout]["op"], json!({"kind":"dropout"}));
    assert_eq!(dag["nodes"][uniform]["op"], json!({"kind":"uniform_like"}));
    assert_eq!(dag["nodes"][dropout]["inputs"].as_array().unwrap().len(), 3);
    assert_eq!(dag["nodes"][uniform]["inputs"].as_array().unwrap().len(), 4);
    let dropout_key = &dag["nodes"][input(&dag, dropout, 2)];
    let uniform_key = &dag["nodes"][input(&dag, uniform, 3)];
    for (key, name) in [(dropout_key, "k"), (uniform_key, "j")] {
        assert_eq!(key["op"], json!({"kind":"load","name":name}));
        assert_eq!(key["inputs"], json!([]));
        assert_eq!(key["output_type"], json!({"dims":[],"precision":"key"}));
    }
    // The runtime rate reaches the primitive as its operand.
    assert_eq!(
        dag["nodes"][input(&dag, dropout, 1)]["op"]["kind"],
        json!("load")
    );
}

#[test]
fn version_16_random_payloads_have_no_version_17_spelling() {
    let dag = lower(KEYED, "sample");
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
    let dag = lower(KEYED, "sample");
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

    // A drawn key derived from as well: a derivation consumes its parent
    // ([05-OP-70]), so the draw and the split are two uses.
    let mut derived = dag.clone();
    let id = derived["nodes"].as_array().unwrap().len();
    derived["nodes"].as_array_mut().unwrap().push(
        json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":id,
        "op":{"kind":"split","branch":"left"},"inputs":[dropout_key],
        "output_type":{"dims":[],"precision":"key"}}),
    );
    derived["roots"].as_array_mut().unwrap().push(json!(id));
    rejects_domain(&derived, "is consumed twice");
}

/// A key computed inside a lowered definition is used once: the draw that
/// consumes it is its only reader, and no compiler-inserted `Drop` reads it
/// again (a `Drop` of a key would be a second use and a key reaching an
/// operation other than a key consumer). Covers `key_from_seed` and
/// `fold_in`.
///
/// Evidentiary status: REGRESSION TEST. At d1b33808f `lower` fails at the
/// schema stage ("node N produces a key, but only a key operation or a Load
/// produces one"): the implicit-drop pass drops every unconsumed non-`Load`
/// value, and slice 3 removed the `DrawKey` exclusion without excluding the
/// key operations that replaced it.
#[test]
fn a_computed_key_is_consumed_by_its_draw_alone() {
    let source = "def sample(k: key, x: tensor[2, f32], rate: f32) -> (tensor[2, f32], tensor[2, f32]) = (dropout(key_from_seed(7i64), copy(x), rate), uniform_like(fold_in(k, 3i64), x, 0.0f32, 1.0f32))\n";
    let dag = lower(source, "sample");
    accepts(&dag);
    for (draw, slot, producer) in [
        ("dropout", 2, "key_from_seed"),
        ("uniform_like", 3, "fold_in"),
    ] {
        let draw = first(&dag, draw);
        let key = input(&dag, draw, slot);
        assert_eq!(dag["nodes"][key]["op"]["kind"], producer);
        let readers = dag["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|node| node["inputs"].as_array().unwrap().contains(&json!(key)))
            .count();
        assert_eq!(readers, 1, "key {key} has another reader: {dag}");
    }
}

/// The readers of `node`'s value in the wire graph.
fn readers(dag: &Value, node: usize) -> Vec<usize> {
    dag["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, reader)| reader["inputs"].as_array().unwrap().contains(&json!(node)))
        .map(|(index, _)| index)
        .collect()
}

/// The sibling of [`a_computed_key_is_consumed_by_its_draw_alone`] for a key
/// tensor and for the `grad` export. A `split_keys` key tensor is read by
/// its batched draw alone, and a computed key in a differentiated program is
/// read by its draw and by that draw's replay (a replay read is not a use),
/// never by a `Drop`: a key is never dropped, at any rank.
///
/// Evidentiary status: the `lower` half is a REGRESSION TEST: at `6f42b4e92`
/// it fails at the schema stage ("node N produces a key, but only a key
/// operation or a Load produces one"). The `grad` half is a DISPOSITION LOCK:
/// it passes there too, because the gradient graph is rebuilt without the
/// forward graph's `Drop`s.
#[test]
fn a_computed_key_tensor_and_a_gradient_export_keep_one_consumer_per_key() {
    let source = "def noisy(r: key, x: tensor[4, f32]) -> tensor[4, f32] = dropout(r, x, 0.5f32)\n\
                  def sample(k: key, xs: tensor[3, 4, f32]) -> tensor[3, 4, f32] = vmap(noisy)(split_keys(k, 3i64), xs)\n";
    let dag = lower(source, "sample");
    accepts(&dag);
    let split_n = first(&dag, "split_n");
    assert_eq!(dag["nodes"][split_n]["output_type"]["precision"], "key");
    assert_eq!(
        dag["nodes"][split_n]["output_type"]["dims"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let keys_readers = readers(&dag, split_n);
    assert_eq!(keys_readers.len(), 1, "{dag}");
    assert_eq!(dag["nodes"][keys_readers[0]]["op"]["kind"], "dropout");

    let grad = chelis_compiler_api::compiler::grad(GradRequest {
        source_kind: SourceKind::Surf,
        source: "x = (x : tensor[4, f32])\n\
                 loss = (sum(dropout(fold_in(key_from_seed(7i64), 1i64), x, 0.5f32), 0i32) : tensor[f32])\n"
            .into(),
        output_name: "loss".into(),
        wrt_names: vec!["x".into()],
        fuse: false,
    })
    .unwrap_or_else(|error| panic!("{error:?}"));
    let grad = serde_json::to_value(grad.dag).unwrap();
    accepts(&grad);
    let fold = first(&grad, "fold_in");
    let mut kinds = readers(&grad, fold)
        .into_iter()
        .map(|reader| {
            grad["nodes"][reader]["op"]["kind"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>();
    kinds.sort();
    assert_eq!(kinds, ["dropout", "dropout_replay"], "{grad}");
}

/// spec/10 §3.2: a key enters a graph as a key operation's output or as a
/// key-precision `Load`, and may be a root; it is never a shape dependency,
/// and no constant carries one.
#[test]
fn a_key_may_be_loaded_or_rooted_and_is_never_a_dependency_or_a_constant() {
    let dag = lower(KEYED, "sample");
    let dropout = first(&dag, "dropout");
    let key = input(&dag, dropout, 2);

    // A key-precision Load, unconsumed: dropping a key is allowed.
    let mut loaded = dag.clone();
    let id = loaded["nodes"].as_array().unwrap().len();
    loaded["nodes"].as_array_mut().unwrap().push(
        json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":id,
        "op":{"kind":"load","name":"unused"},"inputs":[],
        "output_type":{"dims":[],"precision":"key"}}),
    );
    accepts(&loaded);

    // The loaded key as a root. A root is a key's one use, so a key its draw
    // consumed is no root.
    let mut loaded_root = loaded.clone();
    loaded_root["roots"].as_array_mut().unwrap().push(json!(id));
    accepts(&loaded_root);
    let mut key_root = dag.clone();
    key_root["roots"].as_array_mut().unwrap().push(json!(key));
    rejects_domain(&key_root, "is a graph root and is also consumed");

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

/// Insert an input-free node as node 0, renumbering every later reference,
/// so an existing consumer may read it as an earlier node.
fn prepend(graph: &mut Value, op: Value, dims: &[u64], precision: &str) -> usize {
    let shift = |value: &mut Value| {
        *value = json!(value.as_u64().unwrap() + 1);
    };
    for node in graph["nodes"].as_array_mut().unwrap() {
        shift(&mut node["id"]);
        for input in node["inputs"].as_array_mut().unwrap() {
            shift(input);
        }
        for dependency in node["shape_deps"].as_array_mut().unwrap() {
            shift(dependency);
        }
    }
    for root in graph["roots"].as_array_mut().unwrap() {
        shift(root);
    }
    graph["nodes"]
        .as_array_mut()
        .unwrap()
        .insert(0, wire_node(0, op, &[], dims, precision));
    0
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
/// draw's replays, and a replay reads its forward draw's own controls. The
/// codec applies the IR verifier's key rules, so it rejects each payload
/// below exactly when `verify` would.
///
/// Evidentiary status: REGRESSION TEST for the replay payloads (at 7d0b996ca
/// `from_validated_json` accepted them); the counter-stream draw key's own
/// control check has no key-form subject and is gone with `DrawKey`.
#[test]
fn a_replay_reads_only_its_forward_draws_key_under_that_draws_controls() {
    let source = concat!(
        "def loss(k: key, x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(dropout(k, x, 0.5f32), 0i32))\n",
        "def sample(g: key, u: key, x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = (grad(loss, wrt=x)(g, copy(x)), uniform_like(u, x, 0.0f32, 1.0f32))\n"
    );
    let dag = lower(source, "sample");
    accepts(&dag);
    let replay = first(&dag, "dropout_replay");
    let uniform = first(&dag, "uniform_like");
    let forward_key = input(&dag, replay, 2);
    let uniform_key = input(&dag, uniform, 3);
    assert_eq!(
        dag["nodes"][forward_key]["op"],
        json!({"kind":"load","name":"g"})
    );
    let forward = dag["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .position(|node| node["op"]["kind"] == "dropout" && node["inputs"][2] == json!(forward_key))
        .expect("the replay's forward dropout consumes the same key");
    assert_eq!(input(&dag, replay, 1), input(&dag, forward, 1));

    let mut replay_reads_uniform_key = dag.clone();
    replay_reads_uniform_key["nodes"][replay]["inputs"][2] = json!(uniform_key);
    rejects_domain(&replay_reads_uniform_key, "changes its forward node");

    // Another rate than the forward draw's, even of equal value.
    let mut replay_changes_rate = dag.clone();
    let other_rate = prepend(
        &mut replay_changes_rate,
        json!({"kind":"const","value":{"dtype":"f32","bits":"3f000000"}}),
        &[],
        "f32",
    );
    replay_changes_rate["nodes"][replay + 1]["inputs"][1] = json!(other_rate);
    rejects_domain(&replay_changes_rate, "changes its forward node");

    // A loaded key no forward draw consumes.
    let mut replay_reads_unconsumed_key = dag.clone();
    let unconsumed = prepend(
        &mut replay_reads_unconsumed_key,
        json!({"kind":"load","name":"unconsumed"}),
        &[],
        "key",
    );
    replay_reads_unconsumed_key["nodes"][replay + 1]["inputs"][2] = json!(unconsumed);
    rejects_domain(
        &replay_reads_unconsumed_key,
        "that no forward random primitive consumes",
    );

    // A replay read is not a use, but a second draw on the forward key is.
    let mut uniform_consumes_dropout_key = dag.clone();
    uniform_consumes_dropout_key["nodes"][uniform]["inputs"][3] = json!(forward_key);
    rejects_domain(&uniform_consumes_dropout_key, "is consumed twice");
}

#[test]
fn random_controls_seeds_and_keys_keep_their_structural_types() {
    let dag = lower(KEYED, "sample");
    let dropout = first(&dag, "dropout");

    // The rate is a rank-zero value of the data dtype, never an integer.
    let mut integer_rate = dag.clone();
    let seed = prepend(
        &mut integer_rate,
        json!({"kind":"const","value":{"dtype":"int64","value":7}}),
        &[],
        "int64",
    );
    integer_rate["nodes"][dropout + 1]["inputs"][1] = json!(seed);
    rejects_domain(
        &integer_rate,
        "random control must be a value of the draw's dtype",
    );

    // [05-OP-69]: a seed is an int64 tensor, never the draw's float data.
    let mut float_seed = key_chain();
    let float = prepend(
        &mut float_seed,
        json!({"kind":"const","value":{"dtype":"f32","bits":"40e00000"}}),
        &[],
        "f32",
    );
    // `key_chain`'s node 1 is its `key_from_seed`, now node 2.
    float_seed["nodes"][2]["inputs"][0] = json!(float);
    rejects_domain(&float_seed, "key operation reads the wrong operand dtype");

    // A key operation produces a key of its input's shape, never a number.
    let mut numeric_key = key_chain();
    numeric_key["nodes"][2]["output_type"]["precision"] = json!("f32");
    rejects_domain(&numeric_key, "does not produce keys");

    // A missing key operand is a malformed primitive.
    let mut keyless = dag.clone();
    keyless["nodes"][dropout]["inputs"]
        .as_array_mut()
        .unwrap()
        .truncate(2);
    rejects_domain(&keyless, "missing an input");
}

#[test]
fn gradient_random_lowering_admits_only_a_scalar_bool_activation() {
    // spec/06 §2.11 and [05-OP-8]: differentiating through a keyed draw
    // keeps its key a single-use operand, and a draw's optional path
    // activation is exactly one Bool of rank zero on every encode/decode
    // path; a consumer cannot replace it with a numeric scalar.
    let source = concat!(
        "def loss(k: key, x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(copy(x), uniform_like(k, x, 0.0f32, 1.0f32)), 0))\n",
        "def derivative(g: key, x: tensor[2, f32]) -> tensor[2, f32] = grad(loss, wrt=x)(g, x)\n"
    );
    let dag = lower(source, "derivative");
    accepts(&dag);
    // The differentiated copy draws once, keyed by `derivative`'s own key,
    // and its backward pass reads the forward value rather than redrawing.
    let draws = dag["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, node)| node["op"]["kind"] == "uniform_like")
        .map(|(id, _)| id)
        .collect::<Vec<_>>();
    let keyed_by_g = draws
        .iter()
        .copied()
        .filter(|draw| {
            dag["nodes"][input(&dag, *draw, 3)]["op"] == json!({"kind":"load","name":"g"})
        })
        .collect::<Vec<_>>();
    assert_eq!(keyed_by_g.len(), 1, "{draws:?}");
    let uniform = keyed_by_g[0];

    let mut active = dag.clone();
    let activation = prepend(
        &mut active,
        json!({"kind":"load","name":"active"}),
        &[],
        "bool",
    );
    active["nodes"][uniform + 1]["inputs"]
        .as_array_mut()
        .unwrap()
        .push(json!(activation));
    accepts(&active);

    let mut malformed = active.clone();
    malformed["nodes"][activation]["output_type"]["precision"] = json!("f32");
    rejects_domain(&malformed, "exactly one Bool activation");
}

// ---- one operand rule on both sides of the codec ----

/// Axis `name` of unknown extent, or a literal extent.
enum Axis {
    Named(&'static str),
    Lit(usize),
}

fn tensor(axes: &[Axis], precision: Prim) -> TensorType {
    TensorType {
        dims: axes
            .iter()
            .map(|axis| match axis {
                Axis::Named(name) => DimInfo::Named((*name).into(), None),
                Axis::Lit(extent) => DimInfo::Lit(*extent),
            })
            .collect(),
        precision,
    }
}

fn ir_node(dag: &mut Dag, op: RiscOp, inputs: Vec<NodeId>, ty: TensorType) -> NodeId {
    dag.add_node(op, inputs, ty, None)
}

/// The wire form of the loads, constants, key operations and draws below.
fn wire_of(dag: &Dag) -> Value {
    let nodes = dag
        .nodes()
        .iter()
        .map(|node| {
            let op = match &node.op {
                RiscOp::Load { name } => json!({"kind":"load","name":name.as_str()}),
                RiscOp::Const { value } if value.prim() == Prim::Int64 => {
                    json!({"kind":"const","value":{"dtype":"int64","value":value.as_i64_exact().unwrap()}})
                }
                RiscOp::Const { value } => json!({"kind":"const","value":{"dtype":"f32",
                    "bits":format!("{:08x}", (value.as_f64_lossy() as f32).to_bits())}}),
                RiscOp::KeyFromSeed => json!({"kind":"key_from_seed"}),
                RiscOp::Split { branch } => json!({"kind":"split","branch":match branch {
                    KeyBranch::Left => "left",
                    KeyBranch::Right => "right",
                }}),
                RiscOp::SplitN {
                    count: RtDim::Lit(count),
                } => json!({"kind":"split_n","count":{"bound":"lit","value":count}}),
                RiscOp::UniformLike => json!({"kind":"uniform_like"}),
                RiscOp::UniformBoundAdjoint { .. } => {
                    json!({"kind":"uniform_bound_adjoint","bound":"high"})
                }
                other => panic!("no wire form here for {other:?}"),
            };
            let dims = node
                .output_type
                .dims
                .iter()
                .map(|dim| match dim {
                    DimInfo::Named(name, None) => json!({"kind":"named","name":name,"size":null}),
                    DimInfo::Lit(size) => json!({"kind":"lit","size":size}),
                    other => panic!("no wire form here for {other:?}"),
                })
                .collect::<Vec<_>>();
            json!({"shape_deps":[],"span_id":null,"merged_spans":[],"id":node.id.0,"op":op,
                "inputs":node.inputs.iter().map(|input| input.0).collect::<Vec<_>>(),
                "output_type":{"dims":dims,"precision":node.output_type.precision.interchange_name()}})
        })
        .collect::<Vec<_>>();
    let roots = dag.roots().iter().map(|root| root.0).collect::<Vec<_>>();
    json!({"schema_version": WIRE_DAG_SCHEMA_VERSION, "nodes": nodes, "roots": roots})
}

/// Keys `split_n(key(7), 3)`: nodes 0 to 2.
fn three_keys(dag: &mut Dag) -> NodeId {
    let seed = ir_node(
        dag,
        RiscOp::Const {
            value: scalar_from_i64("test", Prim::Int64, 7).unwrap(),
        },
        vec![],
        tensor(&[], Prim::Int64),
    );
    let key = ir_node(dag, RiscOp::KeyFromSeed, vec![seed], tensor(&[], Prim::Key));
    ir_node(
        dag,
        RiscOp::SplitN {
            count: RtDim::Lit(3),
        },
        vec![key],
        tensor(&[Axis::Lit(3)], Prim::Key),
    )
}

fn f32_bound(dag: &mut Dag, value: f64) -> NodeId {
    ir_node(
        dag,
        RiscOp::synth_const(Prim::F32, value),
        vec![],
        tensor(&[], Prim::F32),
    )
}

/// Round 3's witness: a `UniformLike` over `t: [3, 4]` declared `[rows, 4]`.
fn uniform_declaring(rows: usize) -> Dag {
    let mut dag = Dag::new();
    let keys = three_keys(&mut dag);
    let t = ir_node(
        &mut dag,
        RiscOp::Load { name: "t".into() },
        vec![],
        tensor(&[Axis::Lit(3), Axis::Lit(4)], Prim::F32),
    );
    let low = f32_bound(&mut dag, 0.0);
    let high = f32_bound(&mut dag, 1.0);
    let drawn = ir_node(
        &mut dag,
        RiscOp::UniformLike,
        vec![t, low, high, keys],
        tensor(&[Axis::Lit(rows), Axis::Lit(4)], Prim::F32),
    );
    dag.add_root(drawn);
    dag
}

/// A `Split` of `k: [n]` declared `[out]`.
fn split_declaring(out: &'static str) -> Dag {
    let mut dag = Dag::new();
    let k = ir_node(
        &mut dag,
        RiscOp::Load { name: "k".into() },
        vec![],
        tensor(&[Axis::Named("n")], Prim::Key),
    );
    let split = ir_node(
        &mut dag,
        RiscOp::Split {
            branch: KeyBranch::Left,
        },
        vec![k],
        tensor(&[Axis::Named(out)], Prim::Key),
    );
    dag.add_root(split);
    dag
}

/// A forward `UniformLike` over `t: [3, 5]` and its bound adjoint over a
/// cotangent `g: [3, trailing]`.
fn adjoint_over(trailing: usize) -> Dag {
    let mut dag = Dag::new();
    let keys = three_keys(&mut dag);
    let t = ir_node(
        &mut dag,
        RiscOp::Load { name: "t".into() },
        vec![],
        tensor(&[Axis::Lit(3), Axis::Lit(5)], Prim::F32),
    );
    let low = f32_bound(&mut dag, 0.0);
    let high = f32_bound(&mut dag, 1.0);
    let forward = ir_node(
        &mut dag,
        RiscOp::UniformLike,
        vec![t, low, high, keys],
        tensor(&[Axis::Lit(3), Axis::Lit(5)], Prim::F32),
    );
    let g = ir_node(
        &mut dag,
        RiscOp::Load { name: "g".into() },
        vec![],
        tensor(&[Axis::Lit(3), Axis::Lit(trailing)], Prim::F32),
    );
    let adjoint = ir_node(
        &mut dag,
        RiscOp::UniformBoundAdjoint {
            bound: UniformBound::High,
        },
        vec![t, g, keys],
        tensor(&[], Prim::F32),
    );
    dag.add_root(forward);
    dag.add_root(adjoint);
    dag
}

/// Round 3 found the IR verifier and the wire decoder applying different
/// operand rules in three places. Both now call `verify_random_operands`,
/// so each divergent graph gets the same sentence, node number included,
/// from `verify` and from the decoder, and each agreeing graph passes both.
///
/// Evidentiary status: REGRESSION TEST. At c23ec448a `verify` accepted all
/// three divergent graphs, and the decoder's reports named no node.
#[test]
fn the_verifier_and_the_codec_share_one_operand_rule() {
    let cases = [
        (
            uniform_declaring(2),
            uniform_declaring(3),
            "node 6: uniform_like must preserve its float template's exact shape and dtype",
        ),
        (
            split_declaring("m"),
            split_declaring("n"),
            "node 1: key operation must be element-wise over one exact shape, with a split's count axis appended last",
        ),
        (
            adjoint_over(7),
            adjoint_over(5),
            "node 8: a uniform bound adjoint is a value of its template's dtype, shaped like a leading part of its key's shape, over a cotangent of its template's exact type",
        ),
    ];
    for (divergent, agreeing, sentence) in cases {
        let errors = chelis_ir::verify::verify(&divergent);
        assert!(errors.iter().any(|error| error == sentence), "{errors:?}");
        rejects_domain(&wire_of(&divergent), sentence);
        assert_eq!(chelis_ir::verify::verify(&agreeing), Vec::<String>::new());
        accepts(&wire_of(&agreeing));
    }
}
