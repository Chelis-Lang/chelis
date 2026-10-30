use chelis_dag_core::{Dag, DagNode, DimInfo, LoadStoreName, NodeId, RiscOp, TensorType};
use chelis_numeric_core::dtype_semantics::{
    ScalarValue, TensorStorage, scalar_from_i64, tensor_from_scalars,
};
use chelis_numeric_core::prim::Prim;

#[test]
fn checked_extent_owner_is_explicit_and_closed_in_graph_codecs() {
    for owner in ["reshape", "concat"] {
        let json = serde_json::json!({"CheckedExtent": {
            "owner": owner, "claims": ["rows"], "axis": {"Lit": 1}
        }});
        let op: RiscOp = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&op).unwrap(), json);
        let binary = bincode::serialize(&op).unwrap();
        let decoded: RiscOp = bincode::deserialize(&binary).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), json);

        let mut missing = json.clone();
        missing["CheckedExtent"]
            .as_object_mut()
            .unwrap()
            .remove("owner");
        assert!(serde_json::from_value::<RiscOp>(missing).is_err());
        let mut unknown = json;
        unknown["CheckedExtent"]["owner"] = serde_json::json!("expand");
        assert!(serde_json::from_value::<RiscOp>(unknown).is_err());
    }
}

#[test]
fn node_json_and_positional_binary_fields_stay_stable() {
    let node = DagNode {
        id: NodeId(1),
        op: RiscOp::Neg,
        inputs: vec![NodeId(0)],
        output_type: TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        },
        reusable_input: Some(NodeId(0)),
        span_id: Some("op".into()),
        merged_spans: vec!["extra".into()],
        shape_deps: vec![NodeId(0)],
        result_claim_deps: vec![],
        owner: chelis_dag_core::dag::Owner::unconditional(chelis_dag_core::dag::DeclId(0)),
    };
    let json = r#"{"id":1,"op":"Neg","inputs":[0],"output_type":{"dims":[{"Lit":3}],"precision":"F32"},"reusable_input":0,"span_id":"op","merged_spans":["extra"],"shape_deps":[0],"result_claim_deps":[],"owner":{"decl":0,"activation":null}}"#;
    assert_eq!(serde_json::to_string(&node).unwrap(), json);
    let bytes = bincode::serialize(&(
        NodeId(1),
        RiscOp::Neg,
        vec![NodeId(0)],
        TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        },
        Some(NodeId(0)),
        Some(String::from("op")),
        vec![String::from("extra")],
        vec![NodeId(0)],
        Vec::<NodeId>::new(),
        chelis_dag_core::dag::Owner::unconditional(chelis_dag_core::dag::DeclId(0)),
    ))
    .unwrap();
    assert_eq!(bincode::serialize(&node).unwrap(), bytes);
    let decoded: DagNode = bincode::deserialize(&bytes).unwrap();
    assert_eq!(serde_json::to_string(&decoded).unwrap(), json);
    let old_json = r#"{"id":0,"op":"Neg","inputs":[],"output_type":{"dims":[],"precision":"F32"},"reusable_input":null,"owner":{"decl":0,"activation":null}}"#;
    let decoded: DagNode = serde_json::from_str(old_json).unwrap();
    assert_eq!(decoded.span_id, None);
    assert!(decoded.merged_spans.is_empty());
    assert!(decoded.shape_deps.is_empty());
    assert!(decoded.result_claim_deps.is_empty());
    assert!(
        serde_json::from_str::<DagNode>(
            &old_json.replace(",\"owner\":{\"decl\":0,\"activation\":null}", "")
        )
        .is_err()
    );
}

#[test]
fn graph_roots_names_and_numeric_payloads_survive_both_codecs() {
    let exact = 9_007_199_254_740_993_i64;
    let scalar = scalar_from_i64("fixture", Prim::Int64, exact).unwrap();
    let mut bytes = 4_u32.to_le_bytes().to_vec();
    bytes.extend(exact.to_le_bytes());
    assert_eq!(bincode::serialize(&scalar).unwrap(), bytes);
    assert_eq!(
        serde_json::to_string(&scalar).unwrap(),
        r#"{"dtype":"int64","value":9007199254740993}"#
    );
    let decoded: ScalarValue = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded.as_i64_exact(), Some(exact));
    let mut dag = Dag::new();
    dag.declare("fixture");
    let ty = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let x = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load {
            name: "grads.1".into(),
        },
        vec![],
        ty.clone(),
        Some("x".into()),
    );
    let y = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Const { value: scalar },
        vec![],
        ty,
        Some("c".into()),
    );
    dag.set_roots(vec![y, x]);
    let json = serde_json::to_string(&dag).unwrap();
    assert_eq!(
        json,
        r#"{"nodes":[{"id":0,"op":{"Load":{"name":"grads.1"}},"inputs":[],"output_type":{"dims":[],"precision":"Int64"},"reusable_input":null,"span_id":"x","merged_spans":[],"shape_deps":[],"result_claim_deps":[],"owner":{"decl":0,"activation":null}},{"id":1,"op":{"Const":{"value":{"dtype":"int64","value":9007199254740993}}},"inputs":[],"output_type":{"dims":[],"precision":"Int64"},"reusable_input":null,"span_id":"c","merged_spans":[],"shape_deps":[],"result_claim_deps":[],"owner":{"decl":0,"activation":null}}],"roots":[1,0],"declarations":[{"name":"fixture","value":false}]}"#
    );
    let bytes = bincode::serialize(&dag).unwrap();
    for decoded in [
        serde_json::from_str::<Dag>(&json).unwrap(),
        bincode::deserialize::<Dag>(&bytes).unwrap(),
    ] {
        assert_eq!(decoded.roots(), &[y, x]);
        assert_eq!(serde_json::to_string(&decoded).unwrap(), json);
        assert_eq!(bincode::serialize(&decoded).unwrap(), bytes);
        assert!(serde_json::from_str::<Dag>(&json.replace("grads.1", "grads/1")).is_err());
        let mut invalid = bytes.clone();
        let position = invalid
            .windows(7)
            .position(|part| part == b"grads.1")
            .unwrap();
        invalid[position + 5] = b'/';
        assert!(bincode::deserialize::<Dag>(&invalid).is_err());
    }
}

#[test]
fn checked_name_codecs_reject_every_untrusted_byte_class() {
    let names = [
        "", "1x", ".x", "-x", "x\u{1}", "x\u{7f}", "x\0", "x\n", "x\r", "x%", "x/", "é", "xé",
    ];
    for name in names {
        assert!(LoadStoreName::new(name).is_err(), "{name:?}");
        assert!(
            serde_json::from_str::<LoadStoreName>(&serde_json::to_string(name).unwrap()).is_err(),
            "{name:?}"
        );
        assert!(
            bincode::deserialize::<LoadStoreName>(&bincode::serialize(name).unwrap()).is_err(),
            "{name:?}"
        );
    }
    for name in ["x", "_x", "grads.1", "x_1"] {
        let checked = LoadStoreName::new(name).unwrap();
        assert_eq!(checked.as_str(), name);
        assert_eq!(
            serde_json::to_string(&checked).unwrap(),
            serde_json::to_string(name).unwrap()
        );
        assert_eq!(
            bincode::serialize(&checked).unwrap(),
            bincode::serialize(name).unwrap()
        );
    }
}

#[test]
fn tensor_payload_bytes_keep_exact_integer_storage() {
    let values = [
        scalar_from_i64("fixture", Prim::Int64, 9_007_199_254_740_993).unwrap(),
        scalar_from_i64("fixture", Prim::Int64, -7).unwrap(),
    ];
    let tensor = tensor_from_scalars(Prim::Int64, &values);
    assert_eq!(
        serde_json::to_string(&tensor).unwrap(),
        r#"{"dtype":"int64","values":[9007199254740993,-7]}"#
    );
    let mut bytes = 4_u32.to_le_bytes().to_vec();
    bytes.extend(2_u64.to_le_bytes());
    bytes.extend(9_007_199_254_740_993_i64.to_le_bytes());
    bytes.extend((-7_i64).to_le_bytes());
    assert_eq!(bincode::serialize(&tensor).unwrap(), bytes);
    let decoded: TensorStorage = bincode::deserialize(&bytes).unwrap();
    assert_eq!(
        decoded.to_i64_exact_vec(),
        Some(vec![9_007_199_254_740_993, -7])
    );
    assert!(serde_json::from_str::<ScalarValue>(r#"{"dtype":"f16","value":0.1}"#).is_err());
    assert!(serde_json::from_str::<TensorStorage>(r#"{"dtype":"f16","values":[0.1]}"#).is_err());
}

#[test]
fn top_level_names_keep_binding_identity_and_checked_wire_labels() {
    use chelis_dag_core::load_store_name::LoadBinding;
    for source in ["x", "grads.1", "module.value", "x-y"] {
        let name = LoadStoreName::top_level(source);
        assert_eq!(name.binding(), LoadBinding::TopLevel(source));
        assert_ne!(name, LoadStoreName::new(source).unwrap());
        let expected = format!(
            "@chelis_global_{}",
            source
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        assert_eq!(name.as_str(), expected);
        let json = serde_json::to_string(&name).unwrap();
        let binary = bincode::serialize(&name).unwrap();
        assert_eq!(json, serde_json::to_string(&expected).unwrap());
        assert_eq!(binary, bincode::serialize(&expected).unwrap());
        for decoded in [
            serde_json::from_str::<LoadStoreName>(&json).unwrap(),
            bincode::deserialize::<LoadStoreName>(&binary).unwrap(),
        ] {
            assert_eq!(decoded, name);
            assert_eq!(decoded.binding(), LoadBinding::TopLevel(source));
        }
    }
    for label in [
        "@chelis_global_",
        "@chelis_global_7",
        "@chelis_global_zz",
        "@chelis_global_ff",
        "@chelis_global_2f",
        "@chelis_global_4A",
    ] {
        assert!(
            serde_json::from_str::<LoadStoreName>(&serde_json::to_string(label).unwrap()).is_err()
        );
        assert!(
            bincode::deserialize::<LoadStoreName>(&bincode::serialize(label).unwrap()).is_err()
        );
    }
}

#[test]
fn checked_names_cover_all_ascii_control_bytes_and_graph_codecs() {
    for byte in 0_u8..=127 {
        let character = char::from(byte);
        let allowed = byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.');
        let text = format!("x{character}");
        assert_eq!(LoadStoreName::new(&text).is_ok(), allowed, "{byte}");
        assert_eq!(
            serde_json::from_str::<LoadStoreName>(&serde_json::to_string(&text).unwrap()).is_ok(),
            allowed,
            "{byte}"
        );
        assert_eq!(
            bincode::deserialize::<LoadStoreName>(&bincode::serialize(&text).unwrap()).is_ok(),
            allowed,
            "{byte}"
        );
        let first = format!("{character}x");
        let first_allowed = byte.is_ascii_alphabetic() || byte == b'_';
        assert_eq!(
            LoadStoreName::new(&first).is_ok(),
            first_allowed,
            "first {byte}"
        );
        assert_eq!(
            serde_json::from_str::<LoadStoreName>(&serde_json::to_string(&first).unwrap()).is_ok(),
            first_allowed,
            "first {byte}"
        );
        assert_eq!(
            bincode::deserialize::<LoadStoreName>(&bincode::serialize(&first).unwrap()).is_ok(),
            first_allowed,
            "first {byte}"
        );
    }
}

#[test]
fn load_and_store_codecs_enforce_the_checked_name_boundary() {
    for invalid in [
        "", "1x", "x\0", "x\n", "x\r", "x%", "x/", "é", "xé", "x\u{7f}",
    ] {
        for (tag, op) in [
            ("Load", RiscOp::Load { name: "x".into() }),
            ("Store", RiscOp::Store { name: "x".into() }),
        ] {
            let json = serde_json::json!({(tag): {"name": invalid}});
            assert!(
                serde_json::from_value::<RiscOp>(json).is_err(),
                "{tag}: {invalid:?}"
            );
            let mut bytes = bincode::serialize(&op).unwrap()[..4].to_vec();
            bytes.extend(bincode::serialize(invalid).unwrap());
            assert!(
                bincode::deserialize::<RiscOp>(&bytes).is_err(),
                "{tag}: {invalid:?}"
            );
        }
    }
}
