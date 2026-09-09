//! spec/10 §3.4: result reference maps select only their own DAG's nodes.
use chelis_compiler_api::schema::{
    EvaluatedRoot, GradResult, LowerResult, WIRE_DAG_SCHEMA_VERSION,
};
use serde_json::json;

fn dag() -> serde_json::Value {
    json!({"schema_version":WIRE_DAG_SCHEMA_VERSION,"nodes":[{"shape_deps":[],"span_id":null,"merged_spans":[],"id":0,"inputs":[],"op":{"kind":"load","name":"x"},"output_type":{"dims":[],"precision":"f32"}}],"roots":[0]})
}

#[test]
fn lower_reference_names_do_not_change_the_owner_scope() {
    let value = json!({"dag":dag(),"named_roots":{"x":0}});
    let mut lower: LowerResult = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&lower).unwrap(), value);
    for id in [1, u64::MAX] {
        let mut invalid = value.clone();
        invalid["named_roots"]["x"] = json!(id);
        assert!(serde_json::from_value::<LowerResult>(invalid).is_err());
        lower.named_roots.insert("x".into(), id);
        assert!(serde_json::to_string(&lower).is_err());
    }
}

#[test]
fn all_gradient_result_references_select_the_enclosed_dag() {
    let value = json!({"dag":dag(),"output_node":0,"grad_nodes_by_name":{"dx":0},"forward_nodes_by_name":{"x":0}});
    let mut result: GradResult = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&result).unwrap(), value);
    let mut invalid = value.clone();
    invalid["output_node"] = json!(1);
    assert!(serde_json::from_value::<GradResult>(invalid).is_err());
    for field in ["grad_nodes_by_name", "forward_nodes_by_name"] {
        let mut invalid = value.clone();
        invalid[field] = json!({"renamed":u64::MAX});
        assert!(serde_json::from_value::<GradResult>(invalid).is_err());
    }
    result.output_node = 1;
    assert!(serde_json::to_string(&result).is_err());
}

#[test]
fn evaluated_root_identity_remains_opaque_without_its_graph() {
    let value = json!({"node_id":u64::MAX,"value":{"type":"unit"}});
    let root: EvaluatedRoot = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(root).unwrap(), value);
    let mut invalid = value;
    invalid["node_id"] = json!(-1);
    assert!(serde_json::from_value::<EvaluatedRoot>(invalid).is_err());
}
