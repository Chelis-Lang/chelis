//! chelis#3464, spec/06 §7.5: the compiler API's `grad` returns one gradient
//! entry per requested parameter. A parameter read only through a control
//! slot, or not at all, receives no contribution, and its entry is the exact
//! positive zero of its shape and dtype, as Surf lowering returns.
use std::collections::BTreeMap;

use chelis_compiler_api::schema::{GradRequest, SourceKind};
use serde_json::{Value, json};

fn grad(source: &str, wrt: &[&str]) -> (Value, BTreeMap<String, u64>) {
    let grad = chelis_compiler_api::compiler::grad(GradRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        output_name: "loss".into(),
        wrt_names: wrt.iter().map(|name| (*name).to_string()).collect(),
        fuse: false,
    })
    .unwrap_or_else(|error| panic!("{error:?}"));
    (
        serde_json::to_value(&grad.dag).unwrap(),
        grad.grad_nodes_by_name,
    )
}

/// The node `name`'s gradient is a rank-0 f32 constant with the bits of +0.
fn assert_positive_zero(dag: &Value, names: &BTreeMap<String, u64>, name: &str) {
    let node = &dag["nodes"][names[name] as usize];
    assert_eq!(
        node["op"],
        json!({"kind": "const", "value": {"bits": "00000000", "dtype": "f32"}}),
        "{name}: {node}"
    );
}

#[test]
fn a_predicate_only_parameter_has_a_positive_zero_entry() {
    let (dag, names) = grad(
        "x = (x : tensor[f32])\n\
         y = (y : tensor[f32])\n\
         loss = (mul(x, cast(lt(y, scalar_to_tensor(2.0f32)), f32)) : tensor[f32])\n",
        &["x", "y"],
    );
    assert_eq!(
        names.keys().collect::<Vec<_>>(),
        ["x", "y"],
        "every requested parameter has an entry"
    );
    assert_positive_zero(&dag, &names, "y");
}

#[test]
fn a_disconnected_parameter_has_a_positive_zero_entry() {
    let (dag, names) = grad(
        "x = (x : tensor[f32])\n\
         z = (z : tensor[f32])\n\
         loss = (mul(x, x) : tensor[f32])\n",
        &["x", "z"],
    );
    assert_eq!(names.keys().collect::<Vec<_>>(), ["x", "z"]);
    assert_positive_zero(&dag, &names, "z");
}
