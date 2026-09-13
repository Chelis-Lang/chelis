//! Private lowering ownership: numerical root order, independent of API repacking.
use chelis_ir::{
    dag::{DimInfo, TensorType},
    eval::{TensorValue, eval_tensor_roots_with_strict},
    lower::try_lower_subexpr_program,
};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

fn pair_roots(selection: &str, batched: bool) -> Vec<Vec<f64>> {
    let function = "(fn {} (params {} (x {type: (t-tensor {} (t-prim {} f32))}) (w {type: (t-tensor {} (t-prim {} f32))})) (app {} (var {} mul) (var {} x) (var {} w)))";
    let grad = format!("(grad {{}} {function}{selection})");
    let transform = if batched {
        format!("(vmap {{}} {grad} 0)")
    } else {
        grad
    };
    let source = format!("(app {{}} {transform} (var {{}} a) (var {{}} b))");
    let expr = chelis_deep::parser::parse_str(&source).unwrap().remove(0);
    let shape = if batched { vec![2] } else { vec![] };
    let ty = TensorType {
        dims: shape.iter().copied().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let inputs: UnordMap<String, TensorValue> = UnordMap::from([
        (
            "a".into(),
            TensorValue::from_vec(
                shape.clone(),
                if batched { vec![2.0, 7.0] } else { vec![2.0] },
            ),
        ),
        (
            "b".into(),
            TensorValue::from_vec(
                shape.clone(),
                if batched { vec![3.0, 11.0] } else { vec![3.0] },
            ),
        ),
    ]);
    let dag = try_lower_subexpr_program(
        &expr,
        UnordMap::from([("a".into(), ty.clone()), ("b".into(), ty)]),
        UnordMap::new(),
        UnordMap::new(),
    )
    .expect("valid direct grad lowers");
    let values =
        eval_tensor_roots_with_strict(&dag, dag.roots(), |name| inputs.get(name).cloned()).unwrap();
    dag.roots()
        .iter()
        .map(|root| {
            assert_eq!(values[root].shape, shape);
            values[root].to_f64_lossy_vec()
        })
        .collect()
}

#[test]
fn direct_gradient_roots_follow_index_sequence() {
    assert_eq!(pair_roots(" (tuple {} 1 0)", false), [vec![2.0], vec![3.0]]);
    assert_eq!(
        pair_roots(" (tuple {} 1 1 0)", false),
        [vec![2.0], vec![2.0], vec![3.0]]
    );
}

#[test]
fn default_and_single_selection_controls() {
    assert_eq!(pair_roots("", false), [vec![3.0], vec![2.0]]);
    assert_eq!(pair_roots(" 1", false), [vec![2.0]]);
}

#[test]
fn fused_gradient_roots_follow_index_sequence() {
    assert_eq!(
        pair_roots(" (tuple {} 1 0)", true),
        [vec![2.0, 7.0], vec![3.0, 11.0]]
    );
    assert_eq!(
        pair_roots(" (tuple {} 1 1 0)", true),
        [vec![2.0, 7.0], vec![2.0, 7.0], vec![3.0, 11.0]]
    );
}
