//! #2530/#2531: every DAG input is validated at entry in ABI slot order.

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict, eval_tensor_with_strict};
use chelis_types::{RawTensor, finalize_tensor, types::Prim};

fn typed(shape: Vec<usize>, dtype: Prim) -> TensorValue {
    let count = shape.iter().product::<usize>();
    let raw = if dtype.is_float() {
        RawTensor::Float(vec![1.0; count])
    } else {
        RawTensor::Int(vec![1; count])
    };
    TensorValue::from_storage(shape, finalize_tensor("load", dtype, raw).unwrap())
}

fn load(dag: &mut Dag, name: &str, dims: Vec<DimInfo>, dtype: Prim) -> chelis_ir::dag::NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        TensorType {
            dims,
            precision: dtype,
        },
        None,
    )
}

fn assert_one_trap(error: &str, context: &str, dtype: Prim) {
    assert!(error.contains(context), "{error}");
    let traps = error
        .lines()
        .filter(|line| line.starts_with("numeric trap:"))
        .collect::<Vec<_>>();
    let expected = format!("numeric trap: domain in load at {}", dtype.name());
    assert_eq!(traps, [expected.as_str()], "{error}");
}

#[test]
fn scalar_rank_is_checked_before_load_in_whole_and_selected_eval() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec![], Prim::F32);
    dag.add_root(x);
    for selected in [false, true] {
        let input = || Some(typed(vec![4], Prim::F32));
        let error = if selected {
            eval_tensor_roots_with_strict(&dag, &[x], |_| input()).unwrap_err()
        } else {
            eval_tensor_with_strict(&dag, |_| input()).unwrap_err()
        };
        assert_one_trap(&error, "input `x` expected rank 0, got 1", Prim::Int64);
        assert_eq!(
            error,
            "input `x` expected rank 0, got 1\nnumeric trap: domain in load at i64"
        );
    }
    let good = eval_tensor_with_strict(&dag, |_| Some(typed(vec![], Prim::F32))).unwrap();
    assert_eq!(good[&x].shape, Vec::<usize>::new());
    assert_eq!(good[&x].to_f64_lossy_vec(), [1.0]);
}

#[test]
fn first_invalid_input_is_first_abi_slot_for_dtype_and_extent() {
    // Node order intentionally disagrees with lexicographic label order.
    let mut dag = Dag::new();
    let z = load(&mut dag, "z", vec![DimInfo::Lit(2)], Prim::F32);
    let a = load(&mut dag, "a", vec![DimInfo::Lit(2)], Prim::F32);
    dag.add_root(z);
    dag.add_root(a);

    let error = eval_tensor_with_strict(&dag, |name| {
        Some(typed(
            vec![2],
            if name == "z" { Prim::F64 } else { Prim::Int32 },
        ))
    })
    .unwrap_err();
    assert_one_trap(
        &error,
        "input `z` carries dtype f64 but the Load declares f32",
        Prim::F32,
    );
    assert_eq!(
        error,
        "input `z` carries dtype f64 but the Load declares f32; provide a value with the declared dtype (casts are explicit in Chelis)\nnumeric trap: domain in load at f32"
    );
    assert!(!error.contains("input `a`"), "{error}");

    // A later slot's shape failure cannot jump ahead of the first slot's
    // dtype failure merely because the lane groups checks by kind.
    let error = eval_tensor_with_strict(&dag, |name| {
        Some(if name == "z" {
            typed(vec![2], Prim::F64)
        } else {
            typed(vec![3], Prim::F32)
        })
    })
    .unwrap_err();
    assert_one_trap(
        &error,
        "input `z` carries dtype f64 but the Load declares f32",
        Prim::F32,
    );
    assert!(!error.contains("a axis"), "{error}");

    let error = eval_tensor_with_strict(&dag, |_| Some(typed(vec![3], Prim::F32))).unwrap_err();
    assert_one_trap(&error, "z axis 0 = 3", Prim::Int64);
    assert_eq!(
        error,
        "extent `2`: claimed = 2, z axis 0 = 3\nnumeric trap: domain in load at i64"
    );
    assert!(!error.contains("a axis"), "{error}");

    let good = eval_tensor_with_strict(&dag, |_| Some(typed(vec![2], Prim::F32))).unwrap();
    assert_eq!(good[&z].shape, [2]);
    assert_eq!(good[&a].shape, [2]);
    assert_eq!(good[&z].to_f64_lossy_vec(), [1.0, 1.0]);
    assert_eq!(good[&a].to_f64_lossy_vec(), [1.0, 1.0]);
}

#[test]
fn selected_root_uses_its_live_load_declaration_for_a_reused_name() {
    let mut dag = Dag::new();
    let dead = load(&mut dag, "x", vec![], Prim::F64);
    let live = load(&mut dag, "x", vec![DimInfo::Lit(2)], Prim::F32);
    dag.add_root(dead);
    dag.add_root(live);

    let good = eval_tensor_roots_with_strict(&dag, &[live], |_| Some(typed(vec![2], Prim::F32)))
        .expect("the unselected declaration must not reject the selected input");
    assert_eq!(good[&live].shape, [2]);
    assert_eq!(good[&live].to_f64_lossy_vec(), [1.0, 1.0]);

    let error = eval_tensor_roots_with_strict(&dag, &[live], |_| Some(typed(vec![], Prim::F32)))
        .unwrap_err();
    assert_one_trap(&error, "input `x` expected rank 1, got 0", Prim::Int64);

    let error = eval_tensor_roots_with_strict(&dag, &[live], |_| Some(typed(vec![2], Prim::F64)))
        .unwrap_err();
    assert_one_trap(
        &error,
        "input `x` carries dtype f64 but the Load declares f32",
        Prim::F32,
    );
}

#[test]
fn selected_load_reuses_its_original_abi_slot() {
    let mut dag = Dag::new();
    let dead_x = load(&mut dag, "x", vec![], Prim::F64);
    let y = load(&mut dag, "y", vec![DimInfo::Lit(2)], Prim::F32);
    let live_x = load(&mut dag, "x", vec![DimInfo::Lit(2)], Prim::F32);
    dag.add_root(dead_x);
    dag.add_root(y);
    dag.add_root(live_x);

    // Resolution visits y before the selected x. Its validation still runs
    // after x, whose first occurrence assigned ABI slot zero.
    let error =
        eval_tensor_roots_with_strict(&dag, &[y, live_x], |_| Some(typed(vec![2], Prim::F64)))
            .unwrap_err();
    assert_one_trap(
        &error,
        "input `x` carries dtype f64 but the Load declares f32",
        Prim::F32,
    );
    assert!(!error.contains("input `y`"), "{error}");
}

#[test]
fn unverified_conflicting_live_loads_cannot_reuse_one_admitted_dtype() {
    let mut dag = Dag::new();
    let first = load(&mut dag, "x", vec![DimInfo::Lit(2)], Prim::F32);
    let second = load(&mut dag, "x", vec![DimInfo::Lit(2)], Prim::F64);
    dag.add_root(first);
    dag.add_root(second);
    assert!(
        chelis_ir::verify::verify(&dag)
            .iter()
            .any(|error| error.contains("load 'x' has inconsistent tensor types"))
    );

    for input in [
        typed(vec![2], Prim::F32),
        TensorValue::from_vec(vec![2], vec![1.0, 1.0]),
    ] {
        let error = eval_tensor_with_strict(&dag, |_| Some(input.clone())).unwrap_err();
        assert_one_trap(
            &error,
            "input `x` carries dtype f32 but the Load declares f64",
            Prim::F64,
        );
    }
}
