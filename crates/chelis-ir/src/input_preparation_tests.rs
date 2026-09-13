//! Test-first contract for additive load preparation. Registered from evaluation.rs.
use super::*;
use crate::dag::{DimInfo, TensorType};
use crate::eval::{
    eval_tensor_plan_with_strict, eval_tensor_roots_with_strict, prepare_tensor_plan_inputs,
    prepare_tensor_roots_inputs,
};
use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
use chelis_types::types::Prim;

fn value() -> TensorValue {
    TensorValue::from_storage(
        vec![2],
        finalize_tensor("test", Prim::F32, RawTensor::Float(vec![3.0, 5.0])).unwrap(),
    )
}

fn ty() -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::F32,
    }
}

fn loads(names: &[&str]) -> Dag {
    let mut dag = Dag::new();
    for name in names {
        dag.add_node(
            RiscOp::Load {
                name: (*name).into(),
            },
            vec![],
            ty(),
            None,
        );
    }
    dag
}

fn context(seed: u64) -> RandomExecutionContext {
    RandomExecutionContext::new(RandomLoweringState {
        seed: Some(seed),
        counter: 5,
    })
}

fn plan(mut dag: Dag) -> EvaluationPlan {
    let mut metadata = ExecutionMetadata::new(Some(42));
    let x = NodeId(0);
    let out = dag.add_node(
        RiscOp::Dropout {
            rate: 0.5,
            seed: 42,
        },
        vec![x],
        ty(),
        None,
    );
    metadata.forward(out, ScopeId(0));
    dag.add_root(out);
    metadata.spine.record_nodes(&dag);
    metadata.complete(&dag).unwrap();
    EvaluationPlan::new(dag, metadata).unwrap()
}

#[test]
fn input_preparation_roots_preserve_values_order_dedup_and_empty_selection() {
    let mut dag = loads(&["x", "unused", "x"]);
    let root = dag.add_node(RiscOp::Add, vec![NodeId(0), NodeId(2)], ty(), None);
    for roots in [vec![root], vec![]] {
        let mut old_calls = Vec::new();
        let old = eval_tensor_roots_with_strict(&dag, &roots, |name| {
            old_calls.push(name.to_owned());
            Some(value())
        })
        .unwrap();
        let mut calls = Vec::new();
        let prepared = prepare_tensor_roots_inputs(&dag, &roots, |name| {
            calls.push(name.to_owned());
            Ok(Some(value()))
        })
        .unwrap();
        assert_eq!(calls, old_calls);
        assert_eq!(
            calls,
            if roots.is_empty() {
                vec!["x", "unused"]
            } else {
                vec!["x"]
            }
        );
        let actual =
            eval_tensor_roots_with_strict(&dag, &roots, |name| prepared.get(name).cloned())
                .unwrap();
        assert_eq!(actual[&root], old[&root]);
        assert_eq!(actual[&root].storage(), old[&root].storage());
        assert_eq!(
            calls, old_calls,
            "executing the map cannot re-enter the provider"
        );
    }
}

#[test]
fn input_preparation_shape_dependencies_remain_required() {
    let mut dag = loads(&["shape", "unused"]);
    let root = dag.add_node(
        RiscOp::synth_const(Prim::F32, 7.0),
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_shape_dep(root, NodeId(0));
    let mut calls = Vec::new();
    let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
        calls.push(name.to_owned());
        Ok(Some(value()))
    })
    .unwrap();
    assert_eq!(calls, ["shape"]);
    assert!(prepared.contains_key("shape"));
    let old = eval_tensor_roots_with_strict(&dag, &[root], |_| None).unwrap_err();
    assert_eq!(
        prepare_tensor_roots_inputs(&dag, &[root], |_| Ok(None)).unwrap_err(),
        old
    );
    assert_eq!(old, "missing required input `shape`");
}

#[test]
fn input_preparation_optional_absence_is_not_a_strict_load_error() {
    let mut dag = Dag::new();
    let symbolic = TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    for _ in 0..2 {
        dag.add_node(
            RiscOp::Load {
                name: "optional".into(),
            },
            vec![],
            symbolic.clone(),
            None,
        );
    }
    let live = dag.add_node(
        RiscOp::Load {
            name: "live".into(),
        },
        vec![],
        symbolic,
        None,
    );
    for supplied in [false, true] {
        let mut old_calls = Vec::new();
        let old = eval_tensor_roots_with_strict(&dag, &[live], |name| {
            old_calls.push(name.to_owned());
            (name == "live" || supplied).then(value)
        });
        let mut calls = Vec::new();
        let prepared = prepare_tensor_roots_inputs(&dag, &[live], |name| {
            calls.push(name.to_owned());
            Ok((name == "live" || supplied).then(value))
        })
        .unwrap();
        assert_eq!(calls, old_calls);
        assert_eq!(
            calls,
            if supplied {
                vec!["optional", "live"]
            } else {
                vec!["optional", "optional", "live"]
            }
        );
        let actual =
            eval_tensor_roots_with_strict(&dag, &[live], |name| prepared.get(name).cloned());
        assert_eq!(actual, old);
    }
}

#[test]
fn input_preparation_defers_missing_optional_dimension_to_inference() {
    let mut dag = Dag::new();
    let symbolic = TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    dag.add_node(
        RiscOp::Load {
            name: "shape".into(),
        },
        vec![],
        symbolic.clone(),
        None,
    );
    let one = dag.add_node(
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let root = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: crate::dag::RtDim::Sym("n".into()),
        },
        vec![one],
        symbolic,
        None,
    );
    let mut calls = Vec::new();
    let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
        calls.push(name.to_owned());
        Ok(None)
    })
    .unwrap();
    assert_eq!(calls, ["shape"]);
    assert!(prepared.is_empty());
    let old = eval_tensor_roots_with_strict(&dag, &[root], |_| None).unwrap_err();
    let actual = eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned())
        .unwrap_err();
    assert_eq!(actual, old);
    assert!(actual.contains("symbolic dimension"), "{actual}");
}

#[test]
fn input_preparation_preserves_to_end_binding_without_symbolic_dimensions() {
    let mut dag = loads(&["x"]);
    let root = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(crate::dag::RtDim::Lit(0), crate::dag::RtDim::ToEnd)],
        },
        vec![NodeId(0)],
        ty(),
        None,
    );
    let old = eval_tensor_roots_with_strict(&dag, &[root], |_| Some(value())).unwrap();
    let prepared = prepare_tensor_roots_inputs(&dag, &[root], |_| Ok(Some(value()))).unwrap();
    let actual =
        eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned()).unwrap();
    assert_eq!(actual, old);
    assert_eq!(actual[&root], value());
}

#[test]
fn input_preparation_provider_errors_and_missing_live_inputs_stop_in_order() {
    let dag = loads(&["first", "broken", "later"]);
    let mut calls = Vec::new();
    let error = prepare_tensor_roots_inputs(&dag, &[], |name| {
        calls.push(name.to_owned());
        if name == "broken" {
            Err("initializer failed exactly".into())
        } else {
            Ok(Some(value()))
        }
    })
    .unwrap_err();
    assert_eq!(error, "initializer failed exactly");
    assert_eq!(calls, ["first", "broken"]);
    let mut old_calls = Vec::new();
    let old = eval_tensor_roots_with_strict(&dag, &[], |name| {
        old_calls.push(name.to_owned());
        (name == "first").then(value)
    })
    .unwrap_err();
    calls.clear();
    let missing = prepare_tensor_roots_inputs(&dag, &[], |name| {
        calls.push(name.to_owned());
        Ok((name == "first").then(value))
    })
    .unwrap_err();
    assert_eq!(missing, old);
    assert_eq!(calls, old_calls);
    let plan = plan(dag);
    calls.clear();
    let error = prepare_tensor_plan_inputs(&plan, &context(42), |name| {
        calls.push(name.to_owned());
        if name == "broken" {
            Err("initializer failed exactly".into())
        } else {
            Ok(Some(value()))
        }
    })
    .unwrap_err();
    assert_eq!(error, "initializer failed exactly");
    assert_eq!(calls, ["first", "broken"]);
}

#[test]
fn input_preparation_leaves_rank_and_numeric_errors_to_evaluation() {
    let dag = loads(&["x"]);
    for input in [
        TensorValue::from_storage(
            vec![],
            finalize_tensor("test", Prim::F32, RawTensor::Float(vec![3.0])).unwrap(),
        ),
        TensorValue::from_storage(
            vec![1],
            finalize_tensor("test", Prim::F32, RawTensor::Float(vec![3.0])).unwrap(),
        ),
        TensorValue::from_storage(
            vec![2],
            finalize_tensor("test", Prim::F16, RawTensor::Float(vec![3.0, 5.0])).unwrap(),
        ),
    ] {
        let old = eval_tensor_roots_with_strict(&dag, &[], |_| Some(input.clone())).unwrap_err();
        let prepared = prepare_tensor_roots_inputs(&dag, &[], |_| Ok(Some(input.clone()))).unwrap();
        assert_eq!(
            prepared["x"].storage(),
            input.storage(),
            "preparation preserves the tagged input"
        );
        let actual = eval_tensor_roots_with_strict(&dag, &[], |name| prepared.get(name).cloned())
            .unwrap_err();
        assert_eq!(actual, old);
    }
}

#[test]
fn input_preparation_drop_and_bad_axis_fail_before_provider_effects() {
    let mut dropped = loads(&["x"]);
    let root = dropped.add_node(RiscOp::Drop, vec![NodeId(0)], ty(), None);
    let mut bad_axis = loads(&["x"]);
    let bad = bad_axis.add_node(
        RiscOp::Reshape { new_shape: vec![] },
        vec![NodeId(0)],
        ty(),
        None,
    );
    for (dag, root) in [(dropped, root), (bad_axis, bad)] {
        let old = eval_tensor_roots_with_strict(&dag, &[root], |_| {
            panic!("old validation precedes inputs")
        })
        .unwrap_err();
        let actual = prepare_tensor_roots_inputs(&dag, &[root], |_| {
            panic!("preparation validation precedes inputs")
        })
        .unwrap_err();
        assert_eq!(actual, old);
    }
}

#[test]
fn input_preparation_plan_retains_dead_executed_loads_and_random_state() {
    let plan = plan(loads(&["x", "executed_but_value_dead"]));
    let mut old_context = context(42);
    let mut old_calls = Vec::new();
    let old = eval_tensor_plan_with_strict(&plan, &mut old_context, |name| {
        old_calls.push(name.to_owned());
        Some(value())
    })
    .unwrap();
    let mut new_context = context(42);
    let mut calls = Vec::new();
    let prepared = prepare_tensor_plan_inputs(&plan, &new_context, |name| {
        calls.push(name.to_owned());
        Ok(Some(value()))
    })
    .unwrap();
    assert_eq!(
        new_context.state().counter,
        5,
        "no execution frame enters during preparation"
    );
    assert_eq!(calls, ["x", "executed_but_value_dead"]);
    assert_eq!(calls, old_calls);
    let actual =
        eval_tensor_plan_with_strict(&plan, &mut new_context, |name| prepared.get(name).cloned())
            .unwrap();
    assert_eq!(actual, old);
    assert_eq!(new_context.state().counter, old_context.state().counter);
    assert_eq!(new_context.state().counter, 6);
    assert_eq!(calls, old_calls);
}

#[test]
fn input_preparation_plan_seed_and_integrity_precede_provider_effects() {
    let valid = plan(loads(&["x"]));
    let mut invalid = valid.clone();
    invalid.metadata.spine.steps.pop();
    for (plan, seed) in [(valid, 43), (invalid, 42)] {
        let mut context = context(seed);
        let old = eval_tensor_plan_with_strict(&plan, &mut context, |_| {
            panic!("old validation precedes inputs")
        })
        .unwrap_err();
        let actual = prepare_tensor_plan_inputs(&plan, &context, |_| {
            panic!("preparation validation precedes inputs")
        })
        .unwrap_err();
        assert_eq!(actual, old);
        assert_eq!(context.state().counter, 5);
    }
}

#[test]
fn input_preparation_provider_draw_precedes_original_plan_without_relowering() {
    let initializer = plan(loads(&["seed_input"]));
    let consumer = plan(loads(&["weights"]));
    let initializer_root = initializer.dag_for_inspection().roots()[0];
    let mut reference_context = context(42);
    let initialized =
        eval_tensor_plan_with_strict(&initializer, &mut reference_context, |_| Some(value()))
            .unwrap();
    let expected = eval_tensor_plan_with_strict(&consumer, &mut reference_context, |_| {
        Some(initialized[&initializer_root].clone())
    })
    .unwrap();
    let wrong_ordinal = eval_tensor_plan_with_strict(&consumer, &mut context(42), |_| {
        Some(initialized[&initializer_root].clone())
    })
    .unwrap();
    assert_ne!(
        expected, wrong_ordinal,
        "fixture distinguishes consumer ordinals 5 and 6"
    );

    let mut execution_context = context(42);
    let before_provider = execution_context.clone();
    let mut calls = 0;
    let prepared = prepare_tensor_plan_inputs(&consumer, &before_provider, |name| {
        assert_eq!(name, "weights");
        calls += 1;
        eval_tensor_plan_with_strict(&initializer, &mut execution_context, |_| Some(value()))
            .map(|values| Some(values[&initializer_root].clone()))
    })
    .unwrap();
    assert_eq!(before_provider.state().counter, 5);
    assert_eq!(execution_context.state().counter, 6);
    let actual = eval_tensor_plan_with_strict(&consumer, &mut execution_context, |name| {
        prepared.get(name).cloned()
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(actual, expected);
    for root in consumer.dag_for_inspection().roots() {
        assert_eq!(actual[root].storage(), expected[root].storage());
    }
    assert_eq!(
        execution_context.state().seed,
        reference_context.state().seed
    );
    assert_eq!(
        execution_context.state().counter,
        reference_context.state().counter
    );
    assert_eq!(execution_context.state().counter, 7);
}
