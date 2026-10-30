//! spec/04 §3.1 and §4.5.2: selected constructor contracts own their equations.
use super::*;

// Exact connection-closure traversal from parent 799d8f580. Its output order
// is consumed by raw result-origin solving; do not normalize the oracle.
fn parent_sweep(
    mut available: Vec<ResultConstraint<ErrorWitness>>,
    body: &Type<ErrorWitness>,
) -> Vec<ResultConstraint<ErrorWitness>> {
    let mut visible = chelis_type_core::free_tvars(body);
    let mut equations = Vec::new();
    loop {
        let before = equations.len();
        available.retain(|equation| {
            let variables: Vec<_> = equation
                .types()
                .into_iter()
                .flat_map(chelis_type_core::free_tvars)
                .collect();
            if variables.iter().any(|variable| visible.contains(variable)) {
                visible.extend(variables);
                equations.push(equation.clone());
                false
            } else {
                true
            }
        });
        if equations.len() == before {
            break;
        }
    }
    equations
}

#[test]
fn indexed_reachability_matches_parent_sweep_on_four_vertex_graphs() {
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for mask in 0..64 {
        for reverse in [false, true] {
            let mut equations = edges
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << *index) != 0)
                .map(|(index, &(a, b))| {
                    let actual = Type::Var(TypeVar(a));
                    let declared = Type::Var(TypeVar(b));
                    if index % 2 == 0 {
                        ResultConstraint::Annotation { actual, declared }
                    } else {
                        ResultConstraint::Join {
                            inputs: vec![actual],
                            result: declared,
                        }
                    }
                })
                .collect::<Vec<_>>();
            if reverse {
                equations.reverse();
            }
            let subst = Subst::new();
            for equation in &equations {
                subst.record_result_constraint(equation.clone());
            }
            let mut product = InferenceProduct::default();
            product.import_result_constraints(&subst);
            // Both implementations collect annotations before join entries.
            let available = equations
                .iter()
                .filter(|equation| matches!(equation, ResultConstraint::Annotation { .. }))
                .chain(
                    equations
                        .iter()
                        .filter(|equation| matches!(equation, ResultConstraint::Join { .. })),
                )
                .cloned()
                .collect::<Vec<_>>();
            for root in 0..=4 {
                let body = Type::Var(TypeVar(root));
                assert_eq!(
                    product.result_equations_for(&body, &subst),
                    parent_sweep(available.clone(), &body),
                    "mask={mask}, reverse={reverse}, root={root}"
                );
            }
        }
    }
}

#[test]
fn staged_constructor_publication_preserves_parent_child_provenance_order() {
    let variable = |id| Type::Var(TypeVar(id));
    let parent_join = ResultConstraint::Join {
        inputs: vec![variable(2)],
        result: variable(3),
    };
    let child_join = ResultConstraint::Join {
        inputs: vec![variable(1)],
        result: variable(2),
    };
    let parent_annotation = ResultConstraint::Annotation {
        actual: variable(4),
        declared: variable(5),
    };
    let child_annotation = ResultConstraint::Annotation {
        actual: variable(1),
        declared: variable(4),
    };
    let mut product = InferenceProduct::default();
    let subst = Subst::new();
    let parent = product.stage_constructor_result_constraints(
        vec![parent_join.clone(), parent_annotation.clone()],
        &subst,
    );
    let child = product.stage_constructor_result_constraints(
        vec![child_join.clone(), child_annotation.clone()],
        &subst,
    );
    assert!(
        product
            .result_equations_for(&variable(3), &subst)
            .is_empty()
    );
    product.commit_constructor_result_constraints(child);
    assert!(
        product
            .result_equations_for(&variable(3), &subst)
            .is_empty()
    );
    product.commit_constructor_result_constraints(parent);
    let actual = product.result_equations_for(&variable(3), &subst);
    let expected = parent_sweep(
        vec![
            parent_annotation.clone(),
            child_annotation.clone(),
            parent_join.clone(),
            child_join.clone(),
        ],
        &variable(3),
    );
    assert_eq!(actual, expected);
    assert_eq!(
        actual,
        vec![parent_join, child_join, child_annotation, parent_annotation]
    );
}

#[test]
fn indexed_result_reachability_preserves_parent_sweep_discovery_order() {
    let variable = |id| Type::Var(TypeVar(id));
    let first = ResultConstraint::Annotation {
        actual: variable(1),
        declared: variable(2),
    };
    let second = ResultConstraint::Annotation {
        actual: variable(3),
        declared: variable(4),
    };
    let bridge = ResultConstraint::Join {
        inputs: vec![variable(2), variable(4)],
        result: variable(5),
    };
    let subst = Subst::new();
    for equation in [first.clone(), second.clone(), bridge.clone()] {
        subst.record_result_constraint(equation);
    }
    let mut product = InferenceProduct::default();
    product.import_result_constraints(&subst);
    // The result first reaches the join, then the annotations in the next
    // sweep. Emitting annotation-ledger order instead changes the order in
    // which generalization solves the raw result-provenance graph.
    assert_eq!(
        product.result_equations_for(&variable(5), &subst),
        vec![bridge.clone(), first.clone(), second.clone()]
    );
    // A directly visible annotation is processed before its later bridge;
    // newly visible earlier entries wait for the following sweep.
    assert_eq!(
        product.result_equations_for(&variable(3), &subst),
        vec![second, bridge, first]
    );
}

fn equation_chain_work(count: u32, reverse: bool) -> usize {
    let subst = Subst::new();
    let mut equations = (0..count)
        .map(|index| ResultConstraint::Annotation {
            actual: Type::Var(TypeVar(index)),
            declared: Type::Var(TypeVar(index + 1)),
        })
        .collect::<Vec<_>>();
    if reverse {
        equations.reverse();
    }
    for equation in equations {
        subst.record_result_constraint(equation);
    }
    // An unrelated component must never become evidence for the requested type.
    subst.record_result_constraint(ResultConstraint::Annotation {
        actual: Type::Var(TypeVar(count + 10)),
        declared: Type::Var(TypeVar(count + 11)),
    });
    let mut product = InferenceProduct::default();
    product.import_result_constraints(&subst);
    take_result_replay_work();
    let connected = product.result_equations_for(&Type::Var(TypeVar(count)), &subst);
    assert_eq!(connected.len(), count as usize);
    assert!(
        connected
            .iter()
            .flat_map(ResultConstraint::types)
            .all(|ty| { matches!(ty, Type::Var(variable) if variable.0 <= count) })
    );
    take_result_replay_work()[0]
}

#[test]
fn result_equation_reachability_is_independent_of_registration_order() {
    for reverse in [false, true] {
        let small = equation_chain_work(32, reverse);
        let large = equation_chain_work(128, reverse);
        assert!(small > 0);
        assert!(large <= small * 5, "{small} -> {large}, reverse={reverse}");
    }
}

// spec/04 sections 3.1 and 4.5.1: list elements keep rank, precision and
// rigid-axis constraints; a concrete tensor join does not owe a second,
// unbound generic constructor application.
fn tensor_lists(count: usize) -> String {
    let mut source = String::from("def probe[n, c, h, w](x: tensor[n, c, h, w, f32]) -> i64 = {\n");
    for index in 0..count {
        source.push_str(&format!("  values_{index} = [x, x, x]\n"));
    }
    source.push_str("  0i64\n}\n");
    source
}

fn replay_work(source: &str) -> usize {
    take_result_replay_work();
    crate::test_support::run(|operation| checked_surf(source, operation));
    take_result_replay_work().iter().sum()
}

#[test]
fn concrete_tensor_lists_do_not_accumulate_generic_result_replays() {
    let small = replay_work(&tensor_lists(32));
    let large = replay_work(&tensor_lists(128));
    assert!(
        small > 0,
        "generic final Cons still checks its result origin"
    );
    assert!(
        large <= small * 5,
        "32 -> 128 tensor lists: {small} -> {large}"
    );
}

fn checks(source: &str) -> bool {
    let declarations =
        chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    crate::test_support::run(|operation| {
        crate::test_support::check_ir_program(&deep, operation).is_ok()
    })
}

#[test]
fn tensor_join_preserves_rank_precision_and_rigid_axes() {
    assert!(checks(
        "values = [to_tensor([1.0f32]), to_tensor([2.0f32, 3.0f32])]"
    ));
    assert!(!checks(
        "values = [to_tensor([1.0f32]), to_tensor([[2.0f32]])]"
    ));
    assert!(!checks(
        "values = [to_tensor([1.0f32]), to_tensor([2.0f64])]"
    ));
    assert!(!checks(
        "def probe[n, m](x: tensor[n, f32], y: tensor[m, f32]) = [x, y]"
    ));
}

#[test]
fn tensor_join_preserves_singleton_empty_and_wildcard_directions() {
    for (left, right) in [(1, 3), (3, 1), (0, 3), (3, 0), (0, 0)] {
        assert!(checks(&format!(
            "def probe(x: tensor[{left}, f32], y: tensor[{right}, f32]) = [x, y]"
        )));
        if left != right {
            assert!(!checks(&format!(
                "def probe[n](x: tensor[{left}, f32], y: tensor[{right}, f32]) -> List[tensor[n, f32]] = [x, y]"
            )));
        }
    }
    for (left, right) in [("2", "*"), ("*", "2")] {
        assert!(checks(&format!(
            "def probe(x: tensor[{left}, f32], y: tensor[{right}, f32]) = [x, y]"
        )));
        assert!(!checks(&format!(
            "def probe[n](x: tensor[{left}, f32], y: tensor[{right}, f32]) -> List[tensor[n, f32]] = [x, y]"
        )));
    }
}

#[test]
fn constructor_routes_check_eager_producers_and_pending_gradients() {
    assert!(checks(
        "def probe() -> i64 = { values = [to_tensor([1.0f32]), reshape(to_tensor([2.0f32]), [1i64])]\n 0i64\n }"
    ));
    assert!(!checks(
        "def probe() -> i64 = { values = [to_tensor([1.0f32]), reshape(to_tensor([2.0f32]), [2i64])]\n 0i64\n }"
    ));
    assert!(!checks(
        "def probe() = { g = grad(fn (x) -> 1.0f32)\n [g, grad(fn (x: f32) -> 1.0f32)]\n }"
    ));
    assert!(checks(
        "def probe() = { g = grad(fn (x) -> 1.0f32)\n value = g(2.0f32)\n [g, grad(fn (x: f32) -> 1.0f32)]\n }"
    ));
}
