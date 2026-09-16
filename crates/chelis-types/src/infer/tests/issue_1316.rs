use super::*;

#[derive(Clone, Copy)]
enum GraphShape {
    Isolated,
    Chain,
    FanOut,
    BoundedDag,
    Cycles(usize),
}

fn call(name: usize, argument: &str) -> String {
    format!("f{name}({argument})")
}

fn function_body(index: usize, definitions: usize, shape: GraphShape) -> String {
    match shape {
        GraphShape::Isolated => "x".to_string(),
        GraphShape::Chain => {
            if index + 1 < definitions {
                call(index + 1, "x")
            } else {
                "x".to_string()
            }
        }
        GraphShape::FanOut if index == 0 => {
            let mut lines = vec!["{".to_string()];
            for target in 1..definitions {
                lines.push(format!("  y{target} = {}", call(target, "x")));
            }
            lines.push(if definitions > 1 {
                format!("  y{}", definitions - 1)
            } else {
                "  x".to_string()
            });
            lines.push("}".to_string());
            lines.join("\n")
        }
        GraphShape::FanOut => "x".to_string(),
        GraphShape::BoundedDag => {
            let targets = [index + 1, index + 2]
                .into_iter()
                .filter(|target| *target < definitions)
                .collect::<Vec<_>>();
            match targets.as_slice() {
                [] => "x".to_string(),
                [only] => call(*only, "x"),
                [first, second] => {
                    format!("add({}, {})", call(*first, "x"), call(*second, "x"))
                }
                _ => unreachable!(),
            }
        }
        GraphShape::Cycles(groups) => {
            assert!(groups > 0 && definitions.is_multiple_of(groups));
            let group_size = definitions / groups;
            let group_start = (index / group_size) * group_size;
            let target = group_start + ((index - group_start + 1) % group_size);
            format!(
                "if eq(x, 0i32) then x else {}",
                call(target, "sub(x, 1i32)")
            )
        }
    }
}

fn generated_program(definitions: usize, shape: GraphShape) -> Vec<deep::Expr> {
    let mut source = String::from("module Profile.Issue1316\n");
    for index in 0..definitions {
        source.push_str(&format!(
            "def f{index}(x: i32) -> i32 = {}\n",
            function_body(index, definitions, shape)
        ));
    }
    let declarations = chelis_surf::parser::parse_str(&source).expect("generated Surf parses");
    chelis_surf::desugar::desugar_program(&declarations)
}

fn plan_for(program: &[deep::Expr]) -> FunctionInferencePlan {
    let items = top_level_decl_items_with_modules(program);
    FunctionInferencePlan::build(&items)
}

fn component_names(plan: &FunctionInferencePlan) -> Vec<(Vec<String>, bool)> {
    plan.components
        .iter()
        .map(|component| {
            (
                component
                    .members
                    .iter()
                    .map(|member| member.name.clone())
                    .collect(),
                component.recursive,
            )
        })
        .collect()
}

#[test]
fn issue_1316_components_remain_callee_first_and_source_ordered() {
    let chain = plan_for(&generated_program(4, GraphShape::Chain));
    assert_eq!(
        component_names(&chain),
        vec![
            (vec!["f3".into()], false),
            (vec!["f2".into()], false),
            (vec!["f1".into()], false),
            (vec!["f0".into()], false),
        ]
    );

    let cycle = plan_for(&generated_program(4, GraphShape::Cycles(1)));
    assert_eq!(
        component_names(&cycle),
        vec![(
            vec!["f0".into(), "f1".into(), "f2".into(), "f3".into()],
            true,
        )]
    );

    let self_recursive = plan_for(&generated_program(4, GraphShape::Cycles(4)));
    assert!(
        self_recursive
            .components
            .iter()
            .all(|component| { component.members.len() == 1 && component.recursive }),
        "a self edge must retain recursive classification"
    );
}

fn reference_reaches(
    start: usize,
    current: usize,
    graph: &[Vec<usize>],
    visited: &mut chelis_unord::UnordSet<usize>,
) -> bool {
    for &next in &graph[current] {
        if next == start {
            return true;
        }
        if visited.insert(next) && reference_reaches(start, next, graph, visited) {
            return true;
        }
    }
    false
}

fn reference_components(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    use chelis_unord::{UnordMap, UnordSet};

    let mut assigned = UnordSet::new();
    let mut unordered = Vec::new();
    for name in 0..graph.len() {
        if assigned.contains(&name) {
            continue;
        }
        let members = (0..graph.len())
            .filter(|&candidate| {
                candidate == name
                    || (reference_reaches(candidate, name, graph, &mut UnordSet::new())
                        && reference_reaches(name, candidate, graph, &mut UnordSet::new()))
            })
            .collect::<Vec<_>>();
        assigned.extend(members.iter().copied());
        unordered.push(members);
    }

    let mut component_by_vertex = UnordMap::new();
    for (component_index, members) in unordered.iter().enumerate() {
        for member in members {
            component_by_vertex.insert(*member, component_index);
        }
    }
    let mut component_graph = vec![UnordSet::new(); unordered.len()];
    for (caller, callees) in graph.iter().enumerate() {
        for callee in callees {
            let caller_component = component_by_vertex[&caller];
            let callee_component = component_by_vertex[callee];
            if caller_component != callee_component {
                component_graph[caller_component].insert(callee_component);
            }
        }
    }

    fn visit(
        component: usize,
        graph: &[UnordSet<usize>],
        visiting: &mut UnordSet<usize>,
        visited: &mut UnordSet<usize>,
        out: &mut Vec<usize>,
    ) {
        if visited.contains(&component) || !visiting.insert(component) {
            return;
        }
        let dependencies = graph[component]
            .to_sorted()
            .into_iter()
            .copied()
            .collect::<Vec<_>>();
        for dependency in dependencies {
            visit(dependency, graph, visiting, visited, out);
        }
        visiting.remove(&component);
        visited.insert(component);
        out.push(component);
    }

    let mut order = Vec::new();
    let mut visiting = UnordSet::new();
    let mut visited = UnordSet::new();
    for component in 0..unordered.len() {
        visit(
            component,
            &component_graph,
            &mut visiting,
            &mut visited,
            &mut order,
        );
    }
    order
        .into_iter()
        .map(|component| unordered[component].clone())
        .collect()
}

#[test]
fn issue_1316_linear_scc_matches_the_previous_order_on_every_three_vertex_graph() {
    let vertex_count = 3;
    for edge_mask in 0usize..(1 << (vertex_count * vertex_count)) {
        let mut graph = vec![Vec::new(); vertex_count];
        for (caller, callees) in graph.iter_mut().enumerate() {
            for callee in 0..vertex_count {
                let bit = caller * vertex_count + callee;
                if edge_mask & (1 << bit) != 0 {
                    callees.push(callee);
                }
            }
        }
        assert_eq!(
            linear_scc_component_vertices_for_test(&graph),
            reference_components(&graph),
            "component/order drift for edge mask {edge_mask:#x}"
        );
    }
}

#[test]
fn issue_1316_iterative_scc_handles_a_hundred_thousand_vertex_chain() {
    let vertices = 100_000;
    let mut graph = vec![Vec::new(); vertices];
    for (vertex, callees) in graph.iter_mut().enumerate().take(vertices - 1) {
        callees.push(vertex + 1);
    }
    let components = linear_scc_component_vertices_for_test(&graph);
    assert_eq!(components.len(), vertices);
    assert_eq!(components.first(), Some(&vec![vertices - 1]));
    assert_eq!(components.last(), Some(&vec![0]));
}

#[test]
fn issue_1316_mid_scc_cancellation_returns_no_partial_plan() {
    let program = generated_program(64, GraphShape::Chain);
    let token = CancelToken::new();
    let _cancel_guard = crate::cancel::install_cancel_token(token.clone());
    let _plan_hook = cancel_function_plan_after_edges_for_test(17, token);
    reset_function_plan_profile();

    let plan = plan_for(&program);
    let profile = take_function_plan_profile();
    assert!(!plan.complete);
    assert!(plan.components.is_empty());
    assert_eq!(profile.scc_edge_inspections, 17);
}

#[test]
fn issue_1316_planner_work_is_linear_on_generated_graph_axes() {
    for definitions in [16, 32, 64, 128, 256, 512] {
        for (shape, expected_edges) in [
            (GraphShape::Isolated, 0),
            (GraphShape::Chain, definitions - 1),
            (GraphShape::FanOut, definitions - 1),
            (GraphShape::BoundedDag, definitions * 2 - 3),
            (GraphShape::Cycles(definitions / 4), definitions),
        ] {
            reset_function_plan_profile();
            let plan = plan_for(&generated_program(definitions, shape));
            let profile = take_function_plan_profile();
            assert!(plan.complete);
            assert_eq!(profile.plan_builds, 1);
            assert_eq!(profile.reference_graph_builds, 1);
            assert_eq!(profile.graph_vertices, definitions);
            assert_eq!(profile.graph_edges, expected_edges);
            assert_eq!(profile.scc_vertex_entries, definitions);
            assert_eq!(profile.scc_edge_inspections, expected_edges);
        }
    }
}

#[test]
fn issue_1316_each_public_driver_builds_one_shared_plan() {
    let program = generated_program(32, GraphShape::BoundedDag);

    reset_function_plan_profile();
    check_typed_program(&program).expect("primary driver accepts generated DAG");
    let primary_profile = take_function_plan_profile();
    assert_eq!(primary_profile.plan_builds, 1);
    assert_eq!(primary_profile.reference_graph_builds, 1);

    reset_function_plan_profile();
    check_ir_program(&program).expect("IR driver accepts generated DAG");
    let ir_profile = take_function_plan_profile();
    assert_eq!(ir_profile.plan_builds, 1);
    assert_eq!(ir_profile.reference_graph_builds, 1);
}

#[test]
fn issue_1316_malformed_group_rejects_in_both_drivers() {
    let malformed = vec![node_expr(
        DeepTag::Def,
        vec![
            symbol_expr("broken"),
            node_expr(
                DeepTag::Fn,
                vec![node_expr(DeepTag::Params, vec![symbol_expr("x")])],
            ),
        ],
    )];
    assert!(check_typed_program(&malformed).is_err());
    assert!(check_ir_program(&malformed).is_err());
}

#[test]
fn issue_1316_unknown_call_diagnostic_order_is_stable_across_drivers() {
    let declarations = chelis_surf::parser::parse_str(
        "module Profile.Errors\n\
         def first(x: i32) -> i32 = missing_first(x)\n\
         def second(x: i32) -> i32 = missing_second(x)\n",
    )
    .expect("fixture parses");
    let program = chelis_surf::desugar::desugar_program(&declarations);
    let messages = |result: Result<CheckedProgram, InferResult>| {
        result
            .expect_err("unknown calls must reject")
            .errors
            .into_iter()
            .filter(|error| matches!(error.kind, CheckErrorKind::UnboundVariable { .. }))
            .map(|error| error.message)
            .collect::<Vec<_>>()
    };
    let primary = messages(check_typed_program(&program));
    let ir = messages(check_ir_program(&program));
    assert_eq!(primary, ir);
    assert_eq!(primary.len(), 2);
    assert!(primary[0].contains("missing_first"));
    assert!(primary[1].contains("missing_second"));
}
