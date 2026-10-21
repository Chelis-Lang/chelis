use chelis_reef_bootstrap_core::{
    BootstrapGraphError, BootstrapIdentity, BootstrapNode, validate_and_order,
};

fn identity(name: &str, version: &str) -> BootstrapIdentity {
    BootstrapIdentity {
        name: name.into(),
        version: version.into(),
    }
}

fn node(name: &str, version: &str, deps: &[(&str, &str)]) -> BootstrapNode {
    BootstrapNode {
        identity: identity(name, version),
        dependencies: deps.iter().map(|(n, v)| identity(n, v)).collect(),
    }
}

fn ordered(nodes: Vec<BootstrapNode>, expected: &[usize]) {
    let original = nodes.clone();
    assert_eq!(validate_and_order(&nodes).unwrap(), expected);
    assert_eq!(validate_and_order(&nodes).unwrap(), expected);
    assert_eq!(nodes, original);
    let mut indices = expected.to_vec();
    indices.sort_unstable();
    assert_eq!(indices, (0..nodes.len()).collect::<Vec<_>>());
}

#[test]
fn stable_depth_first_orders() {
    ordered(vec![], &[]);
    ordered(
        vec![
            node("A", "1", &[("B", "1")]),
            node("B", "1", &[("C", "1")]),
            node("C", "1", &[]),
        ],
        &[2, 1, 0],
    );
    ordered(
        vec![
            node("A", "1", &[("B", "1"), ("C", "1")]),
            node("B", "1", &[("D", "1")]),
            node("C", "1", &[("D", "1")]),
            node("D", "1", &[]),
        ],
        &[3, 1, 2, 0],
    );
    ordered(
        vec![
            node("Z", "1", &[]),
            node("A", "1", &[]),
            node("M", "1", &[]),
        ],
        &[0, 1, 2],
    );
    ordered(
        vec![
            node("A", "1", &[("B", "1")]),
            node("C", "1", &[("D", "1")]),
            node("D", "1", &[]),
            node("B", "1", &[]),
        ],
        &[3, 0, 2, 1],
    );
    ordered(
        vec![
            node("A", "1", &[("C", "1"), ("B", "1")]),
            node("B", "1", &[]),
            node("C", "1", &[]),
        ],
        &[2, 1, 0],
    );
    ordered(
        vec![
            node("A", "1", &[("B", "1"), ("B", "1")]),
            node("B", "1", &[]),
        ],
        &[1, 0],
    );
}

#[test]
fn occurrences_and_last_target_remain_distinct() {
    ordered(
        vec![
            node("A", "1", &[]),
            node("C", "1", &[("A", "1")]),
            node("A", "1", &[]),
        ],
        &[0, 2, 1],
    );
    ordered(
        vec![
            node("A", "1", &[("B", "1")]),
            node("A", "1", &[("C", "1")]),
            node("B", "1", &[]),
            node("C", "1", &[]),
        ],
        &[2, 0, 3, 1],
    );
    assert_eq!(
        validate_and_order(&[node("A", "1", &[("absent", "1")]), node("A", "1", &[])]),
        Err(BootstrapGraphError::MissingDependency {
            dependent: identity("A", "1"),
            missing: identity("absent", "1")
        })
    );
}

#[test]
fn conflicts_sort_unique_versions_before_other_checks() {
    let nodes = [
        node("A", "2", &[("absent", "1")]),
        node("A", "1", &[]),
        node("A", "2", &[]),
    ];
    assert_eq!(
        validate_and_order(&nodes),
        Err(BootstrapGraphError::DuplicateVersion {
            package: "A".into(),
            versions: vec!["1".into(), "2".into()]
        })
    );
    let multiple = [
        node("A", "2", &[]),
        node("A", "1", &[]),
        node("B", "3", &[]),
        node("B", "2", &[]),
    ];
    match validate_and_order(&multiple).unwrap_err() {
        BootstrapGraphError::DuplicateVersion { package, versions } => {
            assert!(package == "A" || package == "B");
            assert_eq!(
                versions,
                if package == "A" {
                    vec!["1", "2"]
                } else {
                    vec!["2", "3"]
                }
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn exact_missing_edges_use_node_then_edge_order_before_cycles() {
    for (wanted, present) in [("2", "1"), ("1.0", "1.0.0"), (" 1", "1")] {
        assert_eq!(
            validate_and_order(&[node("A", "1", &[("B", wanted)]), node("B", present, &[])]),
            Err(BootstrapGraphError::MissingDependency {
                dependent: identity("A", "1"),
                missing: identity("B", wanted)
            })
        );
    }
    assert_eq!(
        validate_and_order(&[
            node("cycle", "1", &[("cycle", "1")]),
            node("A", "1", &[("second", "7"), ("first", "8")]),
            node("B", "1", &[("other", "9")])
        ]),
        Err(BootstrapGraphError::MissingDependency {
            dependent: identity("A", "1"),
            missing: identity("second", "7")
        })
    );
    assert_eq!(
        validate_and_order(&[node("a", "1", &[("A", "1")])]),
        Err(BootstrapGraphError::MissingDependency {
            dependent: identity("a", "1"),
            missing: identity("A", "1")
        })
    );
}

#[test]
fn cycle_paths_close_and_preserve_identity_suffixes() {
    let cases = [
        (
            vec![node("A", "1", &[("A", "1")])],
            vec![identity("A", "1"), identity("A", "1")],
        ),
        (
            vec![
                node("X", "x", &[("A", "a")]),
                node("A", "a", &[("B", "b")]),
                node("B", "b", &[("C", "c")]),
                node("C", "c", &[("A", "a")]),
            ],
            vec![
                identity("A", "a"),
                identity("B", "b"),
                identity("C", "c"),
                identity("A", "a"),
            ],
        ),
        (
            vec![node("A", "1", &[("A", "1")]), node("A", "1", &[("A", "1")])],
            vec![identity("A", "1"), identity("A", "1"), identity("A", "1")],
        ),
    ];
    for (nodes, cycle) in cases {
        assert_eq!(
            validate_and_order(&nodes),
            Err(BootstrapGraphError::Cycle { cycle })
        );
    }
}

#[test]
fn long_chain_uses_explicit_traversal_state() {
    let count = 30_000;
    let nodes: Vec<_> = (0..count)
        .map(|i| BootstrapNode {
            identity: identity(&i.to_string(), "1"),
            dependencies: if i + 1 < count {
                vec![identity(&(i + 1).to_string(), "1")]
            } else {
                vec![]
            },
        })
        .collect();
    assert_eq!(
        validate_and_order(&nodes).unwrap(),
        (0..count).rev().collect::<Vec<_>>()
    );
}
