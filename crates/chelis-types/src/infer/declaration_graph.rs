//! Top-level declaration dependency analysis.
//!
//! One responsibility: build the canonical reference graph between
//! top-level declarations, report initialization-order defects from it, and
//! derive the callee-first function inference plan from its strongly
//! connected components.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TopLevelDefinitionKind {
    Function,
    EagerValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum TopLevelReferenceKind {
    Read,
    Apply,
}

#[derive(Clone, Debug)]
pub(super) struct TopLevelDefinition {
    pub(super) item_index: usize,
    pub(super) name: String,
    pub(super) kind: TopLevelDefinitionKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct TopLevelReference {
    pub(super) target: usize,
    pub(super) kind: TopLevelReferenceKind,
}

/// One canonical syntactic reference graph for top-level inference.
///
/// Vertices are name-keyed top-level `def`s. `item_references` additionally
/// records the outgoing references of every flattened item so the body
/// scheduler can consume the same lexical walk as cycle detection and the
/// function SCC planner. Edges point from the declaration/item containing the
/// reference to the referenced definition. The edge kind distinguishes a bare
/// read from direct application for diagnostics; reachability and SCCs use
/// both, as [04-INF-7] requires.
///
/// Duplicate definitions are already rejected independently. For totality on
/// such input, the first definition owns the vertex/ordinal while the last
/// occurrence supplies its outgoing edges, matching the previous function
/// planner and cycle detector.
#[derive(Clone, Debug, Default)]
pub(super) struct TopLevelReferenceGraph {
    pub(super) definitions: Vec<TopLevelDefinition>,
    pub(super) vertex_by_name: UnordMap<String, usize>,
    pub(super) definition_vertex_by_item: Vec<Option<usize>>,
    pub(super) outgoing: Vec<Vec<TopLevelReference>>,
    pub(super) item_references: Vec<Vec<TopLevelReference>>,
    pub(super) complete: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LaterEagerValueDependency {
    pub(super) root: usize,
    pub(super) later: usize,
    pub(super) path: Vec<usize>,
    pub(super) root_has_direct_forward_reference: bool,
}

struct TopLevelInitializationAnalysis {
    adjacency: Vec<Vec<usize>>,
    eager_cycle_components: Vec<Vec<usize>>,
    cyclic_eager_vertices: Vec<bool>,
}

#[cfg(test)]
thread_local! {
    static LATER_DEPENDENCY_CANCEL_AFTER_EDGES: RefCell<Option<(usize, CancelToken)>> =
        const { RefCell::new(None) };
}

fn later_dependency_edge_inspected() {
    #[cfg(test)]
    LATER_DEPENDENCY_CANCEL_AFTER_EDGES.with(|hook| {
        let mut hook = hook.borrow_mut();
        let Some((remaining, token)) = hook.as_mut() else {
            return;
        };
        *remaining -= 1;
        if *remaining == 0 {
            token.cancel();
            hook.take();
        }
    });
}

#[cfg(test)]
struct LaterDependencyCancellationHook;

#[cfg(test)]
impl Drop for LaterDependencyCancellationHook {
    fn drop(&mut self) {
        LATER_DEPENDENCY_CANCEL_AFTER_EDGES.with(|hook| hook.borrow_mut().take());
    }
}

#[cfg(test)]
fn cancel_later_dependency_after_edges_for_test(
    inspections: usize,
    token: CancelToken,
) -> LaterDependencyCancellationHook {
    assert!(inspections > 0);
    LATER_DEPENDENCY_CANCEL_AFTER_EDGES.with(|hook| {
        assert!(hook.borrow_mut().replace((inspections, token)).is_none());
    });
    LaterDependencyCancellationHook
}

fn initialization_cancelled(cancel: Option<&CancelToken>, errors: &mut DiagnosticSink<'_>) -> bool {
    if !cancel.is_some_and(CancelToken::is_cancelled) {
        return false;
    }
    if !errors
        .iter()
        .any(|error| crate::cancel::is_cancellation(&error.message))
    {
        errors.push(crate::cancel::cancellation_check_error());
    }
    true
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TopLevelReferenceComponent {
    /// Flattened `def` item ordinals in source order.
    pub(super) members: Vec<usize>,
    pub(super) cyclic: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct TopLevelReferenceComponents {
    /// Components in dependency-first order. An edge `a -> b` means `a`
    /// references `b`, so `b`'s component precedes `a`'s.
    pub(super) components: Vec<TopLevelReferenceComponent>,
    /// Flattened item ordinal -> component index. `defsig`, `deftype`, and
    /// other non-definition items are `None` and remain scheduler singletons.
    pub(super) component_by_item: Vec<Option<usize>>,
    pub(super) complete: bool,
}

impl TopLevelReferenceGraph {
    pub(super) fn build(items: &[(Option<String>, &deep::Expr)]) -> Self {
        profile_reference_graph_build();
        let cancel = crate::cancel::current_cancel_token();
        let cancelled = || cancel.as_ref().is_some_and(CancelToken::is_cancelled);
        let mut graph = Self {
            definition_vertex_by_item: vec![None; items.len()],
            item_references: vec![Vec::new(); items.len()],
            complete: true,
            ..Self::default()
        };

        let mut declared_signature_names = UnordSet::new();
        for (_, expr) in items {
            if cancelled() {
                graph.complete = false;
                return graph;
            }
            let Some((tag, _, kids)) = stamped_parts(expr) else {
                continue;
            };
            let Some(name) = kids.first().and_then(symbol_name) else {
                continue;
            };
            if tag == DeepTag::Defsig {
                declared_signature_names.insert(name.to_string());
            }
        }

        for (item_index, (_, expr)) in items.iter().enumerate() {
            if cancelled() {
                graph.complete = false;
                return graph;
            }
            let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
                continue;
            };
            let (Some(name), Some(body)) = (kids.first().and_then(symbol_name), kids.get(1)) else {
                continue;
            };
            if let Some(vertex) = graph.vertex_by_name.get(name).copied() {
                graph.definition_vertex_by_item[item_index] = Some(vertex);
                continue;
            }
            let vertex = graph.definitions.len();
            graph.vertex_by_name.insert(name.to_string(), vertex);
            graph.definition_vertex_by_item[item_index] = Some(vertex);
            graph.definitions.push(TopLevelDefinition {
                item_index,
                name: name.to_string(),
                kind: if tagged_children(body, DeepTag::Fn).is_some() {
                    TopLevelDefinitionKind::Function
                } else {
                    TopLevelDefinitionKind::EagerValue
                },
            });
            graph.outgoing.push(Vec::new());
        }

        for (item_index, (_, expr)) in items.iter().enumerate() {
            if cancelled() {
                graph.complete = false;
                return graph;
            }
            let mut references = BTreeSet::new();
            let mut bound = Vec::new();
            collect_top_level_references(expr, &graph.vertex_by_name, &mut bound, &mut references);

            // [04-INF-4]: the literal self-reference of an explicitly typed
            // external input is its declaration spelling, not an eager edge.
            if let Some((DeepTag::Def, _, kids)) = stamped_parts(expr)
                && let (Some(name), Some(body)) = (kids.first().and_then(symbol_name), kids.get(1))
                && (body_is_type_stamped_literal_self_ref(body, name)
                    || (declared_signature_names.contains(name)
                        && body_is_literal_self_ref_shape(body, name)))
                && let Some(vertex) = graph.vertex_by_name.get(name).copied()
            {
                references.retain(|reference| reference.target != vertex);
            }

            let item_references = references.into_iter().collect::<Vec<_>>();
            graph.item_references[item_index] = item_references.clone();
            if let Some((DeepTag::Def, _, kids)) = stamped_parts(expr)
                && let Some(name) = kids.first().and_then(symbol_name)
                && let Some(vertex) = graph.vertex_by_name.get(name).copied()
            {
                // Last duplicate body wins, preserving the old planners'
                // deterministic behavior on an already-invalid program.
                graph.outgoing[vertex] = item_references;
            }
        }
        graph
    }

    pub(super) fn definition(&self, vertex: usize) -> &TopLevelDefinition {
        &self.definitions[vertex]
    }

    /// SCC projection of the full [04-INF-7] reference graph. This is the
    /// shared seed for mixed body-inference components: scheduler-only mirror
    /// and hole precedence edges may merge these components, but no consumer
    /// may split one and reintroduce an inference-order `UnboundVariable` for
    /// an eager cycle.
    pub(super) fn inference_components(&self) -> TopLevelReferenceComponents {
        if !self.complete {
            return TopLevelReferenceComponents {
                component_by_item: vec![None; self.item_references.len()],
                complete: false,
                ..TopLevelReferenceComponents::default()
            };
        }
        let adjacency = self.adjacency();
        let cancel = crate::cancel::current_cancel_token();
        let Some(vertex_components) = unprofiled_scc_vertex_components(&adjacency, cancel.as_ref())
        else {
            return TopLevelReferenceComponents {
                component_by_item: vec![None; self.item_references.len()],
                complete: false,
                ..TopLevelReferenceComponents::default()
            };
        };
        let mut components = Vec::with_capacity(vertex_components.len());
        let mut component_by_item = vec![None; self.item_references.len()];
        for vertices in vertex_components {
            let cyclic = vertices.len() > 1
                || vertices
                    .first()
                    .is_some_and(|vertex| adjacency[*vertex].binary_search(vertex).is_ok());
            let mut members = vertices
                .iter()
                .map(|vertex| self.definitions[*vertex].item_index)
                .collect::<Vec<_>>();
            members.sort_unstable();
            let component_index = components.len();
            for member in &members {
                component_by_item[*member] = Some(component_index);
            }
            components.push(TopLevelReferenceComponent { members, cyclic });
        }
        TopLevelReferenceComponents {
            components,
            component_by_item,
            complete: true,
        }
    }

    /// Later eager values whose inferred schemes must be available while one
    /// cyclic full-reference component is co-inferred.
    ///
    /// [04-INF-8] gives `CycleDetected` precedence when an eager root occurs
    /// in its own reachable closure. If a member of that exact SCC also reads
    /// a value declared after one of its eager roots, ordinary [04-INF-4]
    /// visibility would emit `UnboundVariable` before the cycle reporter can
    /// publish the owning verdict. These targets are not cycle members: the
    /// scheduler infers them first and the component scope grants their
    /// already-established bindings temporary visibility. The source-order
    /// rule remains unchanged everywhere else.
    pub(super) fn cycle_precedence_targets(&self, component_items: &[usize]) -> Vec<usize> {
        if !self.complete {
            return Vec::new();
        }
        let member_vertices = component_items
            .iter()
            .filter_map(|item| self.definition_vertex_by_item.get(*item).copied().flatten())
            .collect::<UnordSet<_>>();
        let Some(earliest_eager_root) = member_vertices
            .to_sorted()
            .into_iter()
            .filter_map(|vertex| {
                let definition = &self.definitions[*vertex];
                (definition.kind == TopLevelDefinitionKind::EagerValue)
                    .then_some(definition.item_index)
            })
            .min()
        else {
            return Vec::new();
        };

        let mut targets = BTreeSet::new();
        for &item in component_items {
            let Some(references) = self.item_references.get(item) else {
                continue;
            };
            for reference in references {
                let target = &self.definitions[reference.target];
                if target.kind == TopLevelDefinitionKind::EagerValue
                    && target.item_index > earliest_eager_root
                    && !member_vertices.contains(&reference.target)
                {
                    targets.insert(reference.target);
                }
            }
        }
        targets.into_iter().collect()
    }

    fn adjacency(&self) -> Vec<Vec<usize>> {
        self.outgoing
            .iter()
            .map(|references| {
                let mut targets = references
                    .iter()
                    .map(|reference| reference.target)
                    .collect::<Vec<_>>();
                targets.sort_by_key(|target| {
                    let definition = &self.definitions[*target];
                    (definition.item_index, definition.name.clone())
                });
                targets.dedup();
                targets
            })
            .collect()
    }

    fn initialization_analysis(
        &self,
        cancel: Option<&CancelToken>,
    ) -> Option<TopLevelInitializationAnalysis> {
        let cancelled = || cancel.is_some_and(CancelToken::is_cancelled);
        if !self.complete || cancelled() {
            return None;
        }
        let adjacency = self.adjacency();
        if cancelled() {
            return None;
        }
        let mut components = unprofiled_scc_vertex_components(&adjacency, cancel)?;
        components.retain(|component| {
            let cyclic = component.len() > 1
                || component
                    .first()
                    .is_some_and(|vertex| adjacency[*vertex].binary_search(vertex).is_ok());
            cyclic
                && component.iter().any(|vertex| {
                    self.definitions[*vertex].kind == TopLevelDefinitionKind::EagerValue
                })
        });
        let mut cyclic_eager_vertices = vec![false; self.definitions.len()];
        for component in &mut components {
            for &vertex in component.iter() {
                cyclic_eager_vertices[vertex] = true;
            }
            component.sort_by_key(|vertex| {
                let definition = &self.definitions[*vertex];
                (definition.item_index, definition.name.clone())
            });
        }
        components.sort_by_key(|component| {
            component
                .iter()
                .map(|vertex| self.definitions[*vertex].item_index)
                .min()
                .unwrap_or(usize::MAX)
        });
        if cancelled() {
            return None;
        }
        Some(TopLevelInitializationAnalysis {
            adjacency,
            eager_cycle_components: components,
            cyclic_eager_vertices,
        })
    }

    #[cfg(test)]
    fn cyclic_eager_components(&self) -> Vec<Vec<usize>> {
        let cancel = crate::cancel::current_cancel_token();
        self.initialization_analysis(cancel.as_ref())
            .map(|analysis| analysis.eager_cycle_components)
            .unwrap_or_default()
    }

    fn cycle_path(
        &self,
        adjacency: &[Vec<usize>],
        component: &[usize],
        cancel: Option<&CancelToken>,
    ) -> Option<Vec<usize>> {
        let cancelled = || cancel.is_some_and(CancelToken::is_cancelled);
        if cancelled() {
            return None;
        }
        let members = component.iter().copied().collect::<UnordSet<_>>();
        let start = component
            .iter()
            .copied()
            .filter(|vertex| self.definitions[*vertex].kind == TopLevelDefinitionKind::EagerValue)
            .min_by_key(|vertex| {
                let definition = &self.definitions[*vertex];
                (definition.item_index, definition.name.clone())
            })
            .expect("an eager cycle component contains an eager value");
        if adjacency[start].binary_search(&start).is_ok() {
            return Some(vec![start, start]);
        }

        // Find the deterministic shortest return path from each source-order
        // neighbor. Strong connectivity guarantees that one reaches `start`.
        for &first in adjacency[start]
            .iter()
            .filter(|target| members.contains(target))
        {
            let mut queue = VecDeque::from([first]);
            let mut predecessor = UnordMap::new();
            let mut seen = UnordSet::new();
            seen.insert(first);
            while let Some(vertex) = queue.pop_front() {
                if cancelled() {
                    return None;
                }
                if vertex == start {
                    let mut reverse = vec![start];
                    let mut current = start;
                    while current != first {
                        if cancelled() {
                            return None;
                        }
                        current = predecessor[&current];
                        reverse.push(current);
                    }
                    reverse.reverse();
                    let mut path = vec![start];
                    path.extend(reverse);
                    return Some(path);
                }
                for &next in adjacency[vertex]
                    .iter()
                    .filter(|target| members.contains(target))
                {
                    if seen.insert(next) {
                        predecessor.insert(next, vertex);
                        queue.push_back(next);
                    }
                }
            }
        }
        unreachable!("every member of an SCC reaches its eager start")
    }

    /// [04-INF-8]: deterministic acyclic transitive dependencies from an eager
    /// root to a later eager value. Direct forward reads remain owned by
    /// `infer_var` at their exact source span and are omitted here, so the graph
    /// adds exactly one diagnostic per indirect `(root, later)` pair rather than
    /// duplicating [04-INF-4]'s established diagnostic.
    #[cfg(test)]
    pub(super) fn acyclic_later_eager_value_dependencies(
        &self,
    ) -> Option<Vec<LaterEagerValueDependency>> {
        let cancel = crate::cancel::current_cancel_token();
        let analysis = self.initialization_analysis(cancel.as_ref())?;
        self.later_eager_value_dependencies(&analysis, cancel.as_ref())
    }

    fn later_eager_value_dependencies(
        &self,
        analysis: &TopLevelInitializationAnalysis,
        cancel: Option<&CancelToken>,
    ) -> Option<Vec<LaterEagerValueDependency>> {
        let cancelled = || cancel.is_some_and(CancelToken::is_cancelled);
        let vertex_count = self.definitions.len();
        let mut visit_epoch = vec![0usize; vertex_count];
        let mut predecessor = vec![usize::MAX; vertex_count];
        let mut queue = VecDeque::new();
        let mut reached_later = Vec::new();
        let mut findings = Vec::new();
        let mut epoch = 0usize;

        for root in self
            .definitions
            .iter()
            .enumerate()
            .filter(|(_, definition)| definition.kind == TopLevelDefinitionKind::EagerValue)
            .map(|(vertex, _)| vertex)
        {
            if cancelled() {
                return None;
            }
            // CycleDetected takes precedence only when this eager root is
            // itself in the cycle. A different reachable eager cycle does
            // not erase an otherwise independent root-to-later violation.
            if analysis.cyclic_eager_vertices[root] {
                continue;
            }

            epoch += 1;
            visit_epoch[root] = epoch;
            queue.clear();
            queue.push_back(root);
            reached_later.clear();
            while let Some(vertex) = queue.pop_front() {
                if cancelled() {
                    return None;
                }
                for &next in &analysis.adjacency[vertex] {
                    later_dependency_edge_inspected();
                    if cancelled() {
                        return None;
                    }
                    if vertex == root
                        && self.definitions[next].kind == TopLevelDefinitionKind::EagerValue
                        && self.definitions[next].item_index > self.definitions[root].item_index
                    {
                        // A direct source-forward value read is either owned
                        // by [04-INF-4], or resolves a same-name prior-library
                        // binding. In neither case is that edge a dependency
                        // on the current unit's later definition. Exclude it
                        // from this search while retaining distinct indirect
                        // paths from the same root to the same later value.
                        continue;
                    }
                    if visit_epoch[next] == epoch {
                        continue;
                    }
                    visit_epoch[next] = epoch;
                    predecessor[next] = vertex;
                    queue.push_back(next);
                    if self.definitions[next].kind == TopLevelDefinitionKind::EagerValue
                        && self.definitions[next].item_index > self.definitions[root].item_index
                    {
                        reached_later.push(next);
                    }
                }
            }

            reached_later.sort_by_key(|later| {
                let definition = &self.definitions[*later];
                (definition.item_index, definition.name.clone())
            });
            reached_later.dedup();
            for &later in &reached_later {
                if cancelled() {
                    return None;
                }
                let mut path = vec![later];
                let mut current = later;
                while current != root {
                    if cancelled() {
                        return None;
                    }
                    current = predecessor[current];
                    debug_assert_ne!(
                        current,
                        usize::MAX,
                        "a reached vertex has a predecessor before the root"
                    );
                    path.push(current);
                }
                path.reverse();
                findings.push(LaterEagerValueDependency {
                    root,
                    later,
                    path,
                    root_has_direct_forward_reference: analysis.adjacency[root]
                        .binary_search(&later)
                        .is_ok(),
                });
            }
        }
        findings.sort_by_key(|finding| {
            (
                self.definitions[finding.root].item_index,
                self.definitions[finding.later].item_index,
                self.definitions[finding.root].name.clone(),
                self.definitions[finding.later].name.clone(),
            )
        });
        if cancelled() {
            return None;
        }
        Some(findings)
    }

    pub(super) fn report_initialization_errors(&self, errors: &mut DiagnosticSink<'_>) {
        let cancel = crate::cancel::current_cancel_token();
        let Some(analysis) = self.initialization_analysis(cancel.as_ref()) else {
            initialization_cancelled(cancel.as_ref(), errors);
            return;
        };
        let Some(findings) = self.later_eager_value_dependencies(&analysis, cancel.as_ref()) else {
            initialization_cancelled(cancel.as_ref(), errors);
            return;
        };

        // Stage the complete report before mutating the sink. If cancellation
        // interrupts SCC analysis, cycle-path recovery, or reachability, the
        // policy report stays unpublished and this boundary records only the
        // cancellation diagnostic, even if the sink already has source errors.
        let mut cycle_paths = Vec::with_capacity(analysis.eager_cycle_components.len());
        for component in &analysis.eager_cycle_components {
            let Some(path) = self.cycle_path(&analysis.adjacency, component, cancel.as_ref())
            else {
                initialization_cancelled(cancel.as_ref(), errors);
                return;
            };
            cycle_paths.push(path);
        }
        if initialization_cancelled(cancel.as_ref(), errors) {
            return;
        }

        let existing_unbound_names = errors
            .iter()
            .filter_map(|error| match &error.kind {
                CheckErrorKind::UnboundVariable { identifier } => Some(identifier.clone()),
                _ => None,
            })
            .collect::<UnordSet<_>>();
        let mut staged = Vec::with_capacity(cycle_paths.len() + findings.len());
        for path in cycle_paths {
            if initialization_cancelled(cancel.as_ref(), errors) {
                return;
            }
            let names = path
                .iter()
                .map(|vertex| self.definitions[*vertex].name.as_str())
                .collect::<Vec<_>>();
            staged.push(CheckError::new(
                CheckErrorKind::CycleDetected,
                format!("binding cycle: {}", names.join(" -> ")),
                vec![
                    "Break the cycle by removing one of the self-referential definitions or \
                     replacing it with a concrete value."
                        .to_string(),
                ],
            ));
        }
        for finding in findings {
            if initialization_cancelled(cancel.as_ref(), errors) {
                return;
            }
            let root = &self.definitions[finding.root];
            let later = &self.definitions[finding.later];
            if finding.root_has_direct_forward_reference
                && existing_unbound_names.contains(&later.name)
            {
                continue;
            }
            let path = finding
                .path
                .iter()
                .map(|vertex| self.definitions[*vertex].name.as_str())
                .collect::<Vec<_>>()
                .join(" -> ");
            staged.push(CheckError::new(
                CheckErrorKind::UnboundVariable {
                    identifier: later.name.clone(),
                },
                format!(
                    "top-level eager value `{}` reaches later value `{}` during initialization \
                     through {path} ([04-INF-8])",
                    root.name, later.name
                ),
                vec![format!(
                    "Move `{}` before `{}`, or pass its value explicitly",
                    later.name, root.name
                )],
            ));
        }
        if initialization_cancelled(cancel.as_ref(), errors) {
            return;
        }
        for error in staged {
            errors.push(error);
        }
    }
}

fn collect_top_level_references(
    expr: &deep::Expr,
    vertex_by_name: &UnordMap<String, usize>,
    bound: &mut Vec<UnordSet<String>>,
    references: &mut BTreeSet<TopLevelReference>,
) {
    stack_guard!("collect_top_level_references", expr);
    match expr {
        deep::Expr::Atom(_, _) => {}
        deep::Expr::Map(map, _) => {
            map.visit_syntax(&mut |_, value| {
                collect_top_level_references(value, vertex_by_name, bound, references);
            });
        }
        // Metadata describes the expression; it is not executed as part of a
        // top-level initializer. The stamped expression itself still is.
        deep::Expr::MetaExpr(meta, _) => {
            collect_top_level_references(&meta.expr, vertex_by_name, bound, references)
        }
        deep::Expr::Node(node, _) => match node.tag() {
            DeepTag::App => {
                let kids = node.children_slice();
                if let Some(callee) = kids.first().and_then(var_name_expr)
                    && vertex_by_name.contains_key(callee)
                    && !is_bound_name(callee, bound)
                {
                    references.insert(TopLevelReference {
                        target: vertex_by_name[callee],
                        kind: TopLevelReferenceKind::Apply,
                    });
                } else if let Some(callee) = kids.first() {
                    collect_top_level_references(callee, vertex_by_name, bound, references);
                }
                for argument in kids.iter().skip(1) {
                    collect_top_level_references(argument, vertex_by_name, bound, references);
                }
            }
            DeepTag::Var => {
                if let Some(name) = node.children_slice().first().and_then(symbol_name)
                    && vertex_by_name.contains_key(name)
                    && !is_bound_name(name, bound)
                {
                    references.insert(TopLevelReference {
                        target: vertex_by_name[name],
                        kind: TopLevelReferenceKind::Read,
                    });
                }
            }
            DeepTag::Fn => {
                let kids = node.children_slice();
                if kids.len() >= 2 {
                    bound.push(
                        param_source_infos(&kids[0])
                            .into_iter()
                            .map(|(name, _)| name)
                            .collect(),
                    );
                    // [04-INF-7]: nested lambda bodies are part of the eager
                    // syntactic reference set even if the closure is stored.
                    collect_top_level_references(&kids[1], vertex_by_name, bound, references);
                    bound.pop();
                }
            }
            DeepTag::Let => {
                let kids = node.children_slice();
                if kids.len() < 2 {
                    return;
                }
                bound.push(UnordSet::new());
                if let Some(bind_kids) = kids
                    .first()
                    .and_then(|bindings| tagged_children(bindings, DeepTag::Bind))
                {
                    let mut index = 0;
                    while index + 1 < bind_kids.len() {
                        collect_top_level_references(
                            &bind_kids[index + 1],
                            vertex_by_name,
                            bound,
                            references,
                        );
                        if let Some(name) = symbol_name(&bind_kids[index]) {
                            bound
                                .last_mut()
                                .expect("let scope exists")
                                .insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                collect_top_level_references(&kids[1], vertex_by_name, bound, references);
                bound.pop();
            }
            DeepTag::Match => {
                let kids = node.children_slice();
                if let Some(scrutinee) = kids.first() {
                    collect_top_level_references(scrutinee, vertex_by_name, bound, references);
                }
                for arm in kids.iter().skip(1) {
                    let Some((DeepTag::Arm, _, arm_kids)) = stamped_parts(arm) else {
                        collect_top_level_references(arm, vertex_by_name, bound, references);
                        continue;
                    };
                    let Some(pattern) = arm_kids.first() else {
                        continue;
                    };
                    bound.push(
                        chelis_deep::pattern_binder_names(pattern)
                            .into_iter()
                            .collect(),
                    );
                    for scoped in arm_kids.iter().skip(1) {
                        collect_top_level_references(scoped, vertex_by_name, bound, references);
                    }
                    bound.pop();
                }
            }
            _ => {
                for child in node.children_slice() {
                    collect_top_level_references(child, vertex_by_name, bound, references);
                }
            }
        },
        deep::Expr::BareList(elements, _) => {
            for child in elements {
                collect_top_level_references(child, vertex_by_name, bound, references);
            }
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                collect_top_level_references(child, vertex_by_name, bound, references);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct FunctionInferenceMember {
    pub(super) item_index: usize,
    pub(super) name: String,
}

#[derive(Clone, Debug)]
pub(super) struct FunctionInferenceComponent {
    pub(super) members: Vec<FunctionInferenceMember>,
    pub(super) recursive: bool,
}

#[derive(Clone, Debug)]
pub(super) struct FunctionInferencePlan {
    pub(super) components: Vec<FunctionInferenceComponent>,
    pub(super) complete: bool,
}

impl Default for FunctionInferencePlan {
    fn default() -> Self {
        Self {
            components: Vec::new(),
            complete: true,
        }
    }
}

impl FunctionInferencePlan {
    /// Build the one canonical function dependency plan for an inference run.
    /// SCCs are constructed in O(vertices + edges), returned callee-first,
    /// and retain source order within each component.
    #[cfg(test)]
    pub(super) fn build(items: &[(Option<String>, &deep::Expr)]) -> Self {
        let references = TopLevelReferenceGraph::build(items);
        Self::build_from_reference_graph(&references)
    }

    /// Project the function-only SCC plan from the canonical top-level
    /// reference graph. The main inference driver builds that graph once and
    /// shares it with cycle diagnostics and the mixed-component scheduler.
    pub(super) fn build_from_reference_graph(references: &TopLevelReferenceGraph) -> Self {
        profile_plan_build();
        if !references.complete {
            return Self::incomplete();
        }
        let cancel = crate::cancel::current_cancel_token();
        let cancelled = || cancel.as_ref().is_some_and(CancelToken::is_cancelled);
        let mut compact_by_reference_vertex = vec![None; references.definitions.len()];
        let mut function_vertices = Vec::new();
        for (reference_vertex, definition) in references.definitions.iter().enumerate() {
            if cancelled() {
                return Self::incomplete();
            }
            if definition.kind != TopLevelDefinitionKind::Function {
                continue;
            }
            compact_by_reference_vertex[reference_vertex] = Some(function_vertices.len());
            function_vertices.push(reference_vertex);
        }

        let mut graph = vec![Vec::<usize>::new(); function_vertices.len()];
        for (vertex, &reference_vertex) in function_vertices.iter().enumerate() {
            if cancelled() {
                return Self::incomplete();
            }
            let mut callees = references.outgoing[reference_vertex]
                .iter()
                .filter_map(|reference| compact_by_reference_vertex[reference.target])
                .collect::<Vec<_>>();
            callees.sort_unstable();
            callees.dedup();
            graph[vertex] = callees;
        }
        profile_graph(graph.len(), graph.iter().map(Vec::len).sum());

        let Some(vertex_components) = ordered_scc_vertex_components(&graph, cancel.as_ref()) else {
            return Self::incomplete();
        };
        let mut component_by_vertex = vec![0; graph.len()];
        for (component_index, vertices) in vertex_components.iter().enumerate() {
            for &vertex in vertices {
                component_by_vertex[vertex] = component_index;
            }
        }
        let mut components = vertex_components
            .iter()
            .map(|_| FunctionInferenceComponent {
                members: Vec::new(),
                recursive: false,
            })
            .collect::<Vec<_>>();
        for (vertex, reference_vertex) in function_vertices.into_iter().enumerate() {
            let definition = &references.definitions[reference_vertex];
            components[component_by_vertex[vertex]]
                .members
                .push(FunctionInferenceMember {
                    item_index: definition.item_index,
                    name: definition.name.clone(),
                });
        }
        for (component, vertices) in components.iter_mut().zip(&vertex_components) {
            component.recursive = component.members.len() > 1
                || vertices
                    .iter()
                    .any(|&vertex| graph[vertex].binary_search(&vertex).is_ok());
        }
        Self {
            components,
            complete: true,
        }
    }

    fn incomplete() -> Self {
        Self {
            components: Vec::new(),
            complete: false,
        }
    }

    pub(super) fn ordered_members(&self) -> impl Iterator<Item = &FunctionInferenceMember> {
        self.components
            .iter()
            .flat_map(|component| component.members.iter())
    }

    pub(super) fn recursive_member_names(&self) -> UnordSet<String> {
        self.components
            .iter()
            .filter(|component| component.recursive)
            .flat_map(|component| component.members.iter())
            .map(|member| member.name.clone())
            .collect()
    }
}

#[derive(Clone, Copy)]
struct TarjanFrame {
    vertex: usize,
    next_edge: usize,
}

/// Iterative Tarjan followed by the historical deterministic component
/// postorder. The iterative stack avoids replacing the cubic defect with a
/// native-stack limit on long declaration chains.
fn ordered_scc_vertex_components(
    graph: &[Vec<usize>],
    cancel: Option<&CancelToken>,
) -> Option<Vec<Vec<usize>>> {
    scc_vertex_components(graph, cancel, true)
}

fn unprofiled_scc_vertex_components(
    graph: &[Vec<usize>],
    cancel: Option<&CancelToken>,
) -> Option<Vec<Vec<usize>>> {
    scc_vertex_components(graph, cancel, false)
}

fn scc_vertex_components(
    graph: &[Vec<usize>],
    cancel: Option<&CancelToken>,
    profile: bool,
) -> Option<Vec<Vec<usize>>> {
    let cancelled = || cancel.is_some_and(CancelToken::is_cancelled);
    let mut next_index = 0usize;
    let mut indices = vec![None; graph.len()];
    let mut lowlinks = vec![0usize; graph.len()];
    let mut on_stack = vec![false; graph.len()];
    let mut tarjan_stack = Vec::new();
    let mut frames = Vec::<TarjanFrame>::new();
    let mut raw_component_by_vertex = vec![usize::MAX; graph.len()];
    let mut raw_component_count = 0usize;

    for start in 0..graph.len() {
        if indices[start].is_some() {
            continue;
        }
        if cancelled() {
            return None;
        }
        indices[start] = Some(next_index);
        lowlinks[start] = next_index;
        next_index += 1;
        tarjan_stack.push(start);
        on_stack[start] = true;
        if profile {
            profile_scc_vertex_entry();
        }
        frames.push(TarjanFrame {
            vertex: start,
            next_edge: 0,
        });

        while let Some(frame) = frames.last().copied() {
            if frame.next_edge < graph[frame.vertex].len() {
                if cancelled() {
                    return None;
                }
                let callee = graph[frame.vertex][frame.next_edge];
                frames.last_mut().expect("Tarjan frame exists").next_edge += 1;
                if profile {
                    profile_scc_edge_inspection();
                }
                if indices[callee].is_none() {
                    indices[callee] = Some(next_index);
                    lowlinks[callee] = next_index;
                    next_index += 1;
                    tarjan_stack.push(callee);
                    on_stack[callee] = true;
                    if profile {
                        profile_scc_vertex_entry();
                    }
                    frames.push(TarjanFrame {
                        vertex: callee,
                        next_edge: 0,
                    });
                } else if on_stack[callee] {
                    lowlinks[frame.vertex] =
                        lowlinks[frame.vertex].min(indices[callee].expect("visited vertex"));
                }
                continue;
            }

            let finished = frames.pop().expect("Tarjan frame exists").vertex;
            if lowlinks[finished] == indices[finished].expect("visited vertex") {
                loop {
                    let member = tarjan_stack.pop().expect("SCC root remains on stack");
                    on_stack[member] = false;
                    raw_component_by_vertex[member] = raw_component_count;
                    if member == finished {
                        break;
                    }
                }
                raw_component_count += 1;
            }
            if let Some(parent) = frames.last() {
                lowlinks[parent.vertex] = lowlinks[parent.vertex].min(lowlinks[finished]);
            }
        }
    }

    // Tarjan's discovery order is an implementation detail. Re-form the
    // component list in first-source-occurrence order before applying the
    // old callee-first postorder, preserving diagnostics and serialization.
    let mut compact_by_raw = UnordMap::new();
    let mut unordered = Vec::<Vec<usize>>::new();
    for (vertex, &raw) in raw_component_by_vertex.iter().enumerate() {
        let next = compact_by_raw.len();
        let component = *compact_by_raw.entry(raw).or_insert_with(|| {
            unordered.push(Vec::new());
            next
        });
        unordered[component].push(vertex);
    }
    let mut component_by_vertex = vec![0usize; graph.len()];
    for (component, vertices) in unordered.iter().enumerate() {
        for &vertex in vertices {
            component_by_vertex[vertex] = component;
        }
    }
    let mut component_graph = vec![UnordSet::<usize>::new(); unordered.len()];
    for (caller, callees) in graph.iter().enumerate() {
        if cancelled() {
            return None;
        }
        for &callee in callees {
            let caller_component = component_by_vertex[caller];
            let callee_component = component_by_vertex[callee];
            if caller_component != callee_component {
                component_graph[caller_component].insert(callee_component);
            }
        }
    }
    let dependencies = component_graph
        .into_iter()
        .map(|component| component.into_sorted())
        .collect::<Vec<_>>();

    let mut order = Vec::with_capacity(unordered.len());
    let mut visiting = vec![false; unordered.len()];
    let mut visited = vec![false; unordered.len()];
    for root in 0..unordered.len() {
        if visited[root] {
            continue;
        }
        visiting[root] = true;
        let mut component_frames = vec![(root, 0usize)];
        while let Some((component, next_dependency)) = component_frames.last().copied() {
            if cancelled() {
                return None;
            }
            if next_dependency < dependencies[component].len() {
                let dependency = dependencies[component][next_dependency];
                component_frames
                    .last_mut()
                    .expect("component frame exists")
                    .1 += 1;
                if !visited[dependency] && !visiting[dependency] {
                    visiting[dependency] = true;
                    component_frames.push((dependency, 0));
                }
                continue;
            }
            component_frames.pop();
            visiting[component] = false;
            if !visited[component] {
                visited[component] = true;
                order.push(component);
            }
        }
    }
    let mut slots = unordered.into_iter().map(Some).collect::<Vec<_>>();
    Some(
        order
            .into_iter()
            .filter_map(|component| slots[component].take())
            .collect(),
    )
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FunctionPlanProfile {
    pub(super) plan_builds: usize,
    pub(super) reference_graph_builds: usize,
    pub(super) graph_vertices: usize,
    pub(super) graph_edges: usize,
    pub(super) scc_vertex_entries: usize,
    pub(super) scc_edge_inspections: usize,
}

#[cfg(test)]
thread_local! {
    static FUNCTION_PLAN_PROFILE: RefCell<FunctionPlanProfile> = RefCell::default();
    static FUNCTION_PLAN_CANCEL_AFTER_EDGES: RefCell<Option<(usize, CancelToken)>> =
        const { RefCell::new(None) };
}

fn profile_plan_build() {
    #[cfg(test)]
    FUNCTION_PLAN_PROFILE.with(|profile| profile.borrow_mut().plan_builds += 1);
}

fn profile_reference_graph_build() {
    #[cfg(test)]
    FUNCTION_PLAN_PROFILE.with(|profile| profile.borrow_mut().reference_graph_builds += 1);
}

fn profile_graph(vertices: usize, edges: usize) {
    #[cfg(test)]
    FUNCTION_PLAN_PROFILE.with(|profile| {
        let mut profile = profile.borrow_mut();
        profile.graph_vertices += vertices;
        profile.graph_edges += edges;
    });
    #[cfg(not(test))]
    let _ = (vertices, edges);
}

fn profile_scc_vertex_entry() {
    #[cfg(test)]
    FUNCTION_PLAN_PROFILE.with(|profile| profile.borrow_mut().scc_vertex_entries += 1);
}

fn profile_scc_edge_inspection() {
    #[cfg(test)]
    {
        FUNCTION_PLAN_PROFILE.with(|profile| profile.borrow_mut().scc_edge_inspections += 1);
        FUNCTION_PLAN_CANCEL_AFTER_EDGES.with(|hook| {
            let mut hook = hook.borrow_mut();
            let Some((remaining, token)) = hook.as_mut() else {
                return;
            };
            *remaining -= 1;
            if *remaining == 0 {
                token.cancel();
                hook.take();
            }
        });
    }
}

#[cfg(test)]
pub(super) fn reset_function_plan_profile() {
    FUNCTION_PLAN_PROFILE.with(|profile| *profile.borrow_mut() = FunctionPlanProfile::default());
}

#[cfg(test)]
pub(super) fn take_function_plan_profile() -> FunctionPlanProfile {
    FUNCTION_PLAN_PROFILE.with(|profile| std::mem::take(&mut *profile.borrow_mut()))
}

#[cfg(test)]
pub(super) struct FunctionPlanCancellationHook;

#[cfg(test)]
impl Drop for FunctionPlanCancellationHook {
    fn drop(&mut self) {
        FUNCTION_PLAN_CANCEL_AFTER_EDGES.with(|hook| hook.borrow_mut().take());
    }
}

#[cfg(test)]
pub(super) fn cancel_function_plan_after_edges_for_test(
    inspections: usize,
    token: CancelToken,
) -> FunctionPlanCancellationHook {
    assert!(inspections > 0);
    FUNCTION_PLAN_CANCEL_AFTER_EDGES.with(|hook| {
        assert!(hook.borrow_mut().replace((inspections, token)).is_none());
    });
    FunctionPlanCancellationHook
}

#[cfg(test)]
pub(super) fn linear_scc_component_vertices_for_test(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    ordered_scc_vertex_components(graph, None).expect("uncancelled SCC construction completes")
}

#[cfg(test)]
mod top_level_reference_graph_tests {
    use super::*;

    fn surf_program(source: &str) -> Vec<deep::Expr> {
        let declarations = chelis_surf::parser::parse_str(source)
            .unwrap_or_else(|error| panic!("graph fixture must parse: {error:?}\n{source}"));
        chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar")
    }

    fn graph(source: &str) -> TopLevelReferenceGraph {
        let exprs = surf_program(source);
        let items = top_level_decl_items_with_modules(&exprs);
        TopLevelReferenceGraph::build(&items)
    }

    /// [04-INF-8] regression: the compiled-wrong-answer shape is one
    /// deterministic `(root, later)` finding, with the transitive path that
    /// explains why individually legal references are jointly illegal.
    #[test]
    fn indirect_later_value_dependency_is_reported_once_per_pair() {
        let graph = graph(
            "module IndirectLater\n\n\
             u = ping(1)\n\n\
             v1 = 5\n\n\
             def ping(n: i32) -> i32 = add(n, v1)\n",
        );
        let findings = graph
            .acyclic_later_eager_value_dependencies()
            .expect("uncancelled analysis completes");
        assert_eq!(findings.len(), 1, "{findings:#?}");
        let finding = &findings[0];
        assert_eq!(graph.definition(finding.root).name, "u");
        assert_eq!(graph.definition(finding.later).name, "v1");
        assert_eq!(
            finding
                .path
                .iter()
                .map(|vertex| graph.definition(*vertex).name.as_str())
                .collect::<Vec<_>>(),
            ["u", "ping", "v1"]
        );
    }

    /// [04-INF-4]/[04-INF-8] de-duplication lock: without a prior context
    /// binding, the direct forward reference remains infer_var's one located
    /// error even when an alternative indirect path reaches the same value.
    #[test]
    fn direct_forward_reference_with_an_indirect_alternative_is_reported_once() {
        let exprs = surf_program(
            "module DirectWins\n\n\
             u = add(v1, ping(1))\n\n\
             v1 = 5\n\n\
             def ping(n: i32) -> i32 = add(n, v1)\n",
        );
        let report = crate::check_ir_program(&exprs).expect_err("direct read must reject");
        let matching = report
            .errors
            .iter()
            .filter(|error| {
                matches!(
                    &error.kind,
                    CheckErrorKind::UnboundVariable { identifier } if identifier == "v1"
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "{:#?}", report.errors);
        assert!(
            !matching[0].message.contains("[04-INF-8]"),
            "the existing direct diagnostic owns de-duplication: {:#?}",
            report.errors
        );
    }

    #[test]
    fn pure_direct_forward_reference_remains_one_diagnostic() {
        let exprs = surf_program(
            "module PureDirect\n\n\
             root = add(later, (1 : i32))\n\n\
             later = (5 : i32)\n",
        );
        let report = crate::check_ir_program(&exprs).expect_err("direct read must reject");
        let matching = report
            .errors
            .iter()
            .filter(|error| {
                matches!(
                    &error.kind,
                    CheckErrorKind::UnboundVariable { identifier } if identifier == "later"
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "{:#?}", report.errors);
        assert!(!matching[0].message.contains("[04-INF-8]"));
    }

    /// A direct same-name read may resolve a prior library binding, while a
    /// function declared after the current-unit value captures that new value.
    /// The direct graph edge must not hide the distinct indirect dependency.
    #[test]
    fn prior_context_direct_binding_does_not_hide_current_unit_indirect_dependency() {
        let library = surf_program("module Prior\n\nlater = (1 : i32)\n");
        let context = crate::build_type_env_from_library(&library)
            .expect("the prior i32 binding builds a reusable context");
        let current = surf_program(
            "module Current\n\n\
             root = add(later, read_current(0))\n\n\
             later = (5 : i32)\n\n\
             def read_current(n: i32) -> i32 = add(n, later)\n",
        );
        let report = crate::check_ir_with_context(&context, &current)
            .expect_err("the indirect dependency on current-unit later must reject");
        let matching = report
            .errors
            .iter()
            .filter(|error| {
                matches!(
                    &error.kind,
                    CheckErrorKind::UnboundVariable { identifier } if identifier == "later"
                ) && error.message.contains("`root`")
                    && error.message.contains("[04-INF-8]")
            })
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "{:#?}", report.errors);
    }

    /// An unrelated root's direct [04-INF-4] error for the same identifier
    /// does not own or suppress this root's distinct indirect [04-INF-8] path.
    #[test]
    fn unrelated_direct_error_does_not_suppress_another_roots_indirect_dependency() {
        let exprs = surf_program(
            "module UnrelatedDirect\n\n\
             unrelated = later\n\n\
             root = read_current(0)\n\n\
             later = (5 : i32)\n\n\
             def read_current(n: i32) -> i32 = add(n, later)\n",
        );
        let report = crate::check_ir_program(&exprs)
            .expect_err("both roots' distinct initialization errors must reject");
        let direct = report
            .errors
            .iter()
            .filter(|error| {
                matches!(
                    &error.kind,
                    CheckErrorKind::UnboundVariable { identifier } if identifier == "later"
                ) && !error.message.contains("[04-INF-8]")
            })
            .count();
        let indirect = report
            .errors
            .iter()
            .filter(|error| {
                matches!(
                    &error.kind,
                    CheckErrorKind::UnboundVariable { identifier } if identifier == "later"
                ) && error.message.contains("`root`")
                    && error.message.contains("[04-INF-8]")
            })
            .count();
        assert_eq!(direct, 1, "{:#?}", report.errors);
        assert_eq!(indirect, 1, "{:#?}", report.errors);
    }

    /// A reachable but separate eager cycle does not take precedence over an
    /// independent root-to-later [04-INF-8] violation.
    #[test]
    fn separate_reachable_cycle_preserves_root_later_diagnostic() {
        let exprs = surf_program(
            "module SeparateCycle\n\n\
             root = add(read_later(0), enter_cycle())\n\n\
             later = (5 : i32)\n\n\
             cycle_value = enter_cycle()\n\n\
             def read_later(n: i32) -> i32 = add(n, later)\n\n\
             def enter_cycle() -> i32 = cycle_value\n",
        );
        let report = crate::check_ir_program(&exprs)
            .expect_err("both the eager cycle and indirect later dependency must reject");
        assert!(
            report
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::CycleDetected)),
            "{:#?}",
            report.errors
        );
        let matching = report
            .errors
            .iter()
            .filter(|error| {
                matches!(
                    &error.kind,
                    CheckErrorKind::UnboundVariable { identifier } if identifier == "later"
                ) && error.message.contains("`root`")
                    && error.message.contains("[04-INF-8]")
            })
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "{:#?}", report.errors);
    }

    /// Negative parity: backward and independent values do not acquire an
    /// [04-INF-8] edge merely because a function is involved.
    #[test]
    fn backward_and_independent_values_stay_out_of_the_later_set() {
        for source in [
            "module Backward\n\nv1 = 5\n\nu = ping(1)\n\ndef ping(n: i32) -> i32 = add(n, v1)\n",
            "module Independent\n\nu = ping(1)\n\nv1 = 5\n\ndef ping(n: i32) -> i32 = n\n",
        ] {
            let graph = graph(source);
            assert!(
                graph
                    .acyclic_later_eager_value_dependencies()
                    .expect("uncancelled analysis completes")
                    .is_empty(),
                "{source}"
            );
        }
    }

    /// [04-INF-7] regression: lambda bodies and applications feed the same
    /// full SCC projection that Slice C consumes.
    #[test]
    fn eager_lambda_cycle_is_one_component() {
        let graph = graph(
            "module LambdaCycle\n\n\
             carried = map(fn (x: i32) -> f(x), [1, 2])\n\n\
             def f(n: i32) -> i32 = add(n, carried)\n",
        );
        let carried = graph.vertex_by_name["carried"];
        let f = graph.vertex_by_name["f"];
        let projection = graph.inference_components();
        assert!(projection.complete);
        let component_index = projection.component_by_item[graph.definition(carried).item_index]
            .expect("carried is scheduled in a definition component");
        assert_eq!(
            projection.component_by_item[graph.definition(f).item_index],
            Some(component_index)
        );
        assert!(projection.components[component_index].cyclic);
        assert_eq!(projection.components[component_index].members.len(), 2);
        let components = graph.cyclic_eager_components();
        assert_eq!(components.len(), 1, "{components:#?}");
        assert!(components[0].contains(&carried), "{components:#?}");
        assert!(components[0].contains(&f), "{components:#?}");
        assert!(
            graph
                .acyclic_later_eager_value_dependencies()
                .expect("uncancelled analysis completes")
                .is_empty()
        );
    }

    /// Public-check diagnostic lock for [04-INF-8]: graph reachability adds one
    /// error, not one per traversal path, and the message names both endpoints.
    #[test]
    fn checker_reports_one_root_later_diagnostic_for_the_indirect_shape() {
        let source = "module IndirectLaterDiagnostic\n\n\
                      u = ping(1)\n\n\
                      v1 = 5\n\n\
                      def ping(n: i32) -> i32 = add(n, v1)\n";
        let exprs = surf_program(source);
        let report = crate::check_ir_program(&exprs).expect_err("[04-INF-8] must reject");
        let matching = report
            .errors
            .iter()
            .filter(|error| {
                matches!(
                    &error.kind,
                    CheckErrorKind::UnboundVariable { identifier } if identifier == "v1"
                ) && error.message.contains("`u`")
                    && error.message.contains("`v1`")
                    && error.message.contains("[04-INF-8]")
            })
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "{:#?}", report.errors);
    }

    #[test]
    fn cancellation_during_later_reachability_emits_only_cancellation() {
        let exprs = surf_program(
            "module CancelLaterDependency\n\n\
             root = read_current(0)\n\n\
             later = (5 : i32)\n\n\
             def read_current(n: i32) -> i32 = add(n, later)\n",
        );
        let token = CancelToken::new();
        let _cancel_guard = crate::cancel::install_cancel_token(token.clone());
        let _reachability_hook = cancel_later_dependency_after_edges_for_test(1, token);
        let report = crate::check_ir_program(&exprs)
            .expect_err("cancelled initialization analysis must reject");
        assert_eq!(report.errors.len(), 1, "{:#?}", report.errors);
        assert!(
            report
                .errors
                .iter()
                .all(|error| crate::cancel::is_cancellation(&error.message)),
            "cancelled reachability must publish only the cancellation diagnostic: {:#?}",
            report.errors
        );
    }

    #[test]
    fn cancellation_during_later_reachability_survives_prior_source_errors() {
        let exprs = surf_program(
            "module CancelAfterError\n\n\
             broken = missing\n\n\
             root = read_current(0)\n\n\
             later = (5 : i32)\n\n\
             def read_current(n: i32) -> i32 = add(n, later)\n",
        );
        let token = CancelToken::new();
        let _cancel_guard = crate::cancel::install_cancel_token(token.clone());
        let _reachability_hook = cancel_later_dependency_after_edges_for_test(1, token);
        let report = crate::check_ir_program(&exprs)
            .expect_err("source failure plus cancelled initialization analysis must reject");
        assert!(
            report.errors.iter().any(|error| matches!(
                &error.kind,
                CheckErrorKind::UnboundVariable { identifier } if identifier == "missing"
            )),
            "{:#?}",
            report.errors
        );
        assert!(
            report
                .errors
                .iter()
                .any(|error| crate::cancel::is_cancellation(&error.message)),
            "a pre-existing source error must not bypass cancellation: {:#?}",
            report.errors
        );
        assert!(
            report
                .errors
                .iter()
                .all(|error| !error.message.contains("[04-INF-8]")),
            "cancelled reachability must not publish a partial policy report: {:#?}",
            report.errors
        );
    }
}
