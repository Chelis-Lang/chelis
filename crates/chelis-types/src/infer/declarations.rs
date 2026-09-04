//! Declaration collection, dependency analysis, and signature schedules.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

pub(super) fn top_level_decl_name(expr: &deep::Expr) -> Option<&str> {
    let (tag, _, kids) = stamped_parts(expr)?;
    if !matches!(
        tag,
        DeepTag::Def | DeepTag::Defsig | DeepTag::Deftype | DeepTag::Typealias
    ) {
        return None;
    }
    kids.first().and_then(symbol_name)
}

pub(super) fn top_level_decl_items(exprs: &[deep::Expr]) -> Vec<&deep::Expr> {
    fn push<'a>(expr: &'a deep::Expr, out: &mut Vec<&'a deep::Expr>) {
        if let Some((DeepTag::Module, _, kids)) = stamped_parts(expr) {
            // `(module {} name children...)` — skip the name child.
            for child in kids.iter().skip(1) {
                push(child, out);
            }
            return;
        }
        out.push(expr);
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, &mut out);
    }
    out
}

/// Like [`top_level_decl_items`] but pairs each flattened item with
/// its enclosing lexical module key: nested `(module ...)` names
/// joined with `.`, `None` for items outside any wrapper. This is the
/// module-identity source for checker-enforced opacity (RFC D-CHECK);
/// reef package-linked items carry no wrapper and key through their
/// internal-name stem instead (see `opacity::module_key_for_item`).
pub(super) fn top_level_decl_items_with_modules(
    exprs: &[deep::Expr],
) -> Vec<(Option<String>, &deep::Expr)> {
    fn push<'a>(
        expr: &'a deep::Expr,
        prefix: Option<&str>,
        out: &mut Vec<(Option<String>, &'a deep::Expr)>,
    ) {
        if let Some((DeepTag::Module, _, kids)) = stamped_parts(expr) {
            // `(module {} name children...)` — skip the name child.
            let name = kids.first().and_then(symbol_name);
            let key = match (prefix, name) {
                (Some(p), Some(n)) => Some(format!("{p}.{n}")),
                (None, Some(n)) => Some(n.to_string()),
                (p, None) => p.map(str::to_string),
            };
            for child in kids.iter().skip(1) {
                push(child, key.as_deref(), out);
            }
            return;
        }
        out.push((prefix.map(str::to_string), expr));
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, None, &mut out);
    }
    out
}

/// RFC v4b (RT-1 F2): a named module may be opened by at most one
/// `(module ...)` wrapper per check unit. Module identity is otherwise
/// a forgeable string -- a second wrapper of an opaque type's defining
/// module would construct and inspect the type as if it were inside.
/// Walks every wrapper (including nested ones, keyed by their full
/// `.`-joined path) and emits ONE `DuplicateModule` error per
/// re-opened name. Surf emits one module per file and reef strips
/// wrappers before inference, so this only fires on hand-written `.dp`
/// (the forge surface).
pub(super) fn detect_module_reopens(exprs: &[deep::Expr], errors: &mut DiagnosticSink<'_>) {
    fn walk(
        expr: &deep::Expr,
        prefix: Option<&str>,
        seen: &mut UnordSet<String>,
        reported: &mut UnordSet<String>,
        errors: &mut DiagnosticSink<'_>,
    ) {
        // chelis#1107: carrier-preserving read. A `List`-only destructure
        // returned on every stamped `module`, so the reopen check never ran
        // on `check_typed_program`.
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            return;
        };
        if tag != DeepTag::Module {
            return;
        }
        let name = kids.first().and_then(symbol_name);
        let key = match (prefix, name) {
            (Some(p), Some(n)) => Some(format!("{p}.{n}")),
            (None, Some(n)) => Some(n.to_string()),
            (p, None) => p.map(str::to_string),
        };
        if let Some(key) = &key
            && !seen.insert(key.clone())
            && reported.insert(key.clone())
        {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateModule,
                format!(
                    "module `{key}` is opened by more than one module wrapper in this \
                     check unit; a named module may be opened at most once"
                ),
                vec![format!(
                    "merge the `{key}` wrappers into one, or rename one of them"
                )],
            ));
        }
        for child in kids.iter().skip(1) {
            walk(child, key.as_deref(), seen, reported, errors);
        }
    }
    let mut seen = UnordSet::new();
    let mut reported = UnordSet::new();
    for expr in exprs {
        walk(expr, None, &mut seen, &mut reported, errors);
    }

    // RFC v5 belt-and-suspenders (RT-1 F2 bypass): a stem-derived
    // module identity (from a top-level mangled `deftype`/`def` name)
    // that collides with a lexical wrapper key in the same check unit
    // is also a `DuplicateModule` error. Genuine linker output has NO
    // lexical wrappers, so this never fires on it; it defends the
    // stem-plus-wrapper forge shapes even if the name-format check is
    // somehow bypassed. Runs unconditionally (structural).
    for (lexical, expr) in top_level_decl_items_with_modules(exprs) {
        // Only flat (non-wrapped) mangled declarations introduce a
        // stem-derived module identity; a name inside a lexical
        // wrapper keys to the wrapper, not its stem.
        if lexical.is_some() {
            continue;
        }
        // chelis#1107: carrier-preserving read, as in `walk` above.
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if !matches!(tag, DeepTag::Deftype | DeepTag::Def) {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(stem_key) = crate::opacity::reef_module_stem(name) else {
            continue;
        };
        if seen.contains(&stem_key) && reported.insert(stem_key.clone()) {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateModule,
                format!(
                    "module `{stem_key}` is opened by both a lexical wrapper and a \
                     reef-stem mangled name in this check unit; a named module may be \
                     opened at most once"
                ),
                vec![format!(
                    "rename the mangled declaration; the `{stem_key}` lexical module \
                     already exists"
                )],
            ));
        }
    }
}

/// RFC v5 (RT-1 F2 bypass): the reef package-linker's internal-name
/// format (`Pkg__<pkg>__<Module>__<Name>` / lowercase twin) is the
/// linker's PRIVATE output. A program NOT produced by the linker
/// (raw `.ch` or raw `.dp`) that uses it forges module identity
/// through the reef-stem channel, so any top-level declaration whose
/// binding name matches the format is a declaration error. Skipped
/// entirely when the linked-program provenance flag is set (the
/// linker's own output is accepted). The linker also re-mangles every
/// user source name, so user code inside a real package cannot smuggle
/// a clean mangled name into linked output.
pub(super) fn detect_forged_linker_names(exprs: &[deep::Expr], errors: &mut DiagnosticSink<'_>) {
    if crate::opacity::linked_program() {
        return;
    }
    for expr in top_level_decl_items(exprs) {
        // chelis#1107: carrier-preserving read. A `List`-only destructure
        // skipped every stamped declaration, so this forgery guard ran only
        // on `check_ir_program`.
        //
        // `defmacro` is compiler-internal pre-expansion syntax outside the
        // vocabulary; it stays symbol-headed (raw-string boundary), so it
        // never decodes and is matched on its head string instead.
        let (kids, is_declaration) = match stamped_parts(expr) {
            Some((tag, _, kids)) => (
                kids,
                matches!(
                    tag,
                    DeepTag::Deftype | DeepTag::Def | DeepTag::Defsig | DeepTag::Typealias
                ),
            ),
            None => match expr {
                deep::Expr::List(list, _) => (
                    children(list),
                    list.unknown_tag_symbol() == Some("defmacro"),
                ),
                deep::Expr::UnknownForm(data) => {
                    (data.children.as_slice(), data.head == "defmacro")
                }
                _ => continue,
            },
        };
        if !is_declaration {
            continue;
        }
        if let Some(name) = kids.first().and_then(symbol_name)
            && crate::opacity::is_linker_format_name(name)
        {
            errors.push(crate::opacity::forged_linker_name_error(name));
        }
    }
}

pub(super) fn infer_signature_metadata_with_context_and_headers(
    exprs: &[deep::Expr],
    function_plan: &FunctionInferencePlan,
    type_env: &BTreeMap<String, deep::Expr>,
    signature_context: &SignatureInferenceMetadata,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> SignatureInferenceMetadata {
    let defsig_names = collect_defsig_names(exprs);
    let authored_signature_types = collect_authored_signature_types(exprs, type_headers, errors);
    let recursive_members = function_plan.recursive_member_names();
    let mut functions = BTreeMap::new();
    let mut defs_by_name = UnordMap::<String, VecDeque<&deep::Expr>>::new();
    for expr in top_level_decl_items(exprs) {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if kids
            .get(1)
            .and_then(|body| tagged_children(body, DeepTag::Fn))
            .is_none()
        {
            continue;
        }
        defs_by_name
            .entry(name.to_string())
            .or_default()
            .push_back(expr);
    }
    let ordered_defs = function_plan
        .ordered_members()
        .filter_map(|member| defs_by_name.get_mut(&member.name)?.pop_front())
        .collect::<Vec<_>>();
    let passes = ordered_defs.len().max(1);
    let imported_signatures = signature_context
        .functions
        .iter()
        .map(|(name, inference)| (name.clone(), inference.display_signature.clone()))
        .collect::<UnordMap<_, _>>();

    // chelis#930: cooperative cancellation at declaration granularity. This
    // fixed point runs one full sweep of every def per def (`passes` is the
    // def count), which makes it the front end's other declaration-count-
    // scaling pass — and on a large program the single most expensive one.
    // Polling the inner loop rather than the outer sweep keeps the bound
    // independent of program size: one declaration, not one O(n) sweep.
    let cancel = crate::cancel::current_cancel_token();
    'fixed_point: for _ in 0..passes {
        functions.clear();
        let mut available_signatures = imported_signatures.clone();
        for expr in &ordered_defs {
            if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
                break 'fixed_point;
            }
            let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
                continue;
            };
            let Some(name) = kids.first().and_then(symbol_name) else {
                continue;
            };
            let Some(fn_kids) = kids
                .get(1)
                .and_then(|body| tagged_children(body, DeepTag::Fn))
            else {
                continue;
            };
            let Some(checked_signature) = type_env
                .get(name)
                .and_then(|expr| type_from_deep_expr(expr, type_headers, errors))
            else {
                continue;
            };
            let Type::Fn(checked_args, checked_ret) = checked_signature.clone() else {
                continue;
            };
            let Some(params_expr) = fn_kids.first() else {
                continue;
            };
            let Some(body) = fn_kids.get(1) else {
                continue;
            };
            let param_infos = param_source_infos(params_expr);
            let recursive_cycle = recursive_members.contains(name);
            let all_written_by_defsig =
                defsig_names.contains(name) && param_infos.iter().all(|(_, written)| !*written);
            let mut display_args = checked_args.clone();
            let mut params = Vec::new();

            for (index, (pname, param_written)) in param_infos.iter().enumerate() {
                let Some(checked_type) = checked_args.get(index).cloned() else {
                    continue;
                };
                let written = all_written_by_defsig || *param_written;
                let can_infer = !recursive_cycle
                    && !written
                    && type_contains_tensor(&checked_type)
                    && !matches!(checked_type, Type::Ref(_));
                let inferred_read_only = can_infer
                    && !param_has_consuming_use_with_headers(
                        body,
                        pname,
                        &available_signatures,
                        type_env,
                        type_headers,
                        errors,
                    );
                let display_type = if inferred_read_only {
                    Type::Ref(Box::new(checked_type.clone()))
                } else {
                    checked_type.clone()
                };
                if let Some(slot) = display_args.get_mut(index) {
                    *slot = display_type.clone();
                }
                params.push(ParamSignatureInference {
                    index,
                    name: pname.clone(),
                    written,
                    inferred_read_only,
                    checked_type,
                    display_type,
                });
            }

            let display_signature = Type::Fn(display_args, checked_ret);
            available_signatures.insert(name.to_string(), display_signature.clone());
            functions.insert(
                name.to_string(),
                FunctionSignatureInference {
                    name: name.to_string(),
                    authored_signature: defsig_names.contains(name),
                    authored_signature_type: authored_signature_types.get(name).cloned(),
                    recursive_cycle,
                    checked_signature,
                    display_signature,
                    params,
                },
            );
        }
    }

    SignatureInferenceMetadata { functions }
}

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

    fn cyclic_eager_components(&self) -> Vec<Vec<usize>> {
        if !self.complete {
            return Vec::new();
        }
        let adjacency = self.adjacency();
        let Some(mut components) = unprofiled_scc_vertex_components(
            &adjacency,
            crate::cancel::current_cancel_token().as_ref(),
        ) else {
            return Vec::new();
        };
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
        for component in &mut components {
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
        components
    }

    fn cycle_path(&self, component: &[usize]) -> Vec<usize> {
        let adjacency = self.adjacency();
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
            return vec![start, start];
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
                if vertex == start {
                    let mut reverse = vec![start];
                    let mut current = start;
                    while current != first {
                        current = predecessor[&current];
                        reverse.push(current);
                    }
                    reverse.reverse();
                    let mut path = vec![start];
                    path.extend(reverse);
                    return path;
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

    pub(super) fn report_eager_cycle_errors(&self, errors: &mut DiagnosticSink<'_>) {
        for component in self.cyclic_eager_components() {
            let path = self.cycle_path(&component);
            let names = path
                .iter()
                .map(|vertex| self.definitions[*vertex].name.as_str())
                .collect::<Vec<_>>();
            errors.push(CheckError::new(
                CheckErrorKind::CycleDetected,
                format!("binding cycle: {}", names.join(" -> ")),
                vec![
                    "Break the cycle by removing one of the self-referential definitions or \
                     replacing it with a concrete value."
                        .to_string(),
                ],
            ));
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
            for (_, value) in &map.entries {
                collect_top_level_references(value, vertex_by_name, bound, references);
            }
        }
        // Metadata describes the expression; it is not executed as part of a
        // top-level initializer. The stamped expression itself still is.
        deep::Expr::MetaExpr(meta, _) => {
            collect_top_level_references(&meta.expr, vertex_by_name, bound, references)
        }
        deep::Expr::List(list, _) => match get_tag(list) {
            Some(DeepTag::App) => {
                let kids = children(list);
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
            Some(DeepTag::Var) => {
                if let Some(name) = children(list).first().and_then(symbol_name)
                    && vertex_by_name.contains_key(name)
                    && !is_bound_name(name, bound)
                {
                    references.insert(TopLevelReference {
                        target: vertex_by_name[name],
                        kind: TopLevelReferenceKind::Read,
                    });
                }
            }
            Some(DeepTag::Fn) => {
                let kids = children(list);
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
            Some(DeepTag::Let) => {
                let kids = children(list);
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
            Some(DeepTag::Match) => {
                let kids = children(list);
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
                for child in children(list) {
                    collect_top_level_references(child, vertex_by_name, bound, references);
                }
            }
        },
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            collect_top_level_references(&bridged, vertex_by_name, bound, references);
        }
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

pub(super) fn collect_defsig_names(exprs: &[deep::Expr]) -> UnordSet<String> {
    let mut names = UnordSet::new();
    for expr in top_level_decl_items(exprs) {
        if let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr)
            && let Some(name) = kids.first().and_then(symbol_name)
        {
            names.insert(name.to_string());
        }
    }
    names
}

pub(super) fn collect_authored_signature_types(
    exprs: &[deep::Expr],
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> UnordMap<String, Type> {
    let mut signatures = UnordMap::new();
    for expr in top_level_decl_items(exprs) {
        let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(signature_expr)) = (kids.first().and_then(symbol_name), kids.get(1))
        else {
            continue;
        };
        if let Some(signature) = type_from_deep_expr(signature_expr, type_headers, errors) {
            signatures.insert(name.to_string(), signature);
        }
    }
    signatures
}

pub(super) fn collect_top_level_calls(
    expr: &deep::Expr,
    def_names: &UnordSet<String>,
    bound: &mut Vec<UnordSet<String>>,
    calls: &mut UnordSet<String>,
) {
    // Bail before this walker's own unbounded recursion exhausts the native
    // stack on a deeply-nested `app` body. This pass accumulates into
    // `calls`/`bound` and carries no error vector, so it cannot push a
    // diagnostic itself; the guard records the bail in `STACK_EXHAUSTED` so
    // the check entry boundary still turns it into a hard located failure
    // (never a silent partial collection). See `STACK_RED_ZONE_BYTES`.
    stack_guard!("collect_top_level_calls", expr);
    match expr {
        deep::Expr::Atom(_, _) => {}
        deep::Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                collect_top_level_calls(value, def_names, bound, calls);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            collect_top_level_calls(&meta.expr, def_names, bound, calls)
        }
        deep::Expr::List(list, _) => match get_tag(list) {
            Some(DeepTag::App) => {
                let kids = children(list);
                if let Some(callee) = kids.first().and_then(var_name_expr)
                    && def_names.contains(callee)
                    && !is_bound_name(callee, bound)
                {
                    calls.insert(callee.to_string());
                }
                for child in kids {
                    collect_top_level_calls(child, def_names, bound, calls);
                }
            }
            // A bare reference (alias binding, argument position, returned
            // value) is a dependency edge too: an aliased in-group call is
            // still recursion, and the §3.1.1 uniformity check only sees a
            // group the SCC planner reports (spec/04 §3.1.1).
            Some(DeepTag::Var) => {
                if let Some(name) = children(list).first().and_then(symbol_name)
                    && def_names.contains(name)
                    && !is_bound_name(name, bound)
                {
                    calls.insert(name.to_string());
                }
            }
            Some(DeepTag::Fn) => {
                let kids = children(list);
                if kids.len() >= 2 {
                    bound.push(
                        param_source_infos(&kids[0])
                            .into_iter()
                            .map(|(n, _)| n)
                            .collect(),
                    );
                    collect_top_level_calls(&kids[1], def_names, bound, calls);
                    bound.pop();
                }
            }
            Some(DeepTag::Let) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return;
                }
                let mut let_names = UnordSet::new();
                if let Some(bind_kids) = kids
                    .first()
                    .and_then(|bind| tagged_children(bind, DeepTag::Bind))
                {
                    let mut index = 0;
                    while index + 1 < bind_kids.len() {
                        collect_top_level_calls(&bind_kids[index + 1], def_names, bound, calls);
                        if let Some(name) = symbol_name(&bind_kids[index]) {
                            let_names.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                bound.push(let_names);
                collect_top_level_calls(&kids[1], def_names, bound, calls);
                bound.pop();
            }
            Some(DeepTag::Match) => {
                let kids = children(list);
                if let Some(scrutinee) = kids.first() {
                    collect_top_level_calls(scrutinee, def_names, bound, calls);
                }
                for arm in kids.iter().skip(1) {
                    let Some(arm_kids) = tagged_children(arm, DeepTag::Arm) else {
                        continue;
                    };
                    if arm_kids.len() < 3 {
                        continue;
                    }
                    bound.push(pattern_names_for_signature(&arm_kids[0]));
                    collect_top_level_calls(&arm_kids[1], def_names, bound, calls);
                    collect_top_level_calls(&arm_kids[2], def_names, bound, calls);
                    bound.pop();
                }
            }
            _ => {
                for child in children(list) {
                    collect_top_level_calls(child, def_names, bound, calls);
                }
            }
        },
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            collect_top_level_calls(&bridged, def_names, bound, calls);
        }
        deep::Expr::BareList(elems, _) => {
            for child in elems {
                collect_top_level_calls(child, def_names, bound, calls);
            }
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                collect_top_level_calls(child, def_names, bound, calls);
            }
        }
    }
}

pub(crate) fn param_has_consuming_use(
    expr: &deep::Expr,
    param: &str,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
) -> Result<bool, InferResult> {
    crate::session::param_has_consuming_use(
        expr,
        param,
        available_signatures,
        type_env,
        type_headers,
    )
}

pub(crate) fn param_has_consuming_use_in_session(
    expr: &deep::Expr,
    param: &str,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    param_has_consuming_use_with_headers(
        expr,
        param,
        available_signatures,
        type_env,
        type_headers,
        errors,
    )
}

pub(super) fn param_has_consuming_use_with_headers(
    expr: &deep::Expr,
    param: &str,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let mut bound = Vec::new();
    param_has_consuming_use_inner(
        expr,
        param,
        &mut bound,
        available_signatures,
        type_env,
        type_headers,
        errors,
    )
}

pub(super) fn param_has_consuming_use_inner(
    expr: &deep::Expr,
    param: &str,
    bound: &mut Vec<UnordSet<String>>,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    stack_guard!("param_has_consuming_use_inner", expr, false);
    match expr {
        deep::Expr::Atom(_, _) => false,
        deep::Expr::Map(map, _) => map.entries.iter().any(|(_, value)| {
            param_has_consuming_use_inner(
                value,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            )
        }),
        deep::Expr::MetaExpr(meta, _) => param_has_consuming_use_inner(
            &meta.expr,
            param,
            bound,
            available_signatures,
            type_env,
            type_headers,
            errors,
        ),
        deep::Expr::List(list, _) if list.unknown_tag_symbol() == Some("drop") => {
            // Legacy internal `drop` spelling, outside the vocabulary;
            // recognized at the raw-string boundary.
            children(list)
                .first()
                .is_some_and(|child| expr_mentions_unshadowed_name(child, param, bound))
        }
        deep::Expr::List(list, _) => match get_tag(list) {
            Some(DeepTag::Var) => {
                var_name_list(list) == Some(param) && !is_bound_name(param, bound)
            }
            Some(DeepTag::Borrow) | Some(DeepTag::Copy) => {
                children(list).first().is_some_and(|child| {
                    param_nested_consuming_use(
                        child,
                        param,
                        bound,
                        available_signatures,
                        type_env,
                        type_headers,
                        errors,
                    )
                })
            }
            Some(DeepTag::Realize) => children(list)
                .first()
                .is_some_and(|child| expr_mentions_unshadowed_name(child, param, bound)),
            Some(DeepTag::App) => app_consumes_param(
                list,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            ),
            Some(DeepTag::Pipe) => pipe_consumes_param(
                list,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            ),
            Some(DeepTag::Fn) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                if expr_mentions_unshadowed_name(&kids[1], param, bound) {
                    return true;
                }
                false
            }
            Some(DeepTag::Let) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                let mut let_names = UnordSet::new();
                if let Some(bind_kids) = kids
                    .first()
                    .and_then(|bind| tagged_children(bind, DeepTag::Bind))
                {
                    let mut index = 0;
                    while index + 1 < bind_kids.len() {
                        if param_has_consuming_use_inner(
                            &bind_kids[index + 1],
                            param,
                            bound,
                            available_signatures,
                            type_env,
                            type_headers,
                            errors,
                        ) {
                            return true;
                        }
                        if let Some(name) = symbol_name(&bind_kids[index]) {
                            let_names.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                bound.push(let_names);
                let result = param_has_consuming_use_inner(
                    &kids[1],
                    param,
                    bound,
                    available_signatures,
                    type_env,
                    type_headers,
                    errors,
                );
                bound.pop();
                result
            }
            Some(DeepTag::Match) => {
                let kids = children(list);
                if kids
                    .first()
                    .is_some_and(|scrutinee| expr_mentions_unshadowed_name(scrutinee, param, bound))
                {
                    return true;
                }
                for arm in kids.iter().skip(1) {
                    let Some(arm_kids) = tagged_children(arm, DeepTag::Arm) else {
                        continue;
                    };
                    if arm_kids.len() < 3 {
                        continue;
                    }
                    bound.push(pattern_names_for_signature(&arm_kids[0]));
                    let consumes = param_has_consuming_use_inner(
                        &arm_kids[1],
                        param,
                        bound,
                        available_signatures,
                        type_env,
                        type_headers,
                        errors,
                    ) || param_has_consuming_use_inner(
                        &arm_kids[2],
                        param,
                        bound,
                        available_signatures,
                        type_env,
                        type_headers,
                        errors,
                    );
                    bound.pop();
                    if consumes {
                        return true;
                    }
                }
                false
            }
            _ => children(list).iter().any(|child| {
                param_has_consuming_use_inner(
                    child,
                    param,
                    bound,
                    available_signatures,
                    type_env,
                    type_headers,
                    errors,
                )
            }),
        },
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            param_has_consuming_use_inner(
                &bridged,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            )
        }
        deep::Expr::BareList(elems, _) => elems.iter().any(|child| {
            param_has_consuming_use_inner(
                child,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            )
        }),
        deep::Expr::UnknownForm(data) => data.children.iter().any(|child| {
            param_has_consuming_use_inner(
                child,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            )
        }),
    }
}

pub(super) fn param_nested_consuming_use(
    expr: &deep::Expr,
    param: &str,
    bound: &mut Vec<UnordSet<String>>,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    if is_direct_unshadowed_var(expr, param, bound) {
        return false;
    }
    param_has_consuming_use_inner(
        expr,
        param,
        bound,
        available_signatures,
        type_env,
        type_headers,
        errors,
    )
}

pub(super) fn app_consumes_param(
    list: &deep::List,
    param: &str,
    bound: &mut Vec<UnordSet<String>>,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let kids = children(list);
    let callee = kids.first().and_then(var_name_expr);
    if let Some(func) = kids.first()
        && !matches!(callee, Some(name) if name != param)
        && param_has_consuming_use_inner(
            func,
            param,
            bound,
            available_signatures,
            type_env,
            type_headers,
            errors,
        )
    {
        return true;
    }
    for (index, arg) in kids.iter().skip(1).enumerate() {
        if borrow_inner_for_signature(arg)
            .is_some_and(|inner| is_direct_unshadowed_var(inner, param, bound))
        {
            continue;
        }
        if is_direct_unshadowed_var(arg, param, bound) {
            if callee_arg_is_borrowed(
                callee,
                index,
                available_signatures,
                type_env,
                type_headers,
                errors,
            ) {
                continue;
            }
            return true;
        }
        if param_has_consuming_use_inner(
            arg,
            param,
            bound,
            available_signatures,
            type_env,
            type_headers,
            errors,
        ) {
            return true;
        }
    }
    false
}

pub(super) fn pipe_consumes_param(
    list: &deep::List,
    param: &str,
    bound: &mut Vec<UnordSet<String>>,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let kids = children(list);
    if kids.is_empty() {
        return false;
    }
    let mut current = &kids[0];
    for stage in &kids[1..] {
        // Issue #229 (sibling sweep of chelis#226): peer through any
        // synthesized `__chelis_pipe` lambda the desugarer emits for
        // explicit-arg pipe stages so the borrow-arg classifier sees
        // the inner callee and the piped value's actual arg position
        // — not the lambda's type. Without this peering, every
        // non-bare-var pipe stage is mis-classified as a consuming
        // use, the wrapping function never gets auto-borrow inferred,
        // and downstream calls spuriously consume their argument.
        let (_callee_expr, callee_builtin, piped_arg_index) =
            crate::pipe_stage::resolve_pipe_stage_callee(stage);
        if is_direct_unshadowed_var(current, param, bound) {
            if !callee_arg_is_borrowed(
                callee_builtin,
                piped_arg_index,
                available_signatures,
                type_env,
                type_headers,
                errors,
            ) {
                return true;
            }
        } else if param_has_consuming_use_inner(
            current,
            param,
            bound,
            available_signatures,
            type_env,
            type_headers,
            errors,
        ) {
            return true;
        }
        current = stage;
    }
    false
}

pub(super) fn callee_arg_is_borrowed(
    callee: Option<&str>,
    index: usize,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let Some(callee) = callee else {
        return false;
    };
    if let Some(Type::Fn(args, _)) = available_signatures.get(callee)
        && args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)))
    {
        return true;
    }
    if let Some(Type::Fn(args, _)) = type_env
        .get(callee)
        .and_then(|expr| type_from_deep_expr(expr, type_headers, errors))
        && args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)))
    {
        return true;
    }
    builtin_arg_is_ref(callee, index)
}

pub(super) fn builtin_arg_is_ref(name: &str, index: usize) -> bool {
    let (env, _) = builtins::builtin_env();
    if let Some(Type::Fn(args, _)) = env.lookup(name).map(|scheme| &scheme.body) {
        return args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)));
    }
    false
}

pub(super) fn expr_mentions_unshadowed_name(
    expr: &deep::Expr,
    name: &str,
    bound: &mut Vec<UnordSet<String>>,
) -> bool {
    stack_guard!("expr_mentions_unshadowed_name", expr, false);
    match expr {
        deep::Expr::Atom(_, _) => false,
        deep::Expr::Map(map, _) => map
            .entries
            .iter()
            .any(|(_, value)| expr_mentions_unshadowed_name(value, name, bound)),
        deep::Expr::MetaExpr(meta, _) => expr_mentions_unshadowed_name(&meta.expr, name, bound),
        deep::Expr::List(list, _) => match get_tag(list) {
            Some(DeepTag::Var) => var_name_list(list) == Some(name) && !is_bound_name(name, bound),
            Some(DeepTag::Fn) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                bound.push(
                    param_source_infos(&kids[0])
                        .into_iter()
                        .map(|(n, _)| n)
                        .collect(),
                );
                let result = expr_mentions_unshadowed_name(&kids[1], name, bound);
                bound.pop();
                result
            }
            _ => children(list)
                .iter()
                .any(|child| expr_mentions_unshadowed_name(child, name, bound)),
        },
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            expr_mentions_unshadowed_name(&bridged, name, bound)
        }
        deep::Expr::BareList(elems, _) => elems
            .iter()
            .any(|child| expr_mentions_unshadowed_name(child, name, bound)),
        deep::Expr::UnknownForm(data) => data
            .children
            .iter()
            .any(|child| expr_mentions_unshadowed_name(child, name, bound)),
    }
}

pub(super) fn type_from_deep_expr(
    expr: &deep::Expr,
    headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    let mut vg = VarGen::default();
    DeepTypeResolver::new(
        TypeUseSite::CompilerMetadata,
        BinderMode::TrustedCompilerMetadata,
        headers,
        &mut vg,
        errors,
    )
    .resolve(expr)
    .ok()
    .map(|ty| ty.into_type())
}

pub(super) fn type_contains_tensor(ty: &Type) -> bool {
    match ty {
        Type::Tensor(_, _) => true,
        Type::Ref(inner) => type_contains_tensor(inner),
        Type::Adt(_, args) | Type::Tuple(args) => args.iter().any(type_contains_tensor),
        Type::KindedAdt(_, args) => args
            .iter()
            .any(|argument| argument.as_type().is_some_and(type_contains_tensor)),
        Type::Fn(_, _) | Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => false,
    }
}

/// Issue #256 round 3: does `ty` carry a tensor, consulting `carriers`
/// for the by-name ADT carry decision? This is the `Type`-level mirror
/// of linearity's `type_expr_contains_tensor`: an ADT carries iff its
/// name is in the precomputed carrier set (its definition has a
/// tensor-carrying field) OR one of its type arguments carries (e.g.
/// `Wrapper[tensor[..]]`). Bare `type_contains_tensor` cannot make the
/// by-name decision — it only sees the `Type::Adt` shell, not the
/// variant fields — which is exactly why the deferred-borrow gate must
/// be handed the carrier set rather than trust an args-only check.
pub(super) fn type_carries_tensor_with_carriers(ty: &Type, carriers: &UnordSet<String>) -> bool {
    match ty {
        Type::Tensor(_, _) => true,
        Type::Ref(inner) => type_carries_tensor_with_carriers(inner, carriers),
        Type::Tuple(args) => args
            .iter()
            .any(|a| type_carries_tensor_with_carriers(a, carriers)),
        Type::Adt(name, args) => {
            carriers.contains(name)
                || args
                    .iter()
                    .any(|a| type_carries_tensor_with_carriers(a, carriers))
        }
        Type::KindedAdt(name, args) => {
            carriers.contains(name)
                || args.iter().any(|argument| {
                    argument
                        .as_type()
                        .is_some_and(|ty| type_carries_tensor_with_carriers(ty, carriers))
                })
        }
        Type::Fn(_, _) | Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => false,
    }
}

/// Issue #256 round 3: compute the set of tensor-carrying ADT names from
/// the registry. This is the registry-backed mirror of linearity's
/// `compute_tensor_carrying_adts` (which works off stamped Deep exprs):
/// fixed-point iteration where an ADT joins the carrier set once any of
/// its variant fields carries a tensor against the in-progress set, so a
/// chain `A { f: B }, B { g: tensor }` resolves transitively. Bounded by
/// the ADT count. The two classifiers must agree: the gate uses this set
/// to reject a deferred borrow that resolved to a non-carrying ADT, and
/// linearity uses its own set to reject the concrete (non-deferred) form.
pub(super) fn adt_carrier_set(adt_reg: &AdtRegistry) -> UnordSet<String> {
    let mut carriers: UnordSet<String> = UnordSet::new();
    loop {
        let mut grew = false;
        for (name, def) in &adt_reg.defs {
            if carriers.contains(name) {
                continue;
            }
            let carries = def.variants.iter().any(|variant| {
                variant
                    .fields
                    .iter()
                    .any(|(_, field_ty)| type_carries_tensor_with_carriers(field_ty, &carriers))
            });
            if carries {
                carriers.insert(name.clone());
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    carriers
}

/// Issue #256 round 2 soundness gate. The `borrow` inference arm accepts a
/// borrow whose inner type is still an unresolved `Type::Var`, recording
/// the variable in the substitution's deferred-borrow ledger. That
/// deferral is sound only when the variable is *eventually* pinned to a
/// tensor or tensor-carrying type by a later unification (the surrounding
/// `&tensor[..]` / `&Carrier[..]` parameter). This pass drains the ledger
/// after a def body's inference completes and re-checks each recorded
/// variable against the now-complete substitution:
///
///   - `Tensor` / `Ref(Tensor)`: pinned to a tensor — sound, accept.
///   - `Adt` / `Tuple` / `Ref(Adt|Tuple)`: an aggregate that *may* carry a
///     tensor. Round 3 (#256 soundness): classify it here against the
///     registry-backed carrier set rather than blanket-accepting and
///     deferring to linearity. Deferring was unsound — round 1 loosened
///     linearity's `expr_is_owned_or_borrow_linear` to accept a stale
///     `(t-var ..)` stamp (so a tensor that resolved late is not
///     rejected), and a deferred borrow that resolves to a *non*-carrying
///     ADT/tuple keeps that same `(t-var ..)` stamp at the linearity
///     layer. Both gates would then wave it through. So the gate, which
///     already holds the final `Type`, must make the carry decision: a
///     tensor-carrying aggregate is accepted, a non-carrying one rejected.
///   - still `Var`: never pinned. A fully-polymorphic consumer (e.g.
///     `consume_any[a](t: a)`) unifies the parameter to `&a` without ever
///     forcing a tensor, so a genuinely-non-tensor value would slip past
///     every other gate. Reject.
///   - `Prim` / `Unit` / `Fn`: pinned to a concretely-non-tensor scalar
///     only after the borrow arm ran (so the arm's own `_ => TypeMismatch`
///     could not fire). Reject.
pub(super) fn validate_deferred_borrow_vars(
    subst: &Subst,
    adt_reg: &AdtRegistry,
    declared_type_names: &UnordMap<TypeVar, String>,
    errors: &mut DiagnosticSink<'_>,
) {
    let deferred = subst.take_deferred_borrow_vars();
    if deferred.is_empty() {
        return;
    }
    // Computed lazily: only programs that actually deferred a borrow pay
    // the fixed-point pass, and only once per drain.
    let carriers = adt_carrier_set(adt_reg);
    for tv in deferred {
        let resolved = subst.apply(&Type::Var(tv));
        // Peel every `Ref` layer: the recorded variable is the borrow
        // inner, but a later unification may have wrapped it in one or
        // more `&` layers (e.g. the parameter type was itself `&T`).
        let mut peeled = &resolved;
        while let Type::Ref(inner) = peeled {
            peeled = inner.as_ref();
        }
        let sound = match peeled {
            // Pinned to a tensor: always sound.
            Type::Tensor(_, _) => true,
            // Pinned to an aggregate: sound iff it actually carries a
            // tensor against the registry carrier set (round 3). A
            // non-carrying record/tuple resolved through the deferred
            // path is rejected here — linearity's loosened classifier
            // can no longer be relied on to catch it.
            Type::Adt(_, _) | Type::KindedAdt(_, _) | Type::Tuple(_) => {
                type_carries_tensor_with_carriers(peeled, &carriers)
            }
            // Don't double-report an inner that already failed inference.
            Type::Error(_) => true,
            // Never pinned, or pinned to a concretely-non-tensor value.
            Type::Var(_) | Type::Prim(_) | Type::Unit | Type::Fn(_, _) => false,
            // `Ref` is fully peeled above; treat as sound to avoid a
            // spurious reject if a future shape reaches here.
            Type::Ref(_) => true,
        };
        if !sound {
            // chelis#260 Site 2 / spec/04 [04-FIT-9]: name the source type
            // parameter when this signature declared one.
            //
            // The lookup is on the RESOLVED variable, not the deferred one.
            // Measured on `def go[t](x: t)`: the resolver mints `t` as ?343,
            // the instantiation for the body renames it to ?344, the borrow
            // site defers a later variable ?345, and ?345 resolves to ?344.
            // The recorded map is keyed by what the instantiation minted, so
            // ?344 is the key that carries the name -- and ?344 is also what
            // this diagnostic prints, which is the coincidence that makes the
            // rendering correct rather than merely adjacent.
            //
            // [04-FIT-10]: with no recorded name the internal identity still
            // renders, because a variable with no source provenance must not
            // be given an invented one.
            let subject = match peeled {
                Type::Var(resolved_var) => match declared_type_names.get(resolved_var) {
                    Some(name) => format!("`{name}`"),
                    None => format!("{peeled}"),
                },
                _ => format!("{peeled}"),
            };
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("borrow requires tensor or tensor-carrying input, got {subject}"),
                vec!["Use `&x` only with tensor values".to_string()],
            ));
        }
    }
}

/// RFC D-CHECK: drain the deferred-access ledger after a def body's
/// inference completes and re-check each recorded target variable
/// against the final substitution: a target pinned to an
/// out-of-module opaque ADT (e.g. an unannotated lambda parameter
/// pinned by a later call) is rejected with the same action text as
/// the typed path. Draining per def keeps attribution exact and
/// prevents one def's deferrals from leaking into the next,
/// mirroring `validate_deferred_borrow_vars`.
pub(super) fn validate_deferred_opaque_uses(
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    let mut any_opaque_in_scope: Option<bool> = None;
    for (tv, use_kind) in subst.take_deferred_opaque_uses() {
        let resolved = subst.apply(&Type::Var(tv));
        let mut peeled = &resolved;
        while let Type::Ref(inner) = peeled {
            peeled = inner.as_ref();
        }
        let action = match use_kind {
            crate::unify::DeferredOpaqueUse::Access => crate::opacity::OpaqueAction::FieldAccess,
            crate::unify::DeferredOpaqueUse::RecordUpdate => {
                crate::opacity::OpaqueAction::RecordUpdate
            }
        };
        match peeled {
            Type::Adt(adt_name, _) | Type::KindedAdt(adt_name, _) => {
                crate::opacity::check_opaque_use(action, adt_name, adt_reg, errors);
            }
            // Never pinned: let-generalization makes an unannotated
            // accessor polymorphic, so callers instantiate FRESH
            // variables and the recorded one stays unbound -- a
            // laundering channel for opaque values. Mirror the
            // deferred-borrow ledger's never-pinned rejection,
            // fail-closed, scoped to check units that declare any
            // opaque type so opaque-free programs keep the lenient
            // status quo.
            Type::Var(_) => {
                let opaque_in_scope = *any_opaque_in_scope
                    .get_or_insert_with(|| adt_reg.defs.values().any(|def| def.opaque));
                if opaque_in_scope
                    && let Some(error) = crate::opacity::with_context(|ctx| {
                        crate::opacity::unresolved_target_error(ctx, action)
                    })
                {
                    errors.push(error);
                }
            }
            _ => {}
        }
    }
}

pub(super) fn param_source_infos(expr: &deep::Expr) -> Vec<(String, bool)> {
    let params = match expr {
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => node.children_slice(),
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Params) => children(list),
        deep::Expr::BareList(elements, _) => elements.as_slice(),
        _ => return Vec::new(),
    };
    params
        .iter()
        .filter_map(|param| match param {
            deep::Expr::Atom(deep::Atom::Name(name), _) => Some((name.clone(), false)),
            deep::Expr::MetaExpr(meta, _) => {
                let deep::Expr::Atom(deep::Atom::Name(name), _) = meta.expr.as_ref() else {
                    return None;
                };
                Some((
                    name.clone(),
                    meta.entries.iter().any(|(key, _)| key == "type"),
                ))
            }
            deep::Expr::List(param_list, _) => {
                let name = param_list.elements.first().and_then(symbol_name)?;
                let written = get_meta(param_list)
                    .is_some_and(|meta| meta.entries.iter().any(|(key, _)| key == "type"));
                Some((name.to_string(), written))
            }
            deep::Expr::BareList(elements, _) => {
                let name = elements.first().and_then(symbol_name)?;
                let written = elements.get(1).is_some_and(|expr| {
                    let deep::Expr::Map(meta, _) = expr else {
                        return false;
                    };
                    meta.entries.iter().any(|(key, _)| key == "type")
                });
                Some((name.to_string(), written))
            }
            _ => None,
        })
        .collect()
}

pub(super) fn pattern_names_for_signature(expr: &deep::Expr) -> UnordSet<String> {
    let mut names = UnordSet::new();
    collect_pattern_names_for_signature(expr, &mut names);
    names
}

pub(super) fn collect_pattern_names_for_signature(expr: &deep::Expr, names: &mut UnordSet<String>) {
    // Bail before unbounded recursion exhausts the native stack on a
    // deeply-nested pattern. No error vector here; the guard records the bail
    // so the check entry boundary fails hard with a located diagnostic. See
    // `STACK_RED_ZONE_BYTES`.
    stack_guard!("collect_pattern_names_for_signature", expr);
    // chelis#1107: carrier-preserving read; a `List`-only destructure collected
    // no pattern binder at all from a stamped `match`.
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                names.insert(name.to_string());
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                names.insert(name.to_string());
            }
            if let Some(inner) = kids.get(1) {
                collect_pattern_names_for_signature(inner, names);
            }
        }
        _ => {
            for child in kids {
                collect_pattern_names_for_signature(child, names);
            }
        }
    }
}

pub(super) fn tagged_children(expr: &deep::Expr, tag: DeepTag) -> Option<&[deep::Expr]> {
    stamped_parts(expr).and_then(|(found, _, children)| (found == tag).then_some(children))
}

pub(super) fn var_name_expr(expr: &deep::Expr) -> Option<&str> {
    tagged_children(expr, DeepTag::Var)?
        .first()
        .and_then(symbol_name)
}

pub(super) fn var_name_list(list: &deep::List) -> Option<&str> {
    if get_tag(list) != Some(DeepTag::Var) {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

pub(super) fn borrow_inner_for_signature(expr: &deep::Expr) -> Option<&deep::Expr> {
    // chelis#1107 amendment: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag != DeepTag::Borrow {
        return None;
    }
    kids.first()
}

pub(super) fn is_direct_unshadowed_var(
    expr: &deep::Expr,
    name: &str,
    bound: &[UnordSet<String>],
) -> bool {
    var_name_expr(expr) == Some(name) && !is_bound_name(name, bound)
}

pub(super) fn is_bound_name(name: &str, bound: &[UnordSet<String>]) -> bool {
    bound.iter().rev().any(|scope| scope.contains(name))
}

/// Check whether a def body carries its own explicit type stamp and is a
/// literal self-reference, as produced by `x = (x : T)`. A declaration-level
/// `x: T = x` is recognized separately by the cycle detector because Surf
/// represents its type as a sibling `defsig`, not as body metadata.
pub(super) fn body_is_type_stamped_literal_self_ref(body: &deep::Expr, name: &str) -> bool {
    if expr_type_expr(body, &IrTypeEnv::new()).is_none() {
        return false;
    }
    body_is_literal_self_ref_shape(body, name)
}

/// Check the literal self-reference shape independently of its explicit type
/// owner. Callers must first prove either a body type stamp or the matching
/// declaration signature; bare `x = x` must never earn this carve-out.
pub(super) fn body_is_literal_self_ref_shape(body: &deep::Expr, name: &str) -> bool {
    let mut current = body;
    loop {
        match current {
            deep::Expr::MetaExpr(meta, _) => current = &meta.expr,
            deep::Expr::Node(node, _) => {
                return node.tag() == DeepTag::Var
                    && node.children_slice().first().and_then(symbol_name) == Some(name);
            }
            deep::Expr::List(list, _) => {
                // Surf `x = (x : T)` desugars to a `var` node carrying `type`
                // metadata, which the arms above and below recognize. A
                // symbol-headed `(ascribe x T)` / `(: x T)` list is a legacy
                // hand-written Deep spelling outside the closed vocabulary;
                // it is still unwrapped here so a `defsig`-backed self
                // reference in that form keeps its external-input reading.
                if matches!(list.unknown_tag_symbol(), Some("ascribe" | ":")) {
                    match children(list).first() {
                        Some(inner) => current = inner,
                        None => return false,
                    }
                    continue;
                }
                match get_tag(list) {
                    Some(DeepTag::Var) => {
                        return children(list).first().and_then(symbol_name) == Some(name);
                    }
                    _ => return false,
                }
            }
            _ => return false,
        }
    }
}

/// Detect cycles among top-level `def` bindings.
///
/// An explicitly typed external-input pattern (`x: T = x` or `x = (x : T)`)
/// is permitted: its literal self-loop declares an input rather than reading
/// an eager value. An untyped `x = x`, or any cycle with an intermediate hop,
/// is a real binding cycle and is reported as a `CycleDetected` error.
pub(super) fn detect_top_level_binding_cycles(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) {
    let items = top_level_decl_items_with_modules(exprs);
    TopLevelReferenceGraph::build(&items).report_eager_cycle_errors(errors);
}

pub(super) fn param_name_for_refs(param: &deep::Expr) -> Option<String> {
    stack_guard!("param_name_for_refs", param, None);
    match param {
        deep::Expr::Atom(deep::Atom::Name(name), _) => Some(name.clone()),
        deep::Expr::MetaExpr(meta, _) => param_name_for_refs(&meta.expr),
        // A Deep param is `(name {type: ...})` — a List with the name as
        // the FIRST element and the meta map as the second. `children()`
        // skips first two (tag + meta) and returns nothing for a 2-elem
        // list, so read elements[0] directly.
        deep::Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .map(str::to_string),
        _ => None,
    }
}

#[cfg(test)]
mod top_level_reference_graph_tests {
    use super::*;

    fn graph(source: &str) -> TopLevelReferenceGraph {
        let declarations = chelis_surf::parser::parse_str(source)
            .unwrap_or_else(|error| panic!("graph fixture must parse: {error:?}\n{source}"));
        let exprs = chelis_surf::desugar::desugar_program(&declarations);
        let items = top_level_decl_items_with_modules(&exprs);
        TopLevelReferenceGraph::build(&items)
    }

    /// [04-INF-7] regression: lambda bodies and applications feed the same
    /// full SCC projection that Slice C consumes.
    #[test]
    fn eager_lambda_cycle_is_one_component() {
        let graph = graph(
            "module LambdaCycle\n\n\
             carried = map(fn (x: int32) -> f(x), [1, 2])\n\n\
             def f(n: int32) -> int32 = add(n, carried)\n",
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
    }
}
