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

struct FunctionDefItem<'a> {
    vertex: usize,
    item_index: usize,
    name: String,
    expr: &'a deep::Expr,
}

impl FunctionInferencePlan {
    /// Build the one canonical function dependency plan for an inference run.
    /// SCCs are constructed in O(vertices + edges), returned callee-first,
    /// and retain source order within each component.
    pub(super) fn build(items: &[(Option<String>, &deep::Expr)]) -> Self {
        profile_plan_build();
        let cancel = crate::cancel::current_cancel_token();
        let cancelled = || cancel.as_ref().is_some_and(CancelToken::is_cancelled);
        let mut vertex_by_name = UnordMap::<String, usize>::new();
        let mut def_items = Vec::new();
        for (item_index, (_, expr)) in items.iter().enumerate() {
            if cancelled() {
                return Self::incomplete();
            }
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
            let next_vertex = vertex_by_name.len();
            let vertex = *vertex_by_name
                .entry(name.to_string())
                .or_insert(next_vertex);
            def_items.push(FunctionDefItem {
                vertex,
                item_index,
                name: name.to_string(),
                expr,
            });
        }

        let def_names = vertex_by_name
            .to_sorted()
            .into_iter()
            .map(|(name, _)| name.clone())
            .collect::<UnordSet<_>>();
        let mut graph = vec![Vec::<usize>::new(); vertex_by_name.len()];
        for item in &def_items {
            if cancelled() {
                return Self::incomplete();
            }
            let Some((DeepTag::Def, _, kids)) = stamped_parts(item.expr) else {
                continue;
            };
            let Some(fn_kids) = kids
                .get(1)
                .and_then(|body| tagged_children(body, DeepTag::Fn))
            else {
                continue;
            };
            let (Some(params), Some(body)) = (fn_kids.first(), fn_kids.get(1)) else {
                continue;
            };
            let mut bound = vec![
                param_source_infos(params)
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect(),
            ];
            let mut calls = UnordSet::new();
            collect_top_level_calls(body, &def_names, &mut bound, &mut calls);
            let mut callees = calls
                .into_sorted()
                .into_iter()
                .filter_map(|name| vertex_by_name.get(&name).copied())
                .collect::<Vec<_>>();
            callees.sort_unstable();
            callees.dedup();
            // Preserve the previous duplicate-declaration behavior: the last
            // well-formed body for one name supplies that name's adjacency.
            graph[item.vertex] = callees;
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
        for item in def_items {
            components[component_by_vertex[item.vertex]]
                .members
                .push(FunctionInferenceMember {
                    item_index: item.item_index,
                    name: item.name,
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
        profile_scc_vertex_entry();
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
                profile_scc_edge_inspection();
                if indices[callee].is_none() {
                    indices[callee] = Some(next_index);
                    lowlinks[callee] = next_index;
                    next_index += 1;
                    tarjan_stack.push(callee);
                    on_stack[callee] = true;
                    profile_scc_vertex_entry();
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
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("borrow requires tensor or tensor-carrying input, got {peeled}"),
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

/// Detect cycles among top-level `def` bindings.
///
/// The Nautilus external-input pattern `x = (x : tensor[...])` is permitted —
/// a self-loop (the def body references the same name it is binding, with no
/// intermediate hops) is treated as a declaration of an external input, not as
/// a cycle. Any cycle of length >= 2 (e.g. `a -> b -> a`, `a -> b -> c -> a`)
/// is a real binding cycle and is reported as a `CycleDetected` error.
pub(super) fn detect_top_level_binding_cycles(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) {
    let mut def_names: Vec<String> = Vec::new();
    let mut def_name_set: UnordSet<String> = UnordSet::new();
    let mut def_bodies: UnordMap<String, &deep::Expr> = UnordMap::new();
    // Descend through `(module {} name ...)` wrappers so this check works
    // on idiomatic Surf sources (every `.ch` file starts with `module X`,
    // which desugars to a single top-level `module` list wrapping every
    // declaration). Without this, the cycle check is a no-op in practice.
    for expr in top_level_decl_items(exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some(DeepTag::Def)
        {
            let kids = children(list);
            let Some(name) = kids.first().and_then(symbol_name) else {
                continue;
            };
            let Some(body) = kids.get(1) else { continue };
            if !def_name_set.contains(name) {
                def_name_set.insert(name.to_string());
                def_names.push(name.to_string());
            }
            def_bodies.insert(name.to_string(), body);
        }
    }

    // Phase 1: per-def, collect both the vars referenced EAGERLY (outside fn
    // bodies) and the top-level fns APPLIED eagerly. Lazy refs inside fn
    // bodies are captured separately so we can chain them in on demand.
    let mut direct_refs: UnordMap<String, UnordSet<String>> = UnordMap::new();
    let mut applied_fns: UnordMap<String, UnordSet<String>> = UnordMap::new();
    let mut fn_body_refs: UnordMap<String, (UnordSet<String>, UnordSet<String>)> = UnordMap::new();
    for name in &def_names {
        let Some(body) = def_bodies.get(name) else {
            continue;
        };
        let mut refs: UnordSet<String> = UnordSet::new();
        let mut applied: UnordSet<String> = UnordSet::new();
        let mut bound: UnordSet<String> = UnordSet::new();
        collect_eager_refs(body, &mut bound, &mut refs, &mut applied);
        direct_refs.insert(name.clone(), refs);
        applied_fns.insert(name.clone(), applied);
        // If this def's body IS itself a fn, also collect what its body
        // references so callers of this def can chain.
        if let deep::Expr::List(list, _) = body
            && get_tag(list) == Some(DeepTag::Fn)
            && let Some(fn_body) = children(list).get(1)
        {
            let mut inner_refs: UnordSet<String> = UnordSet::new();
            let mut inner_applied: UnordSet<String> = UnordSet::new();
            let mut inner_bound: UnordSet<String> = UnordSet::new();
            // Bind the fn's own params so they aren't flagged as refs.
            if let Some(params_list) = children(list).first()
                && let deep::Expr::List(params, _) = params_list
                && get_tag(params) == Some(DeepTag::Params)
            {
                for param in children(params) {
                    if let Some(pname) = param_name_for_refs(param) {
                        inner_bound.insert(pname);
                    }
                }
            }
            collect_eager_refs(
                fn_body,
                &mut inner_bound,
                &mut inner_refs,
                &mut inner_applied,
            );
            fn_body_refs.insert(name.clone(), (inner_refs, inner_applied));
        }
    }

    // Phase 2: build two edge sets per def.
    //   - value_edges[d]: names read AS VALUES in d's body (i.e. `(var x)`
    //     where x is not a fn callee). Reading a value requires that value
    //     to already be bound — a cycle here is a real binding cycle.
    //   - call_edges[d]: names d CALLS (`(app (var f) ...)`). Calling a fn
    //     pushes its body into eager evaluation but does NOT require f's
    //     value — f is a callable, not a scalar. Recursive calls with base
    //     cases terminate and don't close a cycle.
    //
    // Cycle condition: DFS from each VALUE def X, traversing both edge
    // kinds transitively. Track the stack of VALUE defs we're currently
    // evaluating. If a value-edge lands on a stack member, that's a real
    // binding cycle. Fn names aren't pushed onto the stack — they are
    // intermediates in the path.
    let mut value_edges: UnordMap<String, Vec<String>> = UnordMap::new();
    let mut call_edges: UnordMap<String, Vec<String>> = UnordMap::new();
    for name in &def_names {
        let body = def_bodies.get(name).copied();
        let is_nautilus_literal_self = body.is_some_and(|b| body_is_literal_self_ref(b, name));
        let body_is_fn = matches!(
            body,
            Some(deep::Expr::List(list, _)) if get_tag(list) == Some(DeepTag::Fn)
        );

        let (raw_refs, raw_applied) = if body_is_fn {
            let empty_refs: UnordSet<String> = UnordSet::new();
            let empty_applied: UnordSet<String> = UnordSet::new();
            fn_body_refs
                .get(name)
                .map(|(r, a)| (r.clone(), a.clone()))
                .unwrap_or((empty_refs, empty_applied))
        } else {
            (
                direct_refs.get(name).cloned().unwrap_or_default(),
                applied_fns.get(name).cloned().unwrap_or_default(),
            )
        };

        let mut value_out: Vec<String> = raw_refs
            .into_sorted()
            .into_iter()
            .filter(|r| {
                if r == name && is_nautilus_literal_self {
                    return false;
                }
                def_name_set.contains(r)
            })
            .collect();
        let mut call_out: Vec<String> = raw_applied
            .into_sorted()
            .into_iter()
            .filter(|r| def_name_set.contains(r))
            .collect();
        value_out.sort();
        call_out.sort();
        value_edges.insert(name.clone(), value_out);
        call_edges.insert(name.clone(), call_out);
    }

    let mut reported: UnordSet<Vec<String>> = UnordSet::new();

    // DFS from each value def. Track:
    //   - `value_stack`: the value defs we're "currently evaluating". A
    //     value_edge landing on a member of this stack is a cycle.
    //   - `visited`: nodes we've already fully explored from some starting
    //     value def. Avoids re-walking fn bodies we've cleared.
    //   - `path`: the traversal path for error reporting (includes both
    //     values and fns as intermediates).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        node: &str,
        is_value: &dyn Fn(&str) -> bool,
        value_edges: &UnordMap<String, Vec<String>>,
        call_edges: &UnordMap<String, Vec<String>>,
        color: &mut UnordMap<String, Color>,
        value_stack: &mut Vec<String>,
        path: &mut Vec<String>,
        reported: &mut UnordSet<Vec<String>>,
        errors: &mut DiagnosticSink<'_>,
    ) {
        color.insert(node.to_string(), Color::Gray);
        let this_is_value = is_value(node);
        if this_is_value {
            value_stack.push(node.to_string());
        }
        path.push(node.to_string());

        if let Some(neighbors) = value_edges.get(node) {
            for next in neighbors {
                if let Some(start) = value_stack.iter().position(|s| s == next) {
                    // Value-edge landing on a value currently being
                    // evaluated → real binding cycle.
                    let value_seg_start = path.iter().position(|s| s == &value_stack[start]);
                    let cycle: Vec<String> = if let Some(s) = value_seg_start {
                        path[s..].to_vec()
                    } else {
                        value_stack[start..].to_vec()
                    };
                    let mut canon = cycle.clone();
                    if let Some((min_idx, _)) = canon.iter().enumerate().min_by(|a, b| a.1.cmp(b.1))
                    {
                        canon.rotate_left(min_idx);
                    }
                    if reported.insert(canon.clone()) {
                        let mut pathstr = canon.clone();
                        pathstr.push(canon[0].clone());
                        let message = format!("binding cycle: {}", pathstr.join(" -> "));
                        errors.push(CheckError::new(
                            CheckErrorKind::CycleDetected,
                            message,
                            vec![
                                "Break the cycle by removing one of the \
                                 self-referential definitions or replacing it \
                                 with a concrete value."
                                    .to_string(),
                            ],
                        ));
                    }
                } else if color.get(next).copied().unwrap_or(Color::White) == Color::White {
                    dfs(
                        next,
                        is_value,
                        value_edges,
                        call_edges,
                        color,
                        value_stack,
                        path,
                        reported,
                        errors,
                    );
                }
            }
        }

        if let Some(neighbors) = call_edges.get(node) {
            for next in neighbors {
                // Fn calls don't require the callee's VALUE — they just
                // push the callee's body into eager evaluation. Gray nodes
                // are mid-exploration (recursive reentry) — skip to avoid
                // infinite DFS.
                if color.get(next).copied().unwrap_or(Color::White) == Color::White {
                    dfs(
                        next,
                        is_value,
                        value_edges,
                        call_edges,
                        color,
                        value_stack,
                        path,
                        reported,
                        errors,
                    );
                }
            }
        }

        path.pop();
        if this_is_value {
            value_stack.pop();
        }
        color.insert(node.to_string(), Color::Black);
    }

    let is_value = |name: &str| -> bool {
        def_bodies
            .get(name)
            .map(|body| !matches!(body, deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Fn)))
            .unwrap_or(false)
    };
    let mut color: UnordMap<String, Color> = def_names
        .iter()
        .map(|n| (n.clone(), Color::White))
        .collect();
    let mut value_stack: Vec<String> = Vec::new();
    let mut path: Vec<String> = Vec::new();
    for name in &def_names {
        if !is_value(name) {
            continue;
        }
        if color.get(name).copied() == Some(Color::White) {
            dfs(
                name,
                &is_value,
                &value_edges,
                &call_edges,
                &mut color,
                &mut value_stack,
                &mut path,
                &mut reported,
                errors,
            );
        }
    }
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

/// Collect both eager references and top-level fn applications.
///
/// `refs` gets free `(var name)` references that fire at definition time
/// (i.e., NOT inside an enclosing `fn` body).
///
/// `applied` gets names of fns called as `(app (var F) ...)` at definition
/// time (again, not inside a nested fn body). Callers use `applied` to
/// chain in the called fn's own eager refs for cycle detection — this is
/// what catches top-level value cycles that route through a fn call:
///
/// ```text
/// a = f()
/// b = g()
/// def f() = b
/// def g() = a
/// ```
///
/// Plain `collect_free_var_refs` (kept below for backward compatibility)
/// ignores fn bodies entirely, which correctly permits mutual recursion
/// between fn defs never called eagerly — but misses the cycle above.
pub(super) fn collect_eager_refs(
    expr: &deep::Expr,
    bound: &mut UnordSet<String>,
    refs: &mut UnordSet<String>,
    applied: &mut UnordSet<String>,
) {
    stack_guard!("collect_eager_refs", expr);
    match expr {
        deep::Expr::List(list, _) => match get_tag(list) {
            Some(DeepTag::Var) => {
                if let Some(name) = children(list).first().and_then(symbol_name)
                    && !bound.contains(name)
                {
                    refs.insert(name.to_string());
                }
            }
            Some(DeepTag::Fn) => {
                // Skip fn body — only its application at this site (if any)
                // is eager; the body itself is deferred.
            }
            Some(DeepTag::App) => {
                let kids = children(list);
                if let Some(callee) = kids.first()
                    && let deep::Expr::List(clist, _) = callee
                    && get_tag(clist) == Some(DeepTag::Var)
                    && let Some(fname) = children(clist).first().and_then(symbol_name)
                    && !bound.contains(fname)
                {
                    // Callee is in `applied` only — NOT in `refs`. For cycle
                    // detection, reading `g` as a value is different from
                    // calling `g()`: the former requires g's value now, the
                    // latter just pushes g's body into eager evaluation and
                    // may terminate at a base case.
                    applied.insert(fname.to_string());
                } else if let Some(callee) = kids.first() {
                    collect_eager_refs(callee, bound, refs, applied);
                }
                for arg in kids.iter().skip(1) {
                    collect_eager_refs(arg, bound, refs, applied);
                }
            }
            Some(DeepTag::Let) => {
                let kids = children(list);
                let mut added: Vec<String> = Vec::new();
                if let Some(deep::Expr::List(bind_list, _)) = kids.first()
                    && get_tag(bind_list) == Some(DeepTag::Bind)
                {
                    let bind_kids = children(bind_list);
                    let mut i = 0;
                    while i + 1 < bind_kids.len() {
                        collect_eager_refs(&bind_kids[i + 1], bound, refs, applied);
                        if let Some(name) = symbol_name(&bind_kids[i])
                            && bound.insert(name.to_string())
                        {
                            added.push(name.to_string());
                        }
                        i += 2;
                    }
                }
                if let Some(body) = kids.get(1) {
                    collect_eager_refs(body, bound, refs, applied);
                }
                for name in added {
                    bound.remove(&name);
                }
            }
            Some(DeepTag::Match) => {
                let kids = children(list);
                // The scrutinee is evaluated in the enclosing scope. Pattern
                // binders exist only inside their own arm's guard and body.
                if let Some(scrutinee) = kids.first() {
                    collect_eager_refs(scrutinee, bound, refs, applied);
                }
                for arm in kids.iter().skip(1) {
                    let Some((DeepTag::Arm, _, arm_kids)) = stamped_parts(arm) else {
                        collect_eager_refs(arm, bound, refs, applied);
                        continue;
                    };
                    let mut added = Vec::new();
                    if let Some(pattern) = arm_kids.first() {
                        for name in chelis_deep::pattern_binder_names(pattern) {
                            if bound.insert(name.clone()) {
                                added.push(name);
                            }
                        }
                    }
                    for scoped in arm_kids.iter().skip(1) {
                        collect_eager_refs(scoped, bound, refs, applied);
                    }
                    for name in added {
                        bound.remove(&name);
                    }
                }
            }
            _ => {
                for elem in &list.elements {
                    collect_eager_refs(elem, bound, refs, applied);
                }
            }
        },
        deep::Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                collect_eager_refs(v, bound, refs, applied);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, v) in &meta.entries {
                collect_eager_refs(v, bound, refs, applied);
            }
            collect_eager_refs(&meta.expr, bound, refs, applied);
        }
        deep::Expr::Atom(_, _) => {}
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            collect_eager_refs(&bridged, bound, refs, applied);
        }
        deep::Expr::BareList(elems, _) => {
            for child in elems {
                collect_eager_refs(child, bound, refs, applied);
            }
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                collect_eager_refs(child, bound, refs, applied);
            }
        }
    }
}
