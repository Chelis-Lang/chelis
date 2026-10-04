//! The primary inference schedule.
//!
//! One responsibility: decide the order and grouping in which top-level
//! declarations are inferred, and prebind a cyclic component's members
//! before their bodies are inferred. The drivers in `program.rs` consume the
//! groups and run them.

use super::*;

/// The hoist order: the total order the body-inference schedule used before
/// chelis#1134 and still uses as its priority. Every module function is
/// spliced at the earliest module-function ordinal in the planner's
/// callee-first order, and every other item keeps textual order.
///
/// This order is a linear extension of every precedence edge
/// [`primary_inference_schedule`] builds except a barrier into a hoisted
/// function (an eager value declared before a module function that reads it),
/// which is exactly the [04-INF-4] defect the schedule exists to repair. The
/// schedule therefore reproduces this order on every program that carries no
/// such barrier, up to the contiguity of a contracted recursive component, so
/// the grouped order [`primary_inference_groups`] infers is identical.
fn hoist_order(
    function_plan: &FunctionInferencePlan,
    item_count: usize,
    module_fn_indices: &BTreeSet<usize>,
) -> Vec<usize> {
    let Some(&insertion) = module_fn_indices.first() else {
        return (0..item_count).collect();
    };
    let ordered_module_fns = function_plan
        .ordered_members()
        .map(|member| member.item_index)
        .filter(|index| module_fn_indices.contains(index))
        .collect::<Vec<_>>();
    let mut order = Vec::with_capacity(item_count);
    for index in 0..item_count {
        if index == insertion {
            order.extend(ordered_module_fns.iter().copied());
        }
        if !module_fn_indices.contains(&index) {
            order.push(index);
        }
    }
    order
}

/// Eager (non-function) top-level value `def`s, as name -> flattened ordinal.
/// The first `def` of a duplicated name owns the position, matching
/// `Env::note_top_level_value_ordinal`; the duplicate is already an error.
fn eager_value_definition_ordinals(
    items: &[(Option<String>, &deep::Expr)],
) -> BTreeMap<String, usize> {
    let mut ordinals = BTreeMap::new();
    for (index, (_, expr)) in items.iter().enumerate() {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if definition_owns_function_metadata_prebind(expr) {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        ordinals.entry(name.to_string()).or_insert(index);
    }
    ordinals
}

/// Does this Deep TYPE expression contain an inference hole?
///
/// chelis#1486 / [04-INF-5]: a wildcard slot is `(t-var {} _)` in a type
/// position, `(d-var {} _)` in a dimension position, and `(d-rank {} _)` in a
/// rank position. `DeepTypeResolver::resolve_type_var`, `resolve_dim_var`, and
/// `resolve_rank_var` are the three places that mint a fresh variable for the
/// name `_`, so these are exactly the spellings that can produce a hole.
///
/// The scan is syntactic on purpose: the schedule runs before any signature is
/// resolved, so it cannot ask the resolver. A bail on a nearly-exhausted stack
/// answers `true`, which can only ADD a precedence edge and never drop one;
/// the guard's record still turns the bail into a hard failure at the check
/// boundary, so the over-approximation is never observed on a passing program.
fn deep_type_contains_hole(expr: &deep::Expr) -> bool {
    stack_guard!("deep_type_contains_hole", expr, true);
    if let Some((tag, _, kids)) = stamped_parts(expr) {
        if matches!(tag, DeepTag::TVar | DeepTag::DVar | DeepTag::DRank)
            && kids.first().and_then(symbol_name) == Some("_")
        {
            return true;
        }
        return kids.iter().any(deep_type_contains_hole);
    }
    match expr {
        deep::Expr::BareList(elements, _) => elements.iter().any(deep_type_contains_hole),
        deep::Expr::MetaExpr(meta, _) => deep_type_contains_hole(&meta.expr),
        _ => false,
    }
}

/// The declared-signature facts the schedule needs about each name, from a
/// syntactic scan of the unit's `defsig` items (chelis#1486).
///
/// `signed` is every name that carries a `defsig` at all; `holed` is the
/// subset whose signature contains an inference hole. The two answer the two
/// halves of [04-INF-5] and [04-INF-6]: a `defsig`-less function has no header
/// for a reader to use at all, and a hole header is not honest until the body
/// has filled it, while a complete or authored-binder header is honest before
/// the body and needs no edge.
struct DeclaredSignatureScan {
    signed: UnordSet<String>,
    holed: UnordSet<String>,
}

fn scan_declared_signatures(items: &[(Option<String>, &deep::Expr)]) -> DeclaredSignatureScan {
    let mut signed = UnordSet::new();
    let mut holed = UnordSet::new();
    for (_, expr) in items {
        let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        signed.insert(name.to_string());
        if defsig_parts(kids).is_some_and(|(_, _, type_expr)| deep_type_contains_hole(type_expr)) {
            holed.insert(name.to_string());
        }
    }
    DeclaredSignatureScan { signed, holed }
}

/// Primary body-inference schedule: the order in which top-level declaration
/// bodies are inferred. The returned values are original flattened ordinals,
/// so scheduling never changes diagnostic ownership, collected-type origins,
/// or output order.
///
/// The schedule is the [`hoist_order`]-least linear extension of a
/// precedence graph whose every edge is a real reference in the program:
///
/// - a module function is inferred after every module function it calls,
///   so a forward helper's body-derived scheme is available to its caller
///   (the planner's callee-first order);
/// - an item is inferred after every eager value it reads that is declared
///   before it ([04-INF-4] makes exactly those reads legal; an unannotated
///   value has no header anywhere, so its type exists only once its own
///   `def` has been inferred). For a module function this is the barrier
///   that keeps the hoist from carrying it across the value;
/// - an item at or after the earliest module-function ordinal is inferred
///   after every `defsig`-LESS module function it reads (the mirror edge). A
///   declared signature is in the global header environment from the first
///   pass, and under [04-INF-6] an authored binder is rigid, so a complete or
///   authored-binder header IS the function's scheme before its body runs and
///   a reader needs no edge. A `defsig`-less function has no header at all,
///   which is the availability role this edge keeps; it mirrors what the hoist
///   supplied implicitly, bounded to the same region, so function visibility
///   ([04-INF-2]/[04-INF-3]) is neither narrowed nor widened;
/// - an item is inferred after every declaration it references whose `defsig`
///   contains a wildcard slot (the hole edge, chelis#1486 / [04-INF-5]). A
///   hole is not a binder and is not quantified as one: the slot's type is
///   whatever the body determines, so a reader that instantiates the header
///   early observes a variable the body has not filled and accepts programs
///   the checker rejects when the body is inferred first. Unlike the mirror
///   edge this one is bounded by no region: it holds below the hoist floor and
///   in a bare unit, because the dishonest header is global from the first
///   pass. Kahn's priority keeps the displacement minimal, so the function
///   holds its hoist position and only its readers slide after it;
/// - every strongly connected component of the full syntactic reference graph
///   is one vertex. This includes mixed function/value components as well as
///   ordinary recursive function groups, so an edge one member earns
///   constrains them all.
///
/// Bare functions keep textual availability (no call or mirror edge), and a
/// forward value reference produces no edge because the scope rule leaves it
/// unbound. Nothing else orders the graph: in particular there is no textual
/// chain over non-function items and no chain over the planner order. Such
/// chains are not dependencies, and they close cycles on legal programs.
/// The canonical reference collector follows lambda bodies and both read and
/// application edges. Contracting all of its SCCs therefore makes the
/// remaining precedence graph acyclic by construction; the old
/// hoist-order-least stall release is unnecessary and deliberately absent.
/// A cyclic component containing an eager value is still rejected as
/// `CycleDetected`, but its bodies co-infer under provisional bindings so the
/// checker reaches that one ingress-independent verdict without first leaking
/// an inference-order `UnboundVariable` (chelis#1485).
///
/// This is availability, not ordinary source visibility. Whether a name is in
/// scope is decided by `Env::top_level_value_visibility` from source position.
/// The sole extra capability is owned by the exact active cyclic component:
/// its provisional members see one another, and [04-INF-8] precedence targets
/// that the graph scheduled first remain visible, only while that rejected
/// component is co-inferred. The component scope restores the prior capability
/// on every exit. No schedule position can otherwise widen or narrow
/// [04-INF-4].
/// `crates/chelis-types/src/infer/tests/schedule_invariants.rs` asserts the
/// invariants above directly on the returned order.
#[cfg(test)]
pub(super) fn primary_inference_schedule(
    function_plan: &FunctionInferencePlan,
    items: &[(Option<String>, &deep::Expr)],
) -> Vec<usize> {
    let references = TopLevelReferenceGraph::build(items);
    primary_inference_schedule_with_reference_graph(function_plan, items, &references)
}

fn primary_inference_schedule_with_reference_graph(
    function_plan: &FunctionInferencePlan,
    items: &[(Option<String>, &deep::Expr)],
    references: &TopLevelReferenceGraph,
) -> Vec<usize> {
    let reference_components = references.inference_components();
    if !function_plan.complete || !reference_components.complete {
        return Vec::new();
    }
    let module_fn_indices = function_plan
        .ordered_members()
        .filter(|member| items[member.item_index].0.is_some())
        .map(|member| member.item_index)
        .collect::<BTreeSet<_>>();
    let mut key = vec![0usize; items.len()];
    for (position, index) in hoist_order(function_plan, items.len(), &module_fn_indices)
        .into_iter()
        .enumerate()
    {
        key[index] = position;
    }

    // Contract every SCC of the full reference graph to one vertex, named by
    // its lowest member ordinal and carrying its lowest hoist key. This
    // includes mixed function/value components, not only recursive functions.
    let mut vertex_of = (0..items.len()).collect::<Vec<_>>();
    let mut component_members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for component in &reference_components.components {
        let members = component.members.clone();
        let Some(&representative) = members.iter().min() else {
            continue;
        };
        for &member in &members {
            vertex_of[member] = representative;
        }
        component_members.insert(representative, members);
    }
    let vertex_key = |vertex: usize| -> usize {
        component_members
            .get(&vertex)
            .map_or(key[vertex], |members| {
                members
                    .iter()
                    .map(|member| key[*member])
                    .min()
                    .unwrap_or(key[vertex])
            })
    };

    // Reference edges, `(before, after)`, deduplicated and self-edge free.
    let eager_value_ordinals = eager_value_definition_ordinals(items);
    let mut module_fn_by_name: BTreeMap<String, usize> = BTreeMap::new();
    for &index in &module_fn_indices {
        if let Some(name) = top_level_decl_name(items[index].1) {
            module_fn_by_name.entry(name.to_string()).or_insert(index);
        }
    }
    // chelis#1486: the hole edge is not bounded by the planner's region, so it
    // needs every `def`'s ordinal, not only a module function's.
    let signatures = scan_declared_signatures(items);
    let mut hole_signature_definitions: BTreeMap<String, usize> = BTreeMap::new();
    for (index, (_, expr)) in items.iter().enumerate() {
        let Some((DeepTag::Def, _, _)) = stamped_parts(expr) else {
            continue;
        };
        if let Some(name) = top_level_decl_name(expr)
            && signatures.holed.contains(name)
        {
            hole_signature_definitions
                .entry(name.to_string())
                .or_insert(index);
        }
    }
    let floor = module_fn_indices.first().copied();
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (index, item_references) in references.item_references.iter().enumerate() {
        let reader = vertex_of[index];
        let reader_is_module_fn = module_fn_indices.contains(&index);
        for reference in item_references {
            let name = &references.definition(reference.target).name;
            if let Some(&value) = eager_value_ordinals.get(name)
                && value < index
            {
                edges.insert((vertex_of[value], reader));
            }
            if let Some(&function) = module_fn_by_name.get(name)
                && (reader_is_module_fn
                    || (floor.is_some_and(|floor| index >= floor)
                        && !signatures.signed.contains(name)))
            {
                edges.insert((vertex_of[function], reader));
            }
            // The hole edge, in every region and in a bare unit too: a
            // header with a wildcard slot is honest only after its body.
            if let Some(&declaration) = hole_signature_definitions.get(name) {
                edges.insert((vertex_of[declaration], reader));
            }
        }
    }

    // [04-INF-8] cycle precedence: an eager root that occurs in its own
    // full-reference component is rejected by CycleDetected alone. Infer any
    // later eager value referenced from that exact component first, so the
    // component's narrow visibility capability can resolve the real scheme
    // instead of leaking an earlier [04-INF-4] UnboundVariable. These edges
    // cannot cycle after SCC contraction: a target with a path back into the
    // component would already be one of its members.
    for component in &reference_components.components {
        if !component.cyclic {
            continue;
        }
        let Some(&representative) = component.members.iter().min() else {
            continue;
        };
        for target in references.cycle_precedence_targets(&component.members) {
            edges.insert((
                vertex_of[references.definition(target).item_index],
                vertex_of[representative],
            ));
        }
    }
    edges.retain(|(before, after)| before != after);

    // Kahn's algorithm, releasing the hoist-order-least ready component. Full
    // reference SCC contraction makes this graph a DAG, so there is no
    // user-program stall and no arbitrary release path.
    let mut successors: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut remaining_predecessors: BTreeMap<usize, usize> = BTreeMap::new();
    for &(before, after) in &edges {
        successors.entry(before).or_default().push(after);
        *remaining_predecessors.entry(after).or_default() += 1;
    }
    let mut pending = (0..items.len())
        .filter(|index| vertex_of[*index] == *index)
        .map(|vertex| (vertex_key(vertex), vertex))
        .collect::<BTreeSet<_>>();
    let mut ready = pending
        .iter()
        .copied()
        .filter(|(_, vertex)| remaining_predecessors.get(vertex).copied().unwrap_or(0) == 0)
        .collect::<BTreeSet<_>>();
    let mut emitted = vec![false; items.len()];
    let mut schedule = Vec::with_capacity(items.len());
    while let Some(&(key, vertex)) = ready.first() {
        ready.remove(&(key, vertex));
        pending.remove(&(key, vertex));
        if emitted[vertex] {
            continue;
        }
        emitted[vertex] = true;
        match component_members.get(&vertex) {
            Some(members) => schedule.extend(members.iter().copied()),
            None => schedule.push(vertex),
        }
        for &successor in successors.get(&vertex).into_iter().flatten() {
            if emitted[successor] {
                continue;
            }
            let remaining = remaining_predecessors
                .get_mut(&successor)
                .expect("every successor was counted when its edge was added");
            *remaining -= 1;
            if *remaining == 0 {
                ready.insert((vertex_key(successor), successor));
            }
        }
    }
    debug_assert!(
        pending.is_empty(),
        "reference-component DAG must schedule every item"
    );
    schedule
}

#[derive(Debug)]
pub(super) struct PrimaryInferenceGroup {
    pub(super) indices: Vec<usize>,
    pub(super) cyclic: bool,
    pub(super) recursive_function_indices: Vec<usize>,
}

pub(super) fn primary_inference_groups_with_reference_graph(
    function_plan: &FunctionInferencePlan,
    items: &[(Option<String>, &deep::Expr)],
    references: &TopLevelReferenceGraph,
) -> Vec<PrimaryInferenceGroup> {
    let schedule =
        primary_inference_schedule_with_reference_graph(function_plan, items, references);
    primary_inference_groups_for_schedule(function_plan, references, schedule)
}

/// Group one explicit schedule: every full-reference component is emitted
/// whole in source order. Function recursive-instantiation membership stays a
/// separate projection and is never inferred from a mixed component's cycle.
pub(super) fn primary_inference_groups_for_schedule(
    function_plan: &FunctionInferencePlan,
    references: &TopLevelReferenceGraph,
    schedule: Vec<usize>,
) -> Vec<PrimaryInferenceGroup> {
    let reference_components = references.inference_components();
    if !reference_components.complete {
        return Vec::new();
    }
    let recursive_function_indices = function_plan
        .components
        .iter()
        .filter(|component| component.recursive)
        .flat_map(|component| component.members.iter().map(|member| member.item_index))
        .collect::<UnordSet<_>>();

    let mut emitted_components = UnordSet::new();
    let mut groups = Vec::new();
    for index in schedule {
        let Some(component_index) = reference_components.component_by_item[index] else {
            groups.push(PrimaryInferenceGroup {
                indices: vec![index],
                cyclic: false,
                recursive_function_indices: Vec::new(),
            });
            continue;
        };
        if emitted_components.insert(component_index) {
            let component = &reference_components.components[component_index];
            groups.push(PrimaryInferenceGroup {
                indices: component.members.clone(),
                cyclic: component.cyclic,
                recursive_function_indices: component
                    .members
                    .iter()
                    .copied()
                    .filter(|index| recursive_function_indices.contains(index))
                    .collect(),
            });
        }
    }
    groups
}

/// A cyclic component's members decide their obligations together, when the
/// component completes (chelis#2584); every other declaration at its own close.
pub(super) fn close_scope(cyclic: bool) -> CloseScope {
    if cyclic {
        CloseScope::ComponentMember
    } else {
        CloseScope::Declaration
    }
}

/// Install monomorphic types for the un-signed members of one cyclic
/// full-reference component. Functions receive arity-shaped function types;
/// eager values receive one fresh type variable. The component is removed and
/// generalized as a unit after every body has unified with its provisional.
///
/// A member's body and its own references are typed at its provisional type.
/// A sibling's reference is typed at a fresh copy of it, which the component's
/// completion unifies with it (`group_link::sibling_instance`): no reference
/// observes a member's type before the group has determined it ([04-INF-5]),
/// so nothing a body infers depends on which sibling was inferred first.
pub(super) fn prebind_cyclic_component_schemes(
    indices: &[usize],
    items: &[(Option<String>, &deep::Expr)],
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    metadata_prebound_names: &UnordSet<String>,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &Subst,
) -> UnordMap<usize, Type> {
    let mut provisional = UnordMap::new();
    env.begin_group_level(subst.current_level());
    for index in indices {
        let expr = items[*index].1;
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(body)) = (kids.first().and_then(symbol_name), kids.get(1)) else {
            continue;
        };
        if metadata_prebound_names.contains(name) {
            continue;
        }
        // A declared header is the member's scheme already, unless it omits a
        // type: then the group types the member at its provisional type
        // ([04-INF-5], [04-INF-2], chelis#2590).
        if declared_signatures.contains_key(name) {
            env.bind_holed_group_member(name, vg, subst);
            continue;
        }
        let ty = match tagged_children(body, DeepTag::Fn) {
            Some(fn_kids) => {
                let Some(params) = fn_kids.first() else {
                    continue;
                };
                let arity = match params {
                    deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => {
                        node.child_count()
                    }
                    deep::Expr::BareList(elements, _) => elements.len(),
                    _ => continue,
                };
                Type::Fn(
                    (0..arity).map(|_| vg.fresh_type()).collect(),
                    Box::new(vg.fresh_type()),
                )
            }
            None => vg.fresh_type(),
        };
        env.bind(name.to_string(), Scheme::mono(ty.clone()));
        provisional.insert(*index, ty);
    }
    provisional
}
