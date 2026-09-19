//! chelis#1134 / [04-INF-4]: structural oracle for `primary_inference_schedule`.
//!
//! The ordering matrices in `tests/issue_1134_forward_reference_parity.rs`
//! enumerate accept/reject verdicts over declaration layouts. Four repairs of
//! the schedule each shipped a green verdict suite and were found by a
//! reviewer constructing a layout the suite could not express, because the
//! defects lived in the emitted ORDER, which no verdict can see. This oracle
//! asserts the schedule's invariants directly on the returned `Vec<usize>`:
//!
//! - I1: a permutation of the item ordinals, with every full-reference
//!   component's members contiguous;
//! - I2: when the reference graph is acyclic, every reference edge is
//!   respected (referenced before referencer);
//! - I3: when the hoist order already respects every edge, the schedule IS
//!   the hoist order once components are collapsed, so every program without
//!   a barrier keeps the exact order it had before chelis#1134;
//! - I4: contracting every full-reference component makes the precedence
//!   graph acyclic, so there is no scheduler stall-release path.
//!
//! The reference graph is derived from the generator's own declaration of
//! what each named item references and which recursive component it belongs
//! to, never from the collectors the schedule uses. Item ordinals come from
//! the real flattening, so a synthesized `defsig` cannot desynchronize them.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Function,
    Value,
}

/// What header the generator says a declaration publishes, which is what
/// decides whether a reader needs a precedence edge to it (chelis#1486).
///
/// Declared, not measured, so a desugarer change that stopped synthesizing a
/// `defsig` would show up as a disagreement rather than silently changing the
/// model the invariants are checked against; `reference` asserts each value
/// against the real flattened `defsig` items.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Header {
    /// No `defsig` anywhere: `def helper(x) = x`. Nothing is available to a
    /// reader until the body is inferred, so the mirror edge applies.
    Absent,
    /// Every slot annotated: `def anchor() -> i32 = 1`. Honest from the
    /// first pass under [04-INF-6], so no edge is owed.
    Complete,
    /// At least one wildcard slot: `def f(n: i32) = add(v, n)`. Not honest
    /// until the body fills it ([04-INF-5]), so the hole edge applies.
    Holed,
}

/// One named declaration as the generator wrote it.
#[derive(Clone, Debug)]
struct Declaration {
    name: &'static str,
    kind: Kind,
    header: Header,
    source: String,
    references: Vec<&'static str>,
    component: Option<&'static str>,
}

impl Declaration {
    fn with_header(self, header: Header) -> Self {
        Declaration { header, ..self }
    }
}

fn function(name: &'static str, source: &str, references: &[&'static str]) -> Declaration {
    Declaration {
        name,
        kind: Kind::Function,
        header: Header::Absent,
        source: source.to_string(),
        references: references.to_vec(),
        component: None,
    }
}

/// A function whose signature is complete: no wildcard slot, so no hole edge.
fn signed_function(name: &'static str, source: &str, references: &[&'static str]) -> Declaration {
    function(name, source, references).with_header(Header::Complete)
}

/// A function whose synthesized signature carries a wildcard slot.
fn holed_function(name: &'static str, source: &str, references: &[&'static str]) -> Declaration {
    function(name, source, references).with_header(Header::Holed)
}

fn member(
    name: &'static str,
    source: &str,
    references: &[&'static str],
    component: &'static str,
) -> Declaration {
    Declaration {
        component: Some(component),
        ..function(name, source, references)
    }
}

/// A recursive-component member with a complete signature.
fn signed_member(
    name: &'static str,
    source: &str,
    references: &[&'static str],
    component: &'static str,
) -> Declaration {
    member(name, source, references, component).with_header(Header::Complete)
}

fn value(name: &'static str, source: &str, references: &[&'static str]) -> Declaration {
    Declaration {
        name,
        kind: Kind::Value,
        header: Header::Absent,
        source: source.to_string(),
        references: references.to_vec(),
        component: None,
    }
}

/// A value with a declaration type (`carried: i32 = 7`), which desugars to a
/// complete `defsig` beside the `def`.
fn signed_value(name: &'static str, source: &str, references: &[&'static str]) -> Declaration {
    value(name, source, references).with_header(Header::Complete)
}

#[derive(Clone, Debug)]
struct Program {
    wrapped: bool,
    declarations: Vec<Declaration>,
}

impl Program {
    fn source(&self) -> String {
        let mut source = String::new();
        if self.wrapped {
            source.push_str("module ScheduleOracle\n\n");
        }
        for declaration in &self.declarations {
            source.push_str(&declaration.source);
            source.push_str("\n\n");
        }
        source
    }
}

/// One flattened item, matched back to the generator's declaration by name
/// and tag. A `defsig` item references nothing and owns no value ordinal.
#[derive(Clone, Copy, Debug)]
enum FlatItem {
    Signature,
    Declared(usize),
}

struct Measured {
    schedule: Vec<usize>,
    plan: FunctionInferencePlan,
    flat: Vec<FlatItem>,
    module_fn_indices: BTreeSet<usize>,
    /// The header each name actually publishes, read off the real flattened
    /// `defsig` items. Compared against the generator's declared `Header` so a
    /// mis-annotated call site or a desugarer change is a loud disagreement.
    headers: BTreeMap<&'static str, Header>,
}

/// The three wildcard spellings, restated here rather than borrowed from the
/// implementation, so the model this oracle checks against is independent of
/// the scanner it is checking (the same reason `reference` rebuilds the hoist
/// order by hand).
///
/// Iterative rather than recursive, so it needs no `stack_guard!`: a walker
/// with an explicit worklist cannot exhaust the native stack, and a guard in a
/// test model would only give it a way to answer wrongly.
fn deep_type_has_hole(root: &chelis_deep::Expr) -> bool {
    let mut worklist = vec![root];
    while let Some(expr) = worklist.pop() {
        if let Some((tag, _, kids)) = stamped_parts(expr) {
            if matches!(tag, DeepTag::TVar | DeepTag::DVar | DeepTag::DRank)
                && kids.first().and_then(symbol_name) == Some("_")
            {
                return true;
            }
            worklist.extend(kids.iter());
            continue;
        }
        match expr {
            chelis_deep::Expr::List(list, _) => worklist.extend(list.elements.iter()),
            chelis_deep::Expr::BareList(elements, _) => worklist.extend(elements.iter()),
            chelis_deep::Expr::MetaExpr(meta, _) => worklist.push(&meta.expr),
            _ => {}
        }
    }
    false
}

fn measure(program: &Program) -> Measured {
    let source = program.source();
    let declarations = chelis_surf::parser::parse_str(&source)
        .unwrap_or_else(|error| panic!("generated Surf must parse: {error:?}\n{source}"));
    let exprs =
        chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar");
    let items = top_level_decl_items_with_modules(&exprs);
    let flat = items
        .iter()
        .map(|(_, expr)| {
            let (tag, _, _) = stamped_parts(expr).expect("generated declarations are tagged");
            if tag == DeepTag::Defsig {
                return FlatItem::Signature;
            }
            let name = top_level_decl_name(expr).expect("generated defs are named");
            let position = program
                .declarations
                .iter()
                .position(|declaration| declaration.name == name)
                .unwrap_or_else(|| {
                    panic!("flattened item `{name}` is not in the generator\n{source}")
                });
            FlatItem::Declared(position)
        })
        .collect::<Vec<_>>();
    let plan = FunctionInferencePlan::build(&items);
    assert!(plan.complete, "plan must complete\n{source}");
    let module_fn_indices = plan
        .ordered_members()
        .filter(|member| items[member.item_index].0.is_some())
        .map(|member| member.item_index)
        .collect::<BTreeSet<_>>();
    let schedule = primary_inference_schedule(&plan, &items);
    let mut headers: BTreeMap<&'static str, Header> = BTreeMap::new();
    for declaration in &program.declarations {
        headers.insert(declaration.name, Header::Absent);
    }
    for (_, expr) in &items {
        let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(declared) = program
            .declarations
            .iter()
            .find(|declaration| declaration.name == name)
        else {
            panic!("a `defsig` for `{name}` has no generator declaration\n{source}");
        };
        let header =
            if defsig_parts(kids).is_some_and(|(_, _, type_expr)| deep_type_has_hole(type_expr)) {
                Header::Holed
            } else {
                Header::Complete
            };
        headers.insert(declared.name, header);
    }
    Measured {
        schedule,
        plan,
        flat,
        module_fn_indices,
        headers,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeKind {
    Read,
    Call,
    Mirror,
    Hole,
    CyclePrecedence,
}

/// The reference graph, derived from the generator's declarations over the
/// real item ordinals. `vertex[i]` is the contracted vertex of item `i`.
struct Reference {
    vertex: Vec<usize>,
    edges: BTreeMap<(usize, usize), EdgeKind>,
    hoist: Vec<usize>,
    cyclic_components: BTreeSet<usize>,
}

fn reference(program: &Program, measured: &Measured) -> Reference {
    let item_count = measured.flat.len();
    let declared = |index: usize| match measured.flat[index] {
        FlatItem::Signature => None,
        FlatItem::Declared(position) => Some(&program.declarations[position]),
    };
    // The generator's kinds must agree with what the planner saw, or the
    // reference graph would be describing a different program.
    for index in 0..item_count {
        let Some(declaration) = declared(index) else {
            continue;
        };
        let planned_as_module_fn = measured.module_fn_indices.contains(&index);
        let expected = program.wrapped && declaration.kind == Kind::Function;
        assert_eq!(
            planned_as_module_fn,
            expected,
            "item {index} `{}`: generator kind and planner disagree\n{}",
            declaration.name,
            program.source()
        );
        // chelis#1486: the header decides which precedence edge a reader is
        // owed, so a wrong declaration would model a different program.
        assert_eq!(
            measured
                .headers
                .get(declaration.name)
                .copied()
                .unwrap_or(Header::Absent),
            declaration.header,
            "item {index} `{}`: generator header and desugared signature disagree\n{}",
            declaration.name,
            program.source()
        );
    }
    let mut value_ordinal: BTreeMap<&str, usize> = BTreeMap::new();
    let mut function_index: BTreeMap<&str, usize> = BTreeMap::new();
    let mut definition_index: BTreeMap<&str, usize> = BTreeMap::new();
    for index in 0..item_count {
        let Some(declaration) = declared(index) else {
            continue;
        };
        definition_index.entry(declaration.name).or_insert(index);
        match declaration.kind {
            Kind::Value => {
                value_ordinal.entry(declaration.name).or_insert(index);
            }
            Kind::Function => {
                function_index.entry(declaration.name).or_insert(index);
            }
        }
    }
    // Independent full-reference SCC model. The implementation uses Tarjan;
    // this small generated oracle uses transitive closure so sharing an SCC
    // bug cannot make both sides green. Non-definition items remain singleton
    // vertices in the scheduling graph.
    let mut reaches = vec![vec![false; item_count]; item_count];
    let mut has_self_edge = vec![false; item_count];
    for index in 0..item_count {
        let Some(declaration) = declared(index) else {
            continue;
        };
        reaches[index][index] = true;
        for name in &declaration.references {
            if let Some(&target) = definition_index.get(name) {
                reaches[index][target] = true;
                has_self_edge[index] |= index == target;
            }
        }
    }
    for intermediate in 0..item_count {
        let through_intermediate = reaches[intermediate].clone();
        for row in &mut reaches {
            if !row[intermediate] {
                continue;
            }
            for (reachable, &through) in row.iter_mut().zip(&through_intermediate) {
                *reachable |= through;
            }
        }
    }
    let mut vertex = (0..item_count).collect::<Vec<_>>();
    for (index, representative) in vertex.iter_mut().enumerate() {
        if declared(index).is_none() {
            continue;
        }
        *representative = (0..item_count)
            .filter(|other| declared(*other).is_some())
            .filter(|other| reaches[index][*other] && reaches[*other][index])
            .min()
            .expect("each definition reaches itself");
    }
    let mut members_by_vertex: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, &representative) in vertex.iter().enumerate() {
        if declared(index).is_some() {
            members_by_vertex
                .entry(representative)
                .or_default()
                .push(index);
        }
    }
    let cyclic_components = members_by_vertex
        .iter()
        .filter(|(_, members)| members.len() > 1 || has_self_edge[members[0]])
        .map(|(representative, _)| *representative)
        .collect::<BTreeSet<_>>();
    // The older generator annotation describes the function-only planner SCC.
    // Every such pair must remain in one full-reference component, though an
    // eager value may now join it.
    let mut representative_by_label = BTreeMap::new();
    for (index, &component) in vertex.iter().enumerate() {
        let Some(label) = declared(index).and_then(|declaration| declaration.component) else {
            continue;
        };
        let prior = representative_by_label.entry(label).or_insert(component);
        assert_eq!(
            *prior,
            component,
            "function component `{label}` was split by the full-reference model\n{}",
            program.source()
        );
    }
    let floor = measured.module_fn_indices.first().copied();
    let header_of = |name: &str| -> Header {
        program
            .declarations
            .iter()
            .find(|declaration| declaration.name == name)
            .map_or(Header::Absent, |declaration| declaration.header)
    };
    let mut edges = BTreeMap::new();
    for index in 0..item_count {
        let Some(declaration) = declared(index) else {
            continue;
        };
        let reader_is_module_fn = program.wrapped && declaration.kind == Kind::Function;
        for name in &declaration.references {
            if let Some(&ordinal) = value_ordinal.get(name)
                && ordinal < index
                && vertex[ordinal] != vertex[index]
            {
                edges.insert((vertex[ordinal], vertex[index]), EdgeKind::Read);
            }
            if let Some(&function) = function_index.get(name)
                && program.wrapped
                && vertex[function] != vertex[index]
            {
                if reader_is_module_fn {
                    edges.insert((vertex[function], vertex[index]), EdgeKind::Call);
                } else if floor.is_some_and(|floor| index >= floor)
                    && header_of(name) == Header::Absent
                {
                    // `defsig`-less only (chelis#1486). A complete or
                    // authored-binder header is honest before its body under
                    // [04-INF-6], so its reader is owed nothing; a hole header
                    // is covered by the hole edge below. Only a function with
                    // no header at all leaves a reader with nothing to use.
                    edges.insert((vertex[function], vertex[index]), EdgeKind::Mirror);
                }
            }
            // The hole edge (chelis#1486 / [04-INF-5]). Bounded by no region:
            // the dishonest header is global from the first pass, so it holds
            // below the floor and in a bare unit, for a function reader and a
            // value reader alike.
            if header_of(name) == Header::Holed
                && let Some(&target) = function_index.get(name).or_else(|| value_ordinal.get(name))
                && vertex[target] != vertex[index]
            {
                edges.insert((vertex[target], vertex[index]), EdgeKind::Hole);
            }
        }
    }
    // [04-INF-8] cycle precedence adds one availability edge for each later
    // eager target read by a cyclic component containing an eager root. This
    // independent model derives the set from the generator's own reference
    // declarations and transitive-closure SCCs, not the implementation graph.
    for &component in &cyclic_components {
        let eager_members = (0..item_count)
            .filter(|index| vertex[*index] == component)
            .filter(|index| {
                declared(*index).is_some_and(|declaration| declaration.kind == Kind::Value)
            })
            .collect::<Vec<_>>();
        let Some(&earliest_eager_root) = eager_members.iter().min() else {
            continue;
        };
        for member in 0..item_count {
            if vertex[member] != component {
                continue;
            }
            let Some(declaration) = declared(member) else {
                continue;
            };
            for name in &declaration.references {
                if let Some(&target) = value_ordinal.get(name)
                    && target > earliest_eager_root
                    && vertex[target] != component
                {
                    edges.insert((vertex[target], component), EdgeKind::CyclePrecedence);
                }
            }
        }
    }
    // The hoist order, written out independently of the implementation:
    // every module function at the earliest module-function ordinal in the
    // planner's callee-first order, everything else textual.
    let hoist = match floor {
        None => (0..item_count).collect(),
        Some(insertion) => {
            let ordered = measured
                .plan
                .ordered_members()
                .map(|member| member.item_index)
                .filter(|index| measured.module_fn_indices.contains(index))
                .collect::<Vec<_>>();
            let mut hoist = Vec::with_capacity(item_count);
            for index in 0..item_count {
                if index == insertion {
                    hoist.extend(ordered.iter().copied());
                }
                if !measured.module_fn_indices.contains(&index) {
                    hoist.push(index);
                }
            }
            hoist
        }
    };
    Reference {
        vertex,
        edges,
        hoist,
        cyclic_components,
    }
}

fn positions(order: &[usize]) -> Vec<usize> {
    let mut position = vec![usize::MAX; order.len()];
    for (slot, index) in order.iter().enumerate() {
        position[*index] = slot;
    }
    position
}

/// Position of a contracted vertex: the slot of its first emitted member.
fn vertex_position(order: &[usize], vertex: &[usize], target: usize) -> usize {
    order
        .iter()
        .position(|index| vertex[*index] == target)
        .expect("every vertex is scheduled")
}

fn respects(order: &[usize], vertex: &[usize], edge: (usize, usize)) -> bool {
    vertex_position(order, vertex, edge.0) < vertex_position(order, vertex, edge.1)
}

fn is_acyclic(reference: &Reference) -> bool {
    let vertices = reference.vertex.iter().copied().collect::<BTreeSet<_>>();
    let mut successors: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(before, after) in reference.edges.keys() {
        successors.entry(before).or_default().push(after);
    }
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Gray,
        Black,
    }
    fn visit(
        node: usize,
        successors: &BTreeMap<usize, Vec<usize>>,
        color: &mut BTreeMap<usize, Color>,
    ) -> bool {
        match color.get(&node).copied().unwrap_or(Color::White) {
            Color::Gray => return false,
            Color::Black => return true,
            Color::White => {}
        }
        color.insert(node, Color::Gray);
        for &next in successors.get(&node).into_iter().flatten() {
            if !visit(next, successors, color) {
                return false;
            }
        }
        color.insert(node, Color::Black);
        true
    }
    let mut color = BTreeMap::new();
    vertices
        .into_iter()
        .all(|vertex| visit(vertex, &successors, &mut color))
}

/// Collapse each contracted component to its first emitted member, so two
/// orders that group the same components at the same slots compare equal.
fn collapsed(order: &[usize], vertex: &[usize]) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    order
        .iter()
        .copied()
        .filter(|index| seen.insert(vertex[*index]))
        .map(|index| vertex[index])
        .collect()
}

/// Check every invariant on one program and return the violations.
fn violations(program: &Program) -> Vec<String> {
    let measured = measure(program);
    let reference = reference(program, &measured);
    let source = program.source();
    let schedule = &measured.schedule;
    let mut failures = Vec::new();
    let item_count = measured.flat.len();

    // I1: permutation, components contiguous.
    let mut sorted = schedule.clone();
    sorted.sort_unstable();
    if sorted != (0..item_count).collect::<Vec<_>>() {
        failures.push(format!("I1: not a permutation: {schedule:?}\n{source}"));
        return failures;
    }
    let position = positions(schedule);
    let mut members_by_vertex: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, &vertex) in reference.vertex.iter().enumerate() {
        members_by_vertex.entry(vertex).or_default().push(index);
    }
    for (vertex, members) in &members_by_vertex {
        let slots = members
            .iter()
            .map(|member| position[*member])
            .collect::<Vec<_>>();
        let contiguous = slots.windows(2).all(|pair| pair[1] == pair[0] + 1);
        if !contiguous {
            failures.push(format!(
                "I1: component at {vertex} is not contiguous (members {members:?} at slots {slots:?}) in {schedule:?}\n{source}"
            ));
        }
    }

    if is_acyclic(&reference) {
        // I2: every edge respected.
        for (&edge, kind) in &reference.edges {
            if !respects(schedule, &reference.vertex, edge) {
                failures.push(format!(
                    "I2: {kind:?} edge {edge:?} violated by {schedule:?}\n{source}"
                ));
            }
        }
        // I3: the hoist order is kept whenever it already satisfies every edge.
        let hoist_respects_all = reference
            .edges
            .keys()
            .all(|&edge| respects(&reference.hoist, &reference.vertex, edge));
        if hoist_respects_all
            && collapsed(schedule, &reference.vertex)
                != collapsed(&reference.hoist, &reference.vertex)
        {
            failures.push(format!(
                "I3: hoist order {:?} satisfies every edge but the schedule is {schedule:?}\n{source}",
                reference.hoist
            ));
        }
    } else {
        failures.push(format!(
            "I4: precedence graph is still cyclic after full-reference contraction: {schedule:?}\n{source}"
        ));
    }
    failures
}

fn assert_invariants(programs: &[Program]) {
    let mut failures = Vec::new();
    for program in programs {
        failures.extend(violations(program));
    }
    assert!(
        failures.is_empty(),
        "{}/{} programs violate the schedule invariants:\n\n{}",
        failures.len(),
        programs.len(),
        failures.join("\n\n")
    );
}

fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for (index, head) in items.iter().enumerate() {
        let mut rest = items.to_vec();
        rest.remove(index);
        for mut tail in permutations(&rest) {
            tail.insert(0, head.clone());
            out.push(tail);
        }
    }
    out
}

fn programs_over(layouts: Vec<Vec<Declaration>>) -> Vec<Program> {
    let mut programs = Vec::new();
    for declarations in layouts {
        for wrapped in [true, false] {
            programs.push(Program {
                wrapped,
                declarations: declarations.clone(),
            });
        }
    }
    programs
}

// ---------------------------------------------------------------------------
// Generated alphabets
// ---------------------------------------------------------------------------

/// Round 8's alphabet: a caller of a `defsig`-less helper, the helper reading
/// an eager value, the value calling a third function, and a spare anchor.
/// This is the family in which the previous cycle break released a function
/// out of planner order (`D4_min.ch`), and in which textual chains closed a
/// cycle no program required (`C2_stall_no_cycle.ch`).
fn round_eight_layouts() -> Vec<Vec<Declaration>> {
    let mut layouts = Vec::new();
    for annotated in [false, true] {
        let carried = if annotated {
            signed_value("carried", "carried: i32 = tailfn(1)", &["tailfn"])
        } else {
            value("carried", "carried = tailfn(1)", &["tailfn"])
        };
        let alphabet = vec![
            function("caller", "def caller(n) = helper(n)", &["helper"]),
            function("helper", "def helper(x) = carried", &["carried"]),
            function("tailfn", "def tailfn(n) = n", &[]),
            carried,
        ];
        for mut layout in permutations(&alphabet) {
            layouts.push(layout.clone());
            layout.insert(0, function("anchor", "def anchor(x) = x", &[]));
            layouts.push(layout);
        }
    }
    layouts
}

#[test]
fn round_eight_alphabet_satisfies_the_schedule_invariants() {
    let programs = programs_over(round_eight_layouts());
    assert_eq!(programs.len(), 192);
    assert_invariants(&programs);
}

/// Rounds 5 and 6: an anchor function, an eager value, and a reader that is
/// either a function or a value, in every ordering, both value spellings.
fn matrix_layouts() -> Vec<Vec<Declaration>> {
    let mut layouts = Vec::new();
    for annotated in [false, true] {
        let carried = if annotated {
            signed_value("carried", "carried: i32 = 7", &[])
        } else {
            value("carried", "carried = 7", &[])
        };
        for reader in [
            signed_function("reader", "def reader() -> i32 = carried", &["carried"]),
            value("echoed", "echoed = carried", &["carried"]),
        ] {
            let alphabet = vec![
                signed_function("anchor", "def anchor() -> i32 = 1", &[]),
                carried.clone(),
                reader.clone(),
            ];
            layouts.extend(permutations(&alphabet));
            layouts.push(vec![carried.clone(), reader.clone()]);
            layouts.push(vec![reader, carried.clone()]);
        }
    }
    layouts
}

#[test]
fn anchor_value_reader_matrix_satisfies_the_schedule_invariants() {
    let programs = programs_over(matrix_layouts());
    assert_eq!(programs.len(), 64);
    assert_invariants(&programs);
}

/// Round 6: a mutually recursive pair straddling the value one member reads,
/// with a stamped and a header-less value spelling. The component must move
/// as one unit.
fn recursive_layouts() -> Vec<Vec<Declaration>> {
    let mut layouts = Vec::new();
    for computed in [false, true] {
        let carried = if computed {
            value("carried", "carried = seed()", &["seed"])
        } else {
            value("carried", "carried = 7", &[])
        };
        let alphabet = vec![
            signed_member(
                "ping",
                "def ping(n: i32) -> i32 = if (n <= 0) then 0 else pong((n - 1))",
                &["pong"],
                "pair",
            ),
            signed_member(
                "pong",
                "def pong(n: i32) -> i32 = if (n <= 0) then carried else ping((n - 1))",
                &["ping", "carried"],
                "pair",
            ),
            carried,
        ];
        for mut layout in permutations(&alphabet) {
            layout.insert(0, signed_function("seed", "def seed() -> i32 = 3", &[]));
            layouts.push(layout);
        }
    }
    layouts
}

#[test]
fn recursive_component_layouts_satisfy_the_schedule_invariants() {
    let programs = programs_over(recursive_layouts());
    assert_eq!(programs.len(), 24);
    assert_invariants(&programs);
}

/// Round 7: a `defsig`-less helper, its caller, and a value reading the
/// caller, in every ordering. The mirror edge is what keeps the caller ahead
/// of the value once the hoist is no longer blunt.
fn function_visibility_layouts() -> Vec<Vec<Declaration>> {
    permutations(&[
        function("helper", "def helper(x) = x", &[]),
        function("caller", "def caller(n) = helper(n)", &["helper"]),
        value("consumer", "consumer = caller(1)", &["caller"]),
    ])
}

#[test]
fn function_visibility_layouts_satisfy_the_schedule_invariants() {
    let programs = programs_over(function_visibility_layouts());
    assert_eq!(programs.len(), 12);
    assert_invariants(&programs);
}

// ---------------------------------------------------------------------------
// Named regressions, each with the order fact the round found missing
// ---------------------------------------------------------------------------

/// Slot of a named declaration's `def` item in the schedule.
fn slot_of(program: &Program, measured: &Measured, name: &str) -> usize {
    let index = measured
        .flat
        .iter()
        .position(|item| {
            matches!(item, FlatItem::Declared(position) if program.declarations[*position].name == name)
        })
        .unwrap_or_else(|| panic!("`{name}` is not a def item"));
    positions(&measured.schedule)[index]
}

fn assert_before(program: &Program, measured: &Measured, before: &str, after: &str) {
    assert!(
        slot_of(program, measured, before) < slot_of(program, measured, after),
        "`{before}` must be inferred before `{after}`, schedule {:?}\n{}",
        measured.schedule,
        program.source()
    );
}

fn named(wrapped: bool, declarations: Vec<Declaration>) -> Program {
    Program {
        wrapped,
        declarations,
    }
}

#[test]
fn round_five_hoisted_reader_follows_the_value_it_reads() {
    // `M_backward_computed.ch`: the reader is hoisted to the anchor's slot on
    // `main`, ahead of the header-less value it legally reads.
    let program = named(
        true,
        vec![
            signed_function("seed", "def seed() -> i32 = 3", &[]),
            signed_function("anchor", "def anchor() -> i32 = 1", &[]),
            value("carried", "carried = seed()", &["seed"]),
            signed_function(
                "later_reader",
                "def later_reader() -> i32 = carried",
                &["carried"],
            ),
        ],
    );
    assert!(violations(&program).is_empty());
    let measured = measure(&program);
    assert_before(&program, &measured, "seed", "carried");
    assert_before(&program, &measured, "carried", "later_reader");
}

#[test]
fn round_six_recursive_component_follows_the_value_a_member_reads() {
    // `L_scc_lit.ch` / `J_scc_computed.ch` wrapped, and `bare_scc.ch` bare:
    // the component is emitted at `ping`, which precedes the value on `main`.
    for wrapped in [true, false] {
        for carried in [
            value("carried", "carried = 7", &[]),
            value("carried", "carried = seed()", &["seed"]),
        ] {
            let program = named(
                wrapped,
                vec![
                    signed_function("seed", "def seed() -> i32 = 3", &[]),
                    signed_member(
                        "ping",
                        "def ping(n: i32) -> i32 = if (n <= 0) then 0 else pong((n - 1))",
                        &["pong"],
                        "pair",
                    ),
                    carried,
                    signed_member(
                        "pong",
                        "def pong(n: i32) -> i32 = if (n <= 0) then carried else ping((n - 1))",
                        &["ping", "carried"],
                        "pair",
                    ),
                ],
            );
            assert!(violations(&program).is_empty(), "{}", program.source());
            let measured = measure(&program);
            assert_before(&program, &measured, "carried", "ping");
            assert_before(&program, &measured, "carried", "pong");
        }
    }
}

#[test]
fn round_seven_value_reading_a_caller_waits_for_the_defsig_less_chain() {
    // `P7_dep_order_no_barrier.ch`: `consumer` reads `caller`, whose scheme
    // exists only after `helper`'s body is inferred.
    let program = named(
        true,
        vec![
            function("caller", "def caller(n) = helper(n)", &["helper"]),
            value("consumer", "consumer = caller(1)", &["caller"]),
            function("helper", "def helper(x) = x", &[]),
        ],
    );
    assert!(violations(&program).is_empty());
    let measured = measure(&program);
    assert_before(&program, &measured, "helper", "caller");
    assert_before(&program, &measured, "caller", "consumer");
}

#[test]
fn round_eight_stall_free_program_keeps_planner_order() {
    // `D4_min.ch`: with chains, the barrier, the mirror and the module-function
    // chain closed a cycle and the break released `caller` before `helper`.
    // With reference edges only there is no cycle at all.
    let program = named(
        true,
        vec![
            function("caller", "def caller(n) = helper(n)", &["helper"]),
            signed_value("carried", "carried: i32 = tailfn(1)", &["tailfn"]),
            function("helper", "def helper(x) = carried", &["carried"]),
            function("tailfn", "def tailfn(n) = n", &[]),
        ],
    );
    let measured = measure(&program);
    let reference = reference(&program, &measured);
    assert!(
        reference.cyclic_components.is_empty(),
        "D4_min has no reference cycle"
    );
    assert!(violations(&program).is_empty());
    assert_before(&program, &measured, "tailfn", "carried");
    assert_before(&program, &measured, "carried", "helper");
    assert_before(&program, &measured, "helper", "caller");
}

#[test]
fn a_value_reading_a_later_function_does_not_stall() {
    // `C2_stall_no_cycle.ch`: the textual chain `carried -> head` plus the
    // mirror `tail -> carried` plus the barrier `carried -> head` stalled a
    // program with no reference cycle. Reference edges alone are acyclic.
    let program = named(
        true,
        vec![
            function("anchor", "def anchor(x) = x", &[]),
            value("carried", "carried = tail(1)", &["tail"]),
            function("head", "def head(n) = carried", &["carried"]),
            function("tail", "def tail(n) = n", &[]),
        ],
    );
    let measured = measure(&program);
    let reference = reference(&program, &measured);
    assert!(
        reference.cyclic_components.is_empty(),
        "C2 has no reference cycle"
    );
    assert!(violations(&program).is_empty());
    assert_before(&program, &measured, "tail", "carried");
    assert_before(&program, &measured, "carried", "head");
}

#[test]
fn a_textual_chain_would_close_a_cycle_this_graph_does_not_have() {
    // Two values in textual order where the first reads a module function
    // that reads the second. A chain `first -> second` would make this
    // cyclic; the real edges (`second -> reads`, `reads -> first`) are not.
    let program = named(
        true,
        vec![
            signed_function("anchor", "def anchor() -> i32 = 1", &[]),
            value("first", "first = reads()", &["reads"]),
            value("second", "second = 7", &[]),
            function("reads", "def reads() = second", &["second"]),
        ],
    );
    let measured = measure(&program);
    let reference = reference(&program, &measured);
    assert!(reference.cyclic_components.is_empty());
    assert!(violations(&program).is_empty());
    assert_before(&program, &measured, "second", "reads");
    assert_before(&program, &measured, "reads", "first");
}

/// chelis#1486 / [04-INF-5]: a reader of a hole-signature function follows it
/// even below the hoist floor and in a bare unit, and a reader of a
/// complete-header function is not moved at all.
///
/// The mirror edge was bounded to the planner's region, so a value declared
/// before the first module function was inferred first and instantiated the
/// quantified hole; the layout, not the spelling, decided the verdict. The
/// hole edge carries no such bound. The control is the half that proves the
/// edge is a hole edge and not a blunt "every reader waits" rule: with a
/// complete header the reader keeps the hoist position it had.
///
/// Regression test for the two hole rows. Before this change `f` did not
/// precede `r` in either layout, because no edge existed to move it; the
/// complete-header row is a disposition lock and was green in both states.
#[test]
fn a_below_floor_reader_of_a_hole_signature_function_follows_it() {
    let holed_wrapped = named(
        true,
        vec![
            value("r", "r = f(2)", &["f"]),
            signed_function("anchor", "def anchor() -> i32 = 1", &[]),
            holed_function("f", "def f(n: i32) = n", &[]),
        ],
    );
    assert!(
        violations(&holed_wrapped).is_empty(),
        "{}",
        holed_wrapped.source()
    );
    let measured = measure(&holed_wrapped);
    assert_before(&holed_wrapped, &measured, "f", "r");

    // A bare unit has no planner region at all, so the mirror edge could never
    // have reached this layout; the hole edge still does.
    let holed_bare = named(
        false,
        vec![
            value("r", "r = f(2)", &["f"]),
            holed_function("f", "def f(n: i32) = n", &[]),
        ],
    );
    assert!(
        violations(&holed_bare).is_empty(),
        "{}",
        holed_bare.source()
    );
    let measured = measure(&holed_bare);
    assert_before(&holed_bare, &measured, "f", "r");

    // Control: a complete header is honest before its body ([04-INF-6]), so
    // its reader is owed no edge and keeps its hoist slot ahead of the
    // function.
    let complete = named(
        true,
        vec![
            value("r", "r = g(2)", &["g"]),
            signed_function("anchor", "def anchor() -> i32 = 1", &[]),
            signed_function("g", "def g(n: i32) -> i32 = n", &[]),
        ],
    );
    assert!(violations(&complete).is_empty(), "{}", complete.source());
    let measured = measure(&complete);
    assert_before(&complete, &measured, "r", "g");
}

/// chelis#1485: every spelling is a cyclic component in the full reference
/// graph even where the scheduler-only mirror edge is absent. The component
/// is emitted contiguously and the scheduler never exercises a stall release.
#[test]
fn a_value_naming_a_function_that_reads_it_back_is_a_recorded_stall() {
    for (label, carried, helper, reader) in [
        (
            "signed reader",
            value("carried", "carried = wrap(f)", &["wrap", "f"]),
            function("wrap", "def wrap(g) = g", &[]),
            signed_function(
                "f",
                "def f(n: i32) -> i32 = if (n <= 0) then 0 else carried((n - 1))",
                &["carried"],
            ),
        ),
        (
            "lambda naming a signed reader",
            value(
                "carried",
                "carried = pick(fn (x: i32) -> f(x))",
                &["pick", "f"],
            ),
            function("pick", "def pick(g) = 5", &[]),
            signed_function("f", "def f(n: i32) -> i32 = add(n, carried)", &["carried"]),
        ),
        (
            "defsig-less reader",
            value("carried", "carried = wrap(g)", &["wrap", "g"]),
            function("wrap", "def wrap(h) = h", &[]),
            function(
                "g",
                "def g(n) = if (n <= 0) then 0 else carried((n - 1))",
                &["carried"],
            ),
        ),
    ] {
        let program = named(
            true,
            vec![
                signed_function("anchor", "def anchor() -> i32 = 1", &[]),
                carried,
                helper,
                reader,
            ],
        );
        let measured = measure(&program);
        let reference = reference(&program, &measured);
        let item_named = |name: &str| {
            measured
                .flat
                .iter()
                .position(|item| {
                    matches!(item, FlatItem::Declared(position) if program.declarations[*position].name == name)
                })
                .unwrap_or_else(|| panic!("missing `{name}` item"))
        };
        let carried_item = item_named("carried");
        let reader_item = item_named(if label == "defsig-less reader" {
            "g"
        } else {
            "f"
        });
        assert_eq!(
            reference.vertex[carried_item],
            reference.vertex[reader_item],
            "{label}: carried and its reader must share a full-reference component\n{}",
            program.source()
        );
        assert!(
            reference
                .cyclic_components
                .contains(&reference.vertex[carried_item]),
            "{label}: component must be cyclic\n{}",
            program.source()
        );
        assert!(violations(&program).is_empty(), "{}", program.source());
        assert_eq!(measured.schedule.len(), measured.flat.len());
    }
}

#[test]
fn a_genuine_binding_cycle_stays_total_as_one_component() {
    // `F1_three.ch`: `carried` reads the `defsig`-less `caller`, which needs
    // `helper`, which reads `carried`. A real reference cycle, which
    // `TopLevelReferenceGraph::report_initialization_errors` reports; the
    // schedule must emit every item once with the whole reference component
    // contiguous.
    let three = named(
        true,
        vec![
            function("caller", "def caller(n) = helper(n)", &["helper"]),
            signed_value("carried", "carried: i32 = caller(1)", &["caller"]),
            function("helper", "def helper(x) = carried", &["carried"]),
        ],
    );
    let measured = measure(&three);
    let three_reference = reference(&three, &measured);
    assert!(
        !three_reference.cyclic_components.is_empty(),
        "{}",
        three.source()
    );
    assert!(violations(&three).is_empty(), "{}", three.source());
    assert_eq!(measured.schedule.len(), measured.flat.len());
    let three_positions = positions(&measured.schedule);
    let component_slots = three_reference
        .vertex
        .iter()
        .enumerate()
        .filter(|(_, vertex)| three_reference.cyclic_components.contains(vertex))
        .map(|(index, _)| three_positions[index])
        .collect::<Vec<_>>();
    assert!(
        component_slots
            .windows(2)
            .all(|pair| pair[1] == pair[0] + 1),
        "full-reference component must be contiguous: {:?}",
        measured.schedule
    );

    // `C1_mirror_cycle.ch`: the same runtime cycle through a SIGNED pair.
    // The signed spelling has no mirror edge, but its calls and eager read
    // still form one full-reference component. The cycle detector owns the
    // verdict and the scheduler owns contiguous co-inference.
    let mirror = named(
        true,
        vec![
            signed_member(
                "ping",
                "def ping(n: i32) -> i32 = if (n <= 0) then 0 else pong((n - 1))",
                &["pong"],
                "pair",
            ),
            value("carried", "carried = ping(1)", &["ping"]),
            signed_member(
                "pong",
                "def pong(n: i32) -> i32 = if (n <= 0) then carried else ping((n - 1))",
                &["ping", "carried"],
                "pair",
            ),
        ],
    );
    let measured = measure(&mirror);
    let mirror_reference = reference(&mirror, &measured);
    assert!(
        !mirror_reference.cyclic_components.is_empty(),
        "{}",
        mirror.source()
    );
    assert!(violations(&mirror).is_empty(), "{}", mirror.source());
    assert_eq!(measured.schedule.len(), measured.flat.len());
    let mirror_positions = positions(&measured.schedule);
    let component_slots = mirror_reference
        .vertex
        .iter()
        .enumerate()
        .filter(|(_, vertex)| mirror_reference.cyclic_components.contains(vertex))
        .map(|(index, _)| mirror_positions[index])
        .collect::<Vec<_>>();
    assert!(
        component_slots
            .windows(2)
            .all(|pair| pair[1] == pair[0] + 1),
        "full-reference component must be contiguous: {:?}",
        measured.schedule
    );
}

#[test]
fn a_mixed_bare_and_module_component_follows_the_value_a_member_reads() {
    // Surf cannot spell this: the component has one member outside the module
    // wrapper and one inside, so only the module member is hoisted while the
    // group is emitted at whichever member the schedule reaches first.
    let exprs = chelis_deep::parse_and_stamp_file(
        "(def {} ping (fn {} (params {} n) (app {} (var {} pong) (var {} n))))\n\
         (module {} Mixed\n  \
           (def {} carried (lit {type: (t-prim {} i32)} 7))\n  \
           (def {} pong (fn {} (params {} n) (app {} (var {} add) (var {} carried) (app {} (var {} ping) (var {} n))))))\n",
    )
    .expect("mixed Deep fixture parses");
    let items = top_level_decl_items_with_modules(&exprs);
    assert_eq!(items.len(), 3);
    let plan = FunctionInferencePlan::build(&items);
    let schedule = primary_inference_schedule(&plan, &items);
    // carried (1) must precede both members; members contiguous.
    let position = positions(&schedule);
    assert!(
        position[1] < position[0] && position[1] < position[2],
        "{schedule:?}"
    );
    assert_eq!(
        (position[0] as isize - position[2] as isize).abs(),
        1,
        "{schedule:?}"
    );
}

/// [04-INF-8]: when an eager root is itself cyclic, CycleDetected owns the
/// verdict even if that root also reads a later eager value. The later value's
/// body must therefore be inferred before the rejected cyclic component so
/// its exact type can be made temporarily visible without deleting the
/// ordinary [04-INF-4] diagnostic after it has been produced.
#[test]
fn cycle_precedence_value_is_available_before_the_cyclic_component() {
    let source = "module CyclePrecedence\n\n\
                  root = add(read_root(), later)\n\n\
                  later = 5i32\n\n\
                  def read_root() -> i32 = root\n";
    let declarations = chelis_surf::parser::parse_str(source).expect("fixture parses");
    let exprs =
        chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar");
    let items = top_level_decl_items_with_modules(&exprs);
    let plan = FunctionInferencePlan::build(&items);
    let schedule = primary_inference_schedule(&plan, &items);
    let item_named = |name: &str| {
        items
            .iter()
            .position(|(_, expr)| {
                matches!(stamped_parts(expr), Some((DeepTag::Def, _, _)))
                    && top_level_decl_name(expr) == Some(name)
            })
            .unwrap_or_else(|| panic!("missing `{name}` item"))
    };
    let root = item_named("root");
    let later = item_named("later");
    let read_root = item_named("read_root");
    let position = positions(&schedule);
    assert!(
        position[later] < position[root] && position[later] < position[read_root],
        "the exact later value must precede both cyclic members: {schedule:?}"
    );
    assert_eq!(
        (position[root] as isize - position[read_root] as isize).abs(),
        1,
        "the cyclic component must remain contiguous: {schedule:?}"
    );
}
