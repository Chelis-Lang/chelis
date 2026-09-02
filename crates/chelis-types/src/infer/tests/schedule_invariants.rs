//! chelis#1134 / [04-INF-4]: structural oracle for `primary_inference_schedule`.
//!
//! The ordering matrices in `tests/issue_1134_forward_reference_parity.rs`
//! enumerate accept/reject verdicts over declaration layouts. Four repairs of
//! the schedule each shipped a green verdict suite and were found by a
//! reviewer constructing a layout the suite could not express, because the
//! defects lived in the emitted ORDER, which no verdict can see. This oracle
//! asserts the schedule's invariants directly on the returned `Vec<usize>`:
//!
//! - I1: a permutation of the item ordinals, with every recursive component's
//!   members contiguous;
//! - I2: when the reference graph is acyclic, every reference edge is
//!   respected (referenced before referencer);
//! - I3: when the hoist order already respects every edge, the schedule IS
//!   the hoist order once components are collapsed, so every program without
//!   a barrier keeps the exact order it had before chelis#1134;
//! - I4: when the reference graph has a cycle, the schedule is still total
//!   and every call edge is respected, so a callee is still inferred before
//!   its caller while a genuine binding cycle is being broken.
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

/// One named declaration as the generator wrote it.
#[derive(Clone, Debug)]
struct Declaration {
    name: &'static str,
    kind: Kind,
    source: String,
    references: Vec<&'static str>,
    component: Option<&'static str>,
}

fn function(name: &'static str, source: &str, references: &[&'static str]) -> Declaration {
    Declaration {
        name,
        kind: Kind::Function,
        source: source.to_string(),
        references: references.to_vec(),
        component: None,
    }
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

fn value(name: &'static str, source: &str, references: &[&'static str]) -> Declaration {
    Declaration {
        name,
        kind: Kind::Value,
        source: source.to_string(),
        references: references.to_vec(),
        component: None,
    }
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
}

fn measure(program: &Program) -> Measured {
    let source = program.source();
    let declarations = chelis_surf::parser::parse_str(&source)
        .unwrap_or_else(|error| panic!("generated Surf must parse: {error:?}\n{source}"));
    let exprs = chelis_surf::desugar::desugar_program(&declarations);
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
    Measured {
        schedule,
        plan,
        flat,
        module_fn_indices,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeKind {
    Read,
    Call,
    Mirror,
}

/// The reference graph, derived from the generator's declarations over the
/// real item ordinals. `vertex[i]` is the contracted vertex of item `i`.
struct Reference {
    vertex: Vec<usize>,
    edges: BTreeMap<(usize, usize), EdgeKind>,
    hoist: Vec<usize>,
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
    }
    let mut value_ordinal: BTreeMap<&str, usize> = BTreeMap::new();
    let mut function_index: BTreeMap<&str, usize> = BTreeMap::new();
    let mut first_member: BTreeMap<&str, usize> = BTreeMap::new();
    for index in 0..item_count {
        let Some(declaration) = declared(index) else {
            continue;
        };
        match declaration.kind {
            Kind::Value => {
                value_ordinal.entry(declaration.name).or_insert(index);
            }
            Kind::Function => {
                function_index.entry(declaration.name).or_insert(index);
            }
        }
        if let Some(label) = declaration.component {
            first_member.entry(label).or_insert(index);
        }
    }
    let vertex = (0..item_count)
        .map(|index| {
            declared(index)
                .and_then(|declaration| declaration.component)
                .map_or(index, |label| first_member[label])
        })
        .collect::<Vec<_>>();
    let floor = measured.module_fn_indices.first().copied();
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
                } else if floor.is_some_and(|floor| index >= floor) {
                    // Unconditional, signed or not: a declared header may be
                    // partial or generic and is only the function's scheme
                    // once its body has narrowed it (chelis#1486).
                    edges.insert((vertex[function], vertex[index]), EdgeKind::Mirror);
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
        let mut slots = members
            .iter()
            .map(|member| position[*member])
            .collect::<Vec<_>>();
        slots.sort_unstable();
        let contiguous = slots.windows(2).all(|pair| pair[1] == pair[0] + 1);
        if !contiguous {
            failures.push(format!(
                "I1: component at {vertex} is not contiguous (members {members:?} at slots {slots:?}) in {schedule:?}\n{source}"
            ));
        }
    }

    let acyclic = is_acyclic(&reference);
    if acyclic {
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
        // I4: total (I1 above) and call edges still respected.
        for (&edge, kind) in &reference.edges {
            if *kind == EdgeKind::Call && !respects(schedule, &reference.vertex, edge) {
                failures.push(format!(
                    "I4: call edge {edge:?} violated while breaking a cycle: {schedule:?}\n{source}"
                ));
            }
        }
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
            value("carried", "carried: int32 = tailfn(1)", &["tailfn"])
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
            value("carried", "carried: int32 = 7", &[])
        } else {
            value("carried", "carried = 7", &[])
        };
        for reader in [
            function("reader", "def reader() -> int32 = carried", &["carried"]),
            value("echoed", "echoed = carried", &["carried"]),
        ] {
            let alphabet = vec![
                function("anchor", "def anchor() -> int32 = 1", &[]),
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
            member(
                "ping",
                "def ping(n: int32) -> int32 = if (n <= 0) then 0 else pong((n - 1))",
                &["pong"],
                "pair",
            ),
            member(
                "pong",
                "def pong(n: int32) -> int32 = if (n <= 0) then carried else ping((n - 1))",
                &["ping", "carried"],
                "pair",
            ),
            carried,
        ];
        for mut layout in permutations(&alphabet) {
            layout.insert(0, function("seed", "def seed() -> int32 = 3", &[]));
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
            function("seed", "def seed() -> int32 = 3", &[]),
            function("anchor", "def anchor() -> int32 = 1", &[]),
            value("carried", "carried = seed()", &["seed"]),
            function(
                "later_reader",
                "def later_reader() -> int32 = carried",
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
                    function("seed", "def seed() -> int32 = 3", &[]),
                    member(
                        "ping",
                        "def ping(n: int32) -> int32 = if (n <= 0) then 0 else pong((n - 1))",
                        &["pong"],
                        "pair",
                    ),
                    carried,
                    member(
                        "pong",
                        "def pong(n: int32) -> int32 = if (n <= 0) then carried else ping((n - 1))",
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
            value("carried", "carried: int32 = tailfn(1)", &["tailfn"]),
            function("helper", "def helper(x) = carried", &["carried"]),
            function("tailfn", "def tailfn(n) = n", &[]),
        ],
    );
    let measured = measure(&program);
    let reference = reference(&program, &measured);
    assert!(is_acyclic(&reference), "D4_min has no reference cycle");
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
    assert!(is_acyclic(&reference), "C2 has no reference cycle");
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
            function("anchor", "def anchor() -> int32 = 1", &[]),
            value("first", "first = reads()", &["reads"]),
            value("second", "second = 7", &[]),
            function("reads", "def reads() = second", &["second"]),
        ],
    );
    let measured = measure(&program);
    let reference = reference(&program, &measured);
    assert!(is_acyclic(&reference));
    assert!(violations(&program).is_empty());
    assert_before(&program, &measured, "second", "reads");
    assert_before(&program, &measured, "reads", "first");
}

#[test]
fn a_value_naming_a_function_that_reads_it_back_is_a_recorded_stall() {
    // chelis#1485: `carried` names `f` without applying it and `f` reads
    // `carried`. Every reference is legal and there is no runtime cycle, but
    // the mirror edge `f -> carried` and the read edge `carried -> f` point
    // both ways, so the reference graph is cyclic and the schedule stalls.
    // The mirror cannot be dropped for a signed `f` (chelis#1486), so this
    // pins the stall: the reference graph must be cyclic and the schedule
    // total with callees first. It reddens when the mirror rule changes;
    // the parity suite and the CLI oracle pin the verdict itself and redden
    // when #1485 closes.
    for f in [
        "def f(n: int32) -> int32 = if (n <= 0) then 0 else carried((n - 1))",
        "def f(n) = if (n <= 0) then 0 else carried((n - 1))",
    ] {
        let program = named(
            true,
            vec![
                function("anchor", "def anchor() -> int32 = 1", &[]),
                value("carried", "carried = wrap(f)", &["wrap", "f"]),
                function("wrap", "def wrap(g) = g", &[]),
                function("f", f, &["carried"]),
            ],
        );
        let measured = measure(&program);
        let stall_reference = reference(&program, &measured);
        assert!(
            !is_acyclic(&stall_reference),
            "chelis#1485 is closed for this spelling; retire the ratchet\n{}",
            program.source()
        );
        assert!(violations(&program).is_empty(), "{}", program.source());
        assert_eq!(measured.schedule.len(), measured.flat.len());
    }
}

#[test]
fn a_genuine_binding_cycle_stays_total_with_callees_first() {
    // `F1_three.ch`: `carried` reads the `defsig`-less `caller`, which needs
    // `helper`, which reads `carried`. A real reference cycle, which
    // `detect_top_level_binding_cycles` reports; the schedule must still emit
    // every item once and keep the callee before its caller.
    let three = named(
        true,
        vec![
            function("caller", "def caller(n) = helper(n)", &["helper"]),
            value("carried", "carried: int32 = caller(1)", &["caller"]),
            function("helper", "def helper(x) = carried", &["carried"]),
        ],
    );
    let measured = measure(&three);
    let three_reference = reference(&three, &measured);
    assert!(!is_acyclic(&three_reference), "{}", three.source());
    assert!(violations(&three).is_empty(), "{}", three.source());
    assert_eq!(measured.schedule.len(), measured.flat.len());
    assert_before(&three, &measured, "helper", "caller");

    // `C1_mirror_cycle.ch`: the same runtime cycle through a signed pair.
    // The mirror edge is unconditional, so the reference graph is cyclic
    // here too; the schedule stays total and the detector owns the verdict.
    let mirror = named(
        true,
        vec![
            member(
                "ping",
                "def ping(n: int32) -> int32 = if (n <= 0) then 0 else pong((n - 1))",
                &["pong"],
                "pair",
            ),
            value("carried", "carried = ping(1)", &["ping"]),
            member(
                "pong",
                "def pong(n: int32) -> int32 = if (n <= 0) then carried else ping((n - 1))",
                &["ping", "carried"],
                "pair",
            ),
        ],
    );
    let measured = measure(&mirror);
    let mirror_reference = reference(&mirror, &measured);
    assert!(!is_acyclic(&mirror_reference), "{}", mirror.source());
    assert!(violations(&mirror).is_empty(), "{}", mirror.source());
    assert_eq!(measured.schedule.len(), measured.flat.len());
}

#[test]
fn a_mixed_bare_and_module_component_follows_the_value_a_member_reads() {
    // Surf cannot spell this: the component has one member outside the module
    // wrapper and one inside, so only the module member is hoisted while the
    // group is emitted at whichever member the schedule reaches first.
    let exprs = chelis_deep::parse_and_stamp_file(
        "(def {} ping (fn {} (params {} n) (app {} (var {} pong) (var {} n))))\n\
         (module {} Mixed\n  \
           (def {} carried (lit {type: (t-prim {} int32)} 7))\n  \
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
