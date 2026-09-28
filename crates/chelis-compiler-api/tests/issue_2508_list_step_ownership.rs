//! chelis#2508, chelis#2505 and chelis#2332: every higher-order list operation
//! must return every allocation it touches when its items own heap payloads.
//!
//! The ownership IR models each loop step (`list_push` for `map` and `scan`,
//! `list_extend` for `flat_map`, `filter_step` and `partition_step`) as moving
//! the step's item into the accumulator. The C emitter realized that move with
//! the cloning accumulator entry points and never released the moved owner,
//! and `filter` never released an item its predicate rejected. Direct `index`
//! reads were balanced; the leak #2332 attributed to `chelis_list_index` is
//! this one. The `index_reads`, `zip`, `enumerate` and scalar-accumulator
//! `fold` shapes already balanced on the base, so for them the corpus is a
//! lock rather than a regression test.
//!
//! Two emitter defects in the same loops made some shapes fail to compile: a
//! `filter` or `partition` whose callback body is a bare variable assigned the
//! predicate's result origin before declaring it, and a `fold` whose unread
//! accumulator has a struct C type was cleared with a pointer `NULL`.
//!
//! chelis#2579, chelis#2580 and chelis#2578 are `scan`'s accumulator: the copy
//! paying for a seed the caller keeps ran after the loop, the body's state
//! named the variable the callback's result had overwritten, and a named
//! callback's result was bound to the output list. The seeded rows cross
//! `fold` and `scan` with every seed owner and callback form.
//!
//! The corpus crosses every operation with every heap item kind. Oracle: the
//! compiled program runs against the `ownership-ledger` runtime, every
//! allocation must be finalized with no live owner left, and stdout must equal
//! the in-process evaluator's rendering of the same roots. Every case runs
//! before the test reports, so one failing shape cannot hide another.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// One item type whose values own a heap payload, with the expressions each
/// operation needs over a value `x` of that type.
struct ItemKind {
    name: &'static str,
    ty: &'static str,
    /// Declarations the kind needs ahead of the case.
    prelude: &'static str,
    /// A list literal of two freshly allocated items.
    items: &'static str,
    /// An expression of the item type built from `x`.
    transform: &'static str,
    /// A `bool` expression over `x`.
    predicate: &'static str,
    /// An `i64` expression that reads `x`.
    measure: &'static str,
    /// A freshly allocated item used as an initial accumulator.
    seed: &'static str,
    /// An expression of the item type that reads both an accumulator `acc`
    /// and an item `x`.
    combine: &'static str,
    /// Whether the checker treats a value of this type as linear: a closure
    /// that captures one consumes it, so a later read is rejected.
    linear: bool,
}

const KINDS: &[ItemKind] = &[
    ItemKind {
        name: "string",
        ty: "string",
        prelude: "",
        items: "[string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")]",
        transform: "string_concat(x, \"!\")",
        predicate: "gt(string_len(x), 2i64)",
        measure: "string_len(x)",
        seed: "string_concat(\"in\", \"it\")",
        combine: "string_concat(acc, x)",
        linear: false,
    },
    ItemKind {
        name: "list_of_string",
        ty: "List[string]",
        prelude: "",
        items: "[[string_concat(\"ab\", \"c\"), \"x\"], [string_concat(\"y\", \"z\")]]",
        transform: "map(fn (s: string) -> string_concat(s, \"!\"), x)",
        predicate: "gt(len(x), 1i64)",
        measure: "len(x)",
        seed: "[string_concat(\"in\", \"it\")]",
        combine: "concat(acc, x)",
        linear: false,
    },
    ItemKind {
        name: "adt_with_string",
        ty: "Named",
        prelude: "type Named =\n  | Named(string, i64)\n",
        items: "[Named(string_concat(\"ab\", \"c\"), 3i64), Named(string_concat(\"x\", \"\"), 1i64)]",
        transform: "match x with {\n    | Named(s, k) => Named(string_concat(s, \"!\"), k)\n  }",
        predicate: "match x with {\n    | Named(s, k) => gt(k, 2i64)\n  }",
        measure: "match x with {\n    | Named(s, k) => add(string_len(s), k)\n  }",
        seed: "Named(string_concat(\"in\", \"it\"), 0i64)",
        combine: "match acc with {\n    | Named(s, k) => match x with {\n        | Named(t, j) => Named(string_concat(s, t), add(k, j))\n      }\n  }",
        linear: false,
    },
    ItemKind {
        name: "tensor",
        ty: "tensor[3, f32]",
        prelude: "",
        items: "[to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0])]",
        transform: "(x + x)",
        predicate: "gt(tensor_to_scalar(sum(x, 0)), 7.0)",
        measure: "len([x])",
        seed: "to_tensor([0.0, 0.0, 0.0])",
        combine: "(acc + x)",
        linear: true,
    },
];

/// One operation shape. `{T}`, `{items}`, `{transform}`, `{predicate}`,
/// `{measure}` and `{seed}` are replaced from the item kind.
struct Shape {
    name: &'static str,
    result: &'static str,
    body: &'static str,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "map_owned_result",
        result: "List[{T}]",
        body: "map(fn (x: {T}) -> {transform}, {items})",
    },
    Shape {
        name: "map_scalar_result",
        result: "List[i64]",
        body: "map(fn (x: {T}) -> {measure}, {items})",
    },
    Shape {
        name: "map_in_map",
        result: "List[List[{T}]]",
        body: "map(fn (xs: List[{T}]) -> map(fn (x: {T}) -> {transform}, xs), [{items}, {items}])",
    },
    Shape {
        name: "map_in_map_scalar",
        result: "List[List[i64]]",
        body: "map(fn (xs: List[{T}]) -> map(fn (x: {T}) -> {measure}, xs), [{items}, {items}])",
    },
    Shape {
        name: "flat_map",
        result: "List[{T}]",
        body: "flat_map(fn (x: {T}) -> [{transform}], {items})",
    },
    Shape {
        name: "flat_map_scalar",
        result: "List[i64]",
        body: "flat_map(fn (x: {T}) -> [{measure}, 1i64], {items})",
    },
    Shape {
        name: "filter",
        result: "List[{T}]",
        body: "filter(fn (x: {T}) -> {predicate}, {items})",
    },
    Shape {
        name: "filter_captured_flag",
        result: "List[{T}]",
        body: "filter(fn (x: {T}) -> flag, {items})",
    },
    Shape {
        name: "filter_branching",
        result: "List[{T}]",
        body: "filter(fn (x: {T}) -> if flag then {predicate} else false, {items})",
    },
    Shape {
        name: "partition",
        result: "(List[{T}], List[{T}])",
        body: "partition(fn (x: {T}) -> {predicate}, {items})",
    },
    Shape {
        name: "partition_captured_flag",
        result: "(List[{T}], List[{T}])",
        body: "partition(fn (x: {T}) -> flag, {items})",
    },
    Shape {
        name: "fold_scalar_accumulator",
        result: "i64",
        body: "fold(fn (acc: i64, x: {T}) -> add(acc, {measure}), 0i64, {items})",
    },
    Shape {
        name: "fold_unread_accumulator",
        result: "{T}",
        body: "fold(fn (acc: {T}, x: {T}) -> {transform}, {seed}, {items})",
    },
    Shape {
        name: "fold_selected_accumulator",
        result: "{T}",
        body: "fold(fn (acc: {T}, x: {T}) -> if flag then acc else x, {seed}, {items})",
    },
    Shape {
        name: "scan_unread_accumulator",
        result: "List[{T}]",
        body: "scan(fn (acc: {T}, x: {T}) -> {transform}, {seed}, {items})",
    },
    Shape {
        name: "scan_selected_accumulator",
        result: "List[{T}]",
        body: "scan(fn (acc: {T}, x: {T}) -> if flag then acc else x, {seed}, {items})",
    },
    Shape {
        name: "index_reads",
        result: "i64",
        body: "fold(fn (acc: i64, i: i64) -> add(acc, len([index(xs, i)])), 0i64, [0i64, 1i64, 0i64, 1i64])",
    },
    Shape {
        name: "zip",
        result: "List[({T}, i64)]",
        body: "zip({items}, [1i64, 2i64])",
    },
    Shape {
        name: "enumerate",
        result: "List[(i64, {T})]",
        body: "enumerate({items})",
    },
];

fn instantiate(kind: &ItemKind, shape: &Shape) -> String {
    let fill = |text: &str| {
        text.replace("{items}", kind.items)
            .replace("{transform}", kind.transform)
            .replace("{predicate}", kind.predicate)
            .replace("{measure}", kind.measure)
            .replace("{seed}", kind.seed)
            .replace("{T}", kind.ty)
    };
    // `index_reads` reads a bound list, so its function takes the list; every
    // other case builds its items inside the function so each call owns them.
    if shape.name == "index_reads" {
        return format!(
            "{}def case(flag: bool, xs: List[{}]) -> {} = {}\na = case(true, {})\nb = case(false, {})\n",
            kind.prelude,
            kind.ty,
            fill(shape.result),
            fill(shape.body),
            kind.items,
            kind.items,
        );
    }
    format!(
        "{}def case(flag: bool) -> {} =\n  {}\na = case(true)\nb = case(false)\n",
        kind.prelude,
        fill(shape.result),
        fill(shape.body),
    )
}

/// Callback shapes whose body returns, or reads, an owner from outside the
/// loop by bare reference: `{y}` is an outer item and `{ys}` an outer list of
/// items. The loop's step must not run before the copy that pays for that
/// owner, or a consuming step releases the outer owner itself.
const OUTER_SHAPES: &[Shape] = &[
    Shape {
        name: "map_returns_outer",
        result: "List[{T}]",
        body: "map(fn (x: {T}) -> {y}, {items})",
    },
    Shape {
        name: "flat_map_returns_outer",
        result: "List[{T}]",
        body: "flat_map(fn (x: {T}) -> {ys}, {items})",
    },
    Shape {
        name: "filter_reads_outer",
        result: "List[{T}]",
        body: "filter(fn (x: {T}) -> gt(len({ys}), 0i64), {items})",
    },
    Shape {
        name: "partition_reads_outer",
        result: "(List[{T}], List[{T}])",
        body: "partition(fn (x: {T}) -> gt(len({ys}), 0i64), {items})",
    },
    Shape {
        name: "fold_returns_outer",
        result: "{T}",
        body: "fold(fn (acc: {T}, x: {T}) -> {y}, {seed}, {items})",
    },
    Shape {
        name: "scan_returns_outer",
        result: "List[{T}]",
        body: "scan(fn (acc: {T}, x: {T}) -> {y}, {seed}, {items})",
    },
];

/// Where the outer owner lives.
#[derive(Clone, Copy)]
enum OuterOwner {
    /// A local of the calling function, not used after the loop.
    Local,
    /// A local of the calling function, read again after the loop.
    LocalUsedAfter,
    /// A parameter, so the caller's storage owns it.
    Parameter,
    /// A top-level binding.
    Global,
}

const OUTER_OWNERS: &[(&str, OuterOwner)] = &[
    ("local", OuterOwner::Local),
    ("local_used_after", OuterOwner::LocalUsedAfter),
    ("parameter", OuterOwner::Parameter),
    ("global", OuterOwner::Global),
];

fn instantiate_outer(kind: &ItemKind, shape: &Shape, owner: OuterOwner) -> String {
    let (y, ys) = match owner {
        OuterOwner::Global => ("outer_y", "outer_ys"),
        _ => ("y", "ys"),
    };
    let fill = |text: &str| {
        text.replace("{items}", kind.items)
            .replace("{seed}", kind.seed)
            .replace("{y}", y)
            .replace("{ys}", ys)
            .replace("{T}", kind.ty)
    };
    let result = fill(shape.result);
    let body = fill(shape.body);
    let seed = kind.seed;
    let prelude = kind.prelude;
    let ty = kind.ty;
    match owner {
        OuterOwner::Local => format!(
            "{prelude}def case(flag: bool) -> {result} = {{\n  y = {seed}\n  ys = [{seed}]\n  {body}\n}}\n\
             a = case(true)\nb = case(false)\n"
        ),
        OuterOwner::LocalUsedAfter => format!(
            "{prelude}def case(flag: bool) -> ({result}, {ty}, i64) = {{\n  y = {seed}\n  ys = [{seed}]\n  \
             r = {body}\n  (r, y, len(ys))\n}}\n\
             a = case(true)\nb = case(false)\n"
        ),
        OuterOwner::Parameter => format!(
            "{prelude}def case(flag: bool, y: {ty}, ys: List[{ty}]) -> {result} =\n  {body}\n\
             a = case(true, {seed}, [{seed}])\nb = case(false, {seed}, [{seed}])\n"
        ),
        OuterOwner::Global => format!(
            "{prelude}outer_y = {seed}\nouter_ys = [{seed}]\n\
             def case(flag: bool) -> {result} =\n  {body}\n\
             a = case(true)\nb = case(false)\n"
        ),
    }
}

/// A `fold` or `scan` accumulator callback. `{T}` and `{combine}` are
/// replaced from the item kind; `{z}` names an owner the callback captures.
struct SeededCallback {
    name: &'static str,
    /// A top-level definition the callback names, or empty for an inline one.
    definition: &'static str,
    callback: &'static str,
}

const SEEDED_CALLBACKS: &[SeededCallback] = &[
    SeededCallback {
        name: "inline_reads_acc",
        definition: "",
        callback: "fn (acc: {T}, x: {T}) -> {combine}",
    },
    SeededCallback {
        name: "inline_returns_acc",
        definition: "",
        callback: "fn (acc: {T}, x: {T}) -> acc",
    },
    SeededCallback {
        name: "inline_nested_returns_acc",
        definition: "",
        callback: "fn (acc: {T}, x: {T}) -> fold(fn (inner: {T}, y: {T}) -> acc, x, [x])",
    },
    SeededCallback {
        name: "inline_returns_captured",
        definition: "",
        callback: "fn (acc: {T}, x: {T}) -> z",
    },
    SeededCallback {
        name: "named_reads_acc",
        definition: "def step(acc: {T}, x: {T}) -> {T} = {combine}\n",
        callback: "step",
    },
    SeededCallback {
        name: "named_returns_acc",
        definition: "def keep(acc: {T}, x: {T}) -> {T} = acc\n",
        callback: "keep",
    },
];

/// Where a `fold` or `scan` seed comes from.
#[derive(Clone, Copy, PartialEq)]
enum SeedOwner {
    /// A fresh expression written in the call.
    Fresh,
    /// A local of the calling function, not used after the loop.
    Local,
    /// A local of the calling function, read again after the loop.
    LocalUsedAfter,
    /// A parameter, so the caller's storage owns it.
    Parameter,
    /// A parameter, read again after the loop.
    ParameterUsedAfter,
    /// A top-level binding, printed after the calls.
    Global,
}

const SEED_OWNERS: &[(&str, SeedOwner)] = &[
    ("fresh", SeedOwner::Fresh),
    ("local", SeedOwner::Local),
    ("local_used_after", SeedOwner::LocalUsedAfter),
    ("parameter", SeedOwner::Parameter),
    ("parameter_used_after", SeedOwner::ParameterUsedAfter),
    ("global", SeedOwner::Global),
];

fn instantiate_seeded(
    kind: &ItemKind,
    op: &str,
    callback: &SeededCallback,
    owner: SeedOwner,
) -> String {
    let fill = |text: &str| {
        text.replace("{combine}", kind.combine)
            .replace("{T}", kind.ty)
    };
    let ty = kind.ty;
    let seed_name = match owner {
        SeedOwner::Fresh => kind.seed,
        SeedOwner::Global => "g0",
        _ => "s0",
    };
    let call = format!(
        "{op}({}, {seed_name}, {})",
        fill(callback.callback),
        kind.items
    );
    let result = if op == "scan" {
        format!("List[{ty}]")
    } else {
        ty.to_string()
    };
    let (result, value) = match owner {
        SeedOwner::LocalUsedAfter | SeedOwner::ParameterUsedAfter => {
            (format!("({result}, {ty})"), format!("({call}, s0)"))
        }
        _ => (result, call),
    };
    let parameters = match owner {
        SeedOwner::Parameter | SeedOwner::ParameterUsedAfter => format!("flag: bool, s0: {ty}"),
        _ => "flag: bool".to_string(),
    };
    let mut locals = String::new();
    if callback.name == "inline_returns_captured" {
        locals.push_str(&format!("  z = {}\n", kind.seed));
    }
    if matches!(owner, SeedOwner::Local | SeedOwner::LocalUsedAfter) {
        locals.push_str(&format!("  s0 = {}\n", kind.seed));
    }
    let (globals, arguments, after) = match owner {
        SeedOwner::Parameter | SeedOwner::ParameterUsedAfter => (
            String::new(),
            (
                format!("true, {}", kind.seed),
                format!("false, {}", kind.seed),
            ),
            String::new(),
        ),
        SeedOwner::Global => (
            format!("g0 = {}\n", kind.seed),
            ("true".to_string(), "false".to_string()),
            "c = g0\n".to_string(),
        ),
        _ => (
            String::new(),
            ("true".to_string(), "false".to_string()),
            String::new(),
        ),
    };
    // A block needs a binding before its tail, so a body with no locals is
    // the bare expression.
    let body = if locals.is_empty() {
        format!("\n  {value}\n")
    } else {
        format!(" {{\n{locals}  {value}\n}}\n")
    };
    format!(
        "{}{globals}{}def case({parameters}) -> {result} ={body}\
         a = case({})\nb = case({})\n{after}",
        kind.prelude,
        fill(callback.definition),
        arguments.0,
        arguments.1,
    )
}

/// Run every seeded `fold` and `scan` over one item kind.
fn every_seeded_accumulator_over(kind_name: &str) {
    let kind = KINDS
        .iter()
        .find(|kind| kind.name == kind_name)
        .expect("declared item kind");
    let cases: Vec<(String, String, Expectation)> = ["fold", "scan"]
        .iter()
        .flat_map(|op| {
            SEEDED_CALLBACKS.iter().flat_map(move |callback| {
                SEED_OWNERS.iter().map(move |(owner_name, owner)| {
                    (
                        format!("{op}_{}_{owner_name}_{}", callback.name, kind.name),
                        instantiate_seeded(kind, op, callback, *owner),
                        Expectation::AgreesWithEval,
                    )
                })
            })
        })
        .collect();
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|(name, source, expectation)| check_expecting(name, source, *expectation).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} `{}` seeded-accumulator cases failed:\n\n{}",
        failures.len(),
        cases.len(),
        kind.name,
        failures.join("\n\n")
    );
}

/// The evaluator's rejection of a program it must not run.
fn eval_error(source: &str) -> String {
    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .expect_err("the evaluator must reject this case");
    format!("{error:?}")
}

/// The evaluator's rendering of every root, in the compiled driver's format.
fn evaluated(source: &str) -> String {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|error| panic!("evaluator rejected the case: {error:?}"));
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().expect("named root"),
                root.display.as_deref().expect("rendered root")
            )
        })
        .collect()
}

fn check(name: &str, source: &str) -> Result<(), String> {
    check_expecting(name, source, Expectation::AgreesWithEval)
}

/// What one corpus case must do.
#[derive(Clone, Copy)]
enum Expectation {
    /// Compiled C returns every allocation and prints what eval prints.
    AgreesWithEval,
    /// The checker rejects the program with this diagnostic kind.
    CheckerRejects(&'static str),
}

fn check_expecting(name: &str, source: &str, expectation: Expectation) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| match expectation {
        Expectation::AgreesWithEval => {
            let expected = evaluated(source);
            let generated = ownership_support::emit(source, name);
            let (summary, stdout) = ownership_support::run_program(&generated);
            ownership_support::balanced(&summary);
            assert_eq!(stdout, expected, "compiled output differs from eval");
        }
        Expectation::CheckerRejects(kind) => {
            let error = eval_error(source);
            assert!(
                error.contains(kind),
                "expected a `{kind}` rejection: {error}"
            );
        }
    }))
    .map_err(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        let head: String = message.chars().take(600).collect();
        format!("{name}:\n{source}\n  -> {head}")
    })
}

/// Run every shape over one item kind and report every failing case.
fn every_shape_over(kind_name: &str) {
    let kind = KINDS
        .iter()
        .find(|kind| kind.name == kind_name)
        .expect("declared item kind");
    let failures: Vec<String> = SHAPES
        .iter()
        .filter_map(|shape| {
            let name = format!("{}_{}", shape.name, kind.name);
            check(&name, &instantiate(kind, shape)).err()
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} `{}` cases failed:\n\n{}",
        failures.len(),
        SHAPES.len(),
        kind.name,
        failures.join("\n\n")
    );
}

/// Run every outer-owner shape and owner form over one item kind.
fn every_outer_owner_over(kind_name: &str) {
    let kind = KINDS
        .iter()
        .find(|kind| kind.name == kind_name)
        .expect("declared item kind");
    let failures: Vec<String> = OUTER_SHAPES
        .iter()
        .flat_map(|shape| OUTER_OWNERS.iter().map(move |owner| (shape, owner)))
        .filter_map(|(shape, (owner_name, owner))| {
            let name = format!("{}_{owner_name}_{}", shape.name, kind.name);
            let expectation = match owner {
                // A closure capture consumes a linear value, so the program
                // that reads it after the loop is not a program.
                OuterOwner::LocalUsedAfter if kind.linear => {
                    Expectation::CheckerRejects("UseAfterConsume")
                }
                _ => Expectation::AgreesWithEval,
            };
            check_expecting(&name, &instantiate_outer(kind, shape, *owner), expectation).err()
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} `{}` outer-owner cases failed:\n\n{}",
        failures.len(),
        OUTER_SHAPES.len() * OUTER_OWNERS.len(),
        kind.name,
        failures.join("\n\n")
    );
}

// REGRESSION TESTS for the whole family, one per item kind so the kinds run
// in parallel. On `32e4122e4` 51 of the 76 cases failed: every owned-result
// `map`, nested `map`, `flat_map`, `filter`, `partition` and `scan` over a
// heap item kind left live owners; `filter` and `partition` with a captured
// `flag` body did not compile, nor did `fold` with an unread `string`
// accumulator; and `fold` with an unread ADT accumulator aborted, scanning
// the result origin of the accumulator it had already released.

#[test]
fn every_list_step_over_string_items_returns_every_allocation() {
    every_shape_over("string");
}

#[test]
fn every_list_step_over_list_items_returns_every_allocation() {
    every_shape_over("list_of_string");
}

#[test]
fn every_list_step_over_adt_items_returns_every_allocation() {
    every_shape_over("adt_with_string");
}

#[test]
fn every_list_step_over_tensor_items_returns_every_allocation() {
    every_shape_over("tensor");
}

// REGRESSION TESTS, chelis#2571 round 1: a callback that returns an outer
// owner by bare reference. At `89d706ee9` the loop's consuming step ran before
// the copy that pays for that owner, so `flat_map` released the caller's,
// the closure's or the global's list on the first iteration.

#[test]
fn every_list_step_over_an_outer_string_owner_runs_its_copy_first() {
    every_outer_owner_over("string");
}

#[test]
fn every_list_step_over_an_outer_list_owner_runs_its_copy_first() {
    every_outer_owner_over("list_of_string");
}

#[test]
fn every_list_step_over_an_outer_adt_owner_runs_its_copy_first() {
    every_outer_owner_over("adt_with_string");
}

#[test]
fn every_list_step_over_an_outer_tensor_owner_runs_its_copy_first() {
    every_outer_owner_over("tensor");
}

// REGRESSION TESTS, chelis#2579, chelis#2580 and chelis#2578: a `scan`
// accumulator seeded from an owner the caller keeps, or whose callback reads
// or is a named definition, with `fold` beside it on every row.

#[test]
fn every_seeded_accumulator_over_string_items_leaves_its_seed_intact() {
    every_seeded_accumulator_over("string");
}

#[test]
fn every_seeded_accumulator_over_list_items_leaves_its_seed_intact() {
    every_seeded_accumulator_over("list_of_string");
}

#[test]
fn every_seeded_accumulator_over_adt_items_leaves_its_seed_intact() {
    every_seeded_accumulator_over("adt_with_string");
}

#[test]
fn every_seeded_accumulator_over_tensor_items_leaves_its_seed_intact() {
    every_seeded_accumulator_over("tensor");
}

/// Run one named witness and fail with its report.
fn witness(name: &str, source: &str) {
    if let Err(report) = check(name, source) {
        panic!("{report}");
    }
}

/// REGRESSION TEST, round-1 witness c03: at `89d706ee9` the extend released
/// the captured local on the first iteration and the run aborted.
#[test]
fn flat_map_returning_a_captured_local_list_keeps_it_for_later_use() {
    witness(
        "c03_flat_map_captured_list",
        "def case(flag: bool) -> (List[string], List[string]) = {\n  \
         ys = [string_concat(\"y\", \"s\")]\n  \
         r = flat_map(fn (x: string) -> ys, [string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")])\n  \
         (r, ys)\n}\n\
         a = case(true)\n",
    );
}

/// REGRESSION TEST, round-1 witness c21: the captured list is the caller's
/// argument, and the extend released the caller's storage.
#[test]
fn flat_map_returning_a_captured_parameter_leaves_the_caller_its_list() {
    witness(
        "c21_flat_map_captured_param",
        "def case(ys: List[string]) -> List[string] =\n  \
         flat_map(fn (x: string) -> ys, [string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")])\n\
         a = case([string_concat(\"y\", \"s\")])\n",
    );
}

/// REGRESSION TEST, round-1 witness c28: the captured list is a top-level
/// binding, printed after the call.
#[test]
fn flat_map_returning_a_global_list_leaves_the_global_intact() {
    witness(
        "c28_flat_map_global_list",
        "ys = [string_concat(\"y\", \"s\")]\n\
         def case(flag: bool) -> List[string] =\n  \
         flat_map(fn (x: string) -> ys, [string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")])\n\
         a = case(true)\n\
         c = ys\n",
    );
}

/// REGRESSION TEST, chelis#2580 witness w03: the `scan` state is read by a
/// nested `flat_map`, and the copy that pays for it retained the `flat_map`
/// destination the callback had already written over the state's variable.
#[test]
fn scan_state_read_by_a_nested_flat_map_is_copied_before_it_is_replaced() {
    witness(
        "w03_scan_inner_flat_map_acc",
        "def case(xs: List[string]) -> List[List[string]] =\n  \
         scan(fn (acc: List[string], x: string) -> flat_map(fn (s: string) -> acc, [x, x]), [string_concat(\"s\", \"d\")], xs)\n\
         a = case([string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")])\n",
    );
}

/// Every declared kind has a test above, so a kind added to the table
/// cannot go unrun.
#[test]
fn every_item_kind_has_a_test() {
    let source = include_str!("issue_2508_list_step_ownership.rs");
    for kind in KINDS {
        assert!(
            source.contains(&format!("every_shape_over(\"{}\");", kind.name))
                && source.contains(&format!("every_outer_owner_over(\"{}\");", kind.name))
                && source.contains(&format!(
                    "every_seeded_accumulator_over(\"{}\");",
                    kind.name
                )),
            "item kind `{}` has no test",
            kind.name
        );
    }
}
