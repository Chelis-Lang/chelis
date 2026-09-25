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
    catch_unwind(AssertUnwindSafe(|| {
        let expected = evaluated(source);
        let generated = ownership_support::emit(source, name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
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

/// Every declared kind has a test above, so a kind added to the table
/// cannot go unrun.
#[test]
fn every_item_kind_has_a_test() {
    let source = include_str!("issue_2508_list_step_ownership.rs");
    for kind in KINDS {
        assert!(
            source.contains(&format!("every_shape_over(\"{}\");", kind.name)),
            "item kind `{}` has no test",
            kind.name
        );
    }
}
