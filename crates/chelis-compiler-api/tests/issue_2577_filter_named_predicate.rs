//! chelis#2577: a `filter` or `partition` whose predicate passes the loop item
//! to a named definition must keep the item for the step that follows.
//!
//! A by-value call consumes its argument, and the ownership lowering treated
//! the predicate's use of the item as the item's last, so the call took the
//! loop item and the `filter_step` or `partition_step` that keeps the item
//! read a dead owner: the build failed in the ownership verifier with
//! `owner %N ... is not live`. The step keeps the item after the predicate
//! reads it, so no use inside the predicate is the item's last.
//!
//! The corpus crosses every way a predicate reaches a named definition with
//! every heap item kind. Oracle: the compiled program runs against the
//! `ownership-ledger` runtime, every allocation must be finalized with no live
//! owner left, and stdout must equal the in-process evaluator's rendering of
//! the same roots. Every case runs before the test reports, so one failing
//! shape cannot hide another.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// One item type whose values own a heap payload, with a named predicate
/// `keep` that takes the item by value and a named function `pass` that
/// returns it.
struct ItemKind {
    name: &'static str,
    ty: &'static str,
    /// Declarations the kind needs ahead of the case, including `keep` and
    /// `pass`.
    prelude: &'static str,
    /// A list literal of two freshly allocated items, one kept and one not.
    items: &'static str,
}

const KINDS: &[ItemKind] = &[
    ItemKind {
        name: "string",
        ty: "string",
        prelude: "def keep(v: string) -> bool = gt(string_len(v), 2i64)\n\
                  def pass(v: string) -> string = v\n",
        items: "[string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")]",
    },
    ItemKind {
        name: "list_of_string",
        ty: "List[string]",
        prelude: "def keep(v: List[string]) -> bool = gt(len(v), 1i64)\n\
                  def pass(v: List[string]) -> List[string] = v\n",
        items: "[[string_concat(\"ab\", \"c\"), \"x\"], [string_concat(\"y\", \"z\")]]",
    },
    ItemKind {
        name: "adt_with_string",
        ty: "Named",
        prelude: "type Named =\n  | Named(string, i64)\n\
                  def keep(v: Named) -> bool =\n  match v with {\n    | Named(s, k) => gt(k, 2i64)\n  }\n\
                  def pass(v: Named) -> Named = v\n",
        items: "[Named(string_concat(\"ab\", \"c\"), 3i64), Named(string_concat(\"x\", \"\"), 1i64)]",
    },
    ItemKind {
        name: "tensor",
        ty: "tensor[3, f32]",
        prelude: "def keep(v: tensor[3, f32]) -> bool = gt(tensor_to_scalar(sum(v, 0)), 7.0)\n\
                  def pass(v: tensor[3, f32]) -> tensor[3, f32] = v\n",
        items: "[to_tensor([1.0, 2.0, 3.0], f32), to_tensor([4.0, 5.0, 6.0], f32)]",
    },
];

/// One operation shape. `{T}` and `{items}` are replaced from the item kind.
struct Shape {
    name: &'static str,
    result: &'static str,
    body: &'static str,
    /// Declarations this shape needs after the kind's prelude.
    helpers: &'static str,
    /// Whether this shape failed to build on `d029224fd`, the base of the
    /// fix. The other shapes are locks on the neighbouring lowering.
    regression: bool,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "filter_callback_calls_named",
        result: "List[{T}]",
        body: "filter(fn (x: {T}) -> keep(x), {items})",
        helpers: "",
        regression: true,
    },
    Shape {
        name: "filter_named_callback",
        result: "List[{T}]",
        body: "filter(keep, {items})",
        helpers: "",
        regression: true,
    },
    Shape {
        name: "filter_callback_block_calls_named",
        result: "List[{T}]",
        body: "filter(fn (x: {T}) -> {\n    kept = keep(x)\n    and(kept, flag)\n  }, {items})",
        helpers: "",
        regression: false,
    },
    Shape {
        name: "filter_through_callable_parameter",
        result: "List[{T}]",
        body: "sift(keep, {items})",
        helpers: "def sift(p: ({T}) -> bool, xs: List[{T}]) -> List[{T}] =\n  filter(fn (x: {T}) -> p(x), xs)\n",
        regression: true,
    },
    Shape {
        name: "partition_callback_calls_named",
        result: "(List[{T}], List[{T}])",
        body: "partition(fn (x: {T}) -> keep(x), {items})",
        helpers: "",
        regression: true,
    },
    Shape {
        name: "partition_named_callback",
        result: "(List[{T}], List[{T}])",
        body: "partition(keep, {items})",
        helpers: "",
        regression: true,
    },
    Shape {
        name: "map_callback_calls_named",
        result: "List[{T}]",
        body: "map(fn (x: {T}) -> pass(x), {items})",
        helpers: "",
        regression: false,
    },
    Shape {
        name: "flat_map_callback_calls_named",
        result: "List[{T}]",
        body: "flat_map(fn (x: {T}) -> [pass(x)], {items})",
        helpers: "",
        regression: false,
    },
    Shape {
        name: "fold_callback_calls_named",
        result: "List[{T}]",
        body: "fold(fn (acc: List[{T}], x: {T}) -> append(acc, pass(x)), [], {items})",
        helpers: "",
        regression: false,
    },
];

fn instantiate(kind: &ItemKind, shape: &Shape) -> String {
    let fill = |text: &str| text.replace("{items}", kind.items).replace("{T}", kind.ty);
    format!(
        "{}{}def case(flag: bool) -> {} =\n  {}\na = case(true)\nb = case(false)\n",
        kind.prelude,
        fill(shape.helpers),
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

// REGRESSION TESTS, one per item kind so the kinds run in parallel. On
// `d029224fd` the five shapes marked `regression` failed to build with
// `owner %N ... is not live` for every item kind, 20 of the 36 cases. The
// others built and balanced there and lock the neighbouring lowering: a call
// bound inside the predicate's block, whose item use was never its last, and
// `map`, `flat_map` and `fold`, whose step does not read the item again.

#[test]
fn a_named_predicate_over_string_items_leaves_the_item_to_the_step() {
    every_shape_over("string");
}

#[test]
fn a_named_predicate_over_list_items_leaves_the_item_to_the_step() {
    every_shape_over("list_of_string");
}

#[test]
fn a_named_predicate_over_adt_items_leaves_the_item_to_the_step() {
    every_shape_over("adt_with_string");
}

#[test]
fn a_named_predicate_over_tensor_items_leaves_the_item_to_the_step() {
    every_shape_over("tensor");
}

/// Every declared kind has a test above, so a kind added to the table
/// cannot go unrun, and the table still holds its regression shapes.
#[test]
fn every_item_kind_has_a_test() {
    let source = include_str!("issue_2577_filter_named_predicate.rs");
    for kind in KINDS {
        assert!(
            source.contains(&format!("every_shape_over(\"{}\");", kind.name)),
            "item kind `{}` has no test",
            kind.name
        );
    }
    assert!(SHAPES.iter().any(|shape| shape.regression));
}
