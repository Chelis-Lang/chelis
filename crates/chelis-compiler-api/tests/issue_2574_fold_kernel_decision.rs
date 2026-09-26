//! chelis#2574: a `fold` inside a tensor-returning def body that is not the
//! body's head (a block whose tail is, or binds, the fold) must keep the def
//! on the host lane.
//!
//! Outside a differentiated body the kernel lowering has no `fold` arm: it
//! lowers to an unresolved `Load("fold")`, which the kernel decision then
//! refused after the fact ("the kernel body of `case` reaches the host-only
//! builtin `fold`"). The C lane fell through to host code (chelis#1515) and
//! built the program; the eval lane, which never falls through, rejected it.
//! The kernel decision (`chelis_ir::host::host_def_kernel`) must name the
//! class before lowering, so both lanes choose host code for the same reason.
//!
//! The callback's result is varied over every source a value can come from:
//! a block-local capture, a parameter, a top-level value, the accumulator and
//! the item, and a computation over the item. The head-position `fold` is the
//! control that already evaluated.

#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{
    EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use chelis_ir::host::{HostLoweringSession, host_def_kernel};

const ITEMS: &str = "[to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0])]";

/// A block body whose tail is the fold, with `y` bound in the block.
fn block_tail(callback_result: &str) -> String {
    format!(
        "def case(flag: bool) -> tensor[3, f32] = {{\n  \
         y = to_tensor([7.0, 8.0, 9.0])\n  \
         fold(fn (acc: tensor[3, f32], x: tensor[3, f32]) -> {callback_result}, \
         to_tensor([0.0, 0.0, 0.0]), {ITEMS})\n\
         }}\n\
         a = case(true)\n"
    )
}

/// A block body that binds the fold and returns the binding.
fn block_bound(callback_result: &str) -> String {
    format!(
        "def case(flag: bool) -> tensor[3, f32] = {{\n  \
         s = fold(fn (acc: tensor[3, f32], x: tensor[3, f32]) -> {callback_result}, \
         to_tensor([0.0, 0.0, 0.0]), {ITEMS})\n  \
         s\n\
         }}\n\
         a = case(true)\n"
    )
}

fn eval_root(source: &str) -> String {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|error| panic!("the evaluator rejected:\n{source}\n{error:?}"));
    root_display(&result, "a")
}

fn root_display(result: &EvalResult, name: &str) -> String {
    result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .and_then(|root| root.display.clone())
        .unwrap_or_else(|| panic!("no rendered root `{name}` in {:?}", result.roots))
}

fn kernel_decision(source: &str, name: &str) -> Result<bool, String> {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expansion")
    .into_exprs();
    let checked = chelis_types::check_ir_program(&exprs).expect("type check");
    let checked = chelis_effects::check_program(&checked).expect("effects");
    let checked = chelis_types::check_linearity(&checked).expect("linearity");
    host_def_kernel(&HostLoweringSession::new(&checked), name)
        .map(|kernel| kernel.is_some())
        .map_err(|diagnostic| diagnostic.to_string())
}

fn tensor(values: [f32; 3]) -> String {
    let data = values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("tensor(shape=[3], data=[{data}])")
}

#[test]
fn a_block_tail_fold_returning_a_block_local_capture_evaluates() {
    // The issue's witness.
    let source = block_tail("y");
    assert_eq!(kernel_decision(&source, "case"), Ok(false));
    assert_eq!(eval_root(&source), tensor([7.0, 8.0, 9.0]));
}

#[test]
fn every_callback_result_source_in_a_block_tail_fold_evaluates() {
    for (callback_result, expected) in [
        ("x", [4.0, 5.0, 6.0]),
        ("acc", [0.0, 0.0, 0.0]),
        ("(acc + x)", [5.0, 7.0, 9.0]),
        ("(x + x)", [8.0, 10.0, 12.0]),
        ("(y + x)", [11.0, 13.0, 15.0]),
    ] {
        let source = block_tail(callback_result);
        assert_eq!(kernel_decision(&source, "case"), Ok(false), "{source}");
        assert_eq!(eval_root(&source), tensor(expected), "{source}");
    }
}

#[test]
fn a_fold_bound_in_a_block_and_returned_evaluates() {
    let source = block_bound("(acc + x)");
    assert_eq!(kernel_decision(&source, "case"), Ok(false));
    assert_eq!(eval_root(&source), tensor([5.0, 7.0, 9.0]));
}

#[test]
fn a_block_fold_returning_a_parameter_evaluates() {
    let source = "def case(y: tensor[3, f32]) -> tensor[3, f32] = {\n  \
                  s = fold(fn (acc: tensor[3, f32], x: tensor[3, f32]) -> y, \
                  to_tensor([0.0, 0.0, 0.0]), [to_tensor([1.0, 2.0, 3.0])])\n  \
                  s\n\
                  }\n\
                  a = case(to_tensor([7.0, 8.0, 9.0]))\n";
    assert_eq!(kernel_decision(source, "case"), Ok(false));
    assert_eq!(eval_root(source), tensor([7.0, 8.0, 9.0]));
}

#[test]
fn a_block_fold_returning_a_top_level_value_evaluates() {
    let source = "g = to_tensor([7.0, 8.0, 9.0])\n\
                  def case(flag: bool) -> tensor[3, f32] = {\n  \
                  s = fold(fn (acc: tensor[3, f32], x: tensor[3, f32]) -> g, \
                  to_tensor([0.0, 0.0, 0.0]), [to_tensor([1.0, 2.0, 3.0])])\n  \
                  s\n\
                  }\n\
                  a = case(true)\n";
    assert_eq!(kernel_decision(source, "case"), Ok(false));
    assert_eq!(eval_root(source), tensor([7.0, 8.0, 9.0]));
}

/// Control: the head-position fold was already host code and evaluated.
#[test]
fn a_head_position_fold_stays_host_and_evaluates() {
    let source = format!(
        "def case(flag: bool) -> tensor[3, f32] = \
         fold(fn (acc: tensor[3, f32], x: tensor[3, f32]) -> (acc + x), \
         to_tensor([0.0, 0.0, 0.0]), {ITEMS})\n\
         a = case(true)\n"
    );
    assert_eq!(kernel_decision(&source, "case"), Ok(false));
    assert_eq!(eval_root(&source), tensor([5.0, 7.0, 9.0]));
}

/// A local seed and no capture at all: the block binds the seed, not the
/// callback's result.
#[test]
fn a_block_fold_over_a_local_seed_evaluates() {
    let source = format!(
        "def case(flag: bool) -> tensor[3, f32] = {{\n  \
         s0 = to_tensor([0.0, 0.0, 0.0])\n  \
         fold(fn (acc: tensor[3, f32], x: tensor[3, f32]) -> (acc + x), s0, {ITEMS})\n\
         }}\n\
         a = case(true)\n"
    );
    assert_eq!(kernel_decision(&source, "case"), Ok(false));
    assert_eq!(eval_root(&source), tensor([5.0, 7.0, 9.0]));
}

/// A fold reached through a callee: the decision walks the inlined callee.
#[test]
fn a_fold_reached_through_a_callee_evaluates() {
    let source = format!(
        "def g(flag: bool) -> tensor[3, f32] = {{\n  \
         s0 = to_tensor([0.0, 0.0, 0.0])\n  \
         fold(fn (acc: tensor[3, f32], x: tensor[3, f32]) -> (acc + x), s0, {ITEMS})\n\
         }}\n\
         def f(flag: bool) -> tensor[3, f32] = g(flag)\n\
         a = f(true)\n"
    );
    assert_eq!(kernel_decision(&source, "g"), Ok(false));
    assert_eq!(kernel_decision(&source, "f"), Ok(false));
    assert_eq!(eval_root(&source), tensor([5.0, 7.0, 9.0]));
}

/// Negative parity: a block body with no host-only form stays a kernel.
#[test]
fn a_block_body_without_a_fold_stays_a_kernel() {
    let source = "def case(flag: bool) -> tensor[3, f32] = {\n  \
                  y = to_tensor([7.0, 8.0, 9.0])\n  \
                  (y + y)\n\
                  }\n\
                  a = case(true)\n";
    assert_eq!(kernel_decision(source, "case"), Ok(true));
    assert_eq!(eval_root(source), tensor([14.0, 16.0, 18.0]));
}

/// The same def applied from an embedding with its tensor input bound: the
/// selected entry must evaluate, whichever evaluator the manifest routes it
/// to.
#[test]
fn a_bound_block_fold_entry_evaluates() {
    let source = "def case(y: tensor[3, f32]) -> tensor[3, f32] = {\n  \
                  s = fold(fn (acc: tensor[3, f32], x: tensor[3, f32]) -> (acc + y), \
                  to_tensor([0.0, 0.0, 0.0]), [to_tensor([1.0, 2.0, 3.0]), \
                  to_tensor([4.0, 5.0, 6.0])])\n  \
                  s\n\
                  }\n";
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::from([(
                "y".to_string(),
                TensorValue {
                    shape: vec![3],
                    data: wire_values::storage_f32(vec![7.0, 8.0, 9.0]),
                },
            )]),
        },
        &["case".to_string()],
    )
    .unwrap_or_else(|error| panic!("the bound entry was rejected: {error:?}"));
    let [root] = result.roots.as_slice() else {
        panic!("one root expected: {:?}", result.roots);
    };
    match &root.value {
        ExecutionValue::Tensor { value } => {
            assert_eq!(value.shape, [3]);
            assert_eq!(
                serde_json::to_value(&value.data).unwrap(),
                serde_json::to_value(wire_values::storage_f32(vec![14.0, 16.0, 18.0])).unwrap()
            );
        }
        other => panic!("the root is not a tensor: {other:?}"),
    }
}
