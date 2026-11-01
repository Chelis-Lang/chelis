//! chelis#4105, [05-OP-15]: lowering labels every `argmax_reduce` and
//! `argmin_reduce` result i64. A computed-reshape operand gives the checker's
//! result type symbolic placeholder extents, and the reduction lowering then
//! took the operand dtype; the compiled lane stored i32 indices and copied
//! them into an i64 buffer by byte count.
//!
//! These tests inspect the tensor helpers the compiled lane emits from
//! (`try_lower_compiled_program`), so every index reduction a native build
//! would see is covered, then require each helper DAG to verify.

use chelis_dag_core::dag::{Dag, RiscOp};
use chelis_dag_core::verify;
use chelis_ir::host::try_lower_compiled_program;
use chelis_numeric_core::prim::Prim;

fn checked_surf(source: &str) -> chelis_types::CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("Surf fixture parses");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture desugars"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("Surf fixture expands")
    .into_exprs();
    let checked = chelis_types::check_ir_program(&exprs)
        .unwrap_or_else(|report| panic!("Surf fixture checks: {:#?}", report.errors));
    let checked = chelis_effects::check_program(&checked).expect("Surf fixture effects check");
    chelis_types::check_linearity(&checked).expect("Surf fixture linearity check")
}

/// Every tensor-helper DAG the compiled lane lowers for `source`.
fn helper_dags(source: &str) -> Vec<Dag> {
    let compiled = try_lower_compiled_program(&checked_surf(source)).expect("host lowering");
    let mut dags: Vec<Dag> = compiled.dag.into_iter().collect();
    if let Some(host) = compiled.host {
        dags.extend(
            host.global_tensor_helpers
                .into_iter()
                .map(|helper| helper.dag),
        );
        for function in host.functions {
            dags.extend(function.tensor_helpers.into_iter().map(|helper| helper.dag));
        }
    }
    dags
}

/// `(op, operand dtype, result dtype)` for each node matching `filter`.
fn reductions(dags: &[Dag], filter: fn(&RiscOp) -> bool) -> Vec<(String, Prim, Prim)> {
    dags.iter()
        .flat_map(|dag| {
            assert_eq!(
                verify::verify(dag),
                Vec::<String>::new(),
                "every emitted helper verifies"
            );
            dag.nodes()
                .iter()
                .filter(|node| filter(&node.op))
                .map(|node| {
                    let operand = dag
                        .get(node.inputs[0])
                        .expect("reduction operand exists")
                        .output_type
                        .precision;
                    (
                        format!("{:?}", node.op),
                        operand,
                        node.output_type.precision,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

const COMPUTED: &str = "reshape(reshape(x, [floor_div(mul(shape(x, 0), shape(x, 1)), shape(x, 1)), shape(x, 1)]), [shape(x, 0), shape(x, 1)])";

fn program(op: &str, dtype: &str, result: &str, composed: bool, input: &str) -> String {
    let operand = if composed { COMPUTED } else { "x" };
    format!(
        "module Probe.Lowering\n\
         def pool[b, s](x: tensor[b, s, {dtype}]) -> tensor[b, {result}] = {{\n\
         \x20 y = cast({operand}, {dtype})\n\
         \x20 reshape({op}(y, 1), [shape(x, 0)])\n\
         }}\n\
         out = pool({input})\n"
    )
}

#[test]
fn index_reductions_lower_to_i64_for_direct_and_computed_reshape_operands() {
    for (dtype, input) in [
        ("i32", "to_tensor([[1, 9, 2], [2, 1, 9]], i32)"),
        ("i64", "to_tensor([[1i64, 9i64, 2i64], [2i64, 1i64, 9i64]])"),
        ("f32", "to_tensor([[1.0, 9.0, 2.0], [2.0, 1.0, 9.0]], f32)"),
    ] {
        for op in ["argmax_reduce", "argmin_reduce"] {
            for composed in [false, true] {
                let source = program(op, dtype, "i64", composed, input);
                let found = reductions(&helper_dags(&source), |op| {
                    matches!(op, RiscOp::Argmax { .. } | RiscOp::Argmin { .. })
                });
                assert!(!found.is_empty(), "{source}: no index reduction lowered");
                for (node, _operand, result) in found {
                    assert_eq!(
                        result,
                        Prim::Int64,
                        "{node} over {dtype} (composed={composed}) must yield i64 indices"
                    );
                }
            }
        }
    }
}

/// Negative control: the value reductions sharing the lowering arm keep the
/// operand dtype, so the index rule did not widen them.
#[test]
fn value_reductions_in_the_same_arm_keep_the_operand_dtype() {
    for op in ["min_reduce", "prod_reduce"] {
        for composed in [false, true] {
            let source = program(
                op,
                "i32",
                "i32",
                composed,
                "to_tensor([[1, 9, 2], [2, 1, 9]], i32)",
            );
            let found = reductions(&helper_dags(&source), |op| {
                matches!(op, RiscOp::MinReduce { .. } | RiscOp::ProdReduce { .. })
            });
            assert!(!found.is_empty(), "{source}: no value reduction lowered");
            for (node, operand, result) in found {
                assert_eq!(result, operand, "{node} (composed={composed})");
                assert_eq!(result, Prim::Int32, "{node} (composed={composed})");
            }
        }
    }
}
