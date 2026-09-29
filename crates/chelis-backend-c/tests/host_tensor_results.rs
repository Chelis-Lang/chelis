//! spec/08 §2 and [05-OP-31]: helper result tensors materialize at the
//! resolved host type, preserving dtype, rank and ownership.
mod support;
use chelis_ir::host::{
    ConcreteHostExpr as Expr, ConcreteHostExprKind as Kind, ConcreteHostFunction as Function,
    ConcreteHostParam as Param, ConcreteHostProgram as Program, HostFunctionOrigin,
    HostTensorHelper, HostTensorInput,
};
use chelis_ir::{ConcreteHostType as Ty, Dag, RiscOp, TensorType};
use chelis_types::types::Prim;
use std::{fs, process::Command};

fn identity_result(precision: Prim, result: Ty, dims: Vec<chelis_ir::DimInfo>) -> Program {
    let tensor = TensorType { dims, precision };
    let mut dag = Dag::new();
    let owner = dag.declare("identity");
    let root = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor.clone(),
        None,
    );
    dag.add_root(root);
    Program {
        functions: vec![Function {
            helper_result_claim_axes: vec![],
            name: "the_fn".into(),
            entry_contract: Default::default(),
            params: vec![Param {
                name: "x".into(),
                ty: Ty::Tensor(tensor.clone()),
            }],
            ret_ty: result.clone(),
            body: Expr::new(Kind::TensorCall {
                helper: 0,
                args: vec![Expr::new(Kind::Var("x".into(), Ty::Tensor(tensor.clone())))],
                ty: result,
            }),
            tensor_helpers: vec![HostTensorHelper {
                name: "identity".into(),
                dag,
                inputs: vec![HostTensorInput {
                    name: "x".into(),
                    ty: tensor.clone(),
                }],
                output: tensor,
                specialization: None,
                summary_rejection: None,
            }],
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: vec![],
        }],
        ..Program::default()
    }
}

fn run(program: Program, declaration: &str, assertion: &str, dtype: &str, bits: &str) {
    let generated = support::codegen_host_program(program, "results").expect("codegen");
    let dir = tempfile::tempdir().unwrap();
    let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();
    fs::write(dir.path().join("results.c"), &generated.c_source).unwrap();
    let input = if dtype == "CHELIS_DTYPE_KEY" {
        format!(
            "chelis_tensor *input = chelis_alloc(0, NULL, CHELIS_DTYPE_KEY); chelis_tensor_write *guard = chelis_tensor_begin_write(input); chelis_write_view view = chelis_tensor_write_view(guard); ((uint64_t *)view.data)[0] = {bits}; chelis_tensor_end_write(guard);"
        )
    } else {
        format!(
            "chelis_tensor *input = chelis_scalar_tensor(chelis_scalar_from_bits({dtype}, {bits}));"
        )
    };
    fs::write(
        dir.path().join("main.c"),
        format!(
            r#"
#include "chelis_runtime.h"
#include <stdint.h>
#include <assert.h>
{declaration}
int main(void) {{
    {input}
    {assertion}
    chelis_tensor_release(input);
    return 0;
}}
"#
        ),
    )
    .unwrap();
    let toolchain = chelis_backend_c::toolchain::test_toolchain(Default::default());
    let binary = dir.path().join("run");
    let compile = Command::new(toolchain.compiler)
        .current_dir(dir.path())
        .args([
            "-O2",
            "-fsanitize=address,undefined",
            "-Werror=incompatible-pointer-types",
            "results.c",
            "main.c",
        ])
        .args(toolchain.compile_flags)
        .arg(staged.archive)
        .args(toolchain.link_flags)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&compile.stderr),
        generated.c_source
    );
    let output = Command::new(binary)
        .env("ASAN_OPTIONS", "detect_leaks=0")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
}

#[test]
fn helper_scalar_results_preserve_every_exact_dtype() {
    for (precision, c_type, dtype, bits, expected) in [
        (Prim::Int8, "int8_t", "I8", "255", "-1"),
        (Prim::Int16, "int16_t", "I16", "65535", "-1"),
        (Prim::Int32, "int32_t", "I32", "2147483647", "2147483647"),
        (
            Prim::Int64,
            "int64_t",
            "I64",
            "9007199254740993ULL",
            "9007199254740993LL",
        ),
        (Prim::F16, "uint16_t", "F16", "0x8000", "0x8000"),
        (Prim::Bf16, "uint16_t", "BF16", "0x7fc1", "0x7fc1"),
        (Prim::F32, "float", "F32", "0x40400000", "3.0f"),
        (Prim::F64, "double", "F64", "0x4008000000000000ULL", "3.0"),
        (Prim::Bool, "bool", "BOOL", "1", "true"),
    ] {
        run(
            identity_result(precision, Ty::Scalar(precision), vec![]),
            &format!("extern {c_type} chelis_fn_7468655f666e(chelis_tensor *);"),
            &format!("assert(chelis_fn_7468655f666e(input) == {expected});"),
            &format!("CHELIS_DTYPE_{dtype}"),
            bits,
        );
    }
}

#[test]
fn rank_zero_tensor_result_keeps_its_owned_tensor_identity() {
    run(
        identity_result(
            Prim::F32,
            Ty::Tensor(TensorType {
                dims: vec![],
                precision: Prim::F32,
            }),
            vec![],
        ),
        "extern chelis_tensor *chelis_fn_7468655f666e(chelis_tensor *);",
        "chelis_tensor *output = chelis_fn_7468655f666e(input); assert(chelis_tensor_rank(output) == 0); assert(chelis_tensor_read_view(output).dtype == CHELIS_DTYPE_F32); chelis_tensor_release(output);",
        "CHELIS_DTYPE_F32",
        "0x40400000",
    );
}

#[test]
fn nonscalar_helper_cannot_materialize_as_a_scalar() {
    let result = support::codegen_host_program(
        identity_result(Prim::F32, Ty::Float32, vec![chelis_ir::DimInfo::Lit(1)]),
        "invalid_rank",
    );
    assert!(
        result.is_err(),
        "rank-one output must not silently become a scalar"
    );
}

#[test]
fn helper_result_dtype_mismatch_is_rejected() {
    let result = support::codegen_host_program(
        identity_result(Prim::F32, Ty::Float64, vec![]),
        "invalid_dtype",
    );
    assert!(
        result.is_err(),
        "helper dtype must agree with scalar result type"
    );
}

#[test]
fn mixed_tuple_results_materialize_fields_in_output_order() {
    let tensor = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };
    let result = Ty::Tuple(vec![Ty::Float32, Ty::Tensor(tensor.clone())]);
    let mut program = identity_result(Prim::F32, result, vec![]);
    let helper = &mut program.functions[0].tensor_helpers[0];
    let root = helper.dag.roots()[0];
    let owner = helper.dag.declare("second");
    let second = helper
        .dag
        .add_node(owner, RiscOp::Neg, vec![root], tensor, None);
    helper.dag.add_root(second);
    run(
        program,
        "extern chelis_tuple *chelis_fn_7468655f666e(chelis_tensor *);",
        "chelis_tuple *output = chelis_fn_7468655f666e(input); chelis_value first_value = chelis_tuple_get(output, 0); chelis_scalar first = chelis_value_unbox_scalar(first_value); assert(first.dtype == CHELIS_DTYPE_F32); chelis_value_release(first_value); chelis_tensor *second = chelis_tensor_take_value(chelis_tuple_get(output, 1)); assert(((const float *)chelis_tensor_read_view(second).data)[0] == -3.0f); chelis_tensor_release(second); chelis_tuple_release(output);",
        "CHELIS_DTYPE_F32",
        "0x40400000",
    );
}

#[test]
fn singleton_tuple_is_not_replaced_by_its_only_tensor() {
    run(
        identity_result(Prim::F32, Ty::Tuple(vec![Ty::Float32]), vec![]),
        "extern chelis_tuple *chelis_fn_7468655f666e(chelis_tensor *);",
        "chelis_tuple *output = chelis_fn_7468655f666e(input); chelis_value value = chelis_tuple_get(output, 0); chelis_scalar scalar = chelis_value_unbox_scalar(value); assert(scalar.dtype == CHELIS_DTYPE_F32); chelis_value_release(value); chelis_tuple_release(output);",
        "CHELIS_DTYPE_F32",
        "0x40400000",
    );
}

#[test]
fn helper_result_arity_mismatch_is_rejected() {
    let result = support::codegen_host_program(
        identity_result(Prim::F32, Ty::Tuple(vec![Ty::Float32, Ty::Float32]), vec![]),
        "invalid_arity",
    );
    assert!(
        result.is_err(),
        "one output cannot initialize two tuple fields"
    );
}

#[test]
fn scalar_key_result_keeps_all_key_bits() {
    run(
        identity_result(Prim::Key, Ty::Scalar(Prim::Key), vec![]),
        "extern chelis_key chelis_fn_7468655f666e(chelis_tensor *);",
        "assert(chelis_fn_7468655f666e(input).bits == UINT64_MAX);",
        "CHELIS_DTYPE_KEY",
        "UINT64_MAX",
    );
}
