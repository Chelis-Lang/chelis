//! Host-emit dtype dispatch and exact scalar transport.
//!
//! Elementwise operations use dtype-typed guarded/read views. Scalar inputs
//! to tensor helpers use the tagged scalar carrier and its exact constructor,
//! preserving the dtype and stored bits without hand-written buffer packing.
//! Source checks lock those ABI boundaries; CLI standard-lowering tests also
//! execute bool, signed-integer, and float scalar inputs through generated C.

use std::fs;
use std::process::Command;

mod support;
use chelis_ir::ConcreteHostType as HostType;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::host::{
    ConcreteHostBinding as HostBinding, ConcreteHostExpr as HostExpr,
    ConcreteHostExprKind as HostExprKind, ConcreteHostFunction as HostFunction,
    ConcreteHostParam as HostParam, ConcreteHostProgram as HostProgram, HostFunctionOrigin,
    HostTensorHelper, HostTensorInput,
};
use chelis_types::types::Prim;
use support::emit_host_program;

mod common;

fn vec_ty(n: usize, prim: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: prim,
    }
}

fn make_binary_program(op_name: &str, prim: Prim) -> HostProgram {
    let tt = vec_ty(4, prim);
    let body = HostExpr::new(HostExprKind::Builtin {
        name: op_name.to_string(),
        args: vec![
            HostExpr::new(HostExprKind::Var(
                "a".to_string(),
                HostType::Tensor(tt.clone()),
            )),
            HostExpr::new(HostExprKind::Var(
                "b".to_string(),
                HostType::Tensor(tt.clone()),
            )),
        ],
        ty: HostType::Tensor(tt.clone()),
    });
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "the_fn".to_string(),
            params: vec![
                HostParam {
                    name: "a".to_string(),
                    ty: HostType::Tensor(tt.clone()),
                },
                HostParam {
                    name: "b".to_string(),
                    ty: HostType::Tensor(tt.clone()),
                },
            ],
            ret_ty: HostType::Tensor(tt),
            body,
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
        adt_layouts: Vec::new(),
    }
}

fn make_unary_program(op_name: &str, prim: Prim) -> HostProgram {
    let tt = vec_ty(4, prim);
    let body = HostExpr::new(HostExprKind::Builtin {
        name: op_name.to_string(),
        args: vec![HostExpr::new(HostExprKind::Var(
            "a".to_string(),
            HostType::Tensor(tt.clone()),
        ))],
        ty: HostType::Tensor(tt.clone()),
    });
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "the_fn".to_string(),
            params: vec![HostParam {
                name: "a".to_string(),
                ty: HostType::Tensor(tt.clone()),
            }],
            ret_ty: HostType::Tensor(tt),
            body,
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
        adt_layouts: Vec::new(),
    }
}

fn make_tensor_to_scalar_program(prim: Prim, scalar_ty: HostType) -> HostProgram {
    let tensor_ty = TensorType {
        dims: Vec::new(),
        precision: prim,
    };
    let body = HostExpr::new(HostExprKind::Builtin {
        name: "tensor_to_scalar".to_string(),
        args: vec![HostExpr::new(HostExprKind::Var(
            "input".to_string(),
            HostType::Tensor(tensor_ty.clone()),
        ))],
        ty: scalar_ty.clone(),
    });
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "the_fn".to_string(),
            params: vec![HostParam {
                name: "input".to_string(),
                ty: HostType::Tensor(tensor_ty),
            }],
            ret_ty: scalar_ty,
            body,
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
        adt_layouts: Vec::new(),
    }
}

fn make_scalar_to_tensor_program() -> HostProgram {
    let tensor_ty = TensorType {
        dims: Vec::new(),
        precision: Prim::Int64,
    };
    let body = HostExpr::new(HostExprKind::Builtin {
        name: "scalar_to_tensor".to_string(),
        args: vec![HostExpr::new(HostExprKind::Var(
            "input".to_string(),
            HostType::Int64,
        ))],
        ty: HostType::Tensor(tensor_ty.clone()),
    });
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "the_fn".to_string(),
            params: vec![HostParam {
                name: "input".to_string(),
                ty: HostType::Int64,
            }],
            ret_ty: HostType::Tensor(tensor_ty),
            body,
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
        adt_layouts: Vec::new(),
    }
}

#[test]
fn tensor_to_scalar_i64_never_round_trips_through_f64() {
    let source = emit_host_program(
        &make_tensor_to_scalar_program(Prim::Int64, HostType::Int64),
        "tensor_to_scalar_i64_exact",
    )
    .expect("rank-zero i64 extraction must emit");

    assert!(
        source.contains("chelis_host_scalar_as_i64("),
        "i64 tensor_to_scalar must read back through the dtype-checked \
         exact scalar reader of the tagged-carrier ABI:\n{source}"
    );
    assert!(
        source.contains(
            "CHELIS_DTYPE_I64: { int64_t out; memcpy(&out, &value.bits, sizeof out); return out; }"
        ),
        "the exact reader must recover i64 bits at their declared width:\n{source}"
    );
    assert!(
        !source.contains("__result = chelis_tensor_to_f64("),
        "i64 tensor_to_scalar must not pass through double:\n{source}"
    );
}

#[test]
fn tensor_to_scalar_f64_keeps_the_float_extractor() {
    let source = emit_host_program(
        &make_tensor_to_scalar_program(Prim::F64, HostType::Float64),
        "tensor_to_scalar_f64",
    )
    .expect("rank-zero f64 extraction must emit");

    assert!(
        source.contains("chelis_host_scalar_as_float("),
        "f64 tensor_to_scalar must keep the floating reader:\n{source}"
    );
    assert!(
        source.contains("CHELIS_DTYPE_F64"),
        "the floating reader must be dtype-checked at F64:\n{source}"
    );
}

#[test]
fn scalar_to_tensor_i64_uses_exact_i64_storage() {
    let source = emit_host_program(
        &make_scalar_to_tensor_program(),
        "scalar_to_tensor_i64_exact",
    )
    .expect("rank-zero i64 packing must emit");

    assert!(
        source.contains("chelis_host_scalar_from_i64("),
        "i64 scalar_to_tensor must pack through the tagged exact-width \
         scalar of the tagged-carrier ABI:\n{source}"
    );
    assert!(
        source.contains("chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)value)"),
        "the packer must tag i64 bits at their declared width:\n{source}"
    );
    assert!(
        !source.contains("__result = chelis_scalar_tensor_from_i64("),
        "i64 scalar_to_tensor must not call the legacy I32 storage helper:\n{source}"
    );
}

fn make_checked_tensor_cast_program(source: Prim, target: Prim) -> HostProgram {
    let source_ty = vec_ty(2, source);
    let target_ty = vec_ty(2, target);
    let body = HostExpr::new(HostExprKind::Builtin {
        name: "cast".to_string(),
        args: vec![HostExpr::new(HostExprKind::Var(
            "input".to_string(),
            HostType::Tensor(source_ty.clone()),
        ))],
        ty: HostType::Tensor(target_ty.clone()),
    });
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "the_fn".to_string(),
            params: vec![HostParam {
                name: "input".to_string(),
                ty: HostType::Tensor(source_ty),
            }],
            ret_ty: HostType::Tensor(target_ty),
            body,
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
        adt_layouts: Vec::new(),
    }
}

#[test]
fn checked_tensor_cast_host_emission_covers_the_active_product_without_fallback_identity() {
    let active = [
        Prim::F64,
        Prim::F32,
        Prim::F16,
        Prim::Bf16,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
    ];

    for source in active {
        for target in active {
            let program = make_checked_tensor_cast_program(source, target);
            let generated = emit_host_program(
                &program,
                &format!("checked_cast_{}_to_{}", source.name(), target.name()),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "active checked-cast pair {} -> {} must emit: {error}",
                    source.name(),
                    target.name()
                )
            });
            let marker = format!(
                "/* checked cast plan: {} -> {} */",
                source.name(),
                target.name()
            );
            assert!(
                generated.contains(&marker),
                "host emission must consume the exact plan for {} -> {}; generated:\n{generated}",
                source.name(),
                target.name()
            );
            assert_eq!(
                generated.contains("/* checked cast identity */"),
                source == target,
                "host identity is legal exactly on the equal-Prim diagonal: {} -> {}; generated:\n{generated}",
                source.name(),
                target.name()
            );
        }
    }
}

fn generated_dtype_arm<'a>(source: &'a str, dtype_macro: &str) -> &'a str {
    let marker = format!("case {dtype_macro}: {{");
    let (_, rest) = source
        .split_once(&marker)
        .unwrap_or_else(|| panic!("generated C contains no `{marker}` arm:\n{source}"));
    let end = rest
        .find("case CHELIS_")
        .or_else(|| rest.find("default:"))
        .unwrap_or(rest.len());
    &rest[..end]
}

fn compile_generated_i32_binary_assignment(op_name: &str, lhs: i32, rhs: i32) -> i32 {
    let program = make_binary_program(op_name, Prim::Int32);
    let source = emit_host_program(&program, &format!("{op_name}_i32_exact")).unwrap();
    let arm = generated_dtype_arm(&source, "CHELIS_DTYPE_I32");
    let assignment = arm
        .lines()
        .find(|line| line.contains("__target_data[i] ="))
        .unwrap_or_else(|| panic!("the i32 arm contains no assignment:\n{arm}"))
        .trim();

    let c_source = format!(
        r#"#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {{
    int32_t target_storage[1] = {{0}};
    const int32_t lhs_storage[1] = {{{lhs}}};
    const int32_t rhs_storage[1] = {{{rhs}}};
    int32_t *__target_data = target_storage;
    const int32_t *__lhs_data = lhs_storage;
    const int32_t *__rhs_data = rhs_storage;
    int i = 0;
    int idx_lhs = 0;
    int idx_rhs = 0;
    {assignment}
    printf("%" PRId32 "\n", __target_data[0]);
    return 0;
}}
"#
    );

    let probe = common::probe_dir(&format!("host_emit_{op_name}"));
    let temp_dir = probe.path().to_path_buf();
    let source_path = temp_dir.join("probe.c");
    let binary_path = temp_dir.join("probe");
    fs::write(&source_path, c_source).unwrap();

    let compile = Command::new("gcc")
        .args(["-std=c11", "-O0"])
        .arg(&source_path)
        .args(["-lm", "-o"])
        .arg(&binary_path)
        .output()
        .unwrap_or_else(|error| panic!("failed to run gcc: {error}"));
    assert!(
        compile.status.success(),
        "generated i32 assignment failed to compile:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let output = Command::new(&binary_path).output().unwrap();
    assert!(
        output.status.success(),
        "generated i32 assignment failed to run:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

// ---- L1587 binary operator ---------------------------------------------

#[test]
fn binary_elementwise_emits_dtype_switch_at_f32() {
    let program = make_binary_program("add", Prim::F32);
    let src = emit_host_program(&program, "binop_f32").unwrap();
    assert!(
        src.contains("switch (") && src.contains(".dtype)"),
        "binary elementwise must emit an outer switch on dtype; got:\n{src}"
    );
    assert!(
        src.contains("(float*)") && src.contains(".data"),
        "binary elementwise must cast view data through a typed pointer for f32; got:\n{src}"
    );
    assert!(
        !src.contains("    {target}->data[i] ="),
        "raw `{{target}}->data[i]` template token must be substituted; got:\n{src}"
    );
    // The legacy untyped pattern `t->data[idx] OP t->data[idx]` reads
    // 4 bytes as float. The new code paths cast view data to a typed
    // pointer before indexing.
    let buggy_pattern = "->data[i] = ";
    let bare_lhs_rhs = src.lines().any(|l| {
        l.contains(buggy_pattern)
            && !l.contains("(float*)")
            && !l.contains("(double*)")
            && !l.contains("(int64_t*)")
    });
    assert!(
        !bare_lhs_rhs,
        "binary elementwise must not emit bare `->data[i] = ...` without a typed cast; got:\n{src}"
    );
}

#[test]
fn binary_elementwise_emits_dtype_switch_at_f64() {
    let program = make_binary_program("add", Prim::F64);
    let src = emit_host_program(&program, "binop_f64").unwrap();
    assert!(
        src.contains("(double*)") && src.contains(".data"),
        "binary elementwise at f64 must cast view data through `(double*)`; got:\n{src}"
    );
}

#[test]
fn binary_elementwise_emits_dtype_switch_at_i64() {
    let program = make_binary_program("add", Prim::Int64);
    let src = emit_host_program(&program, "binop_i64").unwrap();
    assert!(
        src.contains("(int64_t*)") && src.contains(".data"),
        "binary elementwise at i64 must cast view data through `(int64_t*)`; got:\n{src}"
    );
}

#[test]
fn binary_elementwise_int32_arm_uses_int32_t_pointers() {
    let program = make_binary_program("add", Prim::Int32);
    let src = emit_host_program(&program, "binop_i32").unwrap();
    let arm = generated_dtype_arm(&src, "CHELIS_DTYPE_I32");

    assert!(
        arm.contains("int32_t *__target_data = (int32_t*)"),
        "the i32 target pointer must use int32_t; arm:\n{arm}"
    );
    assert!(
        arm.contains("const int32_t *__lhs_data = (const int32_t*)")
            && arm.contains("const int32_t *__rhs_data = (const int32_t*)"),
        "the i32 input pointers must use int32_t; arm:\n{arm}"
    );
}

#[test]
fn binary_elementwise_int32_arm_contains_no_float_pointer() {
    let program = make_binary_program("add", Prim::Int32);
    let src = emit_host_program(&program, "binop_i32_no_float").unwrap();
    let arm = generated_dtype_arm(&src, "CHELIS_DTYPE_I32");

    assert!(
        !arm.contains("float"),
        "the i32 arm must not use an IEEE binary32 pointer; arm:\n{arm}"
    );
}

#[test]
fn binary_elementwise_f32_arm_keeps_float_pointers() {
    let program = make_binary_program("add", Prim::F32);
    let src = emit_host_program(&program, "binop_f32_control").unwrap();
    let arm = generated_dtype_arm(&src, "CHELIS_DTYPE_F32");

    assert!(
        arm.contains("float *__target_data = (float*)")
            && arm.contains("const float *__lhs_data = (const float*)")
            && arm.contains("const float *__rhs_data = (const float*)"),
        "the f32 arm must keep float pointers; arm:\n{arm}"
    );
}

// ---- L1623 binary func -------------------------------------------------

#[test]
fn binary_func_elementwise_emits_typed_pointer_access() {
    let program = make_binary_program("max_elem", Prim::F32);
    let src = emit_host_program(&program, "binfunc_f32").unwrap();
    // Direct extrema selection requires an f32-typed access pattern. The
    // migrated emission must cast `->data` to a typed pointer before indexing
    // rather than reading through the public `float *data` field declaration
    // unconditionally.
    assert!(
        src.contains("(float*)") || src.contains("(const float*)"),
        "binary func elementwise must cast `->data` to a typed pointer; got:\n{src}"
    );
    assert!(
        !src.lines().any(|l| l.contains("->data[i] = ")
            && !l.contains("(float*)")
            && !l.contains("(double*)")
            && !l.contains("(int64_t*)")),
        "binary func elementwise must not emit a bare `->data[i]` assignment; got:\n{src}"
    );
}

#[test]
fn binary_func_f32_extrema_use_exact_first_operand_selectors() {
    for (op, comparison, forbidden) in [
        ("max_elem", ">=", ["fmaxf(", "fmax("]),
        ("min_elem", "<=", ["fminf(", "fmin("]),
    ] {
        let program = make_binary_program(op, Prim::F32);
        let src = emit_host_program(&program, "binfunc_f32_extrema").unwrap();
        let arm = generated_dtype_arm(&src, "CHELIS_DTYPE_F32");

        assert!(arm.contains("isnan(__lhs_data[idx_lhs])"), "{arm}");
        assert!(arm.contains("!isnan(__rhs_data[idx_rhs])"), "{arm}");
        assert!(
            arm.contains(&format!(
                "__lhs_data[idx_lhs] {comparison} __rhs_data[idx_rhs]"
            )),
            "{arm}"
        );
        assert!(
            arm.contains("? __lhs_data[idx_lhs] : __rhs_data[idx_rhs]"),
            "{arm}"
        );
        for function in forbidden {
            assert!(
                !arm.contains(function),
                "direct extrema must select an operand, not call {function}:\n{arm}"
            );
        }
    }
}

#[test]
fn binary_func_int32_max_preserves_values_above_f32_exact_range() {
    assert_eq!(
        compile_generated_i32_binary_assignment("max_elem", 16_777_217, 0),
        16_777_217
    );
}

#[test]
fn binary_func_int32_min_preserves_values_below_f32_exact_range() {
    assert_eq!(
        compile_generated_i32_binary_assignment("min_elem", -16_777_217, 0),
        -16_777_217
    );
}

// ---- L1649 unary operator ----------------------------------------------

#[test]
fn unary_elementwise_emits_dtype_switch_at_f32() {
    let program = make_unary_program("neg", Prim::F32);
    let src = emit_host_program(&program, "unop_f32").unwrap();
    assert!(
        src.contains("switch (") && src.contains(".dtype)"),
        "unary elementwise must emit an outer switch on dtype; got:\n{src}"
    );
    assert!(
        src.contains("(float*)"),
        "unary elementwise must cast view data to a typed pointer; got:\n{src}"
    );
}

#[test]
fn unary_elementwise_emits_dtype_switch_at_f64() {
    let program = make_unary_program("neg", Prim::F64);
    let src = emit_host_program(&program, "unop_f64").unwrap();
    assert!(
        src.contains("(double*)"),
        "unary elementwise at f64 must cast `->data` through `(double*)`; got:\n{src}"
    );
}

#[test]
fn unary_elementwise_emits_dtype_switch_at_i64() {
    let program = make_unary_program("neg", Prim::Int64);
    let src = emit_host_program(&program, "unop_i64").unwrap();
    assert!(
        src.contains("(int64_t*)"),
        "unary elementwise at i64 must cast `->data` through `(int64_t*)`; got:\n{src}"
    );
}

// ---- L1675 unary func --------------------------------------------------

#[test]
fn unary_func_elementwise_emits_typed_pointer_access() {
    let program = make_unary_program("exp", Prim::F32);
    let src = emit_host_program(&program, "unfunc_f32").unwrap();
    assert!(
        src.contains("(float*)") || src.contains("(const float*)"),
        "binary32 unary functions must use float pointers; got:\n{src}"
    );
}

/// chelis#2734: a host elementwise loop over a dtype none of its arms covers
/// is refused before emission with a typed diagnostic, never emitted with a
/// default arm that aborts when the program runs. Host lowering runs these
/// operations in the tensor lane, which covers every admitted dtype.
#[test]
fn an_elementwise_loop_outside_its_arms_is_refused_before_emission() {
    for (program, name) in [
        (make_unary_program("exp", Prim::Int32), "exp"),
        (make_unary_program("exp", Prim::F64), "exp"),
        (make_unary_program("sin", Prim::F16), "sin"),
        (make_unary_program("sqrt", Prim::Bf16), "sqrt"),
        (make_binary_program("add", Prim::F16), "add"),
        (make_binary_program("mul", Prim::Bf16), "mul"),
        (make_binary_program("sub", Prim::Int8), "sub"),
        (make_binary_program("add", Prim::Int16), "add"),
    ] {
        let error = emit_host_program(&program, &format!("{name}_refused"))
            .expect_err("a dtype outside the loop's arms must be refused");
        let message = error.to_string();
        assert!(
            message.contains(&format!("`{name}` over a"))
                && message.contains("has no arm for this dtype"),
            "{name}: {message}"
        );
    }
}

/// The negative twin: every dtype an arm covers still emits its loop, with
/// no run-time abort arm for a named dtype.
#[test]
fn an_elementwise_loop_inside_its_arms_still_emits() {
    for prim in [Prim::F32, Prim::F64, Prim::Int32, Prim::Int64] {
        let program = make_binary_program("add", prim);
        let source = emit_host_program(&program, "add_admitted").unwrap();
        assert!(source.contains("switch ("), "{prim:?}: {source}");
    }
    let program = make_unary_program("exp", Prim::F32);
    let source = emit_host_program(&program, "exp_admitted").unwrap();
    assert!(
        source.contains("chelis_cr_expf") && !source.contains("unsupported for dtype"),
        "{source}"
    );
}

// ---- Exact scalar-to-tensor helper arguments -------------------------
// `assign_tensor_call` transports a scalar through chelis_scalar, then
// constructs the rank-zero tensor with the same dtype and stored bits.

fn make_tensor_call_with_scalar_arg(scalar_ty: HostType, scalar_val: HostExpr) -> HostProgram {
    // The host-side fallback path uses `HostExprKind::TensorCall` to
    // bind through a tensor helper.  We construct a TensorCall against
    // a synthetic helper (index 0) whose input is the scalar-coerced
    // tensor, then read the body through a top-level binding to drive
    // emission.  The verified backend boundary requires the referenced
    // helper to exist, so this fixture carries the exact rank-zero identity
    // helper consumed by the call.
    let precision = match scalar_ty {
        HostType::Bool => Prim::Bool,
        HostType::Float64 => Prim::F64,
        HostType::Int64 => Prim::Int64,
        ref other => panic!("unsupported scalar helper fixture type: {other:?}"),
    };
    let tt = TensorType {
        dims: Vec::new(),
        precision,
    };
    let mut helper_dag = Dag::new();
    let helper_dag_decl = helper_dag.declare("test");
    let helper_root = helper_dag.add_node(
        helper_dag_decl,
        RiscOp::Load {
            name: "input".into(),
        },
        Vec::new(),
        tt.clone(),
        None,
    );
    helper_dag.add_root(helper_root);
    let helper = HostTensorHelper {
        name: "scalar_identity".into(),
        dag: helper_dag,
        inputs: vec![HostTensorInput {
            name: "input".into(),
            ty: tt.clone(),
        }],
        output: tt.clone(),
        specialization: None,
        summary_rejection: None,
    };
    let call = HostExpr::new(HostExprKind::TensorCall {
        helper: 0,
        args: vec![scalar_val],
        ty: HostType::Tensor(tt.clone()),
    });
    HostProgram {
        globals: vec![HostBinding {
            name: "result".to_string(),
            display_name: Some("result".to_string()),
            display_roots: Vec::new(),
            ty: HostType::Tensor(tt.clone()),
            value: call,
        }],
        global_tensor_helpers: vec![helper],
        functions: Vec::new(),
        summary_rejections: Vec::new(),
        adt_layouts: Vec::new(),
    }
}

#[test]
fn to_tensor_list_ingress_uses_only_the_exact_registered_constructor() {
    let list_ty = HostType::List(Box::new(HostType::Float64));
    let tensor_ty = vec_ty(2, Prim::F64);
    let list = HostExpr::new(HostExprKind::List(
        vec![
            HostExpr::new(HostExprKind::Float(1.0)),
            HostExpr::new(HostExprKind::Float(2.0)),
        ],
        list_ty,
    ));
    let to_tensor = HostExpr::new(HostExprKind::Builtin {
        name: "to_tensor".to_string(),
        args: vec![list],
        ty: HostType::Tensor(tensor_ty.clone()),
    });
    let program = HostProgram {
        globals: vec![HostBinding {
            name: "result".to_string(),
            display_name: None,
            display_roots: Vec::new(),
            ty: HostType::Tensor(tensor_ty),
            value: to_tensor,
        }],
        global_tensor_helpers: Vec::new(),
        functions: Vec::new(),
        summary_rejections: Vec::new(),
        adt_layouts: Vec::new(),
    };

    let source = emit_host_program(&program, "exact_list_ingress").unwrap();
    assert!(
        source.contains("chelis_tensor_from_values(") && source.contains("CHELIS_DTYPE_F64"),
        "to_tensor must emit the registered exact tagged constructor:\n{source}"
    );
    assert!(
        source.contains("chelis_list_release(__arg0_"),
        "the exact constructor borrows its fresh list-literal argument, so the temporary must be released:\n{source}"
    );
    for retired in [
        "chelis_tensor_from_value_list_typed(",
        "chelis_tensor_from_value_list(",
    ] {
        assert!(
            !source.contains(retired),
            "to_tensor restored retired constructor `{retired}`:\n{source}"
        );
    }
}

#[test]
fn scalar_to_tensor_coercion_bool_uses_exact_tagged_carrier() {
    let program =
        make_tensor_call_with_scalar_arg(HostType::Bool, HostExpr::new(HostExprKind::Bool(true)));
    let src = emit_host_program(&program, "scalar_bool").unwrap();
    assert!(
        src.contains("= chelis_scalar_tensor(chelis_scalar_from_bits(CHELIS_DTYPE_BOOL,")
            && src.contains("__tensor_scalar0_0 ? 1 : 0"),
        "bool helper input must use the exact tagged Bool8 carrier:\n{src}"
    );
    assert!(
        !src.lines().any(|line| line.contains("? 1.0f : 0.0f")),
        "bool helper input must not use four-byte float storage:\n{src}"
    );
}

#[test]
fn scalar_to_tensor_coercion_f64_preserves_tag_and_bits() {
    let program = make_tensor_call_with_scalar_arg(
        HostType::Float64,
        HostExpr::new(HostExprKind::Float(7.5)),
    );
    let src = emit_host_program(&program, "scalar_f64").unwrap();
    assert!(
        src.contains("= chelis_scalar_tensor(chelis_scalar_from_bits(CHELIS_DTYPE_F64, chelis_host_f64_bits(__tensor_scalar0_0)))"),
        "f64 helper input must retain its F64 tag and bits (#381):\n{src}"
    );
    assert!(
        !src.lines()
            .any(|line| line.contains("_write.data)[0] = (float)(")),
        "f64 helper input must not narrow through a hand-packed f32 buffer:\n{src}"
    );
}

#[test]
fn scalar_to_tensor_coercion_int64_preserves_tag_and_integer_bits() {
    let program = make_tensor_call_with_scalar_arg(
        HostType::Int64,
        HostExpr::new(HostExprKind::Int(9_007_199_254_740_993)),
    );
    let src = emit_host_program(&program, "scalar_i64").unwrap();
    assert!(
        src.contains("= chelis_scalar_tensor(chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t)__tensor_scalar0_0))"),
        "i64 helper input must preserve its tag and all integer bits:\n{src}"
    );
    assert!(
        !src.contains("chelis_host_f64_bits(__tensor_scalar0_0)"),
        "i64 helper input must not round-trip through f64:\n{src}"
    );
}

// ---- logical operators over tensor operands ----------------------------

/// `op(a, b)` where `a` is a bool tensor and `b` is a bool scalar.
fn make_mixed_logical_program(op_name: &str) -> HostProgram {
    let mut program = make_binary_program(op_name, Prim::Bool);
    let function = &mut program.functions[0];
    function.params[1].ty = HostType::Bool;
    let tt = vec_ty(4, Prim::Bool);
    function.body = HostExpr::new(HostExprKind::Builtin {
        name: op_name.to_string(),
        args: vec![
            HostExpr::new(HostExprKind::Var(
                "a".to_string(),
                HostType::Tensor(tt.clone()),
            )),
            HostExpr::new(HostExprKind::Var("b".to_string(), HostType::Bool)),
        ],
        ty: HostType::Tensor(tt),
    });
    program
}

// A logical operation over two tensors computed on the host combines them
// element by element through typed bool storage, after the operand agreement
// check, rather than combining the two `chelis_tensor *` with C's scalar
// `&&` or `||`.
#[test]
fn logical_binary_over_tensor_operands_is_elementwise_in_host_emission() {
    for (op, c_op) in [("and", "&&"), ("or", "||")] {
        let program = make_binary_program(op, Prim::Bool);
        let src = emit_host_program(&program, "logical_tensor").unwrap();
        assert!(
            src.contains("chelis_host_require_elementwise_agreement(")
                && src.contains("(uint8_t*)")
                && src.contains(&format!("__lhs_data[idx_lhs] {c_op} __rhs_data[idx_rhs]")),
            "{op}: tensor operands must combine element by element; got:\n{src}"
        );
        assert!(
            !src.lines()
                .any(|line| line.contains(c_op) && line.contains("__arg")),
            "{op}: tensor pointers must never meet a scalar `{c_op}`; got:\n{src}"
        );
    }
}

// The negative twin: an operand pair the tensor arm does not take reaches the
// scalar arm, which refuses any tensor operand with a typed unsupported
// ([04-TOT-2]) rather than writing the pointer combination.
#[test]
fn logical_binary_with_one_tensor_operand_is_refused_in_host_emission() {
    for op in ["and", "or"] {
        let error = emit_host_program(&make_mixed_logical_program(op), "logical_mixed")
            .expect_err("a tensor operand must not reach scalar `&&`/`||`");
        let rendered = format!("{error}");
        assert!(
            rendered.contains(&format!("builtin `{op}`"))
                && rendered.contains("tensor operands in `chelis build` host emission"),
            "{op}: {rendered}"
        );
    }
}

// `not` keeps its elementwise tensor arm, which reads and writes the bool
// storage through typed pointers.
#[test]
fn logical_not_over_a_tensor_operand_keeps_its_elementwise_arm() {
    let program = make_unary_program("not", Prim::Bool);
    let src = emit_host_program(&program, "not_tensor").unwrap();
    assert!(
        src.contains("(uint8_t*)") && src.contains("= !__"),
        "tensor `not` must negate each element through typed bool storage; got:\n{src}"
    );
}
