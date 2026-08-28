//! Host-emit elementwise code-generation dtype-dispatch pin.
//!
//! W2 PR 3 of the 0.7.8 compiler cleanup workstream
//! (`CRuntime-F32Coupling`). Locks the invariant that the six
//! `host_emit.rs` code-generation sites listed below emit C code
//! that accesses `chelis_tensor->data` through dtype-typed pointers
//! (e.g. `((double*)t->data)[i]`) rather than the legacy untyped
//! `t->data[i]` form. The legacy form is a 4-byte float load
//! regardless of dtype against the public `float *data` declaration
//! in `crates/chelis-runtime/include/chelis_runtime.h`, mirroring
//! the bug class closed by PR #64 (CastMemcpy), PR #67
//! (ReshapeMemcpy), and PR #72 (PrintTensorF64).
//!
//! Migrated sites (line numbers in `crates/chelis-backend-c/src/host_emit.rs`):
//!   * L1587 elementwise binary operator (`add`, `sub`, `mul`, `div`).
//!   * L1623 elementwise binary func (direct extrema selectors).
//!   * L1649 elementwise unary operator (`neg`, `not`).
//!   * L1675 elementwise unary func (`expf`, `logf`, `sinf`, ...).
//!   * L1749/L1753/L1757 scalar-to-tensor coercion arms for int64,
//!     bool, and f32 helper-call inputs respectively.
//!
//! Each fixture builds a tiny `HostProgram` whose body forces the
//! emitter through one of the six sites, then asserts the generated
//! C source contains the expected typed-pointer pattern (the new
//! shape) and does NOT contain the unmigrated bare-data pattern.
//!
//! Diagnosis: `docs/investigations/c_runtime_dtype_accessors_diagnosis.md`.
//! Spec lock: `docs/design/compiler_cleanup_0_7_8_spec_lock.md` Contract 2.
//! §5 entry: `docs/gap_synthesis.md` `CRuntime-F32Coupling`.

use std::fs;
use std::process::Command;

use chelis_backend_c::host_emit::emit_host_program;
use chelis_ir::ConcreteHostType as HostType;
use chelis_ir::dag::{DimInfo, TensorType};
use chelis_ir::host::{
    ConcreteHostBinding as HostBinding, ConcreteHostExpr as HostExpr,
    ConcreteHostExprKind as HostExprKind, ConcreteHostFunction as HostFunction,
    ConcreteHostParam as HostParam, ConcreteHostProgram as HostProgram, HostFunctionOrigin,
};
use chelis_types::types::Prim;

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
    }
}

#[test]
fn tensor_to_scalar_i64_never_round_trips_through_f64() {
    let source = emit_host_program(
        &make_tensor_to_scalar_program(Prim::Int64, HostType::Int64),
        "tensor_to_scalar_i64_exact",
    )
    .expect("rank-zero int64 extraction must emit");

    assert!(
        source.contains("chelis_host_scalar_as_i64("),
        "int64 tensor_to_scalar must read back through the dtype-checked \
         exact scalar reader of the tagged-carrier ABI:\n{source}"
    );
    assert!(
        source.contains(
            "CHELIS_DTYPE_I64: { int64_t out; memcpy(&out, &value.bits, sizeof out); return out; }"
        ),
        "the exact reader must recover int64 bits at their declared width:\n{source}"
    );
    assert!(
        !source.contains("__result = chelis_tensor_to_f64("),
        "int64 tensor_to_scalar must not pass through double:\n{source}"
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
    .expect("rank-zero int64 packing must emit");

    assert!(
        source.contains("chelis_host_scalar_from_i64("),
        "int64 scalar_to_tensor must pack through the tagged exact-width \
         scalar of the tagged-carrier ABI:\n{source}"
    );
    assert!(
        source.contains("chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)value)"),
        "the packer must tag int64 bits at their declared width:\n{source}"
    );
    assert!(
        !source.contains("__result = chelis_scalar_tensor_from_i64("),
        "int64 scalar_to_tensor must not call the legacy I32 storage helper:\n{source}"
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
        .unwrap_or_else(|| panic!("the int32 arm contains no assignment:\n{arm}"))
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

    let temp_dir =
        std::env::temp_dir().join(format!("chelis_host_emit_{op_name}_{}", std::process::id()));
    fs::create_dir_all(&temp_dir).unwrap();
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
        "generated int32 assignment failed to compile:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let output = Command::new(&binary_path).output().unwrap();
    assert!(
        output.status.success(),
        "generated int32 assignment failed to run:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value = String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let _ = fs::remove_dir_all(temp_dir);
    value
}

// ---- L1587 binary operator ---------------------------------------------

#[test]
fn binary_elementwise_emits_dtype_switch_at_f32() {
    let program = make_binary_program("add", Prim::F32);
    let src = emit_host_program(&program, "binop_f32").unwrap();
    assert!(
        src.contains("switch (") && src.contains("->dtype)"),
        "binary elementwise must emit an outer switch on dtype; got:\n{src}"
    );
    assert!(
        src.contains("(float*)") && src.contains("->data"),
        "binary elementwise must cast `->data` through a typed pointer for f32; got:\n{src}"
    );
    assert!(
        !src.contains("    {target}->data[i] ="),
        "raw `{{target}}->data[i]` template token must be substituted; got:\n{src}"
    );
    // The legacy untyped pattern `t->data[idx] OP t->data[idx]` reads
    // 4 bytes as float. The new code paths cast `->data` to a typed
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
        src.contains("(double*)") && src.contains("->data"),
        "binary elementwise at f64 must cast `->data` through `(double*)`; got:\n{src}"
    );
}

#[test]
fn binary_elementwise_emits_dtype_switch_at_i64() {
    let program = make_binary_program("add", Prim::Int64);
    let src = emit_host_program(&program, "binop_i64").unwrap();
    assert!(
        src.contains("(int64_t*)") && src.contains("->data"),
        "binary elementwise at i64 must cast `->data` through `(int64_t*)`; got:\n{src}"
    );
}

#[test]
fn binary_elementwise_int32_arm_uses_int32_t_pointers() {
    let program = make_binary_program("add", Prim::Int32);
    let src = emit_host_program(&program, "binop_i32").unwrap();
    let arm = generated_dtype_arm(&src, "CHELIS_DTYPE_I32");

    assert!(
        arm.contains("int32_t *__target_data = (int32_t*)"),
        "the int32 target pointer must use int32_t; arm:\n{arm}"
    );
    assert!(
        arm.contains("const int32_t *__lhs_data = (const int32_t*)")
            && arm.contains("const int32_t *__rhs_data = (const int32_t*)"),
        "the int32 input pointers must use int32_t; arm:\n{arm}"
    );
}

#[test]
fn binary_elementwise_int32_arm_contains_no_float_pointer() {
    let program = make_binary_program("add", Prim::Int32);
    let src = emit_host_program(&program, "binop_i32_no_float").unwrap();
    let arm = generated_dtype_arm(&src, "CHELIS_DTYPE_I32");

    assert!(
        !arm.contains("float"),
        "the int32 arm must not use an IEEE binary32 pointer; arm:\n{arm}"
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
        src.contains("switch (") && src.contains("->dtype)"),
        "unary elementwise must emit an outer switch on dtype; got:\n{src}"
    );
    assert!(
        src.contains("(float*)"),
        "unary elementwise must cast `->data` to typed pointer; got:\n{src}"
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

#[test]
fn unary_func_int32_arm_aborts_without_binary32_conversion() {
    let program = make_unary_program("exp", Prim::Int32);
    let src = emit_host_program(&program, "unfunc_i32_reject").unwrap();
    let arm = generated_dtype_arm(&src, "CHELIS_DTYPE_I32");

    assert!(
        arm.contains("abort();"),
        "the int32 arm must abort; arm:\n{arm}"
    );
    assert!(
        !arm.contains("__target_data")
            && !arm.contains("(float*)")
            && !arm.contains("(const float*)"),
        "the int32 arm must not convert through binary32; arm:\n{arm}"
    );
}

// ---- L1749/L1753/L1757 scalar-to-tensor coercion -----------------------
//
// `assign_tensor_call` handles the case where a host helper takes a
// scalar argument; the emitter allocates a rank-0 tensor and writes
// the scalar through `tensor_name->data[0]`.  Three arms today: int64
// (already typed via `(int64_t*)` cast), bool, and the f32-default
// fallback.  The bool and f32 arms must cast `->data` to a typed
// pointer.  Sites 5 and 6 are exercised together by a single fixture
// that constructs a TensorCall taking a scalar arg of each type.

fn make_tensor_call_with_scalar_arg(scalar_ty: HostType, scalar_val: HostExpr) -> HostProgram {
    // The host-side fallback path uses `HostExprKind::TensorCall` to
    // bind through a tensor helper.  We construct a TensorCall against
    // a synthetic helper (index 0) whose input is the scalar-coerced
    // tensor, then read the body through a top-level binding to drive
    // emission.  The path under test runs before helper lookup, so
    // an empty `tensor_helpers` slice is sufficient -- the scalar
    // coercion arm executes regardless of whether the helper exists.
    let tt = vec_ty(1, Prim::F32);
    let call = HostExpr::new(HostExprKind::TensorCall {
        helper: 0,
        args: vec![scalar_val],
        ty: HostType::Tensor(tt.clone()),
    });
    let _ = scalar_ty;
    HostProgram {
        globals: vec![HostBinding {
            name: "result".to_string(),
            display_name: Some("result".to_string()),
            display_roots: Vec::new(),
            ty: HostType::Tensor(tt.clone()),
            value: call,
        }],
        global_tensor_helpers: Vec::new(),
        functions: Vec::new(),
        summary_rejections: Vec::new(),
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
    };

    let source = emit_host_program(&program, "exact_list_ingress").unwrap();
    assert!(
        source.contains("chelis_tensor_from_values(") && source.contains("CHELIS_DTYPE_F64"),
        "to_tensor must emit the registered exact tagged constructor:\n{source}"
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
fn scalar_to_tensor_coercion_bool_uses_typed_pointer() {
    let program =
        make_tensor_call_with_scalar_arg(HostType::Bool, HostExpr::new(HostExprKind::Bool(true)));
    let src = emit_host_program(&program, "scalar_bool").unwrap();
    // [05-OP-31] fixes Bool tensor storage at one canonical byte. The host
    // scalar bridge must therefore write through uint8_t and preserve only
    // the two valid Bool8 bit patterns.
    assert!(
        src.contains("((uint8_t*)")
            && src.contains("->data)[0]")
            && src.contains("? UINT8_C(1) : UINT8_C(0)"),
        "bool scalar-to-tensor coercion must write canonical Bool8 bytes; got:\n{src}"
    );
    assert!(
        !src.lines().any(|line| line.contains("? 1.0f : 0.0f")),
        "bool scalar-to-tensor coercion must not retain four-byte float storage; got:\n{src}"
    );
}

#[test]
fn scalar_to_tensor_coercion_f64_uses_f64_typed_pointer() {
    // #381: a captured f64 scalar fed to a tensor helper must pack into a
    // CHELIS_DTYPE_F64 rank-0 tensor written through a `(double*)`. The pre-fix
    // code packed an f64 scalar into a CHELIS_DTYPE_F32 tensor via `(float)value`
    // (only 4 bytes), so the f64 kernel read garbage and the value collapsed
    // to ~0. The dtype tag and the typed-pointer width must match the f64
    // operand.
    let program = make_tensor_call_with_scalar_arg(
        HostType::Float64,
        HostExpr::new(HostExprKind::Float(7.5)),
    );
    let src = emit_host_program(&program, "scalar_f64").unwrap();
    assert!(
        src.contains("chelis_alloc(0, NULL, CHELIS_DTYPE_F64)"),
        "f64 scalar-to-tensor coercion must allocate a CHELIS_DTYPE_F64 rank-0 tensor (#381); got:\n{src}"
    );
    assert!(
        src.contains("(double*)") && src.contains("->data"),
        "f64 scalar-to-tensor coercion must cast `->data` to a `(double*)` (#381); got:\n{src}"
    );
    // Must NOT pack an f64 scalar through the f32 path (the pre-fix bug).
    assert!(
        !src.lines()
            .any(|l| l.contains("->data[0] = (float)(") && !l.contains("(double*)")),
        "f64 scalar-to-tensor coercion must not pack through the f32 `(float)(...)` path (#381); got:\n{src}"
    );
}

// Note: the host lane classifies every float literal as `Float64`
// (`host_type(HostExprKind::Float)` is coarse per issue #308), so a bare
// f32 scalar arg cannot be synthesized through `make_tensor_call_with_scalar_arg`;
// the f32 packing arm is exercised end-to-end by the eval-vs-C parity test
// `ws2b_numeric_identifier_divergence` (f32 captured-scalar programs) instead.

#[test]
fn scalar_to_tensor_coercion_int64_keeps_typed_pointer() {
    // Already-typed today: `((int64_t*)tensor_name->data)[0] =
    // value;`.  Locks the invariant that this arm's typed cast
    // survives the host_emit migration.
    let program =
        make_tensor_call_with_scalar_arg(HostType::Int64, HostExpr::new(HostExprKind::Int(42)));
    let src = emit_host_program(&program, "scalar_i64").unwrap();
    assert!(
        src.contains("(int64_t*)") && src.contains("->data"),
        "int64 scalar-to-tensor coercion must use `(int64_t*)` cast; got:\n{src}"
    );
}
