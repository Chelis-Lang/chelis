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
//!   * L1623 elementwise binary func (`fmaxf`, `fminf`, ...).
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

use chelis_backend_c::host_emit::emit_host_program;
use chelis_ir::dag::{DimInfo, TensorType};
use chelis_ir::host::{
    HostBinding, HostExpr, HostExprKind, HostFunction, HostParam, HostProgram, HostType,
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
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
    }
}

// ---- L1587 binary operator ---------------------------------------------

#[test]
fn binary_elementwise_emits_dtype_switch_at_f32() {
    let program = make_binary_program("add", Prim::F32);
    let src = emit_host_program(&program, "binop_f32");
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
    let src = emit_host_program(&program, "binop_f64");
    assert!(
        src.contains("(double*)") && src.contains("->data"),
        "binary elementwise at f64 must cast `->data` through `(double*)`; got:\n{src}"
    );
}

#[test]
fn binary_elementwise_emits_dtype_switch_at_i64() {
    let program = make_binary_program("add", Prim::Int64);
    let src = emit_host_program(&program, "binop_i64");
    assert!(
        src.contains("(int64_t*)") && src.contains("->data"),
        "binary elementwise at i64 must cast `->data` through `(int64_t*)`; got:\n{src}"
    );
}

// ---- L1623 binary func -------------------------------------------------

#[test]
fn binary_func_elementwise_emits_typed_pointer_access() {
    let program = make_binary_program("max_elem", Prim::F32);
    let src = emit_host_program(&program, "binfunc_f32");
    // `max_elem` -> `fmaxf` requires an f32-typed access pattern.  The
    // migrated emission must cast `->data` to a typed pointer before
    // indexing rather than reading through the public `float *data`
    // field declaration unconditionally.
    assert!(
        src.contains("(float*)") || src.contains("(const float*)"),
        "binary func elementwise must cast `->data` to a typed pointer; got:\n{src}"
    );
    assert!(
        !src.lines().any(|l| l.contains("->data[i] = ")
            && !l.contains("(float*)")
            && !l.contains("(double*)")
            && !l.contains("(int64_t*)")),
        "binary func elementwise must not emit bare `->data[i] = fmaxf(...)`; got:\n{src}"
    );
}

// ---- L1649 unary operator ----------------------------------------------

#[test]
fn unary_elementwise_emits_dtype_switch_at_f32() {
    let program = make_unary_program("neg", Prim::F32);
    let src = emit_host_program(&program, "unop_f32");
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
    let src = emit_host_program(&program, "unop_f64");
    assert!(
        src.contains("(double*)"),
        "unary elementwise at f64 must cast `->data` through `(double*)`; got:\n{src}"
    );
}

#[test]
fn unary_elementwise_emits_dtype_switch_at_i64() {
    let program = make_unary_program("neg", Prim::Int64);
    let src = emit_host_program(&program, "unop_i64");
    assert!(
        src.contains("(int64_t*)"),
        "unary elementwise at i64 must cast `->data` through `(int64_t*)`; got:\n{src}"
    );
}

// ---- L1675 unary func --------------------------------------------------

#[test]
fn unary_func_elementwise_emits_typed_pointer_access() {
    let program = make_unary_program("exp", Prim::F32);
    let src = emit_host_program(&program, "unfunc_f32");
    assert!(
        src.contains("(float*)") || src.contains("(const float*)"),
        "unary func elementwise must cast `->data` to typed pointer; got:\n{src}"
    );
    assert!(
        !src.lines()
            .any(|l| l.contains("->data[i] = expf") && !l.contains("(float*)")),
        "unary func elementwise must not emit bare `->data[i] = expf(...)`; got:\n{src}"
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
            ty: HostType::Tensor(tt.clone()),
            value: call,
        }],
        global_tensor_helpers: Vec::new(),
        functions: Vec::new(),
        summary_rejections: Vec::new(),
    }
}

#[test]
fn scalar_to_tensor_coercion_bool_uses_typed_pointer() {
    let program =
        make_tensor_call_with_scalar_arg(HostType::Bool, HostExpr::new(HostExprKind::Bool(true)));
    let src = emit_host_program(&program, "scalar_bool");
    // The legacy bool arm writes `tensor_name->data[0] = value ? 1.0f
    // : 0.0f;` against `float *data`.  The migrated code must cast
    // `->data` through a typed pointer first.
    assert!(
        src.contains("(float*)") && src.contains("->data"),
        "bool scalar-to-tensor coercion must cast `->data` to typed pointer; got:\n{src}"
    );
    assert!(
        !src.lines().any(|l| l.contains("->data[0] = ")
            && l.contains("? 1.0f : 0.0f")
            && !l.contains("(float*)")),
        "bool scalar-to-tensor coercion must not emit bare `->data[0] = value ? 1.0f : 0.0f`; got:\n{src}"
    );
}

#[test]
fn scalar_to_tensor_coercion_f64_uses_f64_typed_pointer() {
    // #381: a captured f64 scalar fed to a tensor helper must pack into a
    // CHELIS_F64 rank-0 tensor written through a `(double*)`. The pre-fix
    // code packed an f64 scalar into a CHELIS_F32 tensor via `(float)value`
    // (only 4 bytes), so the f64 kernel read garbage and the value collapsed
    // to ~0. The dtype tag and the typed-pointer width must match the f64
    // operand.
    let program = make_tensor_call_with_scalar_arg(
        HostType::Float64,
        HostExpr::new(HostExprKind::Float(7.5)),
    );
    let src = emit_host_program(&program, "scalar_f64");
    assert!(
        src.contains("chelis_alloc(0, NULL, CHELIS_F64)"),
        "f64 scalar-to-tensor coercion must allocate a CHELIS_F64 rank-0 tensor (#381); got:\n{src}"
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
    let src = emit_host_program(&program, "scalar_i64");
    assert!(
        src.contains("(int64_t*)") && src.contains("->data"),
        "int64 scalar-to-tensor coercion must use `(int64_t*)` cast; got:\n{src}"
    );
}
