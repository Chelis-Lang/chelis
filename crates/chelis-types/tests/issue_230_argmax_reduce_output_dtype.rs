//! Issue #230 regression: `argmax_reduce` / `argmin_reduce` output
//! dtype must be `i64`, not the input tensor's dtype.
//!
//! Per `packages/chelis-std/src/tensor/reduce.ch`, the canonical
//! signatures are:
//!
//! ```text
//! sig argmax: &tensor[a, b, p] -> i32 -> tensor[b, i64]
//! sig argmin: &tensor[a, b, p] -> i32 -> tensor[b, i64]
//! ```
//!
//! i.e. the reduced axis is collapsed (rank `n` -> rank `n-1`) and the
//! output element type is always `i64` (indices), regardless of the
//! input element type `p`. This contrasts with `sum`, `max_reduce`,
//! `min_reduce`, `prod_reduce`, and `mean`, which preserve the input
//! dtype (modulo the §5.7.1 widening for `sum` on narrow integer
//! operands).
//!
//! Before the fix, `check_reduction_signature` defaulted every
//! non-`sum` reduction's output precision to the input precision,
//! producing `tensor[b, f32]` for `argmax_reduce` on an `f32` input.
//! Downstream consumers like hydronnx (which emits ONNX ArgMax /
//! ArgMin into chelis) hit a `TypeMismatch` on the declared
//! `tensor[..., i64]` signature.
//!
//! See `crates/chelis-ir/src/dag.rs::RiscOp::Argmax` for the
//! storage-layer note: the host-runtime / IR evaluator still stores
//! integer-valued floats internally per the Phase 3j-pre Batch 1
//! caveat. The type-system label is independent of the storage and is
//! what user-facing surface contracts (signatures, std modules,
//! Surf-level declarations) bind against.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

// ---------------------------------------------------------------------
// Positive: argmax_reduce / argmin_reduce output dtype is i64.
// ---------------------------------------------------------------------

/// EXPECT: `argmax_reduce` on an `f32` tensor produces `tensor[..., i64]`.
/// This is the hydronnx PR #2 reproducer (issue #230).
#[test]
fn issue230_argmax_reduce_f32_input_yields_int64() {
    let src = r#"
def forward(x: tensor[2, 3, f32]) -> tensor[2, i64] = argmax_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for argmax_reduce(f32) -> i64, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `argmax_reduce` on an `i32` tensor also produces
/// `tensor[..., i64]`. The output dtype is independent of the input
/// dtype.
#[test]
fn issue230_argmax_reduce_int32_input_yields_int64() {
    let src = r#"
def forward(x: tensor[2, 3, i32]) -> tensor[2, i64] = argmax_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for argmax_reduce(i32) -> i64, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `argmin_reduce` on an `f32` tensor produces `tensor[..., i64]`.
#[test]
fn issue230_argmin_reduce_f32_input_yields_int64() {
    let src = r#"
def forward(x: tensor[2, 3, f32]) -> tensor[2, i64] = argmin_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for argmin_reduce(f32) -> i64, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `argmin_reduce` on an `i64` tensor still produces
/// `tensor[..., i64]`. This is the identity case for the dtype rule
/// (input dtype happens to match the canonical output dtype).
#[test]
fn issue230_argmin_reduce_int64_input_yields_int64() {
    let src = r#"
def forward(x: tensor[2, 3, i64]) -> tensor[2, i64] = argmin_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for argmin_reduce(i64) -> i64, got {} error(s)",
            rep.errors.len()
        );
    }
}

// ---------------------------------------------------------------------
// Negative: declaring the wrong output dtype is rejected with the
// usual signature-mismatch diagnostic, citing the inferred i64
// body type.
// ---------------------------------------------------------------------

/// EXPECT: declaring `argmax_reduce(tensor[2, 3, f32], 1)` to return
/// `tensor[2, f32]` is rejected. Before the fix this silently passed
/// because the checker computed the body type as `tensor[2, f32]`
/// matching the declaration. After the fix, the body type is
/// `tensor[2, i64]` and unification against the declared signature
/// fails.
#[test]
fn issue230_argmax_reduce_wrong_output_dtype_f32_rejected() {
    let src = r#"
def forward(x: tensor[2, 3, f32]) -> tensor[2, f32] = argmax_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for argmax_reduce -> f32");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("doesn't match declared signature")),
        "expected signature-mismatch diagnostic, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
    assert!(
        rep.errors.iter().any(|e| e.message.contains("i64")),
        "expected the diagnostic to surface the i64 body type, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT: declaring `argmin_reduce(tensor[2, 3, i32], 1)` to return
/// `tensor[2, i32]` is rejected for the same reason. The output dtype
/// rule is independent of the input dtype; even when the input is an
/// integer type, the output is still i64, not the input's i32.
#[test]
fn issue230_argmin_reduce_wrong_output_dtype_int32_rejected() {
    let src = r#"
def forward(x: tensor[2, 3, i32]) -> tensor[2, i32] = argmin_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for argmin_reduce -> i32");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("doesn't match declared signature")),
        "expected signature-mismatch diagnostic, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
    assert!(
        rep.errors.iter().any(|e| e.message.contains("i64")),
        "expected the diagnostic to surface the i64 body type, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------
// Sibling sweep: other reductions still preserve the input dtype.
// argmax/argmin are the only reductions in the family that change
// dtype; sum/max_reduce/min_reduce/prod_reduce/mean must NOT be
// affected by the issue #230 fix.
// ---------------------------------------------------------------------

/// EXPECT: `max_reduce(tensor[2, 3, f32], 1)` still produces
/// `tensor[2, f32]` — input dtype preserved.
#[test]
fn issue230_sibling_sweep_max_reduce_preserves_input_dtype() {
    let src = r#"
def forward(x: tensor[2, 3, f32]) -> tensor[2, f32] = max_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "max_reduce must still preserve input dtype, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `min_reduce(tensor[2, 3, f32], 1)` still produces
/// `tensor[2, f32]` — input dtype preserved.
#[test]
fn issue230_sibling_sweep_min_reduce_preserves_input_dtype() {
    let src = r#"
def forward(x: tensor[2, 3, f32]) -> tensor[2, f32] = min_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "min_reduce must still preserve input dtype, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `prod_reduce(tensor[2, 3, f32], 1)` still produces
/// `tensor[2, f32]` — input dtype preserved.
#[test]
fn issue230_sibling_sweep_prod_reduce_preserves_input_dtype() {
    let src = r#"
def forward(x: tensor[2, 3, f32]) -> tensor[2, f32] = prod_reduce(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "prod_reduce must still preserve input dtype, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `mean(tensor[2, 3, f32], 1)` still produces
/// `tensor[2, f32]` — input dtype preserved.
#[test]
fn issue230_sibling_sweep_mean_preserves_input_dtype() {
    let src = r#"
def forward(x: tensor[2, 3, f32]) -> tensor[2, f32] = mean(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "mean must still preserve input dtype, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `sum(tensor[2, 3, f32], 1)` still produces
/// `tensor[2, f32]` — f32 input is preserved (the §5.7.1 widening
/// only applies to narrow integer inputs).
#[test]
fn issue230_sibling_sweep_sum_f32_preserves_input_dtype() {
    let src = r#"
def forward(x: tensor[2, 3, f32]) -> tensor[2, f32] = sum(x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "sum on f32 must still produce f32, got {} error(s)",
            rep.errors.len()
        );
    }
}
