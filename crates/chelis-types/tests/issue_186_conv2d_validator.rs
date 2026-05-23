//! Issue #186 regression: `validate_ir_builtin_symbolic_requirements`
//! rejected every well-formed `conv2d` call at `chelis check` time.
//!
//! Reproduction surface for this file is the `check_ir_program` entry point
//! (what `chelis check` drives), not the bare `infer_program` entry point.
//! The existing `infer.rs` unit test `builtin_conv2d_accepts_int_stride_padding`
//! passes through `infer_program` and therefore does NOT exercise the
//! `validate_ir_program` pass that runs the symbolic-requirements check; that
//! is why the bug went undetected until a real Surf user hit it.
//!
//! Diagnosis (see commit body of the diagnosis commit on this branch):
//!   - `app_result_type_is_concrete` looks for a `:type` entry on the app
//!     node's metadata. Surf-desugared apps carry only `:span`; the inference
//!     pass does not stamp inferred app types back into Deep metadata, and
//!     the annotation pass that does (`annotate_ir_program_with_context`)
//!     runs AFTER `validate_ir_program`. So the output-dims check is
//!     structurally always-false for any Surf source, concrete or not.
//!   - The `expr_tensor_type_is_concrete` helper resolves args via
//!     `expr_type_expr`, which does not peek through `(borrow {} ...)`.
//!     Surf programs that use `&x` / `&k` (the idiomatic read-only form,
//!     and what the `Std.Nn.Conv.conv2d_small` sig requires) therefore also
//!     fail the tensor-metadata check.
//!   - The original issue's claim that `.skip(3).take(2)` lands on the
//!     scalar `stride`/`padding` was wrong: for a 4-arg call
//!     `(app {} (var conv2d) x k stride padding)` the window correctly
//!     lands on `(var x)` and `(var k)`.

use chelis_deep::Expr;
use chelis_deep::parser::parse_str as parse_deep;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

/// EXPECT: A Surf-source conv2d call with fully-concrete input tensor +
/// kernel tensor + integer stride/padding type-checks cleanly through
/// `chelis check` (the `check_ir_program` entry point).
///
/// Before the fix this failed with
///   `IR builtin `conv2d` requires concrete output tensor dimensions`
/// because the app's `:type` metadata was unpopulated at validator time.
#[test]
fn issue186_surf_conv2d_concrete_shapes_typechecks() {
    let src = r#"
def call_conv(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(x, k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for concrete conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: A Surf-source conv2d call with borrowed args (`&x`, `&k`) and
/// fully-concrete shapes type-checks cleanly. This is the pattern the
/// `Std.Nn.Conv.conv2d_small` signature requires: the `sig` declares
/// `&tensor[...] -> &tensor[...] -> tensor[...]`, so any Surf consumer
/// must pass borrowed args.
///
/// Before the fix this failed with both the concrete-output-dims error and
/// the concrete-tensor-arg-metadata error.
#[test]
fn issue186_surf_conv2d_borrowed_args_typechecks() {
    let src = r#"
def call_conv(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for borrowed conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: A direct-Deep conv2d call with concrete input/kernel tensor
/// dims and integer stride/padding type-checks cleanly. This mirrors the
/// existing `builtin_conv2d_accepts_int_stride_padding` test but drives
/// the `check_ir_program` pass so the validator is actually exercised.
#[test]
fn issue186_deep_conv2d_concrete_tensors_typechecks() {
    let src = "(def {} x (lit {type: (t-tensor {} (d-lit {} 1) (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))} 0)) \
               (def {} k (lit {type: (t-tensor {} (d-lit {} 8) (d-lit {} 3) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0)) \
               (def {type: (t-tensor {} (d-lit {} 1) (d-lit {} 8) (d-lit {} 6) (d-lit {} 6) (t-prim {} f32))} y \
                 (app {} (var {} conv2d) (var {} x) (var {} k) \
                   (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 0)))";
    let deep = parse_deep(src).expect("deep parse");
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for direct-deep concrete conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: A Surf-source conv2d call whose input tensor has a non-concrete
/// spatial dim (`h`) is rejected by the symbolic-requirements validator
/// with the "concrete tensor argument metadata" error. The IR lowering
/// requires concrete spatial dims to compute strides, so this negative
/// remains a real constraint even after the fix.
#[test]
fn issue186_surf_conv2d_nonconcrete_input_dim_rejected() {
    let src = r#"
def call_conv(x: tensor[1, 3, h, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(x, k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete input dim");
    assert!(
        rep.errors.iter().any(|e| e
            .message
            .contains("requires concrete tensor argument metadata")),
        "expected concrete-tensor-arg-metadata error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (sibling sweep, `mean` arm): a `mean(&x, axis)` call where the
/// reduced axis is non-concrete is still rejected. Before the fix the
/// validator's `expr_type_expr` did not peek through `(borrow {} ...)`,
/// so `tensor_dims_from_type_expr` returned None and the
/// "concrete reduced axis extent" error never fired for borrowed inputs.
#[test]
fn issue186_surf_mean_borrowed_nonconcrete_axis_rejected() {
    let src = r#"
def call_mean(x: tensor[32, n, f32]) -> tensor[32, f32] = mean(&x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for borrowed mean with non-concrete axis");
    assert!(
        rep.errors.iter().any(|e| e
            .message
            .contains("requires a concrete reduced axis extent")),
        "expected mean concrete-reduced-axis error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (sibling sweep, `layer_norm` arm): a `layer_norm(&x, &g, &b)`
/// call whose normalized axis is non-concrete is still rejected. Same
/// borrow-blindness root cause as the mean arm above.
#[test]
fn issue186_surf_layer_norm_borrowed_nonconcrete_axis_rejected() {
    let src = r#"
def call_ln(x: tensor[32, n, f32], g: tensor[n, f32], b: tensor[n, f32]) -> tensor[32, n, f32] =
  layer_norm(&x, &g, &b)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep =
        res.expect_err("expected check failure for borrowed layer_norm with non-concrete axis");
    assert!(
        rep.errors.iter().any(|e| e
            .message
            .contains("requires a concrete normalized axis extent")),
        "expected layer_norm concrete-normalized-axis error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT: A direct-Deep conv2d call whose stride argument is a `(var ...)`
/// rather than an integer literal is rejected with a clear error. The
/// IR lowering requires the stride/padding to be statically-knowable
/// integers (`extract_int_literal` on the arg must succeed at lowering).
#[test]
fn issue186_deep_conv2d_nonliteral_stride_rejected() {
    let src = "(def {type: (t-prim {} int32)} stride_v (lit {type: (t-prim {} int32)} 1)) \
               (def {} x (lit {type: (t-tensor {} (d-lit {} 1) (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))} 0)) \
               (def {} k (lit {type: (t-tensor {} (d-lit {} 8) (d-lit {} 3) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0)) \
               (def {type: (t-tensor {} (d-lit {} 1) (d-lit {} 8) (d-lit {} 6) (d-lit {} 6) (t-prim {} f32))} y \
                 (app {} (var {} conv2d) (var {} x) (var {} k) \
                   (var {} stride_v) (lit {type: (t-prim {} int32)} 0)))";
    let deep = parse_deep(src).expect("deep parse");
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-literal stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("requires a literal integer stride")
                || e.message
                    .contains("requires concrete tensor argument metadata")),
        "expected literal-int-stride error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}
