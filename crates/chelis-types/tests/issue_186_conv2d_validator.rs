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

// ─── Red Team #205 follow-up findings ────────────────────────────

/// EXPECT (RT-205 F1): conv2d with stride 0 is rejected at check
/// time. The output spatial dim formula in spec/05 §471-483
/// (`floor((in + 2p - k) / s) + 1`) divides by stride; previously
/// the validator silently accepted `stride == 0` and the back-end
/// ICE'd at codegen with a symbolic-dim crash. Now blocked with a
/// `positive stride` diagnostic.
#[test]
fn red_team_205_f1_zero_stride_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 0, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for stride == 0");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("requires a positive stride")),
        "expected positive-stride error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F2): conv2d with stride -1 is rejected at check
/// time. `extract_int_literal` accepts the neg-of-lit form, so a
/// sign check on the extracted value is required.
#[test]
fn red_team_205_f2_negative_stride_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, -1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for negative stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("requires a positive stride")),
        "expected positive-stride error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F3): conv2d with negative padding is rejected at
/// check time. Negative padding shrinks the effective input below
/// zero in the output formula.
#[test]
fn red_team_205_f3_negative_padding_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 1, -100)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for negative padding");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("requires non-negative padding")),
        "expected non-negative-padding error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F1/F2 positive parity): stride 1 + padding 0 is
/// still accepted. This pins that the new sign checks do not break
/// the canonical happy path.
#[test]
fn red_team_205_f1_f2_stride_one_padding_zero_accepted() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for stride=1 padding=0, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 F4): conv2d with a kernel larger than the padded
/// input is rejected at check time. Input `[1,3,2,2]` + kernel
/// `[8,3,5,5]` + stride 1 + padding 0 evaluates output H/W to
/// `floor((2 + 0 - 5) / 1) + 1 = -2`. Previously slipped through
/// the validator and only failed during back-end lowering.
#[test]
fn red_team_205_f4_oversize_kernel_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 2, 2, f32], k: tensor[8, 3, 5, 5, f32]) -> tensor[1, 8, 1, 1, f32] =
  conv2d(&x, &k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for oversize kernel");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("output height") || e.message.contains("output width")),
        "expected output-extent error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F4): conv2d output dim that evaluates to exactly
/// zero is also rejected. Input H=3, kernel=3, stride=2, padding=0:
/// `floor((3 - 3) / 2) + 1 = 1`. So we use kernel=5 on input H=4 to
/// get `floor(-1/2)+1 = -1+1 = 0`. The validator must reject 0 as
/// well as negative.
#[test]
fn red_team_205_f4_zero_output_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 4, 8, f32], k: tensor[8, 3, 5, 3, f32]) -> tensor[1, 8, 1, 6, f32] =
  conv2d(&x, &k, 2, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for zero-extent output");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("evaluates to 0")),
        "expected zero-output error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F4): conv2d with non-rank-4 input tensor is
/// rejected at check time. Pinned by the validator's rank guard so
/// the error fires before the back-end lowering pass.
#[test]
fn red_team_205_f4_rank3_input_rejected() {
    let src = r#"
def f(x: tensor[3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for rank-3 input");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("rank-4 input tensor")),
        "expected rank-4-input error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F4 positive parity): the canonical conv2d
/// `floor((8 + 0 - 3) / 1) + 1 = 6` shape continues to be accepted.
#[test]
fn red_team_205_f4_canonical_output_accepted() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for canonical conv2d shape, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 F5): chained conv2d via let-binding type-checks
/// cleanly. The validator's per-let-scope type env now registers
/// the conv2d call's derivable output type so downstream
/// shape-sensitive calls that consume the let-bound name can
/// resolve to a concrete tensor type.
#[test]
fn red_team_205_f5_chained_conv2d_via_let_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = conv2d(&x, &k1, 1, 0)
  conv2d(&y, &k2, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for chained conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 F5 negative parity): a chained conv2d whose second
/// call uses ill-formed args (oversize kernel for the first conv2d's
/// output) is still rejected, with the F4 output-extent diagnostic.
/// This pins that the let-bind type registration does not silently
/// blind the validator: the second call's args now resolve, so the
/// F4 formula evaluates against them.
#[test]
fn red_team_205_f5_chained_conv2d_second_call_ill_formed_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 9, 9, f32]) -> tensor[1, 16, 1, 1, f32] = {
  y = conv2d(&x, &k1, 1, 0)
  conv2d(&y, &k2, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for ill-formed second conv2d");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("output height") || e.message.contains("output width")),
        "expected output-extent error from second conv2d, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F6): validator errors flow through CheckError with
/// `DimensionMismatch` kind (severity 0.8) rather than `Other`
/// (severity 0.5), matching the surface DimensionMismatch already
/// uses for inference-layer shape errors.
#[test]
fn red_team_205_f6_validator_error_uses_dimension_mismatch_kind() {
    use chelis_types::errors::CheckErrorKind;
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 0, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for stride 0");
    let err = rep
        .errors
        .iter()
        .find(|e| e.message.contains("positive stride"))
        .expect("positive-stride error must be present");
    assert!(
        matches!(err.kind, CheckErrorKind::DimensionMismatch),
        "expected DimensionMismatch kind, got {:?}",
        err.kind
    );
    assert!(
        (err.severity - 0.8).abs() < 1e-9,
        "expected severity 0.8, got {}",
        err.severity
    );
}

/// EXPECT (RT-205 F6): validator errors include the call site's
/// source span identifier as a suffix `(at surf:offset..end)` so
/// JSON consumers can locate the offending expression. The exact
/// offsets vary with surrounding whitespace; this test pins only the
/// presence of the `(at surf:` prefix and the `..` separator.
#[test]
fn red_team_205_f6_validator_error_has_span_suffix() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 0, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for stride 0");
    let err = rep
        .errors
        .iter()
        .find(|e| e.message.contains("positive stride"))
        .expect("positive-stride error must be present");
    assert!(
        err.message.contains("(at surf:") && err.message.contains(".."),
        "expected span suffix `(at surf:OFFSET..END)`, got message {:?}",
        err.message
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
