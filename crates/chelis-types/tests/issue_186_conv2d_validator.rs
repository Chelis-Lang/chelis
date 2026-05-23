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

/// EXPECT (RT-205 F8): a Surf conv2d that declares a WRONG output
/// shape is rejected. Before the fix the HM signature pass
/// generated fresh dim-vars for output H/W and unified them with
/// any positive declared dim, so an obviously wrong
/// `tensor[1, 8, 100, 100]` for the canonical 8x8 input + 3x3
/// kernel (real output 6x6) silently type-checked. With the fix the
/// inferred output shape is concrete `[1, 8, 6, 6]`, so the def's
/// body-vs-declared-sig check (TypeMismatch) catches the mismatch.
#[test]
fn red_team_205_f8_wrong_declared_output_dims_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 100, 100, f32] =
  conv2d(&x, &k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for wrong declared output dims");
    // Pin the EXACT computed-vs-declared shape so a future regression
    // that re-introduces the fresh-dvar placeholder is caught here.
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("Lit(6), Lit(6)")
                && e.message.contains("Lit(100), Lit(100)")),
        "expected message naming inferred [..6, 6..] vs declared [..100, 100..], got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F8 positive parity): when the declared output
/// dims match the computed `floor((in + 2p - k) / s) + 1` shape
/// the call still type-checks cleanly.
#[test]
fn red_team_205_f8_correct_declared_output_accepted() {
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
            "expected clean check for canonical conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 F8): an output dim that disagrees by exactly 1
/// (off-by-one) is still caught. Real output for input H=8, kernel
/// 3, stride 2, padding 0 is `floor((8-3)/2) + 1 = 3`; declaring
/// `tensor[1, 8, 4, 4]` must be rejected.
#[test]
fn red_team_205_f8_off_by_one_output_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 4, 4, f32] =
  conv2d(&x, &k, 2, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for off-by-one declared output");
    // Computed output is 3x3; declared is 4x4. Pin both values in
    // the error message so off-by-one regressions surface here.
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("Lit(3), Lit(3)") && e.message.contains("Lit(4), Lit(4)")),
        "expected message naming inferred 3x3 vs declared 4x4, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

// ─── Red Team #205 round-2 follow-up findings ────────────────────

/// EXPECT (RT-205 round-2 F1): conv2d with an i64-near-max padding
/// value MUST NOT panic chelis check. Before the fix
/// `conv2d_output_extent` did `input + 2 * padding` unchecked and
/// triggered `attempt to multiply with overflow` at runtime. The
/// fix routes through `checked_mul`/`checked_add`/`checked_sub`
/// and emits a DimensionMismatch instead.
///
/// Driven through the direct-Deep entry so we can pass an i64
/// literal without tripping Surf's int32 default-literal range
/// guard (which would mask the actual validator overflow path).
/// The lit's declared `:type` is left as int32 because
/// `extract_int_literal` reads the atom value (i64-wide) regardless
/// of the declared type tag and the HM signature expects int32
/// stride/padding; using int32 here keeps HM clean so the
/// validator's overflow check is the only diagnostic that fires.
#[test]
fn red_team_205_round2_f1_padding_near_i64_max_does_not_panic() {
    // padding = 4611686018427387905 ~ i64::MAX / 2; padding * 2
    // overflows i64.
    let src = "(def {} x (lit {type: (t-tensor {} (d-lit {} 1) (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))} 0)) \
               (def {} k (lit {type: (t-tensor {} (d-lit {} 8) (d-lit {} 3) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0)) \
               (def {} y \
                 (app {} (var {} conv2d) (var {} x) (var {} k) \
                   (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 4611686018427387905)))";
    let deep = parse_deep(src).expect("deep parse");
    // Must NOT panic. Result is allowed to be Err with the overflow
    // diagnostic.
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected overflow rejection");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("overflows i64")),
        "expected overflow diagnostic, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-2 F1): the `input + 2 * padding` add path
/// also has a checked guard. With a representable `2 * padding`,
/// `input + 2 * padding` overflows when `input` itself is at the
/// i64 ceiling.
///
/// stride/padding lit type tag is kept int32 because
/// `extract_int_literal` reads the atom's i64 value regardless of
/// tag; the input dim is a `d-lit` (dimension-level int) so the
/// out-of-range-for-int32 value-literal validator does not fire on
/// it.
#[test]
fn red_team_205_round2_f1_input_plus_padding_overflow_does_not_panic() {
    // input H = i64::MAX (carried as a d-lit dim, not a value literal);
    // padding = 1. 2 * padding = 2 fits. input + 2 overflows.
    let huge_in = i64::MAX;
    let src = format!(
        "(def {{}} x (lit {{type: (t-tensor {{}} (d-lit {{}} 1) (d-lit {{}} 3) (d-lit {{}} {huge_in}) (d-lit {{}} {huge_in}) (t-prim {{}} f32))}} 0)) \
         (def {{}} k (lit {{type: (t-tensor {{}} (d-lit {{}} 8) (d-lit {{}} 3) (d-lit {{}} 3) (d-lit {{}} 3) (t-prim {{}} f32))}} 0)) \
         (def {{}} y \
           (app {{}} (var {{}} conv2d) (var {{}} x) (var {{}} k) \
             (lit {{type: (t-prim {{}} int32)}} 1) (lit {{type: (t-prim {{}} int32)}} 1)))"
    );
    let deep = parse_deep(&src).expect("deep parse");
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected overflow rejection");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("overflows i64")),
        "expected overflow diagnostic, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-2 F1 positive parity): canonical small
/// shapes still produce a clean check. Overflow guards must not
/// regress the happy path.
#[test]
fn red_team_205_round2_f1_canonical_shape_still_accepted() {
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
            "expected clean check post-R2-F1, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-2 F2): the canonical CNN layer pattern
/// `y = relu(conv2d(...))` followed by `conv2d(&y, ...)` type-checks
/// cleanly. F5 originally only handled direct `conv2d` RHS; the F2
/// fix extends `derive_ir_builtin_output_type` with shape-preserving
/// unary point-wise ops so the wrapper does not break the chain.
#[test]
fn red_team_205_round2_f2_relu_wrapped_chained_conv2d_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = relu(conv2d(&x, &k1, 1, 0))
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
            "expected clean check for relu-wrapped chained conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-2 F2): tanh wrapper is also shape-preserving.
#[test]
fn red_team_205_round2_f2_tanh_wrapped_chained_conv2d_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = tanh(conv2d(&x, &k1, 1, 0))
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
            "expected clean check for tanh-wrapped chained conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-2 F2): shape-preserving binary point-wise
/// `add(conv2d(...), &b)` is handled by the binary passthrough arm.
/// `b` is a same-shape bias-like tensor; the validator can resolve
/// `y` to the conv2d output shape via the binary derivation.
#[test]
fn red_team_205_round2_f2_add_wrapped_chained_conv2d_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = add(conv2d(&x, &k1, 1, 0), &b)
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
            "expected clean check for add-wrapped chained conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-2 F2 negative parity): a rank-changing
/// reduction wrapper (`sum(conv2d(...), 1)`) MUST NOT be handled by
/// the passthrough arm because sum reduces rank. Downstream
/// `conv2d(&y, ...)` should still fail the concrete-tensor-arg
/// check; this pins that the passthrough fix is shape-preserving
/// only, not a blanket "trust the inner op's output" path.
#[test]
fn red_team_205_round2_f2_sum_wrapped_chained_conv2d_still_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = sum(conv2d(&x, &k1, 1, 0), 1)
  conv2d(&y, &k2, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for sum-wrapped chain");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("concrete tensor argument metadata")
                || e.message.contains("dimension mismatch")
                || e.message.contains("body doesn't match declared signature")),
        "expected rejection of sum-wrapped chain, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-2 F3): chained conv2d where the first call
/// fails for a real reason (non-concrete input dim) produces exactly
/// ONE diagnostic, not a duplicate cascade. The validator suppresses
/// the second call's "concrete tensor argument metadata" error when
/// its input is a `(var name)` whose let-binding's own RHS already
/// pushed a diagnostic.
#[test]
fn red_team_205_round2_f3_cascading_errors_dedupe() {
    let src = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = conv2d(&x, &k1, 1, 0)
  conv2d(&y, &k2, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete input dim");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "expected exactly 1 metadata-cascade error, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-2 F3 negative parity): two independent
/// (non-cascading) failures still produce two errors. The
/// suppression key is "let-bound name whose RHS already failed",
/// not "any duplicate-shaped message".
#[test]
fn red_team_205_round2_f3_independent_failures_not_suppressed() {
    let src = r#"
def f(x1: tensor[1, 3, h, 16, f32], x2: tensor[1, 3, h, 16, f32], k: tensor[8, 3, 3, 3, f32]) -> (tensor[1, 8, 6, 6, f32], tensor[1, 8, 6, 6, f32]) = {
  y1 = conv2d(&x1, &k, 1, 0)
  y2 = conv2d(&x2, &k, 1, 0)
  (y1, y2)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete input dim");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        2,
        "expected 2 independent metadata errors (one per call), got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-2 F4): a rank-5 input tensor produces
/// exactly ONE rank-4 diagnostic, not two. Before the fix the HM
/// signature check and the validator's own rank guard both emitted
/// their own version of "rank-4 input ... got rank 5", confusing
/// the user. The validator now suppresses its rank diagnostic when
/// the HM-side has already emitted the equivalent.
#[test]
fn red_team_205_round2_f4_rank5_input_single_diagnostic() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, 2, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for rank-5 input");
    let rank4_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| e.message.contains("rank-4 input tensor"))
        .collect();
    assert_eq!(
        rank4_errors.len(),
        1,
        "expected exactly 1 rank-4 input diagnostic, got {:?}",
        rank4_errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-2 F4): same dedup for rank-5 kernel. The
/// HM-side and validator both check kernel rank; only one diagnostic
/// should reach the user.
#[test]
fn red_team_205_round2_f4_rank5_kernel_single_diagnostic() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, 2, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 1, 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for rank-5 kernel");
    let rank4_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| e.message.contains("rank-4 kernel tensor"))
        .collect();
    assert_eq!(
        rank4_errors.len(),
        1,
        "expected exactly 1 rank-4 kernel diagnostic, got {:?}",
        rank4_errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

// ─── Red Team #205 round-3 follow-up findings ────────────────────

/// EXPECT (RT-205 round-3 F-A): cascade dedup must apply when the
/// let RHS is a shape-passthrough wrapper over a failed
/// shape-sensitive call. Before the fix, `failed_let_names` was
/// only populated when the immediate outer call was itself
/// shape-sensitive; `y = relu(conv2d(bad))` left `y` unmarked and
/// the downstream `conv2d(&y, ...)` produced a redundant cascade
/// error.
#[test]
fn red_team_205_round3_f_a_cascade_through_passthrough_relu() {
    let src = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = relu(conv2d(&x, &k1, 1, 0))
  conv2d(&y, &k2, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete input dim");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "expected exactly 1 metadata-cascade error through relu wrapper, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-3 F-A): same dedup through a binary
/// passthrough wrapper. `add(conv2d(bad), &b)` is the canonical
/// conv+bias pattern; the cascade dedup must reach through it
/// when the inner conv2d fails.
#[test]
fn red_team_205_round3_f_a_cascade_through_passthrough_add() {
    let src = r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = add(conv2d(&x, &k1, 1, 0), &b)
  conv2d(&y, &k2, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for non-concrete input dim");
    let conv2d_metadata_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            e.message
                .contains("requires concrete tensor argument metadata")
        })
        .collect();
    assert_eq!(
        conv2d_metadata_errors.len(),
        1,
        "expected exactly 1 metadata-cascade error through add wrapper, got {:?}",
        conv2d_metadata_errors
            .iter()
            .map(|e| &e.message)
            .collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-3 F-A negative parity): when the
/// relu-wrapped first conv2d is FINE but the second conv2d has an
/// independent issue (wrong kernel rank), both diagnostics still
/// fire. The dedup must not suppress legitimately independent
/// errors.
#[test]
fn red_team_205_round3_f_a_independent_second_failure_not_suppressed() {
    // First conv2d: clean. relu(conv2d(...)) registers y as
    // tensor[1, 8, 6, 6, f32]. Second conv2d: kernel is rank 3
    // (tensor[8, 3, 3, f32]) instead of rank 4, so HM + validator
    // both surface the kernel rank-4 error. There's no f64 cascade
    // here, so we just count any errors with "rank-4 kernel" wording.
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = relu(conv2d(&x, &k1, 1, 0))
  conv2d(&y, &k2, 1, 0)
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure on second conv2d kernel rank");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("rank-4 kernel tensor")),
        "expected rank-4 kernel error on second call, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 round-3 F-B): `max_elem` is the canonical IR name
/// per spec/05 §2.1, not `maximum`. The binary passthrough allowlist
/// previously contained `maximum`/`minimum` which do not exist in
/// the IR vocabulary, so `y = max_elem(conv2d(...), &b)` cascade was
/// silently broken.
#[test]
fn red_team_205_round3_f_b_max_elem_passthrough() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = max_elem(conv2d(&x, &k1, 1, 0), &b)
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
            "expected clean check for max_elem-wrapped chain, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-3 F-B): `min_elem` is the canonical IR name
/// per spec/05 §3.4, not `minimum`. Same shape as the max_elem fix.
#[test]
fn red_team_205_round3_f_b_min_elem_passthrough() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = min_elem(conv2d(&x, &k1, 1, 0), &b)
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
            "expected clean check for min_elem-wrapped chain, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-3 F-B): one positive probe per
/// newly-added or audit-confirmed UNARY allowlist entry, locking the
/// contract against future allowlist drift. Each test wraps a clean
/// conv2d call with the unary op and chains into a second conv2d;
/// the chain must type-check cleanly. The `softmax(...)` case takes
/// a tensor + axis, but its output shape matches the input shape so
/// the unary passthrough path handles it.
#[test]
fn red_team_205_round3_f_b_unary_passthrough_audit() {
    let unary_ops = [
        "relu", "tanh", "sigmoid", "gelu", "silu", "exp", "log", "neg", "recip", "sqrt", "abs",
        "sin", "cos", "tan", "atan", "floor", "ceil",
    ];
    for op in &unary_ops {
        let src = format!(
            r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {{
  y = {op}(conv2d(&x, &k1, 1, 0))
  conv2d(&y, &k2, 1, 0)
}}
"#
        );
        let deep = surf_to_deep(&src);
        let res = check_ir_program(&deep);
        if let Err(rep) = res {
            for err in &rep.errors {
                eprintln!("op={op} unexpected error: {:?}: {}", err.kind, err.message);
            }
            panic!(
                "unary op `{op}` passthrough failed: got {} error(s)",
                rep.errors.len()
            );
        }
    }
}

/// EXPECT (RT-205 round-3 F-B): one positive probe per
/// newly-added or audit-confirmed BINARY allowlist entry. Each test
/// wraps a clean conv2d with the binary op + same-shape bias and
/// chains into a second conv2d; the chain must type-check cleanly.
///
/// `cmplt`, `gt`, `gte`, `lte`, `eq`, `neq` and `and`/`or` return
/// tensor[D, bool], so the downstream call cannot be a conv2d (which
/// requires f-prec). For those we only check that the let-binder's
/// own validation produces a clean type (no second conv2d).
#[test]
fn red_team_205_round3_f_b_binary_passthrough_audit_arith() {
    // Arithmetic binaries preserve precision and shape; chain into a
    // second conv2d to exercise the downstream registration.
    let arith_ops = ["add", "sub", "mul", "div", "max_elem", "min_elem"];
    for op in &arith_ops {
        let src = format!(
            r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {{
  y = {op}(conv2d(&x, &k1, 1, 0), &b)
  conv2d(&y, &k2, 1, 0)
}}
"#
        );
        let deep = surf_to_deep(&src);
        let res = check_ir_program(&deep);
        if let Err(rep) = res {
            for err in &rep.errors {
                eprintln!("op={op} unexpected error: {:?}: {}", err.kind, err.message);
            }
            panic!(
                "binary arith op `{op}` passthrough failed: got {} error(s)",
                rep.errors.len()
            );
        }
    }
}

/// EXPECT (RT-205 round-3 F-B): the cascade-dedup path triggers
/// when a let-bound name's RHS is a comparison/bool binary whose
/// own validation failed. The passthrough allowlist contains
/// `cmplt`, `lt`, `gt`, `gte`, `lte`, `eq`, `neq`, `and`, `or`, so
/// when one of these chains over a failed inner shape-sensitive
/// call, the cascade-dedup helper sees through them. Probe each by
/// wrapping a non-concrete-dim conv2d and checking that exactly ONE
/// metadata diagnostic fires (not two).
#[test]
fn red_team_205_round3_f_b_binary_compare_cascade_dedup() {
    let compare_ops = ["cmplt", "lt", "gt", "gte", "lte", "eq", "neq"];
    for op in &compare_ops {
        // x has non-concrete h, so conv2d(&x, ...) fails. y = op(conv2d(...), &b);
        // downstream conv2d(&y, ...) should NOT produce its own cascade error.
        let src = format!(
            r#"
def f(x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {{
  y = {op}(conv2d(&x, &k1, 1, 0), &b)
  conv2d(&y, &k2, 1, 0)
}}
"#
        );
        let deep = surf_to_deep(&src);
        let res = check_ir_program(&deep);
        let rep = res.expect_err("expected check failure for non-concrete input");
        let metadata_errors: Vec<_> = rep
            .errors
            .iter()
            .filter(|e| {
                e.message
                    .contains("requires concrete tensor argument metadata")
            })
            .collect();
        assert_eq!(
            metadata_errors.len(),
            1,
            "op `{op}` cascade-dedup failed: expected 1 metadata error, got {:?}",
            metadata_errors
                .iter()
                .map(|e| &e.message)
                .collect::<Vec<_>>()
        );
    }
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
