//! Issue #186 regression coverage for convolution's post-inference semantic
//! checks. PP9 runs the spec-owned checks at every public checker entry and
//! removes the backend-only concrete-metadata refusal.
//!
//! Diagnosis (see commit body of the diagnosis commit on this branch):
//!   - `app_result_type_is_concrete` looks for a `:type` entry on the app
//!     node's metadata. Surf-desugared apps carry only `:span`; the inference
//!     pass does not stamp inferred app types back into Deep metadata, and
//!     the annotation pass that does (`annotate_ir_program_with_context`)
//!     ran after the former validator. So the output-dims check was
//!     structurally always-false for any Surf source, concrete or not.
//!   - The former `expr_tensor_type_is_concrete` helper resolved args via
//!     `expr_type_expr`, which does not peek through `(borrow {} ...)`.
//!     Surf programs that use `&x` / `&k` (the idiomatic read-only form,
//!     and what the `School.Nn.Conv.conv_small` sig requires) therefore also
//!     fail the tensor-metadata check.
//!   - The original issue's claim that `.skip(3).take(2)` lands on the
//!     scalar `stride`/`padding` was wrong: for a 4-arg call
//!     `(app {} (var conv) x k stride padding)` the window correctly
//!     lands on `(var x)` and `(var k)`.

use chelis_deep::Expr;
use chelis_deep::parser::parse_str as parse_deep;
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

/// EXPECT: A Surf-source conv call with fully-concrete input tensor +
/// kernel tensor + integer stride/padding type-checks cleanly through
/// `chelis check` (the `check_ir_program` entry point).
///
/// Before the fix this failed with
///   `IR builtin `conv` requires concrete output tensor dimensions`
/// because the app's `:type` metadata was unpopulated at validator time.
#[test]
fn issue186_surf_conv_concrete_shapes_typechecks() {
    let src = r#"
def call_conv(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(x, k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for concrete conv, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: A Surf-source conv call with borrowed args (`&x`, `&k`) and
/// fully-concrete shapes type-checks cleanly. This is the pattern the
/// `School.Nn.Conv.conv_small` signature requires: the `sig` declares
/// `&tensor[...] -> &tensor[...] -> tensor[...]`, so any Surf consumer
/// must pass borrowed args.
///
/// Before the fix this failed with both the concrete-output-dims error and
/// the concrete-tensor-arg-metadata error.
#[test]
fn issue186_surf_conv_borrowed_args_typechecks() {
    let src = r#"
def call_conv(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for borrowed conv, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: A direct-Deep conv call with concrete input/kernel tensor
/// dims and integer stride/padding type-checks cleanly. This mirrors the
/// existing `builtin_conv_accepts_int_stride_padding` test but drives
/// the `check_ir_program` pass so the validator is actually exercised.
#[test]
fn issue186_deep_conv_concrete_tensors_typechecks() {
    let src = "(def {} x (lit {type: (t-tensor {} (d-lit {} 1) (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))} 0)) \
               (def {} k (lit {type: (t-tensor {} (d-lit {} 8) (d-lit {} 3) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0)) \
               (def {type: (t-tensor {} (d-lit {} 1) (d-lit {} 8) (d-lit {} 6) (d-lit {} 6) (t-prim {} f32))} y \
                 (app {} (var {} conv) (var {} x) (var {} k) \
                   (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (var {} Nil))) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 0)) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 0)) (var {} Nil)))))";
    let deep = parse_deep(src).expect("deep parse");
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for direct-deep concrete conv, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// PP9 / [05-OP-51]: symbolic tensor metadata is a legal checker input.
/// Backends that cannot lower it refuse it under chelis#730 instead.
#[test]
fn issue186_surf_conv_nonconcrete_input_dim_is_checker_legal() {
    let src = r#"
def call_conv[h](x: tensor[1, 3, h, 8, f32], k: tensor[8, 3, 3, 3, f32]) =
  conv(x, k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic conv metadata is not a type error");
}

/// PP9: a symbolic selected extent is not a `mean` type error.
#[test]
fn issue186_surf_mean_borrowed_nonconcrete_axis_is_checker_legal() {
    let src = r#"
def call_mean[n](x: tensor[32, n, f32]) -> tensor[32, f32] = mean(&x, 1)
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic mean extent is not a type error");
}

/// PP9: a symbolic normalized extent is not a `layer_norm` type error.
#[test]
fn issue186_surf_layer_norm_borrowed_nonconcrete_axis_is_checker_legal() {
    let src = r#"
def call_ln[n](x: tensor[32, n, f32], g: tensor[n, f32], b: tensor[n, f32]) -> tensor[32, n, f32] =
  layer_norm(&x, &g, &b, 0.00001f32)
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic layer_norm extent is not a type error");
}

// ─── Red Team #205 follow-up findings ────────────────────────────

/// EXPECT (RT-205 F1): conv with stride 0 is rejected at check
/// time. The output spatial dim formula in spec/05 §471-483
/// (`floor((in + 2p - k) / s) + 1`) divides by stride; previously
/// the validator silently accepted `stride == 0` and the back-end
/// ICE'd at codegen with a symbolic-dim crash. Now blocked with a
/// `positive stride` diagnostic.
#[test]
fn red_team_205_f1_zero_stride_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])
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

/// EXPECT (RT-205 F2): conv with stride -1 is rejected at check
/// time. `extract_int_literal` accepts the neg-of-lit form, so a
/// sign check on the extracted value is required.
#[test]
fn red_team_205_f2_negative_stride_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [-1i64, -1i64], [(0i64, 0i64), (0i64, 0i64)])
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

/// EXPECT (RT-205 F3): conv with negative padding is rejected at
/// check time. Negative padding shrinks the effective input below
/// zero in the output formula.
#[test]
fn red_team_205_f3_negative_padding_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(-100i64, -100i64), (-100i64, -100i64)])
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
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
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

/// EXPECT (RT-205 F4): conv with a kernel larger than the padded
/// input is rejected at check time. Input `[1,3,2,2]` + kernel
/// `[8,3,5,5]` + stride 1 + padding 0 evaluates output H/W to
/// `floor((2 + 0 - 5) / 1) + 1 = -2`. Previously slipped through
/// the validator and only failed during back-end lowering.
#[test]
fn red_team_205_f4_oversize_kernel_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 2, 2, f32], k: tensor[8, 3, 5, 5, f32]) -> tensor[1, 8, 1, 1, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for oversize kernel");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("output spatial axis")),
        "expected output-extent error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F4): conv output dim that evaluates to exactly
/// zero is also rejected. Input H=3, kernel=3, stride=2, padding=0:
/// `floor((3 - 3) / 2) + 1 = 1`. So we use kernel=5 on input H=4 to
/// get `floor(-1/2)+1 = -1+1 = 0`. The validator must reject 0 as
/// well as negative.
#[test]
fn red_team_205_f4_zero_output_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 4, 8, f32], k: tensor[8, 3, 5, 3, f32]) -> tensor[1, 8, 1, 6, f32] =
  conv(&x, &k, [2i64, 2i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for zero-extent output");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("kernel must fit the padded input")),
        "expected zero-output error, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

/// EXPECT (RT-205 F4): conv rejects an input/kernel rank disagreement at
/// check time, before the back-end lowering pass.
#[test]
fn red_team_205_f4_rank3_input_rejected() {
    let src = r#"
def f(x: tensor[3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for rank-3 input");
    assert!(
        rep.errors.iter().any(|e| {
            matches!(
                e.kind,
                chelis_types::errors::CheckErrorKind::DimensionMismatch
            ) && e.expected.as_deref() == Some("rank 3")
                && e.got.as_deref() == Some("rank 4")
                && e.message.contains("conv argument 2 (kernel)")
        }),
        "expected the rank disagreement on the kernel, got {:?}",
        rep.errors
    );
}

/// EXPECT (RT-205 F4 positive parity): the canonical conv
/// `floor((8 + 0 - 3) / 1) + 1 = 6` shape continues to be accepted.
#[test]
fn red_team_205_f4_canonical_output_accepted() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for canonical conv shape, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 F5): chained conv via let-binding type-checks
/// cleanly. The validator's per-let-scope type env now registers
/// the conv call's derivable output type so downstream
/// shape-sensitive calls that consume the let-bound name can
/// resolve to a concrete tensor type.
#[test]
fn red_team_205_f5_chained_conv_via_let_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for chained conv, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 F5 negative parity): a chained conv whose second
/// call uses ill-formed args (oversize kernel for the first conv's
/// output) is still rejected, with the F4 output-extent diagnostic.
/// This pins that the let-bind type registration does not silently
/// blind the validator: the second call's args now resolve, so the
/// F4 formula evaluates against them.
#[test]
fn red_team_205_f5_chained_conv_second_call_ill_formed_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 9, 9, f32]) -> tensor[1, 16, 1, 1, f32] = {
  y = conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for ill-formed second conv");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("output spatial axis")),
        "expected output-extent error from second conv, got {:?}",
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
  conv(&x, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])
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
  conv(&x, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])
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

/// EXPECT (RT-205 F8): a Surf conv that declares a WRONG output
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
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for wrong declared output dims");
    // Pin the EXACT computed-vs-declared shape so a future regression
    // that re-introduces the fresh-dvar placeholder is caught here.
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("6, 6") && e.message.contains("100, 100")),
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
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for canonical conv, got {} error(s)",
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
  conv(&x, &k, [2i64, 2i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for off-by-one declared output");
    // Computed output is 3x3; declared is 4x4. Pin both values in
    // the error message so off-by-one regressions surface here.
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.contains("3, 3") && e.message.contains("4, 4")),
        "expected message naming inferred 3x3 vs declared 4x4, got {:?}",
        rep.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

// ─── Red Team #205 round-2 follow-up findings ────────────────────

/// EXPECT (RT-205 round-2 F1): conv with an i64-near-max padding
/// value MUST NOT panic chelis check. Before the fix
/// `conv_output_extent` did `input + 2 * padding` unchecked and
/// triggered `attempt to multiply with overflow` at runtime. The
/// fix routes through `checked_mul`/`checked_add`/`checked_sub`
/// and emits a DimensionMismatch instead.
///
/// Driven through the direct-Deep entry so we can pass an i64
/// literal without tripping Surf's i32 default-literal range
/// guard (which would mask the actual validator overflow path).
/// The lit's declared `:type` is left as i32 because
/// `extract_int_literal` reads the atom value (i64-wide) regardless
/// of the declared type tag and the HM signature expects i32
/// stride/padding; using i32 here keeps HM clean so the
/// validator's overflow check is the only diagnostic that fires.
#[test]
fn red_team_205_round2_f1_padding_near_i64_max_does_not_panic() {
    // padding = 4611686018427387905 ~ i64::MAX / 2; padding * 2
    // overflows i64.
    let src = "(def {} x (lit {type: (t-tensor {} (d-lit {} 1) (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))} 0)) \
               (def {} k (lit {type: (t-tensor {} (d-lit {} 8) (d-lit {} 3) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0)) \
               (def {} y \
                 (app {} (var {} conv) (var {} x) (var {} k) \
                   (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (var {} Nil))) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 4611686018427387905) (lit {type: (t-prim {} i64)} 4611686018427387905)) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 4611686018427387905) (lit {type: (t-prim {} i64)} 4611686018427387905)) (var {} Nil)))))";
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
/// stride/padding lit type tag is kept i32 because
/// `extract_int_literal` reads the atom's i64 value regardless of
/// tag; the input dim is a `d-lit` (dimension-level int) so the
/// out-of-range-for-i32 value-literal validator does not fire on
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
           (app {{}} (var {{}} conv) (var {{}} x) (var {{}} k) \
             (app {{}} (var {{}} Cons) (lit {{type: (t-prim {{}} i64)}} 1) (app {{}} (var {{}} Cons) (lit {{type: (t-prim {{}} i64)}} 1) (var {{}} Nil))) (app {{}} (var {{}} Cons) (tuple {{}} (lit {{type: (t-prim {{}} i64)}} 1) (lit {{type: (t-prim {{}} i64)}} 1)) (app {{}} (var {{}} Cons) (tuple {{}} (lit {{type: (t-prim {{}} i64)}} 1) (lit {{type: (t-prim {{}} i64)}} 1)) (var {{}} Nil)))))"
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
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
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
/// `y = relu(conv(...))` followed by `conv(&y, ...)` type-checks
/// cleanly. F5 originally only handled direct `conv` RHS; the F2
/// fix extends `derive_ir_builtin_output_type` with shape-preserving
/// unary point-wise ops so the wrapper does not break the chain.
#[test]
fn red_team_205_round2_f2_relu_wrapped_chained_conv_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = relu(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]))
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for relu-wrapped chained conv, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-2 F2): tanh wrapper is also shape-preserving.
#[test]
fn red_team_205_round2_f2_tanh_wrapped_chained_conv_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = tanh(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]))
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for tanh-wrapped chained conv, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-2 F2): shape-preserving binary point-wise
/// `add(conv(...), &b)` is handled by the binary passthrough arm.
/// `b` is a same-shape bias-like tensor; the validator can resolve
/// `y` to the conv output shape via the binary derivation.
#[test]
fn red_team_205_round2_f2_add_wrapped_chained_conv_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = add(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), &b)
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for add-wrapped chained conv, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-2 F2 negative parity): a rank-changing
/// reduction wrapper (`sum(conv(...), 1)`) MUST NOT be handled by
/// the passthrough arm because sum reduces rank. Downstream
/// `conv(&y, ...)` must still reject the rank mismatch, not inherit the
/// original convolution's rank.
#[test]
fn red_team_205_round2_f2_sum_wrapped_chained_conv_still_rejected() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = sum(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), 1)
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for sum-wrapped chain");
    assert!(
        rep.errors.iter().any(|e| {
            matches!(
                e.kind,
                chelis_types::errors::CheckErrorKind::DimensionMismatch
            ) && e.expected.as_deref() == Some("rank 3")
                && e.got.as_deref() == Some("rank 4")
                && e.message.contains("conv argument 2 (kernel)")
        }),
        "expected rejection of sum-wrapped chain, got {:?}",
        rep.errors
    );
}

/// PP9: symbolic metadata remains legal through a chained convolution.
#[test]
fn red_team_205_round2_f3_symbolic_chain_is_checker_legal() {
    let src = r#"
def f[h](x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic conv metadata is not a type error");
}

/// PP9: independent symbolic conv calls are both checker-legal.
#[test]
fn red_team_205_round2_f3_independent_symbolic_calls_are_checker_legal() {
    let src = r#"
def f[h](x1: tensor[1, 3, h, 16, f32], x2: tensor[1, 3, h, 16, f32], k: tensor[8, 3, 3, 3, f32]) -> (tensor[1, 8, 6, 6, f32], tensor[1, 8, 6, 6, f32]) = {
  y1 = conv(&x1, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  y2 = conv(&x2, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  (y1, y2)
}
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic conv metadata is not a type error");
}

/// EXPECT (RT-205 round-2 F4): a rank-5 input with a rank-4 kernel
/// produces exactly ONE diagnostic naming the kernel disagreement.
/// The validator must not repeat the HM-side rank check.
#[test]
fn red_team_205_round2_f4_rank5_input_single_diagnostic() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, 2, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for rank-5 input");
    let rank_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                chelis_types::errors::CheckErrorKind::DimensionMismatch
            ) && e.expected.as_deref() == Some("rank 5")
                && e.got.as_deref() == Some("rank 4")
                && e.message.contains("conv argument 2 (kernel)")
        })
        .collect();
    assert_eq!(
        rank_errors.len(),
        1,
        "expected one kernel rank error: {rep:?}"
    );
}

/// EXPECT (RT-205 round-2 F4): likewise, a rank-5 kernel with a rank-4
/// input produces one directional rank diagnostic.
#[test]
fn red_team_205_round2_f4_rank5_kernel_single_diagnostic() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, 2, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure for rank-5 kernel");
    let rank_errors: Vec<_> = rep
        .errors
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                chelis_types::errors::CheckErrorKind::DimensionMismatch
            ) && e.expected.as_deref() == Some("rank 4")
                && e.got.as_deref() == Some("rank 5")
                && e.message.contains("conv argument 2 (kernel)")
        })
        .collect();
    assert_eq!(
        rank_errors.len(),
        1,
        "expected one kernel rank error: {rep:?}"
    );
}

// ─── Red Team #205 round-3 follow-up findings ────────────────────

/// PP9: a symbolic conv stays checker-legal through a unary shape-preserving
/// wrapper.
#[test]
fn red_team_205_round3_f_a_symbolic_conv_through_relu_is_checker_legal() {
    let src = r#"
def f[h](x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = relu(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]))
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic conv metadata is not a type error");
}

/// PP9: a symbolic conv stays checker-legal through the canonical conv+bias
/// shape-preserving wrapper.
#[test]
fn red_team_205_round3_f_a_symbolic_conv_through_add_is_checker_legal() {
    let src = r#"
def f[h](x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = add(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), &b)
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic conv metadata is not a type error");
}

/// EXPECT (RT-205 round-3 F-A negative parity): when the
/// relu-wrapped first conv is valid but the second conv has an
/// independent issue (wrong kernel rank), which must still be reported.
#[test]
fn red_team_205_round3_f_a_independent_second_failure_not_suppressed() {
    // First conv: clean. relu(conv(...)) registers y as
    // tensor[1, 8, 6, 6, f32]. The second kernel is rank 3 while
    // its input is rank 4.
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = relu(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]))
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check failure on second conv kernel rank");
    assert!(
        rep.errors.iter().any(|e| {
            matches!(
                e.kind,
                chelis_types::errors::CheckErrorKind::DimensionMismatch
            ) && e.expected.as_deref() == Some("rank 4")
                && e.got.as_deref() == Some("rank 3")
                && e.message.contains("conv argument 2 (kernel)")
        }),
        "expected a rank mismatch on the second kernel, got {:?}",
        rep.errors
    );
}

/// EXPECT (RT-205 round-3 F-B): `max_elem` is the canonical IR name
/// per spec/05 §2.1, not `maximum`. The binary passthrough allowlist
/// previously contained `maximum`/`minimum` which do not exist in
/// the IR vocabulary, so `y = max_elem(conv(...), &b)` cascade was
/// silently broken.
#[test]
fn red_team_205_round3_f_b_max_elem_passthrough() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {
  y = max_elem(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), &b)
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
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
  y = min_elem(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), &b)
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
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
/// conv call with the unary op and chains into a second conv;
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
  y = {op}(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]))
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
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
/// wraps a clean conv with the binary op + same-shape bias and
/// chains into a second conv; the chain must type-check cleanly.
///
/// `cmplt`, `gt`, `gte`, `lte`, `eq`, `neq` and `and`/`or` return
/// tensor[D, bool], so the downstream call cannot be a conv (which
/// requires f-prec). For those we only check that the let-binder's
/// own validation produces a clean type (no second conv).
#[test]
fn red_team_205_round3_f_b_binary_passthrough_audit_arith() {
    // Arithmetic binaries preserve precision and shape; chain into a
    // second conv to exercise the downstream registration.
    let arith_ops = ["add", "sub", "mul", "div", "max_elem", "min_elem"];
    for op in &arith_ops {
        let src = format!(
            r#"
def f(x: tensor[1, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {{
  y = {op}(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), &b)
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
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
/// PP9 removes the backend-only symbolic-metadata diagnostic. The comparison
/// wrappers can still fail for their own result-type reasons, but they must
/// never resurrect that retired checker rejection.
#[test]
fn red_team_205_round3_f_b_binary_compare_does_not_restore_metadata_rejection() {
    let compare_ops = ["cmplt", "lt", "gt", "gte", "lte", "eq", "neq"];
    for op in &compare_ops {
        // Symbolic conv metadata is checker-legal. Comparison inference may
        // still reject its own operands, but it must not recreate the retired
        // backend capability diagnostic.
        let src = format!(
            r#"
def f[h](x: tensor[1, 3, h, 16, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32], b: tensor[1, 8, 6, 6, f32]) -> tensor[1, 16, 4, 4, f32] = {{
  y = {op}(conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), &b)
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}}
"#
        );
        let deep = surf_to_deep(&src);
        let messages = match check_ir_program(&deep) {
            Ok(_) => Vec::new(),
            Err(report) => report
                .errors
                .into_iter()
                .map(|error| error.message)
                .collect::<Vec<_>>(),
        };
        assert!(
            !messages
                .iter()
                .any(|message| message.contains("concrete tensor argument metadata")),
            "op `{op}` restored the retired symbolic-metadata rejection: {messages:?}"
        );
    }
}

/// EXPECT (RT-205 round-3 F-C): a symbolic batch dim is accepted in a
/// conv call. PP9 generalizes that language-level acceptance to symbolic
/// tensor metadata; backend implementation remains a separate capability.
#[test]
fn red_team_205_round3_f_c_symbolic_batch_single_call() {
    let src = r#"
def f(x: tensor[batch, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[batch, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for symbolic batch, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT (RT-205 round-3 F-C): symbolic batch flows through a
/// chained conv. The first call's derived output type registers
/// `y` with `(d-name {} batch)` at axis 0, and the second call
/// resolves `y` to that same symbolic-batch type. The output
/// signature is `tensor[batch, 16, 4, 4, f32]`.
#[test]
fn red_team_205_round3_f_c_symbolic_batch_chained() {
    let src = r#"
def f(x: tensor[batch, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[batch, 16, 4, 4, f32] = {
  y = conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for symbolic-batch chained conv, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// PP9: a symbolic spatial extent is legal at the checker boundary.
#[test]
fn red_team_205_round3_f_c_nonconcrete_spatial_is_checker_legal() {
    let src = r#"
def f[h](x: tensor[1, 3, h, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic spatial metadata is not a type error");
}

/// PP9: a symbolic channel extent is legal when the input/kernel equality
/// constraint is preserved.
#[test]
fn red_team_205_round3_f_c_nonconcrete_in_channels_is_checker_legal() {
    let src = r#"
def f(x: tensor[1, in_c, 8, 8, f32], k: tensor[8, in_c, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
"#;
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("symbolic channel metadata is not a type error");
}

/// EXPECT (RT-205 round-3 F-C negative parity): a chained
/// symbolic-batch conv that disagrees on the batch dim between
/// the two calls is rejected by the def's body-vs-sig check (the
/// declared return type has a different batch dim than the inferred
/// one). The F-C fix preserves batch through the chain; HM
/// signature checking catches an explicit declared/inferred
/// mismatch.
#[test]
fn red_team_205_round3_f_c_batch_dim_mismatch_detected() {
    // Declared return uses `b2`, but the body's chain carries
    // `batch` through, so HM should refuse the def.
    let src = r#"
def f(x: tensor[batch, 3, 8, 8, f32], k1: tensor[8, 3, 3, 3, f32], k2: tensor[16, 8, 3, 3, f32]) -> tensor[b2, 16, 4, 4, f32] = {
  y = conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected rejection on batch dim mismatch");
    assert!(
        rep.errors.iter().any(
            |e| e.message.contains("body doesn't match declared signature")
                || matches!(
                    e.kind,
                    chelis_types::errors::CheckErrorKind::DimensionMismatch
                )
        ),
        "expected batch-dim mismatch error, got {:?}",
        rep.errors
            .iter()
            .map(|e| (&e.kind, &e.message))
            .collect::<Vec<_>>()
    );
}

/// [05-OP-51]: runtime stride values are legal and retain runtime guards.
#[test]
fn issue186_deep_conv_nonliteral_stride_is_checker_legal() {
    let src = "(def {type: (t-prim {} i64)} stride_v (lit {type: (t-prim {} i64)} 1)) \
               (def {} x (lit {type: (t-tensor {} (d-lit {} 1) (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))} 0)) \
               (def {} k (lit {type: (t-tensor {} (d-lit {} 8) (d-lit {} 3) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0)) \
               (def {type: (t-tensor {} (d-lit {} 1) (d-lit {} 8) (d-lit {} 6) (d-lit {} 6) (t-prim {} f32))} y \
                 (app {} (var {} conv) (var {} x) (var {} k) \
                   (app {} (var {} Cons) (var {} stride_v) (app {} (var {} Cons) (var {} stride_v) (var {} Nil))) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 0)) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 0)) (var {} Nil)))))";
    let deep = parse_deep(src).expect("deep parse");
    check_ir_program(&deep).expect("runtime stride metadata is not a type error");
}
