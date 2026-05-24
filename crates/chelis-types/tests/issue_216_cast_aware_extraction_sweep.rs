//! Issue #216 regression: cast-aware int extraction sweep beyond the
//! movement-op family.
//!
//! PR #214 extended cast-aware extraction (`extract_int_for_dim` instead
//! of `extract_int_literal`) across the spec/05 §2.4 movement-op family
//! (shrink, stride, pad, permute, expand). The R3 sibling-sweep audit
//! identified 13 additional infer-time int-literal extractors in
//! `crates/chelis-types/src/infer.rs` that share the same cast-bypass
//! antipattern in different domains: reductions, conv2d helpers,
//! gather/scatter, shape lookup, split, softmax, vmap.
//!
//! Each test in this file pins one cast-wrapped form that MUST trip the
//! infer-time validation check; before the fix the cast wrapper hid the
//! literal from `extract_int_literal` so the validation was silently
//! skipped (host-runtime defense-in-depth still caught it, but the user
//! got the wrong diagnostic at the wrong layer).
//!
//! Three sites in this sweep are NOT user-reachable from idiomatic Surf
//! source (the Surf parser desugars them to bare literal ints): vmap's
//! `axis=N` keyword arg and grad's `wrt=name1, name2` keyword arg. They
//! ARE still reachable via direct Deep input (e.g. macro output, tooling,
//! decompiled IR), so the swap is defense-in-depth on those paths. The
//! tests for those three sites are constructed at the Deep level via
//! `parse_deep`, not the Surf parser.
//!
//! Spec sources of truth:
//!   - spec/05-risc-primitives.md §2.4 (movement) and §5.7.1 (reductions)
//!   - spec/04-type-system.md (axis bounds, no implicit broadcasting)

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

// ---------------------------------------------------------------------------
// Reduction axis: `sum` / `max_reduce` / `min_reduce` / `prod_reduce` / `mean`.
// Site: infer.rs check_reduce_signature axis extraction (~line 12076).
// Spec: spec/05 §5.7.1 -- "axis indexes the reduction dimension; negative
// axes index from the end."
// ---------------------------------------------------------------------------

/// EXPECT: `sum(x, cast(99, int32))` on a rank-2 tensor is rejected at
/// infer with an out-of-bounds axis diagnostic. Without the cast-aware
/// fix the cast wrapper hides the literal and the bounds check is
/// silently skipped.
#[test]
fn issue216_sum_cast_wrapped_oob_axis_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, f32] = sum(x, cast(99, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB sum axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("sum")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a sum out-of-bounds axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `max_reduce(x, cast(7, int32))` on rank-2 tensor is rejected
/// (mirror of sum). Locks the swap is reduction-family-wide, not just
/// `sum`-specific.
#[test]
fn issue216_max_reduce_cast_wrapped_oob_axis_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, f32] = max_reduce(x, cast(7, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB max_reduce axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("max_reduce")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a max_reduce out-of-bounds axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// Positive control: `sum(x, cast(-1, int32))` on rank-2 still
/// type-checks cleanly. The cast-aware extractor must peel the wrapper
/// so the negative axis normalizes (rank + axis = 1) instead of being
/// dropped as "non-literal."
#[test]
fn issue216_sum_cast_wrapped_negative_axis_typechecks() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, f32] = sum(x, cast(-1, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for cast-wrapped negative sum axis, got {} error(s)",
            rep.errors.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Softmax axis bounds check.
// Site: infer.rs softmax axis extraction (~line 7630).
// Spec: softmax is shape-preserving but the axis must be in [0, rank).
// ---------------------------------------------------------------------------

/// EXPECT: `softmax(x, cast(5, int32))` on rank-2 is rejected at infer
/// with an out-of-bounds axis diagnostic.
#[test]
fn issue216_softmax_cast_wrapped_oob_axis_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = softmax(x, cast(5, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB softmax axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("softmax")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a softmax out-of-bounds axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Shape axis lookup (`shape(tensor, axis)`).
// Site: infer.rs shape arm (~line 8238).
// Spec: axis is a non-negative literal index. Issue #206 already locks
// the `shape(x, cast(N, int32))` idiomatic form for runtime-dim reshape.
// ---------------------------------------------------------------------------

/// EXPECT: `shape(x, cast(-1, int32))` is rejected at infer with the
/// `shape requires non-negative axis` diagnostic. Without the cast-aware
/// fix the negative axis slips past the positivity check (cast peels at
/// the extractor, then `axis < 0` fires).
#[test]
fn issue216_shape_cast_wrapped_negative_axis_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> int32 = shape(x, cast(-1, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped negative shape axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shape")
                && (e.message.contains("non-negative") || e.message.contains("axis"))),
        "expected a shape non-negative-axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `shape(x, cast(9, int32))` on rank-2 is rejected with the
/// out-of-bounds axis diagnostic.
#[test]
fn issue216_shape_cast_wrapped_oob_axis_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> int32 = shape(x, cast(9, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB shape axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shape")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a shape out-of-bounds-axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Split axis bounds check (`split(tensor, axis, sizes)`).
// Site: infer.rs split arm (~line 8959).
// Spec: split axis is bounded by tensor rank; negative axes wrap.
// ---------------------------------------------------------------------------

/// EXPECT: `split(x, cast(9, int32), sizes)` is rejected with the
/// `split axis ... out of bounds` diagnostic.
#[test]
fn issue216_split_cast_wrapped_oob_axis_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> List[tensor[2, 4, f32]] = split(x, cast(9, int32), [cast(2, int32), cast(2, int32)])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB split axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("split")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a split out-of-bounds axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Gather axis (`gather(tensor, indices, axis)`).
// Site: resolve_builtin_axis called from the gather arm (~line 4154
// extractor; gather call site routes through `resolve_builtin_axis`).
// Spec: axis must be in [-rank, rank).
// ---------------------------------------------------------------------------

/// EXPECT: `gather(table, ids, cast(9, int32))` on rank-2 table is
/// rejected with an OOB axis diagnostic.
#[test]
fn issue216_gather_cast_wrapped_oob_axis_is_error() {
    let src = r#"
def f(table: tensor[10, 4, f32], ids: tensor[3, int64]) -> tensor[3, 4, f32] = gather(table, ids, cast(9, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB gather axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("gather")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a gather out-of-bounds axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Trace / diagonal pair-axis bounds (`trace(matrix, axis1, axis2)` and
// `diagonal(matrix, axis1, axis2)`).
// Site: `resolve_axis_pair_member` (~line 4110).
// Spec: each axis must be in [-rank, rank).
// ---------------------------------------------------------------------------

/// EXPECT: `trace(m, cast(9, int32), cast(0, int32))` on a rank-2
/// matrix is rejected with an OOB axis diagnostic.
#[test]
fn issue216_trace_cast_wrapped_oob_axis_is_error() {
    let src = r#"
def f(m: tensor[4, 4, f32]) -> f32 = trace(m, cast(9, int32), cast(1, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB trace axis1");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("trace")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a trace out-of-bounds axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `diagonal(m, cast(0, int32), cast(9, int32))` rejects on the
/// second-axis side (locks `resolve_axis_pair_member` swap for both
/// member calls, not just one).
#[test]
fn issue216_diagonal_cast_wrapped_oob_axis2_is_error() {
    let src = r#"
def f(m: tensor[4, 4, f32]) -> tensor[4, f32] = diagonal(m, cast(0, int32), cast(9, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB diagonal axis2");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("diagonal")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a diagonal out-of-bounds axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// conv2d stride/padding (validator + derived-output-type + concrete-spatial).
// Sites: `extract_typed_scalar_literal` (~line 5874), used by the conv2d
// validator (~line 5413); `derive_conv2d_output_type` (~lines 5720, 5721);
// `compute_concrete_conv2d_spatial` (~lines 11722, 11723).
// Spec: spec/05 §471-483 -- stride must be positive; padding non-negative.
// ---------------------------------------------------------------------------

/// EXPECT: `conv2d(&x, &k, cast(0, int32), 0)` is rejected with the
/// conv2d positive-stride diagnostic. Without the cast-aware fix the
/// cast(0, int32) was rejected with the WRONG diagnostic
/// (`requires a literal integer stride`) because `extract_int_literal`
/// returned None on the cast wrapper. After the swap to
/// `extract_int_for_dim`, the cast peels to 0 and the actual
/// `positive stride` check fires.
#[test]
fn issue216_conv2d_cast_wrapped_zero_stride_is_error() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, cast(0, int32), 0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped zero conv2d stride");
    // Lock the post-fix diagnostic: "requires a positive stride, got 0".
    // Pre-fix the message was "requires a literal integer stride", which
    // is the wrong layer (the cast was a literal int, just wrapped).
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("conv2d")
                && e.message.contains("positive")
                && e.message.contains("stride")),
        "expected the conv2d positive-stride diagnostic (post-fix), got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `conv2d(&x, &k, 1, cast(-1, int32))` is rejected with the
/// conv2d non-negative-padding diagnostic. Same pre-vs-post-fix story
/// as the zero-stride case.
#[test]
fn issue216_conv2d_cast_wrapped_negative_padding_is_error() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, 1, cast(-1, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped negative conv2d padding");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("conv2d")
                && e.message.contains("non-negative")
                && e.message.contains("padding")),
        "expected the conv2d non-negative-padding diagnostic (post-fix), got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// Positive control: `conv2d(&x, &k, cast(1, int32), cast(0, int32))`
/// type-checks cleanly. Locks the swap is the cast-peeling itself, not
/// a regression on well-formed cast-wrapped stride/padding.
#[test]
fn issue216_conv2d_cast_wrapped_wellformed_typechecks() {
    let src = r#"
def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] =
  conv2d(&x, &k, cast(1, int32), cast(0, int32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for cast-wrapped well-formed conv2d, got {} error(s)",
            rep.errors.len()
        );
    }
}

// ---------------------------------------------------------------------------
// vmap axis (`(vmap {} f (lit ... N))` at Deep level).
// Site: `infer_vmap` axis extraction (~line 13866).
// Spec: axis must be non-negative.
//
// The Surf parser restricts `vmap(f, axis=N)` to bare-integer N, so this
// site is reachable only via direct Deep input. The swap is
// defense-in-depth for Deep-direct paths (macro output, tooling).
// ---------------------------------------------------------------------------

/// EXPECT: vmap with a cast-wrapped negative-axis literal at Deep level
/// is rejected with the `vmap axis must be non-negative` diagnostic.
/// Without the cast-aware fix the cast wraps the -1, so the literal
/// lookup falls back to `unwrap_or(0)` and the malformed call slips
/// through silently.
#[test]
fn issue216_vmap_cast_wrapped_negative_axis_is_error() {
    let src = "(defsig {} f
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} features) (t-prim {} f32))))
             (def {} f (fn {} (params {} x) (var {} x)))
             (def {} g (vmap {} (var {} f)
                 (cast {} (lit {type: (t-prim {} int32)} -1) (t-prim {} int32))))";
    let deep = parse_deep(src).expect("deep parse");
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped negative vmap axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("vmap")
                && (e.message.contains("non-negative") || e.message.contains("axis"))),
        "expected a vmap non-negative-axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Grad wrt parameter index at Deep level.
// Site: `grad_wrt_indices` (~lines 13793, 13814).
//
// At Surf level, `grad(f, wrt=name1, name2)` resolves parameter names to
// compile-time literal int indices during desugar, so cast-wrapping is
// unreachable from Surf. The swap is defense-in-depth for Deep-direct
// paths (e.g. macro output, decompiled IR, custom tooling). After the
// swap, a Deep `(cast {} (lit ... -1) (t-prim {} int32))` in the wrt
// slot peels to -1 and trips the `must be non-negative` check.
// ---------------------------------------------------------------------------

/// EXPECT: grad with a cast-wrapped negative wrt index at Deep level is
/// rejected with the `grad wrt index must be non-negative` diagnostic.
/// Pre-fix the cast wrapper hides the literal so `extract_int_literal`
/// returns None and the WRONG diagnostic fires (`must be an integer
/// parameter index` -- a tuple-type complaint, not a negative-index
/// complaint). Post-fix the cast peels to -1 and the actual
/// negative-index check fires.
#[test]
fn issue216_grad_wrt_cast_wrapped_negative_index_is_error() {
    let src = "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} hidden) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x w)
                    (lit {type: (t-prim {} f32)} 1.0)))
             (def {} dw
                (grad {} (var {} loss)
                    (cast {} (lit {type: (t-prim {} int32)} -1) (t-prim {} int32))))";
    let deep = parse_deep(src).expect("deep parse");
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped negative grad wrt index");
    // Lock the post-fix diagnostic: "must be non-negative, got -1".
    assert!(
        rep.errors.iter().any(
            |e| e.message.to_lowercase().contains("grad") && e.message.contains("non-negative")
        ),
        "expected the grad wrt non-negative-index diagnostic (post-fix), got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: grad with a cast-wrapped negative wrt index inside a tuple
/// at Deep level is also rejected. Locks the swap is for BOTH grad wrt
/// extraction sites (tuple form ~line 13793, single form ~line 13814),
/// not just one.
#[test]
fn issue216_grad_wrt_cast_wrapped_negative_tuple_index_is_error() {
    let src = "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} hidden) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x w)
                    (lit {type: (t-prim {} f32)} 1.0)))
             (def {} grads
                (grad {} (var {} loss)
                    (tuple {}
                        (lit {type: (t-prim {} int32)} 0)
                        (cast {} (lit {type: (t-prim {} int32)} -1) (t-prim {} int32)))))";
    let deep = parse_deep(src).expect("deep parse");
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped negative tuple wrt index");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("grad")
                && (e.message.contains("non-negative") || e.message.contains("index"))),
        "expected a grad wrt non-negative-index error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}
