//! Issue #187 regression: `shrink` and `stride` are registered as 1-arg
//! `tensor_unop` but the IR lowering / evaluator consume window parameters
//! from `args[1..]`. As a result no callable form exists from Surf source:
//!   - bare 1-arg `shrink(x)` / `stride(x)` type-checks but is not
//!     host-runnable (`unsupported builtin in host runtime`).
//!   - parameterized `shrink(x, bounds)` / `stride(x, s0, s1, ...)`
//!     type-rejects with `arity mismatch: expected 1 args`.
//!
//! This file exercises the type-signature side via `check_ir_program`
//! (the `chelis check` entry point). Host-runtime parity is covered in
//! `chelis-compiler-api/tests/issue_187_shrink_stride_host_runtime.rs`.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §2.4, which documents
//!   shrink: (&tensor[D,p], bounds)  -> tensor[D',p]
//!   stride: (&tensor[D,p], strides) -> tensor[D',p]

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

// ---------------------------------------------------------------------------
// Positive: parameterized forms type-check.
// ---------------------------------------------------------------------------

/// EXPECT: `shrink(&x, [[0i64, 1i64], [1i64, 3i64]])` on a `tensor[2, 4, f32]` type-checks
/// to `tensor[1, 2, f32]` (the windowed sub-tensor shape).
#[test]
fn issue187_surf_shrink_parameterized_typechecks() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[0i64, 1i64], [1i64, 3i64]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for parameterized shrink, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `stride(&x, 1i64, 2i64)` on a `tensor[2, 4, f32]` type-checks to
/// `tensor[2, 2, f32]` (stride 1 on axis 0, stride 2 on axis 1 -> dims
/// 2 and ceil(4/2) = 2).
#[test]
fn issue187_surf_stride_parameterized_typechecks() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, 1i64, 2i64)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for parameterized stride, got {} error(s)",
            rep.errors.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Negative parity: malformed parameterized forms are rejected with clear
// errors. Per the §0 plan, each negative test names ONE pinned failure mode.
// ---------------------------------------------------------------------------

/// EXPECT: bare `shrink(&x)` (no bounds arg) is rejected with an arity
/// error. Before the fix this typed cleanly via the 1-arg `tensor_unop`
/// scheme, but the resulting RISC node had no bounds and the host runtime
/// had no arm for it, so it was a silent-corruption surface.
#[test]
fn issue187_surf_shrink_bare_no_bounds_is_arity_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = shrink(&x)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on bare shrink(&x)");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("bounds") || e.message.contains("arity"))),
        "expected a shrink/arity/bounds error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `shrink(&x, [[0i64, 5i64], [1i64, 3i64]])` on `tensor[2, 4, f32]` is rejected
/// because the axis-0 bound (0, 5) ends past the input dimension (2).
#[test]
fn issue187_surf_shrink_out_of_range_bound_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[5, 2, f32] = shrink(&x, [[0i64, 5i64], [1i64, 3i64]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on out-of-range shrink bound");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("axis") || e.message.contains("bound"))),
        "expected a shrink axis/bound error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `shrink(&x, [[0i64, 1i64]])` on a rank-2 tensor is rejected because
/// the bounds list has fewer pairs than the input rank.
#[test]
fn issue187_surf_shrink_wrong_rank_bounds_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 4, f32] = shrink(&x, [[0i64, 1i64]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on rank-mismatched shrink bounds");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("rank") || e.message.contains("bound"))),
        "expected a shrink rank/bound error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `stride(&x, 0i64, 2i64)` is rejected because a zero stride is not
/// allowed (it would either be a division-by-zero in the strided index
/// arithmetic or a no-shrink/no-advance footgun).
#[test]
fn issue187_surf_stride_zero_step_is_error() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, 0i64, 2i64)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on zero stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("stride")
                && (e.message.contains("zero") || e.message.contains("positive"))),
        "expected a stride zero/positive error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `stride(&x, 1i64)` on a rank-2 tensor is rejected because the
/// strides list has fewer entries than the input rank.
#[test]
fn issue187_surf_stride_wrong_rank_is_error() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = stride(&x, 1i64)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on rank-mismatched stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("stride")
                && (e.message.contains("rank") || e.message.contains("axis"))),
        "expected a stride rank/axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `stride(&x, "two")` is rejected because stride entries must be
/// i64, not strings.
#[test]
fn issue187_surf_stride_non_int_is_error() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = stride(&x, "two")
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on non-int stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("stride")
                && (e.message.contains("i64")
                    || e.message.contains("integer")
                    || e.message.contains("type"))),
        "expected a stride type error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Sibling sweep: `pad` had the same `tensor_unop`-vs-parameterized antipattern
// (registered as 1-arg, lowering reads padding from args[1] and fill from
// args[2]). Fixed in the same PR per the issue #187 sibling-sweep policy.
// ---------------------------------------------------------------------------

/// EXPECT: `pad(&x, [[0i64, 1i64], [0i64, 1i64]], 0.0)` on a `tensor[2, 4, f32]`
/// type-checks to `tensor[3, 5, f32]` (input dim plus lo plus hi per axis).
#[test]
fn issue187_sibling_pad_parameterized_typechecks() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[0i64, 1i64], [0i64, 1i64]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for parameterized pad, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: bare `pad(&x)` is rejected with an arity error.
#[test]
fn issue187_sibling_pad_bare_is_arity_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = pad(&x)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on bare pad(&x)");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("padding") || e.message.contains("arity"))),
        "expected a pad/arity error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `pad(&x, [[0i64, 1i64]], 0.0)` on a rank-2 tensor is rejected
/// (wrong-rank padding list).
#[test]
fn issue187_sibling_pad_wrong_rank_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 4, f32] = pad(&x, [[0i64, 1i64]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on wrong-rank pad padding");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("rank") || e.message.contains("padding"))),
        "expected a pad rank/padding error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Red team round 1 on PR #214: malformed pair literals silently fell through
// to `Dim::Wildcard` at infer time instead of being rejected. Compensating
// host-runtime / IR-verifier safeguards still caught these one layer
// downstream, but the desired behavior is a clean type error at
// `chelis check`.
// ---------------------------------------------------------------------------

/// R1-F1: `shrink(&x, [[0i64, 1i64, 2i64]])` -- inner pair is a triple, not a pair.
/// Must be rejected at infer time with a message naming the offending axis.
#[test]
fn red_team_214_r1_f1_shrink_triple_inner_pair_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[0i64, 1i64, 2i64]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on triple-element inner pair");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("pair")
                    || e.message.contains("2-element")
                    || e.message.contains("got 3"))),
        "expected a shrink pair/length error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F1: `shrink(&x, [[0i64]])` -- inner pair is a singleton missing the end.
#[test]
fn red_team_214_r1_f1_shrink_singleton_inner_pair_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[0i64]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on singleton inner pair");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("pair")
                    || e.message.contains("2-element")
                    || e.message.contains("got 1"))),
        "expected a shrink pair/length error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F1: `pad(&x, [[1i64]], 0.0)` -- inner singleton inside pad's pair list.
#[test]
fn red_team_214_r1_f1_pad_singleton_inner_pair_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1i64]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on singleton inner pair in pad");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("pair")
                    || e.message.contains("2-element")
                    || e.message.contains("got 1"))),
        "expected a pad pair/length error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F1: `pad(&x, [[1i64, 0i64, 99i64]], 0.0)` -- inner triple inside pad's pair
/// list. Mirror of the shrink triple case.
#[test]
fn red_team_214_r1_f1_pad_triple_inner_pair_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1i64, 0i64, 99i64]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on triple inner pair in pad");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("pair")
                    || e.message.contains("2-element")
                    || e.message.contains("got 3"))),
        "expected a pad pair/length error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// R1-F2: pad fill type was not enforced at infer time. A list fill, a
// bool fill, or a string fill all slipped through to host-runtime; spec
// §2.4 says fill is a scalar of the input precision.
// ---------------------------------------------------------------------------

/// R1-F2: `pad(&x, ..., [0.0])` -- list fill rather than scalar. Must be
/// rejected at infer.
#[test]
fn red_team_214_r1_f2_pad_list_fill_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1i64, 0i64], [0i64, 1i64]], [0.0])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on list pad fill");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("fill") || e.message.contains("scalar"))),
        "expected a pad fill/scalar error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F2: `pad(&x, ..., true)` on an f32 tensor -- bool fill, wrong
/// precision. Must be rejected at infer.
#[test]
fn red_team_214_r1_f2_pad_bool_fill_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1i64, 0i64], [0i64, 1i64]], true)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on bool pad fill");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("fill")
                    || e.message.contains("scalar")
                    || e.message.contains("precision"))),
        "expected a pad fill/scalar/precision error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F2 positive control: `pad(&x, ..., 0.0)` on f32 tensor still passes.
/// Locks the constraint to "fill matches input precision," not "fill is
/// forbidden."
#[test]
fn red_team_214_r1_f2_pad_matching_scalar_fill_still_typechecks() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1i64, 0i64], [0i64, 1i64]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for matching f32 pad fill, got {} error(s)",
            rep.errors.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Red team round 2 on PR #214: round-1 missed two further infer-side gaps.
// ---------------------------------------------------------------------------

/// R2-M1: `shrink(&x, [[], [0i64, 1i64]])` -- the inner `[]` desugars to bare
/// `(var Nil)`, which `cons_chain_two_ints` was classifying as
/// `NonLiteral` (an "opaque List[Int32] variable"). The user's intent is
/// a zero-element list, not a polymorphic reference; that's still a
/// malformed pair shape and must be rejected at infer time with the same
/// "got 0-element list" diagnostic the round-1 fix uses for non-pair
/// arities.
#[test]
fn red_team_214_r2_m1_shrink_empty_inner_list_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[], [0i64, 1i64]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on empty inner list");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("0-element") || e.message.contains("empty"))),
        "expected a shrink empty-inner-list error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R2-M1 (pad sibling): same empty-inner-list bug in pad's pair list.
#[test]
fn red_team_214_r2_m1_pad_empty_inner_list_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[], [1i64, 1i64]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on empty inner list in pad");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("0-element") || e.message.contains("empty"))),
        "expected a pad empty-inner-list error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R2-L1: cast-wrapped negative bound. `cast(-1, i64)` is concretely a
/// negative literal at desugar time, but `extract_int_literal` doesn't
/// peel `cast`, so the bounds checks were skipped and the program slipped
/// through to host runtime. Use `extract_int_for_dim` (which already
/// handles cast peeling, per reshape) to bring this check forward.
#[test]
fn red_team_214_r2_l1_shrink_cast_wrapped_negative_bound_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 1, f32] = shrink(&x, [[cast(-1, i64), cast(1, i64)], [cast(0, i64), cast(1, i64)]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped negative bound");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("negative")
                    || e.message.contains("inverted")
                    || e.message.contains("empty"))),
        "expected a shrink negative-bound error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R2-L1: cast-wrapped out-of-range bound. The end endpoint
/// `cast(5, i64)` exceeds axis-0 dim 2.
#[test]
fn red_team_214_r2_l1_shrink_cast_wrapped_out_of_range_bound_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[5, 1, f32] = shrink(&x, [[cast(0, i64), cast(5, i64)], [cast(0, i64), cast(1, i64)]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped out-of-range bound");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("out of range") || e.message.contains("axis 0"))),
        "expected a shrink out-of-range error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R2-L1 positive control: well-formed cast-wrapped bounds must still
/// type-check cleanly with a precise output shape.
#[test]
fn red_team_214_r2_l1_shrink_cast_wrapped_well_formed_typechecks() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[cast(0, i64), cast(1, i64)], [cast(1, i64), cast(3, i64)]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for cast-wrapped well-formed shrink bounds, got {} error(s)",
            rep.errors.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Red team round 3 on PR #214. Cast-wrapped int extraction needs to reach
// all parameterized-builtin sites in the spec/05 section 2.4 movement
// family, not just shrink/pad's inner pair list. Plus the `neg` arm of
// extract_int_literal and the cast arm of extract_int_for_dim must be
// recursive so combinations like neg(cast(...)) and cast(cast(...)) also
// reach the infer-time bounds check instead of falling back to wildcard.
// ---------------------------------------------------------------------------

/// R3-HIGH1: `stride(&x, cast(0, i64), 2i64)`. The stride extractor at
/// `infer.rs:10864` uses the non-cast-aware `extract_int_literal`, so
/// cast-wrapped zero strides bypass the positive-stride check at infer.
/// Host runtime catches it, but the diagnostic should land at check.
#[test]
fn red_team_214_r3_high1_stride_cast_wrapped_zero_is_error() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, cast(0, i64), 2i64)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped zero stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("stride")
                && (e.message.contains("zero") || e.message.contains("positive"))),
        "expected a stride zero/positive error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R3-HIGH1 sibling: cast-wrapped negative stride.
#[test]
fn red_team_214_r3_high1_stride_cast_wrapped_negative_is_error() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, cast(-1, i64), 2i64)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped negative stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("stride")
                && (e.message.contains("positive")
                    || e.message.contains("zero")
                    || e.message.contains("negative"))),
        "expected a stride positive/negative error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R3-HIGH2 (contract lock, inverted by chelis#1112): `cast(N, i32)`
/// endpoints inside shrink's bounds list are rejected at the OUTER
/// bounds-type unification step (the expected type is `List[List[Int64]]`
/// under [05-DIM-1]). The pre-#1112 revision of this lock asserted the
/// mirror image; a future change that loosens the bounds type back must
/// intentionally update this fixture.
#[test]
fn red_team_214_r3_high2_int32_cast_bound_rejected_at_unification() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[cast(0, i32), cast(1, i32)], [cast(1, i32), cast(3, i32)]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on i32 cast endpoints");
    // The diagnostic must come from the outer List[List[Int64]] unification
    // step, naming i64 as the demanded bound type.
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && e.message.contains("i64")
                && (e.message.contains("i32") || e.message.contains("List"))),
        "expected a shrink i64/i32 unification error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R3-MED1: `neg(cast(1, i64))` resolves to -1 numerically but the
/// `extract_int_literal` `neg` arm recurses via `extract_int_literal`
/// (not `extract_int_for_dim`), so the inner cast hides the literal.
/// The negative-endpoint check is skipped and the malformed program
/// slips through.
#[test]
fn red_team_214_r3_med1_neg_of_cast_int_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 1, f32] = shrink(&x, [[neg(cast(1, i64)), cast(1, i64)], [cast(0, i64), cast(1, i64)]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on neg(cast(...))-wrapped negative bound");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("negative")
                    || e.message.contains("inverted")
                    || e.message.contains("empty"))),
        "expected a shrink negative-bound error for neg(cast(...)), got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R3-MED2: doubly-nested cast `cast(cast(-1, i32), i64)`.
/// `extract_int_for_dim`'s cast arm peels exactly one cast layer and
/// then calls `extract_int_literal` (which doesn't peel cast), so any
/// depth greater than one falls back to `NonLiteral`.
#[test]
fn red_team_214_r3_med2_double_cast_negative_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 1, f32] = shrink(&x, [[cast(cast(-1, i32), i64), cast(1, i64)], [cast(0, i64), cast(1, i64)]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on doubly-nested cast negative bound");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("negative")
                    || e.message.contains("inverted")
                    || e.message.contains("empty"))),
        "expected a shrink negative-bound error for cast(cast(...)), got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Sibling-sweep targets within the spec section 2.4 movement family.
// `permute` and `expand` are the other parameterized builtins that take
// int literals from `kids[2..]` via `extract_int_literal`; the same
// cast-bypass surfaces there too. Reductions, conv, gather/scatter
// axis extractors are flagged in the commit body as follow-up domain
// scope -- not fixed in this PR.
// ---------------------------------------------------------------------------

/// R3-sibling-sweep: cast-wrapped permute axes type-check cleanly and
/// produce the correctly-reordered output type (was silently falling
/// back to original-dim order).
#[test]
fn red_team_214_r3_permute_cast_wrapped_axes_typechecks() {
    let src = r#"
def g(k: tensor[2, 4, f32]) -> tensor[4, 2, f32] = permute(&k, cast(1, i32), cast(0, i32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for cast-wrapped permute axes, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// R3-sibling-sweep: cast-wrapped out-of-bounds permute axis rejects at
/// infer (was masked by the wrong-fallback type mismatch).
#[test]
fn red_team_214_r3_permute_cast_wrapped_out_of_bounds_axis_is_error() {
    let src = r#"
def g(k: tensor[2, 4, f32]) -> tensor[4, 2, f32] = permute(&k, cast(99, i32), cast(0, i32))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on cast-wrapped OOB permute axis");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("permute")
                && (e.message.contains("out of bounds") || e.message.contains("axis"))),
        "expected a permute out-of-bounds axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R3-sibling-sweep: cast-wrapped expand axis/size type-check and
/// produce the correct output shape.
#[test]
fn red_team_214_r3_expand_cast_wrapped_axis_and_size_typechecks() {
    let src = r#"
def f(x: tensor[2, f32]) -> tensor[3, 2, f32] = insert(&x, cast(0, i32), cast(3, i64))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for cast-wrapped expand args, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// chelis#1277 Slice A: zero is a valid extent, including through the
/// canonical cast-aware static folder.
#[test]
fn red_team_214_r3_expand_cast_wrapped_zero_size_typechecks() {
    let src = r#"
def f(x: tensor[2, f32]) -> tensor[0, 2, f32] = insert(&x, cast(0, i32), cast(0, i64))
"#;
    let deep = surf_to_deep(src);
    if let Err(rep) = check_ir_program(&deep) {
        panic!(
            "cast-wrapped zero expand size must check cleanly, got {:?}",
            rep.errors
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>()
        );
    }
}
